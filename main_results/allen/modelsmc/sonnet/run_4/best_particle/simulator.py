import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
	def __init__(self):
		super(DiscoveredSimulator, self).__init__()
		return

	def forward(
		self,
		init_voltage: float,
		input_current: torch.Tensor,
		dt: float,
		t: torch.Tensor,
		params: torch.Tensor,
		seed=None,
	):
		"""
		Hodgkin-Huxley neuron extended with a single M-type (Kv7/KCNQ) slow K+ current.

		Changes from previous iteration (two targeted fixes based on diagnostics):

		FIX 1 — E_K corrected from -107 mV to -77 mV:
		  The base template used -107 mV which is ~30 mV more hyperpolarised than the
		  standard mammalian cortical neuron value. This error caused exaggerated AHP depth,
		  a resting potential shifted too low, and corrupted skewness/kurtosis statistics.
		  Because both the Kdr and M-current share E_K as their reversal potential, the
		  distortion was doubled. Correcting to -77 mV (standard for Kv7 and Kdr in
		  cortical neurons; also used by Mainen & Sejnowski 1996) should substantially
		  improve mean-V, variance, skewness, kurtosis, and resting-potential statistics.

		FIX 2 — tau_M reduced from 200 ms to 75 ms:
		  200 ms is at the extreme slow end of the Kv7 range. For typical tonic-spiking
		  cortical neurons with ISIs of 20-80 ms, a 200 ms M-gate barely activates between
		  spikes, forcing the posterior to compensate with implausibly large gbar_M values.
		  75 ms is the commonly reported dominant tau for cortical Kv7 channels (consistent
		  with Mainen & Sejnowski 1996 and Wang 1998) and allows meaningful M-current
		  build-up during sustained stimulation without suppressing spiking.

		PRESERVED from previous iteration:
		  - M-current exclusively uses X1 slot: params[6]=gbar_M, params[7]=|V_half_M|
		  - params[7] is in prior range [1e-4, 120]; negated to give V_half_M in (-120, 0) mV
		  - X2 slot (params[8], params[9]) intentionally unused — parsimony principle
		  - Sigmoid steady-state with 10 mV slope: p_inf = 1/(1+exp(-(V-V_half_M)/10))

		Args:
		    init_voltage  : torch.Tensor (batch_size,)            initial membrane voltage (mV)
		    input_current : torch.Tensor (batch_size, time_steps) injected current (µA/cm²)
		    dt            : float                                  time step (ms)
		    t             : torch.Tensor (time_steps,)             time array (ms)
		    params        : torch.Tensor (batch_size, 10)          biophysical parameters
		    seed          : int or None

		Returns:
		    V : torch.Tensor (batch_size, time_steps)              voltage traces (mV)
		"""
		device = params.device

		# ── random generator ─────────────────────────────────────────────────
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]    # scalar
		time_steps = t.shape[0]         # scalar

		# ── standard parameter extraction ────────────────────────────────────
		gbar_Na   = params[:, 0].float()    # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()    # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()    # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()   # (batch_size,)  mV  (params[3] > 0, sign applied)
		Vt        = -params[:, 4].float()   # (batch_size,)  mV  (params[4] > 0, sign applied)
		nois_fact = params[:, 5].float()    # (batch_size,)  unitless

		# ── X1 slot: M-current (Kv7/KCNQ) — both tunable parameters used ────
		# gbar_M  : M-current maximal conductance; prior range [1e-4, 10] mS/cm²
		# V_half_M: half-activation voltage; params[7] prior [1e-4, 120], negated → (-120, 0) mV
		#           Physiological Kv7 half-activation typically -35 to -20 mV, well within prior
		gbar_M   = params[:, 6].float()    # (batch_size,)  mS/cm²
		V_half_M = -params[:, 7].float()   # (batch_size,)  mV  (e.g. -30 mV when params[7]~30)

		# X2 slot intentionally unused — parsimony.
		# Adding a second channel before the M-current posterior is well-characterised
		# would degrade parameter identifiability and inflate the NLE.
		# params[:, 8] and params[:, 9] are reserved but not read.

		tstep         = float(dt)
		nois_fact_obs = 0.0     # observation noise kept at zero per task constraints
		C    = 1.0              # µF/cm²  membrane capacitance
		E_Na = 53.0             # mV      sodium reversal potential

		# FIX 1: E_K corrected to -77 mV (standard mammalian cortical neuron value).
		# Previous value of -107 mV caused systematic bias in all 7 summary statistics
		# via exaggerated AHP and shifted resting potential.
		E_K  = -77.0            # mV  potassium reversal (Kdr AND M-current)

		# FIX 2: tau_M set to 75 ms (reduced from 200 ms).
		# Allows M-gate to meaningfully activate/deactivate on timescales comparable
		# to typical tonic-spiking ISIs (20-80 ms), providing genuine spike-frequency
		# regulation without suppressing spiking.
		tau_M_fixed = 75.0      # ms  fixed M-current time constant

		# ── numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Safe exponential; clips at -500 to prevent underflow
			# z: (batch_size,) → (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Numerically stable z/(exp(z)-1) used in HH alpha/beta rates
			# z: (batch_size,) → (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── standard HH gate kinetics ─────────────────────────────────────────
		def alpha_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (Kv7/KCNQ) gate steady-state ───────────────────────────
		# Sigmoid activation centred on V_half_M with 10 mV slope.
		# Activates in peri-threshold range; non-inactivating (persistent K+ outward current).
		# 10 mV slope is canonical for Kv7 (Brown & Adams 1980, Wang 1998).
		def p_inf(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		# Time constant is voltage-independent and fixed at tau_M_fixed = 75 ms.
		# Both free X1 parameters (gbar_M, V_half_M) are already used, so a fixed
		# tau keeps the channel well-identified by the inference engine.

		# ── state variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Kdr gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na act gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inact gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-gate

		# ── initial conditions ────────────────────────────────────────────────
		V_init  = init_voltage.to(device)                           # (batch_size,)
		V[:, 0] = V_init
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))         # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))         # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))         # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                    # (batch_size,) M-gate at quasi-SS

		# Precompute the exponential decay factor for M-gate (constant across all steps)
		# Shape: scalar (same for all batches and time steps)
		exp_decay_p = torch.exp(torch.tensor(-tstep / tau_M_fixed, device=device))  # scalar

		# ── simulation loop (exponential-Euler integration) ───────────────────
		for i in range(1, time_steps):
			v_prev = V[:, i - 1]   # (batch_size,)

			# Standard HH gate rate constants at v_prev
			a_m, b_m = alpha_m(v_prev), beta_m(v_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(v_prev), beta_h(v_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(v_prev), beta_n(v_prev)   # (batch_size,), (batch_size,)

			# M-gate steady-state at v_prev
			p_ss = p_inf(v_prev)   # (batch_size,)

			# Effective inverse membrane time constant: sum of all active conductances / C
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  Na contribution
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)  Kdr contribution
				+ g_leak                                         # (batch_size,)  leak contribution
				+ gbar_M * p[:, i - 1]                          # (batch_size,)  M-current contribution
			) / C   # (batch_size,)

			# Voltage steady-state: (sum of g_i * E_i + I_ext + noise) / (sum of g_i)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)  Na
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,)  Kdr → -77 mV
				+ g_leak * E_leak                                        # (batch_size,)  leak
				+ gbar_M * p[:, i - 1] * E_K                           # (batch_size,)  M → -77 mV
				+ input_current[:, i - 1]                               # (batch_size,)  injected
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential-Euler updates (exact solution for piecewise-constant gate values)
			V[:, i] = V_inf + (v_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
			# M-gate: fixed tau → use precomputed scalar decay factor (efficient)
			p[:, i] = p_ss + (p[:, i-1] - p_ss) * exp_decay_p   # (batch_size,)

		# ── return voltage (observation noise currently zero) ─────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)