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
		Hodgkin-Huxley neuron simulator extended with a slow M-type K+ current (Kv7/KCNQ).

		DESIGN RATIONALE:
		The base HH model (Na, K delayed rectifier, leak) produces tonic spiking but may
		deviate from experimental data in resting potential, inter-spike interval regularity,
		and higher-order voltage moments (variance, skewness, kurtosis). The M-current is a
		well-characterised, slow, non-inactivating K+ current active in the subthreshold range
		that regulates these statistics without introducing bursting or high-frequency dynamics.

		PARAMETER ASSIGNMENTS (following X1 slot convention):
		  - params[:,6] → gbar_M  : M-current max conductance (mS/cm²), range [1e-4, 10]
		  - params[:,7] → V_half_M: M-current half-activation voltage = -params[:,7] (mV),
		                            params[:,7] positive in [1e-4, 120], giving V_half_M in
		                            [-120, ~0] mV, covering typical KCNQ range of -35 to -20 mV
		  - params[:,8], params[:,9]: NOT USED — tau_max is fixed at 150 ms (physiological
		                            KCNQ time constant) to avoid identifiability issues

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage, mV
			input_current: torch.Tensor: (batch_size, time_steps) # injected current, uA/cm2
			dt: float # integration time step, ms
			t: torch.Tensor: (time_steps,) # time array, ms
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: int or None # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces, mV
		"""
		device = params.device

		# Set up reproducible random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ── Base HH parameters ───────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ max conductance, mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ delayed rectifier max conductance, mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance, mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal potential, mV
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold offset, mV
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude scaling

		# ── M-current parameters (X1 slot only, two tunable parameters) ─────────
		# M-current (Kv7/KCNQ): slow, non-inactivating, voltage-gated K+ current.
		# Activates in subthreshold range (~-60 to -20 mV). Physiological roles:
		#   1. Stabilises resting potential → improves resting mean/SD statistics
		#   2. Provides spike-frequency adaptation → regularises tonic ISI
		#   3. Shapes subthreshold voltage distribution → improves variance/skewness/kurtosis
		# Does NOT cause bursting (too slow, outward current suppresses high-frequency firing).

		# X1 slot: conductance gbar_M
		gbar_M   = params[:, 6].float()   # (batch_size,)  M-current max conductance, mS/cm2; [1e-4, 10]

		# X1's second tunable slot (params[:,7]) repurposed as half-activation voltage.
		# params[:,7] > 0 in [1e-4, 120], so V_half_M = -params[:,7] ∈ [-120, ~0] mV.
		# Inference will find the physiologically correct value (~-35 to -20 mV).
		V_half_M = -params[:, 7].float()  # (batch_size,)  M-current half-activation voltage, mV

		# tau_max fixed at 150 ms: physiologically motivated for KCNQ channels.
		# Fixed (not inferred) to avoid identifiability problems with gbar_M and V_half_M.
		tau_max_M = 150.0  # scalar float, ms

		tstep = float(dt)  # scalar float, ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (kept at 0)
		C    = 1.0            # membrane capacitance, uF/cm²
		E_Na = 53.0           # Na+ reversal potential, mV
		E_K  = -107.0         # K+ reversal potential, mV (used by delayed rectifier and M-current)

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential, clamped at -500 to prevent underflow
			# z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Rall's efun: z / (exp(z) - 1), regularised at z≈0 via Taylor expansion
			# z: any shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gating kinetics ───────────────────────────────────────────
		def alpha_m(x):
			# Na+ activation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			# Na+ deactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2     # (batch_size,)

		def alpha_h(x):
			# Na+ inactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)          # (batch_size,)

		def beta_h(x):
			# Na+ deinactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))    # (batch_size,)

		def alpha_n(x):
			# K+ delayed rectifier activation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# K+ delayed rectifier deactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)            # (batch_size,)

		def tau_x(alpha, beta):
			# Gating time constant; alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gating steady-state; alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current gating kinetics ─────────────────────────────────────────────
		# Boltzmann steady-state: p_inf = 1 / (1 + exp(-(V - V_half_M) / 10))
		# Slope factor 10 mV is standard for KCNQ/Kv7 (Wang & McKinnon 1995).
		def p_inf(x):
			# M-current steady-state activation; x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))  # (batch_size,)

		def tau_p(x):
			# M-current voltage-dependent time constant (ms); x: (batch_size,) → (batch_size,)
			# Bell-shaped profile via cosh: fastest at voltages far from V_half_M,
			# slowest (≈ tau_max_M) at V_half_M. Standard formulation for KCNQ channels.
			# Using single cosh (not 2*cosh) so tau_p(V_half_M) ≈ tau_max_M = 150 ms.
			dv = (x - V_half_M) / 40.0                           # (batch_size,), normalised deviation
			return tau_max_M / (torch.cosh(dv) + 1e-6)           # (batch_size,), ms

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na inactivation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate

		# ── Initialisation at steady state ────────────────────────────────────────
		V_init = init_voltage.to(device)                                           # (batch_size,)
		V[:, 0] = V_init                                                           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                        # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                        # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                                   # (batch_size,)

		# ── Main simulation loop (exponential Euler integration) ──────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Compute HH gating rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # each (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # each (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # each (batch_size,)

			# Compute M-current gate steady-state and time constant at previous voltage
			p_ss      = p_inf(V_prev)    # (batch_size,)
			tau_p_val = tau_p(V_prev)    # (batch_size,), ms

			# ── Effective conductance sum: g_total / C (= 1/tau_V) ──────────────
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na conductance
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) K delayed rectifier
				+ g_leak                                       # (batch_size,) leak
				+ gbar_M * p[:, i - 1]                        # (batch_size,) M-current (outward K+)
			) / C  # (batch_size,)

			# ── Voltage steady-state numerator: sum(g_i * E_i) + I_ext + noise ──
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)
				+ g_leak * E_leak                                      # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                         # (batch_size,) M-current reversal = E_K
				+ input_current[:, i - 1]                             # (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# ── Exponential Euler updates ─────────────────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                           # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))          # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))          # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))          # (batch_size,)
			# M-current gate: exponential Euler with voltage-dependent tau_p
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / (tau_p_val + 1e-6))                              # (batch_size,)

		# Return voltage traces with optional observation noise (currently 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)