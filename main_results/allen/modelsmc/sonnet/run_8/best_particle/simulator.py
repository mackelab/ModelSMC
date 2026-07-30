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
		Hodgkin-Huxley neuron extended with an M-type slow K+ current (I_M).

		Physiological rationale for I_M:
		- M-current is a slow, non-inactivating subthreshold K+ current
		- It produces spike-frequency adaptation in tonically spiking neurons
		- It does NOT produce bursting or high-frequency firing clusters
		- It shapes resting potential, inter-spike voltage trajectory, and
		  voltage distribution statistics (variance, skewness, kurtosis)

		Args:
			init_voltage: torch.Tensor: (batch_size,) initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) injected current (uA/cm2)
			dt: float time step (ms)
			t: torch.Tensor: (time_steps,) time array (ms)
			params: torch.Tensor: (batch_size, 10) biophysical parameters
			seed: optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) voltage traces (mV)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ── Extract base parameters ──────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (negated from positive prior)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (negated from positive prior)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# ── M-current parameters (X1 slot) ───────────────────────────────────
		# gbar_M  : max conductance of M-type K+ current  (mS/cm2, range ~1e-4 to 10)
		# V_half  : half-activation voltage               (mV, negated → range ~-150 to 0)
		# tau_max : peak time constant of M-current gate  (ms, negated → range ~-3000 to 0,
		#           used as -param_j to give positive tau_max)
		gbar_M  = params[:, 6].float()    # (batch_size,) mS/cm2
		# gbar_X2 unused (kept for slot integrity)
		# gbar_X2 = params[:, 7].float()
		V_half  = -params[:, 8].float()   # (batch_size,) mV  e.g. inferred near -35 mV
		tau_max = -params[:, 9].float()   # (batch_size,) ms  e.g. inferred ~100–500 ms

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²  membrane capacitance
		E_Na = 53.0    # mV      sodium reversal potential
		E_K  = -107.0  # mV      potassium reversal potential (also used by M-current)

		# ── Numerical helper functions ────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; clips at -500 to avoid underflow
			# z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Numerically stable (e^z - 1)^{-1} * z  used in HH alpha/beta rates
			# z: any shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics ─────────────────────────────────────
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

		# ── M-current kinetics (slow non-inactivating K+ channel) ────────────
		# Steady-state activation: sigmoidal centred at V_half with slope 10 mV
		# V_half is inferred (~-35 mV); slope fixed at 10 mV for parsimony
		def p_inf_M(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((V_half - x) / 10.0))

		# Voltage-dependent time constant: bell-shaped centred at V_half
		# tau_max is inferred (~100–500 ms); min-clamped at tstep for stability
		def tau_p_M(x):
			# x: (batch_size,) → (batch_size,)
			dv = (x - V_half) / 20.0                              # (batch_size,)
			return torch.clamp(tau_max / (Exp(dv) + Exp(-dv)), min=tstep)  # (batch_size,)

		# ── Allocate state tensors ────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na act.
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na inact.
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K act.
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M act.

		# ── Initial conditions (steady-state at init_voltage) ─────────────────
		V_init = init_voltage.to(device)                           # (batch_size,)
		V[:, 0] = V_init
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)
		p[:, 0] = p_inf_M(V[:, 0])                                 # (batch_size,)

		# ── Time integration (exponential Euler method) ───────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH alpha/beta rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # each (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # each (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # each (batch_size,)

			# M-current steady-state and time constant at previous voltage
			p_ss  = p_inf_M(V_prev)   # (batch_size,)
			tau_p = tau_p_M(V_prev)   # (batch_size,)

			# ── Effective membrane conductance (denominator of V_inf) ─────────
			# Sum of all active conductances (mS/cm2)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev    # Na conductance  (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K conductance   (batch_size,)
				+ g_leak                             # leak            (batch_size,)
				+ gbar_M * p_prev                   # M-current K+    (batch_size,)
			) / C                                   # (batch_size,)

			# ── Effective voltage steady state (V_inf) ───────────────────────
			# Numerator: sum of conductance-weighted reversal potentials + currents
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na drive     (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K            # K drive      (batch_size,)
				+ g_leak * E_leak                          # leak drive   (batch_size,)
				+ gbar_M * p_prev * E_K                   # M-curr drive (batch_size,)
				+ input_current[:, i - 1]                 # injected     (batch_size,)
				+ nois_fact * torch.randn(
					batch_size, generator=generator, device=device
				) / (tstep ** 0.5)                        # noise        (batch_size,)
			) / (tau_V_inv * C)                           # (batch_size,)

			# ── Exponential Euler updates ─────────────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)          # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			# M-current gate: slow exponential Euler with voltage-dependent tau
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_p)               # (batch_size,)

		# ── Return voltage trace with optional observation noise ───────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)