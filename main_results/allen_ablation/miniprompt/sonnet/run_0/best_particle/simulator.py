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
		Hodgkin-Huxley neuron extended with M-current (slow muscarinic K+).

		Args:
			init_voltage: torch.Tensor: (batch_size,)       initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps)  injected current (uA/cm2)
			dt: float                                        time step (ms)
			t: torch.Tensor: (time_steps,)                  time array (ms)
			params: torch.Tensor: (batch_size, 10)           biophysical parameters
			seed: int or None                                random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)        membrane voltage (mV)
		"""
		device = params.device

		# Random generator setup
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ------------------------------------------------------------------ #
		# Parameter extraction
		# ------------------------------------------------------------------ #
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless
		gbar_M    = params[:, 6].float()   # (batch_size,) mS/cm²  M-current conductance
		# params[:, 7] (gbar_X2) unused
		param_i   = -params[:, 8].float()  # (batch_size,) mV  M-current half-activation voltage
		# params[:, 9] (param_j) unused

		tstep = float(dt)

		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV

		# ------------------------------------------------------------------ #
		# Kinetics helpers
		# ------------------------------------------------------------------ #
		CLIP_LO = -5e2
		CLIP_HI = 88.0  # float32 exp overflows above ~88.7

		def Exp(z):
			# z: any shape — clamp both ends to prevent float32 under/overflow
			z_safe = torch.clamp(z, min=CLIP_LO, max=CLIP_HI)  # same shape as z
			return torch.exp(z_safe)  # same shape as z

		def efun(z):
			# z: any shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))  # same shape as z

		# Na+ channel gates
		def alpha_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 13.0                        # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25     # (batch_size,)

		def beta_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 40.0                        # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2        # (batch_size,)

		def alpha_h(x):
			# x: (batch_size,)
			v1 = x - Vt - 17.0                        # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)            # (batch_size,)

		def beta_h(x):
			# x: (batch_size,)
			v1 = x - Vt - 40.0                        # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))       # (batch_size,)

		# K+ delayed-rectifier gate
		def alpha_n(x):
			# x: (batch_size,)
			v1 = x - Vt - 15.0                        # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2      # (batch_size,)

		def beta_n(x):
			# x: (batch_size,)
			v1 = x - Vt - 10.0                        # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)              # (batch_size,)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,)
			return 1.0 / (alpha + beta)                # (batch_size,)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,)
			return alpha / (alpha + beta)              # (batch_size,)

		# M-current (slow muscarinic K+) gate — Boltzmann steady-state + bell-shaped tau
		def p_inf(x):
			# x: (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - param_i) / 10.0))  # (batch_size,)

		def tau_p(x):
			# x: (batch_size,)
			# Clamp from below at 0.1 ms to prevent numerical blow-up in gate update
			raw = 1.0 / (
				3.3 * Exp((x - param_i) / 20.0) + Exp(-(x - param_i) / 20.0)
			)  # (batch_size,)
			return torch.clamp(raw, min=0.1)           # (batch_size,)

		# ------------------------------------------------------------------ #
		# State variable allocation
		# ------------------------------------------------------------------ #
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na act.
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na inact.
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M gate

		# ------------------------------------------------------------------ #
		# Initialisation at t = 0
		# ------------------------------------------------------------------ #
		V_init  = init_voltage.to(device)                          # (batch_size,)
		V[:, 0] = V_init                                           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                   # (batch_size,)

		# ------------------------------------------------------------------ #
		# Simulation loop — exponential-Euler integration
		# ------------------------------------------------------------------ #
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Gate kinetics at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			p_inf_v = p_inf(V_prev)  # (batch_size,)
			tau_p_v = tau_p(V_prev)  # (batch_size,)  already clamped >= 0.1 ms

			# Effective conductance sum divided by C  (inverse membrane time constant)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,) Na current
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) K delayed-rectifier
				+ g_leak                                        # (batch_size,) leak
				+ gbar_M * p[:, i - 1]                         # (batch_size,) M-current
			) / C  # (batch_size,)

			# Noise sample for this time step
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)  # (batch_size,)

			# Voltage quasi-steady-state numerator (divided by tau_V_inv * C below)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na  # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)
				+ g_leak * E_leak                                       # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                          # (batch_size,) M-current
				+ input_current[:, i - 1]                              # (batch_size,)
				+ noise                                                 # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential-Euler updates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                       # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))     # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))     # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))     # (batch_size,)
			p[:, i] = p_inf_v + (p[:, i - 1] - p_inf_v) * Exp(-tstep / tau_p_v)                               # (batch_size,)

		# Optional observation noise (currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)