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
		Hodgkin-Huxley neuron with an added M-current (IM, Kv7/KCNQ-type).

		Rationale for M-current addition:
		  - The base HH model can produce irregular inter-spike intervals and
		    incorrect voltage distribution statistics (skewness, kurtosis).
		  - The M-current is a slow, non-inactivating K+ current that:
		      * Provides gentle spike-frequency adaptation
		      * Regularizes tonic firing (stabilizes ISI)
		      * Does NOT produce bursting or suppress spiking
		  - It is one of the most common additions to HH models for cortical neurons.

		Args:
			init_voltage: torch.Tensor: (batch_size,)
			input_current: torch.Tensor: (batch_size, time_steps)
			dt: float
			t: torch.Tensor: (time_steps,)
			params: torch.Tensor: (batch_size, 10)
			seed: int or None

		Returns:
			V: torch.Tensor: (batch_size, time_steps)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# --- Extract base parameters ---
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (sign applied)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (sign applied)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# --- M-current (IM) parameters via X1 slot ---
		# gbar_M: slow K+ conductance  (range [1e-4, 10] mS/cm²)
		gbar_M = params[:, 6].float()      # (batch_size,) mS/cm²

		# param_i = -params[:,8], negative, range [-150, ~0] mV
		# Used as the half-activation voltage for the M-current gate p
		# Physiologically, M-current activates around -35 to -50 mV → param_i fits this range
		V_half_M = -params[:, 8].float()   # (batch_size,) mV, negative

		# param_j = -params[:,9], negative, range [-3000, ~0]
		# -param_j gives a positive time constant in [1e-4, 3000] ms
		# M-current has a characteristically slow time constant (~100-500 ms)
		tau_p_M  = -params[:, 9].float()   # (batch_size,) ms, positive (slow gate)

		# gbar_X2 and its param are unused (parsimony: one channel is sufficient)

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (shared reversal for K+ and M-current)

		# ----------------------------------------------------------------
		# Utility functions
		# ----------------------------------------------------------------
		def Exp(z):
			# Numerically safe exponential: (batch_size,) -> (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Hodgkin-Huxley helper: handles z~0 limit
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ----------------------------------------------------------------
		# Standard HH channel kinetics
		# ----------------------------------------------------------------
		def alpha_m(x):
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40)

		def tau_x(alpha, beta):
			# (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ----------------------------------------------------------------
		# M-current gate kinetics (p: single activation variable, no inactivation)
		# Boltzmann steady-state with slope factor 10 mV (typical for M-current)
		# Time constant is slow (tau_p_M), provided as a tunable parameter
		# ----------------------------------------------------------------
		def p_inf(x):
			# Sigmoid activation: (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		# ----------------------------------------------------------------
		# State variable allocation
		# ----------------------------------------------------------------
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na act
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inact
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current gate

		# ----------------------------------------------------------------
		# Initial conditions (steady-state at initial voltage)
		# ----------------------------------------------------------------
		V_init    = init_voltage.to(device)                         # (batch_size,)
		V[:, 0]   = V_init
		n[:, 0]   = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))       # (batch_size,)
		m[:, 0]   = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))       # (batch_size,)
		h[:, 0]   = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))       # (batch_size,)
		p[:, 0]   = p_inf(V[:, 0])                                  # (batch_size,)

		# ----------------------------------------------------------------
		# Simulation loop (exponential Euler integration)
		# ----------------------------------------------------------------
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Gate rate constants at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# Noise sample for this time step
			xi = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# ---- Effective membrane conductance (inverse time constant) ----
			# Includes Na, K, leak, and M-current contributions
			g_Na_eff = (m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)
			g_K_eff  = (n_prev ** 4) * gbar_K             # (batch_size,)
			g_M_eff  = p_prev * gbar_M                     # (batch_size,) slow K+ conductance

			tau_V_inv = (g_Na_eff + g_K_eff + g_leak + g_M_eff) / C   # (batch_size,)

			# ---- Effective voltage steady-state ----
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff * E_K
				+ g_leak * E_leak
				+ g_M_eff * E_K       # M-current reversal same as K+
				+ input_current[:, i - 1]
				+ xi
			) / (tau_V_inv * C)       # (batch_size,)

			# ---- Exponential Euler updates ----
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# M-current gate: exponential Euler with slow time constant tau_p_M
			p_ss    = p_inf(V_prev)                                            # (batch_size,)
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_p_M)          # (batch_size,)

		# Optional observation noise (currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)