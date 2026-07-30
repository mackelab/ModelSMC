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
		Simulates a Hodgkin-Huxley neuron with M-current (I_M) for spike-frequency adaptation.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial voltage
			input_current: torch.Tensor: (batch_size, time_steps) # input current
			dt: float # time step size
			t: torch.Tensor: (time_steps,) # time array
			params: torch.Tensor: (batch_size, 10) # parameters
			seed: optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # voltage traces
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar
		time_steps = t.shape[0]  # scalar

		# Extract parameters
		gbar_Na   = params[:, 0].float()  # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()  # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()  # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float() # (batch_size,) mV
		Vt        = -params[:, 4].float() # (batch_size,) mV
		nois_fact = params[:, 5].float()  # (batch_size,) unitless
		# M-current (slow muscarinic K+ adaptation current)
		gbar_M    = params[:, 6].float()  # (batch_size,) mS/cm2, renamed from gbar_X1, range [1e-4, 10]
		# gbar_X2 unused (second channel not needed)
		param_i   = -params[:, 8].float() # (batch_size,) half-activation voltage shift, range [1e-4, 150] -> negative
		param_j   = -params[:, 9].float() # (batch_size,) time constant scaling, range [1e-4, 3000] -> negative

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV

		####################################
		# Numerical helpers
		def Exp(z):
			# z: any shape
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# z: any shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# Standard HH channel kinetics
		def alpha_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 13.0  # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch_size,)

		def alpha_h(x):
			# x: (batch_size,)
			v1 = x - Vt - 17.0  # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch_size,)

		def beta_h(x):
			# x: (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))  # (batch_size,)

		def alpha_n(x):
			# x: (batch_size,)
			v1 = x - Vt - 15.0  # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch_size,)

		def beta_n(x):
			# x: (batch_size,)
			v1 = x - Vt - 10.0  # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # (batch_size,)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,)
			return 1.0 / (alpha + beta)  # (batch_size,)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,)
			return alpha / (alpha + beta)  # (batch_size,)

		# M-current (I_M) kinetics: slow voltage-gated K+ channel for adaptation
		# p_inf: sigmoid steady-state activation around param_i (half-activation voltage)
		def p_inf(x):
			# x: (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - param_i) / 10.0))  # (batch_size,)

		# tau_p: bell-shaped voltage-dependent time constant scaled by param_j
		def tau_p(x):
			# x: (batch_size,)
			v1 = (x - param_i) / 20.0  # (batch_size,)
			denom = 3.3 * Exp(v1) + Exp(-v1)  # (batch_size,)
			# clamp denominator to avoid div-by-zero
			denom = torch.clamp(denom, min=1e-6)  # (batch_size,)
			return torch.abs(param_j) / denom  # (batch_size,)

		####################################

		# State variable allocation
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate

		# Initialization at steady state
		V_init = init_voltage.to(device)  # (batch_size,)
		V[:, 0] = V_init                                              # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))           # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))           # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))           # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                      # (batch_size,)

		# Simulation loop
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Compute gate alphas/betas at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# M-current gate steady-state and time constant at previous voltage
			p_ss  = p_inf(V_prev)   # (batch_size,)
			tau_p_val = tau_p(V_prev)  # (batch_size,)

			# Conductance contributions for effective time constant inverse
			g_Na_eff = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,)
			g_K_eff  = (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)
			g_M_eff  = p[:, i - 1] * gbar_M                         # (batch_size,)

			tau_V_inv = (
				g_Na_eff
				+ g_K_eff
				+ g_leak
				+ g_M_eff  # M-current contribution to membrane conductance
			) / C  # (batch_size,)

			# Noise sample
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)

			# Voltage steady-state numerator
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff * E_K
				+ g_leak * E_leak
				+ g_M_eff * E_K  # M-current reversal = E_K
				+ input_current[:, i - 1]
				+ noise
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential-Euler updates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                        # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))       # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))       # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))       # (batch_size,)
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_p_val)                                     # (batch_size,)

		# Return voltage with optional observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)