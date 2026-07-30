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
		Simulates a Hodgkin-Huxley neuron with M-current for a specified time duration.

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
		time_steps = t.shape[0]       # scalar

		# Extract parameters
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless
		# M-current (slow muscarinic K+ channel responsible for spike-frequency adaptation)
		gbar_M    = params[:, 6].float()   # (batch_size,) mS/cm2, range [1e-4, 10]
		# params[:, 7] unused (gbar_X2 reserved)
		param_i   = -params[:, 8].float()  # (batch_size,) half-activation voltage (mV), negative
		param_j   = -params[:, 9].float()  # (batch_size,) M-current time constant (ms), positive

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV

		####################################
		# Numerical helpers
		def Exp(z):
			# z: any shape -> same shape
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# z: any shape -> same shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# Standard HH Na+ channel kinetics
		def alpha_m(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		# Standard HH K+ channel kinetics
		def alpha_n(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# M-current (I_M) kinetics: slow non-inactivating K+ channel
		# Sigmoid steady-state activation; voltage-independent time constant
		def p_inf_M(x):
			# x: (batch_size,), param_i: (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp((x - param_i) / (-10.0)))

		# tau_p_M: (batch_size,) — param_j is already positive after sign flip
		tau_p_M = param_j  # (batch_size,) ms, range [1e-4, 3000]

		####################################
		# Allocate state tensors
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate

		# Initialization at steady state
		V_init = init_voltage.to(device)        # (batch_size,)
		V[:, 0] = V_init                         # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		p[:, 0] = p_inf_M(V[:, 0])              # (batch_size,)

		# Simulation loop
		for i in range(1, time_steps):
			# Gate alpha/beta at previous voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])  # (batch_size,), (batch_size,)

			# M-gate steady-state at previous voltage
			p_inf_curr = p_inf_M(V[:, i - 1])  # (batch_size,)

			# Effective membrane conductance (1/tau_V) — includes M-current contribution
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)
				+ g_leak                                         # (batch_size,)
				+ gbar_M * p[:, i - 1]                          # (batch_size,) M-current
			) / C  # (batch_size,)

			# Effective steady-state voltage — includes M-current reversal term
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na  # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)
				+ g_leak * E_leak                                       # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                          # (batch_size,) M-current drives toward E_K
				+ input_current[:, i - 1]                               # (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential-Euler voltage update
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)          # (batch_size,)
			# Standard gate updates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			# M-gate update with voltage-independent time constant
			p[:, i] = p_inf_curr + (p[:, i - 1] - p_inf_curr) * Exp(-tstep / tau_p_M)  # (batch_size,)

		# Return voltage with optional observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)