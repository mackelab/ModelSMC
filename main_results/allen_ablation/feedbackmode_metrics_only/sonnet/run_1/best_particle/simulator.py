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
		Hodgkin-Huxley neuron extended with an M-type K+ current (IM / IKM).

		Physiological rationale for IM:
		  - IM is a slow, non-inactivating (persistent) K+ current present in many
		    tonic-firing cortical and peripheral neurons (Kv7/KCNQ channels).
		  - It activates at subthreshold voltages (-60 to -30 mV range), providing
		    a steady outward current that regulates spike threshold and inter-spike
		    intervals without inducing burst firing or suppressing spikes entirely.
		  - Unlike IA (transient, inactivating), IM is persistent — a single gate 'p'
		    with no inactivation variable — making it simpler and more identifiable.
		  - IM primarily shapes: spike count, mean stimulation voltage, and the
		    subthreshold voltage distribution (mean, variance, skewness, kurtosis).
		  - Half-activation voltage controlled by param_i (V_half = param_i - 60),
		    activation time constant controlled by param_j (ms).

		Args:
		    init_voltage: torch.Tensor: (batch_size,)              initial membrane voltage (mV)
		    input_current: torch.Tensor: (batch_size, time_steps)  injected current (uA/cm2)
		    dt: float                                               time step (ms)
		    t: torch.Tensor: (time_steps,)                         time array (ms)
		    params: torch.Tensor: (batch_size, 10)                 biophysical parameters
		    seed: int or None                                       random seed

		Returns:
		    V: torch.Tensor: (batch_size, time_steps)              voltage traces (mV)
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

		# ---- Base HH parameters ----
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ max conductance (mS/cm2)
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ delayed-rectifier conductance (mS/cm2)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm2)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal potential (mV)
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold shift (mV)
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude (unitless)

		# ---- M-type K+ current parameters ----
		# Slot X1: gbar_KM — max conductance of IM (mS/cm2), range [1e-4, 10]
		# param_i (positive, range [1e-4, 150]): sets half-activation voltage
		#   V_half = param_i - 60  →  range [-60 + 0 ≈ -60 mV] to [-60 + 150 = +90 mV]
		#   Physiologically reasonable: IM activates around -45 to -20 mV, so
		#   inference will select param_i ~ 15-40 for V_half ~ -45 to -20 mV.
		# param_j (positive, range [1e-4, 3000]): slow activation time constant (ms)
		#   IM is characteristically slow: tau_p ~ 20-300 ms (will be inferred)
		gbar_KM = params[:, 6].float()    # (batch_size,)  IM max conductance (mS/cm2)
		param_i = params[:, 8].float()    # (batch_size,)  half-activation offset (mV), positive
		param_j = params[:, 9].float()    # (batch_size,)  activation time constant (ms), positive

		tstep = float(dt)

		# Fixed constants
		nois_fact_obs = 0.0
		C    = 1.0     # membrane capacitance (uF/cm2)
		E_Na = 53.0    # Na+ reversal potential (mV)
		E_K  = -107.0  # K+ reversal potential (mV), shared by delayed-rectifier and IM

		####################################
		# Numerical helpers

		def Exp(z):
			# (batch_size,) -> (batch_size,)  numerically stable exponential
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (batch_size,) -> (batch_size,)  auxiliary function for HH rate equations
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		####################################
		# Standard HH kinetics for Na+ (m, h) and K+ delayed-rectifier (n)

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
			return 4.0 / (1 + Exp(-0.2 * v1))

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

		####################################
		# M-type K+ current (IM) kinetics — single non-inactivating gate p
		#
		# Steady-state activation:
		#   p_inf(V) = 1 / (1 + exp(-(V - V_half) / 10))
		# where V_half = param_i - 60 (mV) — allows inference over a wide range
		#
		# Activation time constant:
		#   tau_p = param_j (ms), constant — IM is characteristically slow
		#   Inference will drive param_j toward physiological values (20-300 ms)
		#
		# Current: I_KM = gbar_KM * p * (V - E_K)
		# Note: no inactivation gate, IM is persistent

		def p_inf(x):
			# x: (batch_size,) -> (batch_size,)  IM activation steady-state
			# V_half = param_i - 60: for param_i~15-40, V_half ~ -45 to -20 mV
			V_half = param_i - 60.0  # (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half) / 10.0))

		# tau_p = param_j (ms), constant scalar per batch element
		# (batch_size,) — used directly in exponential integration
		tau_p = param_j  # (batch_size,)

		####################################
		# Allocate state variable arrays

		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) voltage (mV)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ n-gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ m-gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ h-gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IM p-gate

		####################################
		# Initialize at t=0 to steady-state values

		V_init = init_voltage.to(device)  # (batch_size,)
		V[:, 0] = V_init                                                           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                        # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                        # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                                   # (batch_size,)

		####################################
		# Time integration — exponential Euler method

		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH gate kinetics at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# IM conductance at current state: gKM = gbar_KM * p
			# (single gate, no inactivation — persistent current)
			g_KM_now = gbar_KM * p_prev   # (batch_size,)

			# Effective inverse membrane time constant (total conductance / C)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ contribution   (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K+ delayed-rectifier  (batch_size,)
				+ g_leak                            # leak               (batch_size,)
				+ g_KM_now                          # M-type K+          (batch_size,)
			) / C   # (batch_size,)

			# Noise sample for this time step
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)   # (batch_size,)

			# Effective voltage steady-state
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)
				+ g_leak * E_leak                           # (batch_size,)
				+ g_KM_now * E_K                            # IM drives toward E_K  (batch_size,)
				+ input_current[:, i - 1]                  # (batch_size,)
				+ noise                                     # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler update for voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler update for standard HH gates
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(
				-tstep / tau_x(a_n, b_n)
			)   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(
				-tstep / tau_x(a_m, b_m)
			)   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(
				-tstep / tau_x(a_h, b_h)
			)   # (batch_size,)

			# Exponential Euler update for IM activation gate p
			# tau_p is constant (param_j, ms) — slow dynamics characteristic of IM
			p_ss = p_inf(V_prev)   # (batch_size,)  steady-state at current voltage
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_p)   # (batch_size,)

		# Return voltage traces (observation noise currently 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)