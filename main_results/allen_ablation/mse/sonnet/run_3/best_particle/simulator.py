import math
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
		Hodgkin-Huxley neuron with two additional currents:

		1. M-type K+ current (IM, slot X1):
		   - Slow, non-inactivating, single gate p
		   - Corrects voltage distribution skewness/kurtosis and after-hyperpolarization shape
		   - param_i tunes half-activation voltage; param_j tunes time constant (clamped to [25, 300] ms)
		   - Biophysical tau_M range prevents M-current from collapsing to fast-rectifier behavior

		2. A-type K+ current (IA, slot X2):
		   - Fast transient K+ current with activation (aA) and inactivation (bA) gates
		   - Provides rapid repolarization after each spike, directly controlling ISI and spike count
		   - Fixed kinetic parameters (only gbar_IA is inferred); avoids identifiability issues
		   - Does NOT cause bursting — inactivation removes it between spikes cleanly

		Args:
			init_voltage: torch.Tensor (batch_size,) — initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) — injected current (uA/cm2)
			dt: float — time step (ms)
			t: torch.Tensor (time_steps,) — time array (ms)
			params: torch.Tensor (batch_size, 10) — biophysical parameters
			seed: int or None — random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps) — membrane voltage traces (mV)
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

		# ----------------------------------------------------------------
		# Parameter extraction
		# ----------------------------------------------------------------
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# Slot X1: M-current conductance (inferred), range [1e-4, 10] mS/cm2
		gbar_IM  = params[:, 6].float()   # (batch_size,) mS/cm2

		# Slot X2: A-current conductance (inferred), range [1e-4, 120] mS/cm2
		# Rationale: spike-count MSE requires additional fast K+ conductance beyond IM alone
		gbar_IA  = params[:, 7].float()   # (batch_size,) mS/cm2

		# param_i: M-current half-activation shift; param_i in [-150, -1e-4] mV
		#   effective V_half = param_i + 35.0 mV -> range [-115, ~35] mV
		param_i  = -params[:, 8].float()  # (batch_size,)

		# param_j: M-current time constant; -param_j/10 clamped to [25, 300] ms
		#   Enforcing tau_p >= 25 ms ensures realistic slow adaptation (not a second DR)
		param_j  = -params[:, 9].float()  # (batch_size,)

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm2
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV (reversal for all K+ currents: DR, M, A)

		# Precompute fixed exponential decay factors for IA gates (scalar, computed once)
		# tau_a_IA = 2 ms (fast activation), tau_b_IA = 20 ms (slower inactivation)
		# Using fixed time constants keeps IA simple — only gbar_IA is inferred
		tau_a_IA = 2.0    # ms — fast transient activation of A-current
		tau_b_IA = 20.0   # ms — A-current inactivation time constant
		exp_aIA = math.exp(-tstep / tau_a_IA)   # scalar Python float
		exp_bIA = math.exp(-tstep / tau_b_IA)   # scalar Python float

		# ----------------------------------------------------------------
		# Kinetic helper functions
		# ----------------------------------------------------------------
		def Exp(z):
			# Numerically stable exponential; z: any tensor shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable z / (exp(z) - 1); z: any tensor shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# Na+ channel kinetics (m: activation, h: inactivation)
		def alpha_m(x):
			v1 = x - Vt - 13.0    # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			v1 = x - Vt - 40.0    # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			v1 = x - Vt - 17.0    # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			v1 = x - Vt - 40.0    # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))

		# K+ delayed-rectifier kinetics (n: activation)
		def alpha_n(x):
			v1 = x - Vt - 15.0    # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			v1 = x - Vt - 10.0    # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			return alpha / (alpha + beta)  # (batch_size,)

		# M-current (IM) kinetics — slow non-inactivating K+ gate p
		# Sigmoid steady state; half-activation = param_i + 35 mV; slope = 10 mV
		def p_inf(x):
			# x: (batch_size,)
			v_half = param_i + 35.0   # (batch_size,) half-activation voltage (mV)
			return 1.0 / (1.0 + Exp(-(x - v_half) / 10.0))

		def tau_p_fn():
			# Voltage-independent; clamp to [25, 300] ms for biophysical realism
			# tau_p < 25 ms would make IM resemble a fast channel (not M-type behavior)
			return torch.clamp(-param_j / 10.0, min=25.0, max=300.0)  # (batch_size,)

		# A-type K+ current (IA) kinetics — fast transient K+ gate pair (aA, bA)
		# Activation: fast sigmoid centered at -30 mV (slope 15 mV); tau = 2 ms (fixed)
		# Inactivation: sigmoid centered at -60 mV (slope -8 mV); tau = 20 ms (fixed)
		# The A-current inactivates between spikes, providing ISI control without bursting
		def aA_inf(x):
			# x: (batch_size,)
			return 1.0 / (1.0 + Exp(-(x + 30.0) / 15.0))

		def bA_inf(x):
			# x: (batch_size,)
			return 1.0 / (1.0 + Exp((x + 60.0) / 8.0))

		# ----------------------------------------------------------------
		# Allocate state variable tensors
		# ----------------------------------------------------------------
		V  = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m  = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h  = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n  = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p  = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate
		aA = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA activation
		bA = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA inactivation

		# ----------------------------------------------------------------
		# Initialize all gates at steady state
		# ----------------------------------------------------------------
		V_init = init_voltage.to(device)   # (batch_size,)
		V[:, 0]  = V_init                                                     # (batch_size,)
		m[:, 0]  = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                  # (batch_size,)
		h[:, 0]  = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                  # (batch_size,)
		n[:, 0]  = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                  # (batch_size,)
		p[:, 0]  = p_inf(V[:, 0])                                             # (batch_size,)
		aA[:, 0] = aA_inf(V[:, 0])                                            # (batch_size,)
		bA[:, 0] = bA_inf(V[:, 0])                                            # (batch_size,)

		# Precompute M-current time constant (voltage-independent)
		tau_p = tau_p_fn()   # (batch_size,)

		# ----------------------------------------------------------------
		# Main simulation loop — exponential Euler integration
		# ----------------------------------------------------------------
		for i in range(1, time_steps):
			V_prev  = V[:, i - 1]    # (batch_size,)
			m_prev  = m[:, i - 1]    # (batch_size,)
			h_prev  = h[:, i - 1]    # (batch_size,)
			n_prev  = n[:, i - 1]    # (batch_size,)
			p_prev  = p[:, i - 1]    # (batch_size,)
			aA_prev = aA[:, i - 1]   # (batch_size,)
			bA_prev = bA[:, i - 1]   # (batch_size,)

			# Gating variable rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,)

			# M-gate and IA-gate steady states at current voltage
			p_ss  = p_inf(V_prev)    # (batch_size,)
			aA_ss = aA_inf(V_prev)   # (batch_size,)
			bA_ss = bA_inf(V_prev)   # (batch_size,)

			# Effective conductances (batch_size,)
			g_Na_eff = (m_prev ** 3) * gbar_Na * h_prev    # (batch_size,) fast Na+
			g_K_eff  = (n_prev ** 4) * gbar_K              # (batch_size,) delayed-rectifier K+
			g_IM_eff = gbar_IM * p_prev                     # (batch_size,) M-current K+
			g_IA_eff = gbar_IA * (aA_prev ** 3) * bA_prev  # (batch_size,) A-current K+

			# Effective inverse membrane time constant
			tau_V_inv = (g_Na_eff + g_K_eff + g_leak + g_IM_eff + g_IA_eff) / C  # (batch_size,)

			# Stochastic noise term
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)

			# Voltage steady-state (weighted sum of reversal potentials + inputs)
			V_inf = (
				g_Na_eff * E_Na           # Na+ drives toward E_Na = +53 mV
				+ g_K_eff  * E_K          # DR K+ drives toward E_K = -107 mV
				+ g_leak   * E_leak       # leak drives toward E_leak
				+ g_IM_eff * E_K          # M-current K+ drives toward E_K
				+ g_IA_eff * E_K          # A-current K+ drives toward E_K
				+ input_current[:, i - 1] # (batch_size,) injected current
				+ noise                   # (batch_size,) stochastic component
			) / (tau_V_inv * C)           # (batch_size,)

			# Exponential Euler updates
			V[:, i]  = V_inf + (V_prev  - V_inf)  * Exp(-tstep * tau_V_inv)                             # (batch_size,)
			m[:, i]  = inf_x(a_m, b_m) + (m_prev  - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i]  = inf_x(a_h, b_h) + (h_prev  - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i]  = inf_x(a_n, b_n) + (n_prev  - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			p[:, i]  = p_ss  + (p_prev  - p_ss)  * Exp(-tstep / tau_p)                                  # (batch_size,)
			# IA gates use precomputed scalar exponential factors (fixed tau => constant decay)
			aA[:, i] = aA_ss + (aA_prev - aA_ss) * exp_aIA   # (batch_size,)
			bA[:, i] = bA_ss + (bA_prev - bA_ss) * exp_bIA   # (batch_size,)

		# Return voltage traces with optional observation noise (currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)