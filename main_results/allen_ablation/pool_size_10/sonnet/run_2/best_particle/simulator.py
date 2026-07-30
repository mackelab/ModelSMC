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
		Hodgkin-Huxley neuron simulator extended with a slow M-type K+ current
		(Kv7/KCNQ) to regularize tonic spiking inter-spike intervals.

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

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ---- Extract base parameters ----
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (sign applied internally)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (sign applied internally)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# ---- M-type K+ current parameters (slot X1) ----
		# Physiological rationale:
		#   The M-current is a slowly-activating, non-inactivating K+ conductance
		#   found in cortical and hippocampal neurons. It activates in the
		#   subthreshold voltage range and provides a hyperpolarising drive that
		#   regularises inter-spike intervals, producing evenly-spaced tonic
		#   spiking without bursting — exactly matching the observed data character.
		#
		# gbar_M  : maximal M-current conductance (mS/cm²), range [1e-4, 10]
		# v_half_M: half-activation offset above Vt (mV), range [1e-4, 150]
		#           e.g. 35 mV ⇒ half-activation near Vt + 35 mV
		# tau_p   : slow time constant of p-gate (ms), range [1e-4, 3000]
		#           typical physiological range ~20–300 ms
		gbar_M   = params[:, 6].float()  # (batch_size,) mS/cm²
		# gbar_X2 slot left unused (parsimony)
		v_half_M = params[:, 8].float()  # (batch_size,) mV offset — NOTE: positive, no negation
		tau_p    = params[:, 9].float()  # (batch_size,) ms     — NOTE: positive, no negation

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV

		####################################
		# Numerical helpers
		def Exp(z):
			# Clamp to avoid overflow; (batch_size,) or broadcastable
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Exponential function helper for HH alpha/beta rates; shape preserved
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ---- Standard HH channel kinetics ----
		def alpha_m(x):  # x: (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):   # x: (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2     # (batch_size,)

		def alpha_h(x):  # x: (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)          # (batch_size,)

		def beta_h(x):   # x: (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))       # (batch_size,)

		def alpha_n(x):  # x: (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2    # (batch_size,)

		def beta_n(x):   # x: (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)            # (batch_size,)

		def tau_x(alpha, beta):  # (batch_size,) each
			return 1.0 / (alpha + beta)  # (batch_size,)

		def inf_x(alpha, beta):  # (batch_size,) each
			return alpha / (alpha + beta)  # (batch_size,)

		# ---- M-current gating kinetics ----
		# p_inf: sigmoid steady-state activation of M-gate
		#   Half-activation at V = Vt + v_half_M, slope = 10 mV (fixed, typical)
		#   This places threshold in the subthreshold range for regularisation
		def p_inf_fn(x):  # x: (batch_size,)
			# (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - Vt - v_half_M) / 10.0))

		# tau_p is a learnable slow time constant (ms); protected from zero by clamping
		tau_p_safe = torch.clamp(tau_p, min=1e-3)  # (batch_size,)

		####################################

		# ---- Allocate state arrays ----
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate

		# ---- Initialise at steady state ----
		V_init = init_voltage.to(device)  # (batch_size,)
		V[:, 0] = V_init                  # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		p[:, 0] = p_inf_fn(V[:, 0])                          # (batch_size,) M-gate at steady state

		# ---- Forward Euler + exponential integration loop ----
		for i in range(1, time_steps):
			Vi = V[:, i - 1]  # (batch_size,) current voltage

			# Standard HH gate rates
			a_m, b_m = alpha_m(Vi), beta_m(Vi)  # (batch_size,) each
			a_h, b_h = alpha_h(Vi), beta_h(Vi)  # (batch_size,) each
			a_n, b_n = alpha_n(Vi), beta_n(Vi)  # (batch_size,) each

			# M-gate steady state at current voltage
			p_ss = p_inf_fn(Vi)  # (batch_size,)

			# ---- Effective membrane conductance (inverse time constant) ----
			# Includes Na+, K+ (DR), leak, and M-current contributions
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) DR K+
				+ g_leak                                         # (batch_size,) leak
				+ gbar_M * p[:, i - 1]                          # (batch_size,) M-current
			) / C  # (batch_size,)

			# ---- Voltage steady state ----
			# Numerator includes reversal potentials weighted by conductances,
			# injected current, and stochastic noise
			noise_i = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)  # (batch_size,)

			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,)
				+ g_leak * E_leak                                        # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                           # (batch_size,) M reverses at E_K
				+ input_current[:, i - 1]                               # (batch_size,)
				+ noise_i                                                # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# ---- Exponential integration ----
			V[:, i] = V_inf + (Vi - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Standard HH gates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(
				-tstep / tau_x(a_n, b_n)
			)  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(
				-tstep / tau_x(a_m, b_m)
			)  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(
				-tstep / tau_x(a_h, b_h)
			)  # (batch_size,)

			# M-gate update: slow exponential relaxation toward p_ss with time constant tau_p
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(
				-tstep / tau_p_safe
			)  # (batch_size,)

		# Return voltage traces with (currently zero) observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)