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
		Hodgkin-Huxley neuron simulator extended with an M-type slow K+ current (IM).

		Physiological rationale for M-current addition:
		  - IM is a non-inactivating, voltage-gated K+ current that activates slowly
		    around subthreshold voltages (~-35 mV half-activation).
		  - It provides spike-frequency adaptation and stabilises the resting potential,
		    producing regular tonic spiking without bursting.
		  - Addresses discrepancies in: mean resting potential, voltage variance/skewness,
		    and spike count statistics.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) # injected current (uA/cm2)
			dt: float # time step (ms)
			t: torch.Tensor: (time_steps,) # time array (ms)
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces (mV)
		"""
		device = params.device

		# Set up random generator for stochastic noise
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ── Base HH parameters ───────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  fast Na+ conductance (mS/cm2)
		gbar_K    = params[:, 1].float()   # (batch_size,)  delayed-rectifier K+ conductance (mS/cm2)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm2)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal potential (mV)
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold shift (mV)
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude (unitless)

		# ── M-current parameters (X1 slot) ───────────────────────────────────
		# gbar_M  : maximal M-current conductance (mS/cm2), range [1e-4, 10]
		# V_half_M: half-activation voltage for M-current (mV, negative, range [-150, -1e-4])
		#           param_i = -params[:,8], so typical values around -35 mV are accessible
		# tau_M_scale: time-constant scale for M-current gating (ms, positive via -param_j)
		#              range of -param_j: [1e-4, 3000]; at half-activation tau_p = tau_M_scale/2
		#              typical M-current time constants: 50–500 ms
		gbar_M      = params[:, 6].float()   # (batch_size,)  M-current conductance (mS/cm2)
		# X2 slot intentionally unused to maintain parsimony
		# gbar_X2   = params[:, 7]  -- not used
		V_half_M    = -params[:, 8].float()  # (batch_size,)  M-current half-activation (mV)
		tau_M_scale = -params[:, 9].float()  # (batch_size,)  M-current time-scale (ms); positive

		tstep = float(dt)  # scalar float

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (kept at 0 as instructed)
		C    = 1.0            # membrane capacitance (uF/cm2)
		E_Na = 53.0           # Na+ reversal potential (mV)
		E_K  = -107.0         # K+  reversal potential (mV)
		# M-current flows through K+ channels, so reversal = E_K

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; clamps at -500 to avoid underflow
			# z: (batch_size,)  →  return: (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable evaluation of z/(exp(z)-1) used in HH alpha/beta rates
			# z: (batch_size,)  →  return: (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH kinetics ──────────────────────────────────────────────
		def alpha_m(x):
			# Na+ fast activation opening rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ fast activation closing rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation opening rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation closing rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ delayed-rectifier activation opening rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ delayed-rectifier activation closing rate  (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gating time constant from rates  (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Steady-state gating variable from rates  (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current kinetics (X1 slot) ──────────────────────────────────────
		# M-current uses a single gating variable p (slow activation, no inactivation).
		# alpha_p and beta_p are symmetric exponentials around V_half_M, giving:
		#   p_inf(V) = 1 / (1 + exp(-0.08*(V - V_half_M)))   [sigmoid, k=12.5 mV]
		#   tau_p(V) = tau_M_scale / (2*cosh(0.04*(V - V_half_M)))
		#            = tau_M_scale / 2  at V = V_half_M  (slowest point)
		# This formulation ensures slow subthreshold activation consistent with
		# known M-current properties (Brown & Adams 1980; Wang 1998).

		def alpha_p(x):
			# M-current activation opening rate  (batch_size,) → (batch_size,)
			# Rate scale = 1/tau_M_scale; symmetric exponential around V_half_M
			v1 = x - V_half_M   # (batch_size,)
			return Exp(0.04 * v1) / tau_M_scale   # (batch_size,)

		def beta_p(x):
			# M-current activation closing rate  (batch_size,) → (batch_size,)
			v1 = x - V_half_M   # (batch_size,)
			return Exp(-0.04 * v1) / tau_M_scale  # (batch_size,)

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) membrane voltage (mV)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current activation

		# ── Initialise at steady state from init_voltage ──────────────────────
		V_init = init_voltage.to(device)   # (batch_size,)
		V[:, 0] = V_init                   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		p[:, 0] = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))  # (batch_size,)  M-current at SS

		# ── Forward Euler with exponential integration ────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Evaluate gating rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)
			a_p, b_p = alpha_p(V_prev), beta_p(V_prev)   # (batch_size,), (batch_size,)  M-current rates

			# Effective membrane conductance inverse time constant (1/ms)
			# tau_V_inv = sum(g_i) / C
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  Na+ contribution
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)  K+  contribution
				+ g_leak                                         # (batch_size,)  leak contribution
				+ gbar_M * p[:, i - 1]                          # (batch_size,)  M-current contribution
			) / C   # (batch_size,)

			# Voltage steady-state numerator: sum(g_i * E_i) + I_ext + noise
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,)
				+ g_leak * E_leak                                        # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                           # (batch_size,)  M-current drives to E_K
				+ input_current[:, i - 1]                               # (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential integration for voltage (exact for linear ODE within timestep)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential integration for gating variables
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			p[:, i] = inf_x(a_p, b_p) + (p[:, i - 1] - inf_x(a_p, b_p)) * Exp(-tstep / tau_x(a_p, b_p))  # (batch_size,)  M-current update

		# Return voltage traces with optional observation noise (currently 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)