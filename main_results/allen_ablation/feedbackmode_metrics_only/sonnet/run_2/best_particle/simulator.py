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
		Hodgkin-Huxley neuron with A-type (transient) K+ current (I_A).

		Key structural change from previous iteration (a^3*b with constant tau_b):
		  - The A-current inactivation time constant tau_b is now VOLTAGE-DEPENDENT,
		    following a 1/cosh bell shape centered at the inactivation half-voltage:

		      tau_b(V) = tau_b_max / cosh((V - b_half) / 20.0)

		    This is physiologically accurate: I_A inactivation is fastest near the
		    half-activation/inactivation voltage and slows at hyperpolarized or deeply
		    depolarized potentials (Huguenard & McCormick 1992; Bhattacharjee & Bhattacharjee 2003).
		    This voltage-dependence of tau_b can better capture ISI shape, spike count
		    modulation, and voltage distribution statistics (skewness, kurtosis) compared
		    to a constant tau_b, because the inactivation rate adapts with the membrane
		    trajectory during each spike cycle.

		Parameterization:
		  - gbar_A  = params[:,6] (X1 slot, range [1e-4, 10] mS/cm2)
		  - b_half  = -params[:,8] (range [-150, ~0] mV; typical I_A b_half: -60 to -80 mV)
		  - tau_b_max = params[:,9] (range [1e-4, 3000] ms; clamped to [1.0, 1000] ms)

		I_A formulation:
		  - Activation: instantaneous a_inf(V) = 1/(1+exp(-(V+20)/8)) [Connor-Stevens, fixed]
		  - Inactivation: dynamic gate b with b_inf(V)=1/(1+exp((V-b_half)/6))
		    and tau_b(V) = clamp(tau_b_max,1,1000) / cosh((V-b_half)/20)
		  - Current: I_A = gbar_A * a_inf^3 * b * (V - E_K)

		Args:
		    init_voltage: torch.Tensor (batch_size,)
		    input_current: torch.Tensor (batch_size, time_steps)
		    dt: float (ms)
		    t: torch.Tensor (time_steps,)
		    params: torch.Tensor (batch_size, 10)
		    seed: optional int

		Returns:
		    V: torch.Tensor (batch_size, time_steps)
		"""
		device = params.device

		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# --- Base HH parameters ---
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# --- A-current parameters (X1 slot; X2 unused for parsimony) ---
		# gbar_A: maximal transient K+ conductance
		gbar_A    = params[:, 6].float()   # (batch_size,) mS/cm2, range [1e-4, 10]

		# b_half: inactivation Boltzmann half-voltage
		# params[:,8] is in [1e-4, 150], so -params[:,8] is in [-150, ~0] mV
		# Physiological I_A b_half: typically -60 to -80 mV
		b_half    = -params[:, 8].float()  # (batch_size,) mV

		# tau_b_max: peak inactivation time constant at V = b_half
		# params[:,9] is in [1e-4, 3000]; clamped to [1.0, 1000] ms for stability
		# Physiological I_A inactivation tau: ~10–300 ms
		tau_b_max = torch.clamp(params[:, 9].float(), min=1.0, max=1000.0)  # (batch_size,) ms

		tstep = float(dt)   # scalar ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV (shared by Kdr and A-current)

		# -------------------------------------------------------
		# Numerical helpers
		# -------------------------------------------------------
		def Exp(z):
			# Numerically stable exponential (batch_size,) -> (batch_size,)
			# Clips at -500 to avoid underflow without distorting positive values
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Handles z/(exp(z)-1) near z=0 via L'Hopital: limit = 1 - z/2
			# (batch_size,) -> (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# -------------------------------------------------------
		# Standard HH channel kinetics (unchanged from base model)
		# -------------------------------------------------------
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
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant from forward/backward rates
			# (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady state from forward/backward rates
			# (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# -------------------------------------------------------
		# A-current (I_A) kinetics — Connor-Stevens a^3*b formulation
		#   with VOLTAGE-DEPENDENT inactivation time constant
		#
		# Activation (instantaneous, fixed):
		#   a_inf(V) = 1 / (1 + exp(-(V + 20) / 8))
		#   Half-activation at -20 mV, slope 8 mV (Connor-Stevens canonical)
		#   Instantaneous because tau_a << tau_b and tau_membrane for I_A
		#
		# Inactivation (dynamic gate b):
		#   b_inf(V) = 1 / (1 + exp((V - b_half) / 6))   [tunable b_half]
		#
		#   tau_b(V) = tau_b_max / cosh((V - b_half) / 20)   [voltage-dependent]
		#
		#   Rationale for cosh form:
		#     - cosh((V-b_half)/20) = (exp((V-b_half)/20) + exp(-(V-b_half)/20)) / 2
		#     - Equals 1 at V=b_half (giving maximum tau = tau_b_max)
		#     - Increases symmetrically away from b_half, reducing tau at extreme voltages
		#     - Slope of 20 mV is physiologically appropriate for I_A inactivation
		#     - This means inactivation is SLOWEST near rest/threshold and FASTER
		#       at the peak of action potential and at hyperpolarized potentials
		#     - Qualitatively matches Huguenard & McCormick 1992 I_A voltage clamp data
		#     - tau_b is clamped to [0.5, tau_b_max] to prevent division instabilities
		#
		# Current: I_A = gbar_A * a_inf(V)^3 * b * (V - E_K)
		# -------------------------------------------------------
		def a_inf(x):
			# Instantaneous A-current activation steady state
			# (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp(-(x + 20.0) / 8.0))

		def b_inf(x):
			# A-current inactivation steady state (tunable half-voltage b_half)
			# (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp((x - b_half) / 6.0))

		def tau_b_v(x):
			# Voltage-dependent A-current inactivation time constant
			# Bell-shaped: maximum tau_b_max at V=b_half, decays symmetrically
			# (batch_size,) -> (batch_size,)
			# cosh argument clamped to prevent overflow from cosh at extreme V
			cosh_arg = torch.clamp((x - b_half) / 20.0, min=-20.0, max=20.0)  # (batch_size,)
			# cosh = (exp(z) + exp(-z)) / 2; minimum value is 1.0 at z=0
			cosh_val = 0.5 * (Exp(cosh_arg) + Exp(-cosh_arg))   # (batch_size,), >= 1
			# tau_b(V) = tau_b_max / cosh(...); clamped from below to avoid vanishing tau
			return torch.clamp(tau_b_max / cosh_val, min=0.5)   # (batch_size,) ms

		# -------------------------------------------------------
		# State variable allocation
		# -------------------------------------------------------
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) membrane voltage
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Kdr activation
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inactivation
		b = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) I_A inactivation

		# -------------------------------------------------------
		# Initialization at steady-state
		# -------------------------------------------------------
		V_init  = init_voltage.to(device)           # (batch_size,)
		V[:, 0] = V_init                             # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		# I_A inactivation initialized at Boltzmann steady state for resting potential
		# At hyperpolarized rest (~-65 mV), b_inf is close to 1 (channel available)
		b[:, 0] = b_inf(V[:, 0])   # (batch_size,)

		# -------------------------------------------------------
		# Integration loop — Exponential Euler method
		# -------------------------------------------------------
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) previous step voltage

			# Standard HH gating rates at previous voltage
			a_m_v, b_m_v = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,)
			a_h_v, b_h_v = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,)
			a_n_v, b_n_v = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,)

			# I_A gates evaluated at previous voltage
			a_A     = a_inf(V_prev)     # (batch_size,) instantaneous activation (no dynamics)
			b_A_ss  = b_inf(V_prev)     # (batch_size,) inactivation steady state
			tau_b   = tau_b_v(V_prev)   # (batch_size,) voltage-dependent inactivation tau

			# A-current conductance factor: a^3 * b (Connor-Stevens formulation)
			g_A_factor = (a_A ** 3) * b[:, i - 1]   # (batch_size,)

			# Effective conductance reciprocal time constant for exponential Euler
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # Na channel    (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                  # Kdr channel   (batch_size,)
				+ g_leak                                          # passive leak  (batch_size,)
				+ g_A_factor * gbar_A                            # I_A channel   (batch_size,)
			) / C   # (batch_size,)

			# Voltage steady-state numerator (divided by tau_V_inv * C below)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # Na driving force   (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # Kdr driving force  (batch_size,)
				+ g_leak * E_leak                                        # leak driving force (batch_size,)
				+ g_A_factor * gbar_A * E_K                             # I_A driving force  (batch_size,)
				+ input_current[:, i - 1]                               # injected current   (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler updates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                                # (batch_size,)
			n[:, i] = inf_x(a_n_v, b_n_v) + (n[:, i - 1] - inf_x(a_n_v, b_n_v)) * Exp(-tstep / tau_x(a_n_v, b_n_v))  # (batch_size,)
			m[:, i] = inf_x(a_m_v, b_m_v) + (m[:, i - 1] - inf_x(a_m_v, b_m_v)) * Exp(-tstep / tau_x(a_m_v, b_m_v))  # (batch_size,)
			h[:, i] = inf_x(a_h_v, b_h_v) + (h[:, i - 1] - inf_x(a_h_v, b_h_v)) * Exp(-tstep / tau_x(a_h_v, b_h_v))  # (batch_size,)
			# I_A inactivation: exponential Euler with VOLTAGE-DEPENDENT tau_b
			b[:, i] = b_A_ss + (b[:, i - 1] - b_A_ss) * Exp(-tstep / tau_b)   # (batch_size,)

		# Return voltage trace with optional observation noise (currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)