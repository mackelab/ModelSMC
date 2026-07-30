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
		Hodgkin-Huxley neuron with M-current (IKM) for regular tonic spiking.

		The M-current (Kv7/KCNQ) is a slow, non-inactivating K+ conductance that:
		- Activates near -35 mV (subthreshold), well below spike threshold
		- Has a slow time constant (50-500 ms), providing spike-frequency adaptation
		- Does NOT produce bursting — it suppresses high-frequency firing smoothly
		- Improves mean voltage, variance, skewness, kurtosis statistics simultaneously

		Parameter allocation for M-current:
		  gbar_KM  = params[:,6]  (X1 conductance slot,  range [1e-4, 10]  mS/cm2)
		  V_half_M = -params[:,7] (X2 conductance slot repurposed as voltage, range [-120, 0] mV)
		  tau_KM   = params[:,9]  (param_j slot,          range [1e-4, 3000] ms)
		  params[:,8] (param_i) is left unused — kept free for future use

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage in mV
			input_current: torch.Tensor: (batch_size, time_steps) # applied current in uA/cm2
			dt: float # integration time step in ms
			t: torch.Tensor: (time_steps,) # time array in ms
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int for reproducible stochastic noise

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces in mV
		"""
		device = params.device

		# Random generator setup for reproducible noise
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]         # scalar int

		# ---- Base Hodgkin-Huxley parameters ----
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2, Na+ max conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2, K+ delayed rectifier
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2, passive leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV, leak reversal potential (sign flip)
		Vt        = -params[:, 4].float()  # (batch_size,) mV, voltage threshold offset (sign flip)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless, noise amplitude scaling

		# ---- M-current (IKM) parameters ----
		# Conductance: X1 slot, range [1e-4, 10] mS/cm2
		# Typical cortical M-current: 0.5–5 mS/cm2
		gbar_KM  = params[:, 6].float()   # (batch_size,) mS/cm2

		# Half-activation voltage: X2 conductance slot REPURPOSED as a voltage parameter
		# params[:,7] is positive in [1e-4, 120], so -params[:,7] covers [-120, ~0] mV
		# Typical M-current V_half: -35 mV, comfortably within this range
		V_half_M = -params[:, 7].float()  # (batch_size,) mV, M-current Boltzmann midpoint

		# params[:,8] (param_i) is intentionally unused — left as a free slot
		# This avoids overparameterizing the M-current and improves identifiability

		# Time constant: param_j slot, range [1e-4, 3000] ms
		# M-current is notably slow: 50–500 ms; inference will find the correct scale
		# Voltage-independent tau keeps the formulation parsimonious
		tau_KM   = params[:, 9].float()   # (batch_size,) ms, voltage-independent M-current tau

		tstep = float(dt)

		# ---- Fixed biophysical constants ----
		nois_fact_obs = 0.0   # observation noise (zero per specification)
		C    = 1.0            # uF/cm², membrane capacitance
		E_Na = 53.0           # mV, Na+ reversal potential
		E_K  = -107.0         # mV, K+ reversal (shared by IK and IKM — both are K+ channels)

		####################################
		# Numerical helper functions

		def Exp(z):
			# Numerically stable exponential: clips at -500 to prevent underflow
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)  # (same shape as z)

		def efun(z):
			# Hodgkin-Huxley rate helper: avoids 0/0 near z=0 via Taylor expansion
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,
				z / (Exp(z) - 1.0)
			)  # (same shape as z)

		####################################
		# Standard HH channel kinetics (unchanged from base model)

		def alpha_m(x):
			# (batch_size,) -> (batch_size,): Na+ activation opening rate
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# (batch_size,) -> (batch_size,): Na+ activation closing rate
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# (batch_size,) -> (batch_size,): Na+ inactivation onset rate
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# (batch_size,) -> (batch_size,): Na+ inactivation recovery rate
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# (batch_size,) -> (batch_size,): K+ delayed rectifier activation rate
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# (batch_size,) -> (batch_size,): K+ delayed rectifier deactivation rate
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,) -> (batch_size,): gating variable time constant
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,) -> (batch_size,): gating variable steady-state value
			return alpha / (alpha + beta)

		# ===== BEGIN EDITABLE SECTION =====
		# M-current (IKM) kinetics
		# Boltzmann steady-state with 10 mV slope factor — standard for Kv7 channels
		# V_half_M is inferred; slope factor 10 mV is fixed (well-established physiologically)
		def w_inf(x):
			# (batch_size,) -> (batch_size,): M-current steady-state activation (sigmoid)
			# Activates subthreshold: w_inf = 0.5 at V = V_half_M (~-35 mV typically)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))
		# ===== END EDITABLE SECTION =====

		####################################
		# Allocate state variable arrays

		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ activation
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation

		# ===== BEGIN EDITABLE SECTION =====
		# M-current gating variable (slow activation, no inactivation)
		w = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current activation
		# ===== END EDITABLE SECTION =====

		####################################
		# Initialize all state variables at steady state for init_voltage

		V_init = init_voltage.to(device)                                             # (batch_size,)
		V[:, 0] = V_init                                                              # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                          # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                          # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                          # (batch_size,)

		# ===== BEGIN EDITABLE SECTION =====
		# M-current initialized at its voltage-dependent steady state
		w[:, 0] = w_inf(V[:, 0])  # (batch_size,)
		# ===== END EDITABLE SECTION =====

		####################################
		# Main simulation loop — exponential integration (exact for piecewise-linear)

		for i in range(1, time_steps):
			# Compute gating rates at previous timestep voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])  # (batch_size,), (batch_size,)

			# ===== BEGIN EDITABLE SECTION =====
			# M-current steady-state at current voltage (used for exponential update below)
			w_ss = w_inf(V[:, i - 1])  # (batch_size,): M-current w_inf at V[i-1]
			# ===== END EDITABLE SECTION =====

			# Effective inverse membrane time constant: sum of all active conductances / C
			# Used in exponential integration: V(t+dt) = V_inf + (V - V_inf)*exp(-dt*tau_V_inv)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ conductance
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) K+ delayed rectifier
				+ g_leak                                         # (batch_size,) leak conductance
				# ===== BEGIN EDITABLE SECTION =====
				+ w[:, i - 1] * gbar_KM                         # (batch_size,) M-current K+ conductance
				# ===== END EDITABLE SECTION =====
			) / C  # (batch_size,)

			# Effective voltage steady-state: weighted sum of reversal potentials + inputs
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ contribution
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ contribution
				+ g_leak * E_leak                                        # (batch_size,) leak contribution
				# ===== BEGIN EDITABLE SECTION =====
				+ w[:, i - 1] * gbar_KM * E_K                          # (batch_size,) M-current contribution (reversal = E_K)
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                               # (batch_size,) external current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Voltage update — exponential Euler (exact for linear V equation)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Standard HH gating variable updates — exponential Euler
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# ===== BEGIN EDITABLE SECTION =====
			# M-current update — exponential Euler with voltage-independent tau_KM
			# tau_KM is slow (~50–500 ms), inferred by the optimizer via params[:,9]
			# Voltage-independence keeps the model parsimonious (no extra kinetics functions)
			w[:, i] = w_ss + (w[:, i - 1] - w_ss) * Exp(-tstep / tau_KM)  # (batch_size,)
			# ===== END EDITABLE SECTION =====

		# Return voltage traces with optional observation noise (currently disabled)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)