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
		Hodgkin-Huxley neuron extended with an M-type slow K+ current (IKM).

		Extension rationale:
		- Classic HH (Na, K delayed-rectifier, leak) lacks a mechanism to regulate
		  inter-spike interval regularity and subthreshold voltage statistics.
		- IKM (KCNQ/Kv7): slow, non-inactivating, activates near spike threshold (~-35 mV).
		  It provides spike-frequency adaptation and stabilises tonic firing without
		  introducing burst patterns or sustained high-frequency firing.
		- IKM addresses discrepancies in: spike count, resting Vm mean/std, voltage
		  distribution skewness and kurtosis.

		Parameter mapping (per signature):
		  params[:,0]  gbar_Na         (mS/cm²)
		  params[:,1]  gbar_K          (mS/cm²)
		  params[:,2]  g_leak          (mS/cm²)
		  params[:,3]  |E_leak|        (mV, negated internally)
		  params[:,4]  |Vt|            (mV, negated internally)
		  params[:,5]  nois_fact       (unitless)
		  params[:,6]  gbar_M          (mS/cm², M-current conductance, X1 slot)
		  params[:,7]  gbar_X2         (unused, parsimony)
		  params[:,8]  |halfact_offset|(mV, stored positive [1e-4,150], negated → [-150,0])
		                               shifts IKM half-activation from canonical -35 mV
		  params[:,9]  |param_j|       (unused, parsimony)

		Args:
			init_voltage : torch.Tensor (batch_size,)           initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) injected current (µA/cm²)
			dt           : float                                 time step (ms)
			t            : torch.Tensor (time_steps,)           time array (ms)
			params       : torch.Tensor (batch_size, 10)        parameter vector
			seed         : int or None                           RNG seed

		Returns:
			V            : torch.Tensor (batch_size, time_steps) membrane voltage (mV)
		"""
		device = params.device

		# ---- Random generator setup ----
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int
		tstep      = float(dt)

		# ---- Parameter extraction ----
		gbar_Na = params[:, 0].float()          # (batch_size,) mS/cm²
		gbar_K  = params[:, 1].float()          # (batch_size,) mS/cm²
		g_leak  = params[:, 2].float()          # (batch_size,) mS/cm²
		E_leak  = -params[:, 3].float()         # (batch_size,) mV
		Vt      = -params[:, 4].float()         # (batch_size,) mV
		nois_fact = params[:, 5].float()        # (batch_size,) unitless

		# M-current (IKM) — X1 slot
		# Conductance: params[:,6] ∈ [1e-4, 10] mS/cm²
		gbar_M = params[:, 6].float()           # (batch_size,) mS/cm²

		# Half-activation offset: params[:,8] stored positive ∈ [1e-4, 150] mV
		# Negated so halfact_shift ∈ [-150, ~0]; canonical IKM half-activation ≈ -35 mV
		# Effective half-activation = -35 + halfact_shift, tunable by inference
		halfact_shift = -params[:, 8].float()   # (batch_size,) mV ∈ [-150, 0]

		# params[:,7] (gbar_X2) and params[:,9] (param_j): not used — parsimony

		# ---- Fixed biophysical constants ----
		nois_fact_obs = 0.0
		C    = 1.0     # µF/cm²  membrane capacitance
		E_Na = 53.0    # mV      sodium reversal potential
		E_K  = -107.0  # mV      potassium reversal potential (also IKM reversal)

		# ---- Numerical helpers ----
		def Exp(z):
			# (batch_size,) → (batch_size,)  clipped to avoid overflow
			return torch.where(
				z < -5e2,
				torch.exp(torch.tensor(-5e2, dtype=z.dtype, device=device)).expand_as(z),
				torch.exp(z),
			)

		def efun(z):
			# (batch_size,) → (batch_size,)  L'Hôpital-safe exponential factor
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ---- Standard HH kinetics ----
		# Na+ activation (m)
		def alpha_m(x):
			v1 = x - Vt - 13.0                          # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		# Na+ inactivation (h)
		def alpha_h(x):
			v1 = x - Vt - 17.0                          # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		# K+ delayed rectifier (n)
		def alpha_n(x):
			v1 = x - Vt - 15.0                          # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			v1 = x - Vt - 10.0                          # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		# ---- M-current (IKM) kinetics — p gate ----
		# Symmetric exponential form gives:
		#   p_inf(V)  = 1 / (1 + exp(-0.08*(V + 35 + halfact_shift)))
		#   tau_p(V)  = 1 / (alpha_p + beta_p)
		#             = 1 / (6.6e-3 * cosh(0.04*(V + 35 + halfact_shift)))
		# Peak tau_p ≈ 152 ms at V = -35 + |halfact_shift| — slow, consistent with IKM
		# Rate constant 3.3e-3 ms⁻¹ from Wang & McKinnon (1995) / Mainen & Sejnowski (1996)
		def alpha_p(x):
			v1 = x + 35.0 + halfact_shift               # (batch_size,)
			return 3.3e-3 * Exp(0.04 * v1)

		def beta_p(x):
			v1 = x + 35.0 + halfact_shift               # (batch_size,)
			return 3.3e-3 * Exp(-0.04 * v1)

		# Steady-state and time-constant helpers
		def tau_x(alpha, beta):
			return 1.0 / (alpha + beta)                  # (batch_size,)

		def inf_x(alpha, beta):
			return alpha / (alpha + beta)                 # (batch_size,)

		# ---- State variable allocation ----
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na act
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na inact
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K act
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IKM gate

		# ---- Steady-state initialisation ----
		V[:, 0] = init_voltage.to(device)                                              # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                            # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                            # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                            # (batch_size,)
		p[:, 0] = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))                            # (batch_size,)

		# ---- Simulation loop (exponential Euler integration) ----
		for i in range(1, time_steps):
			Vp = V[:, i - 1]   # (batch_size,) previous voltage

			# Compute rate constants at previous voltage
			a_m, b_m = alpha_m(Vp), beta_m(Vp)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(Vp), beta_h(Vp)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(Vp), beta_n(Vp)   # (batch_size,), (batch_size,)
			a_p, b_p = alpha_p(Vp), beta_p(Vp)   # (batch_size,), (batch_size,)

			# Previous gating variables
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Conductance-weighted inverse membrane time constant (1/ms)
			# IKM uses first-order p gate (no power — standard M-current convention)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,) Na contribution
				+ (n_prev ** 4) * gbar_K            # (batch_size,) K contribution
				+ g_leak                             # (batch_size,) leak contribution
				+ gbar_M * p_prev                   # (batch_size,) IKM contribution
			) / C                                    # (batch_size,) 1/ms

			# Noise draw for this time step
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# Effective voltage steady-state (mV): weighted sum of driving forces + inputs
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na drive
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) K drive
				+ g_leak * E_leak                           # (batch_size,) leak drive
				+ gbar_M * p_prev * E_K                    # (batch_size,) IKM drive (E_K)
				+ input_current[:, i - 1]                  # (batch_size,) injected current
				+ noise                                     # (batch_size,) stochastic term
			) / (tau_V_inv * C)                             # (batch_size,)

			# Exponential Euler updates
			V[:, i] = V_inf + (Vp - V_inf) * Exp(-tstep * tau_V_inv)                                   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			p[:, i] = inf_x(a_p, b_p) + (p_prev - inf_x(a_p, b_p)) * Exp(-tstep / tau_x(a_p, b_p))   # (batch_size,)

		# Return voltage traces (observation noise disabled: nois_fact_obs = 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)