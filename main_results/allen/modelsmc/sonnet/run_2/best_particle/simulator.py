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
		Hodgkin-Huxley neuron with M-type K+ current (IM) in the X1 slot.

		DESIGN RATIONALE:
		  The base HH model (Na, K-DR, leak) produces tonic spiking but may misfit
		  inter-spike intervals, resting potential, and voltage distribution moments
		  (mean/variance/skewness/kurtosis during stimulation).

		  M-current (IM) is a slow, non-inactivating, sub-threshold K+ current that:
		    - Activates near spike threshold (~-35 mV) and deactivates during AHP
		    - Regulates inter-spike intervals, producing regular tonic firing adaptation
		    - Does NOT induce bursting or sustained high-frequency firing
		    - Is the canonical adaptation current for tonic spiking neurons
		    (Reference: Wang 1998; Adams et al. 1982)

		PARAMETER MAPPING (strict 2-parameter-per-channel rule):
		  X1 slot:
		    params[:,6] = gbar_M  — M-current maximal conductance (mS/cm2)
		    params[:,8] = |E_M|   — M-current reversal potential magnitude (mV);
		                            negated internally -> K+-like negative reversal
		    tau_M       = 100.0 ms (FIXED physiological constant, NOT inferred)
		                            Wang 1998 uses ~100 ms; fixing this avoids a 3rd
		                            free parameter and prevents gate freezing artefacts.
		  X2 slot:
		    params[:,7] (gbar_X2) — UNUSED (parsimony: IM alone is sufficient)
		    params[:,9] (param_j) — UNUSED

		Args:
		    init_voltage: torch.Tensor: (batch_size,)              initial voltage (mV)
		    input_current: torch.Tensor: (batch_size, time_steps)  injected current (uA/cm2)
		    dt: float                                               time step (ms)
		    t: torch.Tensor: (time_steps,)                         time array (ms)
		    params: torch.Tensor: (batch_size, 10)                 biophysical parameters
		    seed: int or None                                       random seed

		Returns:
		    V: torch.Tensor: (batch_size, time_steps)              voltage traces (mV)
		"""
		device = params.device

		# ---- Random generator ----
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ---- Base HH parameters ----
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ maximal conductance (mS/cm2)
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ delayed-rectifier conductance (mS/cm2)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm2)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal (mV); negated
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold offset (mV); negated
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude

		# ---- M-current parameters (X1 slot: 2 inferred params only) ----
		# gbar_M: maximal M-current conductance; prior range [1e-4, 10] mS/cm2
		# E_M   : reversal potential; prior on |param_i| in [1e-4, 150] mV,
		#         negated internally -> K+-like range [-150, ~0] mV
		# tau_M : FIXED at 100 ms (physiological; Wang 1998)
		#         Fixing avoids the "frozen gate" problem seen when tau_M_base is inferred
		#         over a wide range and also keeps the channel within its 2-parameter budget.
		gbar_M = params[:, 6].float()   # (batch_size,)  mS/cm2  [X1 conductance slot]
		# params[:,7] intentionally unused  (X2 conductance slot — parsimony)
		E_M    = -params[:, 8].float()  # (batch_size,)  mV      [param_i slot, negated]
		# params[:,9] intentionally unused  (param_j slot — parsimony)
		tau_M  = 100.0                  # ms, fixed physiological constant

		tstep = float(dt)

		# ---- Fixed biophysical constants ----
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm2  membrane capacitance
		E_Na = 53.0   # mV      Na+ reversal potential
		E_K  = -107.0 # mV      K+ reversal potential

		####################################
		# ---- Numerical helpers ----

		def Exp(z):
			# Numerically stable exponential (clamp at -500); z: any shape -> same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable z/(exp(z)-1) for HH rate functions; z: any shape -> same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ---- Standard HH kinetics ----

		def alpha_m(x):
			# Na+ activation forward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation backward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation forward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation backward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ DR activation forward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ DR activation backward rate; x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant; (batch_size,), (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state; (batch_size,), (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ---- M-current (IM) gating kinetics ----
		# Non-inactivating, voltage-gated K+ current activating near threshold.
		# Single gating variable p with instantaneous Boltzmann steady-state.
		# Half-activation at -35 mV, slope ~10 mV (Wang 1998).
		# Time constant fixed at tau_M = 100 ms for identifiability and parsimony.
		# This time constant makes IM slow enough to modulate ISI without bursting.

		def p_inf(x):
			# Boltzmann steady-state for IM gate; x: (batch_size,) -> (batch_size,)
			# Half-activation: -35 mV; slope factor: 10 mV
			return 1.0 / (1.0 + Exp(-(x + 35.0) / 10.0))

		# tau_M = 100.0 ms (fixed scalar; used directly in exponential-Euler update)

		####################################

		# ---- Allocate state variables ----
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-gate

		# ---- Steady-state initialisation ----
		V_init  = init_voltage.to(device)                          # (batch_size,)
		V[:, 0] = V_init
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                   # (batch_size,) M-gate at SS

		# ---- Time integration (exponential-Euler method) ----
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Cache gating variables at previous step
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# HH gate rates at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# Total membrane conductance / C  (effective 1/tau_V)
			# Shape: (batch_size,)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ channel conductance
				+ (n_prev ** 4) * gbar_K            # K+ delayed-rectifier conductance
				+ g_leak                             # passive leak conductance
				+ gbar_M * p_prev                   # M-current conductance (slow K+)
			) / C

			# Stochastic current noise; shape: (batch_size,)
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)

			# Instantaneous voltage steady-state (numerator of exponential-Euler)
			# Shape: (batch_size,)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na+ driving force
				+ (n_prev ** 4) * gbar_K * E_K             # K+ DR driving force
				+ g_leak * E_leak                           # leak driving force
				+ gbar_M * p_prev * E_M                    # M-current driving force (K+-like)
				+ input_current[:, i - 1]                  # externally applied current
				+ noise                                     # stochastic fluctuation
			) / (tau_V_inv * C)

			# Exponential-Euler voltage update; shape: (batch_size,)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)

			# Exponential-Euler updates for HH gating variables; shape: (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))

			# Exponential-Euler M-current gate update (fixed tau_M = 100 ms)
			# Shape: (batch_size,)
			p_ss   = p_inf(V_prev)                          # (batch_size,) steady-state at V_prev
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(torch.full_like(p_ss, -tstep / tau_M))

		# Return voltage trace (observation noise currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)