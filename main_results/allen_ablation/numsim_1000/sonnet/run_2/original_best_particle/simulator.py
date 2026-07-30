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
		Hodgkin-Huxley neuron extended with a single M-type (Kv7/KCNQ) potassium current.

		Design rationale for this iteration:
		  - Previous iteration used gbar_M (X1) and E_M (param_i) but left param_j unused
		  - This iteration assigns param_j as a direct tunable time constant (tau_p, in ms)
		    for the M-current gate, giving the inference engine full kinetic control
		  - Using BOTH free param slots (param_i, param_j) for ONE channel avoids
		    identifiability problems from multi-channel parameter splitting
		  - Simplified M-current: p_inf/tau_p formulation (no alpha/beta split)
		    → cleaner, fewer correlated parameters, easier for SBI to identify

		Physiological rationale for M-current:
		  - Slow (~100-500 ms), non-inactivating sub-threshold K⁺ current
		  - Activated near resting potential, regulates ISI and spike-frequency adaptation
		  - Does NOT produce bursting — only tonic spiking regularity
		  - Addresses: spike count, mean/SD resting potential, voltage variance, skewness

		Args:
			init_voltage: torch.Tensor: (batch_size,)              -- initial voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps)  -- injected current (µA/cm²)
			dt: float                                               -- time step (ms)
			t: torch.Tensor: (time_steps,)                         -- time array (ms)
			params: torch.Tensor: (batch_size, 10)                 -- biophysical parameters
			seed: int or None                                       -- random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)              -- membrane voltage (mV)
		"""
		device = params.device

		# ── Random number generator ───────────────────────────────────────────────
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Parameter extraction ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na⁺ max conductance    [mS/cm²]
		gbar_K    = params[:, 1].float()   # (batch_size,)  K⁺ max conductance     [mS/cm²]
		g_leak    = params[:, 2].float()   # (batch_size,)  Leak conductance        [mS/cm²]
		E_leak    = -params[:, 3].float()  # (batch_size,)  Leak reversal           [mV]
		Vt        = -params[:, 4].float()  # (batch_size,)  Voltage threshold shift [mV]
		nois_fact = params[:, 5].float()   # (batch_size,)  Noise amplitude         [unitless]

		# X1 slot: M-type K⁺ current (slow, non-inactivating, tonic-spiking regulator)
		gbar_M    = params[:, 6].float()   # (batch_size,)  M-current conductance   [mS/cm²], range [1e-4, 10]
		# X2 slot: reserved/unused — parsimony principle (one channel is sufficient)
		# gbar_X2 = params[:, 7].float()   # not used in this iteration

		# param_i → M-current reversal potential (E_M)
		# Stored as positive value in [1e-4, 150]; negated to get physiological range
		# E_M ≈ -107 mV (near E_K) when params[:,8] ≈ 107 — K⁺-selective channel
		E_M       = -params[:, 8].float()  # (batch_size,)  M-current reversal      [mV]

		# param_j → M-current activation time constant (tau_p)
		# Range [1e-4, 3000] maps directly to ms — physiological range: 50–500 ms
		# This gives SBI direct control over M-current kinetics independently of gbar_M
		tau_p     = params[:, 9].float()   # (batch_size,)  M-current time constant [ms], range [1e-4, 3000]

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (disabled)
		C    = 1.0            # membrane capacitance [µF/cm²]
		E_Na = 53.0           # Na⁺ reversal         [mV]
		E_K  = -107.0         # K⁺ reversal          [mV]

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential: clamp lower bound at -500
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),  # z: (batch_size,) or broadcastable
				torch.exp(z)
			)

		def efun(z):
			# Rall's function z/(exp(z)-1) with Taylor series near z=0
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gating kinetics ───────────────────────────────────────────
		def alpha_m(x):   # x: (batch_size,)
			v1 = x - Vt - 13.0              # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):    # x: (batch_size,)
			v1 = x - Vt - 40.0             # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2      # (batch_size,)

		def alpha_h(x):   # x: (batch_size,)
			v1 = x - Vt - 17.0             # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)           # (batch_size,)

		def beta_h(x):    # x: (batch_size,)
			v1 = x - Vt - 40.0             # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))     # (batch_size,)

		def alpha_n(x):   # x: (batch_size,)
			v1 = x - Vt - 15.0             # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2    # (batch_size,)

		def beta_n(x):    # x: (batch_size,)
			v1 = x - Vt - 10.0             # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)             # (batch_size,)

		def tau_x(alpha, beta):   # alpha, beta: (batch_size,)
			return 1.0 / (alpha + beta)               # (batch_size,)

		def inf_x(alpha, beta):   # alpha, beta: (batch_size,)
			return alpha / (alpha + beta)             # (batch_size,)

		# ── M-current (Kv7/KCNQ) gating — simplified p_inf / tau_p formulation ───
		# Replaces the alpha_p/beta_p pair from prior iteration with a direct
		# sigmoid steady-state + fixed (but inferred) time constant.
		#
		# Physiological basis:
		#   p_inf(V) = 1 / (1 + exp(-(V - V_half) / k))
		#   V_half ≈ Vt + 30 mV  (activates ~10–15 mV below spike threshold)
		#   k = 9 mV             (slope — standard Kv7 value from literature)
		#   tau_p = param_j      (inferred, covers 10–500 ms physiological range)
		#
		# This formulation:
		#   (1) avoids correlated alpha_p/beta_p parameters
		#   (2) gives SBI a direct, interpretable handle on kinetics via param_j
		#   (3) maintains biophysical realism for tonic spiking without bursting

		V_half_M = 30.0   # mV offset from Vt (scalar, fixed)
		k_M      = 9.0    # mV slope of sigmoid (scalar, fixed)

		def p_inf(x):   # x: (batch_size,)
			# Steady-state M-current activation: sigmoid centred at Vt + V_half_M
			v1 = x - Vt - V_half_M          # (batch_size,)
			return 1.0 / (1.0 + Exp(-v1 / k_M))     # (batch_size,)

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inactivation
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K activation
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current activation

		# ── Steady-state initialisation ───────────────────────────────────────────
		V_init    = init_voltage.to(device)                         # (batch_size,)
		V[:, 0]   = V_init                                          # (batch_size,)
		m[:, 0]   = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))       # (batch_size,)
		h[:, 0]   = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))       # (batch_size,)
		n[:, 0]   = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))       # (batch_size,)
		p[:, 0]   = p_inf(V[:, 0])                                  # (batch_size,) M-gate at steady state

		# ── Simulation loop (exponential Euler) ───────────────────────────────────
		for i in range(1, time_steps):

			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH gating rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current steady-state at current voltage
			p_inf_prev = p_inf(V_prev)   # (batch_size,)

			# Effective membrane conductance (sum of all active conductances / C)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,) Na⁺
				+ (n_prev ** 4) * gbar_K            # (batch_size,) K⁺
				+ g_leak                            # (batch_size,) leak
				+ gbar_M * p_prev                   # (batch_size,) M-current
			) / C                                   # (batch_size,)

			# Stochastic noise injection (fixed amplitude / sqrt(dt) scaling)
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# Voltage steady-state (numerator = sum of g_ion * E_ion + I_ext + noise)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)
				+ g_leak * E_leak                          # (batch_size,)
				+ gbar_M * p_prev * E_M                    # (batch_size,) M-current drive
				+ input_current[:, i - 1]                  # (batch_size,)
				+ noise                                    # (batch_size,)
			) / (tau_V_inv * C)                            # (batch_size,)

			# Exponential Euler voltage update
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler HH gating variable updates
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)

			# Exponential Euler M-current gate update with inferred time constant tau_p
			# tau_p = param_j ∈ [1e-4, 3000] ms — directly sets M-current kinetics speed
			p[:, i] = p_inf_prev + (p_prev - p_inf_prev) * Exp(-tstep / tau_p)   # (batch_size,)

		# ── Return voltage trace (observation noise = 0) ───────────────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)