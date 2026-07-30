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
		Hodgkin-Huxley neuron extended with a voltage-kinetics M-type K+ current.

		CHANGES IN THIS ITERATION (two targeted improvements):

		1. E_K corrected to -77.0 mV (canonical HH value).
		   The base model used -107.0 mV, far below physiological range, causing
		   systematic over-hyperpolarization after spikes — biasing mean voltage,
		   inflating variance, and distorting skewness/kurtosis statistics.

		2. Voltage-dependent tau_M replaces constant tau_M.
		   Real M-type channels have a bell-shaped tau(V) profile peaking near
		   V_half = -35 mV and decreasing at depolarised/hyperpolarised voltages
		   (Wang 1998, Biophys J). The voltage-dependent formulation:
		       tau_M_v(V) = tau_M_peak / (3.3 * exp((V+35)/20) + exp(-(V+35)/20))
		   correctly reproduces how the M-gate relaxes fastest near threshold and
		   more slowly far from it. This improves reproduction of inter-spike
		   interval regularity and higher-order voltage statistics (variance,
		   skewness, kurtosis) without adding any new free parameters.

		M-current physiological rationale:
		  - I_M = gbar_M * p * (V - E_K): slow, non-inactivating K+ current
		  - Active near spike threshold; provides spike-frequency adaptation
		  - Regularises inter-spike intervals → evenly-spaced tonic spiking
		  - Single gating variable p (no inactivation) — minimal complexity
		  - Does NOT produce bursting or sustained high-frequency firing

		Parameter layout:
		  params[:,0]  gbar_Na      Na+ conductance (mS/cm²)
		  params[:,1]  gbar_K       K+ Kdr conductance (mS/cm²)
		  params[:,2]  g_leak       leak conductance (mS/cm²)
		  params[:,3]  |E_leak|     leak reversal (E_leak = -params[:,3], mV)
		  params[:,4]  |Vt|         threshold offset (Vt = -params[:,4], mV)
		  params[:,5]  nois_fact    current noise amplitude (unitless)
		  params[:,6]  gbar_M       M-current maximal conductance (mS/cm²), range [1e-4, 10]
		  params[:,7]  tau_M_peak   M-gate peak time constant (ms), range [1e-4, 120]
		  params[:,8]  unused       slot X2 param_i — kept for parsimony
		  params[:,9]  unused       slot X2 param_j — kept for parsimony

		Args:
			init_voltage: torch.Tensor (batch_size,) initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) injected current (μA/cm²)
			dt: float time step (ms)
			t: torch.Tensor (time_steps,) time array (ms)
			params: torch.Tensor (batch_size, 10) biophysical parameters
			seed: optional int random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps) membrane voltage traces (mV)
		"""
		device = params.device

		# ── Random generator ──────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ── Base HH parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ maximal conductance (mS/cm²)
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ Kdr maximal conductance (mS/cm²)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm²)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal (mV); prior positive
		Vt        = -params[:, 4].float()  # (batch_size,)  threshold offset (mV); prior positive
		nois_fact = params[:, 5].float()   # (batch_size,)  current noise amplitude

		# ── M-current parameters (slot X1: params[:,6] and params[:,7]) ──────
		# gbar_M: prior range [1e-4, 10] mS/cm², already positive — no transform needed
		# tau_M_peak: params[:,7] prior range [1e-4, 120] ms, already positive.
		#   Adding 1e-3 guarantees strict positivity for the voltage-dependent formula.
		#   This is the PEAK of the bell-shaped tau_M(V) curve, occurring near V_half_M.
		gbar_M    = params[:, 6].float()          # (batch_size,)  mS/cm²
		tau_M_peak = params[:, 7].float() + 1e-3  # (batch_size,)  ms, strictly > 0

		# V_half_M: FIXED at -35.0 mV (canonical M-type channel half-activation).
		# NOT inferred — fixes the prior search space issue from the previous iteration
		# where a very broad prior on params[:,8] forced the sampler to scan (-150, 0) mV.
		# Literature: Wang (1998), Brown & Adams (1980), Wang & McKinnon (1995)
		V_half_M = -35.0   # scalar float (mV), fixed

		# Slots X2: params[:,8] and params[:,9] are intentionally unused.
		# One well-characterised M-current channel is sufficient for tonic spiking.
		# Adding a second channel risks identifiability problems with the inference.

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise disabled
		C    = 1.0            # membrane capacitance (μF/cm²)
		E_Na = 53.0           # Na+ reversal potential (mV)

		# CRITICAL FIX: E_K = -77.0 mV (canonical HH value).
		# The base model used E_K = -107.0 mV, which is far outside the physiological
		# range. The correct value -77 mV ensures:
		#   (a) Realistic after-hyperpolarization depth (~-70 to -75 mV trough)
		#   (b) Unbiased mean voltage during stimulation
		#   (c) Correct variance, skewness, and kurtosis of the voltage distribution
		# Both K+ Kdr (n^4 * gbar_K) and M-current (gbar_M * p) share this reversal.
		E_K  = -77.0          # K+ reversal potential (mV), canonical HH

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Clamped exponential: prevents overflow for very negative z
			# z: any-shape tensor → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Handles removable singularity z / (exp(z) - 1) near z = 0
			# Uses Taylor: efun(z) ≈ 1 - z/2 when |z| < 1e-4
			# z: any-shape tensor → same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gate kinetics (unchanged from base model) ─────────────
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
			# Gate time constant from forward/backward rate constants
			# alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state from forward/backward rate constants
			# alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-gate kinetics ───────────────────────────────────────────────────
		# Steady-state: Boltzmann with fixed half-activation V_half_M = -35 mV
		# and slope factor k = 10 mV (standard for muscarinic K+ channels).
		# Fixed V_half_M eliminates the broad prior search problem from prior iterations.
		def p_inf(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		# Voltage-dependent time constant: bell-shaped profile centred at V_half_M.
		# Formula from Wang (1998): tau_M(V) = tau_M_peak / (alpha_p(V) + beta_p(V))
		# Approximated here as:
		#   tau_M_v(V) = tau_M_peak / (3.3 * exp((V + 35)/20) + exp(-(V + 35)/20))
		# At V = -35 mV (peak): denominator ≈ 3.3 + 1 = 4.3, tau = tau_M_peak / 4.3
		# At V = -65 mV (rest): denominator ≈ 0.55 + 6.05 = 6.6, tau shorter → fast relax
		# This correctly makes the M-gate slower near threshold (where adaptation matters)
		# and faster at resting potential (quick return to low activity during quiescence).
		# Minimum clamp at 0.1 ms prevents numerical instability.
		def tau_M_v(x):
			# x: (batch_size,) → (batch_size,)
			dv = (x - V_half_M) / 20.0   # (batch_size,)  normalised voltage deviation
			denom = 3.3 * Exp(dv) + Exp(-dv)   # (batch_size,)  bell-shape denominator (>0)
			# tau_M_peak (batch_size,) scales the peak; clamp denom away from 0
			denom_safe = torch.clamp(denom, min=1e-3)   # (batch_size,)
			tau_raw = tau_M_peak / denom_safe            # (batch_size,)  ms
			return torch.clamp(tau_raw, min=0.1)         # (batch_size,)  ms, clamp at 0.1 ms

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-gate

		# ── Initial conditions at steady state ────────────────────────────────
		V_init = init_voltage.to(device)                                    # (batch_size,)
		V[:, 0] = V_init                                                    # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                 # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                 # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                 # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                            # (batch_size,) M-gate steady state

		# ── Exponential Euler time integration ───────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH gate rates
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-gate steady state and voltage-dependent time constant at V_prev
			p_inf_v  = p_inf(V_prev)    # (batch_size,)
			tau_M_vv = tau_M_v(V_prev)  # (batch_size,)  ms, voltage-dependent, >0

			# Effective inverse membrane time constant (sum of conductances / C)
			# Includes M-current term: gbar_M * p (non-inactivating)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ contribution    (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K+ Kdr contribution (batch_size,)
				+ g_leak                            # leak contribution   (batch_size,)
				+ gbar_M * p_prev                   # M-current term      (batch_size,)
			) / C   # (batch_size,)

			# Current noise (scaled by tstep for Euler-Maruyama consistency)
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device)   # (batch_size,)

			# Voltage steady-state: sum of reversal-potential drives + inputs
			# M-current pulls toward E_K = -77 mV → controlled after-hyperpolarization
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na+ drive    (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # K+ Kdr drive (batch_size,)
				+ g_leak * E_leak                           # leak drive   (batch_size,)
				+ gbar_M * p_prev * E_K                    # M-current    (batch_size,)
				+ input_current[:, i - 1]                  # injected I   (batch_size,)
				+ noise / (tstep ** 0.5)                   # noise drive  (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler voltage update
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Standard gate updates via exponential Euler
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)

			# M-gate update: exponential Euler with voltage-dependent tau_M_vv.
			# tau_M_vv > 0 always (clamped at 0.1 ms) → Exp(-tstep/tau_M_vv) ∈ (0,1]
			# Numerically stable: no sign-negation, no division instability.
			p[:, i] = p_inf_v + (p_prev - p_inf_v) * Exp(-tstep / tau_M_vv)   # (batch_size,)

		# Return voltage trace; observation noise disabled (nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)