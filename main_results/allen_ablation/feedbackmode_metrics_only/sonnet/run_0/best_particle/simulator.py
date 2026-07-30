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
		Hodgkin-Huxley neuron with A-type transient K+ current (IA).

		HISTORY SUMMARY:
		  Iter 1 (score 25.5): param_i→V_half_a, param_j→k_a [2,20]mV, tau_b=80ms fixed, a^3*b
		  Iter 2 (score 25.2): param_i→V_half_a, param_j→tau_b [10,500]ms, k_a=10mV fixed, a^3*b
		                        b_inf coupled: V_half_b = V_half_a - 20mV
		  Iter 3 (score 28.1): DECOUPLED b_inf (V_half_b=-65mV fixed, k_b=8mV), tau_b [20,400]ms
		                        → WORSE than iter 2; decoupling hurt inference

		  THIS ITERATION — Two targeted changes from iter 2 (best score):
		    1. Activation power: a^4 (was a^3)
		       Rationale: a^4 sharpens the voltage-threshold sensitivity of IA, similar to
		       how n^4 is used for K-DR. More cooperative gating means IA turns on more
		       abruptly above V_half_a, creating a cleaner threshold effect that may better
		       match the voltage distribution statistics (skewness, kurtosis) in the data.
		       a^3 is biophysically reasonable but a^4 is also justified for some IA currents.

		    2. Inactivation coupling offset: 25 mV (was 20 mV)
		       b_inf half-voltage = V_half_a - 25 mV (was V_half_a - 20 mV)
		       Rationale: A deeper inactivation half-voltage shifts b_inf so that IA is more
		       inactivated at the resting potential and de-inactivates more specifically during
		       the AHP. This increases the dynamic range of IA between spike peaks and troughs,
		       potentially improving agreement with observed voltage variance and skewness.
		       Coupled formulation (not fixed) is retained because decoupling (iter 3) was worse.

		    3. Everything else identical to iter 2 (which was best):
		       - V_half_a = -params[:, 8]  (tunable activation half-voltage)
		       - tau_b = clamp(params[:, 9] / 6, 10, 500) ms  (tunable inactivation recovery)
		       - k_a = 10 mV fixed for both gates (shared slope)
		       - tau_a = 2 ms fixed (fast activation)

		Args:
		    init_voltage: torch.Tensor (batch_size,)
		    input_current: torch.Tensor (batch_size, time_steps)
		    dt: float  (ms)
		    t: torch.Tensor (time_steps,)
		    params: torch.Tensor (batch_size, 10)
		    seed: int or None

		Returns:
		    V: torch.Tensor (batch_size, time_steps)
		"""
		device = params.device

		# Random generator setup
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ───────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV
		Vt        = -params[:, 4].float()  # (batch_size,)  mV
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── A-type K+ channel parameters (X1 slot; X2 unused) ────────────────────
		# gbar_KA (X1): max conductance for IA current [1e-4, 10] mS/cm²
		# param_i (raw ∈ [1e-4,150], negated internally):
		#   → V_half_a: half-activation voltage for a_inf gate (mV, negative after negation)
		#     Typical cortical IA: activates around -50 to -30 mV
		# param_j (raw ∈ [1e-4,3000]):
		#   → tau_b: inactivation recovery time constant in ms
		#     clamp(param_j / 6, 10, 500) ms — same mapping as iter 2 (best score)
		#     Controls de-inactivation speed during AHP: short tau_b → fast recovery → more IA
		gbar_KA   = params[:, 6].float()              # (batch_size,)  mS/cm²
		# params[:, 7] unused (X2 slot — maintaining parsimony per task constraints)
		V_half_a  = -params[:, 8].float()             # (batch_size,)  mV (negated → negative)
		tau_b_raw = params[:, 9].float()              # (batch_size,)  raw ∈ [1e-4, 3000]
		tau_b_val = torch.clamp(tau_b_raw / 6.0, min=10.0, max=500.0)  # (batch_size,)  ms

		tstep = float(dt)  # ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV (shared by DR and A-type K+ channels)

		# Fixed IA kinetic parameters:
		#   tau_a = 2 ms: fast activation (cortical IA activates in ~1-5 ms)
		#   k_a = 10 mV:  sigmoid width for both a_inf and b_inf (same as iter 2)
		#   inact_offset = 25 mV: b_inf half-voltage = V_half_a - 25 mV
		#     CHANGE from iter 2 (was 20 mV): deeper offset places b_inf half-voltage
		#     further below V_half_a, increasing IA inactivation range between AHP and peak
		#     b_inf is COUPLED to V_half_a (same as iter 2; decoupling in iter 3 was worse)
		tau_a_fixed   = 2.0    # ms (fixed)
		k_a_fixed     = 10.0  # mV (fixed, shared slope for both gates)
		inact_offset  = 25.0  # mV (CHANGED from 20→25: deeper inactivation half-voltage)

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Clipped exponential for numerical stability; z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Numerically stable z/(exp(z)-1); z: any shape → same shape
			return torch.where(
				torch.abs(z) < 1e-4,
				1 - z / 2,
				z / (Exp(z) - 1)
			)

		# ── Standard HH kinetics ──────────────────────────────────────────────────
		def alpha_m(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,) each → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,) each → (batch_size,)
			return alpha / (alpha + beta)

		# ── A-type K+ channel kinetics ────────────────────────────────────────────
		# Activation gate (a): fast sigmoid with tunable half-voltage V_half_a
		#   a_inf(V) = 1 / (1 + exp(-(V - V_half_a) / k_a_fixed))
		#   Fixed slope k_a=10 mV; fixed tau_a=2 ms
		def a_inf(x):   # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_a) / k_a_fixed))

		# Inactivation gate (b): slower sigmoid, COUPLED half-voltage
		#   b_inf(V) = 1 / (1 + exp((V - (V_half_a - inact_offset)) / k_a_fixed))
		#   Half-inactivation at (V_half_a - 25 mV), 25 mV below activation threshold
		#   CHANGE: inact_offset=25mV vs 20mV in iter 2 — deeper inactivation window
		#   COUPLED to V_half_a (not fixed at -65mV as in iter 3 which was worse)
		#   Slope shared with activation (k_a_fixed=10 mV) for consistency
		def b_inf(x):   # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((x - (V_half_a - inact_offset)) / k_a_fixed))

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		a = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IA activation
		b = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IA inactivation

		# ── Steady-state initialisation ───────────────────────────────────────────
		V_init = init_voltage.to(device)                                          # (batch_size,)
		V[:, 0] = V_init                                                          # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                       # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                       # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                       # (batch_size,)
		a[:, 0] = a_inf(V[:, 0])                                                  # (batch_size,)
		b[:, 0] = b_inf(V[:, 0])                                                  # (batch_size,)

		# Precompute fixed exponential decay factor for tau_a (constant across time & batch)
		# tau_b varies per batch element → exp(-tstep/tau_b_val) computed inside loop
		exp_a_decay = torch.full((batch_size,), -tstep / tau_a_fixed, device=device)  # (batch_size,)

		# ── Time integration (exponential Euler) ──────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			a_prev = a[:, i - 1]   # (batch_size,)
			b_prev = b[:, i - 1]   # (batch_size,)

			# Standard HH rate constants at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) each

			# Instantaneous conductances from current gating states
			# Na:   m^3 * h  (standard HH cubic activation)
			# K-DR: n^4      (standard HH quartic activation)
			# K-A:  a^4 * b  (CHANGE: quartic activation was a^3 in iters 1-3)
			#   a^4 sharpens the voltage-threshold sensitivity: IA activates more
			#   abruptly above V_half_a, similar to K-DR n^4 cooperative gating
			#   This may improve agreement with voltage distribution statistics
			g_Na_eff  = (m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)
			g_K_eff   = (n_prev ** 4) * gbar_K             # (batch_size,)
			g_KA_eff  = (a_prev ** 4) * gbar_KA * b_prev   # (batch_size,)  NOTE: a^4

			# Effective membrane conductance inverse time constant
			tau_V_inv = (g_Na_eff + g_K_eff + g_KA_eff + g_leak) / C   # (batch_size,)

			# Additive noise in current space (Euler-Maruyama with √dt normalization)
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)

			# Effective steady-state voltage (reversal-potential-weighted conductances)
			V_inf = (
				g_Na_eff  * E_Na
				+ g_K_eff  * E_K
				+ g_KA_eff * E_K
				+ g_leak   * E_leak
				+ input_current[:, i - 1]
				+ noise
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler update for membrane voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler for standard HH gates
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)

			# Exponential Euler for IA gates
			a_ss = a_inf(V_prev)   # (batch_size,)
			b_ss = b_inf(V_prev)   # (batch_size,)

			# tau_a = 2 ms (fixed, fast activation) — precomputed scalar decay
			a[:, i] = a_ss + (a_prev - a_ss) * Exp(exp_a_decay)          # (batch_size,)

			# tau_b: tunable per batch [10, 500] ms — inactivation recovery
			# Same mapping as iter 2 (clamp(param_j/6, 10, 500)): best-performing range
			# Fast tau_b → IA quickly de-inactivates during AHP → strong ISI delay
			# Slow tau_b → IA mostly inactivated during firing → weaker ISI shaping
			b[:, i] = b_ss + (b_prev - b_ss) * Exp(-tstep / tau_b_val)   # (batch_size,)

		# Return voltage traces (zero observation noise)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)