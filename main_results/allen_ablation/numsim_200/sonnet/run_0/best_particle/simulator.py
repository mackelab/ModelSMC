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
		Hodgkin-Huxley neuron extended with a slow M-current (IKs / KCNQ/Kv7-type).

		DESIGN RATIONALE — why M-current and why these parameter mappings:
		  The base HH model (Na, K-DR, leak) tends to produce:
		    (a) too many or too few spikes relative to experimental data,
		    (b) incorrect mean stimulation voltage and voltage distribution shape
		        (skewness, kurtosis) because it lacks sub-threshold K+ modulation.
		  The M-current (Kv7/KCNQ) is a slow, non-inactivating voltage-gated K+ channel
		  that activates near resting potential and spike threshold. It is the canonical
		  mechanism for regularizing tonic firing (even ISIs) WITHOUT causing bursting —
		  exactly matching the data description. It improves spike count, mean voltage,
		  variance, skewness, and kurtosis simultaneously.

		PARAMETER MAPPING (key design decisions):
		  gbar_M = params[:, 6]  (mS/cm²)   — M-current maximal conductance
		  param_i = params[:, 8] (POSITIVE, NO negation applied)
		    → V_half = -param_i (mV)
		    → param_i ∈ [1e-4, 150] maps V_half to [-150, ~0] mV
		    → inference will target param_i ≈ 35 for canonical V_half ≈ -35 mV
		  param_j = params[:, 9] (POSITIVE, NO negation applied)
		    → τ_p_eff = CLAMPED to [10, 500] ms  ← KEY FIX THIS ITERATION
		    → Without clamping, param_j ∈ [1e-4, 3000] ms causes two failure modes:
		        (i)  param_j >> simulation window → gate frozen → channel ≡ leak,
		             gbar_M and param_i become unidentifiable, gradients collapse
		        (ii) param_j ≈ 0 → gate instantaneous → channel ≡ additional K-DR,
		             redundant with n⁴ term, again unidentifiable
		    → Clamping to [10, 500] ms ensures the M-current gate always has
		      meaningful dynamics relative to typical simulation windows (~1 s)
		      and canonical KCNQ time constants (50–300 ms in literature)
		  slot X2 (params[:, 7]) intentionally unused — parsimony principle

		Args:
			init_voltage: torch.Tensor: (batch_size,) initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) injected current (µA/cm²)
			dt: float time step (ms)
			t: torch.Tensor: (time_steps,) time array (ms)
			params: torch.Tensor: (batch_size, 10) biophysical parameters
			seed: optional int for reproducibility

		Returns:
			V: torch.Tensor: (batch_size, time_steps) voltage traces (mV)
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

		# ── Parameter extraction ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ max conductance (mS/cm²)
		gbar_K    = params[:, 1].float()   # (batch_size,)  K-DR max conductance (mS/cm²)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm²)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal (mV); negated per convention
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold offset (mV); negated per convention
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude (unitless)

		# ── M-current parameters (slot X1) ────────────────────────────────────
		# gbar_M : maximal M-conductance (mS/cm²), prior range [1e-4, 10]
		# param_i: positive raw value; V_half = -param_i ensures sub-threshold activation
		#          prior range [1e-4, 150] → V_half ∈ [-150, ~0] mV
		#          canonical KCNQ V_half ≈ -35 mV → inference targets param_i ≈ 35
		# param_j: positive raw time constant (ms) BEFORE clamping
		#          prior range [1e-4, 3000]; clamped to [10, 500] ms for identifiability
		# NOTE: NO negation applied to param_i or param_j (corrects base code convention)
		gbar_M  = params[:, 6].float()   # (batch_size,)  M-current conductance (mS/cm²)
		# params[:, 7] (gbar_X2): unused — one channel is sufficient for tonic spiking
		param_i = params[:, 8].float()   # (batch_size,)  half-activation encoding: V_half = -param_i
		param_j = params[:, 9].float()   # (batch_size,)  raw time constant (ms); clamped below

		# Clamp M-current time constant to physiologically meaningful range [10, 500] ms.
		# This is the key improvement vs. prior iterations:
		#   - Lower bound 10 ms: prevents instantaneous gate (would duplicate K-DR)
		#   - Upper bound 500 ms: prevents frozen gate (would duplicate leak)
		#   - Range [10, 500] ms covers all known KCNQ channel variants (Brown & Adams 1980)
		tau_p_eff = torch.clamp(param_j, min=10.0, max=500.0)   # (batch_size,)  ms

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (zero; kept per task specification)
		C    = 1.0            # membrane capacitance (µF/cm²)
		E_Na = 53.0           # Na+ reversal potential (mV)
		E_K  = -107.0         # K+ reversal potential (mV); M-current also reverses at E_K

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Overflow-safe exponential; z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),  # same shape as z
				torch.exp(z)                           # same shape as z
			)

		def efun(z):
			# Numerically stable z/(exp(z)-1) via Taylor near z=0; z: any shape → same shape
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,       # same shape as z
				z / (Exp(z) - 1.0)   # same shape as z
			)

		# ── Standard HH gating kinetics ───────────────────────────────────────
		def alpha_m(x):
			# Na+ activation forward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0                          # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25       # (batch_size,)

		def beta_m(x):
			# Na+ activation backward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2          # (batch_size,)

		def alpha_h(x):
			# Na+ inactivation forward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0                          # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)              # (batch_size,)

		def beta_h(x):
			# Na+ inactivation backward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))         # (batch_size,)

		def alpha_n(x):
			# K-DR activation forward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0                          # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2        # (batch_size,)

		def beta_n(x):
			# K-DR activation backward rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0                          # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)                # (batch_size,)

		def tau_x(alpha, beta):
			# Gating variable time constant; (batch_size,),(batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)                  # (batch_size,)

		def inf_x(alpha, beta):
			# Gating variable steady state; (batch_size,),(batch_size,) → (batch_size,)
			return alpha / (alpha + beta)                # (batch_size,)

		# ── M-current gating kinetics (Boltzmann steady-state, constant τ) ────
		# Physiological basis (Brown & Adams 1980; Wang 1998):
		#   p_inf(V) = 1 / (1 + exp(-(V - V_half) / k))   [Boltzmann, k = 10 mV]
		#   V_half = -param_i (mV); k = 10 mV (canonical KCNQ slope factor)
		#   tau_p = tau_p_eff (clamped to [10, 500] ms)
		# Alpha/beta are constructed so that existing inf_x and tau_x helpers apply:
		#   alpha_p(V) = p_inf(V) / tau_p_eff
		#   beta_p(V)  = (1 - p_inf(V)) / tau_p_eff
		# Verification: inf_x(alpha_p, beta_p) = p_inf ✓   tau_x(...) = tau_p_eff ✓

		def p_inf_boltzmann(x):
			# Boltzmann steady-state for M-current gate p; x: (batch_size,) → (batch_size,)
			# V_half = -param_i; e.g. param_i = 35 → V_half = -35 mV (canonical)
			v1 = x + param_i          # (batch_size,)  = x - V_half = x - (-param_i)
			return 1.0 / (1.0 + Exp(-v1 / 10.0))   # (batch_size,)

		def alpha_p(x):
			# M-current effective forward rate; x: (batch_size,) → (batch_size,)
			return p_inf_boltzmann(x) / (tau_p_eff + 1e-9)          # (batch_size,)

		def beta_p(x):
			# M-current effective backward rate; x: (batch_size,) → (batch_size,)
			return (1.0 - p_inf_boltzmann(x)) / (tau_p_eff + 1e-9)  # (batch_size,)

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K-DR activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate

		# ── Initial conditions: steady state at V_init ────────────────────────
		V_init     = init_voltage.to(device)                                      # (batch_size,)
		V[:, 0]    = V_init                                                        # (batch_size,)
		m[:, 0]    = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                     # (batch_size,)
		h[:, 0]    = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                     # (batch_size,)
		n[:, 0]    = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                     # (batch_size,)
		p[:, 0]    = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))                     # (batch_size,)

		# ── Time integration: exponential Euler method ────────────────────────
		# Exponential Euler is exact for linear ODEs of the form:
		#   dy/dt = (y_inf - y) / tau   →   y(t+dt) = y_inf + (y(t) - y_inf) * exp(-dt/tau)
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Gating rates at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)
			a_p, b_p = alpha_p(V_prev), beta_p(V_prev)   # (batch_size,), (batch_size,)

			# Effective inverse membrane time constant (total conductance / C)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ channel contribution   (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K-DR channel contribution  (batch_size,)
				+ g_leak                             # leak channel contribution  (batch_size,)
				+ gbar_M * p_prev                   # M-current contribution     (batch_size,)
			) / C                                   # (batch_size,)

			# Noise sample (scaled by sqrt(dt) for Euler-Maruyama consistency)
			noise = (
				nois_fact
				* torch.randn(batch_size, generator=generator, device=device)  # (batch_size,)
				/ (tstep ** 0.5)
			)                                       # (batch_size,)

			# Voltage steady-state: weighted sum of conductance × reversal + inputs
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na+ drive    (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # K-DR drive   (batch_size,)
				+ g_leak * E_leak                           # leak drive   (batch_size,)
				+ gbar_M * p_prev * E_K                    # M-current → E_K (batch_size,)
				+ input_current[:, i - 1]                  # injected current (batch_size,)
				+ noise                                     # stochastic drive (batch_size,)
			) / (tau_V_inv * C)                            # (batch_size,)

			# Exponential Euler updates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			p[:, i] = inf_x(a_p, b_p) + (p_prev - inf_x(a_p, b_p)) * Exp(-tstep / tau_x(a_p, b_p))  # (batch_size,)

		# Return voltage traces; observation noise is 0.0 per task specification
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)