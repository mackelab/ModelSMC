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
		Hodgkin-Huxley neuron extended with a slow M-type K⁺ current (Kv7/KCNQ).

		Design decisions (consolidating all prior iteration feedback):

		  CHANGE 1 — E_K corrected: -107.0 → -77.0 mV
		    The original E_K = -107 mV produces excessively deep AHP troughs,
		    biasing mean voltage, variance, skewness and kurtosis away from
		    typical tonic-spiking data. The canonical HH value -77 mV is used.

		  CHANGE 2 — M-current uses exactly the X1 slot (params[:,6:8]):
		    * params[:,6]  →  gbar_M  (max M-conductance, range [1e-4, 10] mS/cm²)
		    * params[:,7]  →  tau_w_M (M-gate time constant, raw positive value
		                      clamped to [10, 120] ms; NOT negated — prior iteration
		                      bug negated params[:,9] yielding always-10 ms)
		    * V_half_M = -35.0 mV FIXED constant (physiological Kv7 half-activation).
		      Freeing V_half_M in earlier iterations created identifiability issues
		      and consumed a parameter slot unnecessarily.
		    * X2 slot (params[:,7] as gbar_X2 original label) is REPURPOSED as tau_w_M.
		      params[:,8] and params[:,9] are intentionally unused.

		  CHANGE 3 — M-current physiological rationale:
		    The M-current (Kv7/KCNQ) is a slow, non-inactivating outward K⁺ current
		    that activates near resting potential. It provides spike-frequency
		    adaptation, regularising inter-spike intervals in tonic spiking neurons
		    without producing burst firing or high-frequency sustained activity.
		    Single state variable w (no inactivation gate) keeps the model parsimonious.

		Args:
			init_voltage: torch.Tensor: (batch_size,)              # initial voltage [mV]
			input_current: torch.Tensor: (batch_size, time_steps)  # injected current [uA/cm²]
			dt: float                                               # time step [ms]
			t: torch.Tensor: (time_steps,)                         # time array [ms]
			params: torch.Tensor: (batch_size, 10)                 # biophysical parameters
			seed: optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)  # membrane potential [mV]
		"""
		device = params.device

		# Random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na⁺ max conductance     [mS/cm²]
		gbar_K    = params[:, 1].float()   # (batch_size,)  K⁺  max conductance     [mS/cm²]
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance         [mS/cm²]
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal potential  [mV]
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold shift  [mV]
		nois_fact  = params[:, 5].float()  # (batch_size,)  current noise amplitude  [unitless]

		# ── M-current parameters — X1 slot (params[:,6] and params[:,7]) ────────
		# Slot X1 provides exactly two tunable inferred parameters.
		# We repurpose both for the M-current to maximise identifiability.

		# gbar_M: max M-conductance [mS/cm²]
		#   Raw range of params[:,6]: [1e-4, 10]  →  used directly (positive)
		gbar_M    = params[:, 6].float()   # (batch_size,)  [mS/cm²]

		# tau_w_M: M-gate time constant [ms]
		#   Raw range of params[:,7]: [1e-4, 120]
		#   FIX vs. prior iterations: use the RAW positive value — do NOT negate.
		#   Negating (as done in iteration 2) maps all values to [-120, -1e-4],
		#   which clamps entirely to the minimum and destroys learnability.
		#   Clamp to [10, 120] ms: physiologically plausible for Kv7, and keeps
		#   the M-current slow enough to regularise ISIs without distorting spikes.
		tau_w_M   = torch.clamp(params[:, 7].float(), min=10.0, max=120.0)  # (batch_size,)  [ms]

		# params[:,8] (|param_i|) and params[:,9] (|param_j|) — intentionally unused.
		# Activating X2 slot without a strong justification would only add noise to
		# the posterior and increase the neg-log-marginal-likelihood.

		tstep = float(dt)  # scalar [ms]

		# ── Fixed biophysical constants ──────────────────────────────────────────
		nois_fact_obs = 0.0    # observation noise (kept at zero per specification)
		C    = 1.0             # membrane capacitance [uF/cm²]
		E_Na = 53.0            # Na⁺ reversal potential [mV]
		# FIX: corrected from -107.0 mV (original base) to canonical HH value.
		# -107 mV produced excessively deep AHPs, biasing all voltage statistics.
		E_K  = -77.0           # K⁺ reversal potential [mV]  (shared by K⁺ and M-current)

		# V_half_M: M-gate half-activation voltage [mV] — FIXED physiological constant.
		# Kv7/KCNQ channels activate sigmoidally near -35 mV with ~10 mV slope.
		# Fixing this eliminates one degree of freedom and improves identifiability.
		V_half_M = -35.0       # scalar constant [mV]
		k_M      = 10.0        # sigmoid slope [mV]

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# (any shape) → (same shape); clamp argument to avoid exp overflow
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (any shape) → (same shape); numerically safe z / (exp(z) - 1)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ────────────────────────────────────────
		def alpha_m(x):  # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):  # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):  # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):  # (batch_size,), (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):  # (batch_size,), (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current gating kinetics (state variable: w) ────────────────────────
		# w_inf: Boltzmann steady-state activation of the M-gate.
		# Fixed half-activation V_half_M = -35 mV and slope k_M = 10 mV.
		# Non-inactivating: w approaches w_inf with a single slow time constant tau_w_M.
		# This is the minimal, most identifiable parameterisation of the M-current.
		def w_inf_fn(x):  # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_M))

		# tau_w_M: (batch_size,) — already clamped, constant per sample across time

		# ── Allocate state tensors ───────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) [mV]
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K⁺ activation
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na⁺ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na⁺ inactivation
		w = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate activation

		# ── Steady-state initialisation ─────────────────────────────────────────
		V_init  = init_voltage.to(device)                          # (batch_size,)
		V[:, 0] = V_init                                           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)
		w[:, 0] = w_inf_fn(V[:, 0])                                # (batch_size,) M-gate starts at steady-state

		# ── Exponential Euler time integration ──────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# HH gating rates evaluated at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-gate steady state at previous voltage
			w_ss = w_inf_fn(V_prev)   # (batch_size,)

			# ── Effective conductances ─────────────────────────────────────────
			g_Na_eff = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,) [mS/cm²]
			g_K_eff  = (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) [mS/cm²]
			g_M_eff  = gbar_M * w[:, i - 1]                          # (batch_size,) [mS/cm²]

			# ── Reciprocal membrane time constant ─────────────────────────────
			# tau_V_inv = (sum_i g_i) / C   units: [mS/cm²] / [uF/cm²] = [1/ms]
			tau_V_inv = (g_Na_eff + g_K_eff + g_leak + g_M_eff) / C  # (batch_size,) [1/ms]

			# ── Noise sample ──────────────────────────────────────────────────
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)   # (batch_size,) current noise [uA/cm²]

			# ── Voltage steady-state ──────────────────────────────────────────
			# V_inf = (sum_i g_i*E_i + I_inj + noise) / (C * tau_V_inv)
			# M-current carries outward K⁺, reverses at E_K (same as delayed rectifier)
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff  * E_K
				+ g_leak   * E_leak
				+ g_M_eff  * E_K       # (batch_size,) M-current outward K⁺ contribution
				+ input_current[:, i - 1]
				+ noise
			) / (tau_V_inv * C)        # (batch_size,) [mV]

			# ── Exponential Euler state updates ───────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                          # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))         # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))         # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))         # (batch_size,)
			# M-gate: constant tau_w_M per batch element, guaranteed ≥ 10 ms → no div-by-zero risk
			w[:, i] = w_ss + (w[:, i - 1] - w_ss) * Exp(-tstep / tau_w_M)                                         # (batch_size,)

		# Optional observation noise (currently zero per specification)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)