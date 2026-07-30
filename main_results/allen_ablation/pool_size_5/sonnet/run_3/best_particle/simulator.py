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
		Hodgkin-Huxley neuron with ONE additional channel: M-current (Kv7/KCNQ).

		DESIGN RATIONALE — Single channel, both param slots:
		  Prior iterations showed that adding two slow K+ channels simultaneously
		  (M-current + AHP) caused double adaptation pressure, suppressing spike
		  count and distorting all higher-order statistics. This iteration returns
		  to parsimony: ONE channel (M-current) using BOTH flexible parameter slots
		  to fully characterize its kinetics without parameter identifiability issues.

		KEY IMPROVEMENT — Tunable V_half_M via param_i:
		  Prior iterations fixed V_half_M at -35 or -45 mV, which either:
		    - Left it too depolarized (-35 mV): weak ISI activation (~5% at -55 mV)
		    - Shifted too hyperpolarized (-45 mV): tonic resting activation (~12%)
		  Making V_half_M a free inferred parameter (via param_i = -params[:,8])
		  eliminates this misspecification risk and lets the posterior find the
		  optimal activation range for the experimental data.

		CRITICAL FIXES maintained from prior iterations:
		  E_K = -77.0 mV (physiological mammalian value; -107.0 caused systematic
		  deep AHP biasing mean voltage and all higher-order statistics).

		Parameter index summary:
		  params[:, 0] = gbar_Na       Na+ maximal conductance (mS/cm²)
		  params[:, 1] = gbar_K        K+ delayed rectifier conductance (mS/cm²)
		  params[:, 2] = g_leak        Leak conductance (mS/cm²)
		  params[:, 3] = |E_leak|      Negated internally → E_leak (mV)
		  params[:, 4] = |Vt|          Negated internally → Vt (mV)
		  params[:, 5] = nois_fact     Noise scaling (unitless)
		  params[:, 6] = gbar_M        M-current conductance (mS/cm²) [1e-4, 10]
		  params[:, 7] = (unused)      X2 slot reserved; no second channel this iter
		  params[:, 8] = |V_half_M|    Negated internally → V_half_M (mV) [1e-4, 150]
		  params[:, 9] = tau_M         M-current time constant (ms) [1e-4, 3000], NOT negated

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) # applied current (uA/cm²)
			dt: float # time step (ms)
			t: torch.Tensor: (time_steps,) # time array (ms)
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces (mV)
		"""
		device = params.device

		# Random generator for reproducibility
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Standard HH parameters ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ maximal conductance mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ delayed rectifier conductance mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal mV (sampler draws positive, negated)
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage offset mV (sampler draws positive, negated)
		nois_fact = params[:, 5].float()   # (batch_size,)  noise scaling (unitless)

		# ── M-current (Kv7/KCNQ) — X1 slot, using BOTH param slots ─────────────
		#
		# Physiological basis (Brown & Adams 1980; Wang & McKinnon 1995):
		#   M-current is a slow, non-inactivating, voltage-gated K+ conductance.
		#   It activates in the subthreshold range (tens of ms timescale), dampens
		#   excitability during inter-spike intervals, and regularizes tonic firing
		#   without producing bursting or silencing. It is the minimal mechanism
		#   required to produce regular, tonic, evenly-spaced action potentials.
		#
		# KEY DESIGN CHOICE — V_half_M is INFERRED (not fixed):
		#   param_i = -params[:, 8]: sampler provides positive [1e-4, 150] mV;
		#   negation gives V_half_M in range (-150, ~0) mV. This allows the
		#   posterior to freely locate the optimal half-activation voltage for the
		#   experimental data, avoiding the bias from fixing V_half_M at a wrong value.
		#   Prior iterations at V_half=-35 mV (too depolarized) and V_half=-45 mV
		#   (tonic resting activation) both gave suboptimal statistics.
		#
		# tau_M = params[:, 9]: sampler provides positive [1e-4, 3000] ms;
		#   NO negation applied — tau_M must be strictly positive for exponential
		#   Euler to be stable. Clamped to min=1e-3 ms as safety measure.
		#   Characteristic slow timescale (tens to hundreds of ms) is the defining
		#   Kv7 feature that shapes inter-spike interval regularity.
		#
		# Boltzmann slope = 10 mV: matches established Kv7 channel characterization.
		#   Fixed (not inferred) to avoid identifiability issues with V_half_M.
		gbar_M   = params[:, 6].float()                          # (batch_size,)  mS/cm², [1e-4, 10]
		# X2 slot (params[:,7]) intentionally unused: parsimony — one channel only
		V_half_M = -params[:, 8].float()                          # (batch_size,)  mV, inferred via negation
		tau_M    = torch.clamp(params[:, 9].float(), min=1e-3)    # (batch_size,)  ms, strictly positive

		tstep = float(dt)  # scalar ms

		# ── Fixed biophysical constants ──────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (disabled per task specification)
		C    = 1.0            # membrane capacitance uF/cm²
		E_Na = 53.0           # Na+ reversal potential mV (standard HH value)
		# CORRECTED from -107.0: E_K = -77.0 mV is the physiological mammalian
		# K+ reversal potential. The prior value caused systematically too-deep
		# afterhyperpolarizations, biasing mean voltage and all distribution statistics.
		E_K  = -77.0          # K+ reversal potential mV (shared by K+ DR and M-current)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential with floor at -500; z: (batch_size,) → (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),  # (batch_size,)  prevent underflow
				torch.exp(z)                           # (batch_size,)
			)

		def efun(z):
			# HH rate denominator: z/(exp(z)-1), Taylor-expanded near zero; z: (batch_size,) → (batch_size,)
			return torch.where(
				torch.abs(z) < 1e-4,
				1 - z / 2,        # (batch_size,)  first-order Taylor expansion avoids 0/0
				z / (Exp(z) - 1)  # (batch_size,)  standard exponential form
			)

		# ── Standard HH channel kinetics ────────────────────────────────────────
		def alpha_m(x):
			# Na+ activation opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			# Na+ activation closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch_size,)

		def alpha_h(x):
			# Na+ inactivation opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch_size,)

		def beta_h(x):
			# Na+ inactivation closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))  # (batch_size,)

		def alpha_n(x):
			# K+ delayed rectifier opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch_size,)

		def beta_n(x):
			# K+ delayed rectifier closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # (batch_size,)

		def tau_x(alpha, beta):
			# Gating time constant (ms); alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)  # (batch_size,)

		def inf_x(alpha, beta):
			# Gating steady-state value (dimensionless); alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)  # (batch_size,)

		# ── M-current (Kv7) Boltzmann gating kinetics ────────────────────────────
		def p_inf(x):
			# Kv7 M-current steady-state activation (Boltzmann); x: (batch_size,) → (batch_size,)
			# V_half_M: inferred per-batch (batch_size,), range ~(-150, 0) mV
			# Slope = 10 mV: fixed, consistent with Kv7 characterization
			# V_half_M is broadcast-compatible with x (both batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))  # (batch_size,)

		# tau_M: used directly in the gating update (no voltage dependence for simplicity)
		# Physiological range: M-current tau typically 30-300 ms (Gutfreund 1995)
		# The posterior will constrain tau_M to the range needed by the data

		# ── State variable allocation ────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ DR activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current activation

		# ── Initial conditions: all gating at voltage-dependent steady states ────
		V_init = init_voltage.to(device)                                     # (batch_size,)
		V[:, 0] = V_init                                                      # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                   # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                   # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                              # (batch_size,)

		# ── Time integration via exponential Euler ───────────────────────────────
		# Exponential Euler is exact for piecewise-linear V dynamics and
		# unconditionally stable — preferred over forward Euler for stiff HH systems.
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Standard HH rate functions at previous timestep voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current steady-state at previous voltage
			p_ss = p_inf(V_prev)   # (batch_size,)

			# Total membrane conductance (determines effective time constant)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,)  Na+ (m³h)
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,)  K+ delayed rectifier (n⁴)
				+ g_leak                                        # (batch_size,)  ohmic leak
				+ gbar_M * p[:, i - 1]                         # (batch_size,)  M-current (Kv7)
			) / C                                               # (batch_size,)  ms⁻¹

			# Voltage steady-state drive: conductance-weighted reversal + external current + noise
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na  # (batch_size,)  Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)  K+ DR drive (E_K=-77 mV)
				+ g_leak * E_leak                                       # (batch_size,)  leak drive
				+ gbar_M * p[:, i - 1] * E_K                          # (batch_size,)  M-current K+ drive
				+ input_current[:, i - 1]                              # (batch_size,)  applied current (uA/cm²)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,)  Wiener noise, correct 1/sqrt(dt) scaling
			) / (tau_V_inv * C)                                        # (batch_size,)  mV

			# Exponential Euler voltage update (exact for linear driving force)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Standard HH gating variable updates (exponential Euler)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)

			# M-current gating: slow exponential relaxation to Boltzmann steady state
			# tau_M inferred from params[:,9] (positive [1e-4, 3000] ms), no voltage dependence
			# Slow timescale shapes ISI without causing bursting or spike suppression
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_M)   # (batch_size,)

		# Return voltage traces (observation noise disabled: nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)