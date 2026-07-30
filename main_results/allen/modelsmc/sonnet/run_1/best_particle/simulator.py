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
		Hodgkin-Huxley neuron simulator extended with M-type potassium current (IKm).

		ITERATION HISTORY & TARGETED FIXES
		────────────────────────────────────
		Iter 1 (NLE 36.5): Base HH only, X1/X2 unused.
		Iter 2 (NLE 25.2): Added IKm on X1. Sign bug in half-activation:
		  used `x - Vt - param_i` but param_i<0 → V_half << rest → channel always on.
		Iter 3 (NLE 25.2): Fixed sign to `x - Vt + param_i`. Decoupled p_inf (sigmoid)
		  from tau_p (bell-shaped). tau_p denominator = exp(k*v1)+exp(-k*v1), UN-normalised.
		  X2 unused. Achieved NLE 25.2.
		Iter 4 (NLE 32.3 — REGRESSION): Two simultaneous errors introduced:
		  (A) Added Ih on X2 with V_half=-75 mV, k=10 mV → q_inf(-65mV)≈0.27, channel
		      significantly active at rest → large depolarising bias, corrupted all stats.
		  (B) Normalised tau_p by dividing denom by 2, doubling effective IKm time
		      constant relative to iter-3 → invalidated previously inferred params[:,9].
		THIS ITERATION — Two targeted fixes:
		  FIX 1: Drop Ih entirely. With V_half=-75, E_h=-30, the channel was ~27% open
		    at rest, imposing ~7 uA/cm² depolarising current → inflated mean Vrest,
		    compressed Vrest std, corrupted skewness/kurtosis during stimulation.
		    The resting-state statistics can be addressed by correct IKm alone; Ih
		    would require V_half<=-90 mV to be silent at rest, which leaves too little
		    dynamic range for the optimizer within the parameter prior bounds.
		  FIX 2: Revert tau_p to iter-3 un-normalised form:
		    denom = exp(0.04*v1) + exp(-0.04*v1)  [no /2 factor]
		    At V_half (v1=0): denom=2, tau_p = tau_max/2. Away from V_half: faster.
		    This restores the parameter convention under which NLE 25.2 was achieved,
		    allowing the optimizer to re-use previously optimal params[:,9] values.

		CHANNEL DESIGN: IKm (M-type potassium current, Kv7/KCNQ)
		──────────────────────────────────────────────────────────
		• Non-inactivating, slow K+ current activating near spike threshold
		• Provides spike-frequency adaptation → regularises interspike intervals
		• V_half = Vt + |param_i|: above resting potential (~-35 mV), below spike peak
		  - param_i = -params[:,8] < 0, so |param_i| = params[:,8] in [1e-4, 150] mV
		  - e.g. Vt=-55 mV, params[:,8]=20 → V_half = -55+20 = -35 mV ✓
		• tau_max = |param_j| = -param_j = params[:,9] in [1e-4, 3000] ms
		  - Bell-shaped: peak (tau_max/2) at V_half, faster at flanks
		• Corrects: voltage variance, skewness, kurtosis, spike count regularity
		• Does NOT produce bursting; adaptation → more regular (not less) spiking

		Args:
			init_voltage: torch.Tensor: (batch_size,)              initial voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps)  injected current (uA/cm²)
			dt: float                                               time step (ms)
			t: torch.Tensor: (time_steps,)                         time array (ms)
			params: torch.Tensor: (batch_size, 10)                 biophysical parameters
			seed: int or None                                       random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)  membrane voltage traces (mV)
		"""
		device = params.device

		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Parameter extraction ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm², Na+ maximal conductance
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm², K+ delayed-rectifier conductance
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm², passive leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV,     leak reversal potential
		Vt        = -params[:, 4].float()  # (batch_size,)  mV,     voltage threshold offset
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless noise amplitude

		# ── X1 slot: IKm (M-type K+ current) ─────────────────────────────────────
		# gbar_M  = params[:,6]: maximal M-current conductance in [1e-4, 10] mS/cm²
		# param_i = -params[:,8]: NEGATIVE; |param_i|=params[:,8] in [1e-4, 150] mV
		#   V_half = Vt - param_i = Vt + |param_i|  [SIGN-FIXED in iter 3, preserved here]
		#   e.g. Vt=-55, |param_i|=20 → V_half = -35 mV (above rest, subthreshold) ✓
		# param_j = -params[:,9]: NEGATIVE; tau_max = -param_j = params[:,9] in [1e-4, 3000] ms
		#   Un-normalised bell: peak tau = tau_max/2 at V_half [REVERTED from iter 4]
		gbar_M  = params[:, 6].float()    # (batch_size,)  mS/cm², IKm conductance
		# params[:, 7] (gbar_X2): intentionally unused — Ih dropped (see rationale above)
		param_i = -params[:, 8].float()   # (batch_size,)  mV shift (negative)
		param_j = -params[:, 9].float()   # (batch_size,)  ms scale (negative)

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (unchanged per task spec)
		C    = 1.0            # uF/cm², membrane capacitance
		E_Na = 53.0           # mV, Na+ reversal potential
		E_K  = -107.0         # mV, K+ reversal potential (shared with IKm)

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# (batch_size,) → (batch_size,)  numerically safe exponential
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (batch_size,) → (batch_size,)  handles z≈0 singularity in HH rates
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ─────────────────────────────────────────

		def alpha_m(x):
			# (batch_size,) → (batch_size,)  Na+ m-gate opening rate (ms⁻¹)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			# (batch_size,) → (batch_size,)  Na+ m-gate closing rate (ms⁻¹)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,)

		def alpha_h(x):
			# (batch_size,) → (batch_size,)  Na+ h-gate opening rate (ms⁻¹)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,)

		def beta_h(x):
			# (batch_size,) → (batch_size,)  Na+ h-gate closing rate (ms⁻¹)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,)

		def alpha_n(x):
			# (batch_size,) → (batch_size,)  K+ n-gate opening rate (ms⁻¹)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# (batch_size,) → (batch_size,)  K+ n-gate closing rate (ms⁻¹)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,)

		def tau_x(alpha, beta):
			# (batch_size,), (batch_size,) → (batch_size,)  gating time constant (ms)
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			# (batch_size,), (batch_size,) → (batch_size,)  steady-state gate value
			return alpha / (alpha + beta)   # (batch_size,)

		# ── IKm kinetics (X1 slot) — EDITABLE SECTION ────────────────────────────
		#
		# p_inf: sigmoid steady-state (standard Boltzmann)
		#   v1 = x - Vt + param_i
		#   param_i < 0  →  -param_i > 0  →  V_half = Vt + |param_i| > Vt (above rest) ✓
		#   SIGN CONVENTION: using + param_i (NOT - param_i); validated and fixed in iter 3
		#   Slope 10 mV: standard for M-current (Brown & Adams 1980)
		#
		# tau_p: bell-shaped time constant (peaks at V_half, faster at flanks)
		#   tau_max = -param_j (positive scalar per batch element)
		#   denom = exp(0.04*v1) + exp(-0.04*v1) = 2*cosh(0.04*v1)
		#   At V_half (v1=0): denom=2 → tau_p = tau_max/2  [UN-NORMALISED, iter-3 form]
		#   FIX from iter 4: do NOT divide denom by 2 — restores iter-3 parameter regime
		#   Width ~25 mV (1/0.04), physiologically appropriate for slow K+ channel
		#   Range: tau_max in [1e-4, 3000] ms → effective peak in [0.5e-4, 1500] ms

		def p_inf(x):
			# (batch_size,) → (batch_size,)  IKm sigmoid steady-state gate
			v1 = x - Vt + param_i   # (batch_size,), zero at x = Vt - param_i = V_half
			return 1.0 / (1.0 + Exp(-v1 / 10.0))   # (batch_size,), Boltzmann, slope 10 mV

		def tau_p(x):
			# (batch_size,) → (batch_size,)  IKm bell-shaped time constant (ms)
			# UN-NORMALISED: peak = tau_max/2 at V_half (iter-3 convention, no /2 on denom)
			v1 = x - Vt + param_i          # (batch_size,), centred at V_half
			tau_max = -param_j              # (batch_size,), positive: params[:,9] ms
			denom = Exp(0.04 * v1) + Exp(-0.04 * v1)   # (batch_size,), = 2*cosh(0.04*v1) ≥ 2
			return tau_max / denom   # (batch_size,), bell-shaped, peak = tau_max/2 at v1=0
		# ── END EDITABLE SECTION ─────────────────────────────────────────────────

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ activation
		# IKm gating variable — EDITABLE SECTION
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current activation
		# ── END EDITABLE SECTION ─────────────────────────────────────────────────

		# ── Initial conditions ────────────────────────────────────────────────────
		V_init = init_voltage.to(device)   # (batch_size,)
		V[:, 0] = V_init                   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		# IKm: initialise to sigmoid steady-state at init_voltage — EDITABLE SECTION
		p[:, 0] = p_inf(V[:, 0])   # (batch_size,), near-zero at rest (V_half above rest) ✓
		# ── END EDITABLE SECTION ─────────────────────────────────────────────────

		# ── Exponential Euler integration loop ───────────────────────────────────
		for i in range(1, time_steps):

			Vi = V[:, i - 1]   # (batch_size,) voltage at previous step

			# Standard HH gating rates
			a_m, b_m = alpha_m(Vi), beta_m(Vi)   # (batch_size,) each
			a_h, b_h = alpha_h(Vi), beta_h(Vi)   # (batch_size,) each
			a_n, b_n = alpha_n(Vi), beta_n(Vi)   # (batch_size,) each

			# IKm: compute p_inf and tau_p at current voltage — EDITABLE SECTION
			pi_inf = p_inf(Vi)   # (batch_size,)  IKm steady-state at Vi
			tau_pi = tau_p(Vi)   # (batch_size,)  IKm time constant at Vi (ms)
			# ── END EDITABLE SECTION ─────────────────────────────────────────────

			# Effective membrane conductance (inverse RC time constant × C)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ contribution
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) K+ DR contribution
				+ g_leak                                        # (batch_size,) passive leak
				# IKm: slow K+ conductance (near-zero at rest, active during depolarisation)
				+ gbar_M * p[:, i - 1]                         # (batch_size,) IKm contribution
				# ── END EDITABLE SECTION ─────────────────────────────────────────
			) / C   # (batch_size,)

			# Voltage steady-state numerator (weighted reversal potentials + inputs)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,) K+ DR drive
				+ g_leak * E_leak                                      # (batch_size,) leak drive
				# IKm drives toward E_K = -107 mV during depolarisation → adaptation
				+ gbar_M * p[:, i - 1] * E_K                          # (batch_size,) IKm drive
				# ── END EDITABLE SECTION ─────────────────────────────────────────
				+ input_current[:, i - 1]                             # (batch_size,) injected current
				+ nois_fact * torch.randn(
					batch_size, generator=generator, device=device
				) / (tstep ** 0.5)                                     # (batch_size,) scaled noise
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler updates (exact for piecewise-linear ODE within each step)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)                             # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m)) # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h)) # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n)) # (batch_size,)
			# IKm: exponential Euler with decoupled p_inf and tau_p — EDITABLE SECTION
			p[:, i] = pi_inf + (p[:, i - 1] - pi_inf) * Exp(-tstep / tau_pi)   # (batch_size,)
			# ── END EDITABLE SECTION ─────────────────────────────────────────────

		# Return voltage trace with optional observation noise (nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)