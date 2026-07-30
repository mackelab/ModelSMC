import torch
import torch.nn as nn
import torch.nn.functional as F


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
		Hodgkin-Huxley neuron — confirmed optimal configuration with SINGLE TARGETED CHANGE.

		Channels:
		  1. Fast Na+ (m³h):            E_Na=50mV LOCKED  (base 53mV → NLE=24.8)
		  2. Delayed-rectifier K+ (n⁴): E_K=-85mV LOCKED  (base -107mV → NLE=29.0)
		  3. Leak:                       ohmic, E_leak=-params[:,3]
		  4. M-type K+ (Kv7/KCNQ, X1):  slow K+ adaptation (CONFIRMED OPTIMAL)
		       gbar_M = softplus(params[:,6])*0.1  ∈ [~0.07, ~1.0] mS/cm²
		       tau_w  = softplus(params[:,7]*0.3)*10+4  ∈ [4, ~404] ms
		       V_HALF_M = -35.0 mV, K_M = 10.0 mV  FROZEN SCALARS
		  5. Persistent Na+ (NaP):       subthreshold depolarizing window current
		       g_NaP = 0.03 mS/cm²   REVERTED (0.02 gave same NLE=24.3 as 0.03)
		       V_half_NaP = -55.0 mV  ← SINGLE CHANGE (was -57.0; see rationale)
		       k_NaP = softplus(params[:,8])*0.1+1.0  TUNABLE ∈ [~1.07, ~16.0] mV
		         Controls window current width (Boltzmann slope), orthogonal to Vt ✓
		         Confirmed NLE improvement: 23.8→23.7

		SINGLE TARGETED CHANGE THIS ITERATION:
		  V_half_NaP shifted from -57.0 → -55.0 mV (+2 mV depolarized direction)

		  RATIONALE:
		    The NaP amplitude fine-grid search is exhausted:
		      g_NaP=0.03 → NLE=24.3 (prior confirmed optimum after Ih revert)
		      g_NaP=0.02 → NLE=24.3 (previous iteration; no improvement)
		      g_NaP=0.025 → untested but feedback suggests diminishing returns
		    Since amplitude scaling and slope (k_NaP) are already optimally tuned,
		    the remaining degree of freedom is the Boltzmann threshold V_half_NaP.

		    Physical effect of shifting V_half_NaP from -57 → -55 mV:
		      At V=-65 mV (resting potential), k_NaP=5mV:
		        V_half=-57: m_NaP_inf = 1/(1+exp(-(-65+57)/5)) = 1/(1+exp(1.6)) ≈ 0.17
		        V_half=-55: m_NaP_inf = 1/(1+exp(-(-65+55)/5)) = 1/(1+exp(2.0)) ≈ 0.12
		      → Reduces tonic NaP at rest by ~30%, less persistent depolarizing bias.
		      → Sharpens the threshold for NaP activation onset.
		      → NaP activates more steeply near V=-55mV (closer to spike threshold).
		      → With tunable k_NaP, the inference can compensate for slope if needed.

		    Why -55 mV specifically:
		      - Literature: persistent Na+ half-activation ranges from -55 to -60 mV
		        (Crill 1996, Magistretti & Alonso 1999). -57 mV was midpoint; -55 mV
		        is the upper end of the physiological range.
		      - Small enough change (+2 mV) to isolate this single variable.
		      - Does NOT co-vary with Vt (Vt shifts fast spike threshold via α/β kinetics;
		        V_half_NaP shifts NaP Boltzmann; these are orthogonal mechanisms).
		        Note: prior test V_half_NaP TUNABLE → NLE=24.4 regression was due to
		        co-linearity WITH Vt when both were simultaneously free. Here V_half_NaP
		        is still a LOCKED SCALAR — only its fixed value changes by +2 mV. ✓

		    Revert condition: if NLE > 24.3, revert to V_half_NaP=-57.0 mV.
		    Next candidate if -55 fails: try V_half_NaP=-56.0 mV (1 mV step).

		CONFIRMED CATASTROPHIC — NEVER REPEAT:
		  - Ih fixed g=0.1 mS/cm²:           NLE 23.7→32.8 (resting current disruption)
		  - g_NaP tunable:                    NLE 23.7→28.1 (amplitude+width degeneracy)
		  - nois_fact_obs nonzero:             NLE→28.8
		  - tau_w V-dep cosh-bell:             NLE→27.3
		  - E_K=-107mV (base):                 NLE→29.0
		  - 18 X2 tunable attempts:            ALL regressed
		  - V_half_NaP tunable (not scalar):   NLE→24.4
		  - input_current[:,i]:                regression (must use [:,i-1])
		  - noise outside V_inf numerator:     NLE→26.0
		  - gate clamping (.clamp):            NLE→25.5
		  - w[:,0]=0 (zero init):              NLE→25.6
		  - tau_w floor=20ms:                  NLE→25.4
		  - V_HALF_M tunable:                  NLE→25.2
		  - tau_w no 0.3 factor:               NLE→24.9
		  - E_Na=53mV (base):                  NLE→24.8
		  - g_NaP=0.02:                        NLE=24.3 (no improvement over 0.03)

		PARAMETER MAPPING:
		  params[:,0] = gbar_Na      mS/cm²  Na+ fast conductance
		  params[:,1] = gbar_K       mS/cm²  K+ DR conductance
		  params[:,2] = g_leak       mS/cm²  leak conductance
		  params[:,3] = |E_leak|     mV      E_leak = -params[:,3]
		  params[:,4] = |Vt|         mV      Vt = -params[:,4]
		  params[:,5] = nois_fact    —       noise amplitude
		  params[:,6] = gbar_M_raw   —       gbar_M = softplus(raw)*0.1 ∈ [~0.07, ~1.0] mS/cm²
		  params[:,7] = tau_w_raw    —       tau_w = softplus(raw*0.3)*10+4 ∈ [4, ~404] ms
		  params[:,8] = k_NaP_raw    —       k_NaP = softplus(raw)*0.1+1.0 ∈ [~1.07, ~16.0] mV
		  params[:,9] = (unused)     —       all tunable X2 attempts regressed; reserved
		"""
		device = params.device

		# ── Random generator setup ─────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ─────────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (sign applied internally)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (sign applied internally)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── X1: M-type K+ (Kv7/KCNQ) ─────────────────────────────────────────────
		# Slow voltage-dependent K+ current providing spike-frequency adaptation.
		# All reparameterizations confirmed optimal and preserved exactly:
		#   softplus*0.1:  maps raw→[~0.07,~1.0] mS/cm²; rescued NLE 25.4→23.8
		#   0.3 factor:    compresses tau_w range; confirmed (no-factor→NLE=24.9)
		#   floor +4ms:    confirmed optimal (floor=20ms→NLE=25.4 regression)
		#   V_HALF=-35mV:  frozen scalar (tunable→NLE=25.2 regression)
		#   K_M=10mV:      frozen scalar (frozen with V_HALF_M)
		#   voltage-indep: V-dep cosh-bell tau_w→NLE=27.3 catastrophic
		gbar_M_raw = params[:, 6].float()                          # (batch_size,)  raw ∈ [1e-4, 10]
		gbar_M     = F.softplus(gbar_M_raw) * 0.1                 # (batch_size,)  mS/cm²  [~0.07, ~1.0]

		tau_w_raw  = params[:, 7].float()                          # (batch_size,)  raw ∈ [1e-4, 120]
		tau_w      = F.softplus(tau_w_raw * 0.3) * 10.0 + 4.0     # (batch_size,)  ms  [4, ~404]

		# ── NaP: persistent Na+ window current ────────────────────────────────────
		# k_NaP TUNABLE via params[:,8]: confirmed NLE improvement 23.8→23.7.
		# Controls Boltzmann slope (window width), orthogonal to Vt mechanism ✓
		# Floor=1.0 mV via softplus+1.0: prevents near-zero division instability.
		k_NaP_raw  = params[:, 8].float()                          # (batch_size,)  raw ∈ [1e-4, 150]
		k_NaP      = F.softplus(k_NaP_raw) * 0.1 + 1.0            # (batch_size,)  mV  [~1.07, ~16.0]

		# params[:,9]: UNUSED. 18 prior X2 tunable attempts ALL regressed.
		# Fixed Ih at g=0.1 → catastrophic NLE=32.8. Reserved for future use.

		tstep = float(dt)   # scalar  ms

		# ── Fixed biophysical constants ────────────────────────────────────────────
		nois_fact_obs = 0.0   # LOCKED: nonzero → NLE=28.8 catastrophic
		C    = 1.0            # uF/cm²  standard HH membrane capacitance

		# Grid-confirmed optimal reversal potentials:
		#   E_Na=50mV: LOCKED BEST (base 53mV→NLE=24.8; 50mV→NLE=24.5)
		#   E_K=-85mV: LOCKED BEST (base -107mV→NLE=29.0; -85mV→NLE=24.4)
		E_Na = 50.0    # mV  LOCKED
		E_K  = -85.0   # mV  LOCKED

		# ── M-current frozen structural parameters ─────────────────────────────────
		# Making tunable → NLE=25.2 regression (confirmed iteration 14).
		# w_inf(-65mV) ≈ 0.047: negligible M-conductance at resting potential ✓
		# w_inf(-35mV) = 0.500: half-activation at typical firing threshold ✓
		V_HALF_M = -35.0   # mV  FROZEN SCALAR
		K_M      = 10.0    # mV  FROZEN SCALAR

		# ── NaP fixed scalar parameters ────────────────────────────────────────────
		# g_NaP=0.03: REVERTED. Fine-grid test g_NaP=0.02 gave same NLE=24.3.
		#   Amplitude fine-tuning exhausted; amplitude remains at confirmed optimum.
		# V_half_NaP=-55.0: SINGLE CHANGE (+2mV from -57.0).
		#   Reduces tonic NaP activation at rest:
		#     At V=-65mV, k_NaP=5mV: activation drops from ~0.17 to ~0.12 (-30%)
		#   This is a LOCKED SCALAR (not tunable). Previously making it tunable
		#   caused NLE=24.4 regression via co-linearity with Vt. Here only the
		#   fixed scalar value is changed by +2mV — no new degrees of freedom added.
		#   V_half_NaP=-55mV is within physiological range (literature: -55 to -60mV).
		g_NaP      = 0.03    # mS/cm²  REVERTED to confirmed optimum
		V_half_NaP = -55.0   # mV      RETRIED: +2mV from -57.0 (see rationale above)

		# ── Numerical helpers ──────────────────────────────────────────────────────

		def Exp(z):
			# Numerically stable exp.
			# Upper clamp at 85.0: prevents float32 overflow (max representable ~3.4e38,
			# exp(88.7)≈3.4e38, so clamp at 85 is safe with margin).
			# Lower sentinel at -500: preserves gradient signal near zero crossing.
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z.clamp(max=85.0))
			)   # (same shape as z)

		def efun(z):
			# Stable z/(exp(z)-1). Taylor expansion at |z|<1e-4 avoids 0/0 at z=0.
			# Used in HH α-rate expressions of the form V*efun(V/kT).
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))   # (same shape as z)

		# ── Mainen & Sejnowski (1996) Na+ and K+ kinetics ─────────────────────────

		def alpha_m(x):
			# Na+ fast activation opening rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			# Na+ fast activation closing rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,)

		def alpha_h(x):
			# Na+ inactivation opening rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,)

		def beta_h(x):
			# Na+ inactivation closing rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,)

		def alpha_n(x):
			# K+ DR activation opening rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# K+ DR activation closing rate (ms^-1). (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,)

		def tau_x(alpha, beta):
			# Gating time constant (ms). alpha+beta > 0 always by construction.
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			# Steady-state gate value strictly in (0,1). (batch_size,)
			return alpha / (alpha + beta)   # (batch_size,)

		# ── M-gate Boltzmann steady-state ─────────────────────────────────────────
		def w_inf(x):
			# M-gate Boltzmann activation in (0,1). Increases with depolarization.
			# Frozen parameters V_HALF_M=-35mV, K_M=10mV (tunable→NLE=25.2 regression).
			# (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_HALF_M) / K_M))   # (batch_size,)

		# ── NaP instantaneous Boltzmann ────────────────────────────────────────────
		# Instantaneous approximation: τ_NaP < 1ms << ISI (10–100ms) ✓
		# k_NaP: per-sample tunable tensor (params[:,8] via softplus+floor).
		# V_half_NaP: locked scalar (-55.0 mV, shifted +2mV from -57.0 this iter).
		# Division by k_NaP safe: k_NaP >= 1.07 mV (softplus+1.0 floor) >> 0 ✓
		def m_NaP_inf(x):
			# NaP Boltzmann activation in (0,1). (batch_size,) -> (batch_size,)
			# With V_half_NaP=-55.0: activates more sharply near spike threshold
			# and has reduced tonic activation at resting potential vs -57.0 mV.
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / k_NaP))   # (batch_size,)

		# ── Precompute M-gate decay factor OUTSIDE time loop ──────────────────────
		# tau_w is voltage-independent (constant per draw after parameterization).
		# Hoisting saves (time_steps-1) redundant Exp() calls per batch element.
		# tau_w >= 4ms > 0 guaranteed → decay_w = exp(-dt/tau_w) ∈ (0,1) always ✓
		decay_w = Exp(-tstep / tau_w)   # (batch_size,)  constant over all timesteps

		# ── State tensor allocation ────────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  K+ DR gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na+ act gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na+ inact gate
		w = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  M-gate (X1)
		# NaP: instantaneous gate → no state tensor (τ_NaP << dt).
		# No X2 state tensor: all 18 tunable X2 attempts regressed; fixed Ih catastrophic.

		# ── Steady-state initialisation at t=0 ────────────────────────────────────
		V_init  = init_voltage.to(device)                                    # (batch_size,)  mV
		V[:, 0] = V_init                                                     # (batch_size,)  mV

		# HH fast gates at kinetic steady state.
		# Prevents initial transients that corrupt pre-stimulus resting statistics
		# (metric 2: mean resting V, metric 3: std resting V).
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                  # (batch_size,)  ~0.316 at -65mV
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                  # (batch_size,)  ~0.053 at -65mV
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                  # (batch_size,)  ~0.596 at -65mV

		# M-gate at Boltzmann steady state: CONFIRMED ESSENTIAL (zero init→NLE=25.6).
		# tau_w_min=4ms slow relative to dt; zero init creates slow transients that
		# corrupt resting statistics even in a brief pre-stimulus window.
		# w_inf(-65mV) ≈ 0.047: negligible M-conductance at resting potential ✓
		w[:, 0] = w_inf(V[:, 0])                                            # (batch_size,)  ~0.047 at -65mV

		# ── Time integration: exponential Euler ────────────────────────────────────
		for i in range(1, time_steps):
			# HH rate functions evaluated at V(t-1). All shapes: (batch_size,) ms^-1.
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,)

			# Gating steady states at V(t-1)
			w_ss  = w_inf(V[:, i - 1])        # (batch_size,)  M-gate Boltzmann ss
			m_NaP = m_NaP_inf(V[:, i - 1])   # (batch_size,)  NaP instantaneous activation

			# ── Effective inverse membrane time constant (ms^-1) ──────────────────
			# tau_V_inv = (sum of all conductances) / C.
			# g_leak > 0 (prior lower bound) → tau_V_inv strictly positive always ✓
			# gbar_M*w ∈ [0, ~1.0]: modest adaptation current, no spike suppression ✓
			# g_NaP*m_NaP ∈ [0, 0.03]: small window current (<<gbar_Na 20–120 mS/cm²) ✓
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  Na+ fast (m³h)
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)  K+ DR (n⁴)
				+ g_leak                                        # (batch_size,)  leak (ohmic)
				+ gbar_M * w[:, i - 1]                         # (batch_size,)  M-current (X1)
				+ g_NaP * m_NaP                                 # (batch_size,)  NaP window current
			) / C                                              # (batch_size,)  ms^-1

			# ── Voltage steady state V_inf (mV) ───────────────────────────────────
			# Noise INSIDE numerator: CONFIRMED OPTIMAL (outside→NLE=26.0 regression).
			# input_current[:,i-1]: CONFIRMED OPTIMAL ([:,i]→regression confirmed).
			# NaP: g_NaP*m_NaP*E_Na = inward window current toward +50mV.
			#   With V_half_NaP=-55.0 (this iter), m_NaP_inf at -65mV is smaller
			#   (~0.12 vs ~0.17 at -57.0 for k_NaP=5mV), reducing tonic bias at rest.
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)  Na+ fast
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,)  K+ DR
				+ g_leak * E_leak                                      # (batch_size,)  leak
				+ gbar_M * w[:, i - 1] * E_K                          # (batch_size,)  M-current
				+ g_NaP * m_NaP * E_Na                                 # (batch_size,)  NaP window
				+ input_current[:, i - 1]                              # (batch_size,)  applied stimulus
				+ nois_fact * torch.randn(
					batch_size, generator=generator, device=device
				) / (tstep ** 0.5)                                    # (batch_size,)  OU-style noise
			) / (tau_V_inv * C)                                       # (batch_size,)  mV

			# ── Exponential Euler voltage update ──────────────────────────────────
			# Exact solution of linear ODE with piecewise-constant conductances.
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# ── HH fast gate updates — NO .clamp(0,1) ─────────────────────────────
			# Gate clamping → NLE=25.5 regression (confirmed). Do not reintroduce.
			# α,β > 0 everywhere by construction → inf_x ∈ (0,1) guaranteed ✓
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)

			# ── M-gate update (X1): hoisted voltage-independent decay_w ────────────
			# Exponential Euler: exact for dw/dt = (w_inf(V) - w) / tau_w.
			# decay_w = exp(-dt/tau_w) precomputed outside loop (tau_w constant per run).
			# V-dep tau_w (cosh-bell) → NLE=27.3 catastrophic. Do not reintroduce.
			w[:, i] = w_ss + (w[:, i - 1] - w_ss) * decay_w   # (batch_size,)

		# ── Return voltage (observation noise locked at zero) ──────────────────────
		# nois_fact_obs=0.0 LOCKED: any nonzero → NLE=28.8 catastrophic regression.
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)  mV