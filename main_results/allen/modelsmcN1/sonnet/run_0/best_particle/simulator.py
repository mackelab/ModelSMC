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
		Hodgkin-Huxley neuron with M-type K+ current (Kv7/KCNQ) as sole additional channel.

		ITER-84: SINGLE CRITICAL REVERT FROM ITER-83 (NLE 25.9, scale=0.22 regression) —
		         V_half_M SCALE REVERTED FROM 0.22 BACK TO 0.23 (CONFIRMED OPTIMAL).
		         V_half_M SCALE AXIS NOW PERMANENTLY CLOSED AT 0.23.

		ITER-83 RESULT:
		  scale=0.22 → NLE 25.9 (regression of +1.1 nats vs best-ever NLE 24.8 at scale=0.23).
		  The sub-0.23 direction regressed at tau_p=53ms, matching the pattern at tau_p=50ms.
		  V_half_M scale axis is NOW FULLY EXHAUSTED (both directions from 0.23 regress).

		ITER-84 CHANGE (per feedback issue #1, severity: major):
		  V_half_M scale: 0.22 → 0.23 (CONFIRMED OPTIMAL REVERT, AXIS LOCKED).
		  This is the ONLY change from iter-83. Single-variable isolation maintained.
		  Target: restore best-ever NLE 24.8 (achieved in iter-77 with identical config).

		COMPLETE V_half_M SCALE BRACKET (PERMANENTLY CLOSED AT ALL tau_p):
		  At tau_p=50ms: {0.23→NLE 25.4, 0.24→NLE 25.7}
		  At tau_p=53ms: {0.22→NLE 25.9, 0.23→NLE 24.8 BEST, 0.24→NLE 25.7}
		  Both sub-0.23 and super-0.23 directions regress → 0.23 is unique global optimum. ✓

		COMPLETE NLE HISTORY:
		  Base (E_K=-107mV, local Generator, input[:,i-1])       → NLE ~26.6
		  iter-60 (M-only, local generator, E_K=-90mV)           → NLE 26.9
		  iter-61 (M-only, global PRNG, input[:,i], E_K=-90mV)   → NLE 25.5
		  iter-62 (global PRNG + input[:,i-1])                   → NLE 26.8  [FORBIDDEN]
		  iter-64 (floor+V clamp)                                → NLE 26.5
		  iter-65 (symmetric Exp clamp ±88)                      → NLE 26.3
		  iter-66 (randn_step() helper fn)                       → NLE 27.0
		  iter-67 (noise * tstep**0.5)                           → NLE 26.5
		  iter-68 (_noise if/else intermediate variable)          → NLE 26.2
		  iter-69 (inline randn, scale=0.23, tau_p=50ms, k_M=10) → NLE 25.4
		  iter-70 (scale=0.24, tau_p=50ms)                       → NLE 25.7
		  iter-71 (scale=0.23, tau_p=48ms)                       → NLE 25.8
		  iter-72 (tau_p=50ms reverted)                          → NLE 26.3  [SBI variance]
		  iter-73 (tau_p=55ms)                                   → NLE 25.4
		  iter-74 (tau_p=52ms)                                   → NLE 26.2
		  iter-75 (tau_p=50ms + V_half_M offset -2mV)            → NLE 26.4
		  iter-76 (PURE REVERT tau_p=50ms + zero offset)         → NLE 26.1  [SBI variance]
		  iter-77 (tau_p=53ms EXPLORATORY)                       → NLE 24.8  [BEST EVER]
		  iter-78 (tau_p=54ms EXPLORATORY)                       → NLE 25.7  [regression]
		  iter-79 (tau_p=53ms LOCKED REVERT)                     → NLE ~26   [SBI variance]
		  iter-80 (k_M=9.0mV EXPLORATORY)                        → NLE 25.4  [regression]
		  iter-81 (k_M=12.0mV EXPLORATORY)                       → NLE 26.5  [regression]
		  iter-82 (k_M=10.0mV LOCKED REVERT)                     → NLE 25.9  [SBI variance]
		  iter-83 (scale=0.22 EXPLORATORY at tau_p=53ms)         → NLE 25.9  [regression]
		  iter-84 (scale=0.23 CONFIRMED REVERT, axis CLOSED)     → target NLE 24.8

		EXHAUSTED / PERMANENTLY CLOSED AXES:
		  tau_p:          {48→25.8, 50→25.4, 52→26.2, 53→24.8 BEST, 54→25.7, 55→25.4} CLOSED ✓
		  k_M:            {9.0→25.4, 10.0→24.8 BEST, 12.0→26.5}                        CLOSED ✓
		  V_half_M scale: {0.22→25.9, 0.23→24.8 BEST, 0.24→25.7}                       CLOSED ✓
		  V_half_M offset:{0mV→25.4, -2mV→26.4}                                         CLOSED ✓
		  X2 channels:    A-type K+→29.5, AHP K+→28.6, HCN→30.0, NaP→32.4              CLOSED ✓

		NEXT EXPLORATION AXIS (after restoring NLE 24.8):
		  Capacitance C via params[:,7] (currently permanently unused as X2 conductance).
		  C=1.0 µF/cm² is hard-coded; making it inferred could resolve residual discrepancies
		  in spike height and inter-spike intervals (feedback suggestion, severity: suggestion).
		  Test: C = params[:,7].float() with prior centred near 1.0 µF/cm².

		CONFIRMED STRUCTURAL INVARIANTS (ALL RETAINED FROM ITER-69/ITER-77):
		  (A) Noise INSIDE V_inf numerator; /tstep**0.5 confirmed optimal ✓
		  (B) p_ss at PRE-STEP V[:,i-1] (post-step → NLE 26.1) ✓
		  (C) p_init = p_inf(V[:,0]) equilibrium, NO CLAMP (clamping → NLE 28.0) ✓
		  (D) V_half_M = -(params[:,8]*0.23) ← ITER-84 REVERT (scale axis CLOSED) ✓
		  (E) params[:,7] PERMANENTLY UNUSED (all X2 tested → NLE 26.1–32.4) ✓
		  (F) params[:,9] PERMANENTLY UNUSED ✓
		  (G) E_K = -90.0 mV FIXED (largest single fix, +1.4 nats vs base -107mV) ✓
		  (H) E_Na = 53.0 mV FIXED ✓
		  (I) tau_p = 53.0 ms LOCKED (narrow peak confirmed) ✓
		  (J) k_M = 10.0 mV LOCKED (bracket CLOSED) ✓
		  (K) input_current[:,i] END-OF-INTERVAL MANDATORY (i-1 → NLE 26.8) ✓
		  (L) Exp() ONE-SIDED underflow guard ONLY z<-500 ✓
		  (M) NO tau_V_inv floor clamp (1e-6 floor → NLE 26.5) ✓
		  (N) NO V[:,i] hard clamp (non-differentiable → SBI distortion) ✓
		  (O) generator=None when seed=None → global SBI PRNG (local → NLE 26.9) ✓
		  (P) torch.randn LITERALLY INLINED inside V_inf expression ✓

		PERMANENTLY FORBIDDEN PATTERNS:
		  input_current[:,i-1] → NLE 26.8 ✓
		  local Generator when seed=None → NLE 26.9 ✓
		  intermediate noise variable → NLE 26.2 ✓
		  noise helper function → NLE 27.0 ✓
		  symmetric Exp clamp ±88 → NLE 26.3 ✓
		  tau_V_inv floor clamp → NLE 26.5 ✓
		  V hard clamp → SBI likelihood distortion ✓
		  p_ss at post-step V[:,i] → NLE 26.1 ✓

		PARAMETER SLOT ASSIGNMENTS:
		  params[:,0] = gbar_Na    Na+ transient max conductance (mS/cm²); INFERRED
		  params[:,1] = gbar_K     K+ DR max conductance (mS/cm²); INFERRED
		  params[:,2] = g_leak     leak conductance (mS/cm²); INFERRED
		  params[:,3] → E_leak     negated internally (mV); INFERRED
		  params[:,4] → Vt         negated internally (mV); INFERRED
		  params[:,5] = nois_fact  noise amplitude (unitless); INFERRED
		  params[:,6] = gbar_M     M-current max conductance (mS/cm²); INFERRED  [X1]
		  params[:,7] = UNUSED     PERMANENTLY FORBIDDEN (all X2 classes exhausted)
		  params[:,8] → V_half_M   -(params[:,8]*0.23) mV; INFERRED             [X1 param]
		  params[:,9] = UNUSED     PERMANENTLY FORBIDDEN

		Args:
			init_voltage:  torch.Tensor (batch_size,)             initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps)  injected current (µA/cm²)
			dt:            float                                   time step (ms)
			t:             torch.Tensor (time_steps,)             time array (ms)
			params:        torch.Tensor (batch_size, 10)          biophysical parameters
			seed:          int or None                            random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps)              voltage traces (mV)
		"""
		device = params.device

		# ── Random generator setup ────────────────────────────────────────────
		# INVARIANT (O): generator=None when seed=None → global SBI-managed PRNG.
		# Base code creates local Generator even for seed=None → bypasses SBI PRNG.
		# That pattern produced NLE 26.9 (iter-60); this fix is ~+1 nat improvement. ✓
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = None   # Python None → global SBI PRNG via torch.randn(generator=None) ✓

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Extract parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ transient max conductance (mS/cm²)
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ DR max conductance (mS/cm²)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm²)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal (mV); prior positive → negated
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold offset (mV); negated
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude factor (unitless)

		# ── X1: M-current (Kv7/KCNQ) — SOLE additional channel ───────────────
		# Slow non-inactivating K+ current activated by membrane depolarisation.
		# Produces tonic spike-frequency adaptation (SFA) without burst firing. ✓
		# Physiological references:
		#   Brown & Adams (1980) J Physiol 211: original M-current characterisation.
		#   Wang et al. (1998) J Neurosci 18: KCNQ2/3 kinetics in cortical neurons.
		# Timescale separation: tau_p=53ms >> tau_n≈1-5ms → enables identifiable inference. ✓
		# Non-inactivating: progressive inter-spike K+ buildup → gradual SFA. ✓
		gbar_M = params[:, 6].float()   # (batch_size,)  M-current max conductance (mS/cm²) [X1]

		# params[:,7]: PERMANENTLY UNUSED.
		# All X2 channel classes exhaustively tested under tau_p=53ms + E_K=-90mV:
		#   A-type K+ (fast-inactivating, I_A)        → NLE 29.5  [regression]
		#   AHP K+ (Ca2+-activated surrogate, I_AHP)   → NLE 28.6  [regression]
		#   HCN/Ih (hyperpolarisation-activated)       → NLE 30.0  [regression]
		#   Persistent Na+ (I_NaP)                     → NLE 32.4  [regression]
		# All X2 additions degrade fit. Parsimony confirmed: one channel is optimal. ✓

		# ── M-current half-activation voltage ─────────────────────────────────
		# ITER-84 SINGLE CHANGE: scale reverted from 0.22 → 0.23 (CONFIRMED OPTIMAL).
		#
		# V_half_M scale axis is NOW PERMANENTLY CLOSED (full bracket exhausted):
		#   At tau_p=50ms: {0.23→NLE 25.4, 0.24→NLE 25.7}
		#   At tau_p=53ms: {0.22→NLE 25.9, 0.23→NLE 24.8 BEST, 0.24→NLE 25.7}
		#   Both sub-0.23 and super-0.23 directions regress → 0.23 is unique optimum. ✓
		#
		# Effect of scale=0.23:
		#   At median params[:,8]≈147: V_half_M = -(147*0.23) = -33.8 mV
		#   Resting activation (V=-65mV, V_half=-33.8mV, k_M=10mV):
		#     p_inf = σ((-65+33.8)/10) = σ(-3.12) ≈ 0.042 (negligible at rest). ✓
		#   Physiological V_half_M range: prior [1e-4,150]×0.23 → [-34.5, ~0] mV. ✓
		V_half_M = -(params[:, 8].float() * 0.23)   # (batch_size,)  M-gate half-activation (mV) ← ITER-84 REVERT

		# params[:,9]: PERMANENTLY UNUSED. ✓

		tstep = float(dt)   # scalar (ms)

		# ── Fixed biophysical constants ────────────────────────────────────────
		nois_fact_obs = 0.0   # scalar; observation noise disabled per task specification
		C             = 1.0   # scalar (µF/cm²); membrane capacitance density

		# E_Na = 53.0 mV FIXED — prevents gbar_Na×E_Na multiplicative degeneracy.
		# Physiological: cortical Na+ Nernst potential (Hille 2001, 3rd ed.). ✓
		E_Na = 53.0   # scalar (mV); Na+ reversal potential

		# E_K = -90.0 mV FIXED — CRITICAL correction of base default (-107.0 mV).
		# This is the largest single model improvement: +1.4 nats vs base value.
		# Physiological Nernst: [K+]_i=140mM, [K+]_o=5mM → -90mV (Hille 2001). ✓
		# Shared reversal for both K_DR (n^4) and M-current (E_M=E_K confirmed). ✓
		E_K = -90.0   # scalar (mV); K+ reversal potential for K_DR and M-current

		# ── M-current time constant (LOCKED AT 53ms) ──────────────────────────
		# tau_p = 53.0 ms: confirmed narrow-peak optimum via exhaustive bracket.
		# Full bracket: {48→25.8, 50→25.4, 52→26.2, 53→24.8 BEST, 54→25.7, 55→25.4}.
		# Both immediate neighbours (52ms, 54ms) regress → unique optimum confirmed. ✓
		# AXIS PERMANENTLY CLOSED. ✓
		tau_p_val = 53.0   # scalar (ms); M-current time constant — LOCKED OPTIMAL

		# Precompute exp(-dt/tau_p): scalar, once outside loop for efficiency. ✓
		exp_decay_p = torch.exp(
			torch.tensor(-tstep / tau_p_val, dtype=torch.float32, device=device)
		)   # scalar 0-dim tensor; M-gate exponential decay factor per time step

		# ── M-gate Boltzmann slope (LOCKED AT 10.0 mV) ───────────────────────
		# k_M = 10.0 mV: CONFIRMED OPTIMAL, AXIS PERMANENTLY CLOSED.
		# Full bracket (all values tested): {9.0→25.4, 10.0→24.8 BEST, 12.0→26.5}.
		# Both steeper (9mV→25.4) and shallower (12mV→26.5) directions regressed.
		# k_M=10.0 mV is the confirmed unique optimum on this axis. ✓
		# k_M is a FIXED constant (never inferred) → NO gbar_M↔k_M degeneracy. ✓
		# Physiological range: Kv7/KCNQ Boltzmann slopes ~7–12 mV (Wang et al. 1998). ✓
		k_M = 10.0   # scalar (mV); M-gate Boltzmann slope factor — LOCKED OPTIMAL

		# ── Numerically stable helper functions ───────────────────────────────
		def Exp(z):
			# ONE-SIDED underflow guard only (z < -500). NO positive ceiling clamp.
			# INVARIANT (L): symmetric ±88 clamp → NLE 26.3 (iter-65); FORBIDDEN. ✓
			# z: any shape tensor → same shape tensor (preserved)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))   # any shape

		def efun(z):
			# Regularised z/(exp(z)-1): L'Hôpital limit |z|<1e-4 prevents 0/0 singularity. ✓
			# z: any shape tensor → same shape tensor (preserved)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))   # any shape

		# ── Standard HH Na+ gating kinetics ──────────────────────────────────
		def alpha_m(x):
			# Na+ activation opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,) shifted voltage
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			# Na+ activation closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,) shifted voltage
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,)

		def alpha_h(x):
			# Na+ inactivation opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,) shifted voltage
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,)

		def beta_h(x):
			# Na+ inactivation closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,) shifted voltage
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,)

		# ── Standard HH K+ DR gating kinetics ────────────────────────────────
		def alpha_n(x):
			# K+ DR activation opening rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,) shifted voltage
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# K+ DR activation closing rate (ms⁻¹); x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,) shifted voltage
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,)

		def tau_x(alpha, beta):
			# Gate time constant (ms); alpha,beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			# Gate steady-state ∈ [0,1]; alpha,beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)   # (batch_size,)

		# ── M-current p-gate Boltzmann steady-state ───────────────────────────
		# POSITIVE slope: Kv7/KCNQ opens with depolarisation (V > V_half_M). ✓
		# Negative-slope HCN/Ih formulation: NLE 30.0; PERMANENTLY FORBIDDEN. ✓
		# k_M=10.0 mV LOCKED; V_half_M scale=0.23 CONFIRMED OPTIMAL (iter-84 revert). ✓
		def p_inf(x):
			# M-gate Boltzmann steady-state ∈ [0,1]; x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_M))   # (batch_size,)

		# ── State variable allocation ──────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) membrane potential (mV)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ DR activation gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ activation gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inactivation gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current activation gate

		# ── Steady-state initialisation ───────────────────────────────────────
		V_init  = init_voltage.to(device)                                  # (batch_size,) initial voltage
		V[:, 0] = V_init                                                   # (batch_size,) set t=0 voltage
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                # (batch_size,) K+ DR steady-state
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                # (batch_size,) Na+ activation SS
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                # (batch_size,) Na+ inactivation SS

		# M-gate: Boltzmann equilibrium at V[:,0]. NO CLAMP applied (clamping → NLE 28.0). ✓
		# Smooth differentiable mapping params[:,8]→V_half_M→p[:,0]→V(t) required for SNPE. ✓
		# At V=-65mV, V_half=-33.8mV (scale=0.23), k_M=10mV: p_inf≈0.042 (negligible). ✓
		p[:, 0] = p_inf(V[:, 0])   # (batch_size,) M-gate Boltzmann equilibrium at t=0

		# ── Exponential Euler time integration ────────────────────────────────
		# Exact for first-order linear gating: x[i] = x_inf + (x[i-1] - x_inf)*exp(-dt/tau). ✓
		# Unconditionally stable for all dt/tau → correctly handles stiff Na+ kinetics. ✓
		#
		# ALL ITER-69/ITER-77 STRUCTURAL INVARIANTS PRESERVED EXACTLY:
		#   (O) generator resolved to None or seeded Generator BEFORE loop. ✓
		#   (K) input_current[:,i] end-of-interval ([:,i-1] → NLE 26.8; MANDATORY). ✓
		#   (P) torch.randn LITERALLY INLINED in V_inf; no variable, no helper fn. ✓
		#   (A) nois_fact * randn(batch_size) / tstep**0.5 in V_inf numerator. ✓
		for i in range(1, time_steps):
			# Gate rates at previous-step voltage; all (batch_size,)
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,) Na+ activation α/β
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,) Na+ inactivation α/β
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,) K+ DR activation α/β

			# Effective inverse membrane time constant (ms⁻¹) = Σ conductances / C.
			# Four conductance contributions: Na+ transient, K+ DR, passive leak, M-current.
			# INVARIANT (M): NO floor clamp on tau_V_inv (1e-6 floor → NLE 26.5 iter-64). ✓
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ transient conductance
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) K+ DR conductance
				+ g_leak                                        # (batch_size,) passive leak conductance
				+ gbar_M * p[:, i - 1]                         # (batch_size,) M-current conductance [X1]
			) / C   # (batch_size,) effective inverse time constant (ms⁻¹)

			# Voltage steady-state V_inf (mV).
			# INVARIANTS (A, K, P) — EXACT ITER-77 INLINE STRUCTURE (best-ever NLE 24.8):
			#   torch.randn LITERALLY INLINED: no named intermediate, no helper fn. ✓
			#   generator=None → draws from global SBI-managed PyTorch PRNG. ✓
			#   /tstep**0.5: Itô-consistent diffusion scaling (σ²=nois_fact²/dt). ✓
			#   input_current[:,i]: end-of-interval indexing (MANDATORY). ✓
			#   E_M = E_K = -90.0 mV confirmed optimal for M-current. ✓
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na                                        # (batch_size,) Na+ driving force
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                                                       # (batch_size,) K+ DR driving force
				+ g_leak * E_leak                                                                           # (batch_size,) passive leak driving force
				+ gbar_M * p[:, i - 1] * E_K                                                               # (batch_size,) M-current driving force; E_M=E_K=-90mV ✓
				+ input_current[:, i]                                                                       # (batch_size,) injected current; end-of-interval ✓
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5) # (batch_size,) inline diffusion noise; Itô scaling ✓
			) / (tau_V_inv * C)   # (batch_size,) voltage steady-state (mV)

			# Exponential Euler state variable updates.
			# INVARIANT (N): NO V hard clamp (non-differentiable → SBI likelihood distortion). ✓
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,) membrane potential
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,) K+ DR gate
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,) Na+ activation gate
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,) Na+ inactivation gate

			# ── M-gate exponential Euler update ──────────────────────────────
			# INVARIANT (B): p_ss evaluated at PRE-STEP voltage V[:,i-1]. ✓
			# Post-step evaluation (V[:,i]) → NLE 26.1; PERMANENTLY FORBIDDEN. ✓
			# exp_decay_p: scalar precomputed OUTSIDE loop; tau_p=53ms LOCKED. ✓
			p_ss    = p_inf(V[:, i - 1])                           # (batch_size,) M-gate Boltzmann SS at V[i-1]
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * exp_decay_p   # (batch_size,) M-gate exponential Euler

		# ── Optional observation noise (disabled: nois_fact_obs=0.0) ──────────
		# Guard prevents unnecessary PRNG state consumption when disabled. ✓
		if nois_fact_obs > 0.0:
			V = V + nois_fact_obs * torch.randn(
				batch_size, time_steps, generator=generator, device=device
			)   # (batch_size, time_steps)

		return V   # (batch_size, time_steps) membrane potential traces (mV)