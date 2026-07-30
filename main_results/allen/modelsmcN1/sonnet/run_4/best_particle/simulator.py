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
		Hodgkin-Huxley neuron + I_A transient K+ (X1) + I_SK Ca2+-activated K+ (X2).

		ITER 54: TWO TARGETED FIXES TO ITER 53 (NLE=31.6) PER EXPLICIT FEEDBACK
		=========================================================================
		ROOT CAUSE OF ITER 53 FAILURE:
		  FIX 1 overcorrected Ca2+ parameters by 20× simultaneously with K_d halving:
		  alpha_Ca=0.04, K_d=0.08 → Ca_ss_spike≈0.467, sk_act_spike≈0.73 (saturating)
		  → Excessive mAHP after every spike suppressed firing rate and distorted all
		    voltage-distribution statistics (mean, variance, skewness, kurtosis)
		  → NLE catastrophe: 26.6 → 31.6

		ITER 54 FIX 1: Moderate Ca2+ parameter rescaling (per feedback recommendation)
		  alpha_Ca: 0.04 → 0.010  (5× from original 0.002; NOT the 20× overcorrection)
		  K_d:      0.08 → 0.18   (per feedback; increased from overcorrected 0.08)
		  Result:
		    Ca_ss_spike = 0.010 × 0.146 × 80 ≈ 0.117
		    sk_act_spike = (0.117 / 0.297)²  ≈ 0.155  ← functional but NOT saturating
		    Ca_ss_rest   = 0.010 × 7.5e-5 × 80 ≈ 6e-4
		    sk_act_rest  = (0.0033)²            ≈ 1e-5  ← effectively zero at rest ✓
		  The channel is now identifiable: gbar_SK can tune mAHP strength in [0, ~1×g_total]
		  without fighting a near-saturated Hill function.

		ITER 54 FIX 2: Revert sk_act to use Ca[:,i-1] (per feedback recommendation)
		  Iter 53 used Ca[:,i] (freshly updated) for sk_act → within-step ordering dependency
		  inconsistent with exponential-Euler scheme applied to all other gates (all use i-1).
		  Iter 54: sk_act computed from Ca[:,i-1], then Ca[:,i] updated after voltage step.
		  This restores uniform first-order Euler structure across all state variables.
		  The one-step lag is negligible vs tau_Ca=80 ms.

		COMPLETE ITERATION HISTORY (lower NLE = better):
		  Iter  4: I_A only, all kinetics fixed, gbar_A inferred      → NLE=24.9 BEST EVER
		  Iter 40: I_M  (X2, slow K+)                                 → NLE=35.9 catastrophe
		  Iter 41: I_h  (X2, mixed cation)                            → NLE=34.0 catastrophe
		  Iter 45: Slow K+ AHP (X2, voltage-gated)                   → NLE=36.4 catastrophe
		  Iter 50: I_NaP (X2, persistent Na+)                        → NLE=35.5 catastrophe
		  Iter 51: Remove I_NaP, pure iter 4 revert                  → NLE=26.2
		  Iter 52: I_SK X2 (alpha_Ca=0.002, K_d=0.3, sk_act_spike≈0.005) → NLE=26.6
		           [SK channel effectively silent; gbar_SK unidentifiable]
		  Iter 53: I_SK FIX 1×20 (alpha_Ca=0.04, K_d=0.08, sk_act_spike≈0.73) → NLE=31.6
		           [SK channel saturating; overwhelming mAHP suppressed firing]
		  Iter 54: I_SK FIX 1×5 (alpha_Ca=0.010, K_d=0.18, sk_act_spike≈0.155)
		           + FIX 2 revert Ca[:,i-1] ordering → TARGET ≤ 25.4

		WHY PRIOR X2 CHANNELS FAILED (common flaw: open at rest):
		  I_M   at -65 mV: ~8% open  → resting K+ outward current → hyperpolarised V_rest
		  I_h   at -65 mV: ~30% open → resting depolarising inward current → shifted V_rest
		  I_NaP at -65 mV: ~12% open → resting Na+ inward current → depolarised V_rest
		  I_SK  at -65 mV: Ca≈6e-4 → (6e-4/0.18)² ≈ 1e-5 open → I_SK≈0 at rest ✓

		PARAMETER ASSIGNMENT:
		  params[:,0] = gbar_Na   (mS/cm2) — Na+ fast transient (Traub HH)
		  params[:,1] = gbar_K    (mS/cm2) — K+ delayed rectifier (Traub HH)
		  params[:,2] = g_leak    (mS/cm2) — leak conductance
		  params[:,3] = |E_leak|  (mV)     — E_leak = -params[:,3]
		  params[:,4] = |Vt|      (mV)     — Vt = -params[:,4] (threshold shift)
		  params[:,5] = nois_fact          — noise amplitude (inside V_inf)
		  params[:,6] = gbar_A    (mS/cm2) — I_A transient K+ conductance (X1)
		  params[:,7] = gbar_SK   (mS/cm2) — I_SK Ca2+-activated K+ conductance (X2)
		  params[:,8] = UNUSED             — all I_SK kinetics fixed; no extra param needed
		  params[:,9] = UNUSED             — not needed

		I_A KINETICS (ALL FIXED SCALARS — iter 4 proven optimal, 14+ failed mods confirm):
		  V_half_a=-40 mV, k_a=5 mV, tau_a=1 ms, V_half_b=-82 mV, k_b=6 mV,
		  tau_b=20 ms, E_A=E_K=-107 mV (iter 48 E_A=-90 mV proved catastrophic)

		I_SK KINETICS (ALL FIXED SCALARS — only gbar_SK inferred, iter 54 calibration):
		  tau_Ca=80 ms, alpha_Ca=0.010, K_d=0.18 (Hill exponent=2, E_SK=E_K=-107 mV)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# Extract base HH parameters (unchanged across all iterations)
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# X1: I_A transient K+ — gbar_A is the ONLY inferred X1 parameter
		# All six I_A kinetic constants are fixed scalars (iter 4 optimal, never change)
		gbar_A  = params[:, 6].float()   # (batch_size,) mS/cm2 [check]

		# X2: I_SK Ca2+-activated K+ — gbar_SK is the ONLY inferred X2 parameter
		# All Ca2+/SK kinetic constants fixed scalars (iter 54 moderate calibration)
		gbar_SK = params[:, 7].float()   # (batch_size,) mS/cm2 [check]

		# params[:,8-9]: UNUSED — all channel kinetics are fixed scalars

		tstep         = float(dt)
		nois_fact_obs = 0.0   # observation noise (0 per task specification)
		C    = 1.0            # uF/cm2 membrane capacitance (standard HH)
		E_Na = 53.0           # mV Traub convention FIXED
		E_K  = -107.0         # mV Traub convention FIXED

		# -----------------------------------------------------------------------
		# I_A kinetic constants — ALL FIXED SCALARS (iter 4 proven optimal)
		# CRITICAL: 14+ failed attempts to modify these caused NLE regression.
		# Do NOT change these values in any future iteration.
		# -----------------------------------------------------------------------
		V_half_a = -40.0   # mV FIXED: I_A activation midpoint (Connor & Stevens 1971)
		k_a      =   5.0   # mV FIXED: I_A activation slope
		tau_a    =   1.0   # ms FIXED: fast I_A activation time constant
		V_half_b = -82.0   # mV FIXED: I_A inactivation midpoint (bisection optimum)
		k_b      =   6.0   # mV FIXED: I_A inactivation slope
		tau_b    =  20.0   # ms FIXED: I_A recovery constant (ISI refractoriness)
		E_A      = E_K     # -107.0 mV FIXED (iter 48 E_A=-90 mV caused NLE=27.0 regression)

		# -----------------------------------------------------------------------
		# I_SK Ca2+ proxy kinetic constants — ALL FIXED SCALARS (iter 54 calibration)
		# FIX 1: alpha_Ca=0.010 (5× from 0.002), K_d=0.18 per feedback recommendation
		# Targeting sk_act_spike ≈ 0.155 (functional, non-saturating):
		#   Ca_ss_spike = 0.010 × 0.146 × 80 ≈ 0.117
		#   sk_act_spike = (0.117/0.297)² ≈ 0.155   ← identifiable by SBI posterior
		#   Ca_ss_rest   = 0.010 × 7.5e-5 × 80 ≈ 6e-4
		#   sk_act_rest  = (6e-4/0.181)² ≈ 1e-5     ← zero resting artifact guaranteed
		# -----------------------------------------------------------------------
		tau_Ca   = 80.0    # ms FIXED: Ca2+ clearance time constant (medium AHP timescale)
		alpha_Ca = 0.010   # FIXED: Ca2+ influx per Na+ open state m³h (5× from iter 52)
		K_d      = 0.18    # FIXED: Ca2+ half-activation constant (normalised units)
		# Hill exponent = 2 (standard SK channels, Hirschberg et al. 1998, Nature Neurosci.)
		# E_SK = E_K = -107.0 mV (SK channels are purely K+-selective)

		# -----------------------------------------------------------------------
		# Numerical helpers (proven iter 4 formulation, unchanged all 54 iterations)
		# -----------------------------------------------------------------------
		def Exp(z):
			# Clamped exponential: prevents float32 underflow for z < -500.
			# torch.full_like(z, fill_value) requires tensor as first arg — z satisfies this.
			# z: any shape -> same shape [check]
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# Numerically stable z/(exp(z)-1) via Taylor expansion at |z|<1e-4.
			# Prevents 0/0 singularity at spike threshold where v1→0.
			# z: any shape -> same shape [check]
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# -----------------------------------------------------------------------
		# Standard Traub HH gate kinetics (unchanged all 54 iterations)
		# -----------------------------------------------------------------------
		def alpha_m(x):
			# Na+ activation opening rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation closing rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation opening rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation closing rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ DR activation opening rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ DR activation closing rate; x: (batch_size,) -> (batch_size,) ms-1 [check]
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant; -> (batch_size,) ms [check]
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state; -> (batch_size,) unitless [check]
			return alpha / (alpha + beta)

		# -----------------------------------------------------------------------
		# I_A gate steady-state functions (ALL scalar kinetic params — iter 4 optimal)
		# -----------------------------------------------------------------------
		def a_inf(x):
			# I_A activation steady-state (Boltzmann sigmoid, Connor & Stevens 1971).
			# a_inf(-65 mV) ≈ 0.007: negligible at rest → no resting I_A artifact [check]
			# x: (batch_size,) -> (batch_size,) [check]
			return 1.0 / (1.0 + Exp(-(x - V_half_a) / k_a))

		def b_inf(x):
			# I_A inactivation steady-state (inverted Boltzmann, bisection optimum iters 29-34).
			# b_inf(-65 mV) ≈ 0.057: a*b ≈ 0.0004 at rest → I_A ≈ 0 guaranteed [check]
			# x: (batch_size,) -> (batch_size,) [check]
			return 1.0 / (1.0 + Exp((x - V_half_b) / k_b))

		# -----------------------------------------------------------------------
		# Scalar exponential decay factors (precomputed outside loop — efficiency)
		# exp_a:  tau_a=1 ms  FIXED — fast I_A activation tracks spike onset
		# exp_b:  tau_b=20 ms FIXED — I_A inactivation recovery (ISI refractoriness)
		# exp_Ca: tau_Ca=80 ms FIXED — Ca2+ clearance (medium AHP timescale)
		# All three are dimensionless scalars that broadcast to (batch_size,) inside loop.
		# -----------------------------------------------------------------------
		exp_a  = torch.exp(torch.tensor(-tstep / tau_a,  device=device))   # scalar [check]
		exp_b  = torch.exp(torch.tensor(-tstep / tau_b,  device=device))   # scalar [check]
		exp_Ca = torch.exp(torch.tensor(-tstep / tau_Ca, device=device))   # scalar [check]

		# -----------------------------------------------------------------------
		# State variable allocation
		# -----------------------------------------------------------------------
		V  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) mV
		n  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) K+ DR gate [check]
		m  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ activation [check]
		h  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ inactivation [check]
		a  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) I_A activation [check]
		b  = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) I_A inactivation [check]
		Ca = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Ca2+ proxy [check]
		# Ca[:,0]=0 by default: at rest m³h≈7.5e-5 → Ca_ss_rest≈6e-4 ≈ 0, valid initialisation

		# -----------------------------------------------------------------------
		# Steady-state initialisation at V_init (~-65 mV at rest)
		# I_A  at rest: a*b ≈ 0.007×0.057 ≈ 0.0004 → I_A≈0 (no t=0 transient)
		# I_SK at rest: Ca[:,0]=0 → sk_act=0 → I_SK=0 (no resting artifact)
		# -----------------------------------------------------------------------
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,) mV
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))            # (batch_size,) [check]
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))            # (batch_size,) [check]
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))            # (batch_size,) [check]
		a[:, 0] = a_inf(V[:, 0])   # (batch_size,) ≈ 0.007 at -65 mV [check]
		b[:, 0] = b_inf(V[:, 0])   # (batch_size,) ≈ 0.057 at -65 mV [check]
		# Ca[:,0] = 0 already from torch.zeros [check]

		# -----------------------------------------------------------------------
		# Exponential-Euler integration loop
		#
		# ITER 54 CHANGES FROM ITER 53 (NLE=31.6):
		#   FIX 1: alpha_Ca 0.04→0.010 (moderate 5× not 20×), K_d 0.08→0.18
		#          → sk_act_spike: 0.73 → 0.155 (non-saturating, identifiable)
		#          → gbar_SK can now tune mAHP strength without saturated Hill function
		#
		#   FIX 2: sk_act reverted to use Ca[:,i-1] (not Ca[:,i] as in iter 53)
		#          → restores uniform exp-Euler structure: all state quantities use i-1
		#          → eliminates within-step ordering inconsistency
		#          → Ca[:,i] update moved to after gate updates (uniform ordering)
		#
		# STATE UPDATE ORDER (uniform exp-Euler, all from i-1):
		#   1. Compute HH gate rates a_m, b_m, a_h, b_h, a_n, b_n from V[:,i-1]
		#   2. Compute I_A gate ss a_ss, b_ss from V[:,i-1]
		#   3. Compute m³h from m[:,i-1], h[:,i-1]
		#   4. Compute sk_act from Ca[:,i-1]  ← FIX 2: uses previous Ca, not current
		#   5. Compute g_total, tau_V_inv, V_inf using all i-1 states
		#   6. Update V[:,i] via exp-Euler
		#   7. Update n,m,h gates via exp-Euler
		#   8. Update a,b gates via exp-Euler
		#   9. Update Ca[:,i] via exp-Euler  ← moved here for uniform ordering
		# -----------------------------------------------------------------------
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) voltage at previous time step [check]

			# Step 1: Traub HH gate rates at V(t-1)
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) ms-1 each [check]
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) ms-1 each [check]
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) ms-1 each [check]

			# Step 2: I_A gate steady states at V(t-1)
			a_ss = a_inf(V_prev)   # (batch_size,) I_A activation ss [check]
			b_ss = b_inf(V_prev)   # (batch_size,) I_A inactivation ss [check]

			# Step 3: Na+ open probability proxy for Ca2+ entry
			# m³h: high during action potential (m≈0.9, h≈0.2 → m³h≈0.146)
			#       low at rest (m≈0.05, h≈0.6 → m³h≈7.5e-5)
			m3h = (m[:, i - 1] ** 3) * h[:, i - 1]   # (batch_size,) Na+ open state [check]

			# Step 4: I_SK activation from Ca[:,i-1] (FIX 2 — uniform exp-Euler ordering)
			# Hill function, exponent=2 (standard SK, purely Ca2+-gated, no voltage dep.)
			# sk_act_rest  ≈ (6e-4/0.181)² ≈ 1e-5 → zero resting artifact guaranteed
			# sk_act_spike ≈ (0.117/0.297)² ≈ 0.155 → functional mAHP current
			sk_act   = (Ca[:, i - 1] / (Ca[:, i - 1] + K_d)) ** 2   # (batch_size,) ∈[0,1] [check]
			g_SK_eff = gbar_SK * sk_act                                # (batch_size,) effective SK cond [check]

			# Step 5a: Effective inverse membrane time constant
			# Conductance sum: Na+ + K+DR + leak + I_A (X1) + I_SK (X2)
			tau_V_inv = (
				m3h * gbar_Na * h[:, i - 1]             # (batch_size,) Na+ fast transient [check]
				+ (n[:, i - 1] ** 4) * gbar_K           # (batch_size,) K+ delayed rectifier [check]
				+ g_leak                                 # (batch_size,) leak [check]
				+ gbar_A * a[:, i - 1] * b[:, i - 1]   # (batch_size,) I_A transient K+ (X1) [check]
				+ g_SK_eff                               # (batch_size,) I_SK Ca2+-activated K+ (X2) [check]
			) / C   # (batch_size,) ms-1 [check]

			# Step 5b: Voltage steady-state
			# NOISE: inside V_inf numerator — MUST NOT CHANGE (iter 47 moving out → NLE=29.2)
			# I_A reversal: E_K=-107 mV FIXED (iter 48 E_A=-90 mV caused NLE=27.0)
			# I_SK reversal: E_K=-107 mV (K+-selective channel)
			V_inf = (
				m3h * gbar_Na * h[:, i - 1] * E_Na            # (batch_size,) Na+ [check]
				+ (n[:, i - 1] ** 4) * gbar_K * E_K           # (batch_size,) K+ DR [check]
				+ g_leak * E_leak                              # (batch_size,) leak [check]
				+ gbar_A * a[:, i - 1] * b[:, i - 1] * E_A   # (batch_size,) I_A: E_K [check]
				+ g_SK_eff * E_K                               # (batch_size,) I_SK: E_K [check]
				+ input_current[:, i - 1]                      # (batch_size,) stimulus [check]
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,) mV [check]

			# Step 6: Exponential-Euler voltage update (exact for piecewise-constant linear ODE)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,) [check]

			# Step 7: Standard Traub HH gate updates (unchanged all 54 iterations)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,) [check]
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,) [check]
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,) [check]

			# Step 8: I_A gate updates (scalar exp_a, exp_b; iter 4 structure preserved)
			# exp_a = exp(-dt/1ms):  fast activation tracks spike onset [check]
			# exp_b = exp(-dt/20ms): slow inactivation recovery controls ISI [check]
			a[:, i] = a_ss + (a[:, i - 1] - a_ss) * exp_a   # (batch_size,) [check]
			b[:, i] = b_ss + (b[:, i - 1] - b_ss) * exp_b   # (batch_size,) [check]

			# Step 9: Ca2+ proxy update (FIX 2 — moved here for uniform ordering)
			# dCa/dt = -Ca/tau_Ca + alpha_Ca * m³h
			# Exp-Euler: Ca[i] = Ca_ss + (Ca[i-1] - Ca_ss) * exp(-dt/tau_Ca)
			# Ca_ss = alpha_Ca * m³h * tau_Ca  (exp-Euler steady-state for current m³h)
			# exp_Ca = exp(-dt/80ms): scalar precomputed [check]
			Ca_ss    = alpha_Ca * m3h * tau_Ca                         # (batch_size,) Ca2+ ss [check]
			Ca[:, i] = Ca_ss + (Ca[:, i - 1] - Ca_ss) * exp_Ca        # (batch_size,) [check]

		# Return voltage traces (observation noise=0 per task specification)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)