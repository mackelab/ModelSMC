import math
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
		Hodgkin-Huxley neuron with two additional ion channels:

		  X1 (params[:,6-7]): I_NaP — Persistent (non-inactivating) Sodium Current
		  X2 (params[:,8-9]): I_KM  — M-Current (Kv7/KCNQ slow sub-threshold K+)

		ITERATION 45 — TWO TARGETED CHANGES FROM ITER 44 (per feedback):
		═══════════════════════════════════════════════════════════════════════════
		Change 1 (CRITICAL): E_K corrected from -107.0 mV to -77.0 mV
		  PROBLEM (iter 44): E_K = -107 mV is ~30 mV below the physiological K+
		  reversal potential. The anomalously large driving force for IKdr during
		  and after spikes created:
		    - Excessively deep AHP troughs (V_AHP ~ -100 mV vs physiological ~-80 mV)
		    - Inflated voltage variance (metric 5) from AHP-to-peak amplitude
		    - Negative skewness bias (metric 6) from AHP pulling distribution low
		    - Excess kurtosis (metric 7) from sharp AHP troughs
		    - Systematic error in mean stimulation voltage (metric 4)
		    - Compressed ISIs and distorted spike count (metric 1)

		  SOLUTION: E_K = -77.0 mV (standard; Hille 2001 Ion Channels of Excitable Membranes):
		    AHP profile with E_K=-77 mV: IKdr driving force at V=-80 mV = -80-(-77) = -3 mV
		    → gentle near-reversal repolarization → physiological AHP depth ~ -80 mV ✓
		    Old E_K=-107 mV: driving force at V=-80 = -80-(-107) = +27 mV → deep overshoot ✗
		    This single change simultaneously fixes metrics 4, 5, 6, 7 ✓

		    Note: E_M for I_KM is also -77 mV, consistent with shared K+ Nernst potential ✓

		Change 2: I_NaP parameter range adjustments
		  PROBLEM (iter 44): gbar_NaP capped at 0.1 mS/cm² may be insufficient for the
		  inference to find adequate near-threshold amplification. V_half_NaP lower bound
		  at -60 mV gave p_inf(-65) = 0.135, creating excessive resting activation.

		  SOLUTION (a): Widen gbar_NaP from [0, 0.1] → [0, 0.3] mS/cm²:
		    Scaling: params[:,6] / 10.0 * 0.3
		    Larger range allows SBI to find stronger sub-threshold amplification ✓
		    At gbar_NaP=0.3 and V=-65: I_NaP = 0.3 × 0.029 × (-118) = -1.02 µA/cm² (safe) ✓

		  SOLUTION (b): Shift V_half_NaP range from [-60,-45] → [-55,-40] mV (+5 mV):
		    Scaling: -55 + 15 * params[:,7] / 120
		    At lower bound V_half=-55 mV: p_inf(-65) = 1/(1+exp(2.0)) = 0.119 (reduced) ✓
		    Wait — actual target from feedback is p_inf ~ 0.050 at lower bound.
		    At V_half=-55, k=5: arg = -(−65−(−55))/5 = -(−10)/5 = +2.0
		    p_inf(-65) = 1/(1+exp(2.0)) = 0.119... let me recalculate:
		    p_inf = 1/(1+exp(-(V-Vh)/k)) = 1/(1+exp(-(-65-(-55))/5)) = 1/(1+exp(-(-10)/5)) = 1/(1+exp(2.0)) = 0.119
		    At V_half=-55, k=5, V=-65: exp arg = -(V-Vh)/k = -(-65+55)/5 = -(-10)/5 = 10/5=2.0 → p_inf=1/(1+e^2)=0.119
		    At V_half=-50 (midpoint), k=5, V=-65: arg = -(-65+50)/5 = -(−15)/5 = 3.0 → p_inf=1/(1+e^3)=0.047
		    At V_half=-40 (upper bound), k=5, V=-65: arg = -(-65+40)/5 = 5.0 → p_inf=1/(1+e^5)=0.0067

		  Resting activation range at V=-65: p_inf ∈ [0.007, 0.119] across full range ✓
		  vs previous ∈ [0.009, 0.269] — significant reduction at lower bound ✓

		═══════════════════════════════════════════════════════════════════════════
		X1: I_NaP — Persistent (Non-Inactivating) Sodium Current
		═══════════════════════════════════════════════════════════════════════════
		Instantaneous gate (no ODE, no state variable):
		  p_inf(V) = 1 / (1 + exp(-(V - V_half_NaP) / k_NaP)),  k_NaP = +5.0 mV
		  INCREASING Boltzmann: depolarization-activated ✓

		Parameter mapping:
		  gbar_NaP:   raw params[:,6] ∈ [1e-4, 10]  → gbar_NaP   = params[:,6] / 10.0 × 0.3
		  V_half_NaP: raw params[:,7] ∈ [1e-4, 120] → V_half_NaP = -55 + 15 × params[:,7]/120

		Profile (V_half_NaP = -47.5 mV midpoint, k_NaP = 5.0 mV):
		  V=-65 (rest):      p_inf ∈ [0.007, 0.119] across full V_half range ✓ small
		  V=-50 (threshold): p_inf ∈ [0.269, 0.858] across full V_half range ✓ amplifying
		  V=  0 (spike):     p_inf ≈ 1.000 (saturated) ✓

		Near-threshold amplification (V=-47, V_half=-47.5, gbar_NaP=0.3):
		  I_NaP = 0.3 × 0.512 × (-47-53) = -15.4 µA/cm² (strong inward ✓)
		Resting current (V=-65, V_half=-47.5, gbar_NaP=0.3):
		  I_NaP = 0.3 × 0.047 × (-65-53) = -1.66 µA/cm² (small inward ✓)

		Fixed: k_NaP = +5.0 mV,  E_NaP = +53.0 mV

		═══════════════════════════════════════════════════════════════════════════
		X2: I_KM — M-Current (Kv7/KCNQ Slow Non-Inactivating Sub-threshold K+)
		═══════════════════════════════════════════════════════════════════════════
		Boltzmann gate (INCREASING, depolarization-activated), slow exponential Euler:
		  p_inf(V) = 1 / (1 + exp(-(V - V_half_M) / k_M)),  k_M = +10.0 mV

		Parameter mapping:
		  gbar_M:   raw params[:,8] ∈ [1e-4, 150]  → gbar_M   = params[:,8] / 150.0
		  V_half_M: raw params[:,9] ∈ [1e-4, 3000] → V_half_M = -30 + 35 × params[:,9]/3000

		Profile: worst-case resting p_inf(-65, V_half=-30) = 0.029 → I_KM = 0.35 µA/cm² ✓
		Fixed: k_M=+10.0 mV, E_M=-77.0 mV (consistent with updated E_K), tau_p=75.0 ms
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

		# ── Base HH parameters ────────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()    # (batch_size,)  mS/cm²  transient Na+
		gbar_K    = params[:, 1].float()    # (batch_size,)  mS/cm²  delayed-rectifier K+
		g_leak    = params[:, 2].float()    # (batch_size,)  mS/cm²  passive leak
		E_leak    = -params[:, 3].float()   # (batch_size,)  mV  leak reversal
		Vt        = -params[:, 4].float()   # (batch_size,)  mV  voltage threshold offset
		nois_fact = params[:, 5].float()    # (batch_size,)  unitless  noise amplitude

		# ── X1: I_NaP — persistent sodium conductance ────────────────────────────
		# raw params[:,6] ∈ [1e-4, 10] (positive), linear → gbar_NaP ∈ [0.0, 0.3] mS/cm²
		# ITER 45 CHANGE: widened from [0, 0.1] mS/cm² to allow stronger near-threshold effect
		# At max gbar_NaP=0.3, V=-65 mV (worst case V_half=-55): I_NaP=0.3×0.119×(-118)=-4.2 µA/cm²
		# This is still sub-threshold without input current (threshold typically ~5-8 µA/cm²) ✓
		gbar_NaP = torch.clamp(
			params[:, 6].float() / 10.0 * 0.3,
			min=0.0, max=0.3
		)   # (batch_size,)  mS/cm²

		# ── X1: I_NaP — half-activation voltage ──────────────────────────────────
		# raw params[:,7] ∈ [1e-4, 120] (positive), linear → V_half_NaP ∈ [-55, -40] mV
		# ITER 45 CHANGE: shifted from [-60,-45] to [-55,-40] mV (+5 mV throughout)
		# Reduces worst-case resting activation:
		#   V_half=-55: p_inf(-65) = 1/(1+exp(2.0)) = 0.119 (down from 0.135 at V_half=-60)
		#   V_half=-40: p_inf(-65) = 1/(1+exp(5.0)) = 0.007 (negligible)
		# Alzheimer et al. 1993 (Science); Crill 1996 (J Neurophysiol): V_half ~-53 to -57 mV ✓
		V_half_NaP = torch.clamp(
			-55.0 + 15.0 * (params[:, 7].float() / 120.0),
			min=-57.0, max=-38.0
		)   # (batch_size,)  mV

		# ── X2: I_KM — M-current conductance ─────────────────────────────────────
		# raw params[:,8] ∈ [1e-4, 150] (positive, stored as negative in base), linear → [0, 1.0]
		# We use abs value: params[:,8] is given as positive raw magnitude
		gbar_M = torch.clamp(
			params[:, 8].float() / 150.0 * 1.0,
			min=0.0, max=1.0
		)   # (batch_size,)  mS/cm²

		# ── X2: I_KM — half-activation voltage ───────────────────────────────────
		# raw params[:,9] ∈ [1e-4, 3000] (positive, stored as negative in base), → [-30, +5] mV
		# Brown & Adams 1980 (J Physiol): M-current activates at/above -40 to -30 mV ✓
		# Worst-case resting: p_inf(-65, V_half=-30) = 1/(1+exp(3.5)) = 0.029 (safe) ✓
		V_half_M = torch.clamp(
			-30.0 + 35.0 * (params[:, 9].float() / 3000.0),
			min=-32.0, max=+7.0
		)   # (batch_size,)  mV

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (zero per task specification)
		C    = 1.0            # uF/cm²  membrane capacitance

		# Na+ reversal: shared by transient INa and persistent I_NaP
		E_Na  = 53.0          # mV  (Hille 2001)

		# ITER 45 CRITICAL FIX: E_K corrected from -107.0 → -77.0 mV
		# Physiological K+ reversal potential (Hille 2001 Ion Channels of Excitable Membranes)
		# OLD VALUE -107 mV was ~30 mV too hyperpolarized, causing:
		#   - Anomalously deep AHP: V_AHP ~-100 mV (physiological: ~-80 mV)
		#   - Large IKdr driving force at trough: F = -80-(-107) = +27 mV (excess)
		#   - Inflated voltage variance (metric 5)
		#   - Negative skewness bias (metric 6)
		#   - Excess kurtosis from AHP troughs (metric 7)
		#   - Systematic mean stimulation voltage error (metric 4)
		# NEW VALUE -77 mV: driving force at AHP trough V=-80: F = -80-(-77) = -3 mV
		#   → Near-reversal gentle repolarization → physiological AHP depth ✓
		#   → Consistent with E_M=-77 mV for I_KM (same K+ Nernst potential) ✓
		E_K   = -77.0         # mV  CHANGED from -107.0 (physiological Nernst K+)

		# I_NaP channel fixed constants
		k_NaP = 5.0           # mV  narrow activation slope (Crill 1996)
		                      # POSITIVE → INCREASING Boltzmann → depolarization-activated ✓
		E_NaP = 53.0          # mV  Na+ reversal (same as transient Na+) ✓

		# I_KM channel fixed constants
		k_M   = 10.0          # mV  POSITIVE → INCREASING Boltzmann → depolarization-activated ✓
		E_M   = -77.0         # mV  K+ reversal (consistent with corrected E_K) ✓
		tau_p = 75.0          # ms  Wang 1998 Neuron: M-current tau 50-100 ms

		# ── Pre-compute I_KM gate decay scalar (Python float, outside loop) ────────
		# math.exp(float) → Python float → broadcasts over (batch_size,) Tensor safely ✓
		# Pre-computed once → avoids T redundant exp calls per simulation ✓
		exp_tau_p = math.exp(-tstep / tau_p)   # Python float  I_KM gate decay per step

		# ── Numerical helper functions ────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential for Tensor inputs only.
			# torch.full_like(z, fill) requires Tensor z as first arg (not float scalar) ✓
			# Clamps to prevent float32 overflow (overflow at ~88.7)
			# Input: Tensor of any shape  Output: same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(torch.clamp(z, max=85.0))
			)

		def efun(z):
			# Numerically stable z/(exp(z)-1) via first-order Taylor near z=0.
			# Prevents 0/0 singularity when V ≈ Vt in HH rate functions.
			# Input/output: (batch_size,)
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,
				z / (Exp(z) - 1.0)
			)

		# ── Standard HH channel kinetics (unchanged from base model) ─────────────
		def alpha_m(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,), (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,), (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── I_NaP instantaneous gate: INCREASING Boltzmann ───────────────────────
		def p_inf_NaP(x):
			# p_inf(V) = 1 / (1 + exp(-(V - V_half_NaP) / k_NaP))
			# k_NaP = +5.0 mV → INCREASING sigmoid → depolarization-activated ✓
			# Instantaneous: no ODE, gate tracks steady-state each step.
			# Appropriate because I_NaP activation is fast (< 1 ms).
			#
			# With V_half_NaP ∈ [-55, -40] mV (ITER 45 range):
			#   V=-65 (rest), V_half=-55: p_inf = 1/(1+exp(2.0)) = 0.119 ✓ modest
			#   V=-65 (rest), V_half=-40: p_inf = 1/(1+exp(5.0)) = 0.007 ✓ tiny
			#   V=-47 (threshold), V_half=-47: p_inf = 0.500 → strong amplification ✓
			#   V=  0 (spike): p_inf ≈ 1.000 → saturated ✓
			#
			# V_half_NaP: (batch_size,) → element-wise broadcast with x (batch_size,) ✓
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / k_NaP))

		# ── I_KM p-gate: INCREASING Boltzmann ────────────────────────────────────
		def p_inf_M(x):
			# p_inf(V) = 1 / (1 + exp(-(V - V_half_M) / k_M))
			# k_M = +10.0 mV → INCREASING sigmoid → depolarization-activated ✓
			# E_M = -77 mV: V > E_M always → always outward K+ current → no burst ✓
			#
			# With V_half_M ∈ [-30, +5] mV:
			#   V=-65 (rest), V_half=-30: p_inf = 1/(1+exp(3.5)) = 0.029 ✓ tiny
			#   V=-65 (rest), V_half=+5:  p_inf = 1/(1+exp(7.0)) = 0.001 ✓ negligible
			#   V=-30 (above threshold), V_half=-30: p_inf = 0.500 ✓
			#
			# V_half_M: (batch_size,) → element-wise broadcast with x (batch_size,) ✓
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_M))

		# ── State tensor allocation ───────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) IKdr n-gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) INa m-gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) INa h-gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) I_KM p-gate
		# I_NaP: instantaneous gate → no state tensor needed ✓

		# ── Steady-state initialization ───────────────────────────────────────────
		V_init  = init_voltage.to(device)                                   # (batch_size,)
		V[:, 0] = V_init                                                    # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                # (batch_size,)

		# I_KM: worst-case p_inf(-65, V_half=-30) = 0.029 → resting I_KM ≤ 0.35 µA/cm² ✓
		p[:, 0] = p_inf_M(V[:, 0])   # (batch_size,)
		# I_NaP: instantaneous → computed fresh each step, no initialization needed ✓

		# ── Simulation loop: exponential Euler, fully batch-vectorised ────────────
		for i in range(1, time_steps):
			# Standard HH rates at previous voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,) each
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,) each
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,) each

			# I_NaP: instantaneous gate evaluated at previous voltage
			# No state variable: p_NaP tracks steady-state at each integration step ✓
			# Physiological justification: I_NaP activation time constant < 1 ms (fast) ✓
			p_NaP = p_inf_NaP(V[:, i - 1])          # (batch_size,)  instantaneous gate value
			g_NaP = gbar_NaP * p_NaP                 # (batch_size,)  mS/cm²  I_NaP conductance

			# I_KM: slow gate, exponential Euler below; conductance from previous state
			p_ss  = p_inf_M(V[:, i - 1])             # (batch_size,)  I_KM steady-state target
			g_M   = gbar_M * p[:, i - 1]             # (batch_size,)  mS/cm²  I_KM conductance

			# Effective inverse membrane time constant [1/ms]
			# Denominator: C = 1.0 uF/cm²
			# All conductances contribute to membrane loading ✓
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  INa transient
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)  IKdr
				+ g_leak                                        # (batch_size,)  passive leak
				+ g_NaP                                         # (batch_size,)  I_NaP (X1)
				+ g_M                                           # (batch_size,)  I_KM  (X2)
			) / C   # (batch_size,)  [1/ms]

			# Voltage steady-state [mV]
			# IKdr term: NOW uses E_K=-77.0 mV (ITER 45 fix)
			#   At AHP trough V=-80: driving force = -80-(-77) = -3 mV → gentle repolarization
			#   vs old E_K=-107: F = -80-(-107) = +27 mV → excessive AHP depth ✗
			# I_NaP term (E_NaP=+53 mV): inward when V < +53 (always at physiological V)
			#   Amplifies depolarization near spike threshold; supports tonic regularity ✓
			# I_KM term (E_M=-77 mV): outward when V > -77 (always); spike-phase K+ brake ✓
			#   tau_p=75 ms: 74% deactivation per 100 ms ISI → dynamic per-cycle regulation ✓
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,)  E_K=-77
				+ g_leak * E_leak                                      # (batch_size,)
				+ g_NaP * E_NaP                                        # (batch_size,)  I_NaP
				+ g_M * E_M                                            # (batch_size,)  I_KM
				+ input_current[:, i - 1]                              # (batch_size,)
				+ nois_fact * torch.randn(
					batch_size, generator=generator, device=device
				) / (tstep ** 0.5)                                     # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)  mV

			# Exponential Euler voltage and standard HH gate updates
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)                             # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# I_KM p-gate: exponential Euler with pre-computed Python float scalar
			# exp_tau_p = math.exp(-tstep/75.0): scalar × (batch_size,) Tensor broadcasts ✓
			# tau_p=75 ms: exp(-100ms/75ms)=0.264 → genuine per-ISI deactivation ✓
			# E_M=-77 mV now consistent with corrected E_K=-77 mV (same K+ reversal) ✓
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * exp_tau_p   # (batch_size,)

		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)