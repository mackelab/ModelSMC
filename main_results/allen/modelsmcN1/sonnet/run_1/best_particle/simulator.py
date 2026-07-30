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
		Hodgkin-Huxley neuron extended with:
		  (1) INaP — Persistent Sodium Current   (params[:,6,7,8], instantaneous steady-state)
		  (2) IKM  — M-type K+ Current           (params[:,9], slow ODE gate, tau_w=50ms)

		COMPLETE ITERATION HISTORY AND FIXES:

		  Iter 2-7: Progressive fixes to k_NaP, gbar_NaP, Ih gating, sigmoid centering,
		            gradient coverage, and TypeError crash from Exp(scalar).

		  Iter 8: gbar_h multiplier 5→1.5; k_NaP scale 14→9 → (1,10) mV.
		         REMAINING: Ih E_h=-30mV always below operating range → structural tonic
		         inward bias. gbar_NaP max ~3 mS/cm² → spontaneous pre-stimulus spiking.

		  Iter 9: Replaced Ih (X2 slot) with IKM (M-type K+). IKM uses E_K=-107mV
		         (always outward), activates only above -50mV (zero at rest via
		         w_inf(-65mV)≈0.0025). gbar_NaP multiplier 3→1. tau_w=100ms.
		         REMAINING:
		         (a) V_half_NaP range (-60,-40)mV too negative: at V=-65mV, V_half=-60mV,
		             k=2mV: p_inf≈0.92 → I_NaP ~109 µA/cm² inward at rest → catastrophic.
		             Even at typical samples, substantial resting INaP biases metrics 2,3.
		         (b) tau_w=100ms ≥ ISI at 10Hz (ISI=100ms): gate barely decays between
		             spikes → nearly constant outward DC offset instead of genuine per-spike
		             adaptation, underregulating voltage distribution moments (metrics 5-7).
		         (c) gbar_KM max 0.497 mS/cm² may be insufficient; posterior clustered
		             near upper bound with poor gradient signal.

		  Iter 10 [THIS ITERATION — two targeted fixes]:

		  Fix 1 [CRITICAL — Shift V_half_NaP range from (-60,-40) to (-50,-30) mV,
		         AND reduce gbar_NaP multiplier from 1.0 to 0.5]:

		    PROBLEM: V_half_NaP lower bound = -60 mV is only 5 mV below rest (-65 mV).
		    At V_half_NaP=-60mV, k_NaP=2mV (both near their lower bounds):
		      p_inf(-65mV) = 1/(1+exp(-(-65-(-60))/2)) = 1/(1+exp(2.5)) ≈ 0.076
		      I_NaP = 1.0 * 0.076 * 118 ≈ 9.0 µA/cm²
		    This is ~9× larger than the leak (~1 µA/cm²) at rest — enough to cause
		    spontaneous depolarisation in many prior samples.
		    At V_half_NaP=-60mV, k_NaP=5.5mV:
		      p_inf(-65mV) = 1/(1+exp(5/5.5)) ≈ 0.40 → even worse.

		    FIX part A: Change V_half_NaP transform from `-(40+20*sigmoid(raw-60))` to
		    `-(30+20*sigmoid(raw-60))`, shifting range from (-60,-40) to (-50,-30) mV.
		    Now at most-negative V_half=-50mV, k=5.5mV (prior center):
		      p_inf(-65mV) = 1/(1+exp(15/5.5)) ≈ 0.059 → I_NaP ≤ 0.5*0.059*118 ≈ 3.5 µA/cm²
		    Still not zero at rest, but manageable and non-dominant.
		    At V_half=-50mV, k=2mV: p_inf(-65mV)=1/(1+exp(7.5))≈0.0005 → negligible.
		    INaP still meaningful near threshold (-45mV): p_inf(-45mV,V_half=-40mV,k=5.5)≈0.40

		    FIX part B: Reduce gbar_NaP multiplier 1.0 → 0.5 → range (0.003, 0.496) mS/cm².
		    At upper bound with V_half=-50mV, k=5.5mV at rest:
		      I_NaP_max = 0.496 * 0.059 * 118 ≈ 3.5 µA/cm² (sub-dominant)
		    At prior center (gbar=0.25, V_half=-40mV, k=5.5mV) at rest:
		      I_NaP = 0.25 * p_inf(-65, -40, 5.5) * 118 ≈ 0.25 * 0.007 * 118 ≈ 0.2 µA/cm²
		    → Negligible at rest across most of prior space; well-controlled.

		  Fix 2 [MAJOR — tau_w 100ms → 50ms, gbar_KM multiplier 0.5 → 1.0]:

		    PROBLEM: At 10Hz tonic firing (ISI=100ms), with tau_w=100ms:
		      w_decay_per_ISI = exp(-100/100) = e^-1 ≈ 0.37
		    Gate retains 37% of activation across ISIs → significant DC accumulation.
		    At 20Hz (ISI=50ms): exp(-50/100) = 0.61 → 61% retention → nearly static gate.
		    Result: IKM acts as a constant outward current offset rather than per-spike
		    adaptation mechanism, making it functionally equivalent to a conductance shift
		    and adding little explanatory power for distribution moments (metrics 5-7).

		    FIX: tau_w = 50ms
		    At 10Hz (ISI=100ms): exp(-100/50) = e^-2 ≈ 0.14 → gate decays ~86% per ISI
		    At 20Hz (ISI=50ms):  exp(-50/50)  = e^-1 ≈ 0.37 → gate decays ~63% per ISI
		    → Genuine per-spike adaptation: activates during AP, decays during ISI,
		      providing cycle-by-cycle outward current that modulates ISI regularity.
		    This better controls voltage variance, skewness, kurtosis (metrics 5-7).

		    ALSO: gbar_KM multiplier 0.5 → 1.0 → range (0.007, 0.993) mS/cm²
		    Doubles available conductance range so inference can find adequate M-current
		    strength without posterior collapsing at the upper bound.
		    Physiological IKM: 0.01-0.8 mS/cm² (Mainen & Sejnowski 1996; Wang 1993).

		FINAL PARAMETER TRANSFORMS:
		  params[:,6]: gbar_NaP  = 0.5*sigmoid(raw-5)          → (0.003, 0.496) mS/cm²  prior [1e-4, 10]
		  params[:,7]: V_half_NaP= -(30+20*sigmoid(raw-60))    → (-50.0, -30.0) mV       prior [1e-4, 120]
		  params[:,8]: k_NaP     = 1+9*sigmoid(raw-75)         → (1.01,  9.99)  mV       prior [1e-4, 150]
		  params[:,9]: gbar_KM   = 1.0*sigmoid((raw-1500)/300) → (0.007, 0.993) mS/cm²   prior [1e-4, 3000]
		  Fixed IKM: V_half_KM=-35 mV, k_KM=5 mV, E_KM=E_K=-107 mV, tau_w=50 ms

		Args:
			init_voltage : torch.Tensor (batch_size,)            initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) injected current (uA/cm²)
			dt           : float                                 time step (ms)
			t            : torch.Tensor (time_steps,)            time array (ms)
			params       : torch.Tensor (batch_size, 10)         biophysical parameters
			seed         : int or None                           random seed

		Returns:
			V            : torch.Tensor (batch_size, time_steps) membrane voltage (mV)
		"""
		device = params.device

		# Set up random generator for reproducibility
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ───────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²  fast Na+ maximal conductance
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²  K+ delayed rectifier
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²  passive leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV      leak reversal (sign-flipped)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV      spike threshold offset (sign-flipped)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless current noise amplitude

		# ── INaP parameters ───────────────────────────────────────────────────────

		# gbar_NaP — INaP maximal conductance
		# Prior [1e-4, 10], argument (raw-5) ∈ (-5, +5) → sigmoid ∈ (0.007, 0.993)
		# FIX (iter 10): multiplier 1.0 → 0.5 → range (0.003, 0.496) mS/cm²
		# At upper bound + V_half=-50mV + k=5.5mV at rest (-65mV):
		#   I_NaP = 0.496 * 0.059 * 118 ≈ 3.5 µA/cm² — sub-dominant, no spontaneous spiking
		# Physiological INaP: 0.01-0.5 mS/cm² (Crill 1996; Magistretti & Alonso 1999)
		gbar_NaP = 0.5 * torch.sigmoid(params[:, 6].float() - 5.0)  # (batch_size,) mS/cm² ∈ (0.003, 0.496)

		# V_half_NaP — INaP half-activation voltage
		# Prior [1e-4, 120], argument (raw-60) ∈ (-60, +60) → sigmoid ∈ (0.001, 0.999)
		# FIX (iter 10): base changed 40→30 → range shifts from (-60,-40) to (-50,-30) mV
		# At most-negative V_half=-50mV (lower bound), rest=-65mV, k=5.5mV (prior center):
		#   p_inf = 1/(1+exp(15/5.5)) ≈ 0.059 — small, controlled resting INaP
		# At V_half=-40mV (upper bound), rest=-65mV, k=5.5mV:
		#   p_inf = 1/(1+exp(25/5.5)) ≈ 0.010 — negligible
		# INaP still active near threshold: p_inf(-45,-40,5.5) ≈ 0.40 — meaningful drive
		# Physiological INaP half-activation: -50 to -35 mV (French et al. 1990)
		V_half_NaP = -(30.0 + 20.0 * torch.sigmoid(params[:, 7].float() - 60.0))  # (batch_size,) mV ∈ (-50,-30)

		# k_NaP — INaP Boltzmann slope (mV)
		# Prior [1e-4, 150], argument (raw-75) ∈ (-75, +75) → sigmoid ∈ (0.001, 0.999)
		# Range (1.01, 9.99) mV; prior center → k_NaP = 1+9*0.5 = 5.5 mV (physiological midpoint)
		# Physiological INaP slope: 4-9 mV (French et al. 1990; Magistretti & Alonso 1999)
		k_NaP = 1.0 + 9.0 * torch.sigmoid(params[:, 8].float() - 75.0)  # (batch_size,) mV ∈ (1.01, 9.99)

		# ── IKM parameters ────────────────────────────────────────────────────────

		# gbar_KM — M-type K+ maximal conductance
		# Prior [1e-4, 3000], argument (raw-1500)/300 ∈ (-5, +5) → smooth gradient throughout
		# FIX (iter 10): multiplier 0.5 → 1.0 → range (0.007, 0.993) mS/cm²
		# Doubles dynamic range; prevents posterior collapsing at upper bound
		# Physiological IKM: 0.01-0.8 mS/cm² (Mainen & Sejnowski 1996; Wang 1993)
		gbar_KM = 1.0 * torch.sigmoid((params[:, 9].float() - 1500.0) / 300.0)  # (batch_size,) mS/cm² ∈ (0.007, 0.993)

		tstep = float(dt)

		# ── Fixed biophysical constants ──────────────────────────────────────────
		nois_fact_obs = 0.0    # observation noise (0 per specification)
		C    = 1.0             # uF/cm²  membrane capacitance
		E_Na = 53.0            # mV      Na+ reversal (fast INa and INaP)
		E_K  = -107.0          # mV      K+ reversal (K+ delayed rectifier and IKM)

		# IKM fixed kinetic constants (Brown & Adams 1980; Wang 1993):
		V_half_KM = -35.0      # mV  IKM half-activation
		                       #     At rest (-65mV): w_inf = 1/(1+exp(30/5)) = 0.0025 → ~zero
		                       #     At spike (+20mV): w_inf ≈ 0.999 → fully open during AP
		k_KM      = 5.0        # mV  IKM Boltzmann slope (depolarisation-activated)
		# FIX (iter 10): tau_w 100ms → 50ms
		# At 10Hz (ISI=100ms): gate decays exp(-100/50)=0.14 → 86% per ISI (genuine adaptation)
		# At 20Hz (ISI=50ms):  gate decays exp(-50/50)=0.37  → 63% per ISI (per-spike modulation)
		# Previous 100ms gave near-DC behaviour at 10Hz (only 37% decay per ISI)
		tau_w     = 50.0       # ms  IKM gate time constant (FIXED); range typical: 20-200ms

		# Precompute IKM gate decay factor as Python scalar (CRITICAL: avoids Exp(float) crash)
		# Exp() requires torch.Tensor; math.exp() handles Python float safely.
		# Argument -tstep/tau_w = -dt/50 ≈ -2e-4 for dt=0.01ms — no overflow risk.
		# w_KM_decay broadcasts correctly with Tensor operands as a Python float scalar.
		w_KM_decay = math.exp(-tstep / tau_w)  # Python float scalar ∈ (0, 1); precomputed once

		####################################
		# Kinetics helper functions
		# NOTE: Exp() requires torch.Tensor as first argument.
		# For scalar constants (e.g., w_KM_decay), use math.exp() directly.

		def Exp(z):
			# z: torch.Tensor of any shape
			# Clamps at -500 to prevent underflow; input must be a Tensor (not Python float)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))  # same shape as z

		def efun(z):
			# Linearised GHK factor z/(exp(z)-1), stable near z=0
			# z: torch.Tensor of any shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))  # same shape as z

		# Standard HH Na+ kinetics (Hodgkin & Huxley 1952)
		def alpha_m(x):
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			v1 = x - Vt - 40     # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2     # (batch_size,)

		def alpha_h(x):
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)          # (batch_size,)

		def beta_h(x):
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))       # (batch_size,)

		# Standard HH K+ delayed-rectifier kinetics
		def alpha_n(x):
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40)              # (batch_size,)

		def tau_x(alpha, beta):
			return 1.0 / (alpha + beta)              # (batch_size,)

		def inf_x(alpha, beta):
			return alpha / (alpha + beta)            # (batch_size,)

		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====

		# INaP steady-state: depolarisation-activated Boltzmann (instantaneous)
		# p_inf(V) = 1 / (1 + exp(-(V - V_half_NaP) / k_NaP))
		# Instantaneous valid: tau_NaP ~ 1ms << ISI ~ 10-100ms (Mainen et al. 1995)
		# FIX (iter 10): V_half_NaP now in (-50,-30)mV → minimal resting activation
		# At rest (-65mV, V_half=-50mV, k=5.5mV): p_inf ≈ 0.059 → I_NaP ≤ 3.5 µA/cm²
		# Near threshold (-45mV, V_half=-40mV, k=5.5mV): p_inf ≈ 0.40 → meaningful drive
		def p_inf_NaP(x):
			# x: (batch_size,) membrane voltage (mV)
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / k_NaP))  # (batch_size,) ∈ (0, 1)

		# IKM steady-state: depolarisation-activated Boltzmann
		# w_inf(V) = 1 / (1 + exp(-(V - V_half_KM) / k_KM))
		# V_half_KM = -35mV, k_KM = 5mV
		# Activation profile:
		#   V = -65mV (rest):       w_inf = 1/(1+exp(30/5))  = 0.0025 → negligible (no resting bias)
		#   V = -50mV (subthresh):  w_inf = 1/(1+exp(15/5))  = 0.047  → modest activation
		#   V = -35mV (half-activ): w_inf = 0.50             → half-maximal
		#   V = +20mV (AP peak):    w_inf ≈ 0.999            → fully open during spike
		# Slow gate (tau_w=50ms) tracks near-spike activation; decays during ISI
		def w_inf_KM(x):
			# x: (batch_size,) membrane voltage (mV)
			return 1.0 / (1.0 + Exp(-(x - V_half_KM) / k_KM))  # (batch_size,) ∈ (0, 1)

		# ===== END EDITABLE SECTION =====

		####################################

		# ── Allocate state variable tensors ──────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ delayed rectifier gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation gate
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation gate
		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# INaP: no ODE state — algebraic instantaneous steady-state recomputed each step
		# IKM: slow ODE gate (tau_w=50ms); tracks near-spike activation, decays during ISI
		w_KM = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IKM gate ∈ (0, 1)
		# ===== END EDITABLE SECTION =====

		# ── Initialise all gates at steady-state for initial voltage ──────────────
		V_init  = init_voltage.to(device)                            # (batch_size,)
		V[:, 0] = V_init                                             # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))          # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))          # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))          # (batch_size,)
		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# IKM gate at steady-state for initial voltage
		# At rest (~-65mV): w_inf ≈ 0.0025 → negligible initial IKM, no resting bias
		w_KM[:, 0] = w_inf_KM(V_init)  # (batch_size,) IKM gate steady-state at t=0
		# ===== END EDITABLE SECTION =====

		# ── Exponential Euler integration loop ───────────────────────────────────
		for i in range(1, time_steps):
			# Standard HH gating rates at previous voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])  # (batch_size,), (batch_size,)
			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# INaP: algebraic instantaneous steady-state
			p_NaP = p_inf_NaP(V[:, i - 1])   # (batch_size,) INaP open probability ∈ (0, 1)
			# IKM: ODE gate from previous step (reflects slow per-spike activation history)
			w_m = w_KM[:, i - 1]             # (batch_size,) IKM gate ∈ (0, 1)
			# ===== END EDITABLE SECTION =====

			# ── Effective inverse time constant ───────────────────────────────────
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) fast Na+ transient
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) K+ delayed rectifier
				+ g_leak                                        # (batch_size,) passive leak
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				+ gbar_NaP * p_NaP                             # (batch_size,) INaP (subthreshold Na+)
				+ gbar_KM  * w_m                               # (batch_size,) IKM (near-spike K+)
				# ===== END EDITABLE SECTION =====
			) / C  # (batch_size,)

			# ── Weighted voltage steady-state ─────────────────────────────────────
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) fast Na+ → E_Na
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ → E_K
				+ g_leak * E_leak                                       # (batch_size,) leak → E_leak
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				# INaP: Na+-selective → E_Na = +53 mV (depolarising near threshold)
				# V_half_NaP now in (-50,-30)mV: p_inf at rest ≤ 0.06 → controlled resting drive
				# Maximum resting I_NaP: 0.496 * 0.06 * 118 ≈ 3.5 µA/cm² — sub-dominant
				+ gbar_NaP * p_NaP * E_Na                             # (batch_size,) INaP → E_Na=+53mV
				# IKM: K+-selective → E_K = -107 mV (always outward/hyperpolarising)
				# (V - E_K) > 0 always since V > -107mV → current direction always outward
				# At rest (-65mV): w_m ≈ 0.0025 → I_KM ≈ negligible → zero resting potential bias
				# During AP (+20mV): w_m builds toward 1.0 → strong outward repolarisation drive
				# FIX (iter 10): tau_w=50ms provides per-spike dynamics at 10-20Hz firing
				+ gbar_KM * w_m * E_K                                  # (batch_size,) IKM → E_K=-107mV
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                              # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler voltage update (exact for piecewise-linear V dynamics)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Standard HH gate exponential Euler updates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# IKM gate: exponential Euler for dw/dt = (w_inf(V) - w) / tau_w
			# w_KM_decay = math.exp(-tstep/tau_w): precomputed Python scalar
			# FIX (iter 10): tau_w=50ms → w_KM_decay larger → faster decay between spikes
			# At 10Hz: gate decays ~86% per ISI (vs 37% with tau_w=100ms) — genuine per-spike
			# At 20Hz: gate decays ~63% per ISI (vs 14% with tau_w=100ms) — adaptive modulation
			# Python float w_KM_decay broadcasts correctly with Tensor operands
			w_inf_now = w_inf_KM(V[:, i - 1])                                       # (batch_size,) IKM steady-state at V(t-1)
			w_KM[:, i] = w_inf_now + (w_KM[:, i - 1] - w_inf_now) * w_KM_decay      # (batch_size,) exponential Euler update
			# ===== END EDITABLE SECTION =====

		# Return voltage with optional observation noise (currently 0 per specification)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)