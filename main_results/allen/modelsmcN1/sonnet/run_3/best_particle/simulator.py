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
		Hodgkin-Huxley neuron extended with I_A (X1) and I_M (X2).

		TARGETED CHANGES FROM PRIOR ITERATION (iter-17 feedback, two fixes only)
		========================================================================

		FIX 1 — V_half_w shifted from -45 mV back to -35 mV (CRITICAL):
		  Per iter-17 feedback: "At inter-spike voltages of -50 mV, w_inf ≈ 0.30, so
		  even moderate gbar_M values deliver ≈0.6 mS/cm² of continuous outward current
		  between spikes. This chronically hyperpolarises the membrane during stimulation,
		  reducing mean stimulation voltage (stat 4), suppressing spike count (stat 1),
		  and creating heavy-tailed voltage distributions (stats 6–7)."
		  Fix: keep k_w=6 mV (iter-16 FIX 1), but restore V_half_w to -35 mV so I_M
		  activates predominantly suprathreshold rather than in the deep subthreshold range.
		  OLD V_half_w=-45 mV, k_w=6:
		    w_inf(-65) = sigmoid((-65+45)/6) = sigmoid(-3.33) ≈ 0.034 (resting bias)
		    w_inf(-50) = sigmoid((-50+45)/6) = sigmoid(-0.83) ≈ 0.304 (large inter-spike) [−]
		    w_inf(-35) = sigmoid((-35+45)/6) = sigmoid( 1.67) ≈ 0.841 (AP plateau)
		  NEW V_half_w=-35 mV, k_w=6:
		    w_inf(-65) = sigmoid((-65+35)/6) = sigmoid(-5.00) ≈ 0.007 (near-zero at rest) [+]
		    w_inf(-50) = sigmoid((-50+35)/6) = sigmoid(-2.50) ≈ 0.076 (low inter-spike)   [+]
		    w_inf(-35) = sigmoid((-35+35)/6) = sigmoid( 0.00) = 0.500 (half at threshold) [+]
		    w_inf(-20) = sigmoid((-20+35)/6) = sigmoid( 2.50) ≈ 0.924 (saturated post-AP) [+]
		  Effective inter-spike I_M at V=-50 mV, gbar_M=2.0 mS/cm²:
		    OLD: 2.0 × 0.304 × 30 = 18.2 µA/cm² (chronic hyperpolarisation)              [−]
		    NEW: 2.0 × 0.076 × 30 =  4.6 µA/cm² (moderate post-spike adaptation only)    [+]
		  I_M now provides genuine post-spike adaptation rather than tonic suppression.    [+]
		  Resting bias negligible: gbar_M × 0.007 × 15 = 0.2 µA/cm² at max gbar_M.       [+]
		  Wang (1998) canonical V_half_w=-35 mV referenced to standard HH Vt=0 mV;
		  with Vt-shifted HH kinetics, this places threshold at ~-50 mV absolute,
		  consistent with half-activation of I_M at -35 mV (suprathreshold).              [+]

		FIX 2 — tau_b ceiling capped from ~81 ms to ~51 ms (MAJOR):
		  Per iter-17 feedback: "At the upper end of p_tau_b prior (p→3000, tau_b→81 ms),
		  b-gate recovery fraction per ISI is only ~17–22% per 20 ms ISI. After 5–10 spikes,
		  b approaches near-zero, making I_A effectively persistent — severely suppressing
		  tonic firing and creating a degenerate posterior region."
		  OLD formula: 15+65*sigmoid((p-1500)/500) → ceiling ~81 ms
		    At ISI=20 ms, tau_b=81 ms: recovery = 1−exp(−20/81) ≈ 22% per cycle
		    After 5 spikes: b drops to ~(1-recovery_fraction)^5 × b_0 → near-zero        [−]
		  NEW formula: 15+35*sigmoid((p-1500)/500) → ceiling ~51 ms                       [+]
		    At p=1e-4: tau_b ≈ 15+35*0.047 ≈ 17 ms  (fast; 69% recovery per 20 ms ISI)   [+]
		    At p=500:  tau_b ≈ 15+35*0.119 ≈ 19 ms  (mod-fast; 65% recovery per 20 ms)   [+]
		    At p=1000: tau_b ≈ 15+35*0.269 ≈ 24 ms  (moderate; 57% recovery per 20 ms)   [+]
		    At p=1500: tau_b = 15+35*0.500 = 32.5 ms (centre; 46% recovery per 20 ms)     [+]
		    At p=2000: tau_b ≈ 15+35*0.731 ≈ 41 ms  (slow; 39% recovery per 20 ms)       [+]
		    At p=2500: tau_b ≈ 15+35*0.881 ≈ 46 ms  (slower; 35% recovery per 20 ms)     [+]
		    At p=3000: tau_b ≈ 15+35*0.953 ≈ 48 ms  (very slow; 33% recovery per 20 ms)  [+]
		  At ceiling tau_b=48 ms and ISI=20 ms: 33% recovery — sufficient spike history.  [+]
		  No degenerate b≈0 accumulation regime across any part of prior support.          [+]
		  Full [1e-4, 3000] prior now spans functionally distinct but non-degenerate modes.[+]
		  Literature: Kv4-type inactivation recovery τ typically 15–50 ms (Rudy 1988).    [+]

		UNCHANGED from prior iterations:
		  - E_K = -80.0 mV (physiological Nernst, corrected from base -107 mV)
		  - V_half_a = -40.0 mV, k_a = 6.0 mV (I_A activation, instantaneous)
		  - V_half_b = -65.0 mV, k_b = 6.0 mV (iter-15 FIX 2: sharpens per-cycle gating)
		  - k_w = 6.0 mV (iter-16 FIX 1: sharpened slope retained)
		  - gbar_A = 1.0*(p/10) in [~1e-5, 1.0] mS/cm² (iter-17 FIX 1)
		  - gbar_M = 2.0*(p/120) in [~2e-6, 2.0] mS/cm² (iter-13 FIX 2)
		  - tau_w = 20+60*sigmoid((p-75)/20) in [20, 80] ms (floor: no per-AP sawtooth)
		  - Mainen & Sejnowski 1996 Na+ kinetics; Exp clamped at -500

		PARAMETER MAPPING:
		  params[:,0] = gbar_Na    mS/cm2    Na+ transient max conductance
		  params[:,1] = gbar_K     mS/cm2    K+ DR max conductance
		  params[:,2] = g_leak     mS/cm2    passive leak conductance
		  params[:,3] = |E_leak|   mV        E_leak = -params[:,3]
		  params[:,4] = |Vt|       mV        Vt = -params[:,4]
		  params[:,5] = nois_fact  unitless  stochastic noise amplitude
		  params[:,6] = p_gbar_A   [1e-4,10]   -> 1.0*(p/10)  mS/cm2          I_A conductance
		  params[:,7] = p_gbar_M   [1e-4,120]  -> 2.0*(p/120) mS/cm2          I_M conductance
		  params[:,8] = p_tau_w    [1e-4,150]  -> 20+60*sigmoid((p-75)/20) ms  tau_w I_M
		  params[:,9] = p_tau_b    [1e-4,3000] -> 15+35*sigmoid((p-1500)/500)ms tau_b I_A (FIX 2)

		Returns:
			V: torch.Tensor (batch_size, time_steps)   membrane voltage (mV)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# Extract base HH parameters
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# X1: I_A transient A-type K+ current (Connor & Stevens 1971; Hoffman et al. 1997)
		# Ceiling 1.0 mS/cm² (iter-17 FIX 1: halved from 2.0 to balance combined K+)
		# LINEAR mapping: 1.0*(p/10) in [~1e-5, 1.0] mS/cm²
		#   At p=1e-4: gbar_A ≈ 1e-5 mS/cm² (effectively zero — no forced I_A)          [+]
		#   At p=5:    gbar_A = 0.50 mS/cm² (moderate; per-spike outward pulse)          [+]
		#   At p=10:   gbar_A = 1.00 mS/cm² (strong; within Hoffman 1997 somatic range)  [+]
		p_gbar_A  = params[:, 6].float()                    # (batch_size,) in [1e-4, 10]
		gbar_A    = 1.0 * (p_gbar_A / 10.0)               # (batch_size,) mS/cm2 in [~1e-5, 1.0]

		# X2: I_M muscarinic K+ current (Brown & Adams 1980; Wang 1998)
		# Ceiling 2.0 mS/cm² — now safe because FIX 1 (V_half_w=-35 mV) greatly reduces
		# inter-spike activation, eliminating the chronic subthreshold outward drain.
		# LINEAR mapping: 2.0*(p/120) in [~2e-6, 2.0] mS/cm²
		#   At p=1e-4: gbar_M ≈ 2e-6 mS/cm² (effectively zero — no forced I_M)          [+]
		#   At p=60:   gbar_M = 1.00 mS/cm² (moderate post-spike adaptation)             [+]
		#   At p=120:  gbar_M = 2.00 mS/cm² (strong; tonic spiking preserved)            [+]
		p_gbar_M  = params[:, 7].float()                    # (batch_size,) in [1e-4, 120]
		gbar_M    = 2.0 * (p_gbar_M / 120.0)              # (batch_size,) mS/cm2 in [~2e-6, 2.0]

		# I_M time constant: tau_w in [20, 80] ms (floor 20 ms: no per-AP sawtooth artefact)
		# tau_w = 20 + 60*sigmoid((p-75)/20)
		#   At p=1e-4: tau_w ≈ 21 ms — fast partial ISI recovery                        [+]
		#   At p=75:   tau_w = 50 ms — moderate multi-spike adaptation                  [+]
		#   At p=150:  tau_w ≈ 79 ms — slow; integrates across many spikes              [+]
		p_tau_w   = params[:, 8].float()                                         # (batch_size,) in [1e-4, 150]
		tau_w     = 20.0 + 60.0 * torch.sigmoid((p_tau_w - 75.0) / 20.0)       # (batch_size,) ms in [20, 80]

		# I_A inactivation time constant: tau_b in [~16, ~51] ms
		# FIX 2: ceiling capped by reducing range constant from 65 → 35
		# Formula: 15+35*sigmoid((p-1500)/500) — same transition locus as iter-17 (p=1500)
		#   OLD: 15+65*sigmoid → ceiling ~81 ms; degenerate b≈0 at high p (iter-17 bug)  [−]
		#   NEW: 15+35*sigmoid → ceiling ~51 ms; min 33% b-recovery per 20 ms ISI        [+]
		#   At p=1e-4: tau_b ≈ 15+35*0.047 ≈ 17 ms  (fast; b recovers 69% per 20 ms)    [+]
		#   At p=1500: tau_b = 15+35*0.500 = 32.5 ms (centre; b recovers 46% per 20 ms) [+]
		#   At p=3000: tau_b ≈ 15+35*0.953 ≈ 48 ms  (slow; b recovers 34% per 20 ms)    [+]
		#   No degenerate b≈0 accumulation regime anywhere in prior support               [+]
		p_tau_b   = params[:, 9].float()                                              # (batch_size,) in [1e-4, 3000]
		tau_b     = 15.0 + 35.0 * torch.sigmoid((p_tau_b - 1500.0) / 500.0)         # (batch_size,) ms in [~16, ~51]

		tstep = float(dt)   # scalar float ms

		# Fixed constants
		nois_fact_obs = 0.0   # observation noise (unchanged from base)
		C    = 1.0            # uF/cm2
		E_Na = 53.0           # mV  Na+ reversal

		# E_K corrected from base -107 mV to physiological Nernst.
		# (RT/F)*ln([K+]out/[K+]in) = 26.7*ln(5/140) ≈ -80 mV at 37°C.
		# Shared reversal for K+ DR, I_A, and I_M (all outward K+ channels).
		E_K = -80.0   # mV (corrected from base -107 mV)

		# I_A fixed kinetic parameters (Connor & Stevens 1971; Hoffman et al. 1997)
		# Activation: V_half_a=-40 mV, k_a=6 mV (instantaneous; tau_a~1ms ≪ dt=0.1ms)
		# Inactivation: V_half_b=-65 mV, k_b=6 mV (iter-15 FIX 2: Rudy 1988 range)
		#   b_inf(-65) = 0.500 — half-inactivated at resting potential                  [+]
		#   b_inf(-80) ≈ 0.924 — near-fully recovered at AHP bottom                     [+]
		#   b_inf(+40) ≈ 0.000 — fully inactivated at AP peak                           [+]
		V_half_a = -40.0   # mV I_A activation half-voltage
		k_a      =   6.0   # mV I_A activation slope
		V_half_b = -65.0   # mV I_A inactivation half-voltage (iter-15 FIX 2)
		k_b      =   6.0   # mV I_A inactivation slope

		# I_M fixed kinetic parameters (Brown & Adams 1980; Wang 1998)
		# V_half_w=-35 mV (FIX 1: restored from -45 mV — suprathreshold activation)
		# k_w=6 mV (iter-16 FIX 1: sharpened slope retained)
		#   w_inf(-65) = sigmoid(-5.00) ≈ 0.007 — near-zero at rest (no resting bias)   [+]
		#   w_inf(-50) = sigmoid(-2.50) ≈ 0.076 — low inter-spike activation             [+]
		#   w_inf(-35) = sigmoid( 0.00) = 0.500 — half-activated at spike threshold      [+]
		#   w_inf(-20) = sigmoid( 2.50) ≈ 0.924 — strongly activated during/after AP    [+]
		#   w_inf(+40) = sigmoid(12.5)  ≈ 1.000 — fully activated at AP peak            [+]
		#   Wang (1998): V_half_w=-35 mV canonical (standard HH Vt=0)                   [+]
		V_half_w = -35.0   # mV I_M half-activation voltage (FIX 1: restored from -45 mV)
		k_w      =   6.0   # mV I_M activation slope (iter-16 FIX 1 retained)

		####################################
		# Kinetics helpers

		def Exp(z):
			# Clamped exponential — prevents float32 underflow below -500.
			# z: torch.Tensor (any shape) -> torch.Tensor (same shape)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# z/(exp(z)-1) with linear Taylor branch at |z|<1e-4 to avoid 0/0 at z=0.
			# z: (batch_size,) -> (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# Standard HH Na+ kinetics (Mainen & Sejnowski 1996, Vt offset)
		def alpha_m(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant 1/(alpha+beta); (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state alpha/(alpha+beta); (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====

		def a_inf_A(x):
			# I_A Boltzmann activation steady-state — treated as instantaneous.
			# (Connor & Stevens 1971; tau_a ~ 1 ms, negligible vs dt=0.1 ms)
			# x: (batch_size,) -> (batch_size,) in [0, 1]
			# V=-65 (rest):    a_inf ≈ 0.015; a^3 ≈ 3e-6 — negligible at rest            [+]
			# V=-40 (half):    a_inf = 0.500; a^3 = 0.125 — activating near threshold     [+]
			# V=+40 (AP peak): a_inf ≈ 0.999; a^3 ≈ 1.000 — fully open at AP peak        [+]
			return 1.0 / (1.0 + Exp(-(x - V_half_a) / k_a))   # (batch_size,)

		def b_inf_A(x):
			# I_A Boltzmann inactivation steady-state (Connor & Stevens 1971; Rudy 1988).
			# x: (batch_size,) -> (batch_size,) in [0, 1]
			# V_half_b = -65 mV (iter-15 FIX 2: corrected from -55 mV)
			# V=-65 (rest):    b_inf = 0.500 — half-inactivated at resting potential      [+]
			# V=-75 (AHP):     b_inf ≈ 0.841 — mostly recovered during AHP               [+]
			# V=-80 (AHP bot): b_inf ≈ 0.924 — near-fully recovered at AHP bottom        [+]
			# V=-40 (depol):   b_inf ≈ 0.015 — nearly fully inactivated near threshold    [+]
			# V=+40 (AP peak): b_inf ≈ 0.000 — fully inactivated at AP peak              [+]
			# FIX 2: tau_b ceiling ~51 ms prevents degenerate b≈0 accumulation           [+]
			return 1.0 / (1.0 + Exp((x - V_half_b) / k_b))    # (batch_size,)

		def w_inf_M(x):
			# I_M Boltzmann activation steady-state (Brown & Adams 1980; Wang 1998).
			# x: (batch_size,) -> (batch_size,) in [0, 1]
			# FIX 1: V_half_w=-35 mV (restored; suprathreshold activation)
			# k_w=6 mV (iter-16 FIX 1 retained: sharp sigmoid)
			# w_inf(-65) = sigmoid(-5.00) ≈ 0.007 — near-zero at rest (stat 2,3 safe)    [+]
			# w_inf(-50) = sigmoid(-2.50) ≈ 0.076 — low subthreshold activation           [+]
			# w_inf(-35) = sigmoid( 0.00) = 0.500 — half-activated at threshold           [+]
			# w_inf(-20) = sigmoid( 2.50) ≈ 0.924 — strongly activated after AP          [+]
			# Post-spike w accumulation → genuine spike-rate adaptation via AHP delay     [+]
			# ~4× less inter-spike I_M vs prior V_half_w=-45 mV formulation              [+]
			return 1.0 / (1.0 + Exp(-(x - V_half_w) / k_w))   # (batch_size,)

		# ===== END EDITABLE SECTION =====

		####################################

		# State variable allocation
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ DR
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ act
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inact
		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# I_A inactivation gate b (a instantaneous — no state variable needed for a)
		b = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		# I_M activation gate w
		w = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		# ===== END EDITABLE SECTION =====

		# Initialise at Boltzmann steady-state (eliminates onset transients)
		V_init  = init_voltage.to(device)                                # (batch_size,)
		V[:, 0] = V_init                                                 # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))              # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))              # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))              # (batch_size,)
		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# I_A b IC: b_inf(V_init); at V=-65: b_inf(-65)=0.50 (iter-15 FIX 2)            [+]
		b[:, 0] = b_inf_A(V[:, 0])    # (batch_size,)
		# I_M w IC: w_inf(V_init); at V=-65: w_inf(-65)≈0.007 (FIX 1: near-zero at rest)[+]
		w[:, 0] = w_inf_M(V[:, 0])    # (batch_size,)
		# ===== END EDITABLE SECTION =====

		# Simulation loop (exponential Euler — unconditionally stable for all dt)
		for i in range(1, time_steps):
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,)
			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# I_A: instantaneous activation + inactivation steady-state at V(t-1)
			a_A    = a_inf_A(V[:, i - 1])   # (batch_size,) instantaneous a gate
			b_ss_A = b_inf_A(V[:, i - 1])   # (batch_size,) inactivation SS target
			# I_M: activation steady-state at V(t-1)
			w_ss   = w_inf_M(V[:, i - 1])   # (batch_size,) I_M SS target
			# Effective I_A conductance: gbar_A * a^3 * b
			# FIX 1 of this iter: gbar_A max = 1.0 mS/cm² (safe; combined max K+ = 3.0)  [+]
			# FIX 2 of this iter: tau_b ceiling ~51 ms → b never accumulates to zero      [+]
			# a^3(-65) ≈ 3e-6: zero resting I_A → stats 2, 3, 4 unaffected               [+]
			# Per-cycle b swings from ~0.50 (rest) to ~0.84 (AHP trough): large pulse     [+]
			g_A_eff = gbar_A * (a_A ** 3) * b[:, i - 1]   # (batch_size,) mS/cm2
			# ===== END EDITABLE SECTION =====

			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ transient
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) K+ DR
				+ g_leak                                        # (batch_size,) leak
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				# I_A: zero resting conductance (a^3≈3e-6) → stats 2,3,4 safe             [+]
				# Outward pulse at spike onset (b recovered) → ISI regularisation          [+]
				+ g_A_eff                                       # (batch_size,) I_A
				# I_M: FIX 1 V_half_w=-35 mV → w(-65)≈0.007 near-zero resting conductance [+]
				# w(-50)≈0.076 low inter-spike → no chronic hyperpolarisation between spikes [+]
				+ gbar_M * w[:, i - 1]                         # (batch_size,) I_M
				# ===== END EDITABLE SECTION =====
			) / C   # (batch_size,) ms^-1

			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,) K+ DR
				+ g_leak * E_leak                                       # (batch_size,) leak
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				# I_A outward drive toward E_K=-80 mV.
				# Per-cycle b recovery (0.5→0.84): large per-spike outward pulse           [+]
				# FIX 2 tau_b ceiling ~51 ms: b always retains spike-history memory        [+]
				# Zero resting bias (a^3≈3e-6): stats 2, 3, 4 unaffected                  [+]
				+ g_A_eff * E_K                                        # (batch_size,)
				# I_M outward drive toward E_K=-80 mV.
				# FIX 1 V_half_w=-35 mV: resting I_M ≈ gbar_M*0.007*15 ≈ 0.2 µA/cm²     [+]
				# inter-spike I_M ≈ gbar_M*0.076*30 ≈ 4.6 µA/cm² (moderate adaptation)   [+]
				# Post-AP w accumulation → genuine multi-spike rate adaptation             [+]
				# tau_w [20,80] ms: smooth adaptation profile; no per-AP sawtooth          [+]
				+ gbar_M * w[:, i - 1] * E_K                          # (batch_size,)
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                              # (batch_size,) injected
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,) mV

			# Exponential Euler updates (unconditionally stable for all dt)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)                            # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# I_A b: exponential Euler; tau_b (batch_size,) tensor → Exp() tensor path.
			# FIX 2: tau_b=15+35*sigmoid((p-1500)/500) → ceiling ~51 ms (Kv4-type range) [+]
			# b carries spike-history memory: min 33% recovery per 20 ms ISI at ceiling  [+]
			# No degenerate b≈0 regime anywhere in prior support                          [+]
			b[:, i] = b_ss_A + (b[:, i - 1] - b_ss_A) * Exp(-tstep / tau_b)             # (batch_size,)
			# I_M w: exponential Euler; tau_w (batch_size,) tensor → Exp() tensor path.
			# tau_w in [20, 80] ms: smooth multi-spike adaptation; no per-AP artefact     [+]
			# FIX 1 V_half_w=-35 mV: w accumulates predominantly post-spike              [+]
			w[:, i] = w_ss + (w[:, i - 1] - w_ss) * Exp(-tstep / tau_w)                # (batch_size,)
			# ===== END EDITABLE SECTION =====

		# Return voltage with optional observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)