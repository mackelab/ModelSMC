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
		Hodgkin-Huxley neuron extended with two additional currents:
		  (X1) Persistent Na+ current (INaP): gbar_NaP = params[:,6], V_half_NaP from params[:,8]
		  (X2) M-type K+ current (IKM):       gbar_KM  = params[:,7], V_half_KM  from params[:,9]

		ITER 38 DESIGN (two targeted fixes from iter 37 feedback):

		FIX 1 — Remove INaP q slow-inactivation; revert to quasi-instantaneous activation-only:
		  Problem in iter 37: tau_q=200ms inactivation caused chronic q-depletion during tonic firing.
		  Each AP drives q → q_inf(+30)≈0; slow 200ms recovery cannot keep up with ISI duration.
		  After 3-5 spikes, q stabilizes near 0.05-0.15, making I_NaP ≈ 0 throughout stimulation.
		  This wasted the X1 slot and corrupted metrics 4-7 (stim period distribution statistics).
		  Fix: I_NaP = gbar_NaP * p_inf(V) * (V - E_Na); p_inf quasi-instantaneous; no q ODE.
		  With V_half_NaP in [-48,-40] mV: p_inf(-65,V_half=-48)≈0.023 [2.3%; negligible at rest].
		  p_inf activates near threshold [-48,-40 mV], providing persistent inward current that
		  modestly amplifies spike rate and shapes inter-spike subthreshold voltage distribution.
		  No ISI-accumulation problem (no inactivation state), and resting I_NaP remains negligible.

		FIX 2 — IKM: tau_z 75ms → 500ms; V_half_KM [-35,-15] → [-50,-40] mV:
		  Problem A (iter 37): tau_z=75ms with V_half in [-35,-15] mV fully deactivated IKM each ISI.
		    At V_half=-35: z_inf(-65)≈0.047 [<5%]; with tau_z=75ms each ~2ms AP only builds
		    z by ~(1-exp(-2/75))≈2.6% → z then decays completely before next spike → zero adaptation.
		  Problem B (iter 35-36): V_half in [-65,-45] mV with tau_z=75ms let z equilibrate to
		    z_inf(V_rest)≈0.5 during 300ms pre-stim rest (300ms >> 75ms), corrupting resting metrics.
		  Combined solution — longer tau + intermediate V_half:
		    tau_z = 500ms >> typical ISI (20-100ms): z accumulates across many spikes ✓
		    tau_z = 500ms > pre-stim rest (300ms): z doesn't fully equilibrate at rest ✓
		    V_half_KM = -50 + params[:,9]*(10/3000); range [-50,-40] mV
		    At lower bound V_half=-50, slope=10:
		      z_inf(-65)=1/(1+exp(-1.5))≈0.182 [18% steady-state at rest]
		      After 300ms rest: z≈0.182*(1-exp(-300/500))≈0.182*0.451≈0.082 [8%; small] ✓
		      z_inf(+30)=1/(1+exp(-8))≈0.9997 [~100% during AP peak] ✓
		      z_inf(-55)=1/(1+exp(-0.5))≈0.378 [38% during sub-threshold depol]
		      z accumulates during repetitive firing (500ms >> 20-100ms ISI) → genuine adaptation ✓
		    At upper bound V_half=-40, slope=10:
		      z_inf(-65)=1/(1+exp(-2.5))≈0.076 [7.6% steady-state at rest]
		      After 300ms rest: z≈0.076*0.451≈0.034 [3.4%; very small] ✓
		      z_inf(+30)=1/(1+exp(-7))≈0.9991 [~100% during AP] ✓
		    Both bounds: negligible resting I_KM (z<9%), strong spike-period activation (z>99%) ✓
		    Net: IKM provides genuine inter-spike accumulating adaptation → regulates spike count ✓

		PARAMETER ASSIGNMENTS (iter 38):
		  params[:,0]: gbar_Na [mS/cm2]        transient Na+ (HH standard)
		  params[:,1]: gbar_K  [mS/cm2]        delayed-rectifier K+ (HH standard)
		  params[:,2]: g_leak  [mS/cm2]        passive leak (HH standard)
		  params[:,3]: |E_leak| [mV]           leak reversal; negated internally
		  params[:,4]: |Vt| [mV]               voltage threshold shift; negated internally
		  params[:,5]: nois_fact               current noise amplitude
		  params[:,6]: gbar_NaP → *0.12/10    → [0, 0.12] mS/cm2; prior [1e-4, 10]    [X1]
		  params[:,7]: gbar_KM  → *0.5/120    → [0, 0.5] mS/cm2;  prior [1e-4, 120]   [X2]
		  params[:,8]: V_half_NaP → -48+x*(8/150)  → [-48,-40] mV; prior [1e-4, 150]  [X1 flex]
		  params[:,9]: V_half_KM  → -50+x*(10/3000) → [-50,-40] mV; prior [1e-4, 3000] [X2 flex REDESIGNED]

		ITERATION HISTORY SUMMARY:
		  Iter  3: E_K -107 → -77 mV (30 mV excess AHP depth in base model)
		  Iter 23: INaP added; quasi-instantaneous; V_half tuning across iters
		  Iter 34: IKM added (tau_z=75ms, V_half [-65,-45]); z equilibrates at rest → bad
		  Iter 35: Exp(float scalar) crash fixed via precomputed decay_z scalar
		  Iter 36: z[:,0]=0; V_half_NaP [-52,-44]→[-48,-40]; z still equilibrates at rest (75ms<<300ms)
		  Iter 37: V_half_KM [-65,-45]→[-35,-15]; tau_z=75ms; q-inactivation on INaP added
		           PROBLEMS: IKM fully deactivates each ISI (z_inf<5%; tau<ISI); q-inactivation
		           depletes chronically during tonic firing; both channels near-zero during stim
		  Iter 38 (THIS):
		    (a) INaP: remove q-inactivation (chronic depletion problem); revert to p_inf only ✓
		    (b) IKM: tau_z 75→500ms + V_half [-35,-15]→[-50,-40] mV (accumulates across spikes) ✓

		Args:
			init_voltage: torch.Tensor: (batch_size,) initial membrane voltage in mV
			input_current: torch.Tensor: (batch_size, time_steps) injected current uA/cm2
			dt: float time step in ms
			t: torch.Tensor: (time_steps,) time array in ms
			params: torch.Tensor: (batch_size, 10) biophysical parameters
			seed: optional int for reproducibility

		Returns:
			V: torch.Tensor: (batch_size, time_steps) membrane voltage in mV
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

		# ---- Standard HH parameters -------------------------------------------------
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2 -- transient Na+
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2 -- delayed-rectifier K+
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2 -- passive leak
		E_leak    = -params[:, 3].float()  # (batch_size,) mV -- leak reversal (negated from positive prior)
		Vt        = -params[:, 4].float()  # (batch_size,) mV -- voltage threshold shift (negated)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless -- current noise amplitude

		# ---- X1: INaP conductance ---------------------------------------------------
		# gbar_NaP = params[:,6] * (0.12/10); prior [1e-4, 10] → range [0, 0.12] mS/cm2
		# Literature: INaP gbar in cortical neurons ~0.01-0.1 mS/cm2 (Magistretti & Alonso 1999)
		gbar_NaP = params[:, 6].float() * (0.12 / 10.0)   # (batch_size,) mS/cm2; [0, 0.12]

		# ---- X2: IKM conductance ----------------------------------------------------
		# gbar_KM = params[:,7] * (0.5/120); prior [1e-4, 120] → range [0, 0.5] mS/cm2
		# Literature: IKM gbar in cortical neurons ~0.01-0.3 mS/cm2 (Wang et al. 1998)
		gbar_KM = params[:, 7].float() * (0.5 / 120.0)    # (batch_size,) mS/cm2; [0, 0.5]

		# ---- X1: INaP half-activation voltage ---------------------------------------
		# V_half_NaP = -48 + params[:,8] * (8/150); prior [1e-4, 150] → range [-48, -40] mV
		# At lower bound V_half=-48, slope=5 mV:
		#   p_inf(-65) = 1/(1+exp(17/5)) ≈ 0.023 [2.3% at rest; negligible resting I_NaP] ✓
		#   p_inf(-48) = 0.500 [50% at half-activation; threshold-zone onset]
		#   p_inf(-40) ≈ 0.880 [88% near peak; strong threshold amplification] ✓
		# At upper bound V_half=-40, slope=5 mV:
		#   p_inf(-65) ≈ 0.007 [0.7% at rest; essentially silent] ✓
		#   p_inf(-40) = 0.500 [50% at half-activation; shifted activation zone] ✓
		# ITER 38: NO q slow-inactivation (removed from iter 37).
		# Rationale: tau_q=200ms caused chronic q-depletion during tonic firing:
		#   each AP drives q → q_inf(+30)≈0; slow 200ms recovery cannot keep up with ISI.
		#   After 3-5 spikes q stabilizes near 0.05-0.15 → I_NaP≈0 throughout stim period.
		#   Without q: I_NaP provides persistent subthreshold-to-threshold inward current,
		#   modestly boosting spike rate and shaping inter-spike voltage distribution ✓
		V_half_NaP = -48.0 + params[:, 8].float() * (8.0 / 150.0)   # (batch_size,) mV; [-48, -40]
		slope_NaP  = 5.0   # mV; fixed (Magistretti & Alonso 1999; physiological)

		# ---- X2: IKM half-activation voltage ----------------------------------------
		# ITER 38 REDESIGN: V_half_KM = -50 + params[:,9] * (10/3000); range [-50, -40] mV
		# Previous iter 37: V_half in [-35,-15] mV with tau_z=75ms → IKM fully deactivated
		#   each ISI; z_inf(-65)<5% AND tau_z=75ms≈ISI → zero net spike-frequency adaptation.
		# Previous iter 34-36: V_half in [-65,-45] mV → z_inf(V_rest)=0.5 at lower bound →
		#   z equilibrated to ≈0.49 during 300ms pre-stim rest → corrupted resting potential.
		# New design at lower bound V_half=-50, slope=10:
		#   z_inf(-65) = 1/(1+exp(-1.5)) ≈ 0.182 [18% steady-state; but tau_z=500ms]
		#   After 300ms pre-stim: z ≈ 0.182*(1-exp(-300/500)) ≈ 0.082 [8%; acceptable] ✓
		#   z_inf(+30) = 1/(1+exp(-8)) ≈ 0.9997 [~100% during AP; accumulates across spikes]
		#   tau_z=500ms >> typical ISI (20-100ms) → z builds progressively across spikes ✓
		# New design at upper bound V_half=-40, slope=10:
		#   z_inf(-65) ≈ 0.076 [7.6%]; after 300ms rest: ≈0.034 [3.4%; very small] ✓
		#   z_inf(+30) ≈ 0.999 [~100% during AP] ✓
		V_half_KM  = -50.0 + params[:, 9].float() * (10.0 / 3000.0)  # (batch_size,) mV; [-50, -40]
		slope_KM   = 10.0   # mV; standard M-current Boltzmann slope (fixed; Brown & Adams 1980)

		# ---- IKM time constant: INCREASED from 75ms to 500ms -----------------------
		# CRITICAL CHANGE (iter 38): tau_z=75ms << typical ISI → z reset each cycle → no adaptation.
		# tau_z=500ms >> typical ISI (20-100ms) → z accumulates across many spikes ✓
		# tau_z=500ms > pre-stim rest (300ms) → z doesn't fully equilibrate at rest ✓
		# Literature: M-current tau in cortical neurons 100-1000ms (Wang et al. 1998) ✓
		tau_z = 500.0   # ms; slow adaptation; accumulates across ISIs (fixed; physiological)

		tstep = float(dt)   # ms

		# ---- Fixed biophysical constants --------------------------------------------
		nois_fact_obs = 0.0   # observation noise disabled per task specification
		C    = 1.0            # uF/cm2 -- membrane capacitance (standard HH)
		E_Na = 53.0           # mV -- Na+ reversal (shared INa and INaP: same ion) ✓
		# ITER 3 FIX: E_K corrected from base model -107 mV → -77 mV
		# Base model -107 mV gave ~30 mV excess AHP depth (physiologically unrealistic).
		# Standard mammalian cortical neuron E_K ≈ -77 mV (Hille 2001) ✓
		E_K  = -77.0          # mV -- K+ reversal (IK_DR and IKM share this; both carry K+) ✓

		# ---- Precompute IKM M-gate decay factor (loop-invariant scalar) -------------
		# ITER 35 FIX (retained): Exp() helper requires Tensor input (uses torch.full_like).
		# Passing a Python float scalar to Exp() causes TypeError.
		# Solution: precompute decay_z as Python float using torch.tensor → float conversion.
		# This also avoids redundant per-step computation (tau_z and dt are loop-invariant).
		# decay_z = exp(-dt/500); for dt=0.01ms: decay_z ≈ 0.999980 [very close to 1; stable] ✓
		decay_z = float(torch.exp(torch.tensor(-tstep / tau_z, dtype=torch.float32)))   # Python float scalar

		# ---- Numerical helpers ------------------------------------------------------
		def Exp(z):
			# Numerically stable exp for Tensor inputs ONLY (do NOT pass Python float scalars)
			# z: Tensor (any shape) → Tensor (same shape); floor at -500 prevents underflow
			# torch.full_like(z, -5e2) requires z to be a Tensor (not float) ← important ✓
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# HH auxiliary function: z/(exp(z)-1); Taylor expansion near z=0 avoids 0/0
			# z: Tensor → Tensor; numerically stable for all physiological voltage ranges
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ---- Standard HH gating kinetics (unchanged from base) ---------------------
		def alpha_m(x):
			# Na+ activation forward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation backward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation forward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation backward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ DR activation forward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ DR activation backward rate; x: (batch_size,) mV → (batch_size,) ms^-1
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gating time constant; alpha, beta: (batch_size,) → (batch_size,) ms
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gating steady state; alpha, beta: (batch_size,) → (batch_size,) [0,1]
			return alpha / (alpha + beta)

		# ---- INaP quasi-instantaneous activation Boltzmann (INCREASING in V) -------
		def p_inf_NaP(x):
			# INaP activation steady-state; INCREASING in V (activates on depolarization) ✓
			# x: (batch_size,) mV; V_half_NaP: (batch_size,) mV → element-wise broadcast ✓
			# Quasi-instantaneous: tau_p << dt at all physiological voltages → no ODE state ✓
			# ITER 38: NO q slow-inactivation (removed); p_inf is the sole INaP gating variable.
			# This avoids chronic q-depletion during tonic firing that made INaP≈0 in iter 37 ✓
			# V_half_NaP in [-48,-40] mV: p_inf(V_rest=-65) ≤ 0.023 [negligible at rest] ✓
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / slope_NaP))   # (batch_size,) [0,1]

		# ---- IKM Boltzmann steady state (INCREASING in V) --------------------------
		def z_inf_KM(x):
			# IKM M-gate steady state; INCREASING in V (activates on depolarization) ✓
			# x: (batch_size,) mV; V_half_KM: (batch_size,) mV → element-wise broadcast ✓
			# ITER 38: V_half_KM in [-50,-40] mV (redesigned from [-35,-15] in iter 37)
			# With tau_z=500ms: z accumulates across spikes; provides genuine adaptation ✓
			# z_inf(V_rest=-65) ≤ 0.182 but equilibrates to only ≤0.082 in 300ms pre-stim ✓
			return 1.0 / (1.0 + Exp(-(x - V_half_KM) / slope_KM))   # (batch_size,) [0,1]

		# ---- Allocate state variable arrays ----------------------------------------
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) mV -- membrane voltage
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) K+ DR gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) Na+ activation gate
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) Na+ inactivation gate
		z = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) IKM M-gate ODE state
		# INaP: quasi-instantaneous activation → p_inf computed inline each step; no state array ✓
		# INaP: no q inactivation (iter 38) → no q state array needed ✓

		# ---- Initialize state variables at steady state ----------------------------
		V_init  = init_voltage.to(device)                                       # (batch_size,) mV
		V[:, 0] = V_init                                                        # (batch_size,) mV
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                     # (batch_size,) K+ DR SS
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                     # (batch_size,) Na+ act SS
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                     # (batch_size,) Na+ inact SS
		# IKM M-gate: initialize to ZERO (iter 36 fix; retained)
		# Rationale: IKM accumulates only after sustained depolarization.
		# A quiescent neuron at rest has z≈0. With tau_z=500ms (iter 38), even with
		# V_half=-50 mV: z_rest after 300ms = 0.082 << 1; initializing to 0 is appropriate.
		# The pre-stimulus rest period smoothly brings z to its correct resting level ✓
		z[:, 0] = torch.zeros(batch_size, device=device)                        # (batch_size,) zero init ✓

		# ---- Simulation loop: exponential Euler integration ------------------------
		for i in range(1, time_steps):
			# Standard HH gating rates at previous voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,) ms^-1 each
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,) ms^-1 each
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,) ms^-1 each

			# INaP: quasi-instantaneous activation (no ODE state; no ISI accumulation)
			# p_ss: (batch_size,) [0,1]; V_half_NaP broadcast element-wise across batch ✓
			# ITER 38: no q multiplication (inactivation removed); p_ss is full INaP gate ✓
			p_ss = p_inf_NaP(V[:, i - 1])   # (batch_size,) INaP activation; INCREASING in V

			# IKM: M-gate steady-state target for exponential Euler (z carries ODE memory)
			# z_ss: (batch_size,) [0,1]; V_half_KM broadcast element-wise across batch ✓
			z_ss = z_inf_KM(V[:, i - 1])    # (batch_size,) M-gate SS target; INCREASING in V

			# ---- Effective conductance sum: tau_V_inv = sum(g_i * gate_i) / C ------
			# (batch_size,) ms^-1; all terms non-negative ✓
			# INaP: gated by p_ss (quasi-instantaneous; no memory) ✓
			# IKM: gated by z[:,i-1] (ODE state; slow 500ms memory across spikes) ✓
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) transient Na+
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) delayed-rectifier K+
				+ g_leak                                       # (batch_size,) passive leak
				+ gbar_NaP * p_ss                              # (batch_size,) INaP [quasi-inst.; no q]
				+ gbar_KM * z[:, i - 1]                        # (batch_size,) IKM [ODE z; 500ms memory]
			) / C   # (batch_size,) ms^-1

			# ---- Effective voltage steady state: V_inf = I_sum / (tau_V_inv * C) ---
			# (batch_size,) mV
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na    # (batch_size,) Na+ → +53 mV
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ DR → -77 mV
				+ g_leak * E_leak                                       # (batch_size,) leak → E_leak
				+ gbar_NaP * p_ss * E_Na                                # (batch_size,) INaP → +53 mV [inward] ✓
				+ gbar_KM * z[:, i - 1] * E_K                          # (batch_size,) IKM → -77 mV [outward] ✓
				+ input_current[:, i - 1]                              # (batch_size,) external current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,) mV

			# ---- Exponential Euler state updates ------------------------------------
			# Membrane voltage: exponential relaxation toward V_inf on timescale 1/tau_V_inv
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)                                  # (batch_size,)
			# K+ DR gate: exponential Euler toward Boltzmann steady state (Tensor arg ✓)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))      # (batch_size,)
			# Na+ activation: exponential Euler toward Boltzmann steady state (Tensor arg ✓)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))      # (batch_size,)
			# Na+ inactivation: exponential Euler toward Boltzmann steady state (Tensor arg ✓)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))      # (batch_size,)
			# IKM M-gate: exponential Euler toward z_ss with tau_z=500ms
			# Uses precomputed scalar decay_z (Python float) to avoid Exp(float) crash ✓
			# decay_z = exp(-dt/500); iter 35 fix pattern retained; tau_z now 500ms (iter 38) ✓
			# With tau_z=500ms >> ISI: z accumulates across many spikes → genuine adaptation ✓
			# With tau_z=500ms > 300ms rest: z doesn't fully equilibrate at rest ✓
			z[:, i] = z_ss + (z[:, i - 1] - z_ss) * decay_z                                                    # (batch_size,)
			# INaP: quasi-instantaneous activation → NO ODE update needed; p_inf recomputed fresh ✓

		# Return voltage traces (observation noise disabled: nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)