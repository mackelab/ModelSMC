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
		Hodgkin-Huxley neuron extended with TWO additional channels:

		X1 = M-type slow K+ current (Kv7/KCNQ): spike-frequency adaptation
		     First-order Boltzmann gating ODE with FLAT per-batch scalar tau_p.
		     This is the ONLY tau_p formulation validated to produce NLE < 25.0.

		     IRONCLAD tau_p prohibitions (all caused confirmed regressions):
		       NO cosh voltage-dependent tau_p      (iter-103: NLE 27.2)
		       NO V>0 conditional tau_fast=20ms     (iter-104: NLE 27.4)
		       NO tau_p upper bound < 300ms          (iter-101: NLE 25.6)

		  params[:,6] = gbar_M    mS/cm2   DIRECT — NO internal clamp
		  params[:,8] = tau_p_raw [1e-4,150]  -> tau_p_base [100, 300] ms  FLAT SCALAR
		  V_half_M  = -45.0 mV  LOCKED (global minimum confirmed iter-41)
		  k_p       = 10.0 mV   LOCKED (k=10 beats k=8)
		  E_K_M     = -90.0 mV  LOCKED (tunable caused NLE 29.1 at iter-35)

		X2 = Persistent Na+ current (INaP): subthreshold depolarising drive
		     Instantaneous Boltzmann activation; no slow inactivation ODE.
		     Cannot structurally produce burst dynamics.

		     ITER-106 CRITICAL REVERT: V_half_NaP restored from [-60,-50]mV
		     (iter-105: NLE 26.6 massive regression) to [-62,-48]mV (iter-90
		     validated: NLE 24.5 BEST; iter-100: NLE 24.8).

		     Physiological basis for [-62,-48]mV being the optimal range:
		       At typical resting potential (-70mV):
		         V_half=-62mV: activation = 1/(1+exp((-70+62)/5)) ≈ 19.8%  (strong drive)
		         V_half=-55mV: activation = 1/(1+exp((-70+55)/5)) ≈  4.7%  (moderate)
		         V_half=-48mV: activation = 1/(1+exp((-70+48)/5)) ≈  1.1%  (weak drive)
		       SBI posterior concentrates near -62mV where INaP provides meaningful
		       subthreshold depolarising current that helps explain tonic spiking
		       characteristics (mean resting V, V statistics during stimulation).
		       Iter-105 raised lower bound to -60mV, excluding the 19.8% region →
		       SBI forced into suboptimal posterior, causing NLE 26.6 regression.

		     IRONCLAD V_half_NaP prohibitions (caused confirmed regressions):
		       NO lower bound > -62 mV            (iter-105: NLE 26.6)
		       NO V_half_NaP in [-55, -40] mV     (iter-99:  NLE 25.7)
		       NO V_half_NaP < -65 mV             (iter-91:  NLE 25.4)

		  params[:,7] = gbar_NaP_raw [1e-4,120]  -> [1e-4, 1.5] mS/cm2
		  params[:,9] = V_half_NaP_raw [1e-4,3000] -> [-62, -48] mV  REVERTED
		  k_NaP = 5.0 mV  LOCKED (k=6.5 caused NLE 28.2 at iter-63)

		IRONCLAD RULES — all violations produced confirmed NLE regressions:
		  NO nan_to_num                           (iter-71: NLE 27.1)
		  NO gate clamps                          (iter-71: NLE 27.1)
		  NO V clamps                             (iter-70: NLE 25.8)
		  NO Ih (HCN)                            (iter-73: NLE 27.4)
		  NO gbar_M internal clamp               (iter-93: regression)
		  NO spurious randn before return        (iter-92: NLE 26.7)
		  NO tau_V_inv floor                     (iter-97: NLE 25.6)
		  NO EM additive noise outside V_inf     (iter-95: amplitude collapse)
		  NO input_current[:,i]                  (iter-96: NLE 26.9)
		  NO INaP ceiling > 1.5 mS/cm2          (iters 79/86: regressions)
		  NO V_half_NaP lower bound > -62 mV    (iter-105: NLE 26.6)
		  NO V_half_NaP in [-55,-40] mV         (iter-99: NLE 25.7)
		  NO V_half_NaP < -65 mV               (iter-91: NLE 25.4)
		  NO tau_p upper bound < 300 ms         (iter-101: NLE 25.6)
		  NO cosh voltage-dependent tau_p       (iter-103: NLE 27.2)
		  NO V>0 conditional tau_fast           (iter-104: NLE 27.4)

		ITERATION HISTORY (NLE):
		  Iter 90:  M+INaP; V_half[-62,-48]; gbar_M DIRECT; tau_p flat [100,300]ms -> NLE 24.5  BEST
		  Iter 91:  V_half_NaP shifted to [-65,-48]                                -> NLE 25.4  regression
		  Iter 99:  V_half_NaP shifted to [-55,-40]                                -> NLE 25.7  regression
		  Iter100:  V_half_NaP REVERTED to [-62,-48]                               -> NLE 24.8
		  Iter101:  tau_p tightened [100,200]ms                                    -> NLE 25.6  regression
		  Iter102:  tau_p REVERTED [100,300]ms flat scalar                         -> NLE 25.8
		  Iter103:  tau_p cosh voltage-dependent                                   -> NLE 27.2  major regression
		  Iter104:  cosh REVERTED + V>0 conditional tau_fast=20ms                 -> NLE 27.4  regression
		  Iter105:  V>0 conditional REVERTED + V_half_NaP narrowed [-60,-50]      -> NLE 26.6  regression
		  Iter106:  V_half_NaP REVERTED to [-62,-48]                              -> TARGET: NLE ≤ 24.8

		PARAMETER LAYOUT (batch_size, 10):
		  params[:,0]  gbar_Na        mS/cm2   Na+ transient                  TUNABLE
		  params[:,1]  gbar_K         mS/cm2   K+ delayed-rectifier           TUNABLE
		  params[:,2]  g_leak         mS/cm2   passive leak                   TUNABLE
		  params[:,3]  |E_leak|       mV       leak reversal (negated)        TUNABLE
		  params[:,4]  |Vt|           mV       threshold shift (negated)      TUNABLE
		  params[:,5]  nois_fact      unitless  noise amplitude                TUNABLE
		  params[:,6]  gbar_M         mS/cm2   M-current X1                   DIRECT (no clamp)
		  params[:,7]  gbar_NaP_raw   [1e-4,120]  -> [1e-4,1.5] mS/cm2       INaP X2
		  params[:,8]  tau_p_raw      [1e-4,150]  -> tau_p_base [100,300]ms   M-gate flat tau
		  params[:,9]  V_half_NaP_raw [1e-4,3000] -> [-62,-48] mV             INaP V_half (REVERTED)
		"""
		device = params.device

		# =====================================================================
		# Random generator setup
		# =====================================================================
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int
		tstep      = float(dt)         # float (ms)

		# =====================================================================
		# Parameter extraction
		# =====================================================================
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm2  Na+ transient
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm2  K+ delayed-rectifier
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm2  passive leak
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (|E_leak| raw, negated)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (|Vt| raw, negated)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless noise amplitude

		# X1: M-type slow K+ current conductance — DIRECT, NO INTERNAL CLAMP
		# Internal clamp creates degenerate many-to-one likelihood at boundary,
		# collapsing SBI posterior density and corrupting inference (iter-93) ✓
		gbar_M = params[:, 6].float()   # (batch_size,)  mS/cm2  DIRECT

		# X2: INaP conductance — ceiling 1.5 mS/cm2
		# raw params[:,7] in [1e-4, 120] -> gbar_NaP in [1e-4, 1.5] mS/cm2
		# Ceiling 1.5 validated across iters 85-106; ceiling 2.0 failed iters 79/86 ✓
		gbar_NaP = (
			1e-4 + (1.5 - 1e-4) * torch.clamp(params[:, 7].float(), min=0.0, max=120.0) / 120.0
		)   # (batch_size,)  mS/cm2 in [1e-4, 1.5]

		# X1: M-gate FLAT per-batch time constant — [100, 300] ms
		# raw params[:,8] in [1e-4, 150] -> tau_p_base in [100, 300] ms
		# IRONCLAD upper bound: >= 300ms (iter-101: 200ms -> NLE 25.6 regression)
		# IRONCLAD formulation: FLAT SCALAR ONLY (iter-103 cosh: 27.2; iter-104 V>0: 27.4)
		tau_p_base = (
			100.0 + torch.clamp(params[:, 8].float(), min=0.0, max=150.0) / 150.0 * 200.0
		)   # (batch_size,)  ms in [100, 300]

		# X2: INaP half-activation voltage — [-62, -48] mV  REVERTED (iter-106)
		# raw params[:,9] in [1e-4, 3000] -> V_half_NaP in [-62, -48] mV
		#
		# CRITICAL REVERT from iter-105 [-60,-50]mV which caused NLE 26.6.
		# The [-62,-48]mV range with 14mV width has produced the BEST results:
		#   iter-90: NLE 24.5 (BEST EVER)
		#   iter-100: NLE 24.8 (second best)
		# The lower bound -62mV is essential because at -70mV resting potential,
		# V_half=-62mV gives ~19.8% INaP activation, providing the subthreshold
		# depolarising drive that SBI posterior concentrates on. Raising the lower
		# bound to -60mV (iter-105) cut off this critical region, causing NLE 26.6.
		#
		# Mapping verification:
		#   raw=0    -> -62.0 mV  (INaP activation at -70mV: ~19.8%)  <- posterior peak
		#   raw=1500 -> -55.0 mV  (INaP activation at -70mV: ~4.7%)
		#   raw=3000 -> -48.0 mV  (INaP activation at -70mV: ~1.1%)
		V_half_NaP = (
			-62.0 + torch.clamp(params[:, 9].float(), min=0.0, max=3000.0) / 3000.0 * 14.0
		)   # (batch_size,)  mV in [-62, -48]

		# =====================================================================
		# Fixed constants (all locked; deviations caused confirmed regressions)
		# =====================================================================
		nois_fact_obs = 0.0   # observation noise disabled (kept at base simulator value)
		C    = 1.0            # uF/cm2  membrane capacitance
		E_Na = 53.0           # mV  Na+ reversal (transient Na+ and INaP share reversal)
		E_K  = -107.0         # mV  LOCKED (E_K=-90mV caused NLE 25.5 at iter-67)

		# M-current fixed constants
		V_half_M = -45.0   # mV  LOCKED (global minimum confirmed iter-41)
		k_p      = 10.0    # mV  LOCKED (k=10 > k=8 in validation)
		E_K_M    = -90.0   # mV  LOCKED (tunable E_K_M caused NLE 29.1 at iter-35)

		# INaP Boltzmann slope (locked since iter-57)
		k_NaP = 5.0   # mV  LOCKED (k=6.5 caused NLE 28.2 at iter-63)

		# =====================================================================
		# Numerical helpers — CLEAN: no nan_to_num (violation caused NLE 27.1)
		# =====================================================================

		def Exp(z):
			# z: torch.Tensor (any shape) -> torch.Tensor (same shape)
			# Float32 overflow guard: evaluate exp(-500) for extreme negative arguments
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z),
			)

		def efun(z):
			# z: torch.Tensor (any shape) -> torch.Tensor (same shape)
			# L'Hopital regularisation for |z| < 1e-4: avoids 0/0 in HH rate expressions
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# =====================================================================
		# Standard HH channel kinetics (exact base simulator formulation)
		# =====================================================================

		def alpha_m(x):
			# x: (batch_size,) -> (batch_size,)  Na+ m-gate forward rate [ms-1]
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			# x: (batch_size,) -> (batch_size,)  Na+ m-gate backward rate [ms-1]
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,)

		def alpha_h(x):
			# x: (batch_size,) -> (batch_size,)  Na+ h-gate forward rate [ms-1]
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,)

		def beta_h(x):
			# x: (batch_size,) -> (batch_size,)  Na+ h-gate backward rate [ms-1]
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,)

		def alpha_n(x):
			# x: (batch_size,) -> (batch_size,)  K+ DR n-gate forward rate [ms-1]
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# x: (batch_size,) -> (batch_size,)  K+ DR n-gate backward rate [ms-1]
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)  gate time constant [ms]
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)  gate Boltzmann SS in (0,1)
			return alpha / (alpha + beta)   # (batch_size,)

		# =====================================================================
		# X1: M-gate Boltzmann steady state
		# V_half_M=-45mV, k_p=10mV both LOCKED:
		#   p_inf(-70mV) ≈ 0.076  (small at rest — no tonic blockade of spiking) ✓
		#   p_inf(-45mV) = 0.500  (half-activation at AP threshold) ✓
		#   p_inf(-20mV) ≈ 0.953  (near-saturated at depolarised AP peak) ✓
		# =====================================================================
		def p_inf(x):
			# x: (batch_size,) -> (batch_size,)  M-gate Boltzmann SS in (0,1)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_p))   # (batch_size,)

		# =====================================================================
		# X2: INaP instantaneous Boltzmann activation
		# No slow inactivation ODE → structurally cannot produce bursting ✓
		# V_half_NaP in [-62,-48]mV per batch; k_NaP=5.0mV scalar broadcasts ✓
		# =====================================================================
		def m_NaP_inf(x):
			# x: (batch_size,) -> (batch_size,)  INaP activation in (0,1)
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / k_NaP))   # (batch_size,)

		# =====================================================================
		# State variable allocation
		# =====================================================================
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ DR gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ m-gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ h-gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-gate (X1)

		# =====================================================================
		# Initialisation at Boltzmann steady states for init_voltage
		# =====================================================================
		V[:, 0] = init_voltage.to(device)                                   # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                 # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                 # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                 # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])   # (batch_size,)  M-gate Boltzmann SS at init voltage

		# =====================================================================
		# Precompute FLAT per-batch M-gate exponential decay factor OUTSIDE loop.
		# This is the ONLY validated tau_p formulation (iter-90 NLE 24.5 BEST).
		# decay_p is constant across ALL timesteps for each batch element.
		# IRONCLAD: NO voltage-dependent modification of decay_p ever again.
		# =====================================================================
		decay_p = torch.exp(-tstep / tau_p_base)   # (batch_size,)  flat per-batch decay in (0,1)

		# =====================================================================
		# Time integration: exponential Euler
		#
		# NOISE MODEL (exact base simulator — validated at NLE 24.5):
		#   nois_fact * randn(batch_size) / sqrt(dt) inside V_inf numerator,
		#   divided by tau_V_inv * C — heteroscedastic conductance-weighted scaling ✓
		#
		# STIMULUS: input_current[:,i-1]
		#   Self-consistent: all gate quantities evaluated at step i-1.
		#   input_current[:,i] confirmed detrimental at iter-96 (NLE 26.9) ✓
		#
		# NO tau_V_inv floor (iter-97: floor dominated resting conductance, NLE 25.6) ✓
		# NO spurious randn before return (iter-92: NLE 26.7 bug) ✓
		# =====================================================================

		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Kinetic rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-gate (X1): Boltzmann SS at previous voltage
			p_ss = p_inf(V_prev)   # (batch_size,)  M-gate SS in (0,1)

			# INaP (X2): instantaneous Boltzmann activation at previous voltage
			m_NaP = m_NaP_inf(V_prev)   # (batch_size,)  INaP Boltzmann activation in (0,1)

			# Effective conductances from gate states at step i-1
			g_Na_eff  = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  mS/cm2
			g_K_eff   = (n[:, i - 1] ** 4) * gbar_K                   # (batch_size,)  mS/cm2
			g_M_eff   = p[:, i - 1] * gbar_M                          # (batch_size,)  mS/cm2
			g_NaP_eff = m_NaP * gbar_NaP                               # (batch_size,)  mS/cm2

			# Inverse membrane time constant: total conductance / C
			# NO floor — floor dominated resting conductance at iter-97 (NLE 25.6) ✓
			tau_V_inv = (
				g_Na_eff + g_K_eff + g_M_eff + g_NaP_eff + g_leak
			) / C   # (batch_size,)  ms-1

			# Voltage steady state — exact base simulator noise formulation ✓
			V_inf = (
				g_Na_eff  * E_Na              # (batch_size,)  transient Na+ inward drive
				+ g_K_eff * E_K               # (batch_size,)  K+ DR repolarising drive
				+ g_M_eff * E_K_M             # (batch_size,)  M-current slow AHP drive
				+ g_NaP_eff * E_Na            # (batch_size,)  INaP persistent subthreshold drive
				+ g_leak  * E_leak            # (batch_size,)  passive leak drive
				+ input_current[:, i - 1]     # (batch_size,)  applied stimulus at i-1 ✓
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)               # (batch_size,)  mV

			# Exponential Euler voltage update — no clamps, no nan_to_num ✓
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Standard HH gate updates — exponential Euler
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)

			# M-gate (X1): exponential Euler with FLAT per-batch decay_p (precomputed)
			# Exact solution to first-order ODE: dp/dt = (p_ss(V) - p) / tau_p_base
			# decay_p = exp(-dt/tau_p_base) is constant across all timesteps ✓
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * decay_p   # (batch_size,)

		# Return voltage trace
		# nois_fact_obs=0.0 → observation noise term is zero
		# NO additional randn before return (iter-92: spurious draw caused NLE 26.7) ✓
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)