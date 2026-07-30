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
		Hodgkin-Huxley neuron with I_M (X1, p^1) + I_mAHP (X2, r^1).

		THIS ITERATION: Probe V_half_mAHP = -22.0mV (was -20.0mV OPT NLE=24.8).
		  All previously confirmed optimal axes restored at confirmed values.
		  Single structural change: V_half_mAHP shifted 2mV in hyperpolarising direction.

		CONFIRMED OPTIMAL CONFIGURATION (NLE=24.8, best to date):
		  delta_Vh        = +2.0mV   PERMANENTLY CLOSED {+3=HARM(26.4), +4=PLATEAU(25.1)}
		  delta_E_leak    = +2.0mV   PERMANENTLY CLOSED {+3=HARM(25.2), +4=HARM(25.8)}
		  E_K             = -107.0mV PERMANENTLY CLOSED {-106=HARM(25.8), -105=HARM(25.0)}
		  E_Na            =  53.0mV  PERMANENTLY CLOSED {55.0=HARM(27.4)}
		  C               =   1.0    PERMANENTLY CLOSED {0.9=HARM(27.6), 1.1=HARM(26.0)}
		  I_M gate power  = p^1      PERMANENTLY CLOSED {p^2=HARM(26.0)}
		  I_mAHP gate pw  = r^1      PERMANENTLY CLOSED {r^2=HARM(26.7)}
		  tau_mAHP coeff  = 25.0     PERMANENTLY CLOSED {coeff=55=HARM(24.9)}
		  V_half_mAHP     = PROBE -22.0mV (was -20.0mV OPT; -18.0mV=HARM(25.4))
		  k_M             = 10.0mV   PERMANENTLY CLOSED {k=9.0=HARM(25.0)}
		  k_mAHP          =  6.0mV   PERMANENTLY CLOSED {4=HARM, 8=HARM}
		  tau_M_sigma     = 20.0mV   PERMANENTLY CLOSED {15=HARM, 25=HARM}
		  tau_M_floor     =  1.0ms   PERMANENTLY CLOSED {0.5=HARM(25.7)}
		  floor_mAHP      =  1.0ms   PERMANENTLY CLOSED {0.5=HARM(26.0), 1.5=HARM(26.1)}
		  V_half_M        = -15.0mV  PERMANENTLY CLOSED {-10=HARM(26.4)}
		  gbar_M scale    =  0.065   PERMANENTLY CLOSED {0.075=HARM(25.4), 0.09=HARM(26.1)}
		  gbar_mAHP scale =  0.065   PERMANENTLY CLOSED {0.080=HARM(26.5)}
		  tau_max_M coeff =  75.0    PERMANENTLY CLOSED {95→[5,100]ms=HARM(26.2)}
		  delta_Vm        =  0.0mV   PERMANENTLY CLOSED {+2=HARM(25.6), -2=HARM(26.2)}
		  delta_Vn        =  0.0mV   PERMANENTLY CLOSED {+2=HARM(26.7), -2=HARM(25.2)}

		PROBE — V_half_mAHP = -22.0mV (2mV hyperpolarising shift from confirmed OPT -20.0mV):
		  Background:
		    V_half_mAHP was explored only in the depolarising direction: -18.0mV=HARM(25.4).
		    The hyperpolarising direction (-22.0mV) has NEVER been evaluated.
		    A 2mV hyperpolarising shift means r_inf activates at slightly lower voltages,
		    producing an earlier onset of post-spike AHP while keeping resting activation
		    negligible (r_inf at -70mV: ~1.1e-4 at -22mV vs ~2.4e-4 at -20mV — both negligible).
		  Proposed change:
		    V_half_mAHP = -22.0mV (was -20.0mV; 2mV hyperpolarising shift)
		    Effect on r_inf at key voltages:
		      V=-70mV (rest):  r_inf(-22) ≈ 1.1e-4 vs r_inf(-20) ≈ 2.4e-4 (both negligible)
		      V=-22mV (V_half): r_inf = 0.500 (by definition)
		      V=-20mV (old V_half): r_inf(-22) ≈ 0.595 > r_inf(-20) = 0.500
		      V=0mV (spike):   r_inf(-22) ≈ 0.973 vs r_inf(-20) ≈ 0.962 (slightly more)
		    Net effect: Boltzmann curve shifted 2mV left; channel activates slightly earlier
		    on the repolarisation slope; marginally stronger AHP at intermediate voltages.
		  Physiological rationale:
		    The mAHP channel half-activation voltage varies across cell types (Sah 1996;
		    Bhattacharjee & Bhattacharjee 2006). A shift to -22mV remains within the
		    physiologically plausible range (-25 to -15mV) for medium-AHP currents.
		    Earlier activation during repolarisation could improve inter-spike interval
		    regularity by providing more consistent hyperpolarisation depth, potentially
		    improving variance, skewness, and kurtosis statistics during stimulation.
		    Resting statistics are entirely unaffected (activation remains negligible).
		  Branching:
		    NLE < 24.8 → helpful; close V_half_mAHP at -22.0mV.
		    NLE ≈ 24.8 → plateau; close V_half_mAHP at -20.0mV (no hyperpolarising benefit).
		    NLE > 24.8 → harmful; close V_half_mAHP permanently at -20.0mV.

		PARAMETER ASSIGNMENT:
		  params[:,0]  gbar_Na    : Na+ max conductance      (mS/cm2)
		  params[:,1]  gbar_K     : K+-DR max conductance    (mS/cm2)
		  params[:,2]  g_leak     : leak conductance         (mS/cm2)
		  params[:,3]  |E_leak|   : leak reversal potential  (mV, negated internally)
		  params[:,4]  |Vt|       : voltage threshold        (mV, negated internally)
		  params[:,5]  nois_fact  : noise amplitude          (unitless)
		  params[:,6]  gbar_X1    : mapped → gbar_M    [0.003, 0.068] mS/cm2 (scale=0.065 CLOSED)
		  params[:,7]  gbar_X2    : mapped → gbar_mAHP [0.003, 0.068] mS/cm2 (scale=0.065 CLOSED)
		  params[:,8]  param_i    : mapped → tau_max_M [5, 80]ms (coeff=75 CLOSED)
		  params[:,9]  param_j    : mapped → tau_mAHP  [5, 30]ms (coeff=25 CLOSED)
		"""
		device = params.device

		# -- Random generator -------------------------------------------------
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# -- Extract base HH parameters ---------------------------------------
		gbar_Na   = params[:, 0].float()   # (batch_size,) Na+ max conductance    mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) K+-DR max conductance  mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) leak conductance        mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) leak reversal potential mV
		Vt        = -params[:, 4].float()  # (batch_size,) voltage threshold       mV
		nois_fact = params[:, 5].float()   # (batch_size,) noise amplitude         unitless

		# -- I_M parameters (X1): sigmoid-mapped; ALL axes PERMANENTLY CLOSED ------
		# gbar_M:    [0.003, 0.068] mS/cm2; scale=0.065 CLOSED {0.075=HARM(25.4), 0.09=HARM(26.1)}
		# tau_max_M: [5.0, 80.0]ms; coeff=75.0 CLOSED {95.0→[5,100]ms=HARM(26.2)}
		# sigmoid maps any real input → (0,1); sign flip of raw param does not restrict range
		gbar_M    = 0.003 + 0.065 * torch.sigmoid(params[:, 6].float())   # (batch_size,) mS/cm2; [0.003, 0.068]
		tau_max_M = 5.00  + 75.0  * torch.sigmoid(params[:, 8].float())   # (batch_size,) ms; [5.0, 80.0] CLOSED

		# -- I_mAHP parameters (X2): ALL axes PERMANENTLY CLOSED except V_half_mAHP PROBE ------
		# gbar_mAHP: [0.003, 0.068] mS/cm2; scale=0.065 PERMANENTLY CLOSED {0.080=HARM(26.5)}
		# tau_mAHP:  [5.0, 30.0]ms; coeff=25.0 PERMANENTLY CLOSED {coeff=55=HARM(24.9)}
		gbar_mAHP = 0.003 + 0.065 * torch.sigmoid(params[:, 7].float())   # (batch_size,) mS/cm2; [0.003, 0.068] CLOSED
		tau_mAHP  = 5.00  + 25.0  * torch.sigmoid(params[:, 9].float())   # (batch_size,) ms; [5.0, 30.0] CLOSED

		tstep = float(dt)   # scalar ms

		# -- Fixed biophysical constants (ALL permanently closed) -------------
		nois_fact_obs = 0.0   # observation noise disabled
		C    = 1.0    # uF/cm2; PERMANENTLY CLOSED {0.9=HARM(27.6), 1.0=OPT, 1.1=HARM(26.0)}
		E_Na = 53.0   # mV;     PERMANENTLY CLOSED {55.0=HARM(27.4), 53.0=OPT}
		E_K  = -107.0 # mV;     PERMANENTLY CLOSED {-107=OPT(24.8), -106=HARM(25.8), -105=HARM(25.0)}

		# -- Fixed I_M kinetic constants (ALL permanently closed) -------------
		V_half_M    = -15.0  # mV;  PERMANENTLY CLOSED {-10.0=HARM(26.4), -15.0=OPT(24.8)}
		k_M         =  10.0  # mV;  PERMANENTLY CLOSED {k=9.0=HARM(25.0), k=10.0=OPT(24.8)}
		tau_M_sigma =  20.0  # mV;  PERMANENTLY CLOSED {15=HARM, 20=OPT(24.8), 25=HARM}
		tau_M_floor =   1.0  # ms;  PERMANENTLY CLOSED {0.5=HARM(25.7), 1.0=OPT}

		# -- Fixed I_mAHP kinetic constants (V_half_mAHP = PROBE; all others CLOSED) ----
		# V_half_mAHP: PROBE -22.0mV (was -20.0mV OPT; -18.0mV=HARM(25.4))
		# 2mV hyperpolarising shift — activates channel slightly earlier on repolarisation
		# Resting activation remains negligible: r_inf(-70mV at -22) ≈ 1.1e-4
		V_half_mAHP = -22.0  # mV;  PROBE {-20.0=OPT(24.8), -18.0=HARM(25.4), -22.0=THIS PROBE}
		k_mAHP      =   6.0  # mV;  PERMANENTLY CLOSED {4=HARM, 6=OPT(24.8), 8=HARM}
		floor_mAHP  =   1.0  # ms;  PERMANENTLY CLOSED {0.5=HARM(26.0), 1.0=OPT(24.8), 1.5=HARM(26.1)}

		# -- Na+ h-gate voltage offset — PERMANENTLY CLOSED AT +2.0mV --------
		# Effective offsets: alpha_h: -17+2=-15mV from Vt; beta_h: -40+2=-38mV from Vt
		# {+2=OPT(24.8), +3=HARM(26.4), +4=PLATEAU(25.1)} — ALL steps exhausted
		delta_Vh = 2.0   # mV; PERMANENTLY CLOSED at +2.0mV

		# -- Leak reversal offset — PERMANENTLY CLOSED AT +2.0mV -------------
		# {+2=OPT(24.8), +3=HARM(25.2), +4=HARM(25.8)} — PERMANENTLY CLOSED at +2.0mV
		delta_E_leak = 2.0   # mV; PERMANENTLY CLOSED at +2.0mV

		# Apply effective leak reversal per batch element
		E_leak_eff = E_leak + delta_E_leak   # (batch_size,) mV; effective leak reversal

		# -- Effective I_mAHP time constant (pre-computed; constant per batch) --------
		# tau_r_eff: [6.0, 31.0]ms = floor_mAHP(1ms) + tau_mAHP([5,30]ms)
		# Bell-shaped voltage-dependent tau was HARMFUL (NLE=25.5) → constant CLOSED
		tau_r_eff = floor_mAHP + tau_mAHP   # (batch_size,) ms; range [6.0, 31.0]ms

		# -- Numerical helpers ------------------------------------------------
		def Exp(z):
			# Safe exponential clamped at z=-500 to prevent numerical underflow.
			# z: torch.Tensor any shape -> torch.Tensor same shape
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# Numerically stable z/(exp(z)-1); L'Hopital for |z| < 1e-4.
			# z: torch.Tensor any shape -> torch.Tensor same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# -- Standard HH gate kinetics ----------------------------------------
		def alpha_m(x):   # x: (batch_size,) mV -> (batch_size,) ms^-1
			# Na+ activation; delta_Vm=0.0 PERMANENTLY CLOSED {+2=HARM(25.6), -2=HARM(26.2)}
			v1 = x - Vt - 13.0   # (batch_size,) mV
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,) ms^-1

		def beta_m(x):    # x: (batch_size,) mV -> (batch_size,) ms^-1
			# Na+ deactivation; delta_Vm=0.0 PERMANENTLY CLOSED
			v1 = x - Vt - 40.0   # (batch_size,) mV
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,) ms^-1

		def alpha_h(x):   # x: (batch_size,) mV -> (batch_size,) ms^-1
			# Na+ inactivation onset; delta_Vh=+2.0mV PERMANENTLY CLOSED
			# Effective offset: -17 + 2 = -15mV from Vt (slows inactivation onset slightly)
			v1 = x - Vt - 17.0 + delta_Vh   # (batch_size,) mV; effective: x - Vt - 15.0
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,) ms^-1

		def beta_h(x):    # x: (batch_size,) mV -> (batch_size,) ms^-1
			# Na+ inactivation recovery; delta_Vh=+2.0mV PERMANENTLY CLOSED
			# Effective offset: -40 + 2 = -38mV from Vt (speeds recovery slightly)
			v1 = x - Vt - 40.0 + delta_Vh   # (batch_size,) mV; effective: x - Vt - 38.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,) ms^-1

		def alpha_n(x):   # x: (batch_size,) mV -> (batch_size,) ms^-1
			# K+-DR activation; delta_Vn=0.0 PERMANENTLY CLOSED {+2=HARM(26.7), -2=HARM(25.2)}
			v1 = x - Vt - 15.0   # (batch_size,) mV
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,) ms^-1

		def beta_n(x):    # x: (batch_size,) mV -> (batch_size,) ms^-1
			# K+-DR deactivation; delta_Vn=0.0 PERMANENTLY CLOSED
			v1 = x - Vt - 10.0   # (batch_size,) mV
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,) ms^-1

		def tau_x(alpha, beta):   # (batch_size,), (batch_size,) -> (batch_size,) ms
			return 1.0 / (alpha + beta)   # (batch_size,) ms

		def inf_x(alpha, beta):   # (batch_size,), (batch_size,) -> (batch_size,) [0,1]
			return alpha / (alpha + beta)   # (batch_size,) [0,1]

		# ===== BEGIN EDITABLE SECTION =====

		# -- I_M gate kinetics (X1): Boltzmann ss + Gaussian bell tau ---------
		def p_inf_M(x):   # x: (batch_size,) mV -> (batch_size,) [0,1]
			# Boltzmann steady-state for muscarinic K+ (KCNQ/Kv7) channel.
			# V_half_M=-15mV CLOSED; k_M=10mV CLOSED.
			# Slow adaptation current; activates near spike threshold; negligible at rest.
			# At V=-70mV(rest): p_inf ≈ 4.2e-3; at V=-15mV(V_half): 0.500; at V=+20mV: ≈0.976
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_M))   # (batch_size,) [0,1]

		def tau_M_bell(x):   # x: (batch_size,) mV -> (batch_size,) ms
			# Gaussian bell-shaped time constant centered at V_half_M=-15mV.
			# Slowest near gate midpoint; faster at extreme voltages.
			# tau_M_sigma=20mV CLOSED; tau_M_floor=1.0ms CLOSED; tau_max_M range [5,80]ms CLOSED.
			# Bell peak range: [tau_M_floor, tau_M_floor + tau_max_M] = [1.0, ~81.0]ms.
			return tau_M_floor + tau_max_M * Exp(-((x - V_half_M) / tau_M_sigma) ** 2)   # (batch_size,) ms

		# -- I_mAHP gate kinetics (X2): Boltzmann ss + constant effective tau --
		def r_inf_mAHP(x):   # x: (batch_size,) mV -> (batch_size,) [0,1]
			# Boltzmann steady-state for medium-AHP K+ channel.
			# V_half_mAHP=-22.0mV PROBE {-20=OPT(24.8), -18=HARM(25.4), -22=THIS PROBE}
			# k_mAHP=6mV PERMANENTLY CLOSED.
			# Spike-triggered AHP; activates during repolarisation; negligible at rest.
			# At V=-70mV(rest): r_inf ≈ 1.1e-4 (negligible; resting statistics unaffected)
			# At V=-22mV(V_half): r_inf = 0.500
			# At V=0mV (spike): r_inf ≈ 0.973 (stronger than at V_half=-20: r_inf≈0.962)
			# Hyperpolarising shift → earlier activation on repolarisation slope
			#   → tighter inter-spike AHP → potentially improved IVI regularity/kurtosis
			return 1.0 / (1.0 + Exp(-(x - V_half_mAHP) / k_mAHP))   # (batch_size,) [0,1]

		# tau_r_eff: (batch_size,) ms in [6.0, 31.0]ms — constant per batch.
		# Voltage-dependent bell tau for I_mAHP was HARMFUL (NLE=25.5) → constant CLOSED.

		# ===== END EDITABLE SECTION =====

		# -- State variable allocation ----------------------------------------
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) K+-DR gate [0,1]
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ act gate [0,1]
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ inact gate [0,1]

		# ===== BEGIN EDITABLE SECTION =====
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) I_M gate p^1 [0,1]
		r = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) I_mAHP gate r^1 [0,1]
		# ===== END EDITABLE SECTION =====

		# -- Steady-state initialisation at V_init ----------------------------
		V_init  = init_voltage.to(device)                                   # (batch_size,) mV
		V[:, 0] = V_init                                                    # (batch_size,) mV
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                 # (batch_size,) [0,1]
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                 # (batch_size,) [0,1]
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                 # (batch_size,) [0,1]

		# ===== BEGIN EDITABLE SECTION =====
		# I_M gate initialisation: Boltzmann ss; p^1 CLOSED
		# At rest V≈-70mV: p_inf_M ≈ 4.2e-3 (negligible initial activation)
		p[:, 0] = p_inf_M(V[:, 0])    # (batch_size,) [0,1]

		# I_mAHP gate initialisation: Boltzmann ss; r^1 CLOSED; V_half=-22mV PROBE
		# At rest V≈-70mV: r_inf_mAHP ≈ 1.1e-4 (negligible initial activation)
		r[:, 0] = r_inf_mAHP(V[:, 0]) # (batch_size,) [0,1]
		# ===== END EDITABLE SECTION =====

		# -- Exponential Euler time integration --------------------------------
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) mV; membrane voltage at previous step
			n_prev = n[:, i - 1]   # (batch_size,) K+-DR gate state [0,1]
			m_prev = m[:, i - 1]   # (batch_size,) Na+ activation gate state [0,1]
			h_prev = h[:, i - 1]   # (batch_size,) Na+ inactivation gate state [0,1]

			# Standard HH gate kinetics; delta_Vh=+2.0mV PERMANENTLY CLOSED in alpha/beta_h
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) ms^-1; delta_Vm=0 CLOSED
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) ms^-1; delta_Vh=+2 CLOSED
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) ms^-1; delta_Vn=0 CLOSED

			# ===== BEGIN EDITABLE SECTION =====
			p_prev = p[:, i - 1]   # (batch_size,) I_M gate state [0,1]
			r_prev = r[:, i - 1]   # (batch_size,) I_mAHP gate state [0,1]

			# I_M: Boltzmann ss + Gaussian bell tau; all constants PERMANENTLY CLOSED
			p_ss      = p_inf_M(V_prev)    # (batch_size,) [0,1]; Boltzmann ss target for p
			# tau_M_bell: peak in [1.0, ~81.0]ms; coeff=75 CLOSED; centred at V_half_M=-15mV
			tau_M_cur = tau_M_bell(V_prev) # (batch_size,) ms; voltage-dependent bell

			# I_mAHP: Boltzmann ss + constant per-batch tau; V_half_mAHP PROBE
			r_ss = r_inf_mAHP(V_prev)      # (batch_size,) [0,1]; Boltzmann ss target for r
			# tau_r_eff: (batch_size,) ms in [6.0, 31.0]ms; constant; PERMANENTLY CLOSED
			# (bell-shaped voltage-dependent tau was HARMFUL NLE=25.5)
			# ===== END EDITABLE SECTION =====

			# Effective inverse membrane time constant: sum(g_i) / C  [ms^-1]
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,) Na+ conductance    mS/cm2
				+ (n_prev ** 4) * gbar_K            # (batch_size,) K+-DR conductance  mS/cm2
				+ g_leak                             # (batch_size,) leak conductance   mS/cm2
				# ===== BEGIN EDITABLE SECTION =====
				+ gbar_M    * p_prev                # (batch_size,) I_M; p^1 CLOSED
				+ gbar_mAHP * r_prev                # (batch_size,) I_mAHP; r^1 CLOSED
				# ===== END EDITABLE SECTION =====
			) / C   # (batch_size,) ms^-1

			# Voltage steady-state: [sum(g_i*E_i) + I_inj + noise] / [sum(g_i)*C]
			# Process noise MUST remain INSIDE V_inf for RC-filtered integration (CRITICAL)
			# E_leak_eff = E_leak + 2.0mV PERMANENTLY CLOSED
			# E_K = -107.0mV PERMANENTLY CLOSED
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na+ driving force
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) K+-DR; E_K=-107mV CLOSED
				+ g_leak * E_leak_eff                       # (batch_size,) leak; E_leak+2mV CLOSED
				# ===== BEGIN EDITABLE SECTION =====
				+ gbar_M    * p_prev * E_K                 # (batch_size,) I_M driving; E_K CLOSED
				+ gbar_mAHP * r_prev * E_K                 # (batch_size,) I_mAHP driving; E_K CLOSED
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                  # (batch_size,) applied current injection
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,) process noise: INSIDE V_inf for RC filtering (CRITICAL)
			) / (tau_V_inv * C)   # (batch_size,) mV

			# Exponential Euler: exact solution for linear ODE dy/dt = (y_inf - y) / tau_y
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                   # (batch_size,) mV
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))       # (batch_size,) [0,1]
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))       # (batch_size,) [0,1]
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))       # (batch_size,) [0,1]

			# ===== BEGIN EDITABLE SECTION =====
			# I_M gate p: p^1 CLOSED; Gaussian bell tau; tau_max_M coeff=75 CLOSED [5,80]ms
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_M_cur)    # (batch_size,) [0,1]

			# I_mAHP gate r: r^1 CLOSED; constant tau_r_eff [6.0,31.0]ms CLOSED
			# V_half_mAHP=-22.0mV PROBE (was -20.0mV OPT NLE=24.8; -18.0mV=HARM(25.4))
			r[:, i] = r_ss + (r_prev - r_ss) * Exp(-tstep / tau_r_eff)    # (batch_size,) [0,1]
			# ===== END EDITABLE SECTION =====

		# Return voltage traces; observation noise disabled (nois_fact_obs=0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps) mV