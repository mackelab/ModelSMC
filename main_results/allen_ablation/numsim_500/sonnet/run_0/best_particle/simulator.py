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
		Hodgkin-Huxley neuron extended with a fast-inactivating A-type K+ current (IA) in slot X1.

		DESIGN RATIONALE — WHY IA INSTEAD OF M-CURRENT:
		  The previous M-current (IM) was rejected because it introduces slow spike-frequency
		  adaptation: the M-gate p progressively accumulates during sustained depolarization,
		  steadily increasing K+ conductance and decelerating firing over the stimulus period.
		  This directly contradicts the data characteristic of "evenly-spaced action potentials."

		  A-type K+ current (IA) is appropriate because:
		    - It activates rapidly near spike threshold but inactivates within ~20 ms
		    - At rest (~-70 mV), the inactivation gate b is near its steady state (de-inactivated)
		    - During each inter-spike interval, b partially recovers from inactivation
		    - This creates cycle-by-cycle regulation of spike timing WITHOUT cumulative adaptation
		    - The net conductance does not increase across spikes → evenly-spaced ISIs preserved
		    - Reversal at E_K → purely hyperpolarizing → no burst-promoting mechanism
		  Reference: Connor & Stevens (1971); Huguenard & McCormick (1992)

		PARAMETER SLOT LAYOUT:
		  [0]  gbar_Na        — Na+ max conductance (mS/cm²)
		  [1]  gbar_K         — K+ delayed rectifier max conductance (mS/cm²)
		  [2]  g_leak         — leak conductance (mS/cm²)
		  [3]  |E_leak|       — leak reversal magnitude (mV), E_leak = -params[3]
		  [4]  |Vt|           — voltage threshold shift magnitude (mV), Vt = -params[4]
		  [5]  nois_fact      — stochastic noise scaling (unitless)
		  [6]  gbar_X1→gbar_A — A-type K+ max conductance (mS/cm²) [X1 conductance, range 1e-4..10]
		  [7]  gbar_X2        — UNUSED: reserved for future second channel
		  [8]  |param_i|      — UNUSED: reserved for future X2 parameter
		  [9]  |param_j| → V_half_inact_A: A-current inactivation half-voltage [X1 kinetic param]
		                   param_j = -params[:,9], prior range params[9]∈[1e-4,3000]
		                   → V_half_inact_A ∈ (-3000, ~0) mV; physiological range -60 to -90 mV
		                   is well covered; SBI posterior will concentrate near -70 to -85 mV

		E_K CORRECTION:
		  Changed from -107 mV (squid axon HH 1952, isolated preparation) to -90 mV
		  (standard for intact mammalian neurons, ~4 mM [K+]_out via Nernst equation).
		  This corrects systematic AHP depth bias affecting resting potential mean/std metrics.

		ACTIVATION KINETICS (fixed, not tunable):
		  - Half-activation V_half_act_A = -50 mV (Connor & Stevens 1971)
		  - Activation slope k = 20 mV (broad, subthreshold-to-suprathreshold range)
		  - Activation is treated as INSTANTANEOUS: a_inf(V) only, no separate gate ODE
		    This is standard for IA where tau_a << dt in most HH implementations
		    and avoids adding an extra tunable parameter

		INACTIVATION KINETICS:
		  - V_half_inact_A = param_j = -params[:,9] (tunable, batch-specific)
		  - Inactivation slope = 6 mV (sharp, typical for IA inactivation)
		  - tau_b_A = 20 ms (fixed): slow enough to persist across ISI, fast enough
		    to recover between spikes at resting potential → cyclic regulation, not adaptation

		Args:
			init_voltage: torch.Tensor (batch_size,)    — initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, T) — injected current (μA/cm²)
			dt: float                                    — integration time step (ms)
			t: torch.Tensor (T,)                        — time array (ms)
			params: torch.Tensor (batch_size, 10)       — parameter vector
			seed: int or None                           — RNG seed

		Returns:
			V: torch.Tensor (batch_size, T)             — membrane voltage traces (mV)
		"""
		device = params.device

		# ── Random generator ─────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int
		tstep      = float(dt)         # ms

		# ── Standard HH base parameters ──────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# ── Slot X1: A-type K+ current (IA) ──────────────────────────────────
		# gbar_A: max conductance for the fast-inactivating A-type K+ current
		# Prior: params[:,6] ∈ [1e-4, 10] mS/cm²
		gbar_A = params[:, 6].float()   # (batch_size,) mS/cm²

		# V_half_inact_A: half-inactivation voltage for the b gate of IA
		# Derived from params[:,9] (positive prior [1e-4, 3000]) via negation.
		# Physiological range: -60 to -90 mV. This is within (-3000, 0) so the prior
		# is wide enough; SBI will concentrate the posterior in the physiological range.
		V_half_inact_A = -params[:, 9].float()   # (batch_size,) mV, typically -60 to -90 mV

		# ── Slot X2: UNUSED ───────────────────────────────────────────────────
		# params[:,7] (gbar_X2) and params[:,8] (param_i) reserved but not used.
		# Following parsimony principle: IA alone is the minimal intervention needed.

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0    # observation noise (disabled)
		C    = 1.0             # μF/cm² — specific membrane capacitance
		E_Na = 53.0            # mV — sodium reversal potential
		# Updated from -107 mV (squid axon, HH 1952) to -90 mV (intact neuron standard).
		# -107 mV causes excessively deep AHP, biasing resting potential and voltage statistics.
		E_K  = -90.0           # mV — potassium reversal potential (all K+ channels)

		# ── Fixed IA kinetic constants ────────────────────────────────────────
		# Activation half-voltage: -50 mV, slope 20 mV (Connor & Stevens 1971)
		V_half_act_A = -50.0   # mV, fixed — activation center
		k_act_A      = 20.0    # mV, fixed — activation slope (broad, near threshold)
		# Inactivation slope: 6 mV (sharp inactivation curve, standard for IA)
		k_inact_A    = 6.0     # mV, fixed — inactivation slope
		# Fixed inactivation time constant: 20 ms
		#   - Long enough to persist during inter-spike interval (partial inactivation)
		#   - Short enough to fully recover between spikes at rest (~-70 mV)
		#   - Does NOT accumulate across spikes → no progressive adaptation
		tau_b_A = 20.0         # ms, fixed

		####################################################################
		# Numerical utility functions
		####################################################################
		def Exp(z):
			# Stable exponential: floor argument at -500 to prevent underflow
			# z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# HH rate integral z/(exp(z)-1), linearized near z=0 to avoid 0/0
			# z: any shape → same shape
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,
				z / (Exp(z) - 1.0)
			)

		####################################################################
		# Standard HH channel kinetics
		####################################################################
		def alpha_m(x):
			# Na+ activation forward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation backward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation forward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation backward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ delayed-rectifier forward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ delayed-rectifier backward rate  x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gating time constant  → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Steady-state gating variable  → (batch_size,)
			return alpha / (alpha + beta)

		####################################################################
		# A-type K+ current (IA) kinetics — slot X1
		#
		# I_A = gbar_A * a_inf(V)^3 * b * (V - E_K)
		#
		# Activation: INSTANTANEOUS (no ODE), Boltzmann steady state only.
		#   a_inf(V) = 1 / (1 + exp(-(V - V_half_act_A) / k_act_A))
		#   Treating activation as instantaneous is standard for IA in HH models
		#   because tau_a (~1 ms) is much smaller than dt and ISI duration.
		#   This reduces the state vector by one dimension without loss of accuracy.
		#
		# Inactivation: FIRST-ORDER ODE with fixed tau_b_A = 20 ms.
		#   db/dt = (b_inf(V) - b) / tau_b_A
		#   b_inf(V) = 1 / (1 + exp((V - V_half_inact_A) / k_inact_A))
		#   Note: b_inf is 1 at hyperpolarized potentials (inactivation removed = b high)
		#         b_inf is 0 at depolarized potentials (inactivation = b low)
		#   This is physiologically correct: IA is de-inactivated at rest, transiently
		#   activates upon depolarization toward threshold, then rapidly inactivates.
		####################################################################
		def a_inf_A(x):
			# Instantaneous activation of IA: peaks near spike threshold
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_act_A) / k_act_A))

		def b_inf_A(x):
			# Inactivation steady state of IA: near 1 at rest, near 0 during spike
			# V_half_inact_A is batch-specific (tunable via params[:,9])
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((x - V_half_inact_A) / k_inact_A))

		####################################################################
		# State variable allocation
		####################################################################
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) K+ DR gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ act gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) Na+ inact gate
		b = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, T) IA inact gate

		####################################################################
		# Initialization at steady state (t=0)
		####################################################################
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))           # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))           # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))           # (batch_size,)
		# IA inactivation gate initialized at its voltage-dependent steady state.
		# At typical resting potential (~-70 mV), b_inf ≈ 1 (fully de-inactivated).
		b[:, 0] = b_inf_A(V[:, 0])                                    # (batch_size,)

		####################################################################
		# Simulation loop — exponential Euler integration
		#
		# For linear ODE: C * dV/dt = -g_total * (V - V_inf)
		# Exact step: V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt * g_total / C)
		# Same exact integration applied to all first-order gating variables.
		####################################################################
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) voltage at step i-1
			n_prev = n[:, i - 1]   # (batch_size,) K+ DR gate at step i-1
			m_prev = m[:, i - 1]   # (batch_size,) Na+ activation gate at step i-1
			h_prev = h[:, i - 1]   # (batch_size,) Na+ inactivation gate at step i-1
			b_prev = b[:, i - 1]   # (batch_size,) IA inactivation gate at step i-1

			# Standard HH gating rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# IA kinetics at current voltage
			a_A_inst = a_inf_A(V_prev)    # (batch_size,) instantaneous activation
			b_A_ss   = b_inf_A(V_prev)    # (batch_size,) inactivation steady state

			# ── Effective conductance g_total / C = tau_V_inv ─────────────────
			# IA uses instantaneous activation cubed: a_A_inst^3
			# This couples IA to the voltage update within the same timestep.
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev          # (batch_size,) Na+ term
				+ (n_prev ** 4) * gbar_K                  # (batch_size,) K+ DR term
				+ g_leak                                   # (batch_size,) leak term
				+ gbar_A * (a_A_inst ** 3) * b_prev       # (batch_size,) IA term
			) / C                                          # (batch_size,) ms⁻¹

			# ── Voltage steady-state V_inf ─────────────────────────────────────
			# All K+ channels (IK, IA) drive toward E_K = -90 mV → hyperpolarizing
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na+ drive
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) K+ DR drive
				+ g_leak * E_leak                           # (batch_size,) leak drive
				+ gbar_A * (a_A_inst ** 3) * b_prev * E_K # (batch_size,) IA drive
				+ input_current[:, i - 1]                  # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                            # (batch_size,) mV

			# ── Exponential Euler updates ──────────────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n)) # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m)) # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h)) # (batch_size,)
			# IA inactivation: fixed tau_b_A = 20 ms (no accumulation between spikes)
			# Scalar tau_b_A used with torch.full_like to create correctly-shaped tensor
			b[:, i] = b_A_ss + (b_prev - b_A_ss) * Exp(torch.full_like(b_prev, -tstep / tau_b_A))   # (batch_size,)

		# Return voltage traces; observation noise currently disabled
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)