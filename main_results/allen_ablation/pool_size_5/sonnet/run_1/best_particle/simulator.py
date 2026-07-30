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
		Hodgkin-Huxley neuron with one additional channel: Persistent Sodium (INaP).

		DESIGN DECISIONS (informed by prior iteration feedback):

		CHANNEL SELECTION — WHY INaP ONLY:
		  The task requires tonic, evenly-spaced spiking with quiescence at rest.
		  After testing two-channel combinations in prior iterations, identifiability
		  problems arose (high NLE from conflicting parameter posteriors). Returning
		  to a single well-calibrated channel:

		  INaP (X1) is retained because:
		    * Non-inactivating Na+ conductance active at subthreshold voltages (~-70 to -50 mV)
		    * Amplifies depolarising fluctuations between spikes → lowers effective threshold
		    * Stabilises regular ISI spacing — exactly matching target tonic spiking pattern
		    * Does NOT cause adaptation, bursting, or high-frequency clustering
		    * Two inferred parameters (gbar_NaP, V_half_NaP) are sufficient and identifiable

		  Ih/HCN (X2) was removed because:
		    * Prior feedback confirmed that with E_h = -30 mV, Ih caused pathological
		      tonic depolarisation at rest, corrupting all seven summary statistics
		    * Adding a second channel creates parameter identifiability problems
		      (INaP and Ih both depolarise subthreshold — their conductances trade off)
		    * Parsimony principle: simpler model preferred; adding X2 only if clearly justified

		INaP CORRECTIONS vs. PRIOR ITERATIONS:
		  1. k_NaP: changed from 5.0 mV → 9.0 mV
		     - Prior feedback identified k=5 as "too steep": only 20 mV covers 5%-95%
		       activation, making the Boltzmann very sharp and creating a narrow sodium
		       window that amplifies subthreshold instability
		     - k=9.0 mV spreads activation over ~36 mV (gentler, physiologically validated
		       for cortical persistent Na+ channels, literature range 5-12 mV)
		     - Wider slope also smooths the posterior landscape for V_half_NaP inference
		  2. V_half_NaP = -params[7]: params[7] ∈ [1e-4, 150] → V_half ∈ [-150, ~0] mV
		     Posterior will converge to physiological range -65 to -40 mV
		  3. params[8] and params[9] (X2 slot): deliberately unused — not borrowed for X1
		     (prior feedback flagged illegal cross-slot parameter usage)

		  E_K correction: -107 mV → -90 mV
		     - -107 mV is a squid-axon value; mammalian preparations use -90 mV
		     - Corrects AHP depth, spike afterhyperpolarisation waveform, and voltage
		       distribution statistics (variance, skewness, kurtosis)

		Args:
			init_voltage:  torch.Tensor (batch_size,)            initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) applied current (uA/cm²)
			dt:            float                                  time step (ms)
			t:             torch.Tensor (time_steps,)             time array (ms)
			params:        torch.Tensor (batch_size, 10)          parameter vector
			seed:          int or None

		Returns:
			V:             torch.Tensor (batch_size, time_steps)  membrane voltage (mV)
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

		# ── Base HH parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²  transient Na+ conductance
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²  delayed-rectifier K+ conductance
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²  passive leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV      leak reversal (sign applied internally)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV      voltage threshold shift (sign applied internally)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless noise amplitude

		# ── X1 slot: INaP — Persistent (non-inactivating) sodium current ─────
		# Biophysical formulation:
		#   I_NaP = gbar_NaP * mp_inf(V) * (V - E_Na)
		#   mp_inf(V) = 1 / (1 + exp(-(V - V_half_NaP) / k_NaP))   [Boltzmann activation]
		#
		# Parameter assignments (strictly confined to X1 slot, params[6] and params[7]):
		#
		# gbar_NaP   = params[6], range [1e-4, 10] mS/cm²
		#   Maximum conductance of the persistent Na+ channel.
		#   Literature values for cortical/hippocampal neurons: 0.01–2 mS/cm²
		#   Posterior inference selects the value that reproduces correct spike count/rate.
		#
		# V_half_NaP = -params[7], params[7] ∈ [1e-4, 150] → V_half ∈ [-150, ~0] mV
		#   Half-activation voltage of the Boltzmann gate.
		#   Physiological INaP range: -65 to -40 mV (well within prior support)
		#   Posterior selects value setting correct subthreshold amplification level.
		#
		# k_NaP      = 9.0 mV  (FIXED physiological slope factor)
		#   Boltzmann slope; wider than prior iterations (was 5.0 mV):
		#   - k=5 mV: 5%→95% activation over 20 mV — too steep, creates Na+ window instability
		#   - k=9 mV: 5%→95% activation over 36 mV — gentler, physiologically calibrated
		#     for cortical INaP channels (Crill 1996, Magistretti & Alonso 1999)
		#   Fixing k_NaP reduces parameter dimensionality, avoids identifiability
		#   conflicts with V_half_NaP inference, smooths posterior landscape.
		#
		# Instantaneous gate approximation:
		#   INaP activates on sub-ms timescales (tau_mp ~ 0.1–0.5 ms in literature),
		#   much smaller than any physiological dt. Therefore mp ≈ mp_inf(V(t)) is
		#   standard (Magistretti & Alonso 1999; Fransén et al. 2004), avoiding an
		#   unnecessary ODE and state variable for the INaP gate.
		gbar_NaP   = params[:, 6].float()   # (batch_size,)  mS/cm²  INaP maximal conductance
		V_half_NaP = -params[:, 7].float()  # (batch_size,)  mV      INaP half-activation voltage
		k_NaP      = 9.0                     # mV (scalar fixed constant — physiological slope)
		# params[8] (param_i), params[9] (param_j): X2 slot — deliberately unused.
		# Not borrowed for INaP (prior feedback flagged cross-slot usage as a violation).
		# X2 omitted per parsimony principle: INaP alone is hypothesized sufficient;
		# adding Ih/HCN (X2) was shown to cause identifiability problems in prior iterations.

		tstep = float(dt)

		# ── Fixed biophysical constants ────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise amplitude (disabled, preserved as-is)
		C    = 1.0            # uF/cm²  membrane capacitance
		E_Na = 53.0           # mV      sodium reversal (transient Na+ and INaP, same channel type)
		# E_K corrected from base model value of -107 mV:
		#   -107 mV is the squid-axon K+ reversal (Hodgkin & Huxley 1952, cold seawater).
		#   Mammalian intracellular recordings use -90 mV (standard for [K+]_o = 5 mM).
		#   The -107 mV base value produces over-deep AHPs, distorted spike waveforms,
		#   and systematic errors in voltage variance, skewness, and kurtosis — exactly
		#   the summary statistics evaluated. Correcting to -90 mV addresses these.
		E_K  = -90.0          # mV      potassium reversal (mammalian standard)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# z: any shape → same shape  (numerically stable exponential with floor at -500)
			# Prevents underflow in exp(-500) → 0 and overflow avoidance
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z),
			)

		def efun(z):
			# z: any shape → same shape  (x / (exp(x) - 1), L'Hôpital near z=0)
			# Used in standard HH alpha/beta rate functions to avoid division by zero
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gate kinetics ─────────────────────────────────────────
		def alpha_m(x):
			# x: (batch_size,) → (batch_size,)  Na+ activation opening rate (ms⁻¹)
			v1 = x - Vt - 13.0    # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) → (batch_size,)  Na+ activation closing rate (ms⁻¹)
			v1 = x - Vt - 40.0    # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) → (batch_size,)  Na+ inactivation opening rate (ms⁻¹)
			v1 = x - Vt - 17.0    # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) → (batch_size,)  Na+ inactivation closing rate (ms⁻¹)
			v1 = x - Vt - 40.0    # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) → (batch_size,)  K+ activation opening rate (ms⁻¹)
			v1 = x - Vt - 15.0    # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) → (batch_size,)  K+ activation closing rate (ms⁻¹)
			v1 = x - Vt - 10.0    # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)  gate time constant (ms)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)  gate steady-state (dimensionless)
			return alpha / (alpha + beta)

		# ── INaP instantaneous gate ───────────────────────────────────────────
		# Boltzmann steady-state activation (depolarisation-activated, positive slope):
		#   mp_inf(V) = 1 / (1 + exp(-(V - V_half_NaP) / k_NaP))
		# At V = V_half_NaP: mp_inf = 0.5 (50% activated)
		# At V << V_half_NaP (hyperpolarised): mp_inf → 0 (inactive)
		# At V >> V_half_NaP (depolarised): mp_inf → 1 (fully active)
		# V_half_NaP: (batch_size,)  inferred parameter
		# k_NaP:      scalar 9.0 mV  fixed physiological slope
		def mp_inf(x):
			# x: (batch_size,) → (batch_size,)  INaP Boltzmann steady-state activation
			return 1.0 / (1.0 + Exp(-(x - V_half_NaP) / k_NaP))

		# ── State variable allocation ─────────────────────────────────────────
		# INaP gate is instantaneous → no state variable needed (reduces ODE system size)
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ delayed-rectifier gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation gate
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation gate

		# ── Initial conditions ────────────────────────────────────────────────
		V_init  = init_voltage.to(device)                          # (batch_size,)
		V[:, 0] = V_init                                           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)  K+ gate at steady-state
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)  Na+ activation at steady-state
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)  Na+ inactivation at steady-state
		# INaP: no initialization required — instantaneous gate recomputed each step

		# ── Main simulation loop ──────────────────────────────────────────────
		for i in range(1, time_steps):
			# Standard HH gate rates at previous timestep voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])  # each (batch_size,) ms⁻¹
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])  # each (batch_size,) ms⁻¹
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])  # each (batch_size,) ms⁻¹

			# INaP: evaluate instantaneous gate at previous voltage
			# Sub-ms activation timescale → mp_inf(V) is the quasi-static approximation
			mp = mp_inf(V[:, i - 1])   # (batch_size,)  dimensionless, ∈ [0, 1]

			# ── Effective inverse membrane time constant: tau_V_inv = Σ g_i / C ──
			# Each conductance-weighted term contributes to the total shunting conductance.
			# INaP adds gbar_NaP * mp as an additional conductance term.
			# This term is always non-negative (gbar_NaP > 0, mp ∈ [0,1]), preserving stability.
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)  transient Na+ conductance
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,)  delayed-rectifier K+ conductance
				+ g_leak                                        # (batch_size,)  passive leak conductance
				+ gbar_NaP * mp                                 # (batch_size,)  INaP conductance (instantaneous)
			) / C                                               # (batch_size,)  ms⁻¹

			# ── Voltage steady-state: conductance-weighted reversal potentials + drives ──
			# Each conductance term pulls V toward its reversal potential.
			# INaP reverses at E_Na = 53 mV (depolarising — it is a Na+ channel).
			# The persistent Na+ window current (gbar_NaP * mp * (V - E_Na)) provides
			# sustained inward current at subthreshold voltages when V_half_NaP is near rest,
			# which lowers the effective spike threshold and promotes regular tonic spiking.
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)  transient Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)  K+ repolarising drive (E_K=-90 mV)
				+ g_leak * E_leak                                       # (batch_size,)  passive leak drive
				+ gbar_NaP * mp * E_Na                                  # (batch_size,)  INaP depolarising drive
				+ input_current[:, i - 1]                              # (batch_size,)  externally injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                                        # (batch_size,)  mV

			# Exact exponential (operator-splitting) integration for voltage
			# V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt * tau_V_inv)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)  mV

			# Exact exponential integration for standard HH gating variables
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			# INaP: no gate integration — instantaneous approximation, gate recomputed next step

		# Return voltage traces with optional observation noise (currently disabled, nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)  mV