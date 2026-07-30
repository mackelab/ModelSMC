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
		Hodgkin-Huxley neuron extended with M-type K+ current (I_M).

		Changes from prior iteration (NLE 25.5):
		──────────────────────────────────────────
		1. tau_p extended from [5, 120] ms → [5, 400] ms via linear rescaling
		   of params[:,7]. Prior iteration hard-clamped at 120 ms, which saturated
		   the optimizer whenever the posterior pushed tau_p beyond that ceiling,
		   producing zero-gradient regions and stalling convergence. Rescaling
		   params[:,7] ∈ [1e-4, 120] linearly to [5, 400] ms removes the ceiling
		   while keeping the minimum at 5 ms for numerical stability.

		2. V_half_M made tunable via params[:,8]:
		   V_half_M = -25.0 - params[:,8] / 6.0, clamped to [-55, -25] mV.
		   This maps the raw [1e-4, 150] range of params[:,8] onto [-25, -50] mV,
		   covering the full physiological range of M-current half-activation
		   (~-35 to -45 mV for cortical neurons). Fixing V_half_M at -40 mV
		   prevented the optimizer from matching spike count and mean stimulation
		   voltage simultaneously, since small shifts in V_half change the amount
		   of tonic M-current between spikes and thus the inter-spike interval.

		Preserved from prior iterations:
		  - E_K = -77.0 mV (corrected from -107 mV two iterations ago)
		  - M-current fully in X1: gbar_M = params[:,6], tau_p ∝ params[:,7]
		  - V_half_M kinetic shape: Boltzmann with fixed slope k = 10 mV
		  - X2 slot (params[:,9]) unused — parsimony principle

		Parameter slot assignment:
		  params[:,6] → gbar_M    (M-current max conductance, mS/cm², range [1e-4, 10])
		  params[:,7] → tau_p     (rescaled to [5, 400] ms, raw range [1e-4, 120])
		  params[:,8] → V_half_M  (M-current half-activation, rescaled to [-55, -25] mV)
		  params[:,9] → unused    (X2 kinetic slot, parsimony)

		Physiological rationale for M-current:
		  I_M is a slow, non-inactivating, voltage-gated K+ current that activates
		  near or below spike threshold. It provides spike-frequency adaptation by
		  accumulating between action potentials, adding a progressive hyperpolarising
		  drive that regularises inter-spike intervals. This produces tonic, evenly-
		  spaced spiking — exactly the target phenotype — without bursting or silencing.

		Args:
			init_voltage : torch.Tensor (batch_size,)           — initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) — injected current (μA/cm²)
			dt           : float                                  — time step (ms)
			t            : torch.Tensor (time_steps,)             — time array (ms)
			params       : torch.Tensor (batch_size, 10)          — biophysical parameters
			seed         : int or None                            — RNG seed

		Returns:
			V: torch.Tensor (batch_size, time_steps) — membrane potential traces (mV)
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
		tstep      = float(dt)         # ms, scalar

		# ── Base HH parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm², Na+ max conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm², K+ delayed rectifier max conductance
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm², passive leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV, leak reversal (sign applied)
		Vt        = -params[:, 4].float()  # (batch_size,) mV, voltage threshold offset (sign applied)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless, noise scaling factor

		# ── X1 slot: M-current parameters (all three M-current d.o.f.) ───────

		# gbar_M: M-current max conductance; raw range [1e-4, 10] mS/cm²
		gbar_M = params[:, 6].float()  # (batch_size,) mS/cm²

		# tau_p: M-current time constant (voltage-independent).
		# FIX: prior hard-clamp at 120 ms prevented the optimizer from reaching
		# the slow-adaptation regime (50–300 ms). Now linearly rescaled:
		#   params[:,7] ∈ [1e-4, 120] → tau_p ∈ [5, ~405] ms
		# The lower clamp at 5 ms ensures numerical stability; no upper clamp is
		# needed because params[:,7] ≤ 120 naturally limits tau_p ≤ 405 ms.
		tau_p = 5.0 + params[:, 7].float() * (395.0 / 120.0)  # (batch_size,) ms, range [5, ~405]

		# V_half_M: M-current half-activation voltage.
		# FIX: previously fixed at -40 mV, now tunable via params[:,8].
		# Mapping: params[:,8] ∈ [1e-4, 150] (positive raw) → V_half_M ∈ [-25, -50] mV
		#   V_half_M = -25.0 - params[:,8] / 6.0
		#   At params[:,8] = 0   → V_half_M = -25.0 mV  (upper physiological bound)
		#   At params[:,8] = 150 → V_half_M = -50.0 mV  (lower physiological bound)
		# Clamped to [-55, -25] mV to prevent extrapolation beyond physiological range.
		# Typical cortical M-current V_half is -35 to -45 mV — fully covered.
		V_half_M = torch.clamp(
			-25.0 - params[:, 8].float() / 6.0,
			min=-55.0, max=-25.0
		)  # (batch_size,) mV, range [-55, -25]

		# X2 slot: params[:,9] intentionally unused — parsimony principle.
		# The three M-current parameters (gbar_M, tau_p, V_half_M) are sufficient
		# to characterise I_M; adding a second channel risks parameter identifiability
		# problems and overfitting without clear justification from the data.

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0    # observation noise (zero per specification)
		C    = 1.0             # μF/cm², membrane capacitance
		E_Na = 53.0            # mV, Na+ reversal potential

		# E_K = -77.0 mV: corrected from the base model default of -107 mV.
		# -77 mV is the standard Nernst value for K+ in typical neuronal preparations.
		# The erroneous -107 mV caused excessively deep after-hyperpolarizations,
		# shifted resting potential, and inflated voltage variance/kurtosis.
		E_K  = -77.0           # mV, K+ reversal potential (corrected from -107 mV)

		# M-current Boltzmann slope: fixed at 10 mV (standard literature value)
		k_M  = 10.0            # mV, slope factor for p-gate sigmoid

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; clips at -500 to prevent underflow
			# z: (batch_size,) -> (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (batch_size,) -> (batch_size,): L'Hôpital regularisation near z = 0
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ──────────────────────────────────────
		def alpha_m(x):
			# (batch_size,) -> (batch_size,): Na+ activation opening rate (ms⁻¹)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# (batch_size,) -> (batch_size,): Na+ activation closing rate (ms⁻¹)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# (batch_size,) -> (batch_size,): Na+ inactivation opening rate (ms⁻¹)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# (batch_size,) -> (batch_size,): Na+ inactivation closing rate (ms⁻¹)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# (batch_size,) -> (batch_size,): K+ delayed rectifier opening rate (ms⁻¹)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# (batch_size,) -> (batch_size,): K+ delayed rectifier closing rate (ms⁻¹)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,), (batch_size,) -> (batch_size,): gating time constant (ms)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,), (batch_size,) -> (batch_size,): steady-state gate value
			return alpha / (alpha + beta)

		# ── M-current p-gate kinetics ─────────────────────────────────────────
		# Steady-state activation: Boltzmann sigmoid with per-batch V_half_M and
		# fixed slope k_M = 10 mV. V_half_M ∈ [-55, -25] mV means the channel
		# begins to activate in the subthreshold range and is near-fully open
		# during the action potential peak, providing graded adaptation.
		def p_inf(x):
			# x: (batch_size,), V_half_M: (batch_size,) -> (batch_size,)
			# M-current steady-state activation (per-batch half-activation voltage)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_M))

		# tau_p is (batch_size,) ms — voltage-independent slow time constant
		# Range [5, ~405] ms ensures the channel is slow relative to Na+/K+ gates

		# ── State variable arrays ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ delayed rectifier gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation gate
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation gate
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate

		# ── Initialise at resting steady state ───────────────────────────────
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))            # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))            # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))            # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                      # (batch_size,) M-gate at resting V

		# ── Exponential Euler integration loop ────────────────────────────────
		for i in range(1, time_steps):

			# Standard HH rate constants at previous time step
			a_m, b_m = alpha_m(V[:, i-1]), beta_m(V[:, i-1])  # (batch_size,) each
			a_h, b_h = alpha_h(V[:, i-1]), beta_h(V[:, i-1])  # (batch_size,) each
			a_n, b_n = alpha_n(V[:, i-1]), beta_n(V[:, i-1])  # (batch_size,) each

			# M-current steady-state activation at previous voltage
			# V_half_M is (batch_size,) — each sample has its own half-activation
			p_ss = p_inf(V[:, i-1])  # (batch_size,)

			# Effective inverse membrane time constant: total conductance / C
			# M-current contributes gbar_M * p, which grows after each spike
			# and progressively slows the next spike → regular tonic intervals
			tau_V_inv = (
				(m[:, i-1] ** 3) * gbar_Na * h[:, i-1]   # (batch_size,) Na+ contribution
				+ (n[:, i-1] ** 4) * gbar_K               # (batch_size,) K+ delayed rectifier
				+ g_leak                                    # (batch_size,) passive leak
				+ gbar_M * p[:, i-1]                       # (batch_size,) M-current adaptation
			) / C  # (batch_size,)

			# Conductance-weighted reversal potential sum
			# With E_K = -77 mV (physiological), K+ drives are no longer anomalously deep
			V_inf = (
				(m[:, i-1] ** 3) * gbar_Na * h[:, i-1] * E_Na   # (batch_size,) Na+ drive → +53 mV
				+ (n[:, i-1] ** 4) * gbar_K * E_K                # (batch_size,) K+ drive → -77 mV
				+ g_leak * E_leak                                  # (batch_size,) leak drive
				+ gbar_M * p[:, i-1] * E_K                        # (batch_size,) M-current drive → -77 mV
				+ input_current[:, i-1]                            # (batch_size,) injected stimulus
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler voltage update (exact for piecewise-constant conductances)
			V[:, i] = V_inf + (V[:, i-1] - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Exponential Euler updates for standard HH gating variables
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# Exponential Euler update for M-current gate p
			# tau_p is (batch_size,), slow and voltage-independent
			# Extended range [5, ~405] ms allows posterior to find correct adaptation rate
			p[:, i] = p_ss + (p[:, i-1] - p_ss) * Exp(-tstep / tau_p)  # (batch_size,)

		# Return voltage trace; observation noise fixed at zero
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)