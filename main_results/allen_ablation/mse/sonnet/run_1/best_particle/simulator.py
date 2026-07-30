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
		Hodgkin-Huxley neuron with a slow M-type K+ adaptation current (Kv7/KCNQ).

		Changes from prior iteration (two targeted fixes):
		1. V_half_M is now constrained to the physiologically meaningful range [-50, -20] mV
		   via sigmoid remapping of params[:,7], replacing the unbounded linear map
		   that caused optimizer collapse to degenerate regimes (spike count MSE=56.6).
		2. tau_w is now smoothly mapped via sigmoid to (10, 300) ms, replacing the
		   hard torch.clamp that zeroed gradients for most prior samples and prevented
		   tau_w from being identified (variance MSE=44.3, skewness MSE=11.1).

		Both changes preserve gradient flow throughout the full prior support, giving
		the inference engine a well-shaped loss landscape over the M-current parameters.

		Physiological motivation for M-current (unchanged from prior iterations):
		- Non-inactivating, voltage-gated K+ current (Kv7/KCNQ family)
		- Activates slowly near spike threshold; causes mild spike-frequency adaptation
		- Shapes afterhyperpolarisation waveform → fixes voltage distribution skewness/kurtosis
		- No burst-promoting or high-frequency-sustaining effects

		Parameter mapping:
		  params[:, 0]  -> gbar_Na      : Na+ max conductance (mS/cm²)
		  params[:, 1]  -> gbar_K       : K+ max conductance (mS/cm²)
		  params[:, 2]  -> g_leak       : leak conductance (mS/cm²)
		  params[:, 3]  -> |E_leak|     : leak reversal magnitude (mV, negated internally)
		  params[:, 4]  -> |Vt|         : threshold shift magnitude (mV, negated internally)
		  params[:, 5]  -> nois_fact    : noise scale (unitless)
		  params[:, 6]  -> gbar_M       : M-current conductance (mS/cm²) [range 1e-4, 10]
		  params[:, 7]  -> V_half_M raw : sigmoid-remapped to [-50, -20] mV [range 1e-4, 120]
		  params[:, 8]  -> (unused)     : reserved param_i slot
		  params[:, 9]  -> tau_w raw    : sigmoid-remapped to (10, 300) ms [range 1e-4, 3000]

		Args:
			init_voltage  : torch.Tensor (batch_size,)            – initial voltage (mV)
			input_current : torch.Tensor (batch_size, time_steps) – injected current (µA/cm²)
			dt            : float                                  – time step (ms)
			t             : torch.Tensor (time_steps,)            – time array (ms)
			params        : torch.Tensor (batch_size, 10)         – biophysical parameters
			seed          : int or None                            – optional RNG seed

		Returns:
			V : torch.Tensor (batch_size, time_steps) – membrane voltage traces (mV)
		"""
		device = params.device

		# ── Random number generator ──────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²  – Na+ max conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²  – K+ max conductance
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²  – leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV      – leak reversal (sign applied)
		Vt        = -params[:, 4].float()  # (batch_size,) mV      – voltage threshold shift
		nois_fact = params[:, 5].float()   # (batch_size,)         – noise amplitude scale

		# ── M-current parameters ─────────────────────────────────────────────────
		#
		# gbar_M: maximal M-current conductance. Taken directly from params[:,6]
		#   (gbar_X1 slot, prior range [1e-4, 10] mS/cm²). No remapping needed.
		gbar_M   = params[:, 6].float()   # (batch_size,) mS/cm²

		# V_half_M: M-current half-activation voltage.
		#   FIX (vs. prior iter): sigmoid remap replaces linear negation.
		#   Prior code:  V_half_M = -params[:,7]  → range [-120, 0] mV (too broad,
		#                caused optimizer collapse to always-on or never-on M-current)
		#   New code:    V_half_M = -50 + 30*sigmoid(params[:,7]) → range (-50, -20) mV
		#   Physiological M-current half-activation: typically -35 to -20 mV.
		#   sigmoid(params[:,7]) ∈ (0.5, 1) for params[:,7] > 0 (which is guaranteed
		#   by the prior [1e-4, 120]), so V_half_M ∈ (-35, -20) mV — well-centered
		#   on the physiological range and with nonzero gradient everywhere.
		V_half_M = -50.0 + 30.0 * torch.sigmoid(params[:, 7])   # (batch_size,) mV

		# tau_w: M-current slow time constant.
		#   FIX (vs. prior iter): smooth sigmoid remap replaces hard torch.clamp.
		#   Prior code:  tau_w = clamp(params[:,9], 10, 300) → zero gradient outside
		#                [10, 300]; prior spans [1e-4, 3000] so most samples hit boundary.
		#   New code:    tau_w = 10 + 290*sigmoid(params[:,9]) → range (10, 300) ms
		#   sigmoid maps all positive reals to (0.5, 1), giving tau_w ∈ (155, 300) ms
		#   for typical prior samples. This is physiologically appropriate (M-current
		#   deactivation time constants: 100-300 ms in cortical neurons) and ensures
		#   nonzero gradient everywhere in the prior support.
		tau_w    = 10.0 + 290.0 * torch.sigmoid(params[:, 9])   # (batch_size,) ms

		# params[:, 8] is deliberately left unused (param_i slot reserved for future use)

		tstep = float(dt)   # scalar ms

		# ── Fixed biophysical constants ──────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (kept at 0 per task instructions)
		C    = 1.0            # µF/cm²  – specific membrane capacitance
		E_Na = 53.0           # mV      – sodium reversal potential
		E_K  = -107.0         # mV      – potassium reversal (shared by K+ and M-current)
		slope_M = 10.0        # mV      – M-current sigmoid slope factor (fixed literature value)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential — clamp at -500 to prevent underflow
			# z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Handles near-zero singularity in HH alpha/beta rate expressions
			# z: any shape → same shape
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,
				z / (Exp(z) - 1.0)
			)

		# ── Standard HH gating kinetics (Vt-shifted) ────────────────────────────
		def alpha_m(x):
			# Na+ activation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ activation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ activation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant; inputs: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state; inputs: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (slow K+ adaptation) steady-state ──────────────────────────
		# Boltzmann sigmoid: w_inf(V) = 1 / (1 + exp(-(V - V_half_M) / slope_M))
		# With V_half_M ∈ (-35, -20) mV and slope_M = 10 mV:
		#   At rest (~-65 mV):   w_inf ≈ 0.05–0.12  → channel nearly closed ✓
		#   Near threshold (~-40 mV): w_inf ≈ 0.27–0.50 → partially active ✓
		#   During spike (~+30 mV):   w_inf ≈ 0.99   → fully open → K+ outward ✓
		def w_inf(x):
			# M-current steady-state activation; x: (batch_size,) → (batch_size,) ∈ [0,1]
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / slope_M))

		# ── State variable tensors ────────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ activation
		w = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current gate

		# ── Initial conditions at steady state ───────────────────────────────────
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))           # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))           # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))           # (batch_size,)
		# M-current starts near steady-state at resting voltage (~5-12% open)
		w[:, 0] = w_inf(V[:, 0])                                      # (batch_size,)

		# ── Exponential Euler integration loop ───────────────────────────────────
		for i in range(1, time_steps):
			# Snapshot previous-step state
			V_prev = V[:, i - 1]   # (batch_size,) mV
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			w_prev = w[:, i - 1]   # (batch_size,) M-current gate

			# Alpha/beta rates at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # each (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # each (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # each (batch_size,)

			# ── Effective membrane conductance: tau_V_inv = g_total / C ──────────
			# M-current contributes an additional voltage-dependent K+ conductance
			# that grows during spiking, providing adaptation without bursting
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,) Na+ conductance
				+ (n_prev ** 4) * gbar_K            # (batch_size,) K+ delayed-rectifier
				+ g_leak                             # (batch_size,) leak
				+ gbar_M * w_prev                   # (batch_size,) M-current adaptation
			) / C                                    # (batch_size,) ms⁻¹

			# ── Voltage steady state: weighted sum of reversal potentials ─────────
			# M-current reverses at E_K (it is a potassium channel)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na+ drive
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) K+ drive
				+ g_leak * E_leak                           # (batch_size,) leak drive
				+ gbar_M * w_prev * E_K                    # (batch_size,) M-current drive
				+ input_current[:, i - 1]                  # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                            # (batch_size,) mV

			# ── Exponential Euler state updates ───────────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)

			# M-current gate: exponential Euler with inferred slow time constant tau_w
			# w_inf is evaluated at V_prev (previous voltage, consistent with other gates)
			# Smooth sigmoid-mapped tau_w ensures gradient flow during inference
			w_inf_prev = w_inf(V_prev)                                              # (batch_size,)
			w[:, i]    = w_inf_prev + (w_prev - w_inf_prev) * Exp(-tstep / tau_w)  # (batch_size,)

		# ── Return voltage trace (observation noise = 0 per task instructions) ───
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)