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
		Hodgkin-Huxley neuron extended with a single A-type (fast-inactivating) K+ current
		(I_KA) in slot X1, using both flexible parameter slots for one well-characterised channel.

		Physiological rationale for I_KA:
		  Standard HH models often over-count or under-space spikes because there is no
		  transient outward current to modulate the inter-spike interval (ISI).  I_KA:
		    - Activates rapidly near spike threshold (Boltzmann, half-activation ~Vt+14 mV)
		    - Inactivates at depolarised potentials and de-inactivates during hyperpolarisation
		    - Provides a transient braking current immediately after each spike, lengthening
		      the ISI and regularising tonic spiking without inducing bursting
		    - Does NOT generate burst firing, sustained high-frequency firing, or quiescence

		Parameter slot usage (correct X1=(6,7), flexible=(8,9) convention):
		  params[:,6] = gbar_KA   — I_KA peak conductance (mS/cm²), range [1e-4, 10]
		  params[:,7] = gbar_X2   — unused this iteration (kept for forward compatibility)
		  params[:,8] = |param_i| — magnitude of inactivation half-voltage shift (mV),
		                            applied as negative shift: v_shift = -|param_i| ∈ (-150,0)
		                            moves inactivation curve to more hyperpolarised voltages,
		                            controlling how much channel is available at rest
		  params[:,9] = |param_j| — inactivation tau multiplier: tau_scale = |param_j|/500 ≥ 0
		                            slows the inactivation time constant without changing
		                            steady-state, controlling ISI duration

		Design decisions vs. prior iterations:
		  - Activation gate `a` uses Boltzmann inf + Gaussian tau instead of efun/alpha-beta
		    form, so it is completely decoupled from Na+ m-gate kinetics and does NOT
		    compete with spike initiation
		  - Only ONE channel added this iteration (parsimony; avoids parameter
		    identifiability collapse that occurred when two channels were added simultaneously)
		  - v_shift applied as subtraction in inactivation kinetics so negative values
		    consistently shift curve leftward (to hyperpolarised voltages)

		Args:
			init_voltage  : torch.Tensor (batch_size,)             — initial voltage (mV)
			input_current : torch.Tensor (batch_size, time_steps)  — injected current (μA/cm²)
			dt            : float                                   — time step (ms)
			t             : torch.Tensor (time_steps,)             — time array (ms)
			params        : torch.Tensor (batch_size, 10)          — biophysical parameters
			seed          : int or None

		Returns:
			V             : torch.Tensor (batch_size, time_steps)  — voltage traces (mV)
		"""
		device = params.device

		# ── random generator ─────────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── parameter extraction (strict slot convention) ────────────────────────
		# Base HH parameters
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV   (sign applied here)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV   (sign applied here)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# Slot X1 → I_KA (A-type fast-inactivating K+)
		# gbar_KA: peak conductance for A-type channel; SBI explores [1e-4, 10] mS/cm²
		gbar_KA  = params[:, 6].float()                       # (batch_size,)  mS/cm²

		# Slot X2 → unused this iteration; kept to preserve parameter vector structure
		# _gbar_X2 = params[:, 7]  — not connected to any current

		# Flexible parameters — both assigned to I_KA for a single well-identified channel
		# v_shift: inactivation half-voltage shift.  params[:,8] ∈ [1e-4,150] mV (positive);
		#          negated here so v_shift ∈ (-150, 0) mV, shifting the inactivation curve
		#          toward hyperpolarised voltages (more channel available at rest).
		v_shift   = -params[:, 8].float()                     # (batch_size,)  mV  ∈ (-150, 0)

		# tau_scale: dimensionless multiplier that slows inactivation time constant.
		# params[:,9] ∈ [1e-4, 3000]; divided by 500 → tau_scale ∈ [~0, 6].
		# At tau_scale=0: standard kinetics; at tau_scale=2: 3× slower inactivation.
		tau_scale = params[:, 9].float() / 500.0              # (batch_size,)  ≥ 0

		tstep = float(dt)

		# ── fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # μF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (shared reversal for K+ DR and I_KA)

		# ── numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# (batch_size,) → (batch_size,)  ;  clamp prevents exp overflow
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z),
			)

		def efun(z):
			# (batch_size,) → (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── HH Na+ channel kinetics ───────────────────────────────────────────────
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

		# ── HH K+ delayed-rectifier kinetics ─────────────────────────────────────
		def alpha_n(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		# ── shared gate helpers ───────────────────────────────────────────────────
		def tau_x(alpha, beta):   # each (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # each (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── I_KA: A-type K+ channel kinetics ─────────────────────────────────────
		#
		# Activation gate `a_gate`  (fast, Boltzmann + Gaussian tau):
		#   Using a Boltzmann steady-state entirely decouples the A-type activation
		#   from the Na+ m-gate efun form, so the two channels do not compete at
		#   exactly the same voltage.  The Gaussian tau is fast (1-5 ms) but
		#   bell-shaped around the half-activation voltage, giving smooth dynamics.
		#
		#   a_inf(V) = 1 / (1 + exp(-(V - Vt - 14) / 16))
		#     Half-activation at Vt + 14 mV (≈ just below spike threshold)
		#     Width 16 mV gives gradual, physiologically realistic activation curve
		#
		#   tau_a(V) = 1 + 4 * exp(-((V - Vt - 14) / 30)^2)  [ms]
		#     Peak tau ~5 ms at half-activation; decays to ~1 ms away from threshold.
		#     Keeps channel fast while avoiding instantaneous jumps.
		#
		def a_inf(x):     # (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - Vt - 14.0) / 16.0))

		def tau_a(x):     # (batch_size,) → (batch_size,)
			z = (x - Vt - 14.0) / 30.0          # (batch_size,)
			return 1.0 + 4.0 * Exp(-(z * z))     # (batch_size,)  ms

		# Inactivation gate `b_gate`  (slow, HH h-gate form, shifted + scaled):
		#   Same functional form as the HH sodium h-gate for numerical consistency.
		#   v_shift (∈ (-150, 0) mV) is SUBTRACTED from the reference voltages,
		#   shifting the inactivation curve toward hyperpolarised potentials, so
		#   SBI can infer how much channel is available at the resting potential.
		#
		#   alpha_b(V) ~ exp(-(V - Vt - 17 - v_shift) / 18)   [mirror of alpha_h]
		#   beta_b(V)  ~ 1 / (1 + exp(-0.2*(V - Vt - 40 - v_shift)))  [mirror of beta_h]
		#
		#   Note on sign: v_shift < 0, so (V - Vt - 17 - v_shift) > (V - Vt - 17),
		#   making alpha_b smaller → b_inf smaller at a given V → more resting inactivation.
		#   This is the physiologically correct direction: shifting leftward = more inactivation.
		#
		#   tau_b_eff = tau_b * (1 + tau_scale)
		#   tau_scale ≥ 0 slows the gate without touching its steady-state.
		#
		def alpha_b(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0 - v_shift          # v_shift<0 → v1 is larger → alpha_b smaller
			return 0.128 * Exp(-v1 / 18.0)

		def beta_b(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0 - v_shift
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def b_ss(x):      # (batch_size,) → (batch_size,)  steady-state inactivation
			ab = alpha_b(x)    # (batch_size,)
			bb = beta_b(x)     # (batch_size,)
			return inf_x(ab, bb)

		def tau_b_eff(x):  # (batch_size,) → (batch_size,)  scaled inactivation time constant
			ab = alpha_b(x)    # (batch_size,)
			bb = beta_b(x)     # (batch_size,)
			return tau_x(ab, bb) * (1.0 + tau_scale)  # (batch_size,)  ms

		# ── state variable allocation ─────────────────────────────────────────────
		V      = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n      = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m      = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h      = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		a_gate = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) I_KA activation
		b_gate = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) I_KA inactivation

		# ── initialisation at steady-state ────────────────────────────────────────
		V_init       = init_voltage.to(device)                            # (batch_size,)
		V[:, 0]      = V_init                                              # (batch_size,)
		n[:, 0]      = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))           # (batch_size,)
		m[:, 0]      = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))           # (batch_size,)
		h[:, 0]      = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))           # (batch_size,)
		a_gate[:, 0] = a_inf(V[:, 0])                                      # (batch_size,)
		b_gate[:, 0] = b_ss(V[:, 0])                                       # (batch_size,)

		# ── simulation loop ───────────────────────────────────────────────────────
		for i in range(1, time_steps):
			Vprev  = V[:, i - 1]        # (batch_size,)
			m_prev = m[:, i - 1]        # (batch_size,)
			h_prev = h[:, i - 1]        # (batch_size,)
			n_prev = n[:, i - 1]        # (batch_size,)
			a_prev = a_gate[:, i - 1]   # (batch_size,)
			b_prev = b_gate[:, i - 1]   # (batch_size,)

			# HH gate rates at previous voltage
			a_m, b_m = alpha_m(Vprev), beta_m(Vprev)   # each (batch_size,)
			a_h, b_h = alpha_h(Vprev), beta_h(Vprev)   # each (batch_size,)
			a_n, b_n = alpha_n(Vprev), beta_n(Vprev)   # each (batch_size,)

			# I_KA gate steady-states and time constants at previous voltage
			a_ss_prev  = a_inf(Vprev)      # (batch_size,)
			ta_prev    = tau_a(Vprev)      # (batch_size,)  ms
			b_ss_prev  = b_ss(Vprev)       # (batch_size,)
			tb_prev    = tau_b_eff(Vprev)  # (batch_size,)  ms (scaled)

			# I_KA instantaneous conductance: g_KA = gbar_KA * a^3 * b
			# Using a^3 (vs. n^4 for DR K+) gives slightly less cooperative opening,
			# appropriate for a different K+ channel subtype (Kv4 family).
			g_KA_now = (a_prev ** 3) * b_prev * gbar_KA   # (batch_size,)  mS/cm²

			# Effective membrane conductance Σg / C  (exponential-Euler denominator)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ contribution      (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K+ DR contribution    (batch_size,)
				+ g_leak                             # leak contribution     (batch_size,)
				+ g_KA_now                           # I_KA contribution     (batch_size,)
			) / C   # (batch_size,)

			# Noise sample (Euler-Maruyama scaling by 1/sqrt(dt))
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# Voltage steady-state numerator: Σ(g * E_rev) + I_inj + noise
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K              # (batch_size,)
				+ g_leak * E_leak                            # (batch_size,)
				+ g_KA_now * E_K                             # I_KA reversal = E_K  (batch_size,)
				+ input_current[:, i - 1]                   # (batch_size,)
				+ noise                                      # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential-Euler update (exact for piecewise-linear I-V)
			V[:, i] = V_inf + (Vprev - V_inf) * Exp(-tstep * tau_V_inv)       # (batch_size,)

			# HH gate exponential-Euler updates
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# I_KA gate exponential-Euler updates
			a_gate[:, i] = a_ss_prev + (a_prev - a_ss_prev) * Exp(-tstep / ta_prev)   # (batch_size,)
			b_gate[:, i] = b_ss_prev + (b_prev - b_ss_prev) * Exp(-tstep / tb_prev)   # (batch_size,)

		# ── return with optional observation noise ────────────────────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)