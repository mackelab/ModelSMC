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
		Hodgkin-Huxley neuron extended with a transient A-type K⁺ current (I_A).

		Physiological rationale for A-current:
		  - Transient, rapidly activating and slowly inactivating K⁺ current (Kv4 family)
		  - Activates near spike threshold, inactivates during depolarization
		  - Controls the delay to first spike and inter-spike interval regularity
		  - Produces evenly-spaced tonic spiking WITHOUT bursting or spike suppression
		  - Unlike M-current, I_A does not suppress spikes—it shapes their timing
		  - Well-characterised in cortical and hippocampal neurons (Schoppa & Westbrook 1999)

		Implementation:
		  - Activation gate 'a': treated as instantaneous (Boltzmann steady-state)
		    Half-activation anchored to Vt + 1 mV, slope 8.5 mV
		  - Inactivation gate 'b': slow, first-order kinetics
		    Half-inactivation anchored to Vt - 56 mV, slope 8.5 mV
		    Time constant tau_b = params[:,7] (inferred, clamped ≥ 1 ms)
		  - I_A = gbar_A * a_inf(V) * b * (V - E_K)

		Parameter assignment (X1 slot only; X2 unused for parsimony):
		  params[:,6] = gbar_A  : A-current maximal conductance (mS/cm²), range [1e-4, 10]
		  params[:,7] = tau_b   : inactivation time constant (ms),         range [1e-4, 3000]
		  params[:,8], params[:,9]: unused (X2 slot reserved)

		Args:
			init_voltage : torch.Tensor (batch_size,)            -- initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) -- injected current (μA/cm²)
			dt           : float                                  -- time step (ms)
			t            : torch.Tensor (time_steps,)             -- time array (ms)
			params       : torch.Tensor (batch_size, 10)          -- biophysical parameters
			seed         : int or None                            -- random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps) -- membrane voltage traces (mV)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]         # int

		# ── Base HH parameters ────────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (negated: sampler draws |E_leak|)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (negated: sampler draws |Vt|)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless noise scale

		# ── X1 slot: A-type K⁺ current ───────────────────────────────────────────
		# gbar_A: maximal A-conductance   (mS/cm²), range [1e-4, 10]
		# tau_b : inactivation time const (ms),     range [1e-4, 3000], clamped ≥ 1 ms
		#
		# Half-voltages anchored to inferred Vt for physiological self-consistency:
		#   V_half_a = Vt + 1   mV  (activation just above resting threshold)
		#   V_half_b = Vt - 56  mV  (inactivation well below threshold, ~-80 mV)
		# Slope factors: 8.5 mV (standard for Kv4/I_A channels)
		gbar_A = params[:, 6].float()   # (batch_size,)  mS/cm²
		tau_b  = params[:, 7].float()   # (batch_size,)  ms

		# X2 slot intentionally unused — parsimony; A-current alone addresses
		# inter-spike interval regularity in tonic spiking without bursting risk
		# params[:,8] and params[:,9] are not used

		tstep = float(dt)

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (zero per task specification)
		C    = 1.0            # uF/cm²  membrane capacitance
		E_Na = 53.0           # mV      sodium reversal potential
		E_K  = -107.0         # mV      potassium reversal (shared by Kdr and A-current)

		# A-current slope factor (fixed, standard for Kv4 channels)
		k_A  = 8.5            # mV  slope for both activation and inactivation

		# Clamp tau_b: minimum 1 ms to avoid numerical instability
		# A-current inactivation is typically 10–200 ms in cortical neurons
		tau_b_safe = torch.clamp(tau_b, min=1.0)  # (batch_size,)

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# z: any shape → same shape; clamped exponential for overflow safety
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# z: any shape → same shape; linearise near z=0
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gate kinetics ─────────────────────────────────────────────
		def alpha_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── A-current gate kinetics ───────────────────────────────────────────────
		# Activation a: instantaneous Boltzmann
		#   a_inf = 1 / (1 + exp(-(V - V_half_a) / k_A))
		#   V_half_a = Vt + 1 mV  (just above resting threshold)
		def a_inf_A(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - (Vt + 1.0)) / k_A))

		# Inactivation b: slow first-order kinetics
		#   b_inf = 1 / (1 + exp((V - V_half_b) / k_A))  [note: reversed sign for inactivation]
		#   V_half_b = Vt - 56 mV  (well hyperpolarized, channel de-inactivates at rest)
		def b_inf_A(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((x - (Vt - 56.0)) / k_A))

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		b = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)  A-inactivation

		# ── Initialisation at steady state ───────────────────────────────────────
		V_init  = init_voltage.to(device)                           # (batch_size,)
		V[:, 0] = V_init                                             # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))          # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))          # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))          # (batch_size,)
		b[:, 0] = b_inf_A(V[:, 0])                                   # (batch_size,)  A-inactivation at SS

		# ── Time integration loop (exponential Euler) ─────────────────────────────
		for i in range(1, time_steps):

			V_prev = V[:, i - 1]  # (batch_size,)

			# Standard HH gate rates
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# A-current: instantaneous activation and slow inactivation steady state
			a_ss = a_inf_A(V_prev)   # (batch_size,)  activation (instantaneous)
			b_ss = b_inf_A(V_prev)   # (batch_size,)  inactivation target

			# Effective A-current conductance: gbar_A * a_inf(V) * b(t)
			# I_A = gbar_A * a_ss * b[:,i-1] * (V - E_K)
			# Effective conductance term for tau_V_inv = gbar_A * a_ss * b
			g_A_eff = gbar_A * a_ss * b[:, i - 1]  # (batch_size,)

			# Effective membrane conductance sum (exponential Euler denominator)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # Na    (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                  # Kdr   (batch_size,)
				+ g_leak                                         # leak  (batch_size,)
				+ g_A_eff                                        # I_A   (batch_size,)
			) / C  # (batch_size,)

			# Weighted reversal sum + injected current + noise
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # Na    (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # Kdr   (batch_size,)
				+ g_leak * E_leak                                        # leak  (batch_size,)
				+ g_A_eff * E_K                                          # I_A   (batch_size,) reversal = E_K
				+ input_current[:, i - 1]                               # Iext  (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler voltage update
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Standard HH gate updates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# A-current inactivation update: exponential Euler with inferred tau_b_safe
			# Slow recovery de-inactivates the channel between spikes → ISI regularity
			b[:, i] = b_ss + (b[:, i-1] - b_ss) * Exp(-tstep / tau_b_safe)  # (batch_size,)

		# Return voltage with optional observation noise (currently 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)