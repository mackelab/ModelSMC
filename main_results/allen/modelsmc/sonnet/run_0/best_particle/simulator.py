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
		Hodgkin-Huxley neuron extended with an M-type (KCNQ/Kv7) potassium current.

		Physiological rationale for M-current addition:
		  - M-current is a slow, non-inactivating, sub-threshold K+ current
		  - It regulates inter-spike intervals and produces mild spike-frequency adaptation
		  - It shapes resting potential, mean voltage during stimulation, and voltage
		    distribution statistics (variance, skewness, kurtosis) WITHOUT causing bursting
		  - Half-activation typically near -35 mV, tuned here via param_i

		Args:
			init_voltage: torch.Tensor: (batch_size,)
			input_current: torch.Tensor: (batch_size, time_steps)
			dt: float
			t: torch.Tensor: (time_steps,)
			params: torch.Tensor: (batch_size, 10)
			seed: int or None

		Returns:
			V: torch.Tensor: (batch_size, time_steps)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ── Base parameter extraction ──────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (sign applied here)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (sign applied here)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current (X1 slot) ────────────────────────────────────────────────
		# gbar_M:  slow K+ conductance tuned by inference, range [1e-4, 10] mS/cm²
		# V_half_M: half-activation voltage (negated → negative mV), range [-150, ~0] mV
		#           typical physiological value: ~-35 mV
		# param_j is not used here to avoid identifiability issues with a single channel
		gbar_M   = params[:, 6].float()   # (batch_size,)  mS/cm²  — M-current conductance
		V_half_M = -params[:, 8].float()  # (batch_size,)  mV      — M-current half-activation voltage

		# gbar_X2 / param_j left unused: parsimony principle, one channel is sufficient
		# gbar_X2 = params[:, 7].float()
		# param_j = -params[:, 9].float()

		tstep = float(dt)

		# ── Fixed biophysical constants ────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (also reversal for M-current)

		# ── Numerical helpers ──────────────────────────────────────────────────
		def Exp(z):
			# Clamped exponential for numerical stability
			# z: arbitrary shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Exponential function regularised near z=0
			# z: arbitrary shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH gating kinetics ───────────────────────────────────────
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
			return 4.0 / (1 + Exp(-0.2 * v1))

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

		# ── M-current (KCNQ/Kv7) gating kinetics ─────────────────────────────
		# Single activation gate w, no inactivation.
		# Steady state: sigmoid centred at V_half_M with slope k=10 mV
		# Time constant: bell-shaped voltage dependence (standard M-current form)
		#   - Slow near V_half_M (~10–50 ms), faster at extreme voltages
		#   - Prevents bursting; promotes regular tonic spiking

		def w_inf(x):
			# Steady-state M-current activation
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		def tau_w_fn(x):
			# Voltage-dependent M-current time constant (ms)
			# Bell-shaped, centred at V_half_M; epsilon guards against division by zero
			# x: (batch_size,) → (batch_size,)
			denom = Exp((x - V_half_M) / 40.0) + Exp(-(x - V_half_M) / 20.0)  # (batch_size,)
			return 1.0 / (3.3 * denom + 1e-7)  # (batch_size,)

		# ── State variable allocation ──────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		w = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) — M-current gate

		# ── Initialisation at steady state ────────────────────────────────────
		V_init = init_voltage.to(device)    # (batch_size,)
		V[:, 0] = V_init                    # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		w[:, 0] = w_inf(V[:, 0])                             # (batch_size,)

		# ── Time-stepping loop ─────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)
			m_prev = m[:, i - 1]  # (batch_size,)
			h_prev = h[:, i - 1]  # (batch_size,)
			n_prev = n[:, i - 1]  # (batch_size,)
			w_prev = w[:, i - 1]  # (batch_size,)

			# Standard HH alpha/beta rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# M-current steady state and time constant at current voltage
			w_ss  = w_inf(V_prev)      # (batch_size,)
			tau_w = tau_w_fn(V_prev)   # (batch_size,)

			# Effective membrane conductance (inverse RC time constant)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na  contribution  (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K   contribution  (batch_size,)
				+ g_leak                            # leak contribution (batch_size,)
				+ w_prev * gbar_M                   # M   contribution  (batch_size,)
			) / C                                   # (batch_size,)

			# Voltage steady state (weighted reversal potentials + input)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na  drive  (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # K   drive  (batch_size,)
				+ g_leak * E_leak                           # leak drive (batch_size,)
				+ w_prev * gbar_M * E_K                     # M   drive (same reversal as K⁺) (batch_size,)
				+ input_current[:, i - 1]                   # injected current (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                            # (batch_size,)

			# Exponential-Euler update for membrane voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Exponential-Euler updates for standard HH gating variables
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)

			# Exponential-Euler update for M-current gate
			w[:, i] = w_ss + (w_prev - w_ss) * Exp(-tstep / tau_w)  # (batch_size,)

		# Return voltage traces with (currently zero) observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)