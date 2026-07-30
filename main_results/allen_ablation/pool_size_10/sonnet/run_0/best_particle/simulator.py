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
		Hodgkin-Huxley neuron simulator extended with a slow M-type (Kv7) potassium current.

		The M-current (IKM) is a non-inactivating, voltage-gated K+ current that activates
		slowly at subthreshold potentials. It is the canonical mechanism for spike-frequency
		adaptation and regularization of tonic firing. Its addition prevents burst-like
		dynamics while keeping inter-spike intervals regular.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial voltage
			input_current: torch.Tensor: (batch_size, time_steps) # input current
			dt: float # time step size
			t: torch.Tensor: (time_steps,) # time array
			params: torch.Tensor: (batch_size, 10) # parameters
			seed: optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # voltage traces
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
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (sign applied internally)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (sign applied internally)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current parameters (slot X1) ────────────────────────────────────
		# gbar_M  : maximal M-current conductance (mS/cm²), range [1e-4, 10]
		# V_half_M: half-activation voltage (mV); param_i is negated → range [-150, ~0]
		#           physiologically ~-35 to -60 mV → well within inference range
		# tau_M   : activation time constant (ms); param_j is negated → use |param_j|
		#           M-current is characteristically slow: 10 – 1000 ms
		gbar_M   = params[:, 6].float()           # (batch_size,)  mS/cm²
		V_half_M = -params[:, 8].float()          # (batch_size,)  mV  (already negative)
		tau_M    = params[:, 9].float().clamp(min=1.0)  # (batch_size,)  ms  (positive, clamped)

		# Unused slot – kept for completeness but not wired in
		# gbar_X2 = params[:, 7].float()

		tstep = float(dt)

		# ── Fixed biophysical constants ────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV

		# ── Numerical helpers ──────────────────────────────────────────────────
		def Exp(z):
			# Clamp to avoid overflow in exp
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),  # (batch_size,)
				torch.exp(z)                           # (batch_size,)
			)

		def efun(z):
			# Numerically stable (x / (e^x - 1)) used in HH rate functions
			return torch.where(
				torch.abs(z) < 1e-4,
				1 - z / 2,          # (batch_size,)
				z / (Exp(z) - 1)    # (batch_size,)
			)

		# ── Standard HH channel kinetics ──────────────────────────────────────
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

		def alpha_n(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (IKM) kinetics ───────────────────────────────────────────
		# Single-gate (p), no inactivation.
		# Boltzmann steady-state with slope k=10 mV (fixed, canonical value).
		# Time constant tau_M is slow and voltage-independent (simplified formulation).
		slope_M = 10.0  # mV, fixed activation slope

		def p_inf(x):   # (batch_size,) → (batch_size,)
			# Steady-state M-current activation: sigmoid centred at V_half_M
			return 1.0 / (1.0 + Exp((V_half_M - x) / slope_M))

		# ── State variable allocation ──────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate

		# ── Initialisation at steady state ────────────────────────────────────
		V_init = init_voltage.to(device)          # (batch_size,)
		V[:, 0] = V_init                           # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                             # (batch_size,)

		# ── Simulation loop ────────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)
			m_prev = m[:, i - 1]  # (batch_size,)
			h_prev = h[:, i - 1]  # (batch_size,)
			n_prev = n[:, i - 1]  # (batch_size,)
			p_prev = p[:, i - 1]  # (batch_size,)

			# Standard HH gate rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,) each

			# M-current steady state at current voltage
			p_ss = p_inf(V_prev)  # (batch_size,)

			# ── Effective membrane conductance (denominator term) ──────────────
			# Includes Na, K (delayed-rectifier), leak, and M-current contributions
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)  INa conductance
				+ (n_prev ** 4) * gbar_K            # (batch_size,)  IK conductance
				+ g_leak                             # (batch_size,)  leak conductance
				+ gbar_M * p_prev                   # (batch_size,)  IM conductance
			) / C  # (batch_size,)

			# ── Voltage steady-state numerator ────────────────────────────────
			noise_term = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)
				+ g_leak * E_leak                           # (batch_size,)
				+ gbar_M * p_prev * E_K                    # (batch_size,)  IM drives toward E_K
				+ input_current[:, i - 1]                  # (batch_size,)
				+ noise_term                               # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# ── Exponential integration (exact for linear system) ─────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Standard HH gates – exponential Euler
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)

			# M-current gate – exponential Euler with slow time constant tau_M
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_M)  # (batch_size,)

		# ── Return with optional observation noise (currently 0) ──────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)