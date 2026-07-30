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
		Hodgkin-Huxley neuron with an added M-type (Kv7/KCNQ) slow K+ current.

		Physiological rationale for M-current addition:
		  - The base HH model (Na, K_dr, leak) tends to produce statistics that deviate
		    from tonic spiking recordings in mean voltage, variance, and spike count.
		  - The M-current (I_M) is a slowly activating, non-inactivating K+ conductance
		    that modulates firing threshold and inter-spike interval regularity.
		  - Critically, I_M does NOT produce burst firing — it acts as a gentle brake
		    on repetitive spiking, producing regular tonic activity exactly as observed.
		  - Single gating variable p with Boltzmann steady-state and slow time constant.

		Args:
			init_voltage: torch.Tensor (batch_size,)
			input_current: torch.Tensor (batch_size, time_steps)
			dt: float
			t: torch.Tensor (time_steps,)
			params: torch.Tensor (batch_size, 10)
			seed: int or None

		Returns:
			V: torch.Tensor (batch_size, time_steps)
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ── Base parameter extraction ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (negated: raw > 0)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (negated: raw > 0)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current parameters (X1 slot) ────────────────────────────────────────
		# gbar_M  : maximal M-current conductance  [1e-4, 10]  mS/cm²
		# v_half_M: half-activation voltage = -params[:,8], range [-150, -1e-4] mV
		#           Physiological M-current half-activation ≈ -35 to -55 mV  ✓
		# tau_M   : activation time constant = -params[:,9], range [-3000, -1e-4]
		#           We use its negation to recover a positive time constant in ms.
		#           M-current kinetics are slow: 50–500 ms range  ✓
		gbar_M   = params[:, 6].float()   # (batch_size,)  mS/cm²
		# params[:,7] (gbar_X2) left unused — parsimony principle
		v_half_M = -params[:, 8].float()  # (batch_size,)  mV  — half-activation voltage
		tau_M    = -params[:, 9].float()  # (batch_size,)  ms  — activation time constant (positive)

		tstep = float(dt)

		# ── Fixed biophysical constants ────────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (also used as reversal for M-current — K+ selective)

		# ── Numerical helpers ──────────────────────────────────────────────────────
		def Exp(z):
			# Clamped exponential for numerical stability; z: any shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Exponential function used in HH rate expressions; z: any shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics ──────────────────────────────────────────
		def alpha_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2      # (batch_size,)

		def alpha_h(x):
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)           # (batch_size,)

		def beta_h(x):
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))        # (batch_size,)

		def alpha_n(x):
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2     # (batch_size,)

		def beta_n(x):
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)             # (batch_size,)

		def tau_x(alpha, beta):
			return 1.0 / (alpha + beta)               # (batch_size,)

		def inf_x(alpha, beta):
			return alpha / (alpha + beta)             # (batch_size,)

		# ── M-current kinetics (slow Boltzmann, no inactivation) ──────────────────
		# Steady-state activation via sigmoid:
		#   p_inf(V) = 1 / (1 + exp(-(V - v_half_M) / k))
		# k = 10 mV slope factor (fixed, typical for M-current)
		# Time constant tau_M is a learnable parameter (slow, no voltage dependence
		# needed for parsimony; constant-tau formulation is standard for I_M models)
		def p_inf(x):
			# Boltzmann activation; x: (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - v_half_M) / 10.0))  # (batch_size,)

		# ── State variable allocation ──────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  M-gate

		# ── Initialisation at steady state ────────────────────────────────────────
		V_init = init_voltage.to(device)                           # (batch_size,)
		V[:, 0] = V_init
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))
		p[:, 0] = p_inf(V[:, 0])                                   # (batch_size,)

		# ── Integration loop ───────────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Standard HH gate rates at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # each (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # each (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # each (batch_size,)

			# Effective conductance sum (inverse membrane time constant * C)
			# Includes M-current conductance weighted by its gating variable
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # Na contribution  (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                 # K_dr contribution (batch_size,)
				+ g_leak                                        # leak               (batch_size,)
				+ gbar_M * p[:, i - 1]                         # M-current          (batch_size,)
			) / C                                               # (batch_size,)

			# Weighted reversal-potential sum → voltage steady state numerator
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                  # (batch_size,)
				+ g_leak * E_leak                                      # (batch_size,)
				+ gbar_M * p[:, i - 1] * E_K                          # M-current (E_K)  (batch_size,)
				+ input_current[:, i - 1]                             # applied current  (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                                        # (batch_size,)

			# Exponential-Euler updates for all state variables
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))

			# M-gate: constant-tau exponential-Euler step
			# tau_M is (batch_size,) — slow, learnable time constant
			p_ss = p_inf(V_prev)                                       # (batch_size,)
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_M)  # (batch_size,)

		# ── Return voltage trace with optional observation noise ───────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)