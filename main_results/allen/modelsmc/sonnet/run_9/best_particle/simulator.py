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
		Hodgkin-Huxley neuron extended with a slow M-type potassium current (I_M).

		Physiological rationale for I_M:
		  - Slowly-activating, non-inactivating K⁺ conductance (KCNQ/Kv7 family)
		  - Active at sub-threshold and peri-threshold voltages (-60 to -20 mV)
		  - Provides spike-frequency adaptation → regularises tonic firing ISIs
		  - Corrects mean voltage, variance, skewness during stimulation
		  - Does NOT produce burst firing or sustained high-frequency activity
		  - Well-characterised in cortical and hippocampal neurons

		Parameter mapping (using X1 slot only for parsimony):
		  gbar_X1  → gbar_M   : M-current maximal conductance (mS/cm²)
		  param_i  = -params[:,8]: used directly as V_half_M (mV), range ~ [-150, 0]
		                           inference will find values near -35 mV
		  param_j  = -params[:,9]: negated to give positive tau_max_M (ms), range [1, 3000]
		                           inference will find values ~ 50-500 ms

		Args:
			init_voltage : torch.Tensor (batch_size,)              initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps)   injected current (µA/cm²)
			dt           : float                                    time step (ms)
			t            : torch.Tensor (time_steps,)              time array (ms)
			params       : torch.Tensor (batch_size, 10)           biophysical parameters
			seed         : int or None

		Returns:
			V            : torch.Tensor (batch_size, time_steps)   voltage traces (mV)
		"""
		device = params.device

		# ── random generator ──────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── extract base parameters ───────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV
		Vt        = -params[:, 4].float()  # (batch_size,)  mV
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── X1 slot: M-current conductance ───────────────────────────────────
		gbar_M    = params[:, 6].float()   # (batch_size,)  mS/cm²  [1e-4, 10]

		# X2 slot: unused (parsimony — one well-characterised channel is sufficient)
		# gbar_X2 = params[:, 7].float()  # reserved, not implemented

		# param_i = -params[:,8]: raw inference value is positive [1e-4,150], negated
		#   → V_half_M lives in (-150, 0) mV; physiological M-current half-activation
		#     is typically -35 to -50 mV, well within this range
		V_half_M  = -params[:, 8].float()  # (batch_size,)  mV

		# param_j = -params[:,9]: raw inference value is positive [1e-4,3000], negated
		#   → negate back to recover positive time constant; clamp to ≥ 1 ms for stability
		tau_max_M = torch.clamp(-params[:, 9].float(), min=1.0)  # (batch_size,)  ms

		tstep = float(dt)

		# ── fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # µF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (M-current uses same reversal as delayed rectifier K⁺)

		# ── numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# (batch_size,) — numerically stable exponential
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (batch_size,) — standard HH auxiliary function
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── standard HH channel kinetics ─────────────────────────────────────
		def alpha_m(x):   # (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):    # (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):   # (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):    # (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):   # (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40)

		def tau_x(alpha, beta):   # (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,)
			return alpha / (alpha + beta)

		# ===== BEGIN EDITABLE SECTION =====
		# M-current kinetics (I_M = gbar_M * p * (V - E_K))
		#
		# Steady-state activation: sigmoid centred at V_half_M, slope ~10 mV
		#   p_inf(V) = 1 / (1 + exp(-(V - V_half_M) / 10))
		# Voltage-dependent time constant: bell-shaped (Brown & Adams 1980 form)
		#   tau_p(V)  = tau_max_M / (3.3*exp((V-V_half_M)/20) + exp(-(V-V_half_M)/20))
		#   clamped ≥ 0.1 ms to prevent numerical divergence in exponential Euler

		def p_inf(x):
			# (batch_size,) — steady-state M-gate activation
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		def tau_p(x):
			# (batch_size,) — time constant in ms; bell-shaped about V_half_M
			denom = 3.3 * Exp((x - V_half_M) / 20.0) + Exp(-(x - V_half_M) / 20.0)
			return torch.clamp(tau_max_M / denom, min=0.1)
		# ===== END EDITABLE SECTION =====

		# ── allocate state arrays ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		# ===== BEGIN EDITABLE SECTION =====
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate
		# ===== END EDITABLE SECTION =====

		# ── initialise gates at steady state ─────────────────────────────────
		V_init   = init_voltage.to(device)                                   # (batch_size,)
		V[:, 0]  = V_init                                                    # (batch_size,)
		m[:, 0]  = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                 # (batch_size,)
		h[:, 0]  = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                 # (batch_size,)
		n[:, 0]  = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                 # (batch_size,)
		# ===== BEGIN EDITABLE SECTION =====
		p[:, 0]  = p_inf(V[:, 0])                                            # (batch_size,)
		# ===== END EDITABLE SECTION =====

		# ── simulation loop ───────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)

			# standard HH gate alpha/beta
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,)
			# ===== BEGIN EDITABLE SECTION =====
			# M-gate quantities at previous time step
			p_prev    = p[:, i - 1]          # (batch_size,)
			p_ss      = p_inf(V_prev)         # (batch_size,)
			tau_p_val = tau_p(V_prev)         # (batch_size,)  ms
			# ===== END EDITABLE SECTION =====

			# ── effective membrane conductance (exponential Euler denominator) ─
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na current   (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K  current   (batch_size,)
				+ g_leak                             # leak current (batch_size,)
				# ===== BEGIN EDITABLE SECTION =====
				+ gbar_M * p_prev                   # M  current   (batch_size,)
				# ===== END EDITABLE SECTION =====
			) / C                                   # (batch_size,)

			# ── noise ─────────────────────────────────────────────────────────
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# ── voltage steady state ──────────────────────────────────────────
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na term  (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # K  term  (batch_size,)
				+ g_leak * E_leak                          # leak term(batch_size,)
				# ===== BEGIN EDITABLE SECTION =====
				+ gbar_M * p_prev * E_K                   # M  term  (batch_size,)
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                 # I_inj    (batch_size,)
				+ noise                                    # stochastic(batch_size,)
			) / (tau_V_inv * C)                           # (batch_size,)

			# ── exponential Euler updates ─────────────────────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                             # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m)) # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h)) # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n)) # (batch_size,)
			# ===== BEGIN EDITABLE SECTION =====
			# M-gate: exponential Euler with direct steady-state / time-constant form
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_p_val)                              # (batch_size,)
			# ===== END EDITABLE SECTION =====

		# ── optional observation noise (currently zero) ───────────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)