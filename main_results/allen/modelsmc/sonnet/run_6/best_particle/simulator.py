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
		Hodgkin-Huxley neuron extended with an M-type K+ current (I_KM).

		Physiological rationale for I_KM:
		  - Slow, non-inactivating K+ conductance (KCNQ/Kv7 family)
		  - Active at subthreshold and near-threshold voltages
		  - Provides spike-frequency adaptation → supports evenly-spaced tonic spiking
		  - Does NOT produce bursting, clustering, or sustained high-frequency firing

		Key fixes applied vs. base model:
		  1. E_K corrected to -90.0 mV (standard cortical HH value; -107.0 was unphysiological)
		  2. param_i, param_j extracted WITHOUT negation (inference supplies positive values)

		Args:
			init_voltage  : torch.Tensor : (batch_size,)              initial voltage (mV)
			input_current : torch.Tensor : (batch_size, time_steps)   applied current (uA/cm2)
			dt            : float                                      time step (ms)
			t             : torch.Tensor : (time_steps,)              time array (ms)
			params        : torch.Tensor : (batch_size, 10)           biophysical parameters
			seed          : int or None                                random seed

		Returns:
			V             : torch.Tensor : (batch_size, time_steps)   voltage traces (mV)
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

		# ── Parameter extraction ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2 — Na+ max conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2 — delayed-rectifier K+ max conductance
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2 — leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV     — leak reversal (sign applied)
		Vt        = -params[:, 4].float()  # (batch_size,) mV     — voltage threshold offset (sign applied)
		nois_fact = params[:, 5].float()   # (batch_size,)        — noise amplitude

		# X1 → M-type K+ current (slow adaptation, non-inactivating)
		gbar_M    = params[:, 6].float()   # (batch_size,) mS/cm2 — M-current max conductance [1e-4, 10]

		# X2 → unused (parsimony: single adaptation channel sufficient for tonic spiking)
		# gbar_X2 = params[:, 7].float()  # reserved, not implemented

		# param_i: half-activation voltage shift above Vt (mV), range [1e-4, 150]
		#   M-current typically activates ~25-45 mV above resting potential
		#   NO negation applied — inference engine provides positive values directly
		param_i   = params[:, 8].float()   # (batch_size,) mV — half-activation offset above Vt

		# param_j: M-current time constant (ms), range [1e-4, 3000]
		#   M-current is characteristically slow: tau ~ 50–500 ms in cortical neurons
		#   NO negation applied — inference engine provides positive values directly
		param_j   = params[:, 9].float()   # (batch_size,) ms — M-current activation time constant

		tstep = float(dt)  # ms

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (zero per task specification)
		C    = 1.0            # uF/cm² — membrane capacitance
		E_Na = 53.0           # mV     — Na+ reversal potential
		# FIX: corrected from -107.0 to -90.0 mV (physiological cortical value)
		# -107.0 caused excessively hyperpolarized resting potential and
		# distorted mean voltage, resting potential, skewness, and kurtosis statistics
		E_K  = -90.0          # mV     — K+ reversal potential (all K+ channels share this)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Numerically clipped exponential — prevents overflow for large negative inputs
			# z: any shape tensor
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable z / (exp(z) - 1) for HH alpha/beta rate functions
			# Uses first-order Taylor expansion near z=0 to avoid 0/0
			# z: any shape tensor
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ──────────────────────────────────────

		# Na+ activation gate m (fast)
		def alpha_m(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,) ms^-1

		def beta_m(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2     # (batch_size,) ms^-1

		# Na+ inactivation gate h
		def alpha_h(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)          # (batch_size,) ms^-1

		def beta_h(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))    # (batch_size,) ms^-1

		# Delayed-rectifier K+ activation gate n
		def alpha_n(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,) ms^-1

		def beta_n(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)            # (batch_size,) ms^-1

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,) ms
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → (batch_size,) dimensionless steady state
			return alpha / (alpha + beta)

		# ── M-type K+ current kinetics (I_KM) ────────────────────────────────
		# Single non-inactivating gating variable p
		# Steady state: sigmoid function centered at (Vt + param_i)
		#   - param_i ~ 25–45 mV places half-activation ~25–45 mV above threshold
		#   - slope factor 9 mV is physiologically standard for KCNQ channels
		# Time constant: constant value param_j (voltage-independent approximation)
		#   - param_j ~ 50–500 ms captures the characteristic slowness of M-current
		#   - Slow activation means it grows during a spike train → progressive adaptation

		def p_inf(x):
			# Steady-state M-current gate activation
			# x: (batch_size,) mV
			v_half = Vt + param_i   # (batch_size,) mV — tunable half-activation voltage
			return 1.0 / (1.0 + Exp(-(x - v_half) / 9.0))  # (batch_size,) dimensionless

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate

		# ── Initial conditions at steady state ───────────────────────────────
		V_init = init_voltage.to(device)            # (batch_size,) mV
		V[:, 0] = V_init                            # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                              # (batch_size,)

		# ── Simulation loop ───────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) mV
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH gate kinetics at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current steady state at V_prev
			p_inf_now = p_inf(V_prev)   # (batch_size,)

			# Effective membrane conductance sum: g_total / C
			# Units: mS/cm2 / (uF/cm2) = ms^-1
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev    # (batch_size,) Na+ contribution
				+ (n_prev ** 4) * gbar_K             # (batch_size,) delayed-rectifier K+
				+ g_leak                              # (batch_size,) leak
				+ p_prev * gbar_M                    # (batch_size,) M-current K+ adaptation
			) / C   # (batch_size,) ms^-1

			# Steady-state voltage numerator: sum(g_i * E_i) + I_inj + noise
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na+ drive
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) delayed-rectifier K+ drive
				+ g_leak * E_leak                           # (batch_size,) leak drive
				+ p_prev * gbar_M * E_K                    # (batch_size,) M-current K+ drive (E_K = -90 mV)
				+ input_current[:, i - 1]                  # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,) stochastic current noise, scaled by sqrt(dt)
			) / (tau_V_inv * C)   # (batch_size,) mV

			# Exponential Euler integration for voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler integration for standard HH gates
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)

			# Exponential Euler for M-current gate p (constant tau = param_j)
			# param_j directly encodes the slow M-current time constant in ms
			p[:, i] = p_inf_now + (p_prev - p_inf_now) * Exp(-tstep / param_j)   # (batch_size,)

		# Return voltage traces with optional observation noise (nois_fact_obs = 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)