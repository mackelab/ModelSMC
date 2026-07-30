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
		Hodgkin-Huxley neuron simulator extended with an M-type K+ current (X1 slot).

		M-current (I_M) rationale:
		  - Slow, non-inactivating voltage-gated K+ current (KCNQ/Kv7 family)
		  - Activates near resting potential (-60 to -40 mV), providing spike-frequency adaptation
		  - Regularises inter-spike intervals → tonic, evenly-spaced spiking (no bursting)
		  - Well-documented in cortical and hippocampal neurons (Brown & Adams 1980; Mainen & Sejnowski 1996)
		  - param_i sets the half-activation voltage (V_half_M ≈ -35 mV typical)
		  - param_j sets the maximum time constant τ_max (slow, 100–1000 ms range)

		Args:
			init_voltage : torch.Tensor (batch_size,)       — initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) — injected current (μA/cm²)
			dt           : float                             — time step (ms)
			t            : torch.Tensor (time_steps,)        — time array (ms)
			params       : torch.Tensor (batch_size, 10)     — biophysical parameters
			seed         : int or None

		Returns:
			V            : torch.Tensor (batch_size, time_steps) — membrane potential (mV)
		"""
		device = params.device

		# ── Random generator ──────────────────────────────────────────────────────
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ── Parameter extraction ──────────────────────────────────────────────────
		gbar_Na  = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K   = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak   = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak   = -params[:, 3].float()  # (batch_size,)  mV  (sign applied)
		Vt       = -params[:, 4].float()  # (batch_size,)  mV  (sign applied)
		nois_fact = params[:, 5].float()  # (batch_size,)  unitless

		# X1 slot → M-type K+ current
		# gbar_M : slow K+ conductance (mS/cm²), range [1e-4, 10]
		# V_half_M: half-activation voltage (mV); param_i is negative → V_half = param_i ≈ -35 mV
		# tau_max_M: max time constant (ms); param_j is negative → tau_max = -param_j ≈ 100–1000 ms
		gbar_M    = params[:, 6].float()   # (batch_size,)  mS/cm²
		# X2 slot — unused in this iteration (parsimony principle)
		# gbar_X2 = params[:, 7].float()
		V_half_M  = -params[:, 8].float()  # (batch_size,)  mV  (negative, e.g. -35)
		tau_max_M = -params[:, 9].float()  # (batch_size,)  ms  (negative of positive param_j)

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Clamp to avoid overflow; shape: same as z
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Exponential function regularised near zero; shape: same as z
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics ──────────────────────────────────────────
		def alpha_m(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current kinetics (slow non-inactivating K+) ─────────────────────────
		# Steady-state activation: sigmoidal, half-activation at V_half_M (≈ -35 mV)
		# Time constant: bell-shaped around V_half_M; maximum = tau_max_M (ms)
		# Reference: Mainen & Sejnowski (1996), Wang (1998)
		def p_inf_M(x):
			# (batch_size,) → (batch_size,)
			# Sigmoid: p_inf = 1 / (1 + exp(-(V - V_half_M) / 10))
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		def tau_p_M(x):
			# (batch_size,) → (batch_size,)
			# Bell-shaped time constant centred on V_half_M
			# tau_p = tau_max_M / (3.3 * exp((V - V_half_M)/40) + exp(-(V - V_half_M)/40))
			dv = x - V_half_M
			return tau_max_M / (3.3 * Exp(dv / 40.0) + Exp(-dv / 40.0))

		def alpha_p(x):
			# (batch_size,) → (batch_size,)
			pinf  = p_inf_M(x)    # (batch_size,)
			tau_p = tau_p_M(x)    # (batch_size,)
			return pinf / tau_p

		def beta_p(x):
			# (batch_size,) → (batch_size,)
			pinf  = p_inf_M(x)    # (batch_size,)
			tau_p = tau_p_M(x)    # (batch_size,)
			return (1.0 - pinf) / tau_p

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  K DR gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na act gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na inact gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  M-current gate

		# ── Initialisation at steady state ───────────────────────────────────────
		V_init = init_voltage.to(device)          # (batch_size,)
		V[:, 0] = V_init                          # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		p[:, 0] = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))   # (batch_size,) — M-gate at rest

		# ── Time integration (exponential Euler) ─────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# Standard HH gate kinetics at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current gate kinetics at V_prev
			a_p, b_p = alpha_p(V_prev), beta_p(V_prev)   # (batch_size,), (batch_size,)

			# Current conductances
			g_Na_now  = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,)
			g_K_now   = (n[:, i - 1] ** 4) * gbar_K                    # (batch_size,)
			g_M_now   = gbar_M * p[:, i - 1]                           # (batch_size,)

			# Effective inverse membrane time constant (sum of conductances / C)
			tau_V_inv = (
				g_Na_now
				+ g_K_now
				+ g_leak
				+ g_M_now   # M-current contribution
			) / C   # (batch_size,)

			# Noise sample
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			# (batch_size,)

			# Steady-state voltage (numerator = sum of conductance-weighted reversals + inputs)
			V_inf = (
				g_Na_now  * E_Na
				+ g_K_now * E_K
				+ g_leak  * E_leak
				+ g_M_now * E_K   # M-current drives toward E_K (same as DR K+)
				+ input_current[:, i - 1]
				+ noise
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler update for voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Gating variable updates (exponential Euler)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
			# M-gate update (slow kinetics via alpha_p / beta_p)
			p[:, i] = inf_x(a_p, b_p) + (p[:, i - 1] - inf_x(a_p, b_p)) * Exp(-tstep / tau_x(a_p, b_p))
			# (batch_size,) for all gate updates

		# Return voltage trace (observation noise currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)