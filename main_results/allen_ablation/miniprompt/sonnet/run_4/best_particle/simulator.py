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
		Hodgkin-Huxley neuron with M-type K+ (IM) and A-type K+ (IA, Connor-Stevens) channels.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage in mV
			input_current: torch.Tensor: (batch_size, time_steps) # stimulation current uA/cm2
			dt: float # time step in ms
			t: torch.Tensor: (time_steps,) # time array in ms
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: int or None # optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces in mV
		"""
		device = params.device

		# Random generator setup
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar
		time_steps = t.shape[0]        # scalar

		# ── Parameter extraction ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (stored as |val|, negated)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (stored as |val|, negated)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# IM: slow non-inactivating K+, spike-frequency adaptation; gbar in [1e-4, 10] mS/cm2
		gbar_IM   = params[:, 6].float()   # (batch_size,) mS/cm2

		# IA: fast transient K+, early repolarization; gbar in [1e-4, 120] mS/cm2
		gbar_IA   = params[:, 7].float()   # (batch_size,) mS/cm2

		# param_i: stored as positive [1e-4, 150]; negated to give voltage midpoint in [-150, 0] mV
		param_i   = -params[:, 8].float()  # (batch_size,) mV, in range [-150, ~0]

		# param_j: stored as positive [1e-4, 3000]; negated here, so -param_j > 0 for τ scaling
		param_j   = -params[:, 9].float()  # (batch_size,) negative; (-param_j) positive

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV  (shared by DR, IM, IA)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# z: any shape -> same shape; overflow-safe
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# z: any shape -> same shape; L'Hopital safe near z=0
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ── Standard HH Na+/K+ kinetics ───────────────────────────────────────
		def alpha_m(x):
			v1 = x - Vt - 13.0                          # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25       # (batch_size,)

		def beta_m(x):
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2          # (batch_size,)

		def alpha_h(x):
			v1 = x - Vt - 17.0                          # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)              # (batch_size,)

		def beta_h(x):
			v1 = x - Vt - 40.0                          # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))         # (batch_size,)

		def alpha_n(x):
			v1 = x - Vt - 15.0                          # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2        # (batch_size,)

		def beta_n(x):
			v1 = x - Vt - 10.0                          # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)                # (batch_size,)

		# ── M-current (IM) p-gate kinetics ───────────────────────────────────
		# Slow non-inactivating K+; prefactor 3.3e-3 gives physiological τ_p (~50-500 ms)
		# param_i shifts activation midpoint; param_j scales the time constant
		def alpha_p(x):
			v1 = x - param_i                             # (batch_size,) ; param_i ~ -35 mV
			return 3.3e-3 * efun(-0.2 * v1) / 0.2       # (batch_size,)

		def beta_p(x):
			v1 = x - param_i                             # (batch_size,)
			return 3.3e-3 * efun(0.2 * v1) / 0.2        # (batch_size,)

		# ── A-type K+ (IA) gate kinetics — Connor-Stevens formulation ─────────
		# Fast transient K+: a⁴ activation, b inactivation
		# Voltage offsets relative to Vt chosen to capture sub-threshold activation
		def alpha_a(x):
			v1 = x - Vt - 18.0                          # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25       # (batch_size,)

		def beta_a(x):
			v1 = x - Vt - 30.0                          # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2          # (batch_size,)

		def alpha_b(x):
			v1 = x - Vt - 5.0                           # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)              # (batch_size,)

		def beta_b(x):
			v1 = x - Vt - 27.0                          # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))         # (batch_size,)

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) DR K+ activation
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IM activation
		a = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA activation
		b = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA inactivation

		# ── Initialisation at t = 0 ───────────────────────────────────────────
		V_init  = init_voltage.to(device)                               # (batch_size,)
		V[:, 0] = V_init                                                # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))             # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))             # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))             # (batch_size,)
		p[:, 0] = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))             # (batch_size,)
		a[:, 0] = inf_x(alpha_a(V[:, 0]), beta_a(V[:, 0]))             # (batch_size,)
		b[:, 0] = inf_x(alpha_b(V[:, 0]), beta_b(V[:, 0]))             # (batch_size,)

		# ── Exponential Euler simulation loop ─────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Gate rate constants at previous voltage: each (batch_size,)
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)
			a_p, b_p = alpha_p(V_prev), beta_p(V_prev)   # (batch_size,), (batch_size,)
			a_a, b_a = alpha_a(V_prev), beta_a(V_prev)   # (batch_size,), (batch_size,)
			a_b, b_b = alpha_b(V_prev), beta_b(V_prev)   # (batch_size,), (batch_size,)

			# Effective conductances at previous state: (batch_size,)
			g_Na_eff = (m[:, i-1] ** 3) * gbar_Na * h[:, i-1]   # (batch_size,)
			g_K_eff  = (n[:, i-1] ** 4) * gbar_K                  # (batch_size,)
			g_IM_eff = p[:, i-1] * gbar_IM                         # (batch_size,)
			g_IA_eff = (a[:, i-1] ** 4) * gbar_IA * b[:, i-1]    # (batch_size,) Connor-Stevens a^4·b

			# Inverse membrane time constant: (batch_size,)
			tau_V_inv = (
				g_Na_eff
				+ g_K_eff
				+ g_leak
				+ g_IM_eff
				+ g_IA_eff
			) / C  # (batch_size,)

			# Noise sample: (batch_size,)
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)

			# Voltage steady-state numerator: (batch_size,)
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff  * E_K
				+ g_leak   * E_leak
				+ g_IM_eff * E_K
				+ g_IA_eff * E_K
				+ input_current[:, i-1]
				+ noise
			) / (tau_V_inv * C)  # (batch_size,)

			# Voltage update — exponential Euler: (batch_size,)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)

			# Standard gate updates — exponential Euler: (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))

			# IM p-gate: slow τ, clamped scale factor to prevent numerical freeze: (batch_size,)
			tau_p_scaled = tau_x(a_p, b_p) * torch.clamp(-param_j, min=1.0, max=500.0)  # (batch_size,)
			p[:, i] = inf_x(a_p, b_p) + (p[:, i-1] - inf_x(a_p, b_p)) * Exp(-tstep / tau_p_scaled)

			# IA gates: fast activation (a) and inactivation (b): (batch_size,)
			a[:, i] = inf_x(a_a, b_a) + (a[:, i-1] - inf_x(a_a, b_a)) * Exp(-tstep / tau_x(a_a, b_a))
			b[:, i] = inf_x(a_b, b_b) + (b[:, i-1] - inf_x(a_b, b_b)) * Exp(-tstep / tau_x(a_b, b_b))

		# Return voltage with optional observation noise: (batch_size, time_steps)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)