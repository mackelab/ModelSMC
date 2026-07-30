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
		Hodgkin-Huxley neuron with one added M-type (Kv7/KCNQ) slow K+ current.

		Physiological rationale:
		  - M-current (I_M) is a non-inactivating, voltage-gated K+ current
		  - Activates slowly at subthreshold voltages → regularises ISI for tonic firing
		  - Does NOT produce bursting; is in fact anti-burst
		  - References: Brown & Adams 1980, Halliwell & Adams 1982

		Design changes vs prior iterations:
		  1. Remove forced-negative constraint on params[:,8]:
		       Prior: param_i = -params[:,8]  (forced negative, range [-150, 0))
		       Now:   voff_M  = params[:,8]   (positive, [1e-4, 150])
		       This allows the half-activation voltage (Vt + voff_M) to be placed
		       anywhere from just above Vt to Vt+150, letting inference explore
		       the full physiologically relevant range without a sign ambiguity.
		  2. Use alpha/beta formulation (as in best-performing iteration NLE=24.4)
		       rather than explicit sigmoid + bell-shaped tau (which degraded to NLE=29.9).
		       The alpha/beta efun formulation naturally provides:
		         - smooth sigmoid p_inf
		         - bell-shaped tau_p via 1/(alpha+beta)
		       while keeping the same code style as the base HH gates.
		  3. tau_scale_M uses params[:,9] directly (positive, clamped [1,200] ms)
		       rather than negating and dividing, matching the natural prior range.

		M-current gating variable 'p':
		  dp/dt = (p_inf(V) - p) / tau_p(V)
		  p_inf(V) = alpha_p / (alpha_p + beta_p)
		  tau_p(V) = tau_scale_M / (alpha_p + beta_p)
		  alpha_p, beta_p: efun-based, half-activation at Vt + voff_M

		Parameters:
		  gbar_M  = params[:,6] : maximal M conductance [1e-4, 10] mS/cm2
		  params[:,7]           : unused (parsimony — one channel is sufficient)
		  voff_M  = params[:,8] : half-activation offset above Vt [1e-4, 150] mV
		  tau_scale_M = clamp(params[:,9]/15, 1, 200): time-constant scale [ms]

		Args:
		    init_voltage : (batch_size,)              initial membrane voltage (mV)
		    input_current: (batch_size, time_steps)   injected current (uA/cm2)
		    dt           : float                      time step (ms)
		    t            : (time_steps,)              time array (ms)
		    params       : (batch_size, 10)           biophysical parameters
		    seed         : int or None

		Returns:
		    V            : (batch_size, time_steps)   simulated voltage traces (mV)
		"""
		device = params.device

		# ── Random generator setup ────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ────────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm2
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm2
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm2
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV
		Vt        = -params[:, 4].float()  # (batch_size,)  mV
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current parameters (X1 slot only; X2 unused for parsimony) ─────────
		gbar_M = params[:, 6].float()      # (batch_size,)  mS/cm2, range [1e-4, 10]

		# Half-activation offset — KEY CHANGE from prior iterations:
		# Use params[:,8] directly as a POSITIVE offset above Vt.
		# Previously forced negative (param_i = -params[:,8]) which restricted
		# half-activation to lie BELOW Vt+35; now it can lie anywhere in [Vt, Vt+150].
		# Physiological M-current typically activates near Vt+30 to Vt+40 mV,
		# so inference will naturally converge to params[:,8] ≈ 30–40.
		voff_M = params[:, 8].float()      # (batch_size,)  mV, positive [1e-4, 150]

		# Time-constant scale — use params[:,9] directly (positive [1e-4, 3000]):
		# clamp to [1, 200] ms to keep I_M slow (consistent with Kv7 kinetics).
		# Dividing by 15 maps the [0, 3000] range to [0, 200] ms.
		tau_scale_M = torch.clamp(
			params[:, 9].float() / 15.0, min=1.0, max=200.0
		)  # (batch_size,)

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (disabled)
		C    = 1.0            # uF/cm²
		E_Na = 53.0           # mV  reversal for Na+
		E_K  = -107.0         # mV  reversal for K+ (and M-current, which is K+-selective)

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; z: any broadcastable tensor shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# z / (exp(z) - 1), regularised near z=0 via Taylor expansion
			# z: any broadcastable tensor shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ──────────────────────────────────────────
		def alpha_m(x):
			# x: (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (I_M) kinetics — alpha/beta formulation ────────────────────
		# Using efun-based rates as in the base HH gates (best-performing pattern).
		# Half-activation at (Vt + voff_M), where voff_M = params[:,8] > 0.
		# This is equivalent to a sigmoidal p_inf and bell-shaped tau_p, but
		# parameterised via rates which the exponential Euler solver handles exactly.
		#
		# Rate constants (1e-3 prefactor) give tau_p on the order of ~50–500 ms
		# before tau_scale_M scaling, consistent with Kv7 channels.
		#
		# The efun formulation guarantees alpha_p, beta_p > 0 for all voltages,
		# which ensures p_inf ∈ (0,1) and tau_p > 0 at all times.

		def alpha_p(x):
			# Forward rate for M-gate; x: (batch_size,)
			# Half-activation at Vt + voff_M
			v1 = x - Vt - voff_M          # (batch_size,)  zero at half-activation
			return 1e-3 * efun(-0.1 * v1) / 0.1   # (batch_size,)

		def beta_p(x):
			# Backward rate for M-gate; x: (batch_size,)
			v1 = x - Vt - voff_M          # (batch_size,)
			return 1e-3 * Exp(-v1 / 80.0) # (batch_size,)
			# Asymmetric beta (Exp rather than efun) gives a steeper inactivation
			# at hyperpolarised potentials, matching observed Kv7 deactivation kinetics

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-gate

		# ── Steady-state initialisation ───────────────────────────────────────────
		V_init  = init_voltage.to(device)                                  # (batch_size,)
		V[:, 0] = V_init                                                   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                # (batch_size,)
		p[:, 0] = inf_x(alpha_p(V[:, 0]), beta_p(V[:, 0]))                # (batch_size,)

		# ── Exponential Euler time-stepping loop ──────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH gate rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) each

			# M-gate rates at previous voltage
			a_p, b_p = alpha_p(V_prev), beta_p(V_prev)   # (batch_size,) each

			# Effective time-constant for M-gate, scaled by tau_scale_M:
			#   tau_p = tau_scale_M / (a_p + b_p)
			# This separates kinetic speed (tau_scale_M) from steady-state shape (a_p/b_p)
			tau_p_eff = tau_scale_M / (a_p + b_p)         # (batch_size,)

			# M-current steady state at current voltage
			p_inf_now = a_p / (a_p + b_p)                 # (batch_size,), in (0,1)

			# ── Effective membrane conductance (denominator of exponential Euler V) ─
			# Each term: g_i (mS/cm2) for each active current channel
			# I_M = gbar_M * p * (V - E_K) → contributes gbar_M * p to tau_V_inv
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)  I_Na
				+ (n_prev ** 4) * gbar_K            # (batch_size,)  I_K
				+ g_leak                            # (batch_size,)  I_leak
				+ p_prev * gbar_M                   # (batch_size,)  I_M
			) / C                                   # (batch_size,)

			# ── Effective voltage steady-state (numerator / denominator * C) ───────
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)                      # (batch_size,)

			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)  I_Na driving force
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)  I_K  driving force
				+ g_leak * E_leak                          # (batch_size,)  leak driving force
				+ p_prev * gbar_M * E_K                    # (batch_size,)  I_M  driving force
				+ input_current[:, i - 1]                  # (batch_size,)  injected current
				+ noise                                    # (batch_size,)  stochastic input
			) / (tau_V_inv * C)                            # (batch_size,)

			# ── State variable updates (exponential Euler, exact for linear ODEs) ──
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)
			# (batch_size,)

			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			# (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
			# (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			# (batch_size,)

			# M-gate: exponential Euler with scaled time constant
			# p(t+dt) = p_inf + (p(t) - p_inf) * exp(-dt / tau_p_eff)
			p[:, i] = p_inf_now + (p_prev - p_inf_now) * Exp(-tstep / tau_p_eff)
			# (batch_size,)

		# ── Return voltage (observation noise currently zero) ─────────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)