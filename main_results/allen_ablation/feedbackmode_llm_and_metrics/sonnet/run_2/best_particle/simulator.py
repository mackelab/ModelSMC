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
		Hodgkin-Huxley neuron with slow M-type K+ current (I_Km).

		Physiological rationale for M-current addition:
		  Standard HH (Na+, K+, leak) tends to produce irregular inter-spike intervals
		  and poor resting-potential statistics when driven tonically. The M-current
		  (KCNQ/Kv7 channels) is a slowly activating, non-inactivating subthreshold
		  K+ conductance (~-35 mV half-activation) that provides slow hyperpolarising
		  drive after each action potential, promoting evenly-spaced tonic spiking
		  without inducing bursts or suppressing firing.

		Parameter slot allocation (strictly respected):
		  params[:,6]  = gbar_Km   – M-current max conductance (mS/cm², X1 slot)
		  params[:,7]  = gbar_X2   – unused (X2 conductance slot, kept for parsimony)
		  params[:,8]  = param_i   – raw value in [1e-4, 150]; mapped to V_half_km (mV)
		                             via V_half = param_i - 80  → range [-80, 70] mV
		                             M-current V_half typically near -35 mV ✓
		  params[:,9]  = param_j   – raw value in [1e-4, 3000]; mapped to tau_km (ms)
		                             via tau = param_j / 10  → range [0, 300] ms
		                             M-current tau typically 50-200 ms ✓

		Args:
		    init_voltage : torch.Tensor (batch_size,)          initial voltage (mV)
		    input_current: torch.Tensor (batch_size, time_steps) injected current (µA/cm²)
		    dt           : float                                 time step (ms)
		    t            : torch.Tensor (time_steps,)            time array (ms)
		    params       : torch.Tensor (batch_size, 10)         biophysical parameters
		    seed         : int or None                           random seed

		Returns:
		    V : torch.Tensor (batch_size, time_steps)            membrane voltage (mV)
		"""
		device = params.device

		# ── Random generator ────────────────────────────────────────────────────
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV (negated per convention)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV (negated per convention)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current parameters (X1 slot + both flexible params) ───────────────
		# gbar_Km: maximal M-current conductance; prior range [1e-4, 10] mS/cm²
		gbar_Km = params[:, 6].float()     # (batch_size,)  mS/cm²

		# params[:,7] (gbar_X2) intentionally unused – preserves parsimony

		# V_half_km: half-activation voltage of M-current gate
		# raw param_i in [1e-4, 150] (positive) → subtract 80 → [-80, 70] mV
		# M-current activates near -35 mV; inference explores this region freely
		V_half_km = params[:, 8].float() - 80.0   # (batch_size,)  mV

		# tau_km: M-current activation time constant (voltage-independent simplification)
		# raw param_j in [1e-4, 3000] → divide by 10 → [0, 300] ms; clamp for stability
		tau_km = (params[:, 9].float() / 10.0).clamp(min=5.0, max=300.0)  # (batch_size,)  ms

		tstep = float(dt)

		# ── Fixed constants ──────────────────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (currently disabled)
		C    = 1.0            # membrane capacitance, µF/cm²
		E_Na = 53.0           # Na+ reversal potential, mV
		E_K  = -107.0         # K+ reversal potential, mV (shared by Kdr and M-current)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; shape mirrors input
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)  # (...)

		def efun(z):
			# Bhaskara efun for HH rate denominators; shape mirrors input
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))  # (...)

		# ── Standard HH gate kinetics (unchanged from base) ─────────────────────
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
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):   # (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current gate kinetics ──────────────────────────────────────────────
		# Steady-state activation: sigmoidal, slope k=10 mV (canonical M-current)
		# Half-activation V_half_km is inferred via param_i mapping
		def p_inf_km(x):   # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_km) / 10.0))

		# Time constant: voltage-independent for simplicity; inferred via param_j mapping
		# Returns a (batch_size,) tensor broadcastable with state updates
		def get_tau_km():   # → (batch_size,)
			return tau_km

		# ── State variable tensors ───────────────────────────────────────────────
		V    = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m    = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h    = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n    = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p_km = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current gate

		# ── Steady-state initialisation ──────────────────────────────────────────
		V_init     = init_voltage.to(device)    # (batch_size,)
		V[:, 0]    = V_init                     # (batch_size,)
		m[:, 0]    = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0]    = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		n[:, 0]    = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		p_km[:, 0] = p_inf_km(V[:, 0])                           # (batch_size,)

		# ── Exponential Euler integration loop ──────────────────────────────────
		for i in range(1, time_steps):
			V_prev    = V[:, i - 1]       # (batch_size,)
			m_prev    = m[:, i - 1]       # (batch_size,)
			h_prev    = h[:, i - 1]       # (batch_size,)
			n_prev    = n[:, i - 1]       # (batch_size,)
			p_km_prev = p_km[:, i - 1]   # (batch_size,)

			# Standard HH alpha/beta rates
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) each

			# M-current gate steady state (same tau at all time points)
			p_km_ss = p_inf_km(V_prev)   # (batch_size,)
			tau_p   = get_tau_km()       # (batch_size,)

			# Effective conductance contributions
			g_Na_eff = (m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)
			g_K_eff  = (n_prev ** 4) * gbar_K             # (batch_size,)
			g_Km_eff = gbar_Km * p_km_prev                # (batch_size,) M-current

			# Inverse effective membrane time constant (sum of all conductances / C)
			tau_V_inv = (g_Na_eff + g_K_eff + g_leak + g_Km_eff) / C   # (batch_size,)

			# Stochastic noise sample
			noise = (
				nois_fact
				* torch.randn(batch_size, generator=generator, device=device)
				/ (tstep ** 0.5)
			)  # (batch_size,)

			# Effective steady-state voltage numerator
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff * E_K
				+ g_leak * E_leak
				+ g_Km_eff * E_K          # M-current pulls toward E_K (hyperpolarising)
				+ input_current[:, i - 1]
				+ noise
			) / (tau_V_inv * C)           # (batch_size,)

			# Exponential Euler updates (exact for linear V equation, approximate for gates)
			V[:, i]    = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			m[:, i]    = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i]    = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i]    = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			p_km[:, i] = p_km_ss + (p_km_prev - p_km_ss) * Exp(-tstep / tau_p)                           # (batch_size,)

		# ── Return voltage (+ optional observation noise, currently zero) ────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)