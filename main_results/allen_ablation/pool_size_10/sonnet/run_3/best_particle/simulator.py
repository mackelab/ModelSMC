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
		Hodgkin-Huxley neuron extended with M-type K+ current (IM / KM).

		Physiological rationale for IM:
		  - Slow, non-inactivating K+ current active near spike threshold (~-35 mV)
		  - Provides spike-frequency adaptation, regularising tonic inter-spike intervals
		  - Does NOT produce bursting or high-frequency sustained firing
		  - Single first-order gate 'p' with Boltzmann steady-state and constant τ_p
		  - Reversal potential shared with K-DR (E_K = -107 mV)

		Key fixes vs. previous iteration:
		  - param_i and param_j are NO LONGER negated (positive raw values used directly)
		  - IM occupies params[:,6] (gbar_KM), params[:,8] (V_half offset), params[:,9] (tau_p)
		  - params[:,7] is explicitly reserved/unused (noted for future Ih channel if needed)

		Args:
			init_voltage: torch.Tensor (batch_size,)   -- initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) -- injected current (µA/cm²)
			dt: float                                  -- time step (ms)
			t: torch.Tensor (time_steps,)              -- time array (ms)
			params: torch.Tensor (batch_size, 10)      -- biophysical parameters
			seed: int or None                          -- optional random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps)   -- membrane voltage traces (mV)
		"""
		device = params.device

		# ── Random generator ────────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size  = params.shape[0]   # int
		time_steps  = t.shape[0]        # int
		tstep       = float(dt)

		# ── Base HH parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()    # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()    # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()    # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()   # (batch_size,) mV  (negation applied here)
		Vt        = -params[:, 4].float()   # (batch_size,) mV  (negation applied here)
		nois_fact = params[:, 5].float()    # (batch_size,) unitless

		# ── M-current parameters ────────────────────────────────────────────────
		# X1 conductance slot: gbar_KM in [1e-4, 10] mS/cm²
		gbar_KM   = params[:, 6].float()    # (batch_size,) mS/cm²

		# params[:,7] (gbar_X2) intentionally unused in this iteration;
		# reserved for a second channel (e.g. Ih) if further discrepancies remain.

		# param_i in [1e-4, 150]: used as half-activation voltage offset (NO negation)
		#   V_half_p = param_i - 75 maps raw range [0,150] → physiological [-75, +75] mV
		#   IM typically activates around -35 mV  →  posterior should find param_i ≈ 40
		param_i   = params[:, 8].float()    # (batch_size,) positive, NOT negated

		# param_j in [1e-4, 3000]: M-current time constant in ms (NO negation)
		#   IM τ_p is typically 50–500 ms; posterior free to explore this range
		param_j   = params[:, 9].float()    # (batch_size,) positive, NOT negated

		# Fixed biophysical constants
		nois_fact_obs = 0.0
		C    = 1.0      # uF/cm²
		E_Na = 53.0     # mV
		E_K  = -107.0   # mV  (shared by K-DR and IM)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Numerically safe exponential; z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable evaluation of z/(exp(z)-1); z: any shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH gating kinetics ─────────────────────────────────────────
		def alpha_m(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):     # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):     # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):     # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,),(batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,),(batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (IM) kinetics ──────────────────────────────────────────────
		# Half-activation voltage derived from param_i (positive, not negated):
		#   V_half_p = param_i - 75  maps [0,150] → [-75, 75] mV
		#   Fixed slope k = 10 mV (standard value for IM Boltzmann; Brown & Adams 1980)
		V_half_p = param_i - 75.0   # (batch_size,) mV

		def p_inf(x):    # x: (batch_size,) → (batch_size,)
			# Boltzmann steady-state for M-current gate p
			# Activates with depolarisation (positive slope Boltzmann)
			return 1.0 / (1.0 + Exp(-(x - V_half_p) / 10.0))

		# Time constant: param_j directly in ms (voltage-independent for parsimony)
		# Posterior can find the appropriate value in [1e-4, 3000] ms
		tau_p = param_j   # (batch_size,) ms

		# ── State variable allocation ────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IM gate

		# ── Steady-state initialisation ──────────────────────────────────────────
		V_init  = init_voltage.to(device)                           # (batch_size,)
		V[:, 0] = V_init
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))         # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))         # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))         # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                    # (batch_size,) IM at rest

		# ── Exponential Euler integration loop ───────────────────────────────────
		for i in range(1, time_steps):

			V_prev = V[:, i - 1]   # (batch_size,)

			# Gating kinetics at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current gate steady state at previous voltage
			p_ss = p_inf(V_prev)   # (batch_size,)

			# Effective membrane conductance (sum of all active conductances / C)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # Na    (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                  # K-DR  (batch_size,)
				+ g_leak                                         # leak  (batch_size,)
				+ gbar_KM * p[:, i - 1]                         # IM    (batch_size,)
			) / C   # (batch_size,)

			# Voltage steady state (weighted sum of reversal potentials + inputs)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na    # Na term   (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                    # K-DR term (batch_size,)
				+ g_leak * E_leak                                         # leak term (batch_size,)
				+ gbar_KM * p[:, i - 1] * E_K                           # IM term   (batch_size,)
				+ input_current[:, i - 1]                                # I_inj     (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,)

			# Voltage update (exponential Euler)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                    # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			# M-current gate: slow first-order relaxation toward p_ss with constant tau_p
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_p)   # (batch_size,)

		# Return voltage traces (observation noise disabled: nois_fact_obs = 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)