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
		Hodgkin-Huxley neuron extended with M-type K+ current (IKm) for tonic spiking.

		The M-current (Kv7/KCNQ channels) is a slow, non-inactivating K+ current that:
		  - Activates near resting potential (~-35 mV half-activation)
		  - Produces mild spike-frequency adaptation without bursting
		  - Regularises inter-spike intervals → consistent with observed tonic spiking
		  - Does NOT produce bursting, quiescence, or high-frequency clusters

		Parameter slots used:
		  X1 slot: gbar_Km (params[:,6]), V_half_Km (-params[:,8]), tau_max_Km (params[:,9])
		  X2 slot: intentionally unused (parsimony principle — one channel is sufficient)

		Args:
			init_voltage: torch.Tensor: (batch_size,)              # initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps)  # injected current (uA/cm2)
			dt: float                                              # time step (ms)
			t: torch.Tensor: (time_steps,)                         # time array (ms)
			params: torch.Tensor: (batch_size, 10)                 # biophysical parameters
			seed: int or None                                      # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)  # membrane voltage traces (mV)
		"""
		device = params.device

		# Random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # int
		time_steps = t.shape[0]       # int

		# ── Base HH parameters ───────────────────────────────────────────────────
		gbar_Na  = params[:, 0].float()   # (batch_size,)  Na+ max conductance      (mS/cm2)
		gbar_K   = params[:, 1].float()   # (batch_size,)  K+  max conductance      (mS/cm2)
		g_leak   = params[:, 2].float()   # (batch_size,)  leak conductance         (mS/cm2)
		E_leak   = -params[:, 3].float()  # (batch_size,)  leak reversal potential  (mV, negative)
		Vt       = -params[:, 4].float()  # (batch_size,)  voltage threshold offset (mV, negative)
		nois_fact = params[:, 5].float()  # (batch_size,)  noise amplitude          (unitless)

		# ── M-current (IKm) parameters — X1 slot only ────────────────────────────
		# Physiological rationale for M-current:
		#   The M-current (carried by Kv7/KCNQ channels) is ubiquitous in cortical
		#   and hippocampal neurons. It activates slowly below spike threshold (~-35 mV)
		#   and does not inactivate. Its primary effect is to reduce excitability and
		#   regularise firing — producing the evenly-spaced tonic spiking pattern
		#   observed in the data, without generating bursting or high-frequency episodes.
		#
		# Design: two inferred parameters fill X1 slot (params[:,6], params[:,8], params[:,9])
		#   gbar_Km   : maximal M-current conductance        (mS/cm2), range [1e-4, 10]
		#   V_half_Km : half-activation voltage              (mV, negative), derived from -params[:,8]
		#                 stored as positive [1e-4, 150] → applied as negative → range [-150, ~0]
		#                 physiologically ~-35 mV for M-current
		#   tau_max_Km: peak time constant at V_half         (ms, positive), from params[:,9]
		#                 range [1e-4, 3000] ms; M-current tau typically 20–300 ms
		#
		# params[:,7] (gbar_X2) intentionally unused — parsimony, one channel suffices
		gbar_Km   = params[:, 6].float()   # (batch_size,)  IKm max conductance     (mS/cm2)
		# params[:, 7] unused — X2 conductance slot left inactive for parsimony
		V_half_Km = -params[:, 8].float()  # (batch_size,)  IKm half-activation     (mV, negative)
		tau_max_Km = params[:, 9].float()  # (batch_size,)  IKm peak time constant  (ms, positive)

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (zero — keep noise model unchanged)
		C    = 1.0            # membrane capacitance (uF/cm2)
		E_Na = 53.0           # Na+ reversal potential (mV)
		E_K  = -107.0         # K+  reversal potential (mV); IKm also uses E_K

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential — clips extreme negative values
			# z: any shape → same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# z/(exp(z)-1), regularised near z=0 via Taylor expansion
			# z: any shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ─────────────────────────────────────────
		def alpha_m(x):
			# Na+ activation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ deactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ deinactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ activation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ deactivation rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant; alpha, beta: (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state; alpha, beta: (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current gating kinetics ─────────────────────────────────────────────
		# Activation variable p (no inactivation — defining feature of M-current)
		#
		# p_inf(V) = 1 / (1 + exp(-(V - V_half_Km) / 10))
		#   Sigmoid with 10 mV slope; standard Kv7 activation slope from literature
		#   (Wang & McKinnon 1995, Bhattacharjee & Bhattacharjee 2003)
		#
		# tau_p(V) = tau_max_Km / (cosh((V - V_half_Km) / 20) + eps)
		#   Bell-shaped time constant peaking at V_half_Km; 20 mV half-width
		#   Epsilon (1e-7) prevents division by zero at extreme voltages

		def p_inf_fn(x):
			# M-current steady-state activation; x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_Km) / 10.0))

		def tau_p_fn(x):
			# M-current voltage-dependent time constant (ms); x: (batch_size,) → (batch_size,)
			return tau_max_Km / (torch.cosh((x - V_half_Km) / 20.0) + 1e-7)

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) voltage (mV)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+  activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IKm activation

		# ── Steady-state initialisation ───────────────────────────────────────────
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))            # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))            # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))            # (batch_size,)
		p[:, 0] = p_inf_fn(V[:, 0])                                    # (batch_size,) IKm at rest

		# ── Exponential Euler integration loop ────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)
			m_prev = m[:, i - 1]  # (batch_size,)
			h_prev = h[:, i - 1]  # (batch_size,)
			n_prev = n[:, i - 1]  # (batch_size,)
			p_prev = p[:, i - 1]  # (batch_size,)

			# Standard HH kinetics at current voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,) each

			# M-current gating: steady-state and time constant at current voltage
			p_inf_v = p_inf_fn(V_prev)  # (batch_size,)
			tau_p_v = tau_p_fn(V_prev)  # (batch_size,)

			# Effective inverse membrane time constant (units: 1/ms)
			# Sum of all active conductances divided by capacitance
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,) Na+ contribution
				+ (n_prev ** 4) * gbar_K            # (batch_size,) K+  contribution
				+ g_leak                             # (batch_size,) leak contribution
				+ gbar_Km * p_prev                   # (batch_size,) IKm contribution (slow K+)
			) / C  # (batch_size,)

			# Voltage steady state numerator (effective driving force sum + inputs)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,) Na+ driving
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,) K+  driving
				+ g_leak * E_leak                           # (batch_size,) leak driving
				+ gbar_Km * p_prev * E_K                   # (batch_size,) IKm drives toward E_K
				+ input_current[:, i - 1]                  # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# ↑ noise model unchanged from base — kept as-is per task requirements
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler updates — exact for linear ODEs, stable for fast gates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			p[:, i] = p_inf_v + (p_prev - p_inf_v) * Exp(-tstep / tau_p_v)                            # (batch_size,)

		# Return voltage traces with optional observation noise (currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)