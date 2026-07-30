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
		Hodgkin-Huxley neuron simulator extended with an M-type (muscarinic)
		slow non-inactivating K+ current (I_M) to correct voltage distribution
		shape errors (skewness, kurtosis) and spike-count discrepancies.

		Physiological rationale for I_M:
		  - Activates slowly at subthreshold voltages (~-35 mV half-activation)
		  - Non-inactivating: provides sustained outward current during tonic firing
		  - Produces spike-frequency adaptation (reduces firing rate over time)
		  - Deepens and prolongs AHP, correcting voltage distribution asymmetry
		  - Does NOT produce bursting; only regulates tonic spiking regularity

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial voltage
			input_current: torch.Tensor: (batch_size, time_steps) # input current
			dt: float # time step size
			t: torch.Tensor: (time_steps,) # time array
			params: torch.Tensor: (batch_size, n_params) # parameters
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
		time_steps = t.shape[0]  # scalar int

		# ── Base parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (sign applied internally)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (sign applied internally)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── M-current parameters (X1 slot) ───────────────────────────────────
		# gbar_M  : M-current maximal conductance (mS/cm²), range [1e-4, 10]
		# param_i : half-activation voltage offset (mV), range [1e-4, 150]
		#           V_half = -90 + param_i  =>  param_i ~ 55 gives V_half ~ -35 mV
		# param_j : M-current activation time constant (ms), range [1e-4, 3000]
		#           Physiological M-current tau ~ 50–300 ms
		# Note: negation removed vs base code so values stay in their positive ranges
		gbar_M  = params[:, 6].float()   # (batch_size,)  mS/cm²  — M-current conductance
		# gbar_X2 reserved / unused — kept simple per parsimony principle
		param_i = params[:, 8].float()   # (batch_size,)  half-activation offset (mV)
		param_j = params[:, 9].float()   # (batch_size,)  slow time constant (ms)

		tstep = float(dt)  # scalar float

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (also reversal for M-current, same ion)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Clamp to avoid overflow; z: (batch_size,) or broadcastable
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable form of z/(exp(z)-1); z: arbitrary shape
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics ─────────────────────────────────────
		def alpha_m(x):  # x: (batch_size,)
			v1 = x - Vt - 13.0  # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):   # x: (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch_size,)

		def alpha_h(x):  # x: (batch_size,)
			v1 = x - Vt - 17.0  # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch_size,)

		def beta_h(x):   # x: (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))  # (batch_size,)

		def alpha_n(x):  # x: (batch_size,)
			v1 = x - Vt - 15.0  # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch_size,)

		def beta_n(x):   # x: (batch_size,)
			v1 = x - Vt - 10.0  # (batch_size,)
			return 0.5 * Exp(-v1 / 40)  # (batch_size,)

		def tau_x(alpha, beta):  # alpha, beta: (batch_size,)
			return 1.0 / (alpha + beta)  # (batch_size,)

		def inf_x(alpha, beta):  # alpha, beta: (batch_size,)
			return alpha / (alpha + beta)  # (batch_size,)

		# ── M-current kinetics (I_M = gbar_M * w * (V - E_K)) ───────────────
		# Slow non-inactivating K+ current; single activation gate w.
		# Steady-state: sigmoidal activation centred at V_half = -90 + param_i
		# (param_i ~ 55 mV => V_half ~ -35 mV, physiologically appropriate)
		# Slope factor fixed at 10 mV for parsimony.
		# Time constant: param_j (ms); slow to avoid burst induction.
		def w_inf(x):  # x: (batch_size,)
			V_half = -90.0 + param_i  # (batch_size,)  half-activation voltage
			return 1.0 / (1.0 + Exp(-(x - V_half) / 10.0))  # (batch_size,)

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		w = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)  M-gate

		# ── Initial conditions ────────────────────────────────────────────────
		V_init   = init_voltage.to(device)  # (batch_size,)
		V[:, 0]  = V_init
		n[:, 0]  = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		m[:, 0]  = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0]  = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		w[:, 0]  = w_inf(V[:, 0])                             # (batch_size,)

		# ── Simulation loop ───────────────────────────────────────────────────
		for i in range(1, time_steps):
			# Voltage at previous step: (batch_size,)
			V_prev = V[:, i - 1]

			# Standard HH gate rates at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,) each
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,) each
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,) each

			# Current gate values
			m_prev = m[:, i - 1]  # (batch_size,)
			h_prev = h[:, i - 1]  # (batch_size,)
			n_prev = n[:, i - 1]  # (batch_size,)
			w_prev = w[:, i - 1]  # (batch_size,)

			# Effective membrane conductance (sum of all active channels)
			# tau_V_inv = total_conductance / C
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na+ conductance  (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K+ conductance   (batch_size,)
				+ g_leak                             # leak conductance (batch_size,)
				+ gbar_M * w_prev                   # M-current conductance (batch_size,)
			) / C  # (batch_size,)

			# Noise sample for this time step
			noise = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,)

			# Voltage steady-state (weighted reversal potentials + injected current)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # Na+ drive   (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # K+ drive    (batch_size,)
				+ g_leak * E_leak                          # leak drive  (batch_size,)
				+ gbar_M * w_prev * E_K                   # M-current drive (reversal = E_K) (batch_size,)
				+ input_current[:, i - 1]                  # injected current (batch_size,)
				+ noise                                    # stochastic drive (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential-Euler updates for voltage and gates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# M-gate: slow exponential-Euler update with fixed time constant param_j
			# Clamp param_j to avoid division by zero or negative time constants
			tau_w = param_j.clamp(min=1e-3)  # (batch_size,)  ms
			w_ss  = w_inf(V_prev)             # (batch_size,)  steady-state M-gate at V_prev
			w[:, i] = w_ss + (w_prev - w_ss) * Exp(-tstep / tau_w)  # (batch_size,)

		# Return voltage trace with optional observation noise (currently 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)