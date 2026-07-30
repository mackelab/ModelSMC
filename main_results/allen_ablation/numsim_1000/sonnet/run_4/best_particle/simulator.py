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
		Hodgkin-Huxley neuron extended with M-type K+ current (IKM).

		IKM rationale:
		  - Slow, non-inactivating voltage-gated K+ current
		  - Activates near spike threshold (~-35 mV), providing graded hyperpolarization
		  - Produces regular, evenly-spaced tonic spiking without bursting
		  - Well-characterised in cortical and hippocampal neurons (Brown & Adams 1980)
		  - Addresses: spike count, mean/variance of voltage, AHP depth

		Args:
			init_voltage: torch.Tensor: (batch_size,)
			input_current: torch.Tensor: (batch_size, time_steps)
			dt: float
			t: torch.Tensor: (time_steps,)
			params: torch.Tensor: (batch_size, 10)
			seed: int or None

		Returns:
			V: torch.Tensor: (batch_size, time_steps)
		"""
		device = params.device

		# Random generator setup
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base parameters ───────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (sign applied here)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (sign applied here)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── X1 slot: M-type K+ current (IKM) ─────────────────────────────────
		# gbar_M  : maximal M-current conductance (mS/cm²), range [1e-4, 10]
		# V_half_M: half-activation voltage (mV), param_i = -params[:,8] ∈ [-150, -1e-4]
		#           Typical IKM V_half ≈ -35 mV  →  params[:,8] ≈ 35
		# tau_w   : slow gating time constant (ms), -param_j ∈ [1e-4, 3000]
		#           IKM tau typically 20–300 ms at physiological temperatures
		gbar_M    = params[:, 6].float()   # (batch_size,)  mS/cm²
		# X2 slot intentionally unused (parsimony principle)
		# gbar_X2 = params[:, 7]           # reserved, not needed
		V_half_M  = -params[:, 8].float()  # (batch_size,)  mV  (negative → subthreshold range)
		tau_w     = -params[:, 9].float()  # (batch_size,)  ms  (positive slow time constant)

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (shared by IKdr and IKM)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential; clamps exponent at -500
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z),
			)

		def efun(z):
			# Exponential function used in HH alpha/beta rate expressions
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics ─────────────────────────────────────
		def alpha_m(x):   # x: (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):    # x: (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):   # x: (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):    # x: (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):   # x: (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # x: (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,), (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,), (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (IKM) gating kinetics ──────────────────────────────────
		# Steady-state activation: sigmoid centred at V_half_M with slope 10 mV
		# V_half_M is inferred (≈ -35 mV); slope fixed for parsimony
		def w_inf(x):   # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		# tau_w is a single inferred scalar per batch element (voltage-independent)
		# Range [1e-4, 3000] ms; slow kinetics consistent with IKM literature

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  mV
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na act
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  Na inact
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  K act
		w = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  M-current gate

		# ── Initial conditions (steady-state at init_voltage) ─────────────────
		V_init = init_voltage.to(device)   # (batch_size,)
		V[:, 0] = V_init                   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		w[:, 0] = w_inf(V[:, 0])                              # (batch_size,)

		# ── Integration loop ──────────────────────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			w_prev = w[:, i - 1]   # (batch_size,)

			# HH gate rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current steady-state at previous voltage
			w_ss = w_inf(V_prev)   # (batch_size,)

			# Effective inverse membrane time constant (Σg / C)
			# Includes Na, K-delayed rectifier, leak, and M-current contributions
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev      # Na channel  (batch_size,)
				+ (n_prev ** 4) * gbar_K               # K-dr channel (batch_size,)
				+ g_leak                               # leak         (batch_size,)
				+ gbar_M * w_prev                      # IKM channel  (batch_size,)
			) / C   # (batch_size,)

			# Voltage steady-state numerator: Σ(g * E) + I_inj + noise
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)
				+ g_leak * E_leak                          # (batch_size,)
				+ gbar_M * w_prev * E_K                    # IKM reversal = E_K  (batch_size,)
				+ input_current[:, i - 1]                  # (batch_size,)
				+ nois_fact * torch.randn(
					batch_size, generator=generator, device=device
				) / (tstep ** 0.5)                         # (batch_size,)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler updates for all state variables
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			# M-current gate: slow exponential relaxation toward w_ss with time constant tau_w
			w[:, i] = w_ss + (w_prev - w_ss) * Exp(-tstep / tau_w)   # (batch_size,)

		# Return voltage trace with optional observation noise (currently 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)