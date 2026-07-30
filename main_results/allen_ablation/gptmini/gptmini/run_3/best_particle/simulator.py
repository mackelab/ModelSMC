import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
	"""
	DiscoveredSimulator: Hodgkin-Huxley base + one added slow non-inactivating K+-like (M-type) channel
	implemented using the X1 slot. This single channel uses both flexible parameters (param_i, param_j)
	for V_half and tau_p mapping respectively, preserving parsimony.

	Notes on shapes are provided on assignment lines as comments.
	"""
	def __init__(self):
		super(DiscoveredSimulator, self).__init__()
		return

	def forward(
		self,
		init_voltage: torch.Tensor,
		input_current: torch.Tensor,
		dt: float,
		t: torch.Tensor,
		params: torch.Tensor,
		seed=None,
	):
		"""
		Args (shapes):
			init_voltage: torch.Tensor, shape (batch_size,)
			input_current: torch.Tensor, shape (batch_size, time_steps)
			dt: float, scalar
			t: torch.Tensor, shape (time_steps,)
			params: torch.Tensor, shape (batch_size, 10)
			seed: int or None
		Returns:
			V: torch.Tensor, shape (batch_size, time_steps)
		"""

		device = params.device  # shape: scalar/device

		# RNG for reproducible stochasticity
		if seed is not None:
			generator = torch.Generator(device=device)  # shape: scalar/generator
			generator.manual_seed(int(seed))  # shape: scalar
		else:
			generator = torch.Generator(device=device)  # shape: scalar/generator

		batch_size = params.shape[0]  # shape: scalar
		time_steps = t.shape[0]  # shape: scalar

		# ----------------------------
		# Parameter extraction (use provided indexing; apply internal sign conventions)
		# ----------------------------
		gbar_Na = params[:, 0].float().clamp(min=1e-12)  # shape: (batch_size,) mS/cm^2
		gbar_K = params[:, 1].float().clamp(min=1e-12)  # shape: (batch_size,) mS/cm^2
		g_leak = params[:, 2].float().clamp(min=1e-12)  # shape: (batch_size,) mS/cm^2
		# params[3] stores |E_leak|; apply negative sign internally (intracellular negative)
		E_leak = -params[:, 3].float().abs()  # shape: (batch_size,) mV
		# params[4] stores |Vt|; keep Vt negative like original HH shift
		Vt = -params[:, 4].float().abs()  # shape: (batch_size,) mV
		nois_fact = params[:, 5].float()  # shape: (batch_size,) unitless (process noise scale)

		# Use only X1 for a single added channel (parsimony). Keep X2 unused.
		gbar_X1 = params[:, 6].float().clamp(min=1e-12)  # shape: (batch_size,) mS/cm^2 (IM conductance)
		_ = params[:, 7].float()  # shape: (batch_size,) placeholder for unused gbar_X2
		param_i_raw = params[:, 8].float().clamp(min=1e-4, max=150.0)  # shape: (batch_size,) in [1e-4,150]
		param_j_raw = params[:, 9].float().clamp(min=1e-4, max=3000.0)  # shape: (batch_size,) in [1e-4,3000]

		tstep = float(dt)  # shape: scalar

		# Fixed biophysical constants
		nois_fact_obs = 0.0  # shape: scalar (observation noise scale)
		C = 1.0  # shape: scalar uF/cm^2
		E_Na = torch.full((batch_size,), 53.0, device=device)  # shape: (batch_size,) mV
		E_K = torch.full((batch_size,), -107.0, device=device)  # shape: (batch_size,) mV

		# ----------------------------
		# Numerically stable helpers and HH rates
		# ----------------------------
		def Exp(z: torch.Tensor) -> torch.Tensor:
			# shape: same as z
			# safe exponential with both-side clamping to avoid overflow/underflow
			z_clamped = torch.clamp(z, min=-500.0, max=500.0)  # shape: same as z
			return torch.exp(z_clamped)  # shape: same as z

		def efun(z: torch.Tensor) -> torch.Tensor:
			# shape: same as z
			# stable implementation of z / (exp(z) - 1)
			z_clamped = torch.clamp(z, min=-500.0, max=500.0)  # shape: same as z
			# use expm1 for better accuracy when z is small
			exp_minus_1 = torch.expm1(z_clamped)  # shape: same as z
			small_mask = torch.abs(z_clamped) < 1e-4  # shape: same as z
			# safe branch: when z ~ 0, use 1 - z/2 approximation
			ans = torch.where(small_mask, 1.0 - z_clamped / 2.0, z_clamped / (exp_minus_1 + 1e-12))  # shape: same as z
			return ans  # shape: same as z

		def alpha_m(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 13.0  # shape: (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # shape: (batch_size,)

		def beta_m(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # shape: (batch_size,)

		def alpha_h(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 17.0  # shape: (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # shape: (batch_size,)

		def beta_h(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))  # shape: (batch_size,)

		def alpha_n(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 15.0  # shape: (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # shape: (batch_size,)

		def beta_n(x: torch.Tensor) -> torch.Tensor:
			# shape: same as x
			v1 = x - Vt - 10.0  # shape: (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # shape: (batch_size,)

		def tau_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			# shape: same as alpha/beta
			return 1.0 / (alpha + beta + 1e-12)  # shape: (batch_size,)

		def inf_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			# shape: same as alpha/beta
			return alpha / (alpha + beta + 1e-12)  # shape: (batch_size,)

		# ----------------------------
		# Implement single IM-like (M-type) slow K+ current using X1 slot
		# Physiological rationale (comments):
		# - M-type channels activate slowly with depolarization and do not inactivate;
		#   they induce spike-frequency adaptation and stabilize subthreshold voltage,
		#   which can adjust spike count, mean/variance without creating bursting.
		# - We map param_i_raw -> V_half in a bounded physiologic range, and param_j_raw -> tau_p.
		# ----------------------------
		# Map param_i_raw ∈ [1e-4,150] -> V_half ∈ [-120, 50] mV (affine)
		Vhalf_min = torch.tensor(-120.0, device=device)  # shape: scalar
		Vhalf_max = torch.tensor(50.0, device=device)  # shape: scalar
		V_half = Vhalf_min + (param_i_raw / 150.0) * (Vhalf_max - Vhalf_min)  # shape: (batch_size,) mV

		# Map param_j_raw ∈ [1e-4,3000] -> tau_p ∈ [5,3000] ms (clamped)
		tau_p_base = param_j_raw.clamp(min=5.0, max=3000.0)  # shape: (batch_size,) ms

		# Fixed slope for p_inf for identifiability (keeps model simple)
		k_p = torch.full((batch_size,), 6.0, device=device)  # shape: (batch_size,) mV

		def p_inf_fun(Vv: torch.Tensor, Vhalf: torch.Tensor) -> torch.Tensor:
			# shape: same as Vv and Vhalf
			# logistic steady-state for p in (0,1)
			return 1.0 / (1.0 + torch.exp(-(Vv - Vhalf) / (k_p + 1e-12)))  # shape: (batch_size,)

		def tau_p_fun(tau_base: torch.Tensor) -> torch.Tensor:
			# shape: same as tau_base
			# simply return tau_base (already clamped)
			return tau_base  # shape: (batch_size,)

		# ----------------------------
		# State containers
		# ----------------------------
		V = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps) (IM gating)

		# Initialization
		V_init = init_voltage.to(device).float()  # shape: (batch_size,)
		V[:, 0] = V_init  # shape: (batch_size, time_steps) column 0 assigned
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # shape: (batch_size, time_steps) column 0
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # shape: (batch_size, time_steps) column 0
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # shape: (batch_size, time_steps) column 0
		p[:, 0] = p_inf_fun(V[:, 0], V_half)  # shape: (batch_size, time_steps) column 0

		# Precompute sqrt(dt) for noise scaling (preserve original noise model)
		noise_den = (tstep ** 0.5)  # shape: scalar

		# ----------------------------
		# Simulation loop (vectorized per batch)
		# ----------------------------
		for i in range(1, time_steps):
			# previous voltage and gating variables
			V_prev = V[:, i - 1]  # shape: (batch_size,)
			a_m = alpha_m(V_prev)  # shape: (batch_size,)
			b_m = beta_m(V_prev)  # shape: (batch_size,)
			a_h = alpha_h(V_prev)  # shape: (batch_size,)
			b_h = beta_h(V_prev)  # shape: (batch_size,)
			a_n = alpha_n(V_prev)  # shape: (batch_size,)
			b_n = beta_n(V_prev)  # shape: (batch_size,)

			# steady states and taus for fast gates
			tau_m = tau_x(a_m, b_m)  # shape: (batch_size,)
			tau_h = tau_x(a_h, b_h)  # shape: (batch_size,)
			tau_n = tau_x(a_n, b_n)  # shape: (batch_size,)
			m_inf = inf_x(a_m, b_m)  # shape: (batch_size,)
			h_inf = inf_x(a_h, b_h)  # shape: (batch_size,)
			n_inf = inf_x(a_n, b_n)  # shape: (batch_size,)

			# IM gating variables using mapped parameters
			p_inf = p_inf_fun(V_prev, V_half)  # shape: (batch_size,)
			tau_p = tau_p_fun(tau_p_base)  # shape: (batch_size,)

			# Effective conductances (batch-wise)
			g_Na_eff = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # shape: (batch_size,)
			g_K_eff = (n[:, i - 1] ** 4) * gbar_K  # shape: (batch_size,)
			g_M_eff = gbar_X1 * p[:, i - 1]  # shape: (batch_size,)
			g_total = g_Na_eff + g_K_eff + g_leak + g_M_eff  # shape: (batch_size,)

			# Membrane inverse time constant
			tau_V_inv = g_total / C  # shape: (batch_size,)

			# Noise term: preserve original noise model (divide by sqrt(dt))
			noise_term = nois_fact * torch.randn((batch_size,), generator=generator, device=device) / (noise_den + 1e-12)  # shape: (batch_size,)

			# Numerator for steady-state voltage: weighted reversal potentials + input + noise
			num = (
				g_Na_eff * E_Na  # shape: (batch_size,)
				+ g_K_eff * E_K  # shape: (batch_size,)
				+ g_leak * E_leak  # shape: (batch_size,)
				+ g_M_eff * E_K  # shape: (batch_size,) IM is K+-like
				+ input_current[:, i - 1]  # shape: (batch_size,)
				+ noise_term  # shape: (batch_size,)
			)  # shape: (batch_size,)

			den = (tau_V_inv * C).clamp(min=1e-12)  # shape: (batch_size,)
			V_inf = num / den  # shape: (batch_size,)

			# Exact exponential update of linearized voltage ODE
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # shape: (batch_size, time_steps) assign col i

			# Update gating variables via exact solution of first-order kinetics
			m[:, i] = m_inf + (m[:, i - 1] - m_inf) * Exp(-tstep / (tau_m + 1e-12))  # shape: (batch_size, time_steps) col i
			h[:, i] = h_inf + (h[:, i - 1] - h_inf) * Exp(-tstep / (tau_h + 1e-12))  # shape: (batch_size, time_steps) col i
			n[:, i] = n_inf + (n[:, i - 1] - n_inf) * Exp(-tstep / (tau_n + 1e-12))  # shape: (batch_size, time_steps) col i
			p[:, i] = p_inf + (p[:, i - 1] - p_inf) * Exp(-tstep / (tau_p + 1e-12))  # shape: (batch_size, time_steps) col i

		# Observation noise left at zero per specification
		obs_noise = nois_fact_obs * torch.randn((batch_size, time_steps), generator=generator, device=device)  # shape: (batch_size, time_steps)
		return V + obs_noise  # shape: (batch_size, time_steps)