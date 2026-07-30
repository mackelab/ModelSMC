import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
	def __init__(self):
		super(DiscoveredSimulator, self).__init__()  # () no state stored
		return

	def forward(
		self,
		init_voltage: torch.Tensor,  # (batch_size,)
		input_current: torch.Tensor,  # (batch_size, time_steps)
		dt: float,  # scalar
		t: torch.Tensor,  # (time_steps,)
		params: torch.Tensor,  # (batch_size, 10)
		seed=None,  # optional int or None
	):
		"""
		Hodgkin-Huxley with one additional slow, non-inactivating K+ (M-type) channel
		implemented using the X1 slot. This single-channel addition is chosen to correct
		discrepancies in firing rate and subthreshold voltage statistics while respecting
		the parsimony constraint (do not introduce bursting).

		All operations are batched over the first (batch) dimension.
		"""

		# Device and RNG
		device = params.device  # scalar-like
		if seed is not None:
			generator = torch.Generator(device=device)  # torch.Generator
			generator.manual_seed(int(seed))  # deterministic RNG when seed provided
		else:
			generator = torch.Generator(device=device)  # torch.Generator (not seeded)

		# Sizes
		batch_size = params.shape[0]  # int
		time_steps = t.shape[0]  # int

		# Extract parameters (apply sign conventions described in signature: magnitudes provided)
		gbar_Na = params[:, 0].float()  # (batch_size,) mS/cm^2
		gbar_K = params[:, 1].float()  # (batch_size,) mS/cm^2
		g_leak = params[:, 2].float()  # (batch_size,) mS/cm^2
		E_leak = -torch.abs(params[:, 3].float())  # (batch_size,) mV (apply negative sign internally)
		Vt = -torch.abs(params[:, 4].float())  # (batch_size,) mV (threshold offset, negative convention)
		nois_fact = params[:, 5].float()  # (batch_size,) unitless scaling for per-step noise
		gbar_X1 = params[:, 6].float()  # (batch_size,) mS/cm^2 - will be used for the M-type channel
		gbar_X2 = params[:, 7].float()  # (batch_size,) mS/cm^2 - left unused in this iteration
		param_i = torch.abs(params[:, 8].float())  # (batch_size,) positive; used as V_half offset (mV)
		param_j = torch.abs(params[:, 9].float())  # (batch_size,) positive; used as tau multiplier (ms)

		# Time step and constants
		tstep = float(dt)  # scalar (ms)
		C = 1.0  # scalar (uF/cm^2)
		nois_fact_obs = 0.0  # scalar (observation noise, kept zero)
		E_Na = torch.full((batch_size,), 53.0, device=device)  # (batch_size,) mV
		E_K = torch.full((batch_size,), -107.0, device=device)  # (batch_size,) mV

		# Safe numerical helpers
		def Exp(z: torch.Tensor) -> torch.Tensor:
			# z: (...,) -> (...,) elementwise safe exponential
			return torch.exp(torch.clamp(z, max=500.0))  # (...,)

		def efun(z: torch.Tensor) -> torch.Tensor:
			# stabilized function for z/(exp(z)-1)
			# z: (...,) -> (...,)
			small = torch.abs(z) < 1e-4  # (...,)
			return torch.where(small, 1.0 - z / 2.0, z / (Exp(z) - 1.0))  # (...,)

		# Hodgkin-Huxley alpha/beta rate functions (vectorized)
		def alpha_m(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0  # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch_size,)

		def alpha_h(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0  # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch_size,)

		def beta_h(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))  # (batch_size,)

		def alpha_n(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0  # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch_size,)

		def beta_n(x: torch.Tensor) -> torch.Tensor:
			# x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0  # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # (batch_size,)

		def tau_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			# alpha,beta: (batch_size,) -> tau: (batch_size,)
			return 1.0 / (alpha + beta + 1e-12)  # (batch_size,)

		def inf_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			# alpha,beta: (batch_size,) -> steady-state (batch_size,)
			return alpha / (alpha + beta + 1e-12)  # (batch_size,)

		# --------------------
		# Added mechanism (one channel): M-type (slow, non-inactivating K+) current
		# Rationale:
		# - Experimental discrepancies: tonic firing rate too high and resting/subthreshold
		#   statistics (mean/variance) mismatched by base HH model.
		# - M-current is a slowly activating K+ current that reduces excitability and
		#   increases subthreshold stability without producing bursting.
		# - Use both available flexible parameters for X1: param_i -> V_half offset (mV),
		#   param_j -> tau multiplier (ms). gbar_X1 is the channel conductance (mS/cm^2).
		#
		# Formulation:
		# - p_inf(V) = 1 / (1 + exp((V_half - V)/k))
		# - tau_p(V) = clamp(param_j, 1..1000) * (baseline_factor(V))
		# - current: I_M = gbar_X1 * p * (V - E_K)
		# - include gbar_X1 * p in total conductance for V steady-state calculation.
		def p_inf(Vx: torch.Tensor) -> torch.Tensor:
			# Vx: (batch_size,) -> (batch_size,)
			V_half = Vt + param_i  # (batch_size,)
			k = 6.0  # scalar slope (mV)
			return 1.0 / (1.0 + torch.exp((V_half - Vx) / k))  # (batch_size,)

		def tau_p(Vx: torch.Tensor) -> torch.Tensor:
			# Vx: (batch_size,) -> (batch_size,)
			V_half = Vt + param_i  # (batch_size,)
			# voltage-dependent baseline between ~5 and ~25 ms
			baseline = 5.0 + 20.0 / (1.0 + torch.exp((Vx - V_half) / 8.0))  # (batch_size,)
			# param_j interpreted as ms-scale multiplier; clamp to physiol. range
			mult = torch.clamp(param_j, min=1.0, max=1000.0)  # (batch_size,)
			return mult * baseline  # (batch_size,)

		# --------------------
		# Allocate state arrays (batched time series)
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)  # M-type activation

		# Initialize states to steady-state at initial voltage
		V_init = init_voltage.to(device)  # (batch_size,)
		V[:, 0] = V_init  # (batch_size,) -> V[:,0] (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])  # (batch_size,) initialize M-type gate at steady state

		# Precompute sqrt(dt) used in the existing noise formulation (kept as-is per constraints)
		sqrt_dt = tstep ** 0.5  # scalar

		# Time integration loop (vectorized over batch dimension)
		for i in range(1, time_steps):
			# previous voltage
			V_prev = V[:, i - 1]  # (batch_size,)

			# HH rates at V_prev
			a_m = alpha_m(V_prev)  # (batch_size,)
			b_m = beta_m(V_prev)  # (batch_size,)
			a_h = alpha_h(V_prev)  # (batch_size,)
			b_h = beta_h(V_prev)  # (batch_size,)
			a_n = alpha_n(V_prev)  # (batch_size,)
			b_n = beta_n(V_prev)  # (batch_size,)

			# gate steady-states and taus
			tau_m = tau_x(a_m, b_m)  # (batch_size,)
			tau_h = tau_x(a_h, b_h)  # (batch_size,)
			tau_n = tau_x(a_n, b_n)  # (batch_size,)
			m_inf = inf_x(a_m, b_m)  # (batch_size,)
			h_inf = inf_x(a_h, b_h)  # (batch_size,)
			n_inf = inf_x(a_n, b_n)  # (batch_size,)

			# M-type properties at V_prev
			p_inf_prev = p_inf(V_prev)  # (batch_size,)
			tau_p_prev = tau_p(V_prev)  # (batch_size,)

			# Total effective inverse membrane time constant (1/ms) contributions
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K  # (batch_size,)
				+ g_leak  # (batch_size,)
				+ gbar_X1 * p[:, i - 1]  # (batch_size,) contribution from M-current
			) / C  # (batch_size,)

			# Noise term: keep same scaling as original model (no change per critical constraint)
			noise_term = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (sqrt_dt + 1e-12)  # (batch_size,)

			# Numerator for steady-state voltage (sum of g*E + injected current + noise)
			V_inf_num = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na  # (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K  # (batch_size,)
				+ g_leak * E_leak  # (batch_size,)
				+ gbar_X1 * p[:, i - 1] * E_K  # (batch_size,) M-current contributes with K reversal
				+ input_current[:, i - 1]  # (batch_size,)
				+ noise_term  # (batch_size,)
			)  # (batch_size,)

			# Avoid division by zero in case tau_V_inv is tiny
			tau_V_inv_safe = torch.clamp(tau_V_inv, min=1e-12)  # (batch_size,)

			# Compute steady-state voltage and exponential integrator step
			V_inf = V_inf_num / (tau_V_inv_safe * C)  # (batch_size,)
			exp_factor = torch.exp(-tstep * tau_V_inv_safe)  # (batch_size,)
			V[:, i] = V_inf + (V_prev - V_inf) * exp_factor  # (batch_size,) -> assign to V[:,i]

			# Update gating variables via exact solution of linear ODEs (exponential update)
			m[:, i] = m_inf + (m[:, i - 1] - m_inf) * torch.exp(-tstep / torch.clamp(tau_m, min=1e-12))  # (batch_size,)
			h[:, i] = h_inf + (h[:, i - 1] - h_inf) * torch.exp(-tstep / torch.clamp(tau_h, min=1e-12))  # (batch_size,)
			n[:, i] = n_inf + (n[:, i - 1] - n_inf) * torch.exp(-tstep / torch.clamp(tau_n, min=1e-12))  # (batch_size,)

			# Update M-type gate p using its own timescale (slow, non-inactivating)
			p[:, i] = p_inf_prev + (p[:, i - 1] - p_inf_prev) * torch.exp(-tstep / torch.clamp(tau_p_prev, min=1e-12))  # (batch_size,)

			# Enforce physiological bounds for gating variables in [0,1]
			m[:, i] = torch.clamp(m[:, i], 0.0, 1.0)  # (batch_size,)
			h[:, i] = torch.clamp(h[:, i], 0.0, 1.0)  # (batch_size,)
			n[:, i] = torch.clamp(n[:, i], 0.0, 1.0)  # (batch_size,)
			p[:, i] = torch.clamp(p[:, i], 0.0, 1.0)  # (batch_size,)

		# Return voltage traces (no observation noise added)
		return V + nois_fact_obs * torch.randn((batch_size, time_steps), generator=generator, device=device)  # (batch_size, time_steps)