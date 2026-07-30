import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
	"""Hodgkin-Huxley with a single added slow, non-inactivating K+ channel (X1, 'M-type').

	Design choices and justifications (inline):
	 - Parsimony: only one extra channel (X1) added and it uses both flexible params (param_i, param_j).
	 - Physiological rationale: M-type K+ channels activate near subthreshold and are slow;
	   they reduce excitability and tune tonic spike frequency without producing bursting.
	 - param mapping: param_i (positive) is mapped linearly into a signed V_half range to allow
	   subthreshold activation (so we don't force V_half positive). param_j is used as tau_base (ms).
	 - Noise model: intentionally left unchanged per critical constraint (keeps original scaling).
	"""

	def __init__(self):
		super(DiscoveredSimulator, self).__init__()  # () no-op

	def forward(
		self,
		init_voltage: torch.Tensor,
		input_current: torch.Tensor,
		dt: float,
		t: torch.Tensor,
		params: torch.Tensor,
		seed=None,
	):
		# Shapes: init_voltage: (batch,) ; input_current: (batch, time) ; dt: scalar ; t: (time,) ; params: (batch,10)
		device = params.device  # () device

		# RNG
		if seed is not None:
			generator = torch.Generator(device=device)  # () rng
			generator.manual_seed(int(seed))  # () seeded
		else:
			generator = torch.Generator(device=device)  # () rng

		batch_size = params.shape[0]  # () batch
		time_steps = t.shape[0]  # () time
		tstep = float(dt)  # () scalar

		# --- Extract & sanitize parameters (all shapes: (batch,))
		gbar_Na = torch.clamp(params[:, 0].float(), min=1e-8, max=1e4)  # (batch,) mS/cm^2
		gbar_K = torch.clamp(params[:, 1].float(), min=1e-8, max=1e4)  # (batch,) mS/cm^2
		g_leak = torch.clamp(params[:, 2].float(), min=1e-12, max=1e4)  # (batch,) mS/cm^2
		E_leak = -torch.clamp(torch.abs(params[:, 3].float()), min=1.0, max=150.0)  # (batch,) mV (sign applied)
		Vt = -torch.clamp(torch.abs(params[:, 4].float()), min=1.0, max=200.0)  # (batch,) mV (offset semantics)
		nois_fact = torch.clamp(params[:, 5].float(), min=0.0, max=1e6)  # (batch,) unitless (kept unchanged)

		# X1 and X2 conductances (we will use only X1) (shapes: (batch,))
		gbar_X1 = torch.clamp(params[:, 6].float(), min=0.0, max=500.0)  # (batch,) mS/cm^2 (X1)
		gbar_X2 = torch.clamp(params[:, 7].float(), min=0.0, max=500.0)  # (batch,) mS/cm^2 (unused placeholder)

		# Flexible parameters (given as positive magnitudes in signature)
		param_i_pos = torch.clamp(params[:, 8].float().abs(), min=1e-4, max=150.0)  # (batch,) positive
		param_j_pos = torch.clamp(params[:, 9].float().abs(), min=1e-4, max=3000.0)  # (batch,) positive

		# Map param_i_pos (0..150) -> V_half range [-120, +40] mV so V_half can be negative (subthreshold)
		# This preserves the positive-only external encoding but yields physiologically-signed V_half.
		V_half_p = param_i_pos * (160.0 / 150.0) - 120.0  # (batch,) mV
		V_half_p = torch.clamp(V_half_p, min=-150.0, max=150.0)  # (batch,) mV (safety)

		# Map param_j_pos directly to tau_base (ms), clamp to physiologically relevant range
		tau_base_p = torch.clamp(param_j_pos, min=1e-4, max=3000.0)  # (batch,) ms

		# --- Constants (broadcastable)
		nois_fact_obs = 0.0  # () observation noise
		C = 1.0  # () uF/cm^2
		E_Na = torch.full((batch_size,), 53.0, device=device, dtype=torch.float32)  # (batch,) mV
		E_K = torch.full((batch_size,), -90.0, device=device, dtype=torch.float32)  # (batch,) mV
		# Note: E_K set to -90 mV (more typical) to improve resting/spiking alignment while preserving structure.

		# --- Numerical helpers
		def Exp(z: torch.Tensor) -> torch.Tensor:
			# safe exponential with shape preservation
			zc = torch.clamp(z, min=-5e2, max=5e2)  # same shape
			return torch.exp(zc)  # same shape

		def efun(z: torch.Tensor) -> torch.Tensor:
			# stable z/(exp(z)-1) fallback for small z; preserves shape
			e = Exp(z)  # same shape
			den = e - 1.0  # same shape
			small = torch.abs(z) < 1e-6  # same shape (bool)
			res = torch.where(small, 1.0 - z / 2.0 + (z * z) / 12.0, z / den)  # same shape
			return torch.nan_to_num(res, nan=1.0, posinf=1e12, neginf=-1e12)  # same shape

		# --- HH alpha/beta kinetics (v shape: (batch,))
		def alpha_m(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 13.0  # (batch,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch,)

		def beta_m(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 40.0  # (batch,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch,)

		def alpha_h(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 17.0  # (batch,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch,)

		def beta_h(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 40.0  # (batch,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))  # (batch,)

		def alpha_n(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 15.0  # (batch,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch,)

		def beta_n(v: torch.Tensor) -> torch.Tensor:
			v1 = v - Vt - 10.0  # (batch,)
			return 0.5 * Exp(-v1 / 40.0)  # (batch,)

		def tau_x(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
			den = a + b  # (batch,)
			return 1.0 / torch.clamp(den, min=1e-12)  # (batch,)

		def inf_x(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
			den = a + b  # (batch,)
			return a / torch.clamp(den, min=1e-12)  # (batch,)

		# --- X1 (M-type K) channel design
		# p_inf: sigmoid centered at V_half_p with slope k_p
		k_p = torch.full((batch_size,), 6.0, device=device, dtype=torch.float32)  # (batch,) mV (slope)
		# tau_p: modest voltage-dependence around tau_base_p (keeps single param_j informative)
		# tau_p(V) = tau_base_p * (1 + alpha / (1 + exp((V - (V_half+offset))/slope)))
		# alpha and offset chosen to allow slow activation near subthreshold without creating bursting.
		alpha_tau = 3.0  # () dimensionless
		offset_tau = 8.0  # () mV
		slope_tau = 6.0  # () mV

		# --- State variables (batch,time)
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch, time)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch, time)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch, time)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch, time)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch, time)  # X1 gating

		# --- Initialization
		V_init = init_voltage.to(device).float()  # (batch,)
		V[:, 0] = V_init  # (batch, time)
		a_n0 = alpha_n(V[:, 0])  # (batch,)
		b_n0 = beta_n(V[:, 0])  # (batch,)
		n[:, 0] = inf_x(a_n0, b_n0)  # (batch, time)
		a_m0 = alpha_m(V[:, 0])  # (batch,)
		b_m0 = beta_m(V[:, 0])  # (batch,)
		m[:, 0] = inf_x(a_m0, b_m0)  # (batch, time)
		a_h0 = alpha_h(V[:, 0])  # (batch,)
		b_h0 = beta_h(V[:, 0])  # (batch,)
		h[:, 0] = inf_x(a_h0, b_h0)  # (batch, time)
		# initialize p at its steady state for V_init
		p_inf0 = 1.0 / (1.0 + Exp((V_half_p - V[:, 0]) / k_p))  # (batch,)
		p[:, 0] = p_inf0  # (batch, time)

		# --- Time stepping (explicit exponential integrator for gating and V)
		for i in range(1, time_steps):
			# previous step voltage (batch,)
			V_prev = V[:, i - 1]  # (batch,)

			# HH kinetics at V_prev (batch,)
			a_m = alpha_m(V_prev)  # (batch,)
			b_m = beta_m(V_prev)  # (batch,)
			a_h = alpha_h(V_prev)  # (batch,)
			b_h = beta_h(V_prev)  # (batch,)
			a_n = alpha_n(V_prev)  # (batch,)
			b_n = beta_n(V_prev)  # (batch,)

			# steady-states and taus for HH gates (batch,)
			inf_m = inf_x(a_m, b_m)  # (batch,)
			tau_m = tau_x(a_m, b_m)  # (batch,)
			inf_h = inf_x(a_h, b_h)  # (batch,)
			tau_h = tau_x(a_h, b_h)  # (batch,)
			inf_n = inf_x(a_n, b_n)  # (batch,)
			tau_n = tau_x(a_n, b_n)  # (batch,)

			# X1 (M-type) gating: p_inf and voltage-dependent tau_p (batch,)
			p_inf = 1.0 / (1.0 + Exp((V_half_p - V_prev) / k_p))  # (batch,)
			tau_p = tau_base_p * (1.0 + alpha_tau / (1.0 + Exp((V_prev - (V_half_p + offset_tau)) / slope_tau)))  # (batch,)
			tau_p = torch.clamp(tau_p, min=1e-4, max=3000.0)  # (batch,)

			# Effective conductances (batch,)
			g_Na_eff = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch,)
			g_K_eff = (n[:, i - 1] ** 4) * gbar_K  # (batch,)
			g_X1_eff = gbar_X1 * p[:, i - 1]  # (batch,)  # M-type
			g_tot = g_Na_eff + g_K_eff + g_leak + g_X1_eff  # (batch,)
			g_tot_safe = torch.clamp(g_tot, min=1e-12)  # (batch,)

			# Currents multiplied by reversal potentials (numerator terms) (batch,)
			I_Na_term = g_Na_eff * E_Na  # (batch,)
			I_K_term = g_K_eff * E_K  # (batch,)
			I_X1_term = g_X1_eff * E_K  # (batch,)
			I_leak_term = g_leak * E_leak  # (batch,)

			# Input current & stochastic current (kept as original scaling per constraints)
			I_inj = input_current[:, i - 1].to(device).float()  # (batch,)
			noise_term = nois_fact * torch.randn((batch_size,), generator=generator, device=device) / (tstep ** 0.5)  # (batch,)

			# Voltage steady-state numerator & V_inf (batch,)
			V_inf_num = I_Na_term + I_K_term + I_X1_term + I_leak_term + I_inj + noise_term  # (batch,)
			V_inf = V_inf_num / g_tot_safe  # (batch,)

			# Exponential Euler update for V (analytic for linearized membrane)
			decay = -tstep * (g_tot / C)  # (batch,)
			decay_clamped = torch.clamp(decay, min=-1e6, max=1e2)  # (batch,)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(decay_clamped)  # (batch, time)

			# Update gating variables via analytic solution for first-order kinetics (batch,)
			exp_m = Exp(torch.clamp(-tstep / tau_m, min=-1e6, max=1e2))  # (batch,)
			exp_h = Exp(torch.clamp(-tstep / tau_h, min=-1e6, max=1e2))  # (batch,)
			exp_n = Exp(torch.clamp(-tstep / tau_n, min=-1e6, max=1e2))  # (batch,)
			exp_p = Exp(torch.clamp(-tstep / tau_p, min=-1e6, max=1e2))  # (batch,)

			m[:, i] = inf_m + (m[:, i - 1] - inf_m) * exp_m  # (batch, time)
			h[:, i] = inf_h + (h[:, i - 1] - inf_h) * exp_h  # (batch, time)
			n[:, i] = inf_n + (n[:, i - 1] - inf_n) * exp_n  # (batch, time)
			p[:, i] = p_inf + (p[:, i - 1] - p_inf) * exp_p  # (batch, time)

		# Return voltage traces with optional observation noise (currently zero)
		V_out = V + nois_fact_obs * torch.randn((batch_size, time_steps), generator=generator, device=device)  # (batch, time)
		return V_out  # (batch, time)