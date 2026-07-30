import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
	def __init__(self):
		super(DiscoveredSimulator, self).__init__()  # shape: scalar
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
		Hodgkin-Huxley with one parsimonious slow M-type K channel (X1).
		Design summary:
		 - Use only X1 (M-type slow K, non-inactivating) to adjust tonic spiking frequency
		   and subthreshold conductance (helps mean/variance of subthreshold voltage).
		 - Both flexible parameters are used for X1: param_i -> V_half mapping; param_j -> tau_q.
		 - dt is in ms (signature-specified); all gating time-constants are in ms.
		"""
		device = params.device  # shape: torch.device

		# RNG (for reproducible stochastic term)
		if seed is not None:
			generator = torch.Generator(device=device)  # shape: torch.Generator
			generator.manual_seed(int(seed))  # shape: scalar
		else:
			generator = torch.Generator(device=device)  # shape: torch.Generator

		batch_size = params.shape[0]  # shape: scalar
		time_steps = t.shape[0]  # shape: scalar

		# --- Parameter extraction and safe transforms ---
		gbar_Na = torch.clamp(params[:, 0].float(), min=1e-8)  # shape: (batch_size,)
		gbar_K = torch.clamp(params[:, 1].float(), min=1e-8)  # shape: (batch_size,)
		g_leak = torch.clamp(params[:, 2].float(), min=0.0)  # shape: (batch_size,)
		# params[3] provided as |E_leak| (mV), apply negative sign internally
		E_leak = -torch.clamp(params[:, 3].float(), min=0.0)  # shape: (batch_size,)
		# params[4] provided as |Vt| (mV), apply negative sign internally for threshold offset usage
		Vt = -torch.clamp(params[:, 4].float(), min=0.0)  # shape: (batch_size,)
		nois_fact = torch.clamp(params[:, 5].float(), min=0.0)  # shape: (batch_size,)
		# Additional conductances (use X1 only)
		gbar_X1 = torch.clamp(params[:, 6].float(), min=0.0)  # shape: (batch_size,)
		gbar_X2 = torch.clamp(params[:, 7].float(), min=0.0)  # shape: (batch_size,) (unused)
		# Flexible parameters used together for X1 (ensure positivity within signature bounds)
		param_i = torch.clamp(params[:, 8].float(), min=1e-4, max=150.0)  # shape: (batch_size,)
		param_j = torch.clamp(params[:, 9].float(), min=1e-4, max=3000.0)  # shape: (batch_size,)

		# dt (ms) as per signature
		tstep = float(dt)  # shape: scalar (ms)

		# Fixed biophysical constants
		nois_fact_obs = 0.0  # shape: scalar
		C = 1.0  # uF/cm² (shape: scalar)
		E_Na = torch.tensor(53.0, device=device).float()  # mV (shape: scalar)
		E_K = torch.tensor(-107.0, device=device).float()  # mV (shape: scalar)

		####################################
		# Numerical helpers (batch-wise)
		def Exp(z: torch.Tensor) -> torch.Tensor:
			# stable exponential (clip very negative exponents)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))  # shape: same as z

		def efun(z: torch.Tensor) -> torch.Tensor:
			# safe function used in HH formulations to avoid 0/0 near small z
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))  # shape: same as z

		# Classic Hodgkin-Huxley channel kinetics (batch-wise; rates per ms)
		def alpha_m(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 13.0  # shape: (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # shape: (batch_size,)

		def beta_m(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # shape: (batch_size,)

		def alpha_h(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 17.0  # shape: (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # shape: (batch_size,)

		def beta_h(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))  # shape: (batch_size,)

		def alpha_n(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 15.0  # shape: (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # shape: (batch_size,)

		def beta_n(x: torch.Tensor) -> torch.Tensor:
			v1 = x - Vt - 10.0  # shape: (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # shape: (batch_size,)

		def tau_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			# returns time-constant in ms; add tiny eps for numerical stability
			return 1.0 / (alpha + beta + 1e-12)  # shape: (batch_size,)

		def inf_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
			return alpha / (alpha + beta + 1e-12)  # shape: (batch_size,)

		####################################
		# ---- Parsimonious extra channel X1: M-type potassium (slow, non-inactivating)
		# Physiological rationale:
		#  - M-type (Kv7) provides slow, voltage-dependent K conductance that accumulates
		#    during depolarization and reduces excitability on 10s-1000s ms timescales.
		#  - This can tune spike rate and subthreshold mean/variance without producing bursting.
		# Parameter usage (both flexible params used for this single channel):
		#  - param_i (1e-4..150) -> mapped to V_half range [-90, +60] mV (physiological span).
		#  - param_j (1e-4..3000) -> used as tau_q in ms, clamped to [1, 3000] ms.
		####################################
		# Map param_i -> V_half (mV) via linear rescale to physiologic interval [-90, +60]
		V_half = (-90.0 + (param_i / 150.0) * 150.0).to(device)  # shape: (batch_size,)
		# Fixed slope for activation (mV)
		k_q = torch.tensor(6.0, device=device).float()  # shape: scalar
		# Map param_j -> tau_q (ms) with clamping to physiologically plausible [1, 3000] ms
		tau_q_base = torch.clamp(param_j, min=1.0, max=3000.0).to(device)  # shape: (batch_size,)

		####################################
		# Allocate state arrays (batch x time)
		V = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		q = torch.zeros((batch_size, time_steps), device=device)  # M-type activation (batch_size, time_steps)

		# Initialization at time 0
		V_init = init_voltage.to(device)  # shape: (batch_size,)
		V[:, 0] = V_init  # shape: (batch_size, time_steps) (assign first column)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # shape: (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # shape: (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # shape: (batch_size,)
		# Initialize q to its steady state at initial voltage (slow variable)
		q_inf_0 = 1.0 / (1.0 + Exp((V_half - V[:, 0]) / k_q))  # shape: (batch_size,)
		q[:, 0] = q_inf_0  # shape: (batch_size, time_steps) (first column assigned)

		# Time-stepping loop (batch-wise)
		for i in range(1, time_steps):  # shape: scalar loop index
			# Compute HH kinetics at previous voltage V[:, i-1]
			a_m = alpha_m(V[:, i - 1])  # shape: (batch_size,)
			b_m = beta_m(V[:, i - 1])  # shape: (batch_size,)
			a_h = alpha_h(V[:, i - 1])  # shape: (batch_size,)
			b_h = beta_h(V[:, i - 1])  # shape: (batch_size,)
			a_n = alpha_n(V[:, i - 1])  # shape: (batch_size,)
			b_n = beta_n(V[:, i - 1])  # shape: (batch_size,)

			# M-type (X1) steady-state and (mostly constant) time-constant at previous voltage
			q_inf = 1.0 / (1.0 + Exp((V_half - V[:, i - 1]) / k_q))  # shape: (batch_size,)
			tau_q = tau_q_base  # shape: (batch_size,) (ms)

			# Effective membrane conductance (total / C) -> used as tau_V_inv
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # Na activation contribution (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K  # delayed rectifier K (batch_size,)
				+ g_leak  # leak conductance (batch_size,)
				+ gbar_X1 * q[:, i - 1]  # M-type K adds voltage-dependent conductance (batch_size,)
			) / C  # shape: (batch_size,)

			# Noise term (current-like) scaled consistently with dt (dt in ms)
			noise_term = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # shape: (batch_size,)

			# Voltage steady-state numerator (sum conductance*reversal + input + noise)
			V_inf_num = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na  # Na (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K  # delayed rectifier K (batch_size,)
				+ g_leak * E_leak  # leak (batch_size,)
				+ gbar_X1 * q[:, i - 1] * E_K  # M-type K steady-state contribution (batch_size,)
				+ input_current[:, i - 1]  # external input current (batch_size,)
				+ noise_term  # stochastic current-like term (batch_size,)
			)  # shape: (batch_size,)

			# Compute V_inf safely (avoid division by zero)
			denom = (tau_V_inv * C).clamp(min=1e-12)  # shape: (batch_size,)
			V_inf = V_inf_num / denom  # shape: (batch_size,)

			# Exponential update for voltage (analytic for linear conductance form)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)  # shape: (batch_size,)

			# Update fast gating variables with first-order kinetics (stable exponential)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # shape: (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # shape: (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # shape: (batch_size,)

			# Slow M-type activation update (single dominant time-constant tau_q)
			q[:, i] = q_inf + (q[:, i - 1] - q_inf) * Exp(-tstep / (tau_q + 1e-12))  # shape: (batch_size,)

		# Return voltage traces; observation noise left unchanged (nois_fact_obs == 0)
		return V + nois_fact_obs * torch.randn(batch_size, time_steps, generator=generator, device=device)  # shape: (batch_size, time_steps)