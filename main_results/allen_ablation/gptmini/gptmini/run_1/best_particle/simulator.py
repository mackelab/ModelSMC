import torch  # shape: ()
import torch.nn as nn  # shape: ()


class DiscoveredSimulator(nn.Module):  # shape: ()
	def __init__(self):  # shape: ()
		super(DiscoveredSimulator, self).__init__()  # shape: ()
		return  # shape: ()

	def forward(  # shape: ()
		self,
		init_voltage: torch.Tensor,  # shape: (batch_size,)
		input_current: torch.Tensor,  # shape: (batch_size, time_steps)
		dt: float,  # shape: ()
		t: torch.Tensor,  # shape: (time_steps,)
		params: torch.Tensor,  # shape: (batch_size, 10)
		seed=None,  # shape: ()
	):  # shape: ()
		"""
		Hodgkin-Huxley + one parsimonious slow non-inactivating K (M-type) current (X1).
		Design choices (rationale in comments inline):
		 - Use X1 (gbar_X1, param_i, param_j) as a slow K current: I_X1 = gbar_X1 * p * (V - E_K)
		 - p_inf(V) = 1/(1+exp(-(V - V_half)/k_p)) with V_half = param_i (per-batch), k_p fixed (6 mV)
		 - tau_p(V) = clamp(param_j (tau_base), min=0.5 ms, max=5000 ms) -- keeps tau positive and identifiable
		 - We use both flexible params (param_i,param_j) for the single channel (per iteration rule)
		 - Noise model kept identical to provided signature (process-like additive term scaled by 1/sqrt(dt))
		"""
		device = params.device  # shape: ()

		# RNG setup (reproducible if seed provided)
		if seed is not None:  # shape: ()
			generator = torch.Generator(device=device)  # shape: ()
			generator.manual_seed(seed)  # shape: ()
		else:  # shape: ()
			generator = torch.Generator(device=device)  # shape: ()

		batch_size = params.shape[0]  # shape: ()
		time_steps = t.shape[0]  # shape: ()

		# ---- Parameter extraction (with shapes) ----
		gbar_Na = params[:, 0].float()  # shape: (batch_size,) mS/cm²
		gbar_K = params[:, 1].float()  # shape: (batch_size,) mS/cm²
		g_leak = params[:, 2].float()  # shape: (batch_size,) mS/cm²
		# signature: params[3] stores |E_leak|, sign applied internally (intracellular leak typically negative)
		E_leak = -params[:, 3].float()  # shape: (batch_size,) mV
		# signature: params[4] stores |Vt| for rate offsets; sign applied internally consistent with base code
		Vt = -params[:, 4].float()  # shape: (batch_size,) mV
		nois_fact = params[:, 5].float()  # shape: (batch_size,) unitless (kept as in signature)
		# Pre-allocated extra conductances
		gbar_X1 = params[:, 6].float()  # shape: (batch_size,) mS/cm² (will be used for M-type K)
		gbar_X2 = params[:, 7].float()  # shape: (batch_size,) mS/cm² (kept unused for parsimony)
		# Flexible parameters (use as magnitudes, do NOT negate)
		param_i = params[:, 8].float()  # shape: (batch_size,) (interpreted as V_half in mV)
		param_j = params[:, 9].float()  # shape: (batch_size,) (interpreted as tau_base in ms)

		tstep = float(dt)  # shape: ()

		# Fixed biophysical constants (tensors on device)
		nois_fact_obs = 0.0  # shape: ()
		C = 1.0  # shape: () uF/cm²
		E_Na = torch.tensor(53.0, device=device)  # shape: () mV
		E_K = torch.tensor(-107.0, device=device)  # shape: () mV

		# ---- Numerically-stable helpers ----
		def Exp(z: torch.Tensor) -> torch.Tensor:  # z shape: (...,)
			# stable exp with clamping to avoid overflow/underflow
			return torch.exp(torch.clamp(z, min=-500.0, max=500.0))  # shape: (...,)

		def efun(z: torch.Tensor) -> torch.Tensor:  # z shape: (...,)
			# stable evaluation of z/(exp(z)-1) near z=0
			abs_z = torch.abs(z)  # shape: (...,)
			return torch.where(abs_z < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))  # shape: (...,)

		# ---- HH rate functions (vectorized) ----
		def alpha_m(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 13.0  # shape: (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # shape: (batch_size,)

		def beta_m(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # shape: (batch_size,)

		def alpha_h(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 17.0  # shape: (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # shape: (batch_size,)

		def beta_h(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 40.0  # shape: (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))  # shape: (batch_size,)

		def alpha_n(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 15.0  # shape: (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # shape: (batch_size,)

		def beta_n(x: torch.Tensor) -> torch.Tensor:  # x shape: (batch_size,)
			v1 = x - Vt - 10.0  # shape: (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)  # shape: (batch_size,)

		def tau_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:  # alpha,beta shape: (batch_size,)
			return 1.0 / (alpha + beta + 1e-12)  # shape: (batch_size,)

		def inf_x(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:  # alpha,beta shape: (batch_size,)
			return alpha / (alpha + beta + 1e-12)  # shape: (batch_size,)

		# ---- New channel (X1) design: slow non-inactivating K (M-type) ----
		# Physiological rationale:
		#  - A slow, non-inactivating K current reduces spike frequency (adaptation)
		#  - It does not produce bursting if tau is large and activation is slow
		# Parameter mapping:
		#  - gbar_X1: conductance amplitude (mS/cm²)
		#  - param_i : V_half of p_inf (mV)
		#  - param_j : tau_base (ms) for p (kept positive and clamped)
		k_p = torch.tensor(6.0, device=device)  # shape: () slope of p_inf (mV)
		tau_p_min = torch.tensor(0.5, device=device)  # shape: () minimum tau_p (ms)
		tau_p_max = torch.tensor(5000.0, device=device)  # shape: () maximum tau_p (ms)

		# ---- State containers (batched) ----
		V = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # shape: (batch_size, time_steps) new gating var p

		# ---- Initialization ----
		V_init = init_voltage.to(device)  # shape: (batch_size,)
		V[:, 0] = V_init  # shape: (batch_size, time_steps) assigned at [:,0]
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # shape: (batch_size, time_steps) assigned [:,0]
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # shape: (batch_size, time_steps) assigned [:,0]
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # shape: (batch_size, time_steps) assigned [:,0]
		# p steady-state initialization using param_i as V_half (no negation)
		V_half_init = param_i  # shape: (batch_size,) mV
		p_inf_init = 1.0 / (1.0 + Exp(-(V[:, 0] - V_half_init) / k_p))  # shape: (batch_size,)
		p[:, 0] = p_inf_init  # shape: (batch_size, time_steps) assigned [:,0]

		# ---- Time stepping (vectorized per time-step, batched across neurons) ----
		for i in range(1, time_steps):  # shape: ()
			# previous voltage (batch)
			V_prev = V[:, i - 1]  # shape: (batch_size,)

			# HH rates at V_prev
			a_m = alpha_m(V_prev)  # shape: (batch_size,)
			b_m = beta_m(V_prev)  # shape: (batch_size,)
			a_h = alpha_h(V_prev)  # shape: (batch_size,)
			b_h = beta_h(V_prev)  # shape: (batch_size,)
			a_n = alpha_n(V_prev)  # shape: (batch_size,)
			b_n = beta_n(V_prev)  # shape: (batch_size,)

			# steady-states and taus for HH gates
			m_inf = inf_x(a_m, b_m)  # shape: (batch_size,)
			tau_m = tau_x(a_m, b_m)  # shape: (batch_size,)
			h_inf = inf_x(a_h, b_h)  # shape: (batch_size,)
			tau_h = tau_x(a_h, b_h)  # shape: (batch_size,)
			n_inf = inf_x(a_n, b_n)  # shape: (batch_size,)
			tau_n = tau_x(a_n, b_n)  # shape: (batch_size,)

			# X1 (p) kinetics using both param slots:
			V_half = param_i  # shape: (batch_size,) V_half for p_inf (mV)
			tau_base = param_j  # shape: (batch_size,) tau base for p (ms)
			# p_inf: sigmoidal activation (non-inactivating)
			p_inf = 1.0 / (1.0 + Exp(-(V_prev - V_half) / k_p))  # shape: (batch_size,)
			# tau_p: clamp tau_base to reasonable physiological bounds to avoid instabilities
			tau_p = torch.clamp(tau_base, min=tau_p_min.item(), max=tau_p_max.item())  # shape: (batch_size,)

			# Conductances at this time step (instantaneous)
			g_Na_t = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # shape: (batch_size,)
			g_K_t = (n[:, i - 1] ** 4) * gbar_K  # shape: (batch_size,)
			g_X1_t = gbar_X1 * p[:, i - 1]  # shape: (batch_size,)
			# total conductance (add tiny eps for numerical stability)
			g_total = g_Na_t + g_K_t + g_leak + g_X1_t + 1e-12  # shape: (batch_size,)
			tau_V_inv = g_total / C  # shape: (batch_size,)

			# steady-state numerator sum (g_i * E_i)
			numer = g_Na_t * E_Na + g_K_t * E_K + g_leak * E_leak + g_X1_t * E_K  # shape: (batch_size,)

			# input drive and process-noise term (KEPT same as signature: noise scaled by 1/sqrt(dt))
			I_drive = input_current[:, i - 1]  # shape: (batch_size,)
			noise_drive = nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # shape: (batch_size,)

			# voltage steady state given instantaneous conductances
			V_inf = (numer + I_drive + noise_drive) / (g_total)  # shape: (batch_size,)

			# exponential update for V with time constant 1/tau_V_inv (stable exact update)
			exp_factor = Exp(-tstep * tau_V_inv)  # shape: (batch_size,)
			V[:, i] = V_inf + (V_prev - V_inf) * exp_factor  # shape: (batch_size, time_steps) assigned [:,i]

			# gating updates (exact first-order solutions)
			m[:, i] = m_inf + (m[:, i - 1] - m_inf) * Exp(-tstep / tau_m)  # shape: (batch_size, time_steps) assigned [:,i]
			h[:, i] = h_inf + (h[:, i - 1] - h_inf) * Exp(-tstep / tau_h)  # shape: (batch_size, time_steps) assigned [:,i]
			n[:, i] = n_inf + (n[:, i - 1] - n_inf) * Exp(-tstep / tau_n)  # shape: (batch_size, time_steps) assigned [:,i]

			# p update (slow M-type K activation)
			p[:, i] = p_inf + (p[:, i - 1] - p_inf) * Exp(-tstep / tau_p)  # shape: (batch_size, time_steps) assigned [:,i]

		# observation noise is zero per signature; keep structure for future extension
		return V + nois_fact_obs * torch.randn((batch_size, time_steps), generator=generator, device=device)  # shape: (batch_size, time_steps)