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
		Hodgkin-Huxley neuron with an added M-type potassium current (K_M / KCNQ).

		Physiological rationale for M-current addition:
		- M-current is a slow, non-inactivating K+ current activating near rest (~-35 mV)
		- It produces spike-frequency adaptation, regularizing inter-spike intervals
		- It supports tonic, evenly-spaced spiking WITHOUT bursting — exactly matching
		  the observed data characteristics (regular tonic spiking, no bursts)
		- It shifts the distribution of subthreshold voltages and affects variance/skewness
		  of the voltage trace, addressing the statistical discrepancies in the base model

		Channel slots used:
		- gbar_KM  = gbar_X1  (M-current conductance, mS/cm²)
		- V_half_p = param_i  (half-activation voltage; negated internally, ~-35 mV)
		- tau_scale = param_j (time constant numerator scale; negated internally, ~100-500 ms)
		- gbar_X2, left unused (only ONE additional channel added per parsimony principle)

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial voltage in mV
			input_current: torch.Tensor: (batch_size, time_steps) # injected current uA/cm²
			dt: float # time step in ms
			t: torch.Tensor: (time_steps,) # time array in ms
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane potential traces in mV
		"""
		device = params.device

		# Set up random generator for reproducibility
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ── Extract base parameters ──────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  — sign flip: raw positive → negative mV
		Vt        = -params[:, 4].float()  # (batch_size,) mV  — sign flip: raw positive → negative mV
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# ── Extract M-current parameters (slot X1 + param_i + param_j) ──────────
		# M-current (K_M): slow non-inactivating potassium current
		# Supports regular tonic spiking via adaptation without causing bursts
		gbar_KM   = params[:, 6].float()   # (batch_size,) mS/cm², range [1e-4, 10]
		# V_half_p: half-activation voltage for p gate
		# Raw param_i ∈ [1e-4, 150], negated → range [−150, −1e-4] mV
		# Physiological M-current activation at ~−35 mV → raw param_i ≈ 35
		V_half_p  = -params[:, 8].float()  # (batch_size,) mV
		# tau_scale: controls the slow time constant of the p gate
		# Raw param_j ∈ [1e-4, 3000], negated → range [−3000, −1e-4]
		# We use -param_j (= raw value) as the positive numerator for tau_p
		# Physiological M-current tau ~ 100–500 ms → raw param_j ≈ 200–1000
		tau_scale = params[:, 9].float()   # (batch_size,) positive raw value used directly

		# gbar_X2 / param_j left intentionally unused — parsimony principle

		tstep = float(dt)  # scalar float, ms

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (kept at 0 as instructed)
		C    = 1.0            # uF/cm² membrane capacitance
		E_Na = 53.0           # mV sodium reversal potential
		E_K  = -107.0         # mV potassium reversal potential (shared by K_DR and K_M)

		# ── Numerical helper functions ────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential with lower clamp at -500
			# z: (batch_size,) → output: (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Handles the x/(exp(x)-1) singularity near zero via L'Hopital
			# z: (batch_size,) → output: (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics (Na+ and K+ delayed rectifier) ──────────
		def alpha_m(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) → output: (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → output: (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → output: (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (K_M) kinetics ──────────────────────────────────────────────
		# Slow, non-inactivating K+ current; single gating variable p
		# Physiological reference: Brown & Adams (1980), Wang (1998)
		#
		# Steady-state activation: sigmoidal centered at V_half_p (~-35 mV)
		# Time constant: bell-shaped, slow (controlled by tau_scale)
		# Reversal at E_K (same potassium reversal as K_DR)

		def p_inf(x):
			# M-current steady-state activation (sigmoid)
			# x: (batch_size,) → output: (batch_size,)
			# V_half_p: (batch_size,), slope fixed at 10 mV (physiologically realistic)
			return 1.0 / (1.0 + Exp(-(x - V_half_p) / 10.0))

		def tau_p(x):
			# M-current activation time constant (ms)
			# x: (batch_size,) → output: (batch_size,)
			# Bell-shaped around V_half_p; tau_scale sets the peak value
			# At x = V_half_p: tau_p ≈ tau_scale / 2 (since cosh(0) = 1, denom = 2)
			# tau_scale ∈ [1e-4, 3000] ms (raw positive value from params[:, 9])
			dv = (x - V_half_p) / 20.0  # (batch_size,) normalized voltage deviation
			# Use hyperbolic cosine structure: denominator = exp(dv) + exp(-dv) >= 2
			denom = Exp(dv) + Exp(-dv)   # (batch_size,) always >= 2.0
			return tau_scale / denom      # (batch_size,) ms, peaks at V_half_p

		# ── Allocate state variable tensors ───────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K_DR gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ act gate
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inact gate
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current gate

		# ── Initialize at steady state ────────────────────────────────────────────
		V_init = init_voltage.to(device)      # (batch_size,)
		V[:, 0] = V_init                       # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))  # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))  # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))  # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])              # (batch_size,) M-gate at steady state

		# ── Time integration (exponential Euler method) ───────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,) current voltage

			# Standard HH gate kinetics at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# M-current gate kinetics at V_prev
			p_ss  = p_inf(V_prev)   # (batch_size,) steady-state value
			tau_p_val = tau_p(V_prev)  # (batch_size,) time constant in ms

			# Effective inverse membrane time constant (sum of all conductances / C)
			# Units: (mS/cm²) / (uF/cm²) = ms⁻¹
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ contribution
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) K_DR contribution
				+ g_leak                                         # (batch_size,) leak contribution
				+ gbar_KM * p[:, i - 1]                         # (batch_size,) K_M contribution
			) / C  # (batch_size,) ms⁻¹

			# Voltage steady-state numerator (weighted sum of reversal potentials + currents)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K_DR drive
				+ g_leak * E_leak                                        # (batch_size,) leak drive
				+ gbar_KM * p[:, i - 1] * E_K                          # (batch_size,) K_M drive
				+ input_current[:, i - 1]                               # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,) scaled noise
			) / (tau_V_inv * C)  # (batch_size,) mV

			# Exponential Euler updates (exact for linear ODEs)
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)           # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			# M-current gate update (exponential Euler with its own slow time constant)
			p[:, i] = p_ss + (p[:, i-1] - p_ss) * Exp(-tstep / tau_p_val)          # (batch_size,)

		# Return voltage trace with optional observation noise (currently 0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)