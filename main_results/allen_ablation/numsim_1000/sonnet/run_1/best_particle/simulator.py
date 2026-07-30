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
		Hodgkin-Huxley neuron with added M-type slow K⁺ current (IKM).

		Physiological rationale for IKM:
		  The M-current is a slow, non-inactivating voltage-gated K⁺ current
		  that activates near spike threshold (~-35 mV). It produces
		  spike-frequency adaptation and regularizes inter-spike intervals,
		  improving agreement with tonic (evenly-spaced) firing statistics
		  (spike count, mean voltage during stimulation, voltage variance/skewness).
		  It does NOT produce bursting or sustained high-frequency firing.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) # applied current (uA/cm2)
			dt: float # time step (ms)
			t: torch.Tensor: (time_steps,) # time array (ms)
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int # random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # membrane voltage traces (mV)
		"""
		device = params.device

		# Set up random generator for reproducibility
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (sign applied: stored positive)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (sign applied: stored positive)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless noise amplitude

		# ── M-current parameters (X1 slot) ────────────────────────────────────
		# gbar_M  : maximal M-current conductance  [1e-4, 10] mS/cm²
		# E_M     : M-current reversal potential   [-150, ~0] mV  (K⁺-like, ~-80 mV expected)
		# tau_p_M : M-gate time constant           [1e-4, 3000] ms (slow activation, ~50-300 ms)
		gbar_M  = params[:, 6].float()          # (batch_size,) mS/cm²
		# params[:, 7] (gbar_X2) intentionally unused — one channel is sufficient
		E_M     = -params[:, 8].float()         # (batch_size,) mV; negative (K⁺ reversal)
		tau_p_M =  params[:, 9].float()         # (batch_size,) ms; positive, slow time constant

		tstep = float(dt)  # scalar ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (kept at 0 per task spec)
		C    = 1.0            # membrane capacitance uF/cm²
		E_Na = 53.0           # mV  Na⁺ reversal
		E_K  = -107.0         # mV  K⁺ reversal

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential — clamp extreme negative values
			# z: (batch_size,) or broadcastable
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable form of z / (exp(z) - 1) used in HH alpha/beta rates
			# z: (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gating kinetics ───────────────────────────────────────
		def alpha_m(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (IKM) gating kinetics ───────────────────────────────────
		# Single activation gate p; no inactivation (non-inactivating current).
		# Boltzmann steady-state: half-activation at -35 mV, slope 10 mV.
		# Time constant tau_p_M is a free (inferred) parameter — slow dynamics.
		def p_inf(x):
			# Steady-state M-gate activation: sigmoidal, activates near -35 mV
			# x: (batch_size,) → returns (batch_size,)
			return 1.0 / (1.0 + Exp(-(x + 35.0) / 10.0))

		# tau_p_M is already (batch_size,) — constant per sample, slow (~50-300 ms)

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na act
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inact
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M gate

		# ── Initial conditions at steady state ────────────────────────────────
		V_init = init_voltage.to(device)      # (batch_size,)
		V[:, 0] = V_init                      # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))   # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                              # (batch_size,) M-gate at rest

		# ── Time integration loop ─────────────────────────────────────────────
		for i in range(1, time_steps):
			# Compute gating variable rates at previous voltage
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,) each
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,) each
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,) each

			# M-gate steady state at previous voltage
			p_ss = p_inf(V[:, i - 1])   # (batch_size,) Boltzmann steady state

			# Effective membrane conductance sum (used for exponential integration)
			# Includes Na, K (delayed-rectifier), leak, and M-current contributions
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na conductance
				+ (n[:, i - 1] ** 4) * gbar_K                 # (batch_size,) K conductance
				+ g_leak                                       # (batch_size,) leak conductance
				+ gbar_M * p[:, i - 1]                        # (batch_size,) M-current conductance
			) / C   # (batch_size,)

			# Voltage steady-state numerator: sum of conductance-weighted reversal potentials
			# plus injected current and stochastic noise
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na    # (batch_size,) Na drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K drive
				+ g_leak * E_leak                                       # (batch_size,) leak drive
				+ gbar_M * p[:, i - 1] * E_M                           # (batch_size,) M-current drive
				+ input_current[:, i - 1]                              # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,) stochastic current noise
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler integration for voltage
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler integration for standard HH gates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)

			# Exponential Euler integration for M-gate (slow, constant tau_p_M)
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_p_M)   # (batch_size,)

		# Return voltage trace with (zero) observation noise
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)