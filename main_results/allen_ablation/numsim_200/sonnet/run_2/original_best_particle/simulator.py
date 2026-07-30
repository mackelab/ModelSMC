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
		Hodgkin-Huxley neuron with an added M-type K+ current (IKm).

		The M-current (Km) is a slow, non-inactivating, voltage-dependent K+
		current that activates near rest (~-35 mV). It provides:
		  - Spike-frequency adaptation stabilising regular tonic firing
		  - Subthreshold voltage stabilisation (improves resting potential statistics)
		  - No burst-promoting dynamics — purely adapting/regularising

		Args:
			init_voltage : torch.Tensor (batch_size,)   initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps)  injected current (uA/cm2)
			dt           : float                         time step (ms)
			t            : torch.Tensor (time_steps,)   time array (ms)
			params       : torch.Tensor (batch_size, 10) biophysical parameters
			seed         : int or None

		Returns:
			V            : torch.Tensor (batch_size, time_steps)  membrane voltage (mV)
		"""
		device = params.device

		# ── Random generator ────────────────────────────────────────────────────
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # scalar
		time_steps = t.shape[0]        # scalar

		# ── Parameter extraction ─────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()        # (batch_size,)  mS/cm2
		gbar_K    = params[:, 1].float()        # (batch_size,)  mS/cm2
		g_leak    = params[:, 2].float()        # (batch_size,)  mS/cm2
		E_leak    = -params[:, 3].float()       # (batch_size,)  mV  (negated)
		Vt        = -params[:, 4].float()       # (batch_size,)  mV  (negated)
		nois_fact = params[:, 5].float()        # (batch_size,)  unitless

		# ── M-type K+ current (IKm) — uses slot X1 + param_i + param_j ──────────
		# gbar_Km   : maximal M-current conductance (mS/cm2), range [1e-4, 10]
		# V_half_p  : half-activation voltage (mV), param_i negated → range [-150, ~0]
		#             canonical value ~-35 mV, inferred by posterior
		# tau_p     : activation time constant (ms), -param_j → range [1e-4, 3000]
		#             canonical M-current tau ~100-300 ms (slow adaptation)
		gbar_Km  = params[:, 6].float()        # (batch_size,)  mS/cm2
		V_half_p = -params[:, 8].float()       # (batch_size,)  mV   half-activation
		tau_p    = -params[:, 9].float()       # (batch_size,)  ms   slow time constant
		# NOTE: gbar_X2 (params[:,7]) intentionally unused — one channel suffices

		tstep = float(dt)

		# ── Fixed biophysical constants ──────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV  (shared reversal for Kdr and Km)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Clamp exponent to avoid overflow; z: (batch_size,) or broadcastable
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Numerically stable (x / (exp(x) - 1)) for HH alpha/beta rates
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ────────────────────────────────────────
		def alpha_m(x):  # x: (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2      # (batch_size,)

		def alpha_h(x):
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)           # (batch_size,)

		def beta_h(x):
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))     # (batch_size,)

		def alpha_n(x):
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2    # (batch_size,)

		def beta_n(x):
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)             # (batch_size,)

		def tau_x(alpha, beta):
			return 1.0 / (alpha + beta)              # (batch_size,)

		def inf_x(alpha, beta):
			return alpha / (alpha + beta)            # (batch_size,)

		# ── M-current (IKm) kinetics ─────────────────────────────────────────────
		# Boltzmann steady-state with fixed slope k=10 mV (canonical for M-current)
		# Activation is purely voltage-dependent (no inactivation)
		def p_inf(x):
			# x: (batch_size,)  →  p_inf: (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_p) / 10.0))

		# ── State variable allocation ────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps)  M-current

		# ── Initial conditions (steady state at init_voltage) ────────────────────
		V_init    = init_voltage.to(device)               # (batch_size,)
		V[:, 0]   = V_init
		m[:, 0]   = inf_x(alpha_m(V_init), beta_m(V_init))
		h[:, 0]   = inf_x(alpha_h(V_init), beta_h(V_init))
		n[:, 0]   = inf_x(alpha_n(V_init), beta_n(V_init))
		p[:, 0]   = p_inf(V_init)                         # M-current at quasi-steady state

		# ── Time integration (exponential Euler) ─────────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)

			# HH gating rates at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # each (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # each (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # each (batch_size,)

			# Gating variables for M-current at previous voltage
			p_ss = p_inf(V_prev)                          # (batch_size,)

			# Effective conductance sum (inverse membrane time constant * C)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # I_Na conductance
				+ (n[:, i - 1] ** 4) * gbar_K                  # I_K conductance
				+ g_leak                                        # leak conductance
				+ gbar_Km * p[:, i - 1]                        # I_Km conductance (M-current)
			) / C   # (batch_size,)

			# Weighted reversal sum (voltage steady state numerator)
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # Na drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # K drive
				+ g_leak * E_leak                                       # leak drive
				+ gbar_Km * p[:, i - 1] * E_K                         # M-current drive (E_K)
				+ input_current[:, i - 1]                              # injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler updates
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)           # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			# M-current: exponential Euler with slow, fixed time constant tau_p
			p[:, i] = p_ss + (p[:, i-1] - p_ss) * Exp(-tstep / tau_p)              # (batch_size,)

		# ── Return voltage trace (+ optional observation noise) ──────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)