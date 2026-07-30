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
		Hodgkin-Huxley neuron with a self-consistent M-type (Kv7/KCNQ) K+ current.

		KEY STRUCTURAL CHANGE FROM PRIOR ITERATIONS:
		Previous versions defined p_inf via a Boltzmann function and tau_p via
		separate exponential rate functions, which were kinetically inconsistent
		(p_inf ≠ alpha_p/(alpha_p+beta_p)). This iteration uses a fully self-consistent
		alpha/beta formulation for the M-current gate — exactly mirroring the HH
		formulation for m, h, n — so that both p_inf and tau_p are derived from the
		same pair of rate functions. This removes a systematic bias and should improve
		accuracy of spike-rate and resting-potential statistics.

		Physiological rationale for M-current:
		  The M-current (Kv7/KCNQ) is a slow, non-inactivating, subthreshold K+
		  current. It activates near resting potential and deactivates slowly after
		  hyperpolarisation. It regulates inter-spike intervals, promotes regular
		  tonic spiking, and prevents burst firing — consistent with the observed
		  data characteristics.

		Args:
			init_voltage: torch.Tensor: (batch_size,)            initial voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) injected current (uA/cm²)
			dt: float                                             time step (ms)
			t: torch.Tensor: (time_steps,)                       time array (ms)
			params: torch.Tensor: (batch_size, 10)               biophysical parameters
			seed: optional int                                    random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)             membrane voltage (mV)
		"""
		device = params.device

		# ── Random generator ──────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na conductance    mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  K-dr conductance  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal     mV
		Vt        = -params[:, 4].float()  # (batch_size,)  spike threshold   mV
		nois_fact  = params[:, 5].float()  # (batch_size,)  noise scale

		# ── M-current parameters (slot X1, param_i, param_j) ─────────────────
		# gbar_M: M-current maximal conductance from slot X1 [1e-4, 10] mS/cm²
		gbar_M    = params[:, 6].float()   # (batch_size,)  mS/cm²

		# param_i stored as negative of positive in [1e-4, 150]:
		#   V_half_M positive ~35 → gate half-activates at V ≈ -35 mV (subthreshold)
		V_half_M  = -params[:, 8].float()  # (batch_size,)  mV (positive, ~35)

		# param_j stored as negative of positive in [1e-4, 3000]:
		#   tau_max_M scales the M-current time constant (50–500 ms is physiological)
		#   In the self-consistent formulation: tau_p(V) = tau_max / (alpha_p + beta_p)
		tau_max_M = -params[:, 9].float()  # (batch_size,)  ms (positive, peak time constant)
		tau_max_M = torch.clamp(tau_max_M, min=1.0, max=3000.0)  # (batch_size,)  safety

		tstep = float(dt)  # scalar ms

		# ── Fixed biophysical constants ───────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # membrane capacitance  uF/cm²
		E_Na = 53.0    # Na reversal potential  mV
		E_K  = -107.0  # K reversal potential   mV (shared by K-dr and M-current)

		# ── Numerical helpers ─────────────────────────────────────────────────
		def Exp(z):
			# Clipped exponential — prevents overflow at large negative z
			# (batch_size,) → (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Numerically stable  z / (exp(z) - 1)
			# (batch_size,) → (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics ─────────────────────────────────────
		def alpha_m(x):  # (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):  # (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):  # (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):  # (batch_size,),(batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):  # (batch_size,),(batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current: self-consistent alpha/beta kinetics ────────────────────
		#
		# CRITICAL STRUCTURAL CHANGE:
		# In prior iterations p_inf was a separate Boltzmann and tau_p came from
		# independent exponentials (Wang 1998) — these are NOT consistent with
		# each other because inf_x(alpha_p, beta_p) ≠ Boltzmann(V).
		#
		# Here we define alpha_p_M and beta_p_M as simple exponentials such that:
		#   p_inf(V) = alpha_p_M / (alpha_p_M + beta_p_M)   [Boltzmann-like]
		#   tau_p(V) = tau_max_M / (alpha_p_M + beta_p_M)   [voltage-dependent]
		#
		# Rate form (symmetric slopes around half-activation):
		#   alpha_p_M = exp(+dV / s)   opening rate — increases with depolarisation
		#   beta_p_M  = exp(-dV / s)   closing rate — increases with hyperpolarisation
		#   where dV = V + V_half_M and s = 10 (mV) matches Boltzmann slope k=10
		#
		# This gives p_inf = 1/(1+exp(-2*dV/s)) ≈ Boltzmann with k=5 mV effective,
		# which is a reasonable M-current slope factor.
		# The time constant peaks at dV=0 (V_half) and decreases symmetrically away.
		#
		# Using the same pair of functions for both p_inf and tau_p ensures full
		# kinetic self-consistency — eliminating systematic bias in spike statistics.

		def alpha_p_M(x):  # (batch_size,) → (batch_size,)
			# Opening rate of M-current gate
			# Increases exponentially with depolarisation beyond half-activation
			dV = x + V_half_M              # (batch_size,)  deviation from half-act
			return Exp(dV / 10.0)          # (batch_size,)  unitless rate coefficient

		def beta_p_M(x):   # (batch_size,) → (batch_size,)
			# Closing rate of M-current gate
			# Increases exponentially with hyperpolarisation below half-activation
			dV = x + V_half_M              # (batch_size,)  deviation from half-act
			return Exp(-dV / 10.0)         # (batch_size,)  unitless rate coefficient

		# Derived self-consistent steady state and time constant
		# p_inf_M(V) = alpha_p_M / (alpha_p_M + beta_p_M) = sigmoid(dV/10)
		# tau_p_M(V) = tau_max_M / (alpha_p_M + beta_p_M)  peaks at V_half, slow elsewhere
		# Both computed inside the loop using inf_x/tau_x helpers for clarity

		# ── State variable allocation ─────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate

		# ── Initialise all gates at steady state ─────────────────────────────
		V_init  = init_voltage.to(device)                               # (batch_size,)
		V[:, 0] = V_init                                                # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))             # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))             # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))             # (batch_size,)
		p[:, 0] = inf_x(alpha_p_M(V[:, 0]), beta_p_M(V[:, 0]))        # (batch_size,) M-gate at ss

		# ── Exponential Euler integration loop ───────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,)
			n_prev = n[:, i - 1]   # (batch_size,)
			m_prev = m[:, i - 1]   # (batch_size,)
			h_prev = h[:, i - 1]   # (batch_size,)
			p_prev = p[:, i - 1]   # (batch_size,)

			# Standard HH rates evaluated at V_prev
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current rates at V_prev (self-consistent alpha/beta)
			a_p, b_p = alpha_p_M(V_prev), beta_p_M(V_prev)   # (batch_size,), (batch_size,)

			# M-current steady state and time constant — both from same rate functions
			p_ss    = inf_x(a_p, b_p)                     # (batch_size,)  p_inf(V_prev)
			tau_p_v = tau_max_M * tau_x(a_p, b_p)         # (batch_size,)  ms, voltage-dependent
			tau_p_v = torch.clamp(tau_p_v, min=1e-3)      # (batch_size,)  numerical safety

			# Noise sample for this time step
			noise = nois_fact * torch.randn(
				batch_size, generator=generator, device=device
			) / (tstep ** 0.5)  # (batch_size,)

			# Total membrane conductance (inverse time constant × C)
			# M-current contributes gbar_M * p_prev (linearised around V_prev)
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # Na-current       (batch_size,)
				+ (n_prev ** 4) * gbar_K            # K-dr current     (batch_size,)
				+ g_leak                             # leak             (batch_size,)
				+ gbar_M * p_prev                   # M-current        (batch_size,)
			) / C  # (batch_size,)

			# Voltage steady state (weighted reversal potentials + injected current)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K             # (batch_size,)
				+ g_leak * E_leak                           # (batch_size,)
				+ gbar_M * p_prev * E_K                    # M pulls toward E_K  (batch_size,)
				+ input_current[:, i - 1]                  # (batch_size,)
				+ noise                                     # (batch_size,)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler voltage update
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)
			# (batch_size,)

			# HH gate updates (standard)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
			# (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
			# (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
			# (batch_size,)

			# M-current gate update using self-consistent voltage-dependent tau
			p[:, i] = p_ss + (p_prev - p_ss) * Exp(-tstep / tau_p_v)
			# (batch_size,)

		# Return voltage traces (observation noise fixed at 0.0 per task spec)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)
		# (batch_size, time_steps)