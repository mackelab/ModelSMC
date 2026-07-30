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
		Hodgkin-Huxley neuron extended with an M-type K+ current (IKM).

		Design rationale for this iteration:
		---------------------------------------------------------------------------
		1. E_K CORRECTION: The base model uses E_K = -107 mV, which is ~30 mV more
		   hyperpolarised than the standard HH value of -77 mV. This caused severely
		   deep after-hyperpolarisations, biasing all seven summary statistics
		   simultaneously. Fixed to E_K = -77 mV throughout.

		2. M-CURRENT (IKM) — Boltzmann formulation:
		   Previous iterations used a symmetric exponential rate-constant formulation
		   (alpha_p / beta_p). That approach couples steady-state and time constant
		   through a single set of rates, making it hard for inference to independently
		   tune activation threshold vs. kinetics speed.

		   This iteration switches to the simpler, more identifiable Boltzmann form:
		     p_inf(V) = 1 / (1 + exp(-(V - V_half_M) / k_slope))
		     tau_p    = param_j   [ms, inferred directly from data]

		   This decouples the two tunable parameters cleanly:
		     - V_half_M = -param_i = -params[:,8]: half-activation voltage
		       params[:,8] in [1e-4, 150] → V_half_M in [-150, ~0] mV
		       Inference will find params[:,8] ~ 35, giving V_half_M ~ -35 mV
		     - tau_p    =  param_j =  params[:,9]: time constant directly in ms
		       params[:,9] in [1e-4, 3000] ms → covers IKM range of 20–300 ms
		       No multiplicative scaling confusion; inference reads tau_p directly

		   k_slope = 9 mV is fixed (standard IKM slope factor from Wang & McKinnon
		   1995; not a free parameter to keep the model parsimonious).

		   The M-current uses E_K as its reversal potential (same as delayed rectifier).

		3. PARAMETER MAPPING (follows signature exactly):
		   params[:,6] = gbar_M  (IKM conductance, mS/cm2, range [1e-4, 10])
		   params[:,7] = gbar_X2 (unused — one channel is sufficient)
		   params[:,8] = |param_i| → V_half_M = -params[:,8]  (mV)
		   params[:,9] = param_j  → tau_p = params[:,9]        (ms, positive)

		Args:
			init_voltage  : torch.Tensor: (batch_size,)             initial voltage (mV)
			input_current : torch.Tensor: (batch_size, time_steps)  injected current (uA/cm2)
			dt            : float                                    time step (ms)
			t             : torch.Tensor: (time_steps,)             time array (ms)
			params        : torch.Tensor: (batch_size, 10)          biophysical parameters
			seed          : int or None                              random seed

		Returns:
			V : torch.Tensor: (batch_size, time_steps)  membrane voltage traces (mV)
		"""
		device = params.device

		# ------------------------------------------------------------------ #
		# Random generator
		# ------------------------------------------------------------------ #
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar
		time_steps = t.shape[0]        # scalar

		# ------------------------------------------------------------------ #
		# Parameter extraction — each (batch_size,)
		# ------------------------------------------------------------------ #
		gbar_Na   = params[:, 0].float()   # (batch_size,)  fast Na+ conductance (mS/cm2)
		gbar_K    = params[:, 1].float()   # (batch_size,)  delayed-rectifier K+ conductance (mS/cm2)
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance (mS/cm2)
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal potential (mV)
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold shift (mV)
		nois_fact = params[:, 5].float()   # (batch_size,)  channel noise scale

		# ---- M-current (IKM): params[:,6], [:,8], [:,9] ---- #
		# gbar_M  : maximal IKM conductance (mS/cm2), prior [1e-4, 10]
		# params[:,7] (gbar_X2) intentionally unused — parsimony, one channel suffices
		# V_half_M: IKM half-activation voltage (mV) = -params[:,8]
		#           params[:,8] prior [1e-4,150] → V_half_M ∈ [-150, ~0]; expect ~-35 mV
		# tau_p   : IKM gating time constant (ms) = params[:,9] directly
		#           params[:,9] prior [1e-4, 3000] → spans 1–300 ms IKM range naturally
		gbar_M   = params[:, 6].float()   # (batch_size,)  IKM conductance (mS/cm2)
		V_half_M = -params[:, 8].float()  # (batch_size,)  IKM half-activation voltage (mV)
		tau_p    = params[:, 9].float()   # (batch_size,)  IKM time constant (ms, positive)

		tstep = float(dt)

		# ------------------------------------------------------------------ #
		# Fixed biophysical constants
		# ------------------------------------------------------------------ #
		nois_fact_obs = 0.0    # observation noise (zero — keep as instructed)
		C    = 1.0             # membrane capacitance (uF/cm2)
		E_Na = 53.0            # Na+ reversal potential (mV)
		# CRITICAL: corrected from base model's erroneous -107 mV to standard HH -77 mV.
		# The error caused ~30 mV too-deep after-hyperpolarisations, biasing every
		# summary statistic (spike count, mean V, variance, skewness, kurtosis).
		E_K  = -77.0           # K+ reversal potential (mV) — for IK and IKM
		# IKM Boltzmann slope factor (mV); fixed at 9 mV (standard value from
		# Wang & McKinnon 1995, J Physiol). Not a free parameter for parsimony.
		k_slope_M = 9.0        # (scalar, mV)

		# ------------------------------------------------------------------ #
		# Numerical helpers
		# ------------------------------------------------------------------ #
		def Exp(z):
			# (batch_size,) -> (batch_size,)  numerically clamped exponential
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# (batch_size,) -> (batch_size,)  HH helper: z / (exp(z) - 1)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ------------------------------------------------------------------ #
		# Standard HH gating kinetics
		# ------------------------------------------------------------------ #
		def alpha_m(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25   # (batch_size,)

		def beta_m(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2   # (batch_size,)

		def alpha_h(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)   # (batch_size,)

		def beta_h(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))   # (batch_size,)

		def alpha_n(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2   # (batch_size,)

		def beta_n(x):
			# (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)   # (batch_size,)

		def tau_x(alpha, beta):
			# (batch_size,), (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)   # (batch_size,)

		def inf_x(alpha, beta):
			# (batch_size,), (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)   # (batch_size,)

		# ------------------------------------------------------------------ #
		# M-current (IKM) steady-state — Boltzmann form
		#
		# p_inf(V) = 1 / (1 + exp(-(V - V_half_M) / k_slope_M))
		#
		# This is the standard Boltzmann steady-state for IKM (non-inactivating
		# slow K+ current; KCNQ/Kv7 channels). The sigmoid activates near the
		# threshold for action potential initiation (~-35 to -45 mV), providing
		# graded negative feedback that regularises ISIs without causing bursting.
		#
		# The time constant tau_p is inferred directly as params[:,9] in ms,
		# giving the inference engine a clean, linear handle on kinetics speed
		# (expected range 20–200 ms for physiological IKM).
		#
		# Advantages over symmetric-exponential (alpha_p/beta_p) formulation used
		# in prior iterations:
		#   • V_half_M and tau_p are orthogonal — no coupling between them
		#   • Prior on params[:,9] maps directly to ms without scaling ambiguity
		#   • Simpler code; fewer opportunities for kinetic instability
		# ------------------------------------------------------------------ #
		def p_inf(x):
			# (batch_size,) -> (batch_size,)  M-current steady-state activation
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / k_slope_M))   # (batch_size,)

		# ------------------------------------------------------------------ #
		# State variable allocation
		# ------------------------------------------------------------------ #
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) voltage
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ act
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inact
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IKM gate

		# ------------------------------------------------------------------ #
		# Initialise all gating variables at steady state for V_init
		# ------------------------------------------------------------------ #
		V_init = init_voltage.to(device)                                     # (batch_size,)
		V[:, 0] = V_init                                                      # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                   # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                   # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                   # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                              # (batch_size,)

		# ------------------------------------------------------------------ #
		# Main simulation loop — exponential Euler integration
		# ------------------------------------------------------------------ #
		for i in range(1, time_steps):
			# HH gating rates at previous timestep
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,), (batch_size,)

			# IKM steady-state at previous timestep voltage
			p_ss = p_inf(V[:, i - 1])   # (batch_size,)

			# Total effective conductance / C (determines membrane time constant)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) fast Na+
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) delayed-rectifier K+
				+ g_leak                                         # (batch_size,) leak
				+ p[:, i - 1] * gbar_M                          # (batch_size,) M-current IKM
			) / C   # (batch_size,)

			# Voltage steady-state: conductance-weighted reversal sum + external inputs
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ drive (E_K=-77)
				+ g_leak * E_leak                                        # (batch_size,) leak drive
				+ p[:, i - 1] * gbar_M * E_K                           # (batch_size,) IKM drive (E_K)
				+ input_current[:, i - 1]                               # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
				# (batch_size,) channel noise ~ N(0, nois_fact / sqrt(dt))
			) / (tau_V_inv * C)   # (batch_size,)

			# Exponential Euler voltage update (exact for linear ODE at fixed conductances)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler updates for standard HH gates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))   # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))   # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))   # (batch_size,)

			# Exponential Euler update for IKM p-gate with inferred tau_p (ms)
			# p(t+dt) = p_inf + (p(t) - p_inf) * exp(-dt / tau_p)
			# tau_p = params[:,9] directly in ms — clean, no scaling ambiguity
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_p)   # (batch_size,)

		# ------------------------------------------------------------------ #
		# Return voltage traces (+ observation noise, currently zero)
		# ------------------------------------------------------------------ #
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)