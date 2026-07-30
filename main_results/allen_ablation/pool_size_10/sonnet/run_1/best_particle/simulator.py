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
		Hodgkin-Huxley neuron extended with a slow M-type K+ current (I_M).

		M-current physiological rationale:
		  The classic HH model (Na+, K+ delayed-rectifier, leak) produces tonic spiking
		  but cannot capture voltage-distribution statistics (resting SD, variance, skewness,
		  kurtosis) or spike-count modulation at different stimulus intensities as well as
		  a model with a slow subthreshold adaptation current.

		  The M-current (Kv7/KCNQ family) is the simplest physiologically justified addition:
		    - Non-inactivating slow K+ conductance, activates near resting potential
		    - Provides tonic hyperpolarising drive that regularises inter-spike intervals
		    - Does NOT cause bursting, silencing, or high-frequency sustained firing
		    - Well-characterised in literature: Wang et al. 1998, Halliwell & Adams 1982

		Parameter mapping (corrected from prior iterations):
		  gbar_M   = params[:,6]          conductance slot X1, range [1e-4, 10] mS/cm2
		  params[:,7] (gbar_X2)           intentionally unused — parsimony, one channel only
		  V_half_w = params[:,8] - 75.0   half-activation voltage [mV]
		                                  raw prior [1e-4,150] maps to approx [-75, +75] mV
		                                  physiological target ~-50 mV => params[:,8] ~ 25
		                                  NO sign flip — direct centred parameterisation
		  tau_w    = params[:,9]          time constant [ms], range [1, 3000] ms
		                                  NO negation (critical fix: prior code used -params[:,9]
		                                  which always yielded negative values, collapsed to 1 ms)
		                                  physiology expects 50-300 ms for slow adaptation

		Args:
			init_voltage : torch.Tensor (batch_size,)      initial voltage [mV]
			input_current: torch.Tensor (batch_size, T)    applied current [uA/cm2]
			dt           : float                           time step [ms]
			t            : torch.Tensor (T,)               time array [ms]
			params       : torch.Tensor (batch_size, 10)   biophysical parameters
			seed         : int or None

		Returns:
			V            : torch.Tensor (batch_size, T)    membrane voltage [mV]
		"""
		device = params.device

		# ── Random generator ──────────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]  # scalar int
		time_steps = t.shape[0]       # scalar int

		# ── Parameter extraction ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  Na+ max conductance      [mS/cm2]
		gbar_K    = params[:, 1].float()   # (batch_size,)  K+ DR max conductance    [mS/cm2]
		g_leak    = params[:, 2].float()   # (batch_size,)  leak conductance          [mS/cm2]
		E_leak    = -params[:, 3].float()  # (batch_size,)  leak reversal (negated)   [mV]
		Vt        = -params[:, 4].float()  # (batch_size,)  voltage threshold (neg)   [mV]
		nois_fact = params[:, 5].float()   # (batch_size,)  noise amplitude           [unitless]

		# M-current: uses slot X1 for conductance, param_i for V_half, param_j for tau
		gbar_M   = params[:, 6].float()              # (batch_size,)  M-current conductance [mS/cm2]
		# params[:,7] (gbar_X2) is left unused for parsimony — one channel is sufficient

		# Half-activation voltage: raw prior [1e-4, 150] shifted by -75 to center near -50 mV
		# No sign flip: inference searches params[:,8] ~ 25 to hit V_half_w ~ -50 mV naturally
		V_half_w = params[:, 8].float() - 75.0      # (batch_size,)  [mV], range ~ [-75, +75]

		# Time constant: raw prior [1e-4, 3000] used DIRECTLY (no negation)
		# Critical: prior iterations applied '-' sign producing always-negative tau_w
		# which clamped universally to 1 ms, eliminating slow adaptation entirely.
		# Here we take params[:,9] as-is, giving access to the full [1, 3000] ms range.
		tau_w    = params[:, 9].float()              # (batch_size,)  [ms], always positive
		tau_w    = torch.clamp(tau_w, min=1.0)       # (batch_size,)  numerical safety floor

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0     # membrane capacitance [uF/cm2]
		E_Na = 53.0    # Na+ reversal potential [mV]
		E_K  = -107.0  # K+ reversal potential [mV] — shared by K+ DR and M-current

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically clamped exponential to prevent overflow
			# z: any shape -> same shape
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Borg-Graham form: avoids 0/0 singularity near z=0
			# z: any shape -> same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH channel kinetics (Vt-shifted Traub-Miles formulation) ────
		def alpha_m(x):  # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):   # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):  # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):   # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):  # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):   # x: (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):  # (batch_size,), (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):  # (batch_size,), (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current gating kinetics ─────────────────────────────────────────────
		# Steady-state: Boltzmann sigmoid with slope 10 mV (Halliwell & Adams 1982)
		# Activates gradually positive to V_half_w; provides subthreshold K+ drive
		def w_inf(x):  # x: (batch_size,) -> (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_w) / 10.0))

		# Time constant tau_w is voltage-independent (constant per batch element)
		# Keeps the model simple; inference learns the appropriate adaptation timescale

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) voltage [mV]
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) K+ DR gate
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) Na+ act. gate
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) Na+ inact. gate
		w = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T) M-current gate

		# ── Initialise at steady state ────────────────────────────────────────────
		V_init  = init_voltage.to(device)                                  # (batch_size,)
		V[:, 0] = V_init
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                # (batch_size,)
		w[:, 0] = w_inf(V[:, 0])                                           # (batch_size,)

		# ── Time integration (exact exponential integrator per step) ──────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)
			n_prev = n[:, i - 1]  # (batch_size,)
			m_prev = m[:, i - 1]  # (batch_size,)
			h_prev = h[:, i - 1]  # (batch_size,)
			w_prev = w[:, i - 1]  # (batch_size,)

			# Evaluate gating kinetics at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# Effective total conductance (scales membrane time constant)
			# M-current adds gbar_M * w — a slow voltage-dependent K+ term
			tau_V_inv = (
				(m_prev ** 3) * gbar_Na * h_prev   # (batch_size,)  Na+ conductance
				+ (n_prev ** 4) * gbar_K             # (batch_size,)  K+ DR conductance
				+ g_leak                              # (batch_size,)  passive leak
				+ gbar_M * w_prev                    # (batch_size,)  M-current conductance
			) / C                                    # (batch_size,)

			# Voltage steady-state numerator: weighted reversal potentials + inputs
			# M-current drives membrane toward E_K (hyperpolarising, adaptation)
			V_inf = (
				(m_prev ** 3) * gbar_Na * h_prev * E_Na   # (batch_size,)
				+ (n_prev ** 4) * gbar_K * E_K              # (batch_size,)
				+ g_leak * E_leak                            # (batch_size,)
				+ gbar_M * w_prev * E_K                     # (batch_size,)  M-current to E_K
				+ input_current[:, i - 1]                   # (batch_size,)  injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)                            # (batch_size,)

			# Exact exponential integration: V, n, m, h
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                              # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n_prev - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m_prev - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h_prev - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# M-current gate: exponential integration toward sigmoid steady-state
			# tau_w is positive (no negation) — allows slow adaptation on 50-300 ms timescale
			w_ss    = w_inf(V_prev)                                           # (batch_size,)
			w[:, i] = w_ss + (w_prev - w_ss) * Exp(-tstep / tau_w)           # (batch_size,)

		# ── Return voltage trace with optional observation noise ──────────────────
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)