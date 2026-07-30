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
		Hodgkin-Huxley neuron extended with a slow M-type K+ current (IKm).

		CHANNEL ADDITION RATIONALE:
		The base HH model produces tonic spiking but cannot simultaneously capture:
		  - Correct inter-spike intervals (mean voltage during stimulation)
		  - Correct voltage distribution moments (variance, skewness, kurtosis)
		  - Correct resting potential statistics
		The M-current (KCNQ/Kv7 family) is the canonical mechanism for spike-frequency
		adaptation in tonically spiking neurons. It activates slowly near subthreshold
		voltages and does NOT produce bursting, matching the data description exactly.

		PARAMETER SLOT ALLOCATION (strictly enforced per slot semantics):
		  params[:,0]  gbar_Na        Na+ max conductance         (mS/cm2)
		  params[:,1]  gbar_K         K+ max conductance          (mS/cm2)
		  params[:,2]  g_leak         leak conductance            (mS/cm2)
		  params[:,3]  |E_leak|       leak reversal (negated)     (mV)
		  params[:,4]  |Vt|           kinetics offset (negated)   (mV)
		  params[:,5]  nois_fact      noise amplitude             (unitless)
		  params[:,6]  gbar_Km        M-current conductance       (mS/cm2) [X1 slot]
		  params[:,7]  UNUSED         (X2 conductance slot reserved for future channel)
		  params[:,8]  param_i        M-gate half-activation      (mV offset from Vt, positive)
		                              range [1e-4, 150]: e.g. ~20-30 mV above Vt
		  params[:,9]  param_j        M-gate time constant        (ms, positive)
		                              range [1e-4, 3000]: M-current tau typically 20-200 ms

		CRITICAL FIX vs. prior iterations:
		  - params[:,8] and params[:,9] are used WITHOUT negation (they are positive by design)
		  - params[:,7] is deliberately left unused (it is the gbar_X2 slot, not a kinetics slot)
		  - This corrects the parameter slot misalignment that caused inference failures

		Args:
			init_voltage : torch.Tensor (batch_size,)             initial membrane voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps)  injected current (uA/cm2)
			dt           : float                                   time step (ms)
			t            : torch.Tensor (time_steps,)             time array (ms)
			params       : torch.Tensor (batch_size, 10)          biophysical parameters
			seed         : int or None                            random seed

		Returns:
			V            : torch.Tensor (batch_size, time_steps)  membrane voltage (mV)
		"""
		device = params.device

		# Random generator for stochastic noise
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # scalar int
		time_steps = t.shape[0]        # scalar int

		# ── Base HH parameters ─────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) Na+ max conductance   (mS/cm2)
		gbar_K    = params[:, 1].float()   # (batch_size,) K+ max conductance    (mS/cm2)
		g_leak    = params[:, 2].float()   # (batch_size,) leak conductance      (mS/cm2)
		E_leak    = -params[:, 3].float()  # (batch_size,) leak reversal         (mV, sign applied)
		Vt        = -params[:, 4].float()  # (batch_size,) kinetics offset       (mV, sign applied)
		nois_fact = params[:, 5].float()   # (batch_size,) noise amplitude       (unitless)

		# ── M-current parameters: X1 conductance slot + param_i/param_j ───────
		# gbar_Km: maximal M-current conductance, range [1e-4, 10] mS/cm2
		# v_half_offset: half-activation measured as positive offset above Vt
		#   v_half = Vt + v_half_offset
		#   With Vt ~ -60 mV and offset ~ 20-30, v_half ~ -35 mV (physiological)
		#   Inferred from params[:,8], range [1e-4, 150] mV — used POSITIVE (no negation)
		# tau_Km: slow M-gate time constant, range [1e-4, 3000] ms
		#   Typical M-current tau: 20–200 ms; inference locates the correct timescale
		#   Inferred from params[:,9], range [1e-4, 3000] ms — used POSITIVE (no negation)
		gbar_Km      = params[:, 6].float()   # (batch_size,) M-current conductance    (mS/cm2)
		# params[:, 7] intentionally unused — this is the gbar_X2 slot, not a kinetics param
		v_half_offset = params[:, 8].float()  # (batch_size,) M-gate half-act offset   (mV, positive)
		tau_Km        = params[:, 9].float()  # (batch_size,) M-gate time constant     (ms, positive)

		tstep = float(dt)   # scalar float (ms)

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (disabled)
		C    = 1.0            # membrane capacitance   (uF/cm2)
		E_Na = 53.0           # Na+ reversal potential (mV)
		E_K  = -107.0         # K+ reversal potential  (mV); also used for M-current (K channel)

		# ── Numerical helper functions ──────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential — clamp exponent at -500 to prevent underflow
			# z: (batch_size,) -> (batch_size,)
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)

		def efun(z):
			# Stable evaluation of z/(exp(z)-1), L'Hopital branch near z=0
			# z: (batch_size,) -> (batch_size,)
			return torch.where(
				torch.abs(z) < 1e-4,
				1.0 - z / 2.0,
				z / (Exp(z) - 1.0)
			)

		# ── Standard HH gate kinetics (unchanged from base model) ─────────────
		def alpha_m(x):
			# Na+ fast activation opening rate  (batch_size,) -> (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ fast activation closing rate  (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation opening rate     (batch_size,) -> (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation closing rate     (batch_size,) -> (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ delayed-rectifier opening rate (batch_size,) -> (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ delayed-rectifier closing rate (batch_size,) -> (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# Gate time constant from rate functions  (batch_size,) -> (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state from rate functions   (batch_size,) -> (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (IKm) gate kinetics ──────────────────────────────────────
		# Slow, non-inactivating K+ activation gate p (Boltzmann steady state).
		# v_half is inferred per batch element: v_half = Vt + v_half_offset
		#   where v_half_offset > 0 places activation above Vt (subthreshold range).
		# Slope factor k = 10 mV: standard physiological value for KCNQ/M-channels.
		# Time constant tau_Km is constant (not voltage-dependent) for parsimony.
		def p_inf(x):
			# M-gate Boltzmann steady-state activation  (batch_size,) -> (batch_size,)
			v_half = Vt + v_half_offset   # (batch_size,) inferred half-activation voltage
			return 1.0 / (1.0 + Exp(-(x - v_half) / 10.0))

		# ── Allocate state variable tensors ────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) voltage (mV)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inactivation
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ activation
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-gate activation

		# ── Initialization: all gates at voltage steady state ─────────────────
		V_init  = init_voltage.to(device)                              # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))            # (batch_size,) Na+ m at rest
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))            # (batch_size,) Na+ h at rest
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))            # (batch_size,) K+ n at rest
		p[:, 0] = p_inf(V[:, 0])                                       # (batch_size,) M-gate p at rest

		# ── Exponential Euler time integration ─────────────────────────────────
		# Exponential Euler is exact for linear (piecewise-constant-conductance) ODEs,
		# providing numerical stability at larger time steps vs. forward Euler.
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]   # (batch_size,) voltage at previous step

			# Compute HH gate rate constants at previous voltage
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-gate steady state at previous voltage
			p_ss = p_inf(V_prev)   # (batch_size,)

			# ── Effective membrane conductance (inverse RC time constant) ──────
			# Sum of all active conductances: determines voltage relaxation rate
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ conductance
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) K+ delayed-rectifier
				+ g_leak                                         # (batch_size,) leak
				+ gbar_Km * p[:, i - 1]                         # (batch_size,) M-current (slow K+)
			) / C   # (batch_size,)

			# ── Voltage steady-state numerator (weighted reversal potentials) ──
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ driving force
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ driving force
				+ g_leak * E_leak                                       # (batch_size,) leak driving force
				+ gbar_Km * p[:, i - 1] * E_K                          # (batch_size,) M-current (K+ rev)
				+ input_current[:, i - 1]                               # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)   # (batch_size,)

			# ── State variable updates (exponential Euler) ────────────────────
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)                                    # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			# M-gate: exponential Euler with inferred constant time constant tau_Km
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_Km)                                   # (batch_size,)

		# Return voltage trace with optional observation noise (currently 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)