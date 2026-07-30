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
		Hodgkin-Huxley neuron extended with a transient A-type K+ current (I_A).

		Physiological rationale for I_A (Kv4 / Shal family):
		  - Rapidly activating (~1 ms) and inactivating (~10-80 ms) outward K+ current.
		  - Active at subthreshold voltages; recovers from inactivation during inter-spike
		    hyperpolarisation, making it available again before each successive spike.
		  - Because I_A inactivates during the spike itself and recovers symmetrically
		    between spikes, it promotes EVENLY SPACED inter-spike intervals WITHOUT
		    cumulative adaptation — directly matching the tonic, non-adapting spiking
		    pattern observed in the experimental data.
		  - Unlike I_M (which accumulates during repetitive firing and causes progressive
		    ISI lengthening), I_A is a non-adapting ISI-regularising mechanism.

		History of design decisions:
		  - I_M was tried in prior iterations but caused spike-frequency adaptation
		    (progressive ISI lengthening), contradicting the regular tonic spiking data.
		  - I_A chosen as replacement: transient inactivation resets each cycle,
		    producing evenly-spaced spikes consistent with all data constraints.

		Parameter slot usage (X1 only, parsimony principle):
		  params[:,6] = gbar_A     : max conductance (mS/cm²), range [1e-4, 10]
		  params[:,7] = tau_b_scale: inactivation time-constant scale (ms), range [1e-4, 120]
		  V_half_act  fixed at -40 mV  (canonical Kv4 activation midpoint)
		  V_half_inact fixed at -60 mV (canonical Kv4 inactivation midpoint)
		  tau_a        fixed at 1.5 ms  (fast activation, typical Kv4)

		Args:
			init_voltage  : torch.Tensor (batch_size,)     - initial voltage (mV)
			input_current : torch.Tensor (batch_size, T)   - injected current (uA/cm²)
			dt            : float                          - time step (ms)
			t             : torch.Tensor (T,)              - time array (ms)
			params        : torch.Tensor (batch_size, 10)  - biophysical parameters
			seed          : int or None

		Returns:
			V             : torch.Tensor (batch_size, T)   - voltage traces (mV)
		"""
		device = params.device

		# ---------- random generator ----------
		generator = torch.Generator(device=device)
		if seed is not None:
			generator.manual_seed(seed)

		batch_size = params.shape[0]   # scalar
		time_steps = t.shape[0]        # scalar

		# ---------- parameter extraction ----------
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (stored positive, negated here)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (stored positive, negated here)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# X1 slot → transient A-type K+ current (I_A)
		# params[:,6]: max conductance (mS/cm²), range [1e-4, 10]
		# params[:,7]: inactivation time constant scale (ms), range [1e-4, 120]
		#   Physiological I_A inactivation: 10-80 ms → inferred values in [1e-4, 120] cover this range.
		# X2 slot (params[:,8], params[:,9]) intentionally unused — parsimony principle.
		gbar_A      = params[:, 6].float()  # (batch_size,)  mS/cm²
		tau_b_scale = params[:, 7].float()  # (batch_size,)  ms, I_A inactivation time-constant scale

		tstep = float(dt)

		# ---------- fixed biophysical constants ----------
		nois_fact_obs = 0.0
		C    = 1.0     # uF/cm²
		E_Na = 53.0    # mV
		E_K  = -107.0  # mV  (reversal for both delayed-rectifier and A-current)

		# I_A kinetic constants (fixed at canonical Kv4 values)
		V_half_act   = -40.0  # mV  — half-activation voltage (scalar)
		V_half_inact = -60.0  # mV  — half-inactivation voltage (scalar)
		k_act        =  8.0   # mV  — activation slope factor
		k_inact      =  6.0   # mV  — inactivation slope factor
		tau_a_fixed  =  1.5   # ms  — fixed fast activation time constant

		# ---------- numerical helpers ----------
		def Exp(z):
			# z: any shape → same shape;  clamps at -500 for numerical stability
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# z: any shape → same shape
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ---------- standard HH channel kinetics ----------
		def alpha_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0  # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0  # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0  # (batch_size,)
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0  # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0  # (batch_size,)
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):
			# (batch_size,), (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# (batch_size,), (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ---------- A-type K+ current (I_A) kinetics ----------
		# Activation gate (a_A): fast Boltzmann sigmoid, centred at V_half_act (-40 mV)
		#   As V rises toward threshold, a_A rapidly increases, opening I_A.
		#   This outward current transiently opposes depolarisation just before each spike,
		#   introducing a brief delay that homogenises inter-spike intervals.
		def a_inf(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_act) / k_act))

		# Inactivation gate (b_A): slower sigmoid, centred at V_half_inact (-60 mV)
		#   At resting potential (~-65 mV), b_A ≈ 0.6 (partial de-inactivation).
		#   During the spike (V >> -60 mV), b_A inactivates → I_A turns off.
		#   During after-hyperpolarisation (V << -60 mV), b_A recovers → I_A available again.
		#   This cycle repeats identically for each spike → evenly spaced ISIs.
		def b_inf(x):
			# x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((x - V_half_inact) / k_inact))

		# Inactivation time constant: tunable via tau_b_scale (params[:,7])
		#   Voltage-independent for simplicity (can be refined if needed).
		#   tau_b_scale directly sets the inactivation time constant in ms.
		#   Larger tau_b_scale → slower inactivation recovery → I_A stays available longer.
		def tau_b(x):
			# x: (batch_size,) → (batch_size,)  [shape kept for consistent API]
			# Mild voltage dependence: slightly faster at depolarised voltages
			v1 = (x - V_half_inact) / 30.0   # (batch_size,)  normalised deviation
			denom = Exp(v1 * 0.5) + Exp(-v1 * 0.5) + 1e-6  # (batch_size,)  ≥ 2
			# Peak value = tau_b_scale at V = V_half_inact
			return 2.0 * tau_b_scale / denom  # (batch_size,)  ms

		# ---------- state variable allocation ----------
		V   = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  mV
		n   = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  K+ gate
		m   = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  Na+ act
		h   = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  Na+ inact
		# A-current gates: activation (a_A) and inactivation (b_A)
		a_A = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  fast act
		b_A = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, T)  slow inact

		# ---------- initialisation at voltage-dependent steady states ----------
		V_init      = init_voltage.to(device)                         # (batch_size,)
		V[:, 0]     = V_init                                          # (batch_size,)
		n[:, 0]     = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))       # (batch_size,)
		m[:, 0]     = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))       # (batch_size,)
		h[:, 0]     = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))       # (batch_size,)
		# Initialise I_A gates at their steady states for the initial voltage
		a_A[:, 0]   = a_inf(V[:, 0])                                  # (batch_size,)
		b_A[:, 0]   = b_inf(V[:, 0])                                  # (batch_size,)

		# ---------- main simulation loop (exponential Euler integration) ----------
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Standard HH rate constants
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)  # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)  # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)  # (batch_size,), (batch_size,)

			# I_A gate steady states and time constants at previous voltage
			a_ss  = a_inf(V_prev)              # (batch_size,)  activation steady state
			b_ss  = b_inf(V_prev)              # (batch_size,)  inactivation steady state
			tau_bv = tau_b(V_prev)             # (batch_size,)  inactivation time constant (ms)

			# Effective conductances weighted by gating variables
			gNa_eff = (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]  # (batch_size,)
			gK_eff  = (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,)
			# I_A: fast activation (4th power by convention, consistent with Kv4),
			# slow inactivation gate; reversal = E_K
			gA_eff  = gbar_A * (a_A[:, i - 1] ** 4) * b_A[:, i - 1]  # (batch_size,)

			# Inverse membrane time constant (exponential Euler denominator)
			tau_V_inv = (
				gNa_eff
				+ gK_eff
				+ g_leak
				+ gA_eff    # I_A contributes transiently near threshold
			) / C  # (batch_size,)

			# Effective voltage steady state (numerator of linearised membrane ODE)
			V_inf = (
				gNa_eff * E_Na
				+ gK_eff  * E_K
				+ g_leak  * E_leak
				+ gA_eff  * E_K    # I_A drives V toward E_K transiently near threshold
				+ input_current[:, i - 1]
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler voltage update
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Exponential Euler updates for standard HH gates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# Exponential Euler updates for I_A gates
			# Fast activation: fixed tau_a_fixed (1.5 ms), tracks voltage changes rapidly
			a_A[:, i] = a_ss + (a_A[:, i-1] - a_ss) * Exp(torch.full_like(V_prev, -tstep / tau_a_fixed))  # (batch_size,)
			# Slow inactivation: tunable tau_bv, recovers during inter-spike hyperpolarisation
			b_A[:, i] = b_ss + (b_A[:, i-1] - b_ss) * Exp(-tstep / (tau_bv + 1e-6))  # (batch_size,)

		# ---------- return voltage traces ----------
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, T)