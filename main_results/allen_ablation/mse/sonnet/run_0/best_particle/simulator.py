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
		Hodgkin-Huxley neuron simulator extended with a transient A-type K+ current (IA).

		Design rationale (incorporating lessons from 3 prior iterations):

		  Iteration 1: Base HH → large skewness error (MSE=64.8), missing AHP mechanism.

		  Iteration 2: Added M-current (IKm) with V_half_m = -params[:,8] ≈ 0 mV at rest.
		    → Pathological over-activation; spike count MSE=19.5, variance MSE=19.2.
		    Problem: M-current was near-maximally open at rest, tonically suppressing firing.

		  Iteration 3: Fixed M-current with V_half_m ∈ [-50,-35] mV.
		    → Still spike count MSE=12.6, variance MSE=8.12.
		    Problem: M-current in [-50,-35] mV range still partially open at the inter-spike
		    trough (~-60 mV), providing a tonic outward K+ current that chronically
		    reduces membrane excitability.

		  Current iteration: Switch to A-type K+ current (IA).
		    Physiological rationale:
		      - IA (Kv4/Kv1 family) has rapid activation near -30 to -10 mV AND
		        fast inactivation with half-voltage near -65 mV (near rest).
		      - At resting potential (~-65 to -70 mV), the inactivation gate b ≈ 0.5-1.0
		        and activation gate a ≈ 0, so I_A ≈ 0: NO tonic suppression at rest.
		      - Upon depolarization toward threshold, IA activates transiently before
		        inactivating, producing a brief outward current that delays the next spike
		        and controls inter-spike interval (ISI).
		      - This shapes the voltage distribution correctly (fixing skewness, variance)
		        without tonically reducing spike count.
		      - Consistent with regular tonic spiking; does not produce bursting.

		Parameter mapping:
		  params[:,0]  → gbar_Na    [mS/cm2] Na+ max conductance
		  params[:,1]  → gbar_K     [mS/cm2] K+ delayed-rectifier
		  params[:,2]  → g_leak     [mS/cm2] leak conductance
		  params[:,3]  → |E_leak|   [mV]     leak reversal (negated internally)
		  params[:,4]  → |Vt|       [mV]     voltage threshold shift (negated internally)
		  params[:,5]  → nois_fact           process noise scale
		  params[:,6]  → gbar_A     [1e-4,10] mS/cm2  A-type K+ conductance
		  params[:,7]  → (unused)   [1e-4,120]         reserved
		  params[:,8]  → |param_i|  [1e-4,150] → V_half_act_A = -10 - 0.2*x ∈ [-40,-10] mV
		  params[:,9]  → |param_j|  [1e-4,3000] → tau_b_A = clamp(0.1*x, 10, 300) ms

		Args:
			init_voltage: torch.Tensor: (batch_size,)            initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) injected current (uA/cm2)
			dt: float                                             time step (ms)
			t: torch.Tensor: (time_steps,)                       time array (ms)
			params: torch.Tensor: (batch_size, 10)               biophysical parameters
			seed: optional int                                    for reproducibility

		Returns:
			V: torch.Tensor: (batch_size, time_steps)            membrane voltage traces (mV)
		"""
		device = params.device

		# Set up random generator for process noise
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int: number of parallel simulations
		time_steps = t.shape[0]        # int: number of time points

		# ---- Standard HH parameters ----
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm2 - Na+ max conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm2 - K+ delayed-rectifier
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm2 - leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV     - leak reversal (negated)
		Vt        = -params[:, 4].float()  # (batch_size,) mV     - voltage threshold shift (negated)
		nois_fact = params[:, 5].float()   # (batch_size,)         - process noise scale

		# ---- A-type K+ current (IA) parameters ----
		# Physiological rationale:
		#   IA is a transient K+ current that:
		#   (1) Fully inactivates at rest → NO tonic suppression of spiking
		#   (2) Activates transiently during depolarization → delays next spike onset
		#   (3) Recovers from inactivation during deep AHP → controls ISI in tonic firing
		#   This was chosen over M-current (IKm) because M-current in prior iterations
		#   caused chronic tonic hyperpolarization even when V_half_m was set to [-50,-35] mV.

		# Conductance: inferred, range [1e-4, 10] mS/cm2
		gbar_A      = params[:, 6].float()   # (batch_size,) mS/cm2

		# params[:,7] reserved/unused for this channel

		# Half-activation voltage: params[:,8] ∈ [1e-4, 150] → V_half_act ∈ [-40, -10] mV
		# Affine map: -10 - 0.2*x so that:
		#   x → 0   gives V_half_act = -10 mV  (least negative, least activation at rest)
		#   x → 150 gives V_half_act = -40 mV  (more negative, earlier activation onset)
		# Typical IA half-activation: -30 to -15 mV (well above rest)
		V_half_act_A = -10.0 - params[:, 8].float() * 0.2   # (batch_size,) mV ∈ [-40, -10]

		# Slow inactivation time constant: params[:,9] ∈ [1e-4, 3000] → tau_b ∈ [10, 300] ms
		# IA inactivation is slow relative to activation, governing ISI control
		# Clamp ensures numerical stability in Exp(-tstep/tau_b)
		tau_b_A = torch.clamp(params[:, 9].float() * 0.1, min=10.0, max=300.0)  # (batch_size,) ms

		# Fixed IA parameters (physiologically well-characterized)
		slope_act_A    = 10.0    # mV - Boltzmann activation slope (typical 8-12 mV)
		V_half_inact_A = -65.0   # mV - inactivation half-voltage (~rest); ensures b≈1 at rest
		slope_inact_A  = 7.0     # mV - inactivation slope (positive: b decreases with depol.)
		tau_a_A        = 1.0     # ms - fast activation time constant (fixed; IA activates in ~1 ms)

		tstep = float(dt)   # scalar ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise (kept at 0)
		C    = 1.0            # uF/cm2 - membrane capacitance
		E_Na = 53.0           # mV - sodium reversal potential
		E_K  = -107.0         # mV - potassium reversal (shared by IKdr and IA)

		####################################
		# Numerical helper functions

		def Exp(z):
			# Numerically stable exponential: clamp argument at -500 to prevent underflow
			# z: arbitrary-shape tensor
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# HH auxiliary: z/(exp(z)-1), L'Hopital near z=0 for stability
			# z: arbitrary-shape tensor
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ---- Standard HH Na+ channel kinetics ----
		def alpha_m(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 13.0   # (batch_size,)
			return 0.32 * efun(-0.25 * v1) / 0.25  # (batch_size,)

		def beta_m(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 40.0   # (batch_size,)
			return 0.28 * efun(0.2 * v1) / 0.2  # (batch_size,)

		def alpha_h(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 17.0   # (batch_size,)
			return 0.128 * Exp(-v1 / 18.0)  # (batch_size,)

		def beta_h(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 40.0   # (batch_size,)
			return 4.0 / (1 + Exp(-0.2 * v1))  # (batch_size,)

		# ---- Standard HH K+ delayed-rectifier kinetics ----
		def alpha_n(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 15.0   # (batch_size,)
			return 0.032 * efun(-0.2 * v1) / 0.2  # (batch_size,)

		def beta_n(x):
			# x: (batch_size,) mV
			v1 = x - Vt - 10.0   # (batch_size,)
			return 0.5 * Exp(-v1 / 40)  # (batch_size,)

		def tau_x(alpha, beta):
			# alpha, beta: (batch_size,) → time constant in ms
			return 1.0 / (alpha + beta)  # (batch_size,)

		def inf_x(alpha, beta):
			# alpha, beta: (batch_size,) → steady-state gate value
			return alpha / (alpha + beta)  # (batch_size,)

		# ---- A-type K+ current (IA) gate kinetics ----
		# Activation gate 'a': fast, Boltzmann steady state
		# At rest (~-65 mV), a_inf ~ 0 since V_half_act_A ∈ [-40,-10] mV >> rest
		# → IA is NOT activated at rest, avoiding tonic hyperpolarization
		def a_inf_A(x):
			# Steady-state fast activation gate (Boltzmann)
			# x: (batch_size,) mV
			return 1.0 / (1.0 + Exp(-(x - V_half_act_A) / slope_act_A))  # (batch_size,)

		# Inactivation gate 'b': slow, Boltzmann steady state
		# At rest (~-65 mV = V_half_inact_A), b_inf ~ 0.5; fully recovered below -80 mV
		# During spike AHP, b recovers → IA available for next spike cycle
		def b_inf_A(x):
			# Steady-state slow inactivation gate (Boltzmann, decreasing with depolarization)
			# x: (batch_size,) mV
			return 1.0 / (1.0 + Exp((x - V_half_inact_A) / slope_inact_A))  # (batch_size,)

		####################################

		# ---- Allocate state variable tensors ----
		V    = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) mV
		n    = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) K+ DR gate
		m    = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ act gate
		h    = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Na+ inact gate
		a_ch = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA activation gate
		b_ch = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) IA inactivation gate

		# ---- Initialize all gates at steady state ----
		V_init = init_voltage.to(device)                              # (batch_size,)
		V[:, 0]    = V_init                                           # (batch_size,)
		n[:, 0]    = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))        # (batch_size,)
		m[:, 0]    = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))        # (batch_size,)
		h[:, 0]    = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))        # (batch_size,)
		a_ch[:, 0] = a_inf_A(V[:, 0])                                 # (batch_size,) IA act SS at init V
		b_ch[:, 0] = b_inf_A(V[:, 0])                                 # (batch_size,) IA inact SS at init V

		# ---- Integration loop with exact exponential updates ----
		for i in range(1, time_steps):

			# HH rate constants at previous time step
			a_m = alpha_m(V[:, i - 1])  # (batch_size,)
			b_m = beta_m(V[:, i - 1])   # (batch_size,)
			a_h = alpha_h(V[:, i - 1])  # (batch_size,)
			b_h = beta_h(V[:, i - 1])   # (batch_size,)
			a_n = alpha_n(V[:, i - 1])  # (batch_size,)
			b_n = beta_n(V[:, i - 1])   # (batch_size,)

			# IA gate steady states at previous voltage
			a_ss = a_inf_A(V[:, i - 1])  # (batch_size,) fast activation SS
			b_ss = b_inf_A(V[:, i - 1])  # (batch_size,) slow inactivation SS

			# Effective membrane conductance inverse (1/tau_V = sum_channels(g)/C)
			# IA contribution: gbar_A * a^3 * b (transient; a≈0 at rest → near-zero contribution at rest)
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]         # (batch_size,) Na+ conductance
				+ (n[:, i - 1] ** 4) * gbar_K                        # (batch_size,) K+ DR conductance
				+ g_leak                                               # (batch_size,) leak conductance
				+ gbar_A * (a_ch[:, i - 1] ** 3) * b_ch[:, i - 1]   # (batch_size,) IA conductance
			) / C  # (batch_size,)

			# Voltage steady state: weighted sum of reversal potentials + external drive
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na          # (batch_size,) Na+ drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                          # (batch_size,) K+ DR drive
				+ g_leak * E_leak                                              # (batch_size,) leak drive
				+ gbar_A * (a_ch[:, i - 1] ** 3) * b_ch[:, i - 1] * E_K     # (batch_size,) IA drive (K+ reversal)
				+ input_current[:, i - 1]                                     # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # (batch_size,) noise
			) / (tau_V_inv * C)  # (batch_size,)

			# Exact exponential (analytic) update for membrane voltage
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)  # (batch_size,)

			# Exact exponential updates for standard HH gating variables
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# Exact exponential updates for IA gating variables
			# Fast activation: tau_a_A = 1.0 ms (fixed scalar, physiologically appropriate)
			a_ch[:, i] = a_ss + (a_ch[:, i - 1] - a_ss) * Exp(torch.full_like(a_ss, -tstep / tau_a_A))  # (batch_size,)

			# Slow inactivation: tau_b_A is (batch_size,) inferred parameter, clamped [10,300] ms
			b_ch[:, i] = b_ss + (b_ch[:, i - 1] - b_ss) * Exp(-tstep / tau_b_A)  # (batch_size,)

		# Return voltage traces with optional observation noise (currently 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)