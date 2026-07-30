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
		Hodgkin-Huxley neuron simulator extended with a slow non-inactivating
		M-type K⁺ current (IKM) to capture spike-frequency adaptation in tonic spiking.

		Design decisions informed by iterative feedback:

		1. E_K corrected from -107 mV to -80 mV.
		   The original -107 mV was far outside physiological range (~-77 to -90 mV),
		   causing systematically too-deep AHPs and distorting voltage variance, skewness,
		   and kurtosis in ways no inferred parameter could compensate. Standard HH models
		   use E_K ≈ -77 to -90 mV; -80 mV is chosen as a conservative central value.

		2. IKM occupies X1 slot with both tunable params (param_i → V_half_M, param_j → tau_p).
		   Parsimony principle: one well-characterized channel with both tunable parameters
		   is preferred over two channels with split parameters (avoids identifiability issues).

		3. V_half_M mapped to [-40, -20] mV (improved from prior [-55,-25] and [-40,-15] ranges).
		   At rest (-65 mV): p_inf ≈ 0.006 (V_half=-20) to 0.067 (V_half=-40) — nearly silent.
		   At threshold (~-30 mV): p_inf ≈ 0.5 (V_half=-30) — robustly activating.
		   This cleanly separates resting and spiking activation regimes.

		4. tau_p mapped to [20, 170] ms via /20.0 + 20.0 (full 150 ms optimizer-accessible range).
		   Physiological IKM time constants are 20-200 ms; this provides genuine sensitivity.

		5. gbar_X2 (params[:,7]) unused — parsimony principle maintained.

		Args:
			init_voltage: torch.Tensor: (batch_size,) # initial membrane voltage (mV)
			input_current: torch.Tensor: (batch_size, time_steps) # injected current (uA/cm2)
			dt: float # time step (ms)
			t: torch.Tensor: (time_steps,) # time array (ms)
			params: torch.Tensor: (batch_size, 10) # biophysical parameters
			seed: optional int random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps) # voltage traces (mV)
		"""
		device = params.device

		# Set up reproducible random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ──────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²  Na maximal conductance
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²  K delayed-rectifier conductance
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²  leak conductance
		E_leak    = -params[:, 3].float()  # (batch_size,) mV      leak reversal (sign applied here)
		Vt        = -params[:, 4].float()  # (batch_size,) mV      voltage threshold shift
		nois_fact = params[:, 5].float()   # (batch_size,) unitless stochastic noise amplitude

		# ── X1 slot: M-type K⁺ current (IKM) ────────────────────────────────────
		# Physiological rationale:
		#   IKM (KCNQ/Kv7 channels) is a slow, subthreshold-activating, non-inactivating
		#   K⁺ current found ubiquitously in cortical and hippocampal neurons.
		#   Key properties:
		#   - Activates slowly near/above spike threshold, deactivates slowly at rest
		#   - Provides progressive spike-frequency adaptation during tonic spiking
		#   - Does NOT produce burst firing — activation simply reduces excitability
		#   - Directly explains discrepancies in: spike count, mean voltage during
		#     stimulation, and voltage distribution shape (skewness, kurtosis)
		#
		# Parameter assignments:
		#   gbar_KM  = params[:,6]: maximal IKM conductance [1e-4, 10] mS/cm²
		#   V_half_M = params[:,8]: half-activation voltage of p-gate
		#     raw range [1e-4, 150] → [-40, -20] mV via: -(param * 0.133 + 20.0)
		#     At param≈0:   V_half_M ≈ -20.0 mV → p_inf(-65mV) ≈ 0.006 (nearly silent at rest)
		#     At param=150: V_half_M ≈ -40.0 mV → p_inf(-65mV) ≈ 0.067 (still small at rest)
		#     At V≈-30 mV:  p_inf ≈ 0.5 to 0.81 (well activated during spiking)
		#   tau_p    = params[:,9]: M-current time constant
		#     raw range [1e-4, 3000] → [20, 170] ms via: param/20.0 + 20.0
		#     Full 150 ms optimizer-accessible range covers physiological IKM kinetics
		#
		# X2 slot (params[:,7]) intentionally unused — parsimony principle.

		gbar_KM  = params[:, 6].float()   # (batch_size,) mS/cm²  IKM maximal conductance

		# Half-activation: raw [1e-4, 150] → [-40, -20] mV
		# Keeps IKM nearly silent at resting potential (~-65 mV)
		# Activates robustly only near/above spike threshold
		V_half_M = -(params[:, 8].float() * 0.133 + 20.0)   # (batch_size,) mV ∈ [-40, -20]

		# Slow time constant: raw [1e-4, 3000] → [20, 170] ms
		# Lower bound 20 ms ensures meaningful adaptation within spike trains
		# Upper bound 170 ms allows slow sustained adaptation across seconds
		tau_p = params[:, 9].float() / 20.0 + 20.0   # (batch_size,) ms ∈ [20, 170]

		tstep = float(dt)   # scalar ms

		# Fixed biophysical constants
		nois_fact_obs = 0.0   # observation noise amplitude (kept at 0.0 per task spec)
		C    = 1.0            # membrane capacitance uF/cm²
		E_Na = 53.0           # Na⁺ reversal potential mV (standard HH)

		# CRITICAL: E_K corrected from original -107 mV to -80 mV
		# Original -107 mV produced pathologically deep AHPs, distorting voltage
		# variance, skewness, and kurtosis beyond optimizer compensation range.
		# -80 mV is physiologically standard (Nernst equation for typical [K⁺] gradients).
		# Both delayed-rectifier K⁺ and IKM share this K⁺ reversal potential.
		E_K  = -80.0          # K⁺ reversal potential mV (corrected from -107 mV)

		# ── Numerical helpers ────────────────────────────────────────────────────
		def Exp(z):
			# Numerically stable exponential: clips extreme negative values to prevent underflow
			# z: (batch_size,) → (batch_size,)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# Exponential-linear function for HH rate expressions
			# Avoids 0/0 singularity at z≈0 via first-order Taylor approximation
			# z: (batch_size,) → (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1.0 - z / 2.0, z / (Exp(z) - 1.0))

		# ── Standard HH gate kinetics (unmodified from base) ────────────────────
		def alpha_m(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1.0 + Exp(-0.2 * v1))

		def alpha_n(x):   # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):    # x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40.0)

		def tau_x(alpha, beta):   # (batch_size,),(batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):   # (batch_size,),(batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── IKM p-gate: Boltzmann steady-state activation ────────────────────────
		# Slope k=10 mV is physiologically standard for M-current (Borda & Cooper 2021)
		# V_half_M ∈ [-40, -20] mV ensures minimal activation at rest (-65 mV)
		# and meaningful activation only during action potential generation
		def p_inf(x):   # x: (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_M) / 10.0))

		# ── State variable allocation ────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K DR activation
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na inactivation
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) IKM activation

		# ── Initial conditions: gating variables at steady state ─────────────────
		V_init = init_voltage.to(device)   # (batch_size,)
		V[:, 0] = V_init                                                       # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))                    # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))                    # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))                    # (batch_size,)
		p[:, 0] = p_inf(V[:, 0])                                               # (batch_size,) ≈ 0.006-0.07 at rest

		# ── Time integration: exponential Euler ──────────────────────────────────
		for i in range(1, time_steps):
			# Gate rate constants at previous time step
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,),(batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,),(batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,),(batch_size,)

			# IKM p-gate steady state at previous voltage
			p_inf_v = p_inf(V[:, i - 1])   # (batch_size,)

			# Effective inverse membrane time constant (total conductance / C)
			# Each term is the conductance contribution from one current
			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # Na conductance    (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K                  # K DR conductance  (batch_size,)
				+ g_leak                                         # leak conductance  (batch_size,)
				+ gbar_KM * p[:, i - 1]                         # IKM conductance   (batch_size,)
			) / C                                                # (batch_size,)

			# Voltage steady state: conductance-weighted reversal potentials + external inputs
			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # Na drive      (batch_size,)
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # K DR drive    (batch_size,)
				+ g_leak * E_leak                                       # leak drive    (batch_size,)
				+ gbar_KM * p[:, i - 1] * E_K                          # IKM drive     (batch_size,) — hyperpolarising, same E_K
				+ input_current[:, i - 1]                              # injected      (batch_size,)
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)  # noise (batch_size,)
			) / (tau_V_inv * C)                                        # (batch_size,)

			# Exponential Euler voltage update (exact for piecewise-constant conductances)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,)

			# Exponential Euler updates for standard HH gating variables
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# Exponential Euler update for IKM p-gate
			# tau_p ∈ [20, 170] ms: constant-tau simplification for parsimony
			# IKM tau varies only ~2-fold over physiological voltage range compared
			# to the orders-of-magnitude variation in faster HH gates
			p[:, i] = p_inf_v + (p[:, i - 1] - p_inf_v) * Exp(-tstep / tau_p)   # (batch_size,)

		# Return voltage traces with optional observation noise (currently 0.0)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps)