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
		Hodgkin-Huxley neuron simulator extended with a slow M-type K+ current (IKM).

		MODIFICATION RATIONALE:
		The experimental data shows regular, tonic spiking with evenly-spaced action
		potentials. The base HH model tends to produce spike-frequency acceleration
		that does not match this pattern. The M-current (Kv7/KCNQ channels) is a
		well-characterized, non-inactivating K+ current that activates subthreshold
		(typically near -40 to -35 mV) with slow kinetics (tau ~ 100-500 ms). It
		acts as a brake on repetitive firing, producing regular, adapting spike trains
		without bursting — exactly the dynamics observed experimentally.

		CHANNEL ADDED:
		- IKM (M-current): slow, non-inactivating K+ current via gate `p`
		  * Slot: X1 (params[:,6] = gbar_M, params[:,8] = half-act offset, params[:,9] = tau_p)
		  * Reversal potential: E_K (same as delayed-rectifier K+)
		  * Gate p: single activation gate, no inactivation
		  * Steady-state: sigmoid with half-activation at (Vt + param_i) mV
		  * Time constant: param_j ms (voltage-independent, slow)

		PARAMETER LAYOUT (params: batch_size x 10):
		  [0] gbar_Na  (mS/cm²)   — Na+ max conductance
		  [1] gbar_K   (mS/cm²)   — delayed-rectifier K+ max conductance
		  [2] g_leak   (mS/cm²)   — passive leak conductance
		  [3] |E_leak| (mV)       — leak reversal magnitude (sign applied: E_leak = -params[:,3])
		  [4] |Vt|     (mV)       — voltage threshold offset (sign applied: Vt = -params[:,4])
		  [5] nois_fact            — process noise scale factor
		  [6] gbar_M   (mS/cm²)   — M-current max conductance [range: 1e-4, 10]
		  [7] gbar_X2  (mS/cm²)   — unused (reserved for future channel)
		  [8] |param_i| (mV)      — M-current p-gate half-activation offset above Vt [range: 1e-4, 150]
		                             typical: ~20-30 mV (so half-act ≈ Vt + 20 ≈ -38 mV)
		  [9] |param_j| (ms)      — M-current p-gate time constant [range: 1e-4, 3000]
		                             typical: ~100-500 ms (characteristically slow)

		Args:
			init_voltage: torch.Tensor (batch_size,)            — initial voltage (mV)
			input_current: torch.Tensor (batch_size, time_steps) — injected current (μA/cm²)
			dt: float                                            — integration time step (ms)
			t: torch.Tensor (time_steps,)                       — time array (ms)
			params: torch.Tensor (batch_size, 10)               — biophysical parameters
			seed: int or None                                    — random seed

		Returns:
			V: torch.Tensor (batch_size, time_steps)            — membrane voltage traces (mV)
		"""
		device = params.device

		# ── Random generator setup ────────────────────────────────────────────────
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Parameter extraction ──────────────────────────────────────────────────
		# Base HH parameters (unchanged from base model)
		gbar_Na   = params[:, 0].float()   # (batch_size,) mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,) mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,) mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,) mV  (stored as positive, negated here)
		Vt        = -params[:, 4].float()  # (batch_size,) mV  (stored as positive, negated here)
		nois_fact = params[:, 5].float()   # (batch_size,) unitless

		# M-current parameters (X1 slot)
		# gbar_M: max conductance of the slow M-type K+ channel
		gbar_M  = params[:, 6].float()    # (batch_size,) mS/cm², range [1e-4, 10]

		# params[:, 7] = gbar_X2: unused this iteration (parsimony: one channel at a time)

		# param_i: half-activation offset above Vt (mV)
		# Stored as POSITIVE value per signature (|param_i|); used directly as positive offset.
		# Example: Vt = -58 mV, param_i = 23 → p-gate half-activation at -35 mV (physiological)
		param_i = params[:, 8].float()    # (batch_size,) mV, range [1e-4, 150], NOT negated

		# param_j: M-current gate time constant (ms)
		# Stored as POSITIVE value per signature (|param_j|); used directly as tau_p.
		# M-current is characteristically slow: physiological range 100-500 ms
		param_j = params[:, 9].float()    # (batch_size,) ms, range [1e-4, 3000], NOT negated

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0   # observation noise (keep exactly as base model)
		C    = 1.0            # membrane capacitance, uF/cm²
		E_Na = 53.0           # Na+ reversal potential, mV
		E_K  = -107.0         # K+ reversal potential, mV (shared by all K+ channels)

		# ── Numerical stability helpers ───────────────────────────────────────────
		def Exp(z):
			# Clamped exponential to prevent overflow; z: (batch_size,) → (batch_size,)
			return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

		def efun(z):
			# Stable evaluation of z/(exp(z)-1) using Taylor expansion near z=0
			# z: (batch_size,) → (batch_size,)
			return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

		# ── Standard HH channel kinetics (unchanged from base model) ─────────────
		def alpha_m(x):
			# Na+ activation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 13.0
			return 0.32 * efun(-0.25 * v1) / 0.25

		def beta_m(x):
			# Na+ activation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40
			return 0.28 * efun(0.2 * v1) / 0.2

		def alpha_h(x):
			# Na+ inactivation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 17.0
			return 0.128 * Exp(-v1 / 18.0)

		def beta_h(x):
			# Na+ inactivation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 40.0
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):
			# K+ activation opening rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):
			# K+ activation closing rate; x: (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40)

		def tau_x(alpha, beta):
			# Gate time constant from opening/closing rates; (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):
			# Gate steady-state from opening/closing rates; (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# M-current (IKM) gate kinetics
		# The p gate is a simple sigmoidal activation (no inactivation for M-current).
		# Half-activation at (Vt + param_i) mV with slope 10 mV (standard Kv7 value).
		# When V >> half-activation: p_inf → 1 (channel open, outward K+ current).
		# When V << half-activation: p_inf → 0 (channel closed at rest/hyperpolarized).
		# The voltage-independent time constant param_j captures the slow M-current kinetics.
		def p_inf(x):
			# Sigmoid steady-state for M-current activation gate
			# x: (batch_size,) → (batch_size,)
			# v1: departure from half-activation voltage (positive = above half-act)
			v1 = x - Vt - param_i   # (batch_size,) mV
			return 1.0 / (1.0 + Exp(-v1 / 10.0))   # (batch_size,) dimensionless, in [0, 1]
		# ===== END EDITABLE SECTION =====

		# ── State variable allocation ─────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) mV
		n = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) K+ gate
		m = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ activation
		h = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) Na+ inactivation

		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# M-current activation gate state variable
		p = torch.zeros((batch_size, time_steps), device=device)   # (batch_size, time_steps) M-current gate
		# ===== END EDITABLE SECTION =====

		# ── Steady-state initialisation ───────────────────────────────────────────
		V_init = init_voltage.to(device)                             # (batch_size,)
		V[:, 0] = V_init                                             # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))          # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))          # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))          # (batch_size,)

		# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
		# Initialise M-current gate at its voltage-dependent steady state
		p[:, 0] = p_inf(V[:, 0])   # (batch_size,) — steady-state at initial voltage
		# ===== END EDITABLE SECTION =====

		# ── Main simulation loop ──────────────────────────────────────────────────
		for i in range(1, time_steps):
			# Standard HH gate kinetic rates at previous time step
			a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])   # (batch_size,), (batch_size,)

			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# Compute M-current gate steady-state at previous voltage
			p_ss = p_inf(V[:, i - 1])   # (batch_size,) — target for p gate update
			# ===== END EDITABLE SECTION =====

			tau_V_inv = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]   # (batch_size,) Na+ conductance term
				+ (n[:, i - 1] ** 4) * gbar_K                  # (batch_size,) delayed-rectifier K+
				+ g_leak                                         # (batch_size,) passive leak
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				# M-current contributes outward K+ conductance proportional to p gate
				# During depolarization: p increases → tau_V_inv increases → faster relaxation
				# This shortens the effective time constant and stabilises membrane potential
				+ gbar_M * p[:, i - 1]                          # (batch_size,) M-current conductance
				# ===== END EDITABLE SECTION =====
			) / C   # (batch_size,) ms⁻¹

			V_inf = (
				(m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na   # (batch_size,) Na+ current drive
				+ (n[:, i - 1] ** 4) * gbar_K * E_K                   # (batch_size,) K+ current drive
				+ g_leak * E_leak                                        # (batch_size,) leak current drive
				# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
				# M-current drives voltage toward E_K (hyperpolarizing during depolarization)
				# This is the key mechanism that regularizes spike timing and prevents bursting
				+ gbar_M * p[:, i - 1] * E_K                           # (batch_size,) M-current drive
				# ===== END EDITABLE SECTION =====
				+ input_current[:, i - 1]                               # (batch_size,) injected current
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep**0.5)
				# (batch_size,) process noise (unchanged from base model)
			) / (tau_V_inv * C)   # (batch_size,) mV — voltage steady state

			# Exponential Euler integration (exact for linear ODE approximation)
			V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)   # (batch_size,) mV
			n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# ===== BEGIN EDITABLE SECTION (only modify within this block) =====
			# M-current gate exponential Euler update with constant time constant param_j
			# Clamp param_j away from zero for numerical safety (avoid division by near-zero)
			tau_p = torch.clamp(param_j, min=1e-3)   # (batch_size,) ms, strictly positive
			p[:, i] = p_ss + (p[:, i - 1] - p_ss) * Exp(-tstep / tau_p)   # (batch_size,)
			# ===== END EDITABLE SECTION =====

		# ── Return voltage trace with optional observation noise (kept as base) ───
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)   # (batch_size, time_steps) mV