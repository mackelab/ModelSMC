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
		Hodgkin-Huxley neuron extended with:
		  1. M-current (I_KM): slow non-inactivating K+ current using X1 slot
		     - Provides spike-frequency adaptation, stabilises tonic firing
		     - Uses voltage-independent tau (more stable during spikes)
		     - gbar_KM = params[:,6], V_half_p = -params[:,8], tau_p = params[:,9]

		  2. Ih current (HCN channel): hyperpolarisation-activated cation current using X2 slot
		     - Opens at hyperpolarised voltages (~-80 mV and below)
		     - Reversal E_h ≈ -30 mV (mixed Na+/K+), depolarising at rest
		     - Corrects resting potential mean, SD, and voltage distribution shape
		     - Fixed kinetics (half-activation -80 mV, tau 200 ms); only gbar_Ih inferred
		     - gbar_Ih = params[:,7], all other Ih parameters are fixed

		Args:
			init_voltage: torch.Tensor: (batch_size,)            initial voltage [mV]
			input_current: torch.Tensor: (batch_size, time_steps) injected current [uA/cm2]
			dt: float                                              time step [ms]
			t: torch.Tensor: (time_steps,)                        time array [ms]
			params: torch.Tensor: (batch_size, 10)                biophysical parameters
			seed: optional random seed

		Returns:
			V: torch.Tensor: (batch_size, time_steps)             voltage traces [mV]
		"""
		device = params.device

		# Set up random generator
		if seed is not None:
			generator = torch.Generator(device=device)
			generator.manual_seed(seed)
		else:
			generator = torch.Generator(device=device)

		batch_size = params.shape[0]   # int
		time_steps = t.shape[0]        # int

		# ── Base HH parameters ────────────────────────────────────────────────────
		gbar_Na   = params[:, 0].float()   # (batch_size,)  mS/cm²
		gbar_K    = params[:, 1].float()   # (batch_size,)  mS/cm²
		g_leak    = params[:, 2].float()   # (batch_size,)  mS/cm²
		E_leak    = -params[:, 3].float()  # (batch_size,)  mV  (negative applied)
		Vt        = -params[:, 4].float()  # (batch_size,)  mV  (negative applied)
		nois_fact = params[:, 5].float()   # (batch_size,)  unitless

		# ── X1 slot: M-current (I_KM, slow non-inactivating K+) ─────────────────
		# Physiological rationale:
		#   M-current activates slowly at subthreshold depolarised voltages (~-35 mV),
		#   providing an outward K+ current that limits repetitive firing rate and
		#   produces spike-frequency adaptation without bursting or quiescence.
		# Parameter mapping:
		#   gbar_KM   = params[:,6] ∈ [1e-4, 10]  mS/cm²
		#   V_half_p  = -params[:,8] ∈ [-150, -1e-4] mV  (typically ~-35 mV)
		#   tau_p     =  params[:,9] ∈ [1e-4, 3000]  ms  (typically 50-500 ms)
		# Note: voltage-independent tau avoids instability during fast action potentials
		gbar_KM  = params[:, 6].float()   # (batch_size,)  mS/cm²
		V_half_p = -params[:, 8].float()  # (batch_size,)  mV
		tau_p    =  params[:, 9].float()  # (batch_size,)  ms — voltage-independent for stability

		# ── X2 slot: Ih current (HCN, hyperpolarisation-activated cation) ────────
		# Physiological rationale:
		#   Ih activates at hyperpolarised voltages (< -60 mV), carries inward
		#   Na+/K+ current with reversal ~-30 mV, depolarising the cell toward
		#   rest. This sets a tonic inward "sag" current that:
		#     - Raises mean resting potential slightly above pure K+ equilibrium
		#     - Increases resting SD (voltage fluctuations due to channel noise)
		#     - Shapes subthreshold distribution (skewness, kurtosis)
		#   It does NOT promote bursting — it acts as a stabilising "pacemaker"
		#   current opposing excessive hyperpolarisation.
		# Parameter mapping:
		#   gbar_Ih = params[:,7] ∈ [1e-4, 120] mS/cm² — only inferred parameter
		#   All kinetic parameters fixed from literature
		gbar_Ih = params[:, 7].float()   # (batch_size,)  mS/cm²

		# Fixed Ih kinetic constants (from Magee 1998, Koch 1999)
		E_h          = -30.0   # mV  — mixed Na+/K+ cation reversal
		V_half_r     = -80.0   # mV  — half-activation (hyperpolarisation-activated)
		k_r          =  7.0    # mV  — activation slope (negative: opens on hyperpol.)
		tau_r_fixed  = 200.0   # ms  — slow, voltage-independent time constant

		tstep = float(dt)

		# ── Fixed biophysical constants ───────────────────────────────────────────
		nois_fact_obs = 0.0
		C    = 1.0    # uF/cm²
		E_Na = 53.0   # mV
		E_K  = -107.0 # mV

		# ── Numerical helpers ─────────────────────────────────────────────────────
		def Exp(z):
			# Numerically safe exponential — prevent overflow at very negative z
			return torch.where(
				z < -5e2,
				torch.exp(torch.full_like(z, -5e2)),
				torch.exp(z)
			)  # same shape as z

		def efun(z):
			# Handles 0/0 limit of z / (exp(z) - 1) near z=0
			return torch.where(
				torch.abs(z) < 1e-4,
				1 - z / 2,
				z / (Exp(z) - 1)
			)  # same shape as z

		# ── Standard HH channel kinetics ─────────────────────────────────────────
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
			return 4.0 / (1 + Exp(-0.2 * v1))

		def alpha_n(x):  # (batch_size,) → (batch_size,)
			v1 = x - Vt - 15.0
			return 0.032 * efun(-0.2 * v1) / 0.2

		def beta_n(x):   # (batch_size,) → (batch_size,)
			v1 = x - Vt - 10.0
			return 0.5 * Exp(-v1 / 40)

		def tau_x(alpha, beta):  # (batch_size,), (batch_size,) → (batch_size,)
			return 1.0 / (alpha + beta)

		def inf_x(alpha, beta):  # (batch_size,), (batch_size,) → (batch_size,)
			return alpha / (alpha + beta)

		# ── M-current (I_KM) gating: Boltzmann steady-state, constant tau ────────
		# p_inf = 1 / (1 + exp(-(V - V_half_p) / 10))
		# Slope k=10 mV is standard for cortical M-current (Wang 1998)
		def inf_p(x):  # (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp(-(x - V_half_p) / 10.0))

		# Voltage-independent tau_p avoids spurious fast tracking during spikes.
		# tau_p is directly the inferred parameter (batch_size,), already positive.

		# ── Ih gating: hyperpolarisation-activated Boltzmann, fixed tau ──────────
		# r_inf = 1 / (1 + exp((V - V_half_r) / k_r))
		# Note positive sign in exponent: more open at V << V_half_r = -80 mV
		def inf_r(x):  # (batch_size,) → (batch_size,)
			return 1.0 / (1.0 + Exp((x - V_half_r) / k_r))

		# tau_r is fixed scalar (200 ms) — no per-batch variation needed

		# ── State variable arrays ─────────────────────────────────────────────────
		V = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		n = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		m = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		h = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps)
		p = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) M-current gate
		r = torch.zeros((batch_size, time_steps), device=device)  # (batch_size, time_steps) Ih gate

		# ── Initial conditions (steady state at initial voltage) ──────────────────
		V_init = init_voltage.to(device)  # (batch_size,)
		V[:, 0] = V_init                                               # (batch_size,)
		n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))            # (batch_size,)
		m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))            # (batch_size,)
		h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))            # (batch_size,)
		p[:, 0] = inf_p(V[:, 0])                                       # (batch_size,)
		r[:, 0] = inf_r(V[:, 0])                                       # (batch_size,)

		# ── Exponential Euler time-integration loop ───────────────────────────────
		for i in range(1, time_steps):
			V_prev = V[:, i - 1]  # (batch_size,)

			# Standard HH gating rates
			a_m, b_m = alpha_m(V_prev), beta_m(V_prev)   # (batch_size,), (batch_size,)
			a_h, b_h = alpha_h(V_prev), beta_h(V_prev)   # (batch_size,), (batch_size,)
			a_n, b_n = alpha_n(V_prev), beta_n(V_prev)   # (batch_size,), (batch_size,)

			# M-current gate steady-state (tau is voltage-independent → use tau_p directly)
			inf_p_val = inf_p(V_prev)    # (batch_size,)

			# Ih gate steady-state (tau is fixed scalar)
			inf_r_val = inf_r(V_prev)    # (batch_size,)

			# Effective conductances at previous time step
			g_Na_eff  = (m[:, i-1] ** 3) * gbar_Na * h[:, i-1]  # (batch_size,)
			g_K_eff   = (n[:, i-1] ** 4) * gbar_K                # (batch_size,)
			g_KM_eff  = gbar_KM * p[:, i-1]                      # (batch_size,)
			g_Ih_eff  = gbar_Ih * r[:, i-1]                      # (batch_size,)

			# Effective inverse membrane time constant (total conductance / C)
			tau_V_inv = (
				g_Na_eff
				+ g_K_eff
				+ g_leak
				+ g_KM_eff   # M-current: outward K+, raises conductance
				+ g_Ih_eff   # Ih: inward mixed cation, raises conductance
			) / C  # (batch_size,)

			# Voltage steady-state numerator (weighted reversal potentials + input)
			V_inf = (
				g_Na_eff * E_Na
				+ g_K_eff  * E_K
				+ g_leak   * E_leak
				+ g_KM_eff * E_K    # M-current drives V toward E_K (hyperpolarising)
				+ g_Ih_eff * E_h    # Ih drives V toward E_h ~ -30 mV (depolarising at rest)
				+ input_current[:, i-1]
				+ nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep ** 0.5)
			) / (tau_V_inv * C)  # (batch_size,)

			# Exponential Euler update for voltage
			V[:, i] = V_inf + (V_prev - V_inf) * Exp(-tstep * tau_V_inv)          # (batch_size,)

			# Standard HH gating variable updates
			n[:, i] = inf_x(a_n, b_n) + (n[:, i-1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))  # (batch_size,)
			m[:, i] = inf_x(a_m, b_m) + (m[:, i-1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))  # (batch_size,)
			h[:, i] = inf_x(a_h, b_h) + (h[:, i-1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))  # (batch_size,)

			# M-current gate update (voltage-independent tau → stable during spikes)
			p[:, i] = inf_p_val + (p[:, i-1] - inf_p_val) * Exp(-tstep / tau_p)  # (batch_size,)

			# Ih gate update (fixed scalar tau)
			r[:, i] = inf_r_val + (r[:, i-1] - inf_r_val) * Exp(
				torch.full_like(V_prev, -tstep / tau_r_fixed)
			)  # (batch_size,)

		# Return voltage traces (+ optional observation noise, currently zero)
		return V + nois_fact_obs * torch.randn(
			batch_size, time_steps, generator=generator, device=device
		)  # (batch_size, time_steps)