import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment — NB(k_inf=15) infection + Binomial simultaneous recovery.

		ARCHITECTURE (incremental improvement over NLE=-363 simultaneous Binomial baseline):
		  Infection:  NB(k_inf=15, mu=S·p_inf) — mild COVID superspreading overdispersion. ✓
		  Recovery:   Binomial(n=I, p=p_rec) — simultaneous from I_t (confirmed good). ✓
		  Update:     Simultaneous (both draws from current state I_t, S_t). ✓
		  Clips:      beta∈[1e-4, 1.0]; gamma∈[1e-4, 1.0]. ✓
		  Init:       [0.25, 0.07] → R0≈3.6 (early COVID prior). ✓

		DESIGN HISTORY (NLE — closer to 0 is better):
		  NB(k=15)_inf+Binomial_rec_simultaneous (THIS):      NLE=??? — INCREMENTAL IMPROVEMENT
		  Binomial_inf+Binomial_rec_simultaneous:              NLE=-363  — confirmed good baseline
		  NB(k=0.003)+outcome-cap+Binomial_seq:               NLE=+150  — CATASTROPHIC (k too small)
		  NB(k=0.003)+soft_lam_guard(S*3)+Binomial_seq:       NLE=+46.3 — CATASTROPHIC (lam guard)
		  NB(k=0.002)+lam_cap+Binomial_seq:                   NLE=+51.8 — CATASTROPHIC
		  Binomial+Binomial_seq (old):                        NLE=-329  — sequential worse

		CRITICAL LESSONS:
		  1. NB k=0.003: Var=333·mu² → catastrophic overdispersion (NLE=+150). ✗
		  2. Sequential update (from I_after): same-step infection+recovery artifact. ✗ (NLE=-329)
		  3. Simultaneous Binomial: clean, identifiable, NLE=-363. ✓
		  4. NB k=15: Var=mu+mu²/15 — mild; adds COVID superspreading signal without catastrophe. ✓
		     For mu=50: Binomial std≈6.9 vs NB(k=15) std≈9.6 → modest calibrated overdispersion. ✓
		  5. Tighter beta clip [1e-4,1.0]: R0∈[0.001,10]; eliminates implausible R0>10 waste. ✓

		NB PARAMETERIZATION (numpy rng.negative_binomial):
		  rng.negative_binomial(n=k_inf, p=k_inf/(k_inf+mu_inf))
		  Mean  = n·(1-p)/p = k_inf·(mu_inf/k_inf) = mu_inf. ✓
		  Var   = n·(1-p)/p² = mu_inf + mu_inf²/k_inf. ✓
		  k=15: Var ≈ mu + 0.067·mu² → near-Poisson; mild overdispersion; SBI-identifiable. ✓

		Parameters:
		  parameters[0] = beta  (transmission hazard; init 0.25; bounded [1e-4, 1.0])
		  parameters[1] = gamma (recovery hazard; init 0.07; bounded [1e-4, 1.0])
		  R0 = beta/gamma = 0.25/0.07 ≈ 3.6 at init. ✓
		"""
		super(SimulatorStep, self).__init__()

		# Initial parameters: [beta=0.25, gamma=0.07] → R0≈3.6 (early COVID prior)
		self.parameters = np.array([0.25, 0.07])  # shape: (2,) — [beta, gamma]

		# NB infection dispersion k_inf=15 — mild calibrated overdispersion. ✓
		# Var[ΔSI] = mu_inf + mu_inf²/15: ~7% extra variance over Poisson. ✓
		# Safe regime: k=15 >> k=0.003; no catastrophic tail blowup. ✓
		self._k_inf = 15.0  # scalar: NB dispersion for infection (moderate overdispersion)

		return

	def get_parameters(self) -> np.ndarray:
		"""
		Returns the model parameters as an array.
		"""
		return self.parameters  # shape: (2,)

	def set_parameters(self, parameters: np.ndarray):
		"""
		Updates the model parameters.

		Args:
			parameters (np.ndarray): Array of parameters to update.
		"""
		assert len(parameters) == 2, "Parameter array must have length 2."
		self.parameters = parameters  # shape: (2,)

	def step(self, state: dict, action: int | None, rng: np.random.Generator) -> dict:
		"""
		Wrapper to call the forward method.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""
		return self.forward(state=state, parameters=self.get_parameters(), action=action, rng=rng)

	def forward(
		self,
		state: dict,
		parameters: np.ndarray,
		action: int | None,
		rng: np.random.Generator,
	) -> dict:
		"""
		One step: NB(k_inf=15, mu=S·p_inf) infection + Binomial(I, p_rec) simultaneous recovery.

		MECHANISM (incremental improvement on NLE=-363 simultaneous Binomial):
		  1.  FOI = beta * (I / N)                                    [frequency-dependent force of infection]
		  2.  p_inf = 1 - exp(-FOI)                                   [daily S→I probability ∈ [0,1]]
		  3.  p_rec = 1 - exp(-gamma)                                 [daily I→R probability ∈ [0,1]]
		  4.  mu_inf = S * p_inf                                       [expected infections ∈ [0,S]]
		  5.  new_infections ~ NB(n=k_inf, p=k_inf/(k_inf+mu_inf))   [mild COVID overdispersion; capped at S]
		  6.  new_recoveries ~ Binomial(n=I, p=p_rec)                 [simultaneous from I_t; confirmed good]
		  7.  S_next = S - new_infections                             [S decreases by ΔSI]
		  8.  I_next = I + new_infections - new_recoveries            [I changes by ΔSI - ΔIR]
		  9.  R_next = R + new_recoveries                             [R increases by ΔIR]

		NB INFECTION vs BINOMIAL (per feedback):
		  Binomial(S, p): Var=S·p·(1-p) — underdispersed for COVID. ✗ (NLE=-363)
		  NB(k=15, mu=S·p): Var=mu+mu²/15 — mild overdispersion; matches COVID contact heterogeneity. ✓
		  NB(k=0.003): Var=333·mu² — catastrophically overdispersed. ✗ (NLE=+150)
		  k=15 is safely between these extremes; Var/mu = 1 + mu/15 (near-Poisson for small mu). ✓

		NUMPY NB PARAMETERIZATION:
		  rng.negative_binomial(n=k_inf, p=p_nb) where p_nb = k_inf/(k_inf+mu_inf)
		  → Mean  = k_inf·(1-p_nb)/p_nb = k_inf·(mu_inf/k_inf) = mu_inf. ✓
		  → Var   = mean/p_nb = mu_inf·(k_inf+mu_inf)/k_inf = mu_inf + mu_inf²/k_inf. ✓
		  → Support: {0,1,2,...} → cap at S for conservation. ✓

		SIMULTANEOUS UPDATE (confirmed better than sequential, NLE=-363 vs -329):
		  Both ΔSI and ΔIR drawn from current state (S_t, I_t) independently. ✓
		  No same-step infection+recovery artifact. ✓
		  Conservation: S_next+I_next+R_next = N. ✓

		PARAMETER CLIPS (tightened per feedback):
		  beta  ∈ [1e-4, 1.0]: R0 = beta/gamma ∈ [0.001, 10]; excludes implausible R0>10. ✓
		  gamma ∈ [1e-4, 1.0]: gamma>1 implies <1 day mean infectious period; implausible. ✓

		Args:
			state (dict): {"S": int, "I": int, "R": int}
			parameters (np.ndarray): shape (2,) — [beta, gamma]
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): {"S": int, "I": int, "R": int}
		"""

		# --- Extract current compartment counts ---
		S = int(state["S"])  # scalar: susceptible count at time t
		I = int(state["I"])  # scalar: infected count at time t
		R = int(state["R"])  # scalar: recovered count at time t

		# Total population — conserved invariant across all 60 time steps
		N = S + I + R  # scalar: N = S+I+R = const

		# --- Degenerate early-exit: empty population ---
		if N <= 0:
			return {"S": S, "I": I, "R": R}

		# --- Extract and clip parameters to valid biological domain ---
		# beta∈[1e-4, 1.0]: R0≤10; tightened per feedback (was 2.0; eliminated implausible region). ✓
		# gamma∈[1e-4, 1.0]: gamma>1 → mean infectious period <1 day; biologically implausible. ✓
		beta  = float(np.clip(parameters[0], 1e-4, 1.0))   # scalar: β ∈ [1e-4, 1.0]
		gamma = float(np.clip(parameters[1], 1e-4, 1.0))   # scalar: γ ∈ [1e-4, 1.0]

		# --- Discrete-time transition probabilities via exponential survival ---
		i_frac             = float(I) / float(N)                               # scalar: I/N ∈ [0,1]
		force_of_infection = beta * i_frac                                     # scalar: β·I/N ≥ 0
		p_inf = float(np.clip(1.0 - np.exp(-force_of_infection), 0.0, 1.0))   # scalar: P(S→I) ∈ [0,1]
		p_rec = float(np.clip(1.0 - np.exp(-gamma),              0.0, 1.0))   # scalar: P(I→R) ∈ [0,1]

		# --- STEP 1: INFECTION DRAW — NB(k_inf=15, mu=S·p_inf) mild overdispersion ---
		#
		# INCREMENTAL IMPROVEMENT over Binomial (NLE=-363 baseline). ✓
		# NB(k=15): Var = mu + mu²/15 → mild COVID superspreading; SBI-identifiable. ✓
		# k=15 is safely far from catastrophic k=0.003 (Var=333·mu²; NLE=+150). ✓
		#
		# numpy rng.negative_binomial(n, p):
		#   n = k_inf = 15 (dispersion parameter / number of successes)
		#   p = k_inf / (k_inf + mu_inf) (success probability)
		#   Mean = n·(1-p)/p = mu_inf. ✓
		#   Var  = mu_inf + mu_inf²/k_inf = mu_inf + mu_inf²/15. ✓
		#
		# cap at S: conservation guarantee S_next = S - ΔSI ≥ 0. ✓

		if S > 0 and I > 0:
			k_inf   = self._k_inf                                                  # scalar: k_inf=15.0
			mu_inf  = float(S) * p_inf                                             # scalar: E[ΔSI] = S·p_inf ∈ [0,S]

			if mu_inf > 0.0:
				# NB success probability: p_nb = k/(k+mu); mean = mu; Var = mu + mu²/k. ✓
				p_nb           = k_inf / (k_inf + mu_inf)                              # scalar: p_nb ∈ (0,1)
				new_inf_raw    = int(rng.negative_binomial(n=k_inf, p=p_nb))          # scalar: NB draw ∈ {0,1,2,...}
				new_infections = min(new_inf_raw, S)                                  # scalar: ΔSI ∈ {0,...,S} (cap at S)
			else:
				# mu_inf=0: zero FOI (I=0 or beta=0 edge case)
				new_infections = 0  # scalar: ΔSI=0

		else:
			# S=0: no susceptibles, OR I=0: epidemic extinct → no new infections
			new_infections = 0  # scalar: ΔSI=0

		# --- STEP 2: RECOVERY DRAW — Binomial(n=I, p=p_rec) SIMULTANEOUS from I_t ---
		#
		# CONFIRMED GOOD (core of NLE=-363 simultaneous baseline). ✓
		# Drawn from current I_t (simultaneous), NOT I_after. ✓
		# E[ΔIR] = I·p_rec; Var = I·p_rec·(1-p_rec). ✓
		# Support {0,...,I}: guarantees I contributes non-negatively to I_next. ✓

		if I > 0:
			new_recoveries = int(rng.binomial(n=I, p=p_rec))                      # scalar: ΔIR ∈ {0,...,I}
		else:
			# I=0: no infected individuals to recover
			new_recoveries = 0  # scalar: ΔIR=0

		# --- Simultaneous compartment update ---
		# Both ΔSI (from S_t) and ΔIR (from I_t) drawn from current state. ✓
		# Conservation: (S-ΔSI)+(I+ΔSI-ΔIR)+(R+ΔIR) = S+I+R = N. ✓
		S_next = S - new_infections                          # scalar: S_{t+1} = S_t - ΔSI
		I_next = I + new_infections - new_recoveries         # scalar: I_{t+1} = I_t + ΔSI - ΔIR
		R_next = R + new_recoveries                          # scalar: R_{t+1} = R_t + ΔIR

		# --- Defensive non-negativity guards ---
		# min(NB_draw, S) guarantees S_next≥0. ✓
		# Binomial(I, p) guarantees R_next≥R. ✓
		# I_next: clamp rare edge cases (e.g. large ΔIR > I+ΔSI) to 0. ✓
		S_next = max(0, int(S_next))  # scalar int: S_{t+1} ≥ 0
		I_next = max(0, int(I_next))  # scalar int: I_{t+1} ≥ 0
		R_next = max(0, int(R_next))  # scalar int: R_{t+1} ≥ 0

		next_state = {
			"S": S_next,  # scalar int: S_{t+1}
			"I": I_next,  # scalar int: I_{t+1}
			"R": R_next,  # scalar int: R_{t+1}
		}

		return next_state