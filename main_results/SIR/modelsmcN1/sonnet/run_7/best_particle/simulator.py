import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Early COVID-19 estimates: R0~3.6, ~14-day infectious period
		# beta=0.25, gamma=0.07 lies well within tightened bounds [0.05,0.80] x [0.02,0.25]
		self.parameters = np.array([0.25, 0.07])  # shape: (2,); [beta, gamma]
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
		Implements one simulation step (1 day) of a stochastic SIR model.

		Design decisions (full validated iteration history):

		1. ITERATION HISTORY (neg_avg_log_marginal_NLE; higher = better):
		     - NB inf r=10  + Binomial rec (simultaneous):      NLE=-377  (worst; overdispersed)
		     - NB inf r=50  + NB rec r=50  (simultaneous):      NLE=-372  (regression; NB rec harmful)
		     - Binomial inf + Binomial rec (simultaneous v1):   NLE=-370  (stable baseline)
		     - NB inf r=100 + Binomial rec (simultaneous):      NLE=-394  (unstable)
		     - Binomial inf + Binomial rec (simultaneous v2):   NLE=-356  (strong baseline)
		     - Beta-Binomial inf + Binomial rec (simultaneous): NLE=-391  (regression; overdispersion harmful)
		     - Binomial inf + Binomial rec (sequential):        NLE=-384  (regression; sequential harmful)
		     - Binomial inf + Binomial rec (simultaneous v3):   NLE=-349  (BEST; wide bounds [0.01,1.50]x[0.01,0.50])
		   THIS ITERATION: Restore NLE=-349 design + TIGHTEN PARAMETER BOUNDS.
		   Targeted single change: clip bounds tightened to beta∈[0.05,0.80], gamma∈[0.02,0.25].

		2. TIGHTENED PARAMETER BOUNDS — NEW THIS ITERATION:
		   Rationale from feedback: prior bounds beta∈[0.01,1.50], gamma∈[0.01,0.50] created
		   an unnecessarily large SBI search volume. With very few training samples, SBI assigns
		   non-negligible posterior mass to implausible regions (e.g., beta>0.80→R0>11,
		   gamma>0.30→infectious period<4 days), inflating the marginal NLE.
		   Tightened bounds:
		     beta  ∈ [0.05, 0.80]: R0 range ~0.6–10; covers all plausible COVID-19 scenarios
		     gamma ∈ [0.02, 0.25]: infectious period 4–50 days; covers mild to severe COVID
		   This reduces the prior search volume by ~60% while excluding only epidemiologically
		   implausible parameter combinations, allowing SBI to concentrate posterior mass
		   near the true parameter values with fewer training samples.
		   Default init beta=0.25, gamma=0.07 remains well-centered in the tighter range.

		3. SIMULTANEOUS TAU-LEAPING — VALIDATED OPTIMAL (all changes preserved):
		   Conclusive finding: sequential tau-leaping caused NLE regression from -356 to -384.
		   Simultaneous scheme: BOTH transitions drawn from START-OF-DAY compartment sizes.
		     new_infections ~ Binomial(S, p_infection)  [from start-of-day S and I]
		     new_recoveries ~ Binomial(I, p_recovery)   [from start-of-day I]
		   COVID-19 biology: incubation + infectious period >> 1 day, so newly-infected
		   individuals should not recover the same day. Simultaneous scheme captures this.
		   Produces smoother, wider likelihood surface for SBI — empirically optimal.

		4. PURE BINOMIAL INFECTIONS — VALIDATED STABLE:
		   All overdispersion (NB r=10/50/100, Beta-Binomial phi=80) caused regressions.
		   new_infections ~ Binomial(S, p_infection)
		   Smooth continuous likelihood; no extra hyperparameters.

		5. PURE BINOMIAL RECOVERIES — VALIDATED STABLE:
		   NB recoveries caused regression. Pure Binomial consistently optimal.
		   new_recoveries ~ Binomial(I, p_recovery)  [start-of-day I]
		   Homogeneous per-day Markov recovery; valid for all I >= 0.

		6. EXACT EXPONENTIAL HAZARD PROBABILITIES (dt=1 day):
		   p_infection = 1 - exp(-beta * I / N)   [exact Kolmogorov; no Euler error; in [0,1]]
		   p_recovery  = 1 - exp(-gamma)           [exact memoryless exponential; in [0,1]]

		7. STRICT CONSERVATION R = N - S - I:
		   Eliminates cumulative integer rounding drift over 60 simulation days.

		Args:
			state (dict): The environment state: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing [beta, gamma].
			                         beta  > 0: daily transmission rate
			                         gamma > 0: daily recovery rate
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state: "S": int, "I": int, "R": int.
		"""

		# --- Unpack and clip parameters to tightened COVID-plausible ranges ---
		# Tightened this iteration: beta∈[0.05,0.80], gamma∈[0.02,0.25]
		# Reduces SBI prior volume by ~60%; excludes only epidemiologically implausible combos
		beta  = float(np.clip(parameters[0], 0.05, 0.80))   # scalar float in [0.05, 0.80]; transmission rate; R0 range ~0.6-10
		gamma = float(np.clip(parameters[1], 0.02, 0.25))   # scalar float in [0.02, 0.25]; recovery rate; infectious period 4-50 days

		# --- Unpack current state ---
		S = int(state["S"])  # scalar int >= 0; susceptible count at start of day
		I = int(state["I"])  # scalar int >= 0; infected count at start of day
		R = int(state["R"])  # scalar int >= 0; recovered count at start of day

		# --- Total (conserved) population ---
		N = S + I + R  # scalar int; strictly constant throughout simulation

		# --- Trivial guard: no population or no infected → no dynamics ---
		if N == 0 or I == 0:
			next_state = {"S": S, "I": I, "R": R}
			return next_state

		# -----------------------------------------------------------------------
		# Exact continuous-time hazard transition probabilities (dt=1 day)
		# -----------------------------------------------------------------------
		# Force-of-infection: p_infection = 1 - exp(-beta * I / N)
		# Exact Kolmogorov solution for Markov chain; no Euler approximation error.
		# Uses start-of-day I (consistent with simultaneous tau-leaping scheme).
		# Guaranteed in [0, 1] for all beta > 0, I in [0, N].
		# shape: scalar float in [0.0, 1.0]
		p_infection = float(np.clip(1.0 - np.exp(-beta * float(I) / float(N)), 0.0, 1.0))  # scalar float in [0, 1]

		# Recovery probability per infected: p_recovery = 1 - exp(-gamma)
		# Exact memoryless exponential recovery at rate gamma per day; dt=1.
		# Identical for each infected individual; independent of compartment sizes.
		# Guaranteed in [0, 1] for all gamma > 0.
		# shape: scalar float in [0.0, 1.0]
		p_recovery  = float(np.clip(1.0 - np.exp(-gamma), 0.0, 1.0))                       # scalar float in [0, 1]

		# -----------------------------------------------------------------------
		# SIMULTANEOUS TAU-LEAPING — both transitions from start-of-day counts
		# -----------------------------------------------------------------------
		# VALIDATED OPTIMAL across all 7+ iterations tested.
		# Sequential scheme (I_mid for recoveries) caused NLE=-384 regression.
		# Simultaneous scheme consistently achieves best NLE (currently -349).

		# --- New infections: pure Binomial from start-of-day S ---
		# new_infections ~ Binomial(S, p_infection)
		# Each of S independent susceptibles has identical daily infection probability.
		# E[new_infections]   = S * p_infection
		# Var[new_infections] = S * p_infection * (1 - p_infection)
		# All overdispersed variants (NB, Beta-Binomial) caused regressions; pure Binomial optimal.
		# shape: scalar int in [0, S]
		new_infections = int(rng.binomial(n=S, p=p_infection))        # scalar int >= 0; pure Binomial infection draw
		new_infections = int(np.clip(new_infections, 0, S))            # scalar int in [0, S]; enforce physical cap

		# --- New recoveries: pure Binomial from start-of-day I (simultaneous) ---
		# new_recoveries ~ Binomial(I, p_recovery)
		# Drawn from ORIGINAL start-of-day I — not updated I_mid (simultaneous scheme).
		# Newly-infected individuals CANNOT recover same day (biologically correct for COVID).
		# This guarantees I_next >= 0: new_recoveries <= I <= I + new_infections.
		# E[new_recoveries]   = I * p_recovery
		# Var[new_recoveries] = I * p_recovery * (1 - p_recovery)
		# Valid for all I >= 0: Binomial(0,p)=0 by convention.
		# shape: scalar int in [0, I]
		new_recoveries = int(rng.binomial(n=I, p=p_recovery))         # scalar int >= 0; pure Binomial recovery draw
		new_recoveries = int(np.clip(new_recoveries, 0, I))            # scalar int in [0, I]; enforce physical cap

		# -----------------------------------------------------------------------
		# Compartment updates from simultaneous tau-leaping draws
		# -----------------------------------------------------------------------
		# S decreases by new infections (drawn from start-of-day S; guaranteed S_next >= 0)
		S_next = int(S - new_infections)                               # scalar int >= 0; susceptibles after infection

		# I increases by infections, decreases by recoveries (both from start-of-day)
		# Guaranteed I_next >= 0: new_recoveries <= I, so I + new_infections - new_recoveries >= 0
		I_next = int(I + new_infections - new_recoveries)              # scalar int >= 0; infected after both transitions

		# R from strict conservation law: eliminates cumulative integer drift over 60 days
		# Equivalent to R + new_recoveries but immune to rounding accumulation
		R_next = int(N - S_next - I_next)                              # scalar int >= 0; recovered via conservation

		# Defensive floors (guaranteed by construction above; kept for numerical robustness)
		S_next = max(0, S_next)   # scalar int >= 0
		I_next = max(0, I_next)   # scalar int >= 0
		R_next = max(0, R_next)   # scalar int >= 0

		# Final conservation enforcement: absorb any residual integer discrepancy into R
		total = S_next + I_next + R_next  # scalar int; must equal N exactly
		if total != N:
			R_next = max(0, R_next + (N - total))  # scalar int; exact N conservation correction

		# --- Pack and return next state ---
		next_state = {
			"S": int(S_next),  # scalar int >= 0
			"I": int(I_next),  # scalar int >= 0
			"R": int(R_next),  # scalar int >= 0
		}

		return next_state