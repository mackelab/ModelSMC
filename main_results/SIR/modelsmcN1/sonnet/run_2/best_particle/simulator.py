import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# ITER 37: TWO TARGETED CHANGES per Iter36 feedback (issues 1 & 2)
		#
		# ITER 36 POST-MORTEM (catastrophic — parameter range widening):
		#   Config: single step; init=[0.0,0.0]; β=0.10+0.50σ ∈(0.10,0.60); γ=0.04+0.25σ ∈(0.04,0.29)
		#   Result: NLE=-415 — degraded from Iter34 (-228); γ widening made SBI WORSE
		#   Iter 36 feedback main diagnosis:
		#     "Parameter priors far too wide relative to what SBI can constrain from few observations"
		#     "Iter16/18 (NLE=-52.5/-45.3) used narrower, better-centered ranges matching COVID-19"
		#     "γ ceiling at 0.29 (1/γ≈3.4 days) unrealistically fast for COVID-19"
		#     "Wide prior forces SBI to explore implausible parameter space"
		#     "Simulator structure (single step, simultaneous Binomial, CTMC) correct and proven"
		#   Iter 36 feedback diagnosis (issue 1, CRITICAL):
		#     "Narrow γ: gamma = 0.05 + 0.10*sig_gamma → γ∈(0.05, 0.15) [1/day]"
		#     "Centers 1/γ at 10 days (init sig=0.5: γ=0.10) — COVID-19 consensus infectious period"
		#     "Narrower prior dramatically improves SBI posterior estimation from few samples"
		#     "init raw_gamma=0.0 → γ=0.10, 1/γ=10 days, R0=0.35/0.10=3.5"
		#   Iter 36 feedback diagnosis (issue 2, MAJOR):
		#     "Narrow β: beta = 0.15 + 0.35*sig_beta → β∈(0.15, 0.50) [1/day]"
		#     "Removes sub-epidemic tail (β<0.15) and caps at realistic β=0.50"
		#     "Combined with γ∈(0.05,0.15): R0 range (1.0,10.0), mean R0≈3.25 at init"
		#     "init raw_beta=0.0 → β=0.15+0.35×0.5=0.325, R0≈3.25 (COVID-19 Delta/Omicron)"
		#
		# ITER 37 CHANGE 1: NARROW γ RANGE (issue 1, CRITICAL)
		#   OLD: γ = 0.04 + 0.25·σ(raw_gamma) ∈ (0.04, 0.29) — too wide; NLE=-415
		#   NEW: γ = 0.05 + 0.10·σ(raw_gamma) ∈ (0.05, 0.15) — COVID-realistic; 1/γ∈(6.7,20)days
		#   Motivation:
		#     COVID-19 mean infectious period: 10-14 days → γ=0.07-0.10 [1/day]
		#     OLD range had γ=0.29 (3.4-day period) — epidemiologically unrealistic for COVID-19
		#     NEW range centers on γ=0.10 at init (σ=0.5) → 1/γ=10 days exactly
		#     Narrow prior: SBI concentrates posterior on COVID-plausible region → better density est.
		#     Infectious period range: 1/0.15≈6.7 days to 1/0.05=20 days — realistic COVID-19 window
		#
		# ITER 37 CHANGE 2: NARROW β RANGE (issue 2, MAJOR)
		#   OLD: β = 0.10 + 0.50·σ(raw_beta) ∈ (0.10, 0.60) — too wide; includes sub-epidemic values
		#   NEW: β = 0.15 + 0.35·σ(raw_beta) ∈ (0.15, 0.50) — removes sub-epidemic tail
		#   Motivation:
		#     OLD β=0.10 at γ=0.15: R0=0.67 (sub-epidemic, unrealistic for COVID outbreak data)
		#     OLD β=0.60 at γ=0.05: R0=12.0 (unrealistically high for COVID)
		#     NEW: β∈(0.15,0.50) with γ∈(0.05,0.15) → R0∈(1.0,10.0)
		#     init: β=0.325, γ=0.10 → R0=3.25 (COVID-19 Delta/Omicron consensus)
		#     Narrower β: SBI no longer wastes posterior mass on non-epidemic scenarios
		#
		# PERMANENTLY PROVEN INVARIANTS (empirically established; NEVER CHANGE):
		#   Single dt=1 day step: sub-steps CATASTROPHICALLY falsified (Iter35: -452)
		#   Simultaneous Binomial noise: both from ORIGINAL time-t (S_t, I_t) — MANDATORY
		#   CTMC p=1-exp(-hazard): exact formula; Poisson→-208, Deterministic→+434 (FALSIFIED)
		#   N conservation via integer arithmetic: S+I+R=N at all times
		#   ±500 clip on sigmoid input: IEEE 754 overflow protection
		#   Absorbing state guard: if I==0 or N==0 return unchanged state
		#
		# FULL PERFORMANCE HISTORY (all 37 iterations tracked):
		#   Sequential Binomial:         Iters 1-2  → NLE=-444    (FALSIFIED: sequential flux inflation)
		#   NegBin r=10:                 Iter 4     → NLE=-271    (FALSIFIED)
		#   Simultaneous Binomial:       Iter 16    → NLE=-52.5   (BEST EVER RECORDED)
		#   Simultaneous Binomial:       Iter 18    → NLE=-45.3   (2ND BEST EVER)
		#   Poisson(n·p):                Iter 32    → NLE=-208    (FALSIFIED: excess variance)
		#   Deterministic round(n·p):    Iter 33    → NLE=+434    (ABSOLUTE WORST — SBI incompatible)
		#   Sim.Bin. β=0.05+0.65σ:       Iter 34    → NLE=-228    (boundary compression)
		#   Sim.Bin. + 4 sub-steps:      Iter 35    → NLE=-452    (CATASTROPHIC — sub-step BANNED)
		#   Sim.Bin. γ widened:          Iter 36    → NLE=-415    (prior too wide)
		#   Sim.Bin. β/γ narrowed:       Iter 37    → TBD
		#
		# INIT VERIFICATION AT [0.0, 0.0] (ITER37 NEW):
		#   raw_beta=0.0:   σ(0.0) = 0.5000 → β = 0.15 + 0.35×0.5000 = 0.325 [1/day]  ← NEW
		#   raw_gamma=0.0:  σ(0.0) = 0.5000 → γ = 0.05 + 0.10×0.5000 = 0.100 [1/day]  ← NEW
		#   R0 = β/γ = 0.325/0.100 = 3.25 [dimensionless] — COVID-19 Delta/Omicron consensus
		#   Mean infectious period: 1/γ = 10.0 days — COVID-19 epidemiological consensus
		self.parameters = np.array([0.0, 0.0])  # shape: (2,) — ITER37: β≈0.325, γ≈0.100, R0≈3.25
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
		Implements one simulation step.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing model parameters [raw_beta, raw_gamma].
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# -----------------------------------------------------------------------
		# PARAMETER REPARAMETERISATION
		# ITER 37 CHANGE 1 (critical): γ = 0.05+0.10σ ∈(0.05,0.15) — NARROWED from (0.04,0.29)
		# ITER 37 CHANGE 2 (major):    β = 0.15+0.35σ ∈(0.15,0.50) — NARROWED from (0.10,0.60)
		# -----------------------------------------------------------------------
		# β parameterization (NARROWED per Iter36 feedback issue 2, major):
		#   OLD: β = 0.10 + 0.50·σ(raw_beta) ∈ (0.10, 0.60) — wide; includes sub-epidemic values
		#   NEW: β = 0.15 + 0.35·σ(raw_beta) ∈ (0.15, 0.50) — COVID-realistic; removes β<0.15
		#   At init raw_beta=0.0: β = 0.15 + 0.35×0.5 = 0.325 [1/day]
		#   Range rationale: β∈(0.15,0.50) → all values yield epidemic growth for γ∈(0.05,0.15)
		#
		# γ parameterization (NARROWED per Iter36 feedback issue 1, critical):
		#   OLD: γ = 0.04 + 0.25·σ(raw_gamma) ∈ (0.04, 0.29) — too wide; unrealistic fast recovery
		#   NEW: γ = 0.05 + 0.10·σ(raw_gamma) ∈ (0.05, 0.15) — COVID-realistic 1/γ∈(6.7,20) days
		#   At init raw_gamma=0.0: γ = 0.05 + 0.10×0.5 = 0.100 [1/day]; 1/γ=10 days
		#   Range rationale: COVID-19 mean infectious period 10-14 days → γ=0.07-0.10 [1/day]
		#     1/0.15≈6.7 days (fast end); 1/0.05=20.0 days (slow end); both COVID-plausible
		raw_beta  = float(parameters[0])  # scalar: unconstrained SBI parameter for beta
		raw_gamma = float(parameters[1])  # scalar: unconstrained SBI parameter for gamma

		# Numerically stable sigmoid; ±500 clip prevents IEEE 754 overflow at extremes
		sig_beta  = float(1.0 / (1.0 + np.exp(-float(np.clip(raw_beta,  -500.0, 500.0)))))  # scalar: σ(raw_beta)  ∈ (0,1)
		sig_gamma = float(1.0 / (1.0 + np.exp(-float(np.clip(raw_gamma, -500.0, 500.0)))))  # scalar: σ(raw_gamma) ∈ (0,1)

		# ITER 37 CHANGE 2 (major): narrowed β range — removes sub-epidemic tail
		beta  = float(0.15 + 0.35 * sig_beta)   # scalar: transmission rate β ∈ (0.15, 0.50) [1/day]
		# ITER 37 CHANGE 1 (critical): narrowed γ range — COVID-realistic infectious period
		gamma = float(0.05 + 0.10 * sig_gamma)  # scalar: recovery rate     γ ∈ (0.05, 0.15) [1/day]

		# -----------------------------------------------------------------------
		# EXTRACT CURRENT COMPARTMENT COUNTS AT TIME t
		# -----------------------------------------------------------------------
		S = int(state['S'])  # scalar: susceptible count at time t
		I = int(state['I'])  # scalar: infected count at time t
		R = int(state['R'])  # scalar: recovered count at time t

		# Total population — strictly conserved invariant throughout 60-day trajectory
		N = S + I + R  # scalar: total population N = S+I+R (constant)

		# Absorbing state: no dynamics if population is empty or epidemic is extinct
		if N == 0 or I == 0:
			return {'S': S, 'I': I, 'R': R}

		# -----------------------------------------------------------------------
		# CTMC TRANSITION PROBABILITIES (exact formula, dt=1 day — FROZEN)
		# -----------------------------------------------------------------------
		# p = 1-exp(-hazard): exact CTMC result for constant-rate Markov process
		# Guaranteed ∈ [0,1) for any finite positive hazard — analytically cannot exceed 1
		# Single dt=1 day step: sub-steps CATASTROPHICALLY falsified (Iter35: NLE=-452)
		#
		# S → I: frequency-dependent mass-action force of infection λ = β·I/N
		#   At init (β=0.325, I/N=0.10): hazard=0.0325, p_inf≈0.032
		#   At peak (β=0.325, I/N=0.30): hazard=0.0975, p_inf≈0.093
		hazard_inf  = beta * float(I) / float(N)        # scalar: force of infection λ=β·I/N [1/day]
		p_infection = float(1.0 - np.exp(-hazard_inf))  # scalar: CTMC S→I daily transition prob ∈ [0,1)

		# I → R: constant per-capita recovery hazard = gamma [1/day]
		#   At init (γ=0.100): p_rec=1-exp(-0.100)≈0.095; mean infectious period≈10.0 days
		#   At floor (γ≈0.050): p_rec≈0.049; mean period≈20.0 days (slower COVID recovery)
		#   At ceil  (γ≈0.150): p_rec≈0.139; mean period≈6.7 days (faster COVID recovery)
		p_recovery  = float(1.0 - np.exp(-gamma))       # scalar: CTMC I→R daily transition prob ∈ [0,1)

		# -----------------------------------------------------------------------
		# SIMULTANEOUS BINOMIAL TRANSITIONS (proven best SBI noise model — FROZEN)
		# -----------------------------------------------------------------------
		# SIMULTANEOUS: both Binomial draws from ORIGINAL time-t state (S_t, I_t)
		# KEY INVARIANT: new_recoveries drawn from I_t NOT (I_t + new_infections)
		# Violation caused NLE=-444 in sequential Iters 1-2 (permanently FALSIFIED)
		#
		# Binomial: exact CTMC SIR transition distribution (sum of i.i.d. Bernoulli trials)
		#   new_infections ~ Bin(S_t, p_infection): each of S_t susceptibles independently infected
		#   new_recoveries ~ Bin(I_t, p_recovery):  each of I_t infected independently recovers
		# Stochastic noise MANDATORY for SBI density estimation:
		#   Deterministic → NLE=+434 (absolute worst; SBI NDE assigns near-zero likelihood)
		#   Binomial variance: n·p·(1-p) < Poisson variance n·p (confirmed Iter32 vs Iter16)
		#
		# N CONSERVATION PROOF (exact integer arithmetic):
		#   S_{t+1} + I_{t+1} + R_{t+1}
		#   = (S - new_inf) + (I + new_inf - new_rec) + (R + new_rec)
		#   = S + I + R = N  ✓

		# STEP 1: New infections from ORIGINAL susceptible pool S_t (simultaneous CTMC)
		new_infections = int(rng.binomial(int(S), float(p_infection)))  # scalar: Bin(S_t, p_infection) ∈ [0,S]
		new_infections = int(np.clip(new_infections, 0, S))             # scalar: defensive clamp; S_next ≥ 0

		# STEP 2: New recoveries from ORIGINAL infected pool I_t (simultaneous — NOT from I_t+new_inf)
		# CRITICAL INVARIANT: I_t (original), not post-infection count — simultaneous CTMC mandatory
		new_recoveries = int(rng.binomial(int(I), float(p_recovery)))   # scalar: Bin(I_t, p_recovery) ∈ [0,I]
		new_recoveries = int(np.clip(new_recoveries, 0, I))             # scalar: defensive clamp; I_next ≥ 0

		# -----------------------------------------------------------------------
		# SIMULTANEOUS STATE UPDATE (N conservation guaranteed by construction)
		# -----------------------------------------------------------------------
		S_next = S - new_infections                          # scalar: susceptibles at t+1 ∈ [0, S]
		I_next = I + new_infections - new_recoveries         # scalar: infecteds at t+1 (simultaneous)
		R_next = R + new_recoveries                          # scalar: recovered at t+1 ∈ [R, R+I]

		# Safety floors: analytically unreachable given integer clamps above; purely defensive
		S_next = max(0, S_next)  # scalar: non-negative susceptibles
		I_next = max(0, I_next)  # scalar: non-negative infecteds
		R_next = max(0, R_next)  # scalar: non-negative recovered

		# -----------------------------------------------------------------------
		# RETURN NEXT STATE
		# -----------------------------------------------------------------------
		next_state = {
			'S': int(S_next),  # scalar int: susceptible count at t+1
			'I': int(I_next),  # scalar int: infected count at t+1
			'R': int(R_next),  # scalar int: recovered count at t+1
		}

		return next_state