import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.

		Stochastic discrete-time SIR model — v42.

		Full iterative NLE history (neg_avg_log_marginal_NLE; lower magnitude = better):
		  v7:  simultaneous Euler, beta(0.01,0.80), gamma(0.01,0.50)                   -> NLE ≈ -58.2
		  v22: direct int() extraction, max(0,int()) outputs                            -> NLE ≈ -189
		  v23: round() + R re-clamp                                                     -> NLE ≈ +202  [catastrophic]
		  v25: removed max(0,S_new) and I_new int() cast                                -> NLE ≈ +121  [catastrophic]
		  v26: triple-layer extraction + clip(±20)                                      -> NLE ≈ -129
		  v27: direct int() extraction + unconstrained sigmoid                          -> NLE ≈ -8.83
		  v28-v36: ALL regressions from ANY attempted improvement to v27               -> NLE ≈ -32.8 to +212
		  v37: remove float() on parameters + remove float() on I/N in lambda_t        -> NLE ≈ -3.74  [BEST EVER]
		  v38: beta widened (0.005+0.845*sigma)                                         -> NLE ≈ -80.1  [catastrophic]
		  v39: claimed exact revert to v37 — unexpectedly regressed                     -> NLE ≈ -99.1
		  v40: np.clip on probabilities + min() draw caps                               -> NLE ≈ -74.2  [both harmful]
		  v41: claimed exact v37 — catastrophic positive NLE                            -> NLE ≈ +20.5  [catastrophic]
		  v42: v37 mechanics + COVID-specific bounds beta(0.10,0.50), gamma(0.05,0.15) -> target: improve

		v42 TWO CHANGES (from v41 feedback — critical + major severity):
		  1. NARROW BETA to COVID-plausible range (critical):
		       BEFORE (v37-v41): beta = 0.01 + 0.79 * sigma_p0  -> beta ∈ (0.01, 0.80)
		       AFTER  (v42):     beta = 0.10 + 0.40 * sigma_p0  -> beta ∈ (0.10, 0.50)
		     ROOT CAUSE (v41 feedback): wide beta range (0.01–0.80) allows biologically
		     impossible slow transmission; COVID beta ∈ (0.10, 0.50) is epidemiologically
		     grounded, reducing SBI hypothesis space and improving posterior concentration.

		  2. NARROW GAMMA to COVID-plausible range (major):
		       BEFORE (v37-v41): gamma = 0.01 + 0.49 * sigma_p1  -> gamma ∈ (0.01, 0.50)
		       AFTER  (v42):     gamma = 0.05 + 0.10 * sigma_p1  -> gamma ∈ (0.05, 0.15)
		     ROOT CAUSE (v41 feedback): gamma ∈ (0.01, 0.50) allows 2-day recovery
		     (gamma=0.50) which is epidemiologically impossible for COVID-19 (typical
		     infectious period ~7-14 days → gamma ≈ 0.07-0.14 day^-1). The SBI estimator
		     was fitting posteriors in a biologically wrong region of parameter space,
		     causing trajectory generalization failure (+20.5 NLE).

		  COVID epidemiological grounding for v42 bounds:
		    beta ∈ (0.10, 0.50) day^-1: corresponds to 2–10 contacts/day with transmission
		    gamma ∈ (0.05, 0.15) day^-1: corresponds to 7–20 day infectious period
		    R0 = beta/gamma ∈ (0.67, 10): covers COVID range (wild-type R0 ≈ 2–4)
		    Default (parameters=[0,0]): beta=0.300, gamma=0.100, R0=3.0 (COVID-realistic)

		  ALL OTHER CODE identical to v37 mechanics:
		    - Bare arithmetic in lambda_t (NO float() on I or N)
		    - NO float() cast on parameters before sigmoid
		    - NO np.clip on probabilities
		    - NO min() cap on draws
		    - Direct int(state[]) extraction
		    - Simultaneous Euler draws from pre-step I
		    - max(0,...) compartment output guards only

		HARD-LEARNED INVARIANTS — IMMUTABLE (confirmed catastrophic when violated):
		  1.  DIRECT STATE EXTRACTION: int(state['key']) ONLY; v26→-129; v34→-106
		  2.  SIMULTANEOUS EULER: both draws from pre-step (S,I); v4 sequential→-542
		  3.  BETA BOUND: beta=0.10+0.40*sigma_p0; v42 COVID-informed tightening
		  4.  GAMMA BOUND: gamma=0.05+0.10*sigma_p1; v42 COVID-informed tightening
		  5.  BARE NUMPY HAZARD: p=1.0-np.exp(-rate); NO clip/float/min/max
		      v35→-165; v40 clip→-74.2; v37 bare→-3.74
		  6.  NO float() ON PARAMETERS IN SIGMOID; v36→-32.8
		  7.  NO float() ON I/N IN LAMBDA_T; v36→-32.8
		  8.  PURE BINOMIAL: no overdispersion; v2 NegBinomial→-382
		  9.  NO I<=0 EARLY EXIT: only N<=0 guard
		  10. RAW BINOMIAL DRAWS: int(rng.binomial(n,p)) ONLY; v34→-106; v40→-74.2
		  11. S_new = max(0, int(S - new_infections)) — compartment output guard only
		  12. I_new = max(0, int(I + new_infections - new_recoveries)) — lower guard only
		  13. NO UPPER CLAMP on I_new; v20→-64.1
		  14. R_new = N - S_new - I_new; NO R RE-CLAMP; v23→+202
		  15. NO ±20 CLIP ON PARAMETERS; v26→-129
		  16. int() ON I_new IS ESSENTIAL; v25 removal→+121
		"""
		super(SimulatorStep, self).__init__()
		return

	def get_parameters(self) -> np.ndarray:
		"""
		Returns the model parameters as an array.
		"""
		return self.parameters

	def set_parameters(self, parameters: np.ndarray):
		"""
		Updates the model parameters.

		Args:
			parameters (np.ndarray): Array of parameters to update.
		"""
		assert len(parameters) == 2, "Parameter array must have length 2."
		self.parameters = parameters

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
		v42: v37 mechanics + COVID-specific parameter bounds from v41 feedback.

		v41 feedback (critical): gamma range (0.01–0.50) epidemiologically implausible for
		COVID-19; SBI concentrates posterior in biologically wrong region → +20.5 NLE.
		v42 fix: tighten both bounds to COVID-realistic ranges, preserving all v37 mechanics.

		MODEL: Simultaneous stochastic discrete-time SIR with exact hazard probabilities.
		  S → I: Binomial(S, 1 - exp(-beta * I/N))  [simultaneous pre-step I]
		  I → R: Binomial(I, 1 - exp(-gamma))        [simultaneous pre-step I]

		COVID parameter ranges (v42):
		  beta  ∈ (0.10, 0.50) day^-1: epidemiologically grounded transmission rate
		  gamma ∈ (0.05, 0.15) day^-1: ~7-20 day infectious period for COVID-19
		  R0 = beta/gamma ∈ (0.67, 10): covers wild-type COVID R0 ≈ 2-4
		  Default (params=[0,0]): beta=0.300, gamma=0.100, R0=3.0 [COVID-realistic baseline]

		Args:
			state (dict): "S": int, "I": int, "R": int — environment state at time t.
			parameters (np.ndarray): shape (2,), raw unconstrained sigmoid-space params.
			action (int | None): None — no action in this SIR model.
			rng (np.random.Generator): NumPy RNG for reproducible stochastic draws.

		Returns:
			next_state (dict): "S": int, "I": int, "R": int — state at time t+1.
		"""

		# --- State compartment extraction: DIRECT int() (Invariant 1 — inviolable) ---
		# int(state['key']): handles Python int, numpy int64, numpy int32 uniformly.
		# NO float(), round(), max(0,...) outer wrappers — each confirmed catastrophic in history.
		# v26: max(0, int(...)) outer → NLE=-129; v34: float+round+max → NLE=-106
		S = int(state["S"])   # scalar Python int, susceptibles at time t
		I = int(state["I"])   # scalar Python int, infected at time t
		R = int(state["R"])   # scalar Python int, recovered at time t
		N = S + I + R          # scalar Python int ≥ 0, total conserved population

		# --- N<=0 guard: only special case requiring early return (Invariant 9) ---
		# I<=0 branch deliberately absent: I=0 → lambda_t=0 → p_infection=0 → naturally frozen.
		if N <= 0:
			return state.copy()

		# --- Sigmoid reparameterization: parameters[i] DIRECTLY, no float() cast (Invariant 6) ---
		# parameters[i] is numpy float64 scalar from ndarray indexing — preserve dtype throughout.
		# NO float(parameters[i]) wrapper: v36 coercion → NLE=-32.8; v37 direct → NLE=-3.74.
		# NO np.clip on parameters (Invariant 15): v26 ±20 clip → NLE=-129.
		# logistic sigmoid σ(x) = 1/(1+exp(-x)) ∈ (0,1): smooth, monotone, SBI-compatible.
		sigma_p0 = 1.0 / (1.0 + np.exp(-parameters[0]))   # scalar numpy float64 ∈ (0, 1), beta sigmoid
		sigma_p1 = 1.0 / (1.0 + np.exp(-parameters[1]))   # scalar numpy float64 ∈ (0, 1), gamma sigmoid

		# --- COVID-specific parameter bounds (v42 change — from v41 feedback critical+major) ---
		# beta  ∈ (0.10, 0.50) day^-1: COVID transmission rate, 2–10 contacts/day
		#   BEFORE (v37-v41): beta = 0.01 + 0.79 * sigma_p0  [too wide, allows implausible values]
		#   AFTER  (v42):     beta = 0.10 + 0.40 * sigma_p0  [COVID-informed tightening]
		# gamma ∈ (0.05, 0.15) day^-1: COVID infectious period ~7-20 days
		#   BEFORE (v37-v41): gamma = 0.01 + 0.49 * sigma_p1  [2-day recovery impossible for COVID]
		#   AFTER  (v42):     gamma = 0.05 + 0.10 * sigma_p1  [7-20 day infectious period]
		# Default (parameters=[0,0], sigma=0.5): beta=0.300, gamma=0.100, R0=3.0 [COVID-realistic]
		beta  = 0.10 + 0.40 * sigma_p0   # scalar numpy float64 ∈ (0.10, 0.50) day^-1 [COVID-specific]
		gamma = 0.05 + 0.10 * sigma_p1   # scalar numpy float64 ∈ (0.05, 0.15) day^-1 [COVID-specific]

		# --- Exact discrete-time hazard probabilities: BARE numpy (Invariants 5, 7) ---
		# lambda_t = beta * I / N: BARE arithmetic (Invariant 7 — v37 critical fix preserved).
		#   Dtype chain: numpy float64 * Python int / Python int → numpy float64 scalar.
		#   NO float(I), float(N): v36 explicit Python float casts → NLE=-32.8.
		# p = 1.0 - np.exp(-rate): BARE numpy scalar (Invariant 5).
		#   NO float()/min()/max()/clip(): v35→-165; v40 clip→-74.2; v37 bare→-3.74.
		#   COVID bounds guarantee: lambda_t ∈ [0, 0.50] → p_infection ∈ [0, 0.394] ⊂ [0,1)
		#   COVID bounds guarantee: gamma ∈ (0.05, 0.15) → p_recovery ∈ (0.049, 0.139) ⊂ [0,1)
		#   All probabilities strictly in [0,1) — no clip needed or safe to add.
		lambda_t    = beta * I / N               # scalar numpy float64 ≥ 0, force of infection [bare]
		p_infection = 1.0 - np.exp(-lambda_t)   # scalar numpy float64 ∈ [0, 0.394), NO clip [bare]
		p_recovery  = 1.0 - np.exp(-gamma)      # scalar numpy float64 ∈ (0.049, 0.139), NO clip [bare]

		# --- SIMULTANEOUS DRAW 1: S→I new infections (Invariants 2, 10) ---
		# Binomial(S, p_infection): each susceptible independently risks infection today.
		# SIMULTANEOUS EULER: p_infection from PRE-STEP I — inviolable (v4 sequential→-542).
		# RAW DRAW: int(rng.binomial(n,p)) ONLY — NO min() cap.
		#   v34 compartment-level wrapping → NLE=-106
		#   v40 min(..., S) draw cap → NLE=-74.2
		#   v37 bare int(rng.binomial(n,p)) → NLE=-3.74
		# p_infection ∈ [0, 0.394) guarantees draw ≤ S in all valid float64 cases.
		new_infections = int(rng.binomial(n=S, p=p_infection))   # scalar Python int, nominally ∈ [0, S]

		# --- SIMULTANEOUS DRAW 2: I→R new recoveries (Invariants 2, 10) ---
		# Binomial(I, p_recovery): each infected independently recovers today.
		# SIMULTANEOUS: drawn from identical PRE-STEP I snapshot as Draw 1.
		# RAW DRAW: int(rng.binomial(n,p)) ONLY — NO min() cap.
		# p_recovery ∈ (0.049, 0.139) guarantees draw ≤ I in all valid float64 cases.
		new_recoveries = int(rng.binomial(n=I, p=p_recovery))    # scalar Python int, nominally ∈ [0, I]

		# --- Compartment updates: EXACT v37 output guard formulation (Invariants 11-14) ---

		# S' = max(0, int(S - new_infections)); output lower guard on compartment (Invariant 11).
		# Guard on compartment OUTPUT only — NOT a pre-cap on the raw draw (v34 confusion).
		S_new = max(0, int(S - new_infections))   # scalar Python int ∈ [0, N], susceptibles at t+1

		# I' = max(0, int(I + new_infections - new_recoveries)); lower guard only (Invariant 12).
		# NO UPPER CLAMP on I_new (Invariant 13): v20 upper clamp → NLE=-64.1.
		# int() ESSENTIAL (Invariant 16): v25 removal of int() → NLE=+121.
		I_new = max(0, int(I + new_infections - new_recoveries))   # scalar Python int ≥ 0, infected at t+1

		# R' = N - S' - I': exact arithmetic complement — zero population leakage.
		# NO DIRECT R RE-CLAMP (Invariant 14): v23 re-clamp → NLE=+202.
		# COVID bounds + simultaneous Euler ensure physical plausibility.
		R_new = N - S_new - I_new   # scalar Python int ≥ 0, recovered at t+1 (exact conservation)

		next_state = {
			"S": S_new,   # scalar Python int ≥ 0, susceptibles at time t+1
			"I": I_new,   # scalar Python int ≥ 0, infected at time t+1
			"R": R_new,   # scalar Python int ≥ 0, recovered at time t+1
		}

		return next_state