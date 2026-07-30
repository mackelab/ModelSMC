import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# NOTE: self.parameters is required by the skeleton's get_parameters() and
		# set_parameters() methods — cannot be renamed despite shadowing nn.Module.parameters().
		#
		# Default initialisation at raw=[0,0]:
		#   beta  = softplus(0) * 0.40 = log(2) * 0.40 ≈ 0.277 /day
		#   gamma = softplus(0) * 0.15 = log(2) * 0.15 ≈ 0.104 /day
		#   R0 = 0.277 / 0.104 ≈ 2.66 — consistent with early COVID-19 (R0 ∈ 2–3)
		#   Infectious period ≈ 1/0.104 ≈ 9.6 days — consistent with COVID-19 (5–14 days)
		self.parameters = np.array([0.0, 0.0])  # shape (2,)
		return

	def get_parameters(self) -> np.ndarray:
		"""
		Returns the model parameters as an array.
		"""
		return self.parameters  # shape (2,)

	def set_parameters(self, parameters: np.ndarray):
		"""
		Updates the model parameters.

		Args:
			parameters (np.ndarray): Array of parameters to update.
		"""
		assert len(parameters) == 2, "Parameter array must have length 2."
		self.parameters = parameters  # shape (2,)

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
			parameters (np.ndarray): Array of size (2,) containing model parameters.
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# Extract current compartment sizes — scalars
		S = int(state["S"])  # ()
		I = int(state["I"])  # ()
		R = int(state["R"])  # ()

		# Total conserved population — scalar
		N = S + I + R  # ()

		# Guard: no dynamics if population empty or no infected individuals
		if N == 0 or I == 0:
			return {"S": S, "I": I, "R": R}

		# --- PURE SCALED-SOFTPLUS PARAMETER TRANSFORM ---
		#
		# Design rationale — key change from prior iterations:
		#
		# PRIOR (softplus + hard offset):
		#   beta = 0.05 + softplus(raw - 1.5)
		#   Asymmetric: hard floor at 0.05 prevents SBI from exploring very low beta.
		#   Shift -1.5 compresses support below default → poor prior coverage for low-R0 scenarios.
		#   May force SBI to saturate at the floor if true beta is very small.
		#
		# CURRENT (pure scaled softplus):
		#   beta  = softplus(raw_beta)  * 0.40
		#   gamma = softplus(raw_gamma) * 0.15
		#   No hard lower bound — softplus → 0 as raw → -∞, giving SBI full control.
		#   Scale factors chosen so default (raw=0) yields R0≈2.66 (early COVID-19 literature).
		#   Gradient = scale * sigmoid(raw) ∈ (0, scale) — non-vanishing everywhere.
		#   Symmetric soft constraint: SBI can explore very low or very high rates freely.
		#
		# softplus(x) = log(1 + exp(x)) — strictly positive, smooth, monotone increasing.
		# Numerically stable two-branch implementation avoids exp overflow for large x:
		#   x > 20:  softplus(x) ≈ x               (linear regime, saves exp computation)
		#   x < -20: softplus(x) ≈ exp(x) ≈ 0      (exponential regime, no underflow)
		#   else:    softplus(x) = log1p(exp(x))    (standard formula, accurate for |x| ≤ 20)
		raw_beta  = float(parameters[0])  # ()  unbounded real, SBI-optimized
		raw_gamma = float(parameters[1])  # ()  unbounded real, SBI-optimized

		# Numerically stable softplus for beta
		if raw_beta >= 20.0:
			sp_beta = raw_beta                               # ()  linear regime: softplus(x) ≈ x
		elif raw_beta <= -20.0:
			sp_beta = float(np.exp(raw_beta))               # ()  exp regime: softplus(x) ≈ exp(x)
		else:
			sp_beta = float(np.log1p(np.exp(raw_beta)))     # ()  general: log(1 + exp(x))

		# beta = softplus(raw_beta) * 0.40
		# At raw=0: beta = log(2) * 0.40 ≈ 0.277 /day (transmission rate)
		# Range: (0, ∞) with practical support ≈ (0.001, 1.5) for SBI's search range
		# Gradient: d(beta)/d(raw) = sigmoid(raw) * 0.40 ∈ (0, 0.40) — non-vanishing
		beta = sp_beta * 0.40  # ()  transmission rate (1/day), ∈ (0, ∞)

		# Numerically stable softplus for gamma
		if raw_gamma >= 20.0:
			sp_gamma = raw_gamma                             # ()  linear regime
		elif raw_gamma <= -20.0:
			sp_gamma = float(np.exp(raw_gamma))             # ()  exp regime
		else:
			sp_gamma = float(np.log1p(np.exp(raw_gamma)))   # ()  general

		# gamma = softplus(raw_gamma) * 0.15
		# At raw=0: gamma = log(2) * 0.15 ≈ 0.104 /day (recovery rate)
		# Infectious period = 1/gamma ≈ 9.6 days — COVID-19: 5–14 days ✓
		# Range: (0, ∞) with practical support ≈ (0.001, 0.6)
		# Gradient: d(gamma)/d(raw) = sigmoid(raw) * 0.15 ∈ (0, 0.15) — non-vanishing
		gamma = sp_gamma * 0.15  # ()  recovery rate (1/day), ∈ (0, ∞)

		# --- 4-SUB-STEP TAU-LEAPING WITH SIMULTANEOUS BINOMIAL TRANSITIONS ---
		#
		# Iteration history — variance vs NLE score:
		#
		# n_sub=1  (dt=1.00): NLE ≈ -548  → single daily draw: too much relative variance at small I.
		#   At I=5, gamma=0.10: std(new_rec)/mean ≈ 140% → frequent extinction, bimodal p(x|theta).
		#   NLE cannot learn an identifiable likelihood surface from multimodal trajectories.
		#
		# n_sub=4  (dt=0.25): NLE ≈ -529  → BEST KNOWN. Reduces rel variance by 2×.
		#   At I=5, gamma=0.10: std(new_rec)/mean ≈ 70% → occasional extinction but unimodal core.
		#   p(x|theta) concentrated enough for NLE to learn useful likelihood surface.
		#
		# n_sub=16 (dt=0.0625): NLE ≈ -551 → REGRESSION. Over-suppresses variance.
		#   Trajectories become near-deterministic — p(x|theta) too narrow vs real data noise.
		#   NLE penalized for mismatch between simulator variance and observed data variance.
		#
		# CONCLUSION: n_sub=4 is the empirically optimal balance between:
		#   (a) Sufficient stochasticity: finite-variance p(x|theta) that NLE can learn from
		#   (b) Controlled variance: tight enough distribution to give identifiable likelihood
		#   n_sub=4 kept unchanged — only parameter transform changed in this iteration.
		#
		# Sub-step mechanics:
		# - Reed-Frost exact discrete hazard: p = 1 - exp(-rate * dt) for each transition
		# - Simultaneous draws from start-of-sub-step state → no ordering bias
		# - Exact conservation: S+I+R = N at every sub-step (integer arithmetic)
		# - S strictly non-increasing, R strictly non-decreasing at every sub-step

		n_sub = 4            # ()  four sub-steps per day (empirically best from iteration history)
		dt    = 1.0 / n_sub  # ()  0.25 days per sub-step

		# Running integer state — updated across all n_sub sub-steps
		S_sub = S  # ()  current susceptibles
		I_sub = I  # ()  current infected
		R_sub = R  # ()  current recovered

		for _ in range(n_sub):

			# Early termination: epidemic extinct, no further dynamics possible
			if I_sub == 0:
				break  # ()

			# INFECTION SUB-STEP PROBABILITY
			# Reed-Frost exact hazard for sub-step dt:
			#   p_inf_sub = 1 - exp(-beta * I_sub / N * dt)
			# Frequency-dependent transmission normalised by total population.
			# For small dt: p_inf_sub ≈ beta * I_sub / N * dt (first-order approximation).
			# Exact exponential avoids systematic Euler bias accumulating over n_sub steps.
			force_sub = beta * float(I_sub) / float(N) * dt    # ()  ∈ [0, beta*dt]
			p_inf_sub = float(1.0 - np.exp(-force_sub))         # ()  ∈ [0, 1)
			p_inf_sub = float(np.clip(p_inf_sub, 0.0, 1.0))     # ()  safety clamp to [0,1]

			# RECOVERY SUB-STEP PROBABILITY
			# Reed-Frost exact hazard for sub-step dt:
			#   p_rec_sub = 1 - exp(-gamma * dt)
			# State-independent: constant for fixed gamma and dt (same every sub-step).
			p_rec_sub = float(1.0 - np.exp(-gamma * dt))        # ()  ∈ [0, 1)
			p_rec_sub = float(np.clip(p_rec_sub, 0.0, 1.0))     # ()  safety clamp to [0,1]

			# SIMULTANEOUS BINOMIAL DRAWS FROM START-OF-SUB-STEP STATE
			# Both drawn before updating state → no ordering dependency.
			# Mirrors independent competing Poisson processes of continuous-time Markov SIR:
			# S→I and I→R governed by independent exponential clocks — their draws are independent.
			# Sequential draws (infection then recovery, or vice versa) would introduce systematic
			# bias: recovery-first underestimates I, infection-first overestimates I.

			# new_inf ~ Binomial(S_sub, p_inf_sub): susceptibles becoming infected this sub-step
			if S_sub > 0 and p_inf_sub > 0.0:
				new_inf = int(rng.binomial(n=S_sub, p=p_inf_sub))  # ()  ∈ [0, S_sub]
			else:
				new_inf = 0  # ()

			# new_rec ~ Binomial(I_sub, p_rec_sub): infected recovering this sub-step
			# Drawn from START-OF-SUB-STEP I_sub, not I_sub + new_inf:
			# newly infected individuals do not recover within the same dt interval
			# (they just entered I compartment at the start of the draw)
			if I_sub > 0 and p_rec_sub > 0.0:
				new_rec = int(rng.binomial(n=I_sub, p=p_rec_sub))  # ()  ∈ [0, I_sub]
			else:
				new_rec = 0  # ()

			# Defensive clamps: rng.binomial guarantees output ∈ [0, n],
			# but guard against any float precision edge cases in the prob clipping above
			new_inf = max(0, min(new_inf, S_sub))  # ()  ∈ [0, S_sub]
			new_rec = max(0, min(new_rec, I_sub))  # ()  ∈ [0, I_sub]

			# SIMULTANEOUS COMPARTMENT UPDATE
			# Applied from start-of-sub-step baseline — both transitions take effect at once.
			#
			# Conservation proof every sub-step:
			#   (S_sub - dI) + (I_sub + dI - dR) + (R_sub + dR)
			#   = S_sub + I_sub + R_sub = N  ✓  (exact integer arithmetic)
			#
			# Non-negativity proof every sub-step:
			#   S_next = S_sub - new_inf  ≥ 0           (new_inf ≤ S_sub by clamp) ✓
			#   I_next = I_sub + new_inf - new_rec
			#          ≥ I_sub - new_rec  ≥ I_sub - I_sub = 0  (new_rec ≤ I_sub by clamp) ✓
			#   R_next = R_sub + new_rec  ≥ R_sub        (new_rec ≥ 0) ✓
			#
			# Monotonicity:
			#   S non-increasing: S_sub decreases by new_inf ≥ 0 at every sub-step ✓
			#   R non-decreasing: R_sub increases by new_rec ≥ 0 at every sub-step ✓
			S_sub = S_sub - new_inf                   # ()  non-increasing every sub-step  ✓
			I_sub = I_sub + new_inf - new_rec         # ()  ≥ 0 by proof above             ✓
			R_sub = R_sub + new_rec                   # ()  non-decreasing every sub-step  ✓

		# Final safety clamps: purely defensive, guaranteed not to trigger
		# given sub-step-level non-negativity proofs above, but included for robustness
		S_next = max(0, S_sub)  # ()
		I_next = max(0, I_sub)  # ()
		R_next = max(0, R_sub)  # ()

		next_state = {"S": S_next, "I": I_next, "R": R_next}

		return next_state