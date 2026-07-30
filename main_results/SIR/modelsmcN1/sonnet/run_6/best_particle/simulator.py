import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		Simultaneous Reed-Frost discrete-time SIR with direct Binomial draws.

		Iteration score history (neg_avg_log_marginal_NLE):
		  Iter 13: phi_i=0.25; Beta-Binomial kappa implicit ≈ large; ALL-TIME BEST: live  -5.92
		  Iter 23: merged clip; NO float() on alpha/beta:                        live  -14.4
		  Iter 26: explicit kappa=(1-phi)/phi=3.0 (too diffuse):                 live -236
		  Iter 27: kappa=1/phi=4.0 (still too diffuse):                          live -237
		  Iter 28: kappa=4.0; float() restored:                                  live -241
		  Iter 29: clean rewrite; float(gamma); kappa=4.0 still:                 live -257
		  Iter 30 (this): REMOVE Beta-Binomial; direct Binomial; widen bounds

		Root cause confirmed by iter-29 feedback:
		  kappa=4.0 → alpha+beta=4.0 → Beta variance = p*(1-p)/5 (extremely diffuse).
		  Over 60 steps, wildly varying per-step infection probability destroys trajectory
		  consistency with observed data → marginal likelihood collapses to -257.
		  Iter-13 'kappa implicit' almost certainly meant NO explicit Beta-Binomial:
		  just direct Binomial(S, p_inf_safe) with Reed-Frost FoI.
		  Feedback prescription: `new_infections = int(rng.binomial(n=S, p=float(p_inf_safe)))`

		Two changes from iter-29:
		  1. REMOVE Beta-Binomial entirely (PRIMARY FIX per iter-29 feedback):
		     - Remove alpha, beta_bb, rng.beta() lines completely
		     - Replace with: new_infections = int(rng.binomial(n=S, p=float(p_inf_safe)))
		     - Eliminates catastrophic trajectory variance from diffuse Beta prior
		     - Matches iter-13 'kappa implicit' formulation → target: recover -5.92

		  2. Widen parameter bounds (SECONDARY FIX per iter-29 feedback):
		     - beta:  [0.05, 0.40] → [0.01, 0.50] (more room for SBI to fit diverse outbreaks)
		     - gamma: [0.05, 0.30] → [0.03, 0.30] (allow longer infectious periods ~33 days)
		     - Improves generalization to unseen state-action data
		     - Feedback: 'will improve generalization' (secondary to Beta-Binomial fix)

		LOCKED SETTINGS carried forward:
		  Reed-Frost force-of-infection: p_inf = 1-(1-beta/N)^I  (confirmed best FoI)
		  p_inf_safe = clip(p_inf, 1e-9, 1-1e-9)                 (confirmed clip bounds)
		  float(p_inf_safe) in rng.binomial infection draw        (Python float stability)
		  float(gamma) in rng.binomial recovery draw              (iter-29 retained, keep)
		  self.parameters = [0.20, 0.15]                         (empirically optimal R0≈1.33)
		  Simultaneous update (both draws from time-t I, S)       (Reed-Frost correctness)
		  Population conservation guard                           (defensive integrity check)
		"""
		super(SimulatorStep, self).__init__()
		# Empirically confirmed optimal defaults from iters 12-13:
		#   R0 = 0.20/0.15 ≈ 1.33; mean infectious period ≈ 6.7 days
		self.parameters = np.array([0.20, 0.15])
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
		Implements one simulation step.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing model parameters.
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# Unpack state
		S = int(state["S"])  # (), susceptible count at time t
		I = int(state["I"])  # (), infected count at time t
		R = int(state["R"])  # (), recovered count at time t
		N = S + I + R        # (), total population; conserved throughout

		# Clamp parameters to valid ranges (widened per iter-29 feedback for generalization)
		beta  = np.clip(parameters[0], 0.01, 0.50)  # (), numpy float64; transmission rate ∈ [0.01,0.50]
		gamma = np.clip(parameters[1], 0.03, 0.30)  # (), numpy float64; recovery rate ∈ [0.03,0.30]

		# Guard: skip dynamics if no population or no infected individuals
		if N == 0 or I == 0:
			return {"S": S, "I": I, "R": R}

		# Reed-Frost force-of-infection: probability a susceptible escapes all I contacts
		# p_inf = 1 - (1 - beta/N)^I; all numpy float64 arithmetic
		p_inf      = 1.0 - (1.0 - beta / N) ** I          # (), numpy float64; infection prob ∈ [0,1]
		p_inf_safe = np.clip(p_inf, 1e-9, 1.0 - 1e-9)     # (), numpy float64; clipped ∈ (1e-9, 1-1e-9)

		# ITER-30 PRIMARY FIX: Direct Binomial infection draw — NO Beta-Binomial
		# Feedback: kappa=4.0 was far too diffuse (Var=p*(1-p)/5), compounding over 60 steps
		# to produce trajectories wildly inconsistent with observations → -257 marginal LL.
		# Iter-13 ALL-TIME BEST (-5.92) used 'kappa implicit' = no explicit Beta-Binomial.
		# Direct Binomial: Var = S*p*(1-p); irreducible minimum variance; matches iter-13.
		# float(p_inf_safe): ensures Python float path in rng.binomial C-layer for stability.
		new_infections = int(rng.binomial(n=S, p=float(p_inf_safe)))   # (), new infections ∈ [0, S]

		# Binomial recovery draw; float(gamma) retained (iter-29 feedback: may help stability)
		new_recoveries = int(rng.binomial(n=I, p=float(gamma)))         # (), new recoveries ∈ [0, I]

		# Simultaneous Reed-Frost compartment update (both draws use time-t values)
		# Population conservation: (S-dI) + (I+dI-dR) + (R+dR) = S+I+R = N ✓
		next_S = int(S - new_infections)                    # (), updated susceptible ≥ 0
		next_I = int(I + new_infections - new_recoveries)   # (), updated infected ≥ 0
		next_R = int(R + new_recoveries)                    # (), updated recovered ≥ 0

		# Population conservation safety guard (defensive; algebraically guaranteed above)
		if (next_S + next_I + next_R) != N:
			next_R = int(max(0, N - next_S - next_I))  # (), residual correction restores N

		return {"S": int(next_S), "I": int(next_I), "R": int(next_R)}