import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
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
			parameters (np.ndarray): Array of size (2,) containing model parameters [beta, gamma].
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# --- Extract current compartments ---
		S = int(state["S"])  # scalar int — susceptible count
		I = int(state["I"])  # scalar int — infected count
		R = int(state["R"])  # scalar int — recovered count

		# --- Conserved total population ---
		N = S + I + R  # scalar int — total population (invariant throughout simulation)

		# --- Extract and clip parameters to COVID-realistic ranges ---
		# Beta in [0.05, 0.60]: daily effective contact-transmission rate.
		# R0 = beta/gamma; with gamma in [0.04, 0.20] this gives R0 in [0.25, 15],
		# consistent with real COVID-19 observed R0 values of 2–6 for the original strain.
		beta  = float(np.clip(parameters[0], 0.05, 0.60))  # scalar float — transmission rate
		# Gamma in [0.04, 0.20]: daily recovery rate.
		# 1/gamma = 5–25 days infectious period, consistent with COVID-19 clinical data.
		gamma = float(np.clip(parameters[1], 0.04, 0.20))  # scalar float — recovery rate

		# --- Step 1: New infections — Binomial with Poisson-exact hazard ---
		# Derivation: Susceptible faces force-of-infection = beta * I / N per day.
		# Exact daily escape probability = exp(-force_of_infection) under Poisson contacts.
		# P(infection in one day) = 1 - exp(-beta * I / N).
		new_infections = 0  # scalar int — default: no infections this step
		if N > 0 and I > 0 and S > 0:
			force_of_infection = beta * float(I) / float(N)                                    # scalar float — Poisson rate
			p_infection        = float(np.clip(1.0 - np.exp(-force_of_infection), 0.0, 1.0))   # scalar float — exact probability
			new_infections     = int(rng.binomial(n=S, p=p_infection))                         # scalar int — stochastic Binomial draw
			new_infections     = int(np.clip(new_infections, 0, S))                            # scalar int — hard clamp to [0, S]

		# --- Step 2: New recoveries — Binomial with SAME Poisson-exact hazard framework ---
		# Derivation: Each infected individual recovers at rate gamma per day (Poisson process).
		# Exact daily survival probability = exp(-gamma).
		# P(recovery in one day) = 1 - exp(-gamma).
		# Mathematically consistent with infection branch — both derived from exp holding times.
		# Draw is from pre-step I (individuals who were already infected before this step).
		new_recoveries = 0  # scalar int — default: no recoveries this step
		if I > 0:
			p_recover      = float(np.clip(1.0 - np.exp(-gamma), 0.0, 1.0))  # scalar float — recovery prob via exact hazard
			new_recoveries = int(rng.binomial(n=I, p=p_recover))              # scalar int — stochastic Binomial draw
			new_recoveries = int(np.clip(new_recoveries, 0, I))               # scalar int — clamp to [0, I]

		# --- Step 3: Enforce I_new >= 0 by capping new_recoveries ---
		# I_new = I + new_infections - new_recoveries.
		# Since new_recoveries is drawn from pre-step I (not augmented I), it is possible
		# that new_recoveries > I + new_infections in pathological edge cases (e.g. very
		# large gamma). Cap new_recoveries to I + new_infections to guarantee I_new >= 0
		# and eliminate the need for post-hoc rebalancing hacks.
		max_recoveries = I + new_infections                              # scalar int — maximum recoveries that keeps I_new >= 0
		new_recoveries = int(min(new_recoveries, max_recoveries))        # scalar int — constrained recovery count

		# --- Step 4: Apply simultaneous compartment transitions ---
		# Both transitions computed from pre-step state, matching discrete-time SIR likelihood.
		S_new = S - new_infections                    # scalar int — susceptibles reduced by new infections
		I_new = I + new_infections - new_recoveries   # scalar int — infected: gain from S, lose to R
		R_new = R + new_recoveries                    # scalar int — cumulative recovered

		# --- Step 5: Clamp to non-negative integers (final numerical safety) ---
		S_new = max(int(S_new), 0)  # scalar int
		I_new = max(int(I_new), 0)  # scalar int
		R_new = max(int(R_new), 0)  # scalar int

		# --- Step 6: Restore N exactly by correcting any residual integer drift into R ---
		N_new = S_new + I_new + R_new   # scalar int — recomputed total after clamping
		if N_new != N:
			R_new = R_new + (N - N_new)  # scalar int — absorb drift into R compartment
			R_new = max(R_new, 0)        # scalar int — non-negative guard

		next_state = {
			"S": S_new,  # scalar int
			"I": I_new,  # scalar int
			"R": R_new,  # scalar int
		}

		return next_state