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

		# Extract compartment counts as integers — scalars ()
		S = int(state["S"])  # ()
		I = int(state["I"])  # ()
		R = int(state["R"])  # ()

		# Total conserved population — scalar ()
		N = S + I + R  # ()

		# Guard: no population or epidemic already resolved
		if N <= 0 or I <= 0:
			return {"S": S, "I": I, "R": R}

		# Extract and clamp parameters to COVID-19 epidemiologically valid ranges — scalars ()
		beta  = float(np.clip(parameters[0], 0.05, 1.0))   # () transmission rate per day
		gamma = float(np.clip(parameters[1], 0.03, 0.2))   # () recovery rate per day (5–33 day infectious period)

		# --- Exponential waiting-time transition probabilities ---
		# Probability a susceptible becomes infected today — scalar ()
		# Derived from Poisson process with rate = beta * I / N per day
		lambda_inf  = beta * float(I) / float(N)        # () per-capita infection hazard
		p_infection = 1.0 - np.exp(-lambda_inf)         # () in [0, 1], no overflow possible
		p_infection = float(np.clip(p_infection, 0.0, 1.0))  # () safety clamp

		# Probability an infected individual recovers today — scalar ()
		# Derived from Poisson process with rate = gamma per day
		p_recovery = 1.0 - np.exp(-gamma)               # () in [0, 1]
		p_recovery = float(np.clip(p_recovery, 0.0, 1.0))    # () safety clamp

		# --- Stochastic binomial transitions ---
		# new_infections: number of susceptibles transitioning S -> I today — scalar (int)
		new_infections = int(rng.binomial(n=S, p=p_infection))   # ()

		# new_recoveries: number of infected transitioning I -> R today — scalar (int)
		new_recoveries = int(rng.binomial(n=I, p=p_recovery))    # ()

		# --- Update compartments ---
		S_next = S - new_infections                     # ()
		I_next = I + new_infections - new_recoveries    # ()
		R_next = R + new_recoveries                     # ()

		# Non-negativity guards — scalars ()
		S_next = max(0, S_next)  # ()
		I_next = max(0, I_next)  # ()
		R_next = max(0, R_next)  # ()

		# Enforce strict population conservation: assign any residual drift to R — scalar ()
		total = S_next + I_next + R_next  # ()
		drift = N - total                  # ()
		R_next = max(0, R_next + drift)    # ()

		next_state = {"S": S_next, "I": I_next, "R": R_next}  # dict of ints

		return next_state