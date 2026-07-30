import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default COVID-realistic parameters: beta=0.25 (transmission), gamma=0.07 (recovery ~14 days, R0~3.6)
		self.parameters = np.array([0.25, 0.07])  # shape: (2,)
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
		Implements one simulation step of the canonical stochastic discrete-time SIR model.
		Uses Binomial draws for both infection and recovery transitions with
		exponential waiting-time probabilities for numerical stability.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing [beta, gamma].
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state: "S": int, "I": int, "R": int.
		"""

		# Extract current compartment counts
		S = int(state["S"])  # scalar int: number of susceptible individuals
		I = int(state["I"])  # scalar int: number of infected individuals
		R = int(state["R"])  # scalar int: number of recovered individuals

		# Total population — conserved throughout simulation
		N = S + I + R  # scalar int

		# Extract parameters and clip to COVID-realistic ranges
		# beta: transmission rate; COVID R0 ~1.5–4 with gamma ~0.07 → beta ~0.10–0.28
		# Clip to [0.05, 0.6] to prevent degenerate instant-infection trajectories
		beta = float(np.clip(parameters[0], 0.05, 0.6))   # scalar float

		# gamma: recovery rate; infectious period ~5–20 days → gamma ~0.05–0.20
		gamma = float(np.clip(parameters[1], 0.05, 0.20))  # scalar float

		# --- Infection process: Binomial with exponential waiting-time probability ---
		if N > 0 and S > 0 and I > 0:
			# Force of infection: rate at which a susceptible encounters an infected individual
			# Use exponential form: p = 1 - exp(-beta * I / N) to ensure p in (0, 1) for any beta > 0
			p_infection = float(1.0 - np.exp(-beta * float(I) / float(N)))  # scalar float in (0, 1)
			p_infection = float(np.clip(p_infection, 0.0, 1.0))             # scalar float: safety clamp

			# Binomial draw: number of new infections today
			new_infections = int(rng.binomial(n=S, p=p_infection))  # scalar int
		else:
			new_infections = 0  # scalar int: no infection possible

		# --- Recovery process: Binomial with exponential waiting-time probability ---
		if I > 0:
			# Probability of recovering today: 1 - exp(-gamma) from exponential distribution of recovery times
			p_recovery = float(1.0 - np.exp(-gamma))           # scalar float in (0, 1)
			p_recovery = float(np.clip(p_recovery, 0.0, 1.0))  # scalar float: safety clamp

			# Binomial draw: number of new recoveries today
			new_recoveries = int(rng.binomial(n=I, p=p_recovery))  # scalar int
		else:
			new_recoveries = 0  # scalar int: no recovery possible

		# Enforce valid range for transition counts (safety against numerical edge cases)
		new_infections = int(np.clip(new_infections, 0, S))   # scalar int
		new_recoveries = int(np.clip(new_recoveries, 0, I))   # scalar int

		# Mass-balance update of compartments
		S_next = S - new_infections                   # scalar int: susceptibles decrease
		I_next = I + new_infections - new_recoveries  # scalar int: infected gain new + lose recovered
		R_next = R + new_recoveries                   # scalar int: recovered increase

		# Final non-negativity guarantee
		S_next = int(max(0, S_next))  # scalar int
		I_next = int(max(0, I_next))  # scalar int
		R_next = int(max(0, R_next))  # scalar int

		# Compose and return next state dictionary
		next_state = {
			"S": S_next,  # scalar int
			"I": I_next,  # scalar int
			"R": R_next,  # scalar int
		}

		return next_state