import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default: beta=0.5 (transmission rate, can exceed 1.0), gamma=0.1 (recovery probability)
		# These reflect plausible COVID-19 dynamics with R0 ~ 5 at initialisation
		self.parameters = np.array([0.5, 0.1])  # shape: (2,)
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
		# beta is a rate (not a probability), valid range (0, ~5] for COVID-like dynamics
		# gamma is a daily recovery probability, valid range (0, 1)
		beta_clipped  = np.clip(parameters[0], 1e-6, 5.0)          # scalar float
		gamma_clipped = np.clip(parameters[1], 1e-6, 1.0 - 1e-6)   # scalar float
		self.parameters = np.array([beta_clipped, gamma_clipped])   # shape: (2,)

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
		Implements one discrete stochastic SIR simulation step.

		parameters[0] = beta:  transmission rate, interpreted as rate in exponential survival
		                        formula; valid range (0, 5.0]; R0 = beta / gamma
		parameters[1] = gamma: per-capita daily recovery rate mapped to probability via
		                        1 - exp(-gamma); valid range (0, 1)

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing model parameters.
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# --- Extract state ---
		S = int(state["S"])  # scalar int: current susceptibles
		I = int(state["I"])  # scalar int: current infected
		R = int(state["R"])  # scalar int: current recovered

		# --- Extract and bound parameters ---
		# beta: transmission rate — can exceed 1.0, upper-bounded at 5.0 for stability
		beta  = float(np.clip(parameters[0], 1e-6, 5.0))           # scalar float
		# gamma: recovery rate — used in exponential survival, treated as rate not raw probability
		gamma = float(np.clip(parameters[1], 1e-6, 1.0 - 1e-6))    # scalar float

		# --- Total population (conserved quantity) ---
		N = S + I + R  # scalar int

		# Edge cases: empty population or no active infection
		if N <= 0:
			return {"S": S, "I": I, "R": R}

		if I <= 0:
			# No infected individuals — epidemic is over, no transitions possible
			return {"S": S, "I": I, "R": R}

		# --- Infection transition probability (S -> I) ---
		# Per-susceptible probability of escaping infection over one day:
		#   P(escape) = exp(-beta * I / N)
		# Per-susceptible probability of becoming infected:
		p_infection = 1.0 - np.exp(-beta * float(I) / float(N))    # scalar float in (0, 1)
		p_infection = float(np.clip(p_infection, 0.0, 1.0))         # scalar float, safety guard

		# --- Recovery transition probability (I -> R) ---
		# Exponential waiting-time model: mean infectious period = 1/gamma days
		# Per-infected probability of recovering in one day:
		p_recovery = 1.0 - np.exp(-gamma)                           # scalar float in (0, 1)
		p_recovery = float(np.clip(p_recovery, 0.0, 1.0))           # scalar float, safety guard

		# --- Stochastic binomial draws ---
		new_infections = int(rng.binomial(n=S, p=p_infection))      # scalar int in [0, S]
		new_recoveries = int(rng.binomial(n=I, p=p_recovery))       # scalar int in [0, I]

		# --- Compartment updates ---
		next_S = S - new_infections                                  # scalar int
		next_I = I + new_infections - new_recoveries                 # scalar int
		next_R = R + new_recoveries                                  # scalar int

		# --- Non-negativity guard (binomial draws should respect this, but belt-and-braces) ---
		next_S = max(0, next_S)  # scalar int
		next_I = max(0, next_I)  # scalar int
		next_R = max(0, next_R)  # scalar int

		next_state = {
			"S": next_S,  # scalar int
			"I": next_I,  # scalar int
			"R": next_R,  # scalar int
		}

		return next_state