import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default parameters: beta=0.3 (transmission), gamma=0.1 (recovery)
		self.parameters = np.array([0.3, 0.1])  # shape: (2,)
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
		# Clip to physically valid ranges: beta in (0,1), gamma in (0,1)
		self.parameters = np.clip(parameters, 1e-6, 1.0 - 1e-6)  # shape: (2,)

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

		# Extract compartment values — all scalars (int)
		S = int(state['S'])  # scalar
		I = int(state['I'])  # scalar
		R = int(state['R'])  # scalar

		# Total population — conserved quantity, scalar
		N = S + I + R  # scalar

		# Guard: if population is zero or no infected, no dynamics possible
		if N <= 0 or I <= 0:
			return {'S': S, 'I': I, 'R': R}

		# Extract and clip parameters to valid ranges — scalars
		beta  = float(np.clip(parameters[0], 1e-6, 1.0 - 1e-6))  # transmission rate, scalar
		gamma = float(np.clip(parameters[1], 1e-6, 1.0 - 1e-6))  # recovery rate, scalar

		# --- Infection process ---
		# Force of infection: lambda = beta * I / N
		# Probability a susceptible becomes infected in one day:
		# p_inf = 1 - exp(-beta * I / N)   [exponential waiting time approximation]
		force_of_infection = beta * float(I) / float(N)  # scalar
		p_inf = 1.0 - np.exp(-force_of_infection)        # scalar, in (0,1)
		p_inf = float(np.clip(p_inf, 0.0, 1.0))          # scalar, safety clip

		# Stochastic new infections drawn from Binomial(S, p_inf)
		new_infections = int(rng.binomial(n=S, p=p_inf))  # scalar, in [0, S]

		# --- Recovery process ---
		# Probability an infected recovers in one day:
		# p_rec = 1 - exp(-gamma)
		p_rec = 1.0 - np.exp(-gamma)                     # scalar, in (0,1)
		p_rec = float(np.clip(p_rec, 0.0, 1.0))          # scalar, safety clip

		# Stochastic new recoveries drawn from Binomial(I, p_rec)
		new_recoveries = int(rng.binomial(n=I, p=p_rec))  # scalar, in [0, I]

		# --- State update ---
		S_new = S - new_infections                        # scalar
		I_new = I + new_infections - new_recoveries       # scalar
		R_new = R + new_recoveries                        # scalar

		# Safety: ensure non-negative compartments (numerical guard)
		S_new = max(0, S_new)  # scalar
		I_new = max(0, I_new)  # scalar
		R_new = max(0, R_new)  # scalar

		next_state = {
			'S': S_new,  # int scalar
			'I': I_new,  # int scalar
			'R': R_new,  # int scalar
		}

		return next_state