import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default plausible parameters for COVID: beta=0.3 (transmission), gamma=0.1 (recovery)
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
		# Clip to epidemiologically valid COVID ranges
		beta  = float(np.clip(parameters[0], 0.05, 1.5))   # scalar: transmission rate
		gamma = float(np.clip(parameters[1], 0.01, 0.3))   # scalar: recovery rate
		self.parameters = np.array([beta, gamma])           # shape: (2,)

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

		# Unpack state — all scalars (int)
		S = int(state["S"])   # scalar: susceptible count
		I = int(state["I"])   # scalar: infected count
		R = int(state["R"])   # scalar: recovered count

		# Total population (conserved throughout)
		N = S + I + R         # scalar: total population

		# Unpack and clip parameters to epidemiologically valid COVID ranges
		beta  = float(np.clip(parameters[0], 0.05, 1.5))   # scalar: transmission rate
		gamma = float(np.clip(parameters[1], 0.01, 0.3))   # scalar: recovery rate

		# ------------------------------------------------------------------ #
		# New infections: S -> I
		# Discrete-time hazard formulation: p = 1 - exp(-beta * I / N)
		# Always use Binomial to ensure genuine stochasticity for SBI
		# ------------------------------------------------------------------ #
		if N > 0 and S > 0 and I > 0:
			prob_infection = float(np.clip(1.0 - np.exp(-beta * float(I) / float(N)), 0.0, 1.0))  # scalar in [0,1]
			new_infected   = int(rng.binomial(n=S, p=prob_infection))                               # scalar int
		else:
			new_infected = 0  # scalar int

		# ------------------------------------------------------------------ #
		# New recoveries: I -> R
		# Discrete-time hazard formulation: p = 1 - exp(-gamma)
		# Drawn from pre-step I only: newly infected cannot recover this same step
		# Always use Binomial to ensure genuine stochasticity for SBI
		# ------------------------------------------------------------------ #
		if I > 0:
			prob_recovery  = float(np.clip(1.0 - np.exp(-gamma), 0.0, 1.0))                        # scalar in [0,1]
			new_recovered  = int(rng.binomial(n=I, p=prob_recovery))                                # scalar int
			# Hard cap: recoveries cannot exceed pre-step I count
			new_recovered  = int(np.clip(new_recovered, 0, I))                                      # scalar int
		else:
			new_recovered = 0  # scalar int

		# ------------------------------------------------------------------ #
		# Update compartments:
		# S loses new_infected
		# I loses new_recovered (from old I), then gains new_infected
		# R gains new_recovered
		# ------------------------------------------------------------------ #
		S_next = int(np.clip(S - new_infected,                  0, N))  # scalar int
		I_next = int(np.clip(I - new_recovered + new_infected,  0, N))  # scalar int
		R_next = int(np.clip(R + new_recovered,                 0, N))  # scalar int

		# ------------------------------------------------------------------ #
		# Enforce strict population conservation: N = S_next + I_next + R_next
		# Absorb any integer rounding residual into R
		# ------------------------------------------------------------------ #
		residual = N - (S_next + I_next + R_next)                        # scalar int (typically 0)
		R_next   = int(np.clip(R_next + residual, 0, N))                 # scalar int

		next_state = {
			"S": S_next,   # int
			"I": I_next,   # int
			"R": R_next,   # int
		}

		return next_state