import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# COVID-19 typical defaults: beta=0.25 (moderate transmission),
		# gamma=0.07 (~14-day mean infectious period, R0 ~ 3.6)
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
		# Tight COVID-19-consistent bounds: beta in [0.05, 0.50], gamma in [0.02, 0.20]
		lower = np.array([0.05, 0.02])  # shape: (2,)
		upper = np.array([0.50, 0.20])  # shape: (2,)
		self.parameters = np.clip(parameters, lower, upper)  # shape: (2,)

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

		# Extract current compartment values
		S = int(state["S"])  # scalar int
		I = int(state["I"])  # scalar int
		R = int(state["R"])  # scalar int

		# Total population (conserved quantity, must remain constant)
		N = S + I + R  # scalar int

		# Extract and clip parameters to COVID-realistic bounds
		beta  = float(np.clip(parameters[0], 0.05, 0.50))  # scalar float, transmission rate
		gamma = float(np.clip(parameters[1], 0.02, 0.20))  # scalar float, recovery rate

		# Handle degenerate cases: no epidemic dynamics possible
		if N <= 0 or I <= 0:
			return {"S": S, "I": I, "R": R}

		# --- Infection transition ---
		# Discrete-time survival probability from continuous-time exponential waiting time
		# Force of infection: lambda = beta * I / N (per-capita daily hazard)
		lambda_infection = beta * float(I) / float(N)              # scalar float, per-capita hazard
		p_infection = 1.0 - np.exp(-lambda_infection)              # scalar float, daily infection prob
		p_infection = float(np.clip(p_infection, 0.0, 1.0))        # scalar float, safety clip

		# Exact Binomial draw: canonical discrete-time stochastic SIR kernel
		new_infections = int(rng.binomial(n=S, p=p_infection))     # scalar int
		new_infections = int(np.clip(new_infections, 0, S))        # scalar int, strict feasibility

		# --- Recovery transition ---
		# Discrete-time survival probability from constant-hazard recovery
		p_recovery = 1.0 - np.exp(-gamma)                          # scalar float, daily recovery prob
		p_recovery = float(np.clip(p_recovery, 0.0, 1.0))          # scalar float, safety clip

		# Exact Binomial draw over ORIGINAL I (before this step's infections enter)
		# Preserves causal ordering: only previously infected individuals can recover today
		new_recoveries = int(rng.binomial(n=I, p=p_recovery))      # scalar int
		new_recoveries = int(np.clip(new_recoveries, 0, I))        # scalar int, strict feasibility

		# --- Update compartments (strict population conservation) ---
		S_new = S - new_infections                    # scalar int
		I_new = I + new_infections - new_recoveries   # scalar int
		R_new = R + new_recoveries                    # scalar int

		# Enforce non-negativity (should already hold from clipping above)
		S_new = int(max(0, S_new))  # scalar int
		I_new = int(max(0, I_new))  # scalar int
		R_new = int(max(0, R_new))  # scalar int

		# Build and return next state
		next_state = {
			"S": S_new,  # scalar int
			"I": I_new,  # scalar int
			"R": R_new,  # scalar int
		}

		return next_state