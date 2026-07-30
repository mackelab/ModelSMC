import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Empirically grounded COVID-19 defaults: beta~0.25 (transmission), gamma~0.07 (~14-day infectious period), R0~3.5
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
		Implements one simulation step.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing model parameters [beta, gamma].
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state: "S": int, "I": int, "R": int.
		"""

		# --- Extract current compartment counts ---
		S = int(state["S"])  # shape: scalar — susceptible count
		I = int(state["I"])  # shape: scalar — infected count
		R = int(state["R"])  # shape: scalar — recovered count

		# Total population (strictly conserved) — scalar
		N = S + I + R  # shape: scalar

		# Edge cases: no population or epidemic already extinct — return unchanged
		if N == 0 or I == 0:
			return {"S": S, "I": I, "R": R}

		# --- Extract and constrain parameters to COVID-plausible ranges ---
		# beta in [0.1, 0.5]: covers R0 in [0.5, 10] for gamma in [0.05, 0.2]
		beta  = float(np.clip(parameters[0], 0.1, 0.5))   # shape: scalar — daily transmission rate
		# gamma in [0.05, 0.2]: infectious periods of 5–20 days, typical for COVID-19
		gamma = float(np.clip(parameters[1], 0.05, 0.2))  # shape: scalar — daily recovery rate

		# --- Compute exponential waiting-time transition probabilities ---
		# Exact mapping from continuous-time hazard rate to discrete daily probability
		lambda_t  = beta * float(I) / float(N)                           # shape: scalar — per-capita infection hazard
		p_infect  = float(np.clip(1.0 - np.exp(-lambda_t), 0.0, 1.0))   # shape: scalar — S→I daily transition probability
		p_recover = float(np.clip(1.0 - np.exp(-gamma),    0.0, 1.0))   # shape: scalar — I→R daily transition probability

		# --- Binomial sampling: canonical stochastic discrete-time SIR transitions ---
		# Each of the S susceptibles independently gets infected with probability p_infect
		# Binomial(S, p_infect) is the exact distribution — naturally bounded in [0, S]
		new_infections = int(rng.binomial(n=S, p=p_infect))   # shape: scalar — S→I flow, in [0, S]

		# Each of the I infected individuals independently recovers with probability p_recover
		# Binomial(I, p_recover) is the exact distribution — naturally bounded in [0, I]
		new_recoveries = int(rng.binomial(n=I, p=p_recover))  # shape: scalar — I→R flow, in [0, I]

		# --- Update compartments (bounds respected by Binomial construction) ---
		S_next = S - new_infections                    # shape: scalar
		I_next = I + new_infections - new_recoveries   # shape: scalar
		R_next = R + new_recoveries                    # shape: scalar

		# --- Enforce non-negativity as a final numerical safety guard ---
		S_next = max(0, S_next)  # shape: scalar
		I_next = max(0, I_next)  # shape: scalar
		R_next = max(0, R_next)  # shape: scalar

		# --- Strict population conservation: absorb any residual drift into R ---
		total_next  = S_next + I_next + R_next   # shape: scalar
		discrepancy = N - total_next              # shape: scalar — should be exactly 0 for Binomial draws
		R_next = max(0, R_next + discrepancy)     # shape: scalar — restore exact conservation

		next_state = {
			"S": int(S_next),  # shape: scalar
			"I": int(I_next),  # shape: scalar
			"R": int(R_next),  # shape: scalar
		}

		return next_state