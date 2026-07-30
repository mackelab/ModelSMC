import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default COVID-19 parameters: beta~0.25 (R0~2.5), gamma~0.10 (~10-day recovery)
		self.parameters = np.array([0.25, 0.10])  # shape: (2,)
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
			parameters (np.ndarray): Array of size (2,) containing model parameters.
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# ── Extract current compartment counts as integers ────────────────────────
		S = int(state["S"])  # scalar int, susceptible
		I = int(state["I"])  # scalar int, infected
		R = int(state["R"])  # scalar int, recovered

		# ── Clip parameters to tight COVID-realistic physiological ranges ─────────
		# beta ∈ [0.10, 0.90]: transmission rate; R0 = beta/gamma ∈ [~0.3, ~18]
		# gamma ∈ [0.05, 0.33]: recovery rate; 1/gamma ∈ [3, 20] day mean recovery
		beta  = float(np.clip(parameters[0], 0.10, 0.90))  # scalar float
		gamma = float(np.clip(parameters[1], 0.05, 0.33))  # scalar float

		# ── Total (conserved) population ──────────────────────────────────────────
		N = S + I + R  # scalar int

		# Guard: skip dynamics if population is empty or no active infections
		if N <= 0 or I <= 0:
			return {"S": S, "I": I, "R": R}

		# ── Reed-Frost transition probabilities (Poisson process approximation) ───
		# P(susceptible becomes infected) = 1 - exp(-beta * I / N)
		# This is the exact continuous-to-discrete conversion for Poisson contacts
		p_SI = float(np.clip(1.0 - np.exp(-beta * float(I) / float(N)), 0.0, 1.0))  # scalar in [0,1]

		# P(infected recovers) = 1 - exp(-gamma)
		# Exact conversion from continuous recovery rate to daily probability
		p_IR = float(np.clip(1.0 - np.exp(-gamma), 0.0, 1.0))  # scalar in [0,1]

		# ── Canonical stochastic SIR Binomial transitions (Reed-Frost model) ──────
		# Both transitions drawn from pre-step compartment sizes (simultaneous update)
		# Variance ~ S * p_SI * (1 - p_SI) for infections; ~ I * p_IR * (1 - p_IR) for recoveries
		new_SI = int(rng.binomial(n=S, p=p_SI))  # scalar int in [0, S]
		new_IR = int(rng.binomial(n=I, p=p_IR))  # scalar int in [0, I]

		# ── Update compartments ───────────────────────────────────────────────────
		S_next = S - new_SI            # scalar int, must be ≥ 0 by Binomial guarantee
		I_next = I + new_SI - new_IR   # scalar int
		R_next = R + new_IR            # scalar int

		# ── Safety clip each compartment to non-negative ──────────────────────────
		# Binomial draws guarantee new_SI ≤ S and new_IR ≤ I, but clip for robustness
		S_next = int(np.clip(S_next, 0, N))  # scalar int
		I_next = int(np.clip(I_next, 0, N))  # scalar int
		R_next = int(np.clip(R_next, 0, N))  # scalar int

		# ── Enforce strict population conservation ────────────────────────────────
		# Absorb any integer drift into R (smoothest compartment at epidemic tail)
		drift  = N - (S_next + I_next + R_next)          # scalar int, typically 0
		R_next = int(np.clip(R_next + drift, 0, N))      # scalar int

		next_state = {"S": S_next, "I": I_next, "R": R_next}

		return next_state