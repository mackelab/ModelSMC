import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default raw unconstrained parameters:
		# raw_beta=0.3  → sigmoid≈0.574 → beta  = 0.05 + 0.45*0.574 ≈ 0.308  (R0 ≈ 3.1 at gamma≈0.10)
		# raw_gamma=-0.5 → sigmoid≈0.378 → gamma = 0.04 + 0.14*0.378 ≈ 0.093  (~11-day infectious period)
		self.parameters = np.array([0.3, -0.5])  # shape: (2,)
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
		# Store raw unconstrained values; sigmoid reparameterisation applied in forward()
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
		Implements one simulation step using the canonical discrete-time stochastic SIR
		Markov kernel.  Both transitions use exact Binomial draws from the time-t state,
		giving the correct transition density for Neural Likelihood Estimation.
		Population is exactly conserved by construction (Binomial respects pool sizes).

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing raw unconstrained parameters [raw_beta, raw_gamma].
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# --- Unpack state ---
		S = int(state["S"])  # scalar int: susceptible count at time t
		I = int(state["I"])  # scalar int: infectious count at time t
		R = int(state["R"])  # scalar int: recovered count at time t

		# --- Sigmoid reparameterisation: unconstrained → COVID-realistic ranges ---
		raw_beta  = float(parameters[0])  # scalar float: unconstrained parameter 0
		raw_gamma = float(parameters[1])  # scalar float: unconstrained parameter 1

		# Numerically stable sigmoid: clip prevents overflow for extreme values
		sigmoid_beta  = 1.0 / (1.0 + np.exp(-np.clip(raw_beta,  -30.0, 30.0)))  # scalar float in (0,1)
		sigmoid_gamma = 1.0 / (1.0 + np.exp(-np.clip(raw_gamma, -30.0, 30.0)))  # scalar float in (0,1)

		# beta  ∈ (0.05, 0.50): COVID transmission rate per day
		# gamma ∈ (0.04, 0.18): COVID recovery rate (5.6 to 25-day infectious period)
		# Focused ranges improve SBI posterior concentration with few training samples
		beta  = 0.05 + 0.45 * sigmoid_beta   # scalar float in (0.05, 0.50): transmission rate
		gamma = 0.04 + 0.14 * sigmoid_gamma  # scalar float in (0.04, 0.18): recovery rate

		# --- Total population (constant) ---
		N = S + I + R  # scalar int: total population

		# --- Guard: trivial cases (no population or no infectious individuals) ---
		if N <= 0 or I <= 0:
			return {"S": S, "I": I, "R": R}

		# ============================================================
		# INFECTION TRANSITION — Binomial(S, p_infect)
		# Continuous-time hazard embedding over dt=1 day:
		#   p_infect = 1 - exp(-beta * I/N)
		# This is the exact discrete-time embedding of the continuous SIR ODE.
		# Binomial draw guarantees new_infections ∈ [0, S] without any clamping.
		# ============================================================
		force_of_infection = beta * float(I) / float(N)              # scalar float: per-capita daily hazard
		p_infect = 1.0 - np.exp(-force_of_infection)                  # scalar float in [0,1]: infection probability
		p_infect = float(np.clip(p_infect, 0.0, 1.0))                 # scalar float: numerical safety clamp

		new_infections = int(rng.binomial(n=S, p=p_infect))           # scalar int in [0, S]: stochastic new cases

		# ============================================================
		# RECOVERY TRANSITION — Binomial(I, p_recover)
		# Continuous-time hazard embedding over dt=1 day:
		#   p_recover = 1 - exp(-gamma)
		# Drawn simultaneously from time-t value of I (not updated I).
		# Binomial draw guarantees new_recoveries ∈ [0, I] without any clamping.
		# ============================================================
		p_recover = 1.0 - np.exp(-gamma)                               # scalar float in [0,1]: recovery probability
		p_recover = float(np.clip(p_recover, 0.0, 1.0))                # scalar float: numerical safety clamp

		new_recoveries = int(rng.binomial(n=I, p=p_recover))           # scalar int in [0, I]: stochastic recoveries

		# ============================================================
		# SIMULTANEOUS STATE UPDATE — both flows applied from time-t state
		# Population conservation: S_new + I_new + R_new = N exactly
		#   new_infections  ≤ S  →  S_new = S - new_infections ≥ 0
		#   new_recoveries  ≤ I  →  I_new = I + new_infections - new_recoveries ≥ 0  (may go negative if new_infections < new_recoveries - I = 0, but Binomials on time-t I guarantee new_recoveries ≤ I)
		# ============================================================
		S_new = S - new_infections                                      # scalar int ≥ 0: updated susceptibles
		I_new = I + new_infections - new_recoveries                     # scalar int ≥ 0: updated infectious
		R_new = R + new_recoveries                                      # scalar int ≥ 0: updated recovered

		next_state = {"S": S_new, "I": I_new, "R": R_new}

		return next_state