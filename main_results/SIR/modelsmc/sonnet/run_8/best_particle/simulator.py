import numpy as np
import torch.nn as nn


class SimulatorStep(nn.Module):
	def __init__(self):
		"""
		COVID SIR environment.
		"""
		super(SimulatorStep, self).__init__()
		# Default parameters: beta=0.3, gamma=0.1
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
		# Clip to biologically valid ranges to avoid numerical issues
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
		Implements one simulation step using deterministic Euler ODE integration
		with light additive Gaussian noise scaled to flow magnitude.

		Args:
			state (dict): The environment state represented by a dictionary: "S": int, "I": int, "R": int
			parameters (np.ndarray): Array of size (2,) containing model parameters.
			action (int | None): None.
			rng (np.random.Generator): Random number generator.

		Returns:
			next_state (dict): The next environment state represented by a dictionary: "S": int, "I": int, "R": int.
		"""

		# --- Extract compartment values as floats to avoid truncation error ---
		S = float(state["S"])  # scalar float
		I = float(state["I"])  # scalar float
		R = float(state["R"])  # scalar float

		# --- Total population (conserved) ---
		N = S + I + R  # scalar float

		# Guard: if population is zero or no infected, trivially return state
		if N <= 0.0:
			return state.copy()

		# --- Extract and clip parameters ---
		beta  = float(np.clip(parameters[0], 1e-6, 1.0 - 1e-6))  # scalar: transmission rate
		gamma = float(np.clip(parameters[1], 1e-6, 1.0 - 1e-6))  # scalar: recovery rate

		# --- Deterministic Euler ODE flows (one day step) ---
		# Infection flow: S → I
		infection_flow = beta * S * I / N   # scalar float, always >= 0
		# Recovery flow: I → R
		recovery_flow  = gamma * I          # scalar float, always >= 0

		# Clamp flows so they cannot exceed available compartment sizes
		infection_flow = float(np.clip(infection_flow, 0.0, S))  # scalar float
		recovery_flow  = float(np.clip(recovery_flow,  0.0, I))  # scalar float

		# --- Light additive Gaussian observation noise scaled to sqrt of flow ---
		# Noise scale: proportional to sqrt(flow) to mimic Poisson-like variance
		# noise_scale << 1 to keep trajectories tight around ODE solution
		noise_scale = 0.5  # scalar: small fraction of Poisson-like std

		# Noise on infection flow — scalar float
		sigma_inf = noise_scale * np.sqrt(max(infection_flow, 1.0))  # scalar float
		delta_inf = float(rng.normal(0.0, sigma_inf))                # scalar float

		# Noise on recovery flow — scalar float
		sigma_rec = noise_scale * np.sqrt(max(recovery_flow, 1.0))   # scalar float
		delta_rec = float(rng.normal(0.0, sigma_rec))                # scalar float

		# Perturbed flows, clamped to valid ranges
		noisy_infection = float(np.clip(infection_flow + delta_inf, 0.0, S))  # scalar float
		noisy_recovery  = float(np.clip(recovery_flow  + delta_rec, 0.0, I))  # scalar float

		# --- Update compartments using float arithmetic ---
		S_new = S - noisy_infection                          # scalar float
		I_new = I + noisy_infection - noisy_recovery         # scalar float

		# Clamp S and I to non-negative values
		S_new = max(0.0, S_new)  # scalar float
		I_new = max(0.0, I_new)  # scalar float

		# Enforce exact population conservation analytically: R = N - S - I
		R_new = max(0.0, N - S_new - I_new)  # scalar float

		# --- Round to nearest integer at output stage only ---
		S_out = int(round(S_new))  # scalar int
		I_out = int(round(I_new))  # scalar int
		R_out = int(round(R_new))  # scalar int

		# Final conservation enforcement after rounding (correct R residually)
		total_out = S_out + I_out + R_out  # scalar int
		N_int = int(round(N))              # scalar int
		if total_out != N_int:
			R_out = max(0, R_out + (N_int - total_out))  # scalar int

		next_state = {
			"S": int(S_out),  # scalar int
			"I": int(I_out),  # scalar int
			"R": int(R_out),  # scalar int
		}

		return next_state