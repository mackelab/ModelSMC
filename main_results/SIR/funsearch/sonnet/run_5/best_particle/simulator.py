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
        return self.forward(state = state, parameters = self.get_parameters(), action = action, rng = rng)


    def forward(self, state: dict, parameters: np.ndarray, action: int | None, rng: np.random.Generator) -> dict:
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
        next_state = state.copy()

        S = state["S"]
        I = state["I"]
        R = state["R"]
        N = S + I + R

        if N <= 0:
            return next_state

        beta, gamma = parameters[0], parameters[1]

        # Apply action-based intervention if action is provided
        if action is not None:
            if action == 0:
                # No intervention
                pass
            elif action == 1:
                # Mild intervention: reduce beta by 20%
                beta = beta * 0.8
            elif action == 2:
                # Moderate intervention: reduce beta by 50%
                beta = beta * 0.5
            elif action == 3:
                # Strict lockdown: reduce beta by 80%
                beta = beta * 0.2
            else:
                # Unknown action: default clamp beta
                beta = max(0.0, beta - 0.05 * action)

        # Clamp parameters to valid ranges
        beta = np.clip(beta, 0.0, 1.0)
        gamma = np.clip(gamma, 0.0, 1.0)

        # Compute stochastic transitions using binomial draws
        new_infections = 0
        new_recoveries = 0

        if S > 0 and I > 0:
            # Probability of a susceptible individual getting infected
            infection_prob = 1.0 - np.exp(-beta * I / N)
            infection_prob = np.clip(infection_prob, 0.0, 1.0)

            # Draw number of new infections stochastically
            new_infections = rng.binomial(S, infection_prob)

        if I > 0:
            # Probability of an infected individual recovering
            recovery_prob = 1.0 - np.exp(-gamma)
            recovery_prob = np.clip(recovery_prob, 0.0, 1.0)

            # Draw number of new recoveries stochastically
            new_recoveries = rng.binomial(I, recovery_prob)

        # Additional complexity: super-spreader events
        super_spreader_prob = 0.05  # 5% chance of a super-spreader event
        if I > 0 and rng.random() < super_spreader_prob:
            extra_infections = rng.poisson(lam=max(1, int(0.1 * I)))
            extra_infections = min(extra_infections, S - new_infections)
            if extra_infections > 0:
                new_infections += extra_infections

        # Additional complexity: spontaneous re-susceptibility (waning immunity)
        re_susceptible = 0
        if R > 0:
            waning_prob = 0.01  # 1% chance per step of losing immunity
            re_susceptible = rng.binomial(R, waning_prob)

        # Additional complexity: external importation of infections
        importation_rate = 0.001
        if rng.random() < importation_rate * N:
            imported_cases = rng.poisson(lam=1)
            # Imported cases come from susceptible pool
            imported_cases = min(imported_cases, S - new_infections)
            if imported_cases > 0:
                new_infections += imported_cases

        # Enforce constraints so counts don't go negative
        new_infections = int(np.clip(new_infections, 0, S))
        new_recoveries = int(np.clip(new_recoveries, 0, I))
        re_susceptible = int(np.clip(re_susceptible, 0, R))

        # Multi-step micro-simulation within one macro step
        micro_steps = 3
        dS = 0
        dI = 0
        dR = 0

        for step in range(micro_steps):
            fraction = 1.0 / micro_steps

            micro_infections = int(np.round(new_infections * fraction))
            micro_recoveries = int(np.round(new_recoveries * fraction))
            micro_re_susceptible = int(np.round(re_susceptible * fraction))

            # Adjust last micro-step to account for rounding errors
            if step == micro_steps - 1:
                micro_infections = new_infections - dS * (-1) - (micro_steps - 1) * int(np.round(new_infections / micro_steps))
                micro_recoveries = new_recoveries - dR - (micro_steps - 1) * int(np.round(new_recoveries / micro_steps))
                micro_re_susceptible = re_susceptible - (micro_steps - 1) * int(np.round(re_susceptible / micro_steps))

            micro_infections = max(0, micro_infections)
            micro_recoveries = max(0, micro_recoveries)
            micro_re_susceptible = max(0, micro_re_susceptible)

            dS += -micro_infections + micro_re_susceptible
            dI += micro_infections - micro_recoveries
            dR += micro_recoveries - micro_re_susceptible

        # Final state update
        new_S = int(np.clip(S + dS, 0, N))
        new_I = int(np.clip(I + dI, 0, N))
        new_R = int(np.clip(R + dR, 0, N))

        # Correct any population conservation issues
        total = new_S + new_I + new_R
        if total != N:
            discrepancy = N - total
            # Add discrepancy to the largest compartment
            compartments = {"S": new_S, "I": new_I, "R": new_R}
            largest = max(compartments, key=compartments.get)
            if largest == "S":
                new_S += discrepancy
            elif largest == "I":
                new_I += discrepancy
            else:
                new_R += discrepancy

        # Final clamp to ensure non-negativity
        new_S = max(0, new_S)
        new_I = max(0, new_I)
        new_R = max(0, new_R)

        next_state["S"] = new_S
        next_state["I"] = new_I
        next_state["R"] = new_R

        return next_state


    def evaluate(self, x):
        return 0

