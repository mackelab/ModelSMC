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

        if N == 0:
            return next_state

        beta, gamma = parameters[0], parameters[1]

        # Clamp parameters to physically meaningful ranges
        beta = np.clip(beta, 0.0, 1.0)
        gamma = np.clip(gamma, 0.0, 1.0)

        # Compute a seasonality modifier based on the infected fraction
        infected_fraction = I / N if N > 0 else 0.0
        seasonality_factor = 1.0
        if infected_fraction > 0.5:
            # High burden: behavioral change reduces effective transmission
            seasonality_factor = 1.0 - 0.3 * (infected_fraction - 0.5) / 0.5
        elif infected_fraction < 0.01:
            # Very low prevalence: stochastic extinction more likely
            seasonality_factor = 0.85
        else:
            seasonality_factor = 1.0 + 0.05 * np.sin(2 * np.pi * infected_fraction)

        beta = np.clip(beta * seasonality_factor, 0.0, 1.0)

        # Action-based intervention with graduated, non-linear effects
        effective_beta = beta
        action_compliance_noise = rng.uniform(0.9, 1.1)  # Compliance variability

        if action is not None:
            if action == 0:
                # No intervention: possible slight super-spreading due to complacency
                effective_beta = beta * np.clip(action_compliance_noise, 1.0, 1.1)
            elif action == 1:
                # Mild intervention: mask mandates, partial social distancing
                reduction = 0.20 * action_compliance_noise
                effective_beta = beta * max(0.0, 1.0 - reduction)
            elif action == 2:
                # Moderate intervention: school closures, WFH policies
                reduction = 0.50 * action_compliance_noise
                effective_beta = beta * max(0.0, 1.0 - reduction)
                # Also slightly improve recovery due to less healthcare burden
                gamma = np.clip(gamma * 1.05, 0.0, 1.0)
            elif action == 3:
                # Strict lockdown: major reduction in contacts
                reduction = 0.80 * action_compliance_noise
                effective_beta = beta * max(0.0, 1.0 - reduction)
                # Healthcare system has more capacity; recovery improved
                gamma = np.clip(gamma * 1.10, 0.0, 1.0)
            elif action == 4:
                # Emergency measures: near-total isolation
                reduction = 0.95 * action_compliance_noise
                effective_beta = beta * max(0.0, 1.0 - reduction)
                gamma = np.clip(gamma * 1.15, 0.0, 1.0)
            else:
                # Generalized unknown action with diminishing returns
                raw_reduction = 1.0 - np.exp(-action * 0.15)
                effective_beta = beta * max(0.05, 1.0 - raw_reduction * action_compliance_noise)

        effective_beta = np.clip(effective_beta, 0.0, 1.0)

        # Adaptive sub-stepping: more sub-steps when epidemic is more active
        if infected_fraction > 0.2:
            num_substeps = 8
        elif infected_fraction > 0.05:
            num_substeps = 6
        else:
            num_substeps = 4

        S_curr = float(S)
        I_curr = float(I)
        R_curr = float(R)

        total_new_infections = 0
        total_new_recoveries = 0
        cumulative_force = 0.0

        for step in range(num_substeps):
            N_curr = S_curr + I_curr + R_curr

            if N_curr == 0:
                break

            if I_curr <= 0:
                I_curr = 0.0
                break

            sub_beta = effective_beta / num_substeps
            sub_gamma = gamma / num_substeps

            # Heterogeneous mixing: adjust infection rate by sub-step to model non-uniform contact patterns
            if step < num_substeps // 2:
                # Early in the day: higher contact rates (commuting, work)
                contact_modifier = 1.0 + 0.1 * (step / max(1, num_substeps // 2))
            else:
                # Later in the day: lower contact rates (home)
                contact_modifier = 1.0 - 0.05 * ((step - num_substeps // 2) / max(1, num_substeps // 2))

            sub_beta_adj = np.clip(sub_beta * contact_modifier, 0.0, 1.0)

            # Force of infection with density-dependent saturation
            if N_curr > 0:
                raw_force = sub_beta_adj * I_curr / N_curr
                # Saturation effect: at very high infected fractions, force saturates
                saturation_denom = 1.0 + 0.5 * (I_curr / N_curr)
                infection_rate = raw_force / saturation_denom
            else:
                infection_rate = 0.0

            cumulative_force += infection_rate

            infection_prob = np.clip(1.0 - np.exp(-infection_rate), 0.0, 1.0)

            # Recovery probability with slight time-varying fatigue effect on healthcare
            healthcare_strain = np.clip(I_curr / max(1, N_curr), 0.0, 1.0)
            if healthcare_strain > 0.3:
                # Strained healthcare: recovery slows
                gamma_adj = sub_gamma * (1.0 - 0.2 * (healthcare_strain - 0.3) / 0.7)
            else:
                gamma_adj = sub_gamma

            recovery_prob = np.clip(1.0 - np.exp(-gamma_adj), 0.0, 1.0)

            # Stochastic transitions using binomial sampling
            if S_curr >= 1:
                new_infections = rng.binomial(int(S_curr), infection_prob)
            else:
                new_infections = 0

            if I_curr >= 1:
                new_recoveries = rng.binomial(int(I_curr), recovery_prob)
            else:
                new_recoveries = 0

            # Safety clamps
            new_infections = int(np.clip(new_infections, 0, S_curr))
            new_recoveries = int(np.clip(new_recoveries, 0, I_curr))

            S_curr -= new_infections
            I_curr += new_infections - new_recoveries
            R_curr += new_recoveries

            total_new_infections += new_infections
            total_new_recoveries += new_recoveries

            # Extinction threshold with probability: small I may go extinct stochastically
            if I_curr > 0 and I_curr < 3:
                extinction_prob = np.clip(1.0 - (I_curr / 3.0) * 0.8, 0.0, 1.0)
                if rng.uniform() < extinction_prob * 0.3:
                    R_curr += I_curr  # Infected recover or die out
                    I_curr = 0.0
                    break

            if I_curr <= 0:
                I_curr = 0.0
                break

        # Multi-scale demographic noise model
        if N > 100:
            # Scale noise with population size and epidemic intensity
            base_noise_scale = max(1, int(np.sqrt(N) * 0.02))
            epidemic_intensity = np.clip(total_new_infections / max(1, N), 0.0, 1.0)
            dynamic_noise_scale = max(1, int(base_noise_scale * (1.0 + 2.0 * epidemic_intensity)))

            # Apply correlated noise across compartments
            for noise_attempt in range(3):
                s_noise = rng.integers(-dynamic_noise_scale, dynamic_noise_scale + 1)
                i_noise = rng.integers(-dynamic_noise_scale, dynamic_noise_scale + 1)
                r_noise = -(s_noise + i_noise)

                candidate_S = S_curr + s_noise
                candidate_I = I_curr + i_noise
                candidate_R = R_curr + r_noise

                if candidate_S >= 0 and candidate_I >= 0 and candidate_R >= 0:
                    S_curr = candidate_S
                    I_curr = candidate_I
                    R_curr = candidate_R
                    break
                # If invalid, try smaller noise on next attempt
                dynamic_noise_scale = max(1, dynamic_noise_scale // 2)

        elif N > 20:
            # Smaller populations: simpler noise
            noise_flip = rng.uniform()
            if noise_flip < 0.1 and S_curr > 1 and I_curr >= 0:
                # Small chance of an extra imported infection
                extra = rng.integers(1, 3)
                extra = min(extra, int(S_curr))
                S_curr -= extra
                I_curr += extra

        # Final safety clamp of all compartments
        S_curr = max(0.0, S_curr)
        I_curr = max(0.0, I_curr)
        R_curr = max(0.0, R_curr)

        # Re-normalize to strictly preserve total population
        total_curr = S_curr + I_curr + R_curr
        if total_curr == 0:
            # Degenerate case: reset to recovered
            R_curr = float(N)
            S_curr = 0.0
            I_curr = 0.0
        elif total_curr != N:
            diff = N - total_curr
            # Distribute discrepancy preferentially to the largest compartment
            compartments = [("S", S_curr), ("I", I_curr), ("R", R_curr)]
            compartments_sorted = sorted(compartments, key=lambda x: x[1], reverse=True)
            for name, val in compartments_sorted:
                candidate = val + diff
                if candidate >= 0:
                    if name == "S":
                        S_curr = candidate
                    elif name == "I":
                        I_curr = candidate
                    else:
                        R_curr = candidate
                    break

        next_state["S"] = int(round(S_curr))
        next_state["I"] = int(round(I_curr))
        next_state["R"] = int(round(R_curr))

        # Final population conservation check after rounding
        final_total = next_state["S"] + next_state["I"] + next_state["R"]
        if final_total != N:
            correction = N - final_total
            if next_state["R"] + correction >= 0:
                next_state["R"] += correction
            elif next_state["S"] + correction >= 0:
                next_state["S"] += correction
            else:
                next_state["I"] = max(0, next_state["I"] + correction)

        return next_state


    def evaluate(self, x):
        return 0

