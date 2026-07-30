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
        """Improved version of `forward_v2`."""
        next_state = state.copy()

        S = int(state["S"])
        I = int(state["I"])
        R = int(state["R"])
        N = S + I + R

        if N == 0 or I < 0 or S < 0 or R < 0:
            return next_state

        beta, gamma = float(parameters[0]), float(parameters[1])

        # Clamp parameters to valid ranges
        beta = np.clip(beta, 0.0, 10.0)
        gamma = np.clip(gamma, 0.0, 1.0)

        # ---------------------------------------------------------------
        # Action-based intervention with compounding effects
        # ---------------------------------------------------------------
        effective_beta = beta
        compliance_rate = 1.0
        intervention_fatigue = 0.0

        if action is not None:
            if action == 0:
                effective_beta = beta
                compliance_rate = 1.0
            elif action == 1:
                effective_beta = beta * 0.75
                compliance_rate = 0.95
                intervention_fatigue = 0.01
            elif action == 2:
                effective_beta = beta * 0.50
                compliance_rate = 0.85
                intervention_fatigue = 0.03
            elif action == 3:
                effective_beta = beta * 0.20
                compliance_rate = 0.70
                intervention_fatigue = 0.08
            elif action == 4:
                # Targeted quarantine of symptomatic individuals
                symptomatic_fraction = 0.6
                effective_beta = beta * (1.0 - symptomatic_fraction * 0.9)
                compliance_rate = 0.90
                intervention_fatigue = 0.05
            else:
                effective_beta = beta * 0.90
                compliance_rate = 0.80
                intervention_fatigue = 0.02

            # Adjust effective_beta by compliance and fatigue
            effective_beta *= (compliance_rate - intervention_fatigue * rng.random())
            effective_beta = np.clip(effective_beta, 0.0, beta)

        # ---------------------------------------------------------------
        # Multi-wave seasonal forcing with stochastic noise
        # ---------------------------------------------------------------
        season_amplitude_primary = 0.15
        season_amplitude_secondary = 0.05
        day_of_year = rng.integers(0, 365)
        noise_term = rng.normal(0.0, 0.02)

        seasonal_factor = (
            1.0
            + season_amplitude_primary * np.sin(2 * np.pi * day_of_year / 365)
            + season_amplitude_secondary * np.sin(4 * np.pi * day_of_year / 365 + np.pi / 4)
            + noise_term
        )
        seasonal_factor = np.clip(seasonal_factor, 0.5, 1.8)
        effective_beta *= seasonal_factor

        # ---------------------------------------------------------------
        # Age-stratified mixing: split population into 3 age groups
        # ---------------------------------------------------------------
        age_fractions = [0.20, 0.55, 0.25]  # young, adult, elderly
        age_susceptibility = [1.2, 1.0, 1.5]  # relative susceptibility
        age_severity = [0.1, 0.3, 0.7]        # fraction needing hospitalization

        age_S = [max(0, int(round(S * f))) for f in age_fractions]
        age_I = [max(0, int(round(I * f))) for f in age_fractions]
        age_R = [max(0, int(round(R * f))) for f in age_fractions]

        # Correct rounding errors to match totals
        for compartment_list, total in [(age_S, S), (age_I, I), (age_R, R)]:
            diff = total - sum(compartment_list)
            compartment_list[-1] = max(0, compartment_list[-1] + diff)

        # ---------------------------------------------------------------
        # Force of infection (heterogeneous mixing)
        # ---------------------------------------------------------------
        # Assortative mixing matrix (higher mixing within same age group)
        mixing_matrix = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ])

        force_of_infection_age = np.zeros(3)
        if N > 1:
            for i in range(3):
                foi = 0.0
                for j in range(3):
                    foi += mixing_matrix[i, j] * (age_I[j] / N)
                force_of_infection_age[i] = effective_beta * age_susceptibility[i] * foi
        else:
            force_of_infection_age = np.zeros(3)

        # ---------------------------------------------------------------
        # Infection dynamics per age group
        # ---------------------------------------------------------------
        mild_fraction = 0.8
        prob_recovery_mild = np.clip(1.0 - np.exp(-gamma * 1.5), 0.0, 1.0)
        prob_recovery_severe = np.clip(1.0 - np.exp(-gamma * 0.5), 0.0, 1.0)

        total_new_infections_from_S = 0
        total_new_recoveries = 0
        total_hospitalizations = 0

        new_age_S = list(age_S)
        new_age_I = list(age_I)
        new_age_R = list(age_R)

        for g in range(3):
            foi_g = force_of_infection_age[g]
            prob_inf_g = np.clip(1.0 - np.exp(-foi_g), 0.0, 1.0)

            # New infections from susceptibles in group g
            inf_g = int(rng.binomial(age_S[g], prob_inf_g)) if age_S[g] > 0 else 0

            # Split infected group g into mild and severe
            I_mild_g = int(round(age_I[g] * mild_fraction))
            I_severe_g = age_I[g] - I_mild_g

            # Recoveries
            rec_mild_g = int(rng.binomial(I_mild_g, prob_recovery_mild)) if I_mild_g > 0 else 0
            rec_severe_g = int(rng.binomial(I_severe_g, prob_recovery_severe)) if I_severe_g > 0 else 0
            rec_g = rec_mild_g + rec_severe_g

            # Hospitalizations (subset of severe group recoveries delayed)
            hosp_g = int(round(I_severe_g * age_severity[g] * 0.1))
            hosp_g = min(hosp_g, I_severe_g)
            total_hospitalizations += hosp_g

            total_new_infections_from_S += inf_g
            total_new_recoveries += rec_g

            new_age_S[g] = age_S[g] - inf_g
            new_age_I[g] = age_I[g] + inf_g - rec_g
            new_age_R[g] = age_R[g] + rec_g

        # ---------------------------------------------------------------
        # Waning immunity with age-dependent rates
        # ---------------------------------------------------------------
        waning_rates = [0.0005, 0.001, 0.002]  # elderly wane faster
        total_reinfections = 0

        for g in range(3):
            waning_rate_g = waning_rates[g]
            prob_reinfection_g = np.clip(
                1.0 - np.exp(-waning_rate_g * effective_beta), 0.0, 0.05
            )
            reinfections_g = int(rng.binomial(new_age_R[g], prob_reinfection_g)) if new_age_R[g] > 0 else 0
            total_reinfections += reinfections_g
            new_age_R[g] = max(0, new_age_R[g] - reinfections_g)
            new_age_I[g] += reinfections_g

        # ---------------------------------------------------------------
        # Superspreader events (multiple types)
        # ---------------------------------------------------------------
        superspreader_events = [
            {"prob": 0.01, "multiplier_range": (5, 15), "label": "mass_gathering"},
            {"prob": 0.02, "multiplier_range": (2, 6),  "label": "workplace"},
            {"prob": 0.005, "multiplier_range": (10, 30), "label": "healthcare"},
        ]

        total_extra_infections = 0
        for event in superspreader_events:
            if rng.random() < event["prob"] and I > 0:
                multiplier = rng.integers(event["multiplier_range"][0], event["multiplier_range"][1])
                current_S_total = sum(new_age_S)
                already_infected = total_new_infections_from_S + total_extra_infections
                extra_pool = max(0, current_S_total - already_infected)
                if extra_pool > 0:
                    baseline_prob = np.clip(1.0 - np.exp(-effective_beta * I / N), 0.0, 1.0)
                    extra_prob = np.clip(baseline_prob * multiplier, 0.0, 1.0)
                    extra_inf = int(rng.binomial(extra_pool, extra_prob))
                    total_extra_infections += extra_inf

                    # Distribute extra infections across age groups proportionally
                    distributed = 0
                    for g in range(3):
                        if g < 2:
                            share = int(round(extra_inf * age_fractions[g]))
                        else:
                            share = extra_inf - distributed
                        share = min(share, new_age_S[g])
                        new_age_S[g] = max(0, new_age_S[g] - share)
                        new_age_I[g] += share
                        distributed += share

        # ---------------------------------------------------------------
        # Vaccination campaign (stochastic uptake)
        # ---------------------------------------------------------------
        vaccination_rate = 0.002  # base daily vaccination rate
        vaccine_efficacy = 0.85
        vaccine_hesitancy_factors = [0.7, 0.85, 0.90]  # lower uptake in young

        for g in range(3):
            uptake = vaccination_rate * vaccine_hesitancy_factors[g]
            vaccinated_g = int(rng.binomial(new_age_S[g], np.clip(uptake, 0.0, 1.0))) if new_age_S[g] > 0 else 0
            # Effective vaccinations move to R (immune) based on efficacy
            effective_vaccinated = int(rng.binomial(vaccinated_g, vaccine_efficacy)) if vaccinated_g > 0 else 0
            new_age_S[g] = max(0, new_age_S[g] - effective_vaccinated)
            new_age_R[g] += effective_vaccinated

        # ---------------------------------------------------------------
        # Aggregate compartments
        # ---------------------------------------------------------------
        new_S = sum(new_age_S)
        new_I = sum(new_age_I)
        new_R = sum(new_age_R)

        # ---------------------------------------------------------------
        # Enforce non-negativity
        # ---------------------------------------------------------------
        if new_S < 0:
            overflow = -new_S
            new_S = 0
            new_I = max(0, new_I - overflow)

        if new_I < 0:
            new_I = 0

        if new_R < 0:
            new_R = 0

        # ---------------------------------------------------------------
        # Population conservation with intelligent redistribution
        # ---------------------------------------------------------------
        total_new = new_S + new_I + new_R
        if total_new != N:
            diff = N - total_new
            if diff > 0:
                new_S += diff
            else:
                abs_diff = abs(diff)
                if new_S >= abs_diff:
                    new_S -= abs_diff
                elif new_S + new_R >= abs_diff:
                    remainder = abs_diff - new_S
                    new_S = 0
                    new_R = max(0, new_R - remainder)
                else:
                    new_S = 0
                    new_R = 0
                    new_I = N

        # ---------------------------------------------------------------
        # Final integrity check with iterative correction
        # ---------------------------------------------------------------
        corrections_applied = 0
        max_corrections = 5

        for _ in range(max_corrections):
            any_negative = False
            for val, label in [(new_S, "S"), (new_I, "I"), (new_R, "R")]:
                if val < 0:
                    any_negative = True
                    if label == "S":
                        new_S = 0
                    elif label == "I":
                        new_I = 0
                    elif label == "R":
                        new_R = 0
                    corrections_applied += 1

            if not any_negative:
                break

            # Re-balance after zeroing
            total_check = new_S + new_I + new_R
            if total_check < N:
                new_S += N - total_check
            elif total_check > N:
                excess = total_check - N
                if new_S >= excess:
                    new_S -= excess
                else:
                    new_S = 0

        # Clamp all final values
        new_S = max(0, int(new_S))
        new_I = max(0, int(new_I))
        new_R = max(0, int(new_R))

        # Final population fix if still off
        final_total = new_S + new_I + new_R
        if final_total != N:
            new_S = max(0, new_S + (N - final_total))

        next_state["S"] = int(new_S)
        next_state["I"] = int(new_I)
        next_state["R"] = int(new_R)
        next_state["hospitalizations"] = int(total_hospitalizations)
        next_state["vaccinated_today"] = int(
            sum(
                int(rng.binomial(0, 0.0))
                for _ in range(3)
            )
        )

        return next_state


    def evaluate(self, x):
        return 0

