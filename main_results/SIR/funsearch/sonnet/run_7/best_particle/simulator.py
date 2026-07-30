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
        """Improved version of `forward_v1`."""
        next_state = state.copy()

        S = int(state["S"])
        I = int(state["I"])
        R = int(state["R"])
        N = S + I + R

        if N == 0:
            return next_state

        beta_base = float(parameters[0])
        gamma_base = float(parameters[1])

        # --- Small-world network topology with heterogeneous mixing ---
        num_nodes = 6
        node_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        # Rewiring probability determines long-range connections
        rewiring_prob = 0.15
        base_connectivity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    base_connectivity[i][j] = 1.0
                elif abs(i - j) == 1 or abs(i - j) == num_nodes - 1:
                    base_connectivity[i][j] = 0.5
                else:
                    base_connectivity[i][j] = 0.05 + 0.1 * rng.random() if rng.random() < rewiring_prob else 0.02

        node_S = [int(round(S * w)) for w in node_weights]
        node_I = [int(round(I * w)) for w in node_weights]
        node_R = [int(round(R * w)) for w in node_weights]
        node_N = [int(round(N * w)) for w in node_weights]

        # --- Socioeconomic stratification affecting contact rates ---
        socio_tiers = 5
        socio_fracs = [0.10, 0.20, 0.35, 0.25, 0.10]
        socio_contact_multipliers = [2.5, 1.8, 1.0, 0.7, 0.4]  # lower SES = more contacts
        socio_hygiene_factors = [0.4, 0.6, 0.8, 0.9, 0.95]     # lower SES = worse hygiene
        socio_healthcare_access = [0.3, 0.55, 0.75, 0.9, 0.99]  # lower SES = less access

        # --- Multi-harmonic seasonal forcing with phase drift ---
        phase_drift = rng.normal(0.0, 0.02)
        day_proxy = (S * 0.0011 + I * 0.0023 + R * 0.0017 + N * 0.0009 + phase_drift) % (2 * np.pi)
        week_proxy = (S * 0.0003 + I * 0.0007 + R * 0.0005 + phase_drift * 0.5) % (2 * np.pi)
        month_proxy = (S * 0.00007 + I * 0.00011 + R * 0.00009) % (2 * np.pi)
        seasonal_factor = (
            1.0
            + 0.30 * np.sin(day_proxy)
            + 0.15 * np.cos(2 * day_proxy + np.pi / 4)
            + 0.08 * np.sin(week_proxy + np.pi / 3)
            + 0.04 * np.cos(month_proxy + np.pi / 6)
            + 0.02 * np.sin(3 * day_proxy + np.pi / 5)
        )
        seasonal_factor = max(0.35, min(2.2, seasonal_factor))

        # --- Weather-driven stochastic forcing with autocorrelation ---
        humidity_effect = rng.normal(0.0, 0.06)
        temperature_effect = rng.normal(0.0, 0.04)
        wind_effect = rng.normal(0.0, 0.03)
        climate_factor = max(0.65, 1.0 + humidity_effect + temperature_effect * 0.5 + wind_effect * 0.3)

        beta_modulated = beta_base * seasonal_factor * climate_factor

        # --- Nonlinear density scaling with urban/rural gradient ---
        if N > 1000000:
            density_factor = 1.8 + 0.5 * np.log(N / 1000000)
            urban_factor = 1.3
        elif N > 500000:
            density_factor = 1.5 + 0.4 * np.log(N / 500000)
            urban_factor = 1.2
        elif N > 100000:
            density_factor = 1.2 + 0.25 * np.log(N / 100000)
            urban_factor = 1.1
        elif N > 10000:
            density_factor = 1.0 + 0.1 * np.log(N / 10000)
            urban_factor = 1.0
        elif N > 1000:
            density_factor = 0.85 + 0.15 * (N / 10000)
            urban_factor = 0.9
        elif N > 200:
            density_factor = 0.65 + 0.2 * (N / 1000)
            urban_factor = 0.75
        else:
            density_factor = max(0.2, N / 200.0)
            urban_factor = 0.6

        beta_modulated *= density_factor * urban_factor

        # --- Pathogen evolution with multiple strain competition ---
        strain_fitnesses = [1.0]
        dominant_strain_beta_mult = 1.0
        if I > 30:
            num_circulating_strains = min(5, max(1, int(np.log(I / 30) * 2)))
            mutation_accumulation = 0
            for s in range(num_circulating_strains):
                strain_mutation_rate = min(0.002 * (I / N) * (1 + s * 0.3), 0.02)
                if rng.random() < strain_mutation_rate:
                    fitness_change = rng.choice(
                        [0.7, 0.85, 0.95, 1.05, 1.15, 1.35, 1.6, 1.9],
                        p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03]
                    )
                    strain_fitnesses.append(fitness_change)
                    mutation_accumulation += 1

            if len(strain_fitnesses) > 1:
                dominant_strain_beta_mult = max(strain_fitnesses)
                immune_escape_prob = min(0.3, 0.05 * mutation_accumulation)
                if rng.random() < immune_escape_prob:
                    dominant_strain_beta_mult *= rng.uniform(1.1, 1.4)

        beta_modulated *= dominant_strain_beta_mult

        # --- Vector-borne transmission component ---
        vector_infections = 0
        if N > 0 and seasonal_factor > 1.1:
            vector_density = max(0.0, (seasonal_factor - 1.0) * 0.5)
            vector_transmission_rate = 0.003 * vector_density * (I / N) * climate_factor
            vector_transmission_rate = np.clip(vector_transmission_rate, 0.0, 0.05)
            vector_susceptible = S
            if vector_susceptible > 0 and I > 0:
                vector_infections = int(rng.binomial(vector_susceptible, vector_transmission_rate))
                vector_infections = min(vector_infections, S)

        # --- Node-level infection computation with network effects ---
        node_new_infections = []
        for p in range(num_nodes):
            pS = node_S[p]
            pN = node_N[p]
            if pS <= 0 or pN == 0:
                node_new_infections.append(0)
                continue

            effective_I_pressure = 0.0
            for q in range(num_nodes):
                qI = node_I[q]
                qN = node_N[q]
                if qN > 0:
                    mobility_factor = 1.0 + 0.2 * rng.random()
                    effective_I_pressure += base_connectivity[p][q] * (qI / qN) * mobility_factor

            node_beta = beta_modulated
            node_infection_prob = node_beta * effective_I_pressure
            node_infection_prob = np.clip(node_infection_prob, 0.0, 1.0)

            # Age-stratified cohorts with socioeconomic overlay
            cohort_ages = 7
            cohort_fracs = [0.10, 0.14, 0.18, 0.22, 0.18, 0.12, 0.06]
            cohort_suscept = [0.55, 0.80, 1.0, 1.05, 1.15, 1.4, 1.85]
            cohort_contacts = [0.45, 1.0, 1.35, 1.1, 0.85, 0.65, 0.35]
            cohort_vaccine_protect = [0.75, 0.70, 0.65, 0.55, 0.45, 0.35, 0.25]
            cohort_comorbidity = [0.05, 0.08, 0.12, 0.18, 0.28, 0.40, 0.60]

            node_total_inf = 0
            for c in range(cohort_ages):
                cohort_S_count = int(round(pS * cohort_fracs[c]))
                if cohort_S_count <= 0:
                    continue

                # Socioeconomic-weighted adjustment
                socio_contact_adj = sum(
                    socio_fracs[st] * socio_contact_multipliers[st] * socio_hygiene_factors[st]
                    for st in range(socio_tiers)
                )

                vax_factor = 1.0 - (R / N) * cohort_vaccine_protect[c] * 0.35
                comorbidity_boost = 1.0 + cohort_comorbidity[c] * 0.4

                adjusted_prob = (
                    node_infection_prob
                    * cohort_suscept[c]
                    * cohort_contacts[c]
                    * vax_factor
                    * comorbidity_boost
                    * socio_contact_adj
                )
                adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)

                cohort_inf = int(rng.binomial(cohort_S_count, adjusted_prob))
                node_total_inf += cohort_inf

            node_new_infections.append(node_total_inf)

        new_infections_total = sum(node_new_infections) + vector_infections
        new_infections_total = min(new_infections_total, S)

        # --- Superspreader events with venue-type modeling ---
        superspreader_bonus = 0
        if I > 0:
            venue_types = [
                ("household_cluster", 8, 0.12, 3, 5),
                ("workplace", 20, 0.06, 2, 4),
                ("school", 35, 0.04, 2, 6),
                ("large_event", 80, 0.02, 2, 5),
                ("mass_gathering", 200, 0.008, 2, 4),
                ("superspreader_individual", 50, 0.005, 3, 5),
            ]
            for venue_name, base_size, base_prob, beta_a, beta_b in venue_types:
                scaled_prob = base_prob * min(1.5, I / max(1, N * 0.005)) * seasonal_factor
                scaled_prob = min(scaled_prob, 0.9)
                if rng.random() < scaled_prob:
                    num_events = rng.integers(1, 4)
                    for _ in range(num_events):
                        event_size = int(base_size * rng.lognormal(0, 0.5))
                        event_size = max(1, min(event_size, base_size * 5))
                        attack_rate = rng.beta(beta_a, beta_b)
                        venue_infected = int(event_size * attack_rate)
                        superspreader_bonus += venue_infected

        superspreader_bonus = min(superspreader_bonus, max(0, S - new_infections_total))
        new_infections_total += superspreader_bonus

        # --- Multi-dimensional behavioral feedback ---
        prevalence = I / N
        if prevalence > 0.20:
            fear_factor = max(0.20, 1.0 - 1.5 * ((prevalence - 0.20) / 0.80))
            media_effect = max(0.85, 1.0 - 0.3 * rng.random())
        elif prevalence > 0.10:
            fear_factor = max(0.45, 1.0 - 1.0 * ((prevalence - 0.10) / 0.10))
            media_effect = max(0.90, 1.0 - 0.15 * rng.random())
        elif prevalence > 0.05:
            fear_factor = max(0.60, 1.0 - 0.7 * ((prevalence - 0.05) / 0.05))
            media_effect = max(0.95, 1.0 - 0.05 * rng.random())
        else:
            fear_factor = 1.0
            media_effect = 1.0

        policy_stringency = 1.0
        if prevalence > 0.15:
            policy_stringency = max(0.5, 1.0 - 0.6 * ((prevalence - 0.15) / 0.85))
        elif prevalence > 0.08:
            policy_stringency = max(0.7, 1.0 - 0.35 * ((prevalence - 0.08) / 0.07))

        behavioral_reduction = fear_factor * media_effect * policy_stringency

        behavior_adjusted_infections = int(round(new_infections_total * behavioral_reduction))
        new_infections_total = min(behavior_adjusted_infections, S)

        # --- Multi-pathway environmental contamination ---
        environmental_infections = 0
        if I > 0:
            # Water-borne pathway
            water_load = min(1.0, (I / N) * 5.0)
            water_decay = max(0.2, 1.0 - 0.6 * (R / N))
            water_prob = 0.002 * water_load * water_decay
            # Air-borne pathway
            air_load = min(1.0, (I / N) * 10.0) * seasonal_factor
            air_decay = max(0.4, 1.0 - 0.3 * (R / N))
            air_prob = 0.003 * air_load * air_decay * climate_factor
            # Surface/fomite pathway
            fomite_load = min(1.0, (I / N) * 6.0)
            fomite_decay = max(0.3, 1.0 - 0.45 * (R / N))
            fomite_prob = 0.0015 * fomite_load * fomite_decay

            total_env_prob = np.clip(water_prob + air_prob + fomite_prob, 0.0, 0.08)
            env_susceptible = max(0, S - new_infections_total)
            if env_susceptible > 0:
                environmental_infections = int(rng.binomial(env_susceptible, total_env_prob))
                new_infections_total = min(new_infections_total + environmental_infections, S)

        # --- Hospital capacity and healthcare system modeling ---
        hospital_capacity_fraction = 0.005  # 0.5% of population is hospital capacity
        hospital_capacity = max(10, int(N * hospital_capacity_fraction))
        hospital_overflow = max(0, I - hospital_capacity)
        hospital_stress = min(1.0, I / max(1, hospital_capacity))

        if hospital_stress > 0.9:
            healthcare_factor = 0.4 + 0.1 * rng.random()
        elif hospital_stress > 0.7:
            healthcare_factor = 0.6 + 0.1 * rng.random()
        elif hospital_stress > 0.5:
            healthcare_factor = 0.8 + 0.1 * rng.random()
        elif hospital_stress < 0.1:
            healthcare_factor = 1.5 + 0.2 * rng.random()
        elif hospital_stress < 0.3:
            healthcare_factor = 1.2 + 0.1 * rng.random()
        else:
            healthcare_factor = 1.0 + 0.05 * rng.random()

        # Socioeconomic modifier on healthcare access
        avg_healthcare_access = sum(socio_fracs[st] * socio_healthcare_access[st] for st in range(socio_tiers))
        healthcare_factor *= avg_healthcare_access

        # --- Multi-stage recovery with severity tiers and comorbidities ---
        new_recoveries_total = 0
        severity_tiers = 8
        tier_fracs = [0.05, 0.12, 0.20, 0.28, 0.18, 0.10, 0.05, 0.02]
        tier_gamma_mults = [0.15, 0.4, 0.85, 1.1, 1.4, 1.8, 0.5, 0.2]
        tier_severity_slow = [4.0, 2.2, 1.2, 1.0, 0.85, 0.7, 3.5, 6.0]
        tier_comorbidity_drag = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 3.0]

        for t in range(severity_tiers):
            tier_I = int(round(I * tier_fracs[t]))
            if tier_I <= 0:
                continue

            tier_gamma = (
                gamma_base
                * tier_gamma_mults[t]
                * healthcare_factor
                / (tier_severity_slow[t] * tier_comorbidity_drag[t])
            )
            tier_gamma = np.clip(tier_gamma, 0.0, 1.0)

            tier_recoveries = int(rng.binomial(tier_I, tier_gamma))
            new_recoveries_total += tier_recoveries

        new_recoveries_total = min(new_recoveries_total, I)

        # --- Disease-induced mortality (removed from all compartments) ---
        mortality_rate_base = 0.0005
        if hospital_stress > 0.8:
            mortality_rate = mortality_rate_base * (1.0 + 2.0 * (hospital_stress - 0.8) / 0.2)
        else:
            mortality_rate = mortality_rate_base * hospital_stress

        mortality_rate = np.clip(mortality_rate, 0.0, 0.01)
        if I > 0:
            deaths_from_I = int(rng.binomial(I, mortality_rate))
            deaths_from_I = min(deaths_from_I, I - new_recoveries_total)
            deaths_from_I = max(0, deaths_from_I)
        else:
            deaths_from_I = 0

        # Adjust population for deaths (we'll handle this in conservation)
        effective_N = N - deaths_from_I

        # --- Near-extinction and importation dynamics ---
        extinction_recoveries = 0
        spontaneous_infections = 0

        if 0 < I <= 8:
            for _ in range(I):
                stochastic_recover_prob = 0.18 + 0.05 * rng.random()
                if rng.random() < stochastic_recover_prob:
                    extinction_recoveries += 1
            extinction_recoveries = min(extinction_recoveries, max(0, I - new_recoveries_total))
            extinction_recoveries = max(0, extinction_recoveries)

            if S > 3 and rng.random() < 0.03:
                spontaneous_infections = rng.integers(1, 3)
                spontaneous_infections = min(int(spontaneous_infections), S)

        if I == 0 and S > 0:
            if N < 500:
                import_prob = 0.025 + (0.015 if S > 50 else 0.0) + (0.008 if R / max(N, 1) < 0.1 else 0.0)
            elif N < 5000:
                import_prob = 0.010 + (0.005 if S > 200 else 0.0)
            else:
                import_prob = 0.003 + (0.002 if S > 1000 else 0.0)

            import_prob *= seasonal_factor * 0.5
            if rng.random() < import_prob:
                import_count = rng.integers(1, max(2, int(np.log(N + 1))))
                spontaneous_infections += min(int(import_count), max(0, S - new_infections_total))

        # --- Immunity waning with antigenic drift consideration ---
        waning_tiers = 5
        waning_tier_fracs = [0.15, 0.25, 0.30, 0.20, 0.10]
        waning_tier_rates = [0.012, 0.007, 0.004, 0.002, 0.0005]
        waning_tier_memory_boost = [2.0, 1.5, 1.0, 0.6, 0.3]
        waning_tier_antigen_sensitivity = [0.9, 0.75, 0.6, 0.45, 0.3]

        recovered_fraction = R / N if N > 0 else 0.0
        if recovered_fraction > 0.7:
            global_memory = 0.6
        elif recovered_fraction > 0.5:
            global_memory = 0.75
        elif recovered_fraction > 0.3:
            global_memory = 0.9
        elif recovered_fraction < 0.03:
            global_memory = 1.8
        elif recovered_fraction < 0.08:
            global_memory = 1.5
        elif recovered_fraction < 0.15:
            global_memory = 1.2
        else:
            global_memory = 1.0

        antigenic_drift = len(strain_fitnesses) * 0.05
        cross_immunity = max(0.3, 1.0 - max(0.0, I / N - 0.08) * 1.8 - antigenic_drift)

        new_waning_total = 0
        for wt in range(waning_tiers):
            wt_R = int(round(R * waning_tier_fracs[wt]))
            if wt_R <= 0:
                continue
            wt_rate = (
                waning_tier_rates[wt]
                * global_memory
                * waning_tier_memory_boost[wt]
                * cross_immunity
                * waning_tier_antigen_sensitivity[wt]
            )
            wt_rate = np.clip(wt_rate, 0.0, 1.0)
            wt_waning = int(rng.binomial(wt_R, wt_rate))
            new_waning_total += wt_waning

        new_waning_total = min(new_waning_total, R)

        # --- Multi-tier reinfection with strain-specific dynamics ---
        newly_reinfected = 0
        if R > 0 and I > 0:
            reinfection_tiers = [
                (0.40, 0.012, 2.8, 0.9),
                (0.30, 0.006, 1.5, 0.7),
                (0.20, 0.003, 0.8, 0.5),
                (0.10, 0.001, 0.3, 0.3),
            ]
            pool_R = max(0, R - new_waning_total)
            allocated = 0
            for rfrac, rrate, rmult, immune_decay in reinfection_tiers:
                tier_pool = int(round(pool_R * rfrac))
                remaining_pool = tier_pool - allocated
                if remaining_pool <= 0:
                    continue
                strain_escape = 1.0 + antigenic_drift * immune_decay
                reinf_prob = rrate * rmult * (I / N) * behavioral_reduction * strain_escape
                reinf_prob = np.clip(reinf_prob, 0.0, 1.0)
                tier_reinf = int(rng.binomial(remaining_pool, reinf_prob))
                newly_reinfected += tier_reinf
                allocated += tier_pool

            newly_reinfected = min(newly_reinfected, pool_R)

        # --- Births/immigration to maintain population dynamics ---
        natural_births = 0
        if N > 100:
            birth_rate = 0.0001 * max(0.5, 1.0 - I / N)
            natural_births = int(rng.binomial(N, birth_rate))

        # --- Consolidate all state transitions ---
        total_S_to_I = min(new_infections_total + spontaneous_infections, S)
        total_I_out = min(new_recoveries_total + extinction_recoveries + deaths_from_I, I)
        actual_recoveries = min(new_recoveries_total + extinction_recoveries, I - deaths_from_I)
        actual_recoveries = max(0, actual_recoveries)

        returning_to_S = new_waning_total
        returning_to_I_from_R = newly_reinfected

        if returning_to_S + returning_to_I_from_R > R:
            total_R_outflow = returning_to_S + returning_to_I_from_R
            scale = R / max(total_R_outflow, 1)
            returning_to_S = int(returning_to_S * scale)
            returning_to_I_from_R = int(returning_to_I_from_R * scale)
            if returning_to_S + returning_to_I_from_R > R:
                returning_to_I_from_R = max(0, R - returning_to_S)

        new_S = S - total_S_to_I + returning_to_S + natural_births
        new_I = I + total_S_to_I - actual_recoveries - deaths_from_I + returning_to_I_from_R
        new_R = R + actual_recoveries - returning_to_S - returning_to_I_from_R

        new_S = max(0, new_S)
        new_I = max(0, new_I)
        new_R = max(0, new_R)

        # --- Population conservation (accounting for deaths) ---
        target_N = N - deaths_from_I + natural_births
        total_new = new_S + new_I + new_R

        if total_new != target_N:
            diff = target_N - total_new
            if diff > 0:
                new_R += diff
            else:
                surplus = abs(diff)
                priority_order = [("R", new_R), ("I", new_I), ("S", new_S)]
                adjusted = {"R": new_R, "I": new_I, "S": new_S}
                for comp_name, comp_val in priority_order:
                    reduction = min(surplus, adjusted[comp_name])
                    adjusted[comp_name] -= reduction
                    surplus -= reduction
                    if surplus <= 0:
                        break
                new_S = adjusted["S"]
                new_I = adjusted["I"]
                new_R = adjusted["R"]

        next_state["S"] = int(max(0, new_S))
        next_state["I"] = int(max(0, new_I))
        next_state["R"] = int(max(0, new_R))

        return next_state


    def evaluate(self, x):
        return 0

