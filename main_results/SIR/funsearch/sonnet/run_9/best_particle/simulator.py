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

        if N <= 0:
            return next_state

        beta = float(np.clip(parameters[0], 0.0, 1.0))
        gamma = float(np.clip(parameters[1], 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # 0. Age-structured compartments (children, adults, elderly)           #
        # ------------------------------------------------------------------ #
        age_fractions = state.get("age_fractions", {"children": 0.20, "adults": 0.60, "elderly": 0.20})
        age_susceptibility = state.get("age_susceptibility", {"children": 0.8, "adults": 1.0, "elderly": 1.6})
        age_mortality = state.get("age_mortality", {"children": 0.0001, "adults": 0.001, "elderly": 0.02})
        age_groups = ["children", "adults", "elderly"]

        # Distribute S, I, R across age groups
        age_S = {g: int(age_fractions.get(g, 0.33) * S) for g in age_groups}
        age_I = {g: int(age_fractions.get(g, 0.33) * I) for g in age_groups}
        age_R = {g: int(age_fractions.get(g, 0.33) * R) for g in age_groups}

        # Correct rounding discrepancies
        age_S["adults"] += S - sum(age_S.values())
        age_I["adults"] += I - sum(age_I.values())
        age_R["adults"] += R - sum(age_R.values())

        # ------------------------------------------------------------------ #
        # 1. Multi-strain pathogen dynamics with immune escape tracking        #
        # ------------------------------------------------------------------ #
        strains = state.get("strains", [{"id": 0, "beta_mult": 1.0, "gamma_mult": 1.0, "prevalence": 1.0, "immune_escape": 0.0, "drug_resistance": 0.0}])
        dominant_strain = max(strains, key=lambda x: x["prevalence"])
        strain_beta_mult = float(np.clip(dominant_strain.get("beta_mult", 1.0), 0.5, 3.0))
        strain_gamma_mult = float(np.clip(dominant_strain.get("gamma_mult", 1.0), 0.3, 1.5))
        dominant_immune_escape = float(np.clip(dominant_strain.get("immune_escape", 0.0), 0.0, 1.0))
        dominant_drug_resistance = float(np.clip(dominant_strain.get("drug_resistance", 0.0), 0.0, 1.0))

        # Evolve strain prevalences via competitive exclusion with drug resistance factor
        updated_strains = []
        for strain in strains:
            base_fitness = strain.get("beta_mult", 1.0) / max(strain.get("gamma_mult", 1.0), 1e-6)
            resistance_bonus = 1.0 + 0.3 * strain.get("drug_resistance", 0.0)
            escape_bonus = 1.0 + 0.2 * strain.get("immune_escape", 0.0)
            fitness = base_fitness * resistance_bonus * escape_bonus
            new_prevalence = strain["prevalence"] * fitness
            # Drift in immune escape and drug resistance
            new_escape = float(np.clip(strain.get("immune_escape", 0.0) + rng.normal(0.001, 0.005), 0.0, 1.0))
            new_resistance = float(np.clip(strain.get("drug_resistance", 0.0) + rng.normal(0.0005, 0.002), 0.0, 1.0))
            updated_strains.append({**strain, "prevalence": new_prevalence, "immune_escape": new_escape, "drug_resistance": new_resistance})

        new_total = sum(s["prevalence"] for s in updated_strains)
        if new_total > 0:
            updated_strains = [{**s, "prevalence": s["prevalence"] / new_total} for s in updated_strains]

        # Prune strains with negligible prevalence (< 0.001) to keep list manageable
        updated_strains = [s for s in updated_strains if s["prevalence"] >= 0.001]
        if len(updated_strains) == 0:
            updated_strains = [{"id": 0, "beta_mult": 1.0, "gamma_mult": 1.0, "prevalence": 1.0, "immune_escape": 0.0, "drug_resistance": 0.0}]
        else:
            norm = sum(s["prevalence"] for s in updated_strains)
            updated_strains = [{**s, "prevalence": s["prevalence"] / norm} for s in updated_strains]

        # Random new strain emergence - affected by population size and current I
        mutation_rate = state.get("mutation_rate", 0.005)
        mutation_threshold = state.get("mutation_threshold_strains", 50)
        emergence_prob = mutation_rate * (1.0 + min(I / max(N, 1) * 5.0, 2.0))
        if rng.random() < emergence_prob and I > mutation_threshold:
            parent_strain = rng.choice(updated_strains, p=[s["prevalence"] for s in updated_strains])
            new_strain = {
                "id": max(s["id"] for s in updated_strains) + 1,
                "beta_mult": float(np.clip(parent_strain["beta_mult"] + rng.normal(0.1, 0.3), 0.5, 3.0)),
                "gamma_mult": float(np.clip(parent_strain["gamma_mult"] + rng.normal(-0.05, 0.2), 0.3, 1.5)),
                "prevalence": 0.01,
                "immune_escape": float(np.clip(parent_strain.get("immune_escape", 0.0) + rng.uniform(0.0, 0.3), 0.0, 1.0)),
                "drug_resistance": float(np.clip(parent_strain.get("drug_resistance", 0.0) + rng.uniform(0.0, 0.1), 0.0, 1.0)),
            }
            updated_strains.append(new_strain)
            total_p = sum(s["prevalence"] for s in updated_strains)
            updated_strains = [{**s, "prevalence": s["prevalence"] / total_p} for s in updated_strains]

        next_state["strains"] = updated_strains

        # ------------------------------------------------------------------ #
        # 2. Climate, seasonality, and environmental modifiers                 #
        # ------------------------------------------------------------------ #
        day = int(state.get("day", 0))
        temperature = state.get("temperature", 15.0)
        humidity = float(np.clip(state.get("humidity", 0.5), 0.0, 1.0))
        season_phase = state.get("season_phase", 0.0)

        # Explicit seasonal forcing using sinusoidal calendar
        seasonal_amplitude = float(np.clip(state.get("seasonal_amplitude", 0.2), 0.0, 0.5))
        season_cycle = float(state.get("season_cycle", 365.0))
        seasonal_factor = 1.0 + seasonal_amplitude * np.sin(2.0 * np.pi * day / season_cycle + season_phase)
        seasonal_factor = float(np.clip(seasonal_factor, 0.5, 1.8))

        # Temperature effect with smooth interpolation
        if temperature < 0.0:
            temp_factor = 1.55
        elif temperature < 5.0:
            temp_factor = 1.40
        elif temperature < 10.0:
            temp_factor = 1.25
        elif temperature < 15.0:
            temp_factor = 1.10
        elif temperature < 20.0:
            temp_factor = 1.00
        elif temperature < 25.0:
            temp_factor = 0.90
        elif temperature < 30.0:
            temp_factor = 0.80
        elif temperature < 35.0:
            temp_factor = 0.72
        else:
            temp_factor = 0.65

        # Humidity effect
        if humidity < 0.15:
            humidity_factor = 1.35
        elif humidity < 0.2:
            humidity_factor = 1.30
        elif humidity < 0.4:
            humidity_factor = 1.15
        elif humidity < 0.6:
            humidity_factor = 1.00
        elif humidity < 0.8:
            humidity_factor = 0.90
        else:
            humidity_factor = 0.80

        # Air quality index effect
        aqi = float(np.clip(state.get("air_quality_index", 50.0), 0.0, 500.0))
        if aqi > 300:
            aqi_factor = 1.25
        elif aqi > 200:
            aqi_factor = 1.15
        elif aqi > 150:
            aqi_factor = 1.08
        elif aqi > 100:
            aqi_factor = 1.03
        else:
            aqi_factor = 1.00

        # Update environmental state
        next_temperature = temperature + rng.normal(0.0, 0.5) + seasonal_amplitude * np.cos(2.0 * np.pi * day / season_cycle) * 0.1
        next_state["temperature"] = float(np.clip(next_temperature, -20.0, 50.0))
        next_state["humidity"] = float(np.clip(humidity + rng.normal(0.0, 0.02), 0.0, 1.0))
        next_state["air_quality_index"] = float(np.clip(aqi + rng.normal(0.0, 5.0), 0.0, 500.0))
        next_state["day"] = day + 1
        next_state["season_phase"] = season_phase

        env_factor = temp_factor * humidity_factor * aqi_factor * seasonal_factor
        effective_beta = float(np.clip(beta * strain_beta_mult * env_factor, 0.0, 1.0))
        effective_gamma = float(np.clip(gamma * strain_gamma_mult, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # 3. Behavioral response with social fatigue and trust dynamics        #
        # ------------------------------------------------------------------ #
        awareness_level = float(np.clip(state.get("awareness_level", 0.0), 0.0, 1.0))
        media_coverage = float(np.clip(state.get("media_coverage", 0.5), 0.0, 1.0))
        social_fatigue = float(np.clip(state.get("social_fatigue", 0.0), 0.0, 1.0))
        institutional_trust = float(np.clip(state.get("institutional_trust", 0.7), 0.0, 1.0))

        # Awareness dynamics: increases with prevalence, decreases with fatigue
        prevalence_rate = I / max(N, 1)
        awareness_increase = prevalence_rate * 0.3 * media_coverage * institutional_trust
        awareness_decay = 0.05 * (1.0 - prevalence_rate) * (1.0 + social_fatigue * 0.5)
        new_awareness = float(np.clip(awareness_level + awareness_increase - awareness_decay, 0.0, 1.0))
        next_state["awareness_level"] = new_awareness

        # Social fatigue grows with sustained interventions and declines slowly
        fatigue_increase = 0.002 * new_awareness
        fatigue_decay = 0.001 * (1.0 - prevalence_rate)
        new_social_fatigue = float(np.clip(social_fatigue + fatigue_increase - fatigue_decay, 0.0, 1.0))
        next_state["social_fatigue"] = new_social_fatigue

        # Trust erodes with prolonged epidemic and bad outcomes
        trust_erosion = 0.001 * prevalence_rate
        trust_recovery = 0.0005 * (1.0 - prevalence_rate)
        new_trust = float(np.clip(institutional_trust - trust_erosion + trust_recovery, 0.1, 1.0))
        next_state["institutional_trust"] = new_trust

        # Effective behavioral reduction modified by fatigue
        behavioral_compliance = (1.0 - social_fatigue * 0.6) * institutional_trust
        behavioral_reduction = 1.0 - 0.5 * new_awareness * behavioral_compliance
        effective_beta = float(np.clip(effective_beta * behavioral_reduction, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # 4. Economic feedback on healthcare resources                         #
        # ------------------------------------------------------------------ #
        gdp_index = float(np.clip(state.get("gdp_index", 1.0), 0.1, 3.0))
        healthcare_budget = float(np.clip(state.get("healthcare_budget", 1.0), 0.1, 5.0))

        # Economic contraction due to epidemic
        economic_impact = prevalence_rate * 0.5 + new_social_fatigue * 0.1
        gdp_change = -economic_impact * 0.01 + rng.normal(0.0, 0.002)
        new_gdp = float(np.clip(gdp_index + gdp_change, 0.1, 3.0))
        next_state["gdp_index"] = new_gdp

        # Healthcare budget adjusts with GDP and government response
        budget_multiplier = float(np.clip(state.get("government_response_strength", 1.0), 0.0, 3.0))
        new_budget = float(np.clip(healthcare_budget + (new_gdp - gdp_index) * 0.5 * budget_multiplier, 0.1, 5.0))
        next_state["healthcare_budget"] = new_budget

        # ------------------------------------------------------------------ #
        # 5. Action handling: interventions with fatigue and trust effects     #
        # ------------------------------------------------------------------ #
        newly_vaccinated = 0
        quarantine_count = 0
        treatment_boost = float(np.clip(state.get("treatment_boost", 0.0), 0.0, 0.5))
        reinfection_shield = float(np.clip(state.get("reinfection_shield", 0.0), 0.0, 0.8))

        if action is not None:
            if action > 0:
                # Tiered vaccination with trust-based uptake and immune escape awareness
                vax_target = min(action, S)
                if vax_target > 0:
                    priority_groups = state.get("priority_groups", [("elderly", 0.20, 0.95), ("essential", 0.25, 0.88), ("children", 0.15, 0.75), ("general", 0.40, 0.80)])
                    total_vax = 0
                    remaining_vax_target = vax_target
                    for group_name, group_frac, base_efficacy in priority_groups:
                        if remaining_vax_target <= 0:
                            break
                        # Efficacy reduced by immune escape
                        adjusted_efficacy = float(np.clip(base_efficacy * (1.0 - dominant_immune_escape * 0.5), 0.1, 1.0))
                        # Uptake limited by trust and fatigue
                        uptake_rate = float(np.clip(new_trust * (1.0 - new_social_fatigue * 0.4), 0.1, 1.0))
                        group_S = int(min(group_frac * S, remaining_vax_target))
                        if group_S > 0:
                            effective_group_S = int(group_S * uptake_rate)
                            if effective_group_S > 0:
                                vaxed = int(np.clip(rng.binomial(effective_group_S, adjusted_efficacy), 0, effective_group_S))
                                total_vax += vaxed
                            remaining_vax_target -= group_S
                    newly_vaccinated = min(total_vax, S)
                    S = max(0, S - newly_vaccinated)
                    R = R + newly_vaccinated
                    # Trust improves slightly with successful vaccination
                    next_state["institutional_trust"] = float(np.clip(new_trust + 0.005 * newly_vaccinated / max(N, 1), 0.1, 1.0))

            elif action == -1:
                # Enhanced quarantine with contact tracing and digital surveillance
                contact_tracing_efficiency = float(np.clip(state.get("contact_tracing_efficiency", 0.5), 0.0, 1.0))
                digital_surveillance = float(np.clip(state.get("digital_surveillance", 0.3), 0.0, 1.0))
                base_quarantine_rate = float(np.clip(state.get("quarantine_effectiveness", 0.60), 0.0, 1.0))
                # Digital tools improve tracing but may reduce trust
                combined_rate = float(np.clip(base_quarantine_rate + contact_tracing_efficiency * 0.3 + digital_surveillance * 0.15, 0.0, 1.0))
                # Fatigue reduces compliance
                combined_rate = float(np.clip(combined_rate * (1.0 - new_social_fatigue * 0.3), 0.0, 1.0))
                if I > 0:
                    quarantine_count = int(np.clip(rng.binomial(I, combined_rate), 0, I))
                    I = max(0, I - quarantine_count)
                    next_state["quarantined"] = state.get("quarantined", 0) + quarantine_count
                # Trust impact from surveillance
                trust_delta = -0.002 * digital_surveillance + 0.001 * combined_rate
                next_state["institutional_trust"] = float(np.clip(new_trust + trust_delta, 0.1, 1.0))

            elif action == -2:
                # Tiered lockdown with economic cost tracking
                lockdown_tier = int(state.get("lockdown_tier", 2))
                if lockdown_tier == 1:
                    reduction = rng.uniform(0.20, 0.40)
                    economic_penalty = 0.02
                elif lockdown_tier == 2:
                    reduction = rng.uniform(0.10, 0.25)
                    economic_penalty = 0.01
                elif lockdown_tier == 3:
                    reduction = rng.uniform(0.05, 0.15)
                    economic_penalty = 0.005
                else:
                    reduction = rng.uniform(0.02, 0.10)
                    economic_penalty = 0.002
                # Fatigue reduces lockdown effectiveness
                fatigue_penalty = new_social_fatigue * 0.4
                effective_reduction = float(np.clip(reduction * (1.0 - fatigue_penalty), 0.01, 1.0))
                effective_beta = float(np.clip(effective_beta * effective_reduction, 0.0, 1.0))
                next_state["awareness_level"] = float(np.clip(new_awareness + 0.10 * (1.0 - new_social_fatigue), 0.0, 1.0))
                next_state["social_fatigue"] = float(np.clip(new_social_fatigue + 0.02, 0.0, 1.0))
                next_state["gdp_index"] = float(np.clip(new_gdp - economic_penalty, 0.1, 3.0))

            elif action == -3:
                # Medical treatment with drug resistance consideration
                treatment_investment = float(np.clip(state.get("treatment_investment", 0.1), 0.0, 1.0))
                # Drug resistance reduces treatment effectiveness
                resistance_penalty = dominant_drug_resistance * 0.5
                effective_investment = treatment_investment * (1.0 - resistance_penalty)
                additional_boost = effective_investment * rng.uniform(0.05, 0.20)
                # Budget-scaled treatment boost
                budget_scale = float(np.clip(new_budget / 1.0, 0.5, 2.0))
                treatment_boost = float(np.clip(treatment_boost + additional_boost * budget_scale, 0.0, 0.5))
                next_state["treatment_boost"] = treatment_boost
                # Boost reinfection shield
                reinfection_shield = float(np.clip(reinfection_shield + 0.01 * effective_investment, 0.0, 0.8))
                next_state["reinfection_shield"] = reinfection_shield

            elif action == -4:
                # Border control with smuggling/leakage
                border_control_eff = float(np.clip(state.get("border_control_effectiveness", 0.7), 0.0, 1.0))
                smuggling_rate = float(np.clip(state.get("smuggling_rate", 0.05), 0.0, 0.3))
                external_seed_rate = float(np.clip(state.get("external_seed_rate", 0.001), 0.0, 0.05))
                net_reduction = border_control_eff * (1.0 - smuggling_rate)
                next_state["external_seed_rate"] = float(np.clip(external_seed_rate * (1.0 - net_reduction), 0.0, 0.05))
                # Economic cost of border closure
                next_state["gdp_index"] = float(np.clip(new_gdp - 0.003 * border_control_eff, 0.1, 3.0))

            elif action == -5:
                # Mass testing and isolation: identify and isolate asymptomatic cases
                testing_capacity = float(np.clip(state.get("testing_capacity", 0.1), 0.0, 1.0))
                test_sensitivity = float(np.clip(state.get("test_sensitivity", 0.85), 0.0, 1.0))
                asymptomatic_fraction_local = float(np.clip(state.get("asymptomatic_fraction", 0.30), 0.0, 0.8))
                if I > 0:
                    tested = int(np.clip(rng.binomial(I, testing_capacity), 0, I))
                    detected_asymptomatic = int(np.clip(rng.binomial(int(asymptomatic_fraction_local * tested), test_sensitivity), 0, tested))
                    isolated = int(np.clip(rng.binomial(detected_asymptomatic, new_trust), 0, detected_asymptomatic))
                    I = max(0, I - isolated)
                    next_state["quarantined"] = state.get("quarantined", 0) + isolated
                    next_state["testing_positive_rate"] = isolated / max(tested, 1)

        # ------------------------------------------------------------------ #
        # 6. Super-spreader event modeling                                     #
        # ------------------------------------------------------------------ #
        super_spreader_prob = float(np.clip(state.get("super_spreader_prob", 0.005), 0.0, 0.1))
        super_spreader_multiplier = float(np.clip(state.get("super_spreader_multiplier", 5.0), 2.0, 20.0))
        event_beta_boost = 1.0

        if rng.random() < super_spreader_prob * (1.0 - new_awareness * 0.5):
            event_size = int(rng.uniform(10, 500))
            event_beta_boost = float(np.clip(super_spreader_multiplier * (event_size / max(N, 1)) * 100.0 + 1.0, 1.0, 3.0))
            next_state["last_super_spreader_day"] = day
            next_state["last_super_spreader_size"] = event_size

        # ------------------------------------------------------------------ #
        # 7. Spatial heterogeneity: multi-patch metapopulation                 #
        # ------------------------------------------------------------------ #
        urban_fraction = float(np.clip(state.get("urban_fraction", 0.60), 0.0, 1.0))
        rural_fraction = 1.0 - urban_fraction
        urban_density_factor = float(np.clip(state.get("urban_density_factor", 2.0), 1.0, 5.0))
        rural_density_factor = float(np.clip(state.get("rural_density_factor", 0.5), 0.1, 1.0))
        suburban_fraction = float(np.clip(state.get("suburban_fraction", 0.0), 0.0, min(urban_fraction, rural_fraction)))

        # Adjust fractions for three-patch system if suburban exists
        if suburban_fraction > 0:
            adjusted_urban = urban_fraction - suburban_fraction * 0.5
            adjusted_rural = rural_fraction - suburban_fraction * 0.5
        else:
            adjusted_urban = urban_fraction
            adjusted_rural = rural_fraction

        patches = [
            {"name": "urban", "frac": adjusted_urban, "density": urban_density_factor},
            {"name": "rural", "frac": adjusted_rural, "density": rural_density_factor},
        ]
        if suburban_fraction > 0:
            suburban_density = float(np.clip(state.get("suburban_density_factor", 1.2), 0.8, 2.5))
            patches.append({"name": "suburban", "frac": suburban_fraction, "density": suburban_density})

        patch_S = {p["name"]: int(p["frac"] * S) for p in patches}
        patch_I = {p["name"]: int(p["frac"] * I) for p in patches}

        # Correct rounding
        dominant_patch = "urban"
        patch_S[dominant_patch] += S - sum(patch_S.values())
        patch_I[dominant_patch] += I - sum(patch_I.values())

        patch_new_infections = {p["name"]: 0 for p in patches}

        for patch in patches:
            pname = patch["name"]
            p_S = patch_S[pname]
            p_I = patch_I[pname]
            p_density = patch["density"]

            if p_I <= 0 or p_S <= 0 or N <= 0:
                continue

            p_beta = float(np.clip(effective_beta * p_density * event_beta_boost, 0.0, 1.0))
            foi = float(np.clip(p_beta * p_I / N, 0.0, 1.0))

            # Urban areas: multi-round mixing
            if pname == "urban":
                remaining = p_S
                for _ in range(3):
                    if remaining <= 0:
                        break
                    sub_foi = float(np.clip(foi / 3.0, 0.0, 1.0))
                    new_inf = int(np.clip(rng.binomial(remaining, sub_foi), 0, remaining))
                    patch_new_infections[pname] += new_inf
                    remaining = max(0, remaining - new_inf)
            else:
                new_inf = int(np.clip(rng.binomial(p_S, foi), 0, p_S))
                patch_new_infections[pname] += new_inf

        # Cross-patch transmission
        cross_exposure_rate = float(np.clip(state.get("cross_exposure_rate", 0.05), 0.0, 0.3))
        for src_patch in patches:
            src = src_patch["name"]
            for dst_patch in patches:
                dst = dst_patch["name"]
                if src == dst:
                    continue
                src_I = patch_I[src]
                dst_S = patch_S[dst]
                if src_I <= 0 or dst_S <= 0 or N <= 0:
                    continue
                already_infected = patch_new_infections[dst]
                remaining_dst_S = max(0, dst_S - already_infected)
                if remaining_dst_S <= 0:
                    continue
                cross_foi = float(np.clip(effective_beta * cross_exposure_rate * src_I / N, 0.0, 1.0))
                cross_inf = int(np.clip(rng.binomial(remaining_dst_S, cross_foi), 0, remaining_dst_S))
                patch_new_infections[dst] += cross_inf

        total_new_infections = sum(patch_new_infections.values())
        total_new_infections = int(np.clip(total_new_infections, 0, S))
        S = max(0, S - total_new_infections)

        # ------------------------------------------------------------------ #
        # 8. External seeding with variant importation                         #
        # ------------------------------------------------------------------ #
        external_seed_rate = float(np.clip(state.get("external_seed_rate", 0.0005), 0.0, 0.05))
        if external_seed_rate > 0.0 and S > 0:
            external_seeds = int(rng.poisson(external_seed_rate * N))
            external_seeds = min(external_seeds, S)
            S = max(0, S - external_seeds)
            total_new_infections += external_seeds
            # Small chance of importing a new variant with external seed
            if external_seeds > 0 and rng.random() < 0.01:
                imported_strain = {
                    "id": max(s["id"] for s in updated_strains) + 1,
                    "beta_mult": float(np.clip(rng.normal(1.4, 0.4), 0.5, 3.0)),
                    "gamma_mult": float(np.clip(rng.normal(0.85, 0.2), 0.3, 1.5)),
                    "prevalence": 0.005,
                    "immune_escape": float(np.clip(rng.uniform(0.1, 0.6), 0.0, 1.0)),
                    "drug_resistance": float(np.clip(rng.uniform(0.0, 0.2), 0.0, 1.0)),
                }
                updated_strains.append(imported_strain)
                total_p = sum(s["prevalence"] for s in updated_strains)
                updated_strains = [{**s, "prevalence": s["prevalence"] / total_p} for s in updated_strains]
                next_state["strains"] = updated_strains

        # ------------------------------------------------------------------ #
        # 9. Differentiated recovery: ICU, mild, asymptomatic, severe         #
        # ------------------------------------------------------------------ #
        asymptomatic_fraction = float(np.clip(state.get("asymptomatic_fraction", 0.30), 0.0, 0.8))
        icu_fraction = float(np.clip(state.get("icu_fraction", 0.05), 0.0, 0.3))
        severe_fraction = float(np.clip(state.get("severe_fraction", 0.10), 0.0, 0.4))
        mild_fraction = max(0.0, 1.0 - asymptomatic_fraction - icu_fraction - severe_fraction)

        I_asymptomatic = int(asymptomatic_fraction * I)
        I_icu = int(icu_fraction * I)
        I_severe = int(severe_fraction * I)
        I_mild = max(0, I - I_asymptomatic - I_icu - I_severe)

        # Drug resistance reduces treatment effectiveness for severe and ICU
        treatment_effectiveness = 1.0 - dominant_drug_resistance * 0.4
        gamma_asymptomatic = float(np.clip(effective_gamma * 1.5, 0.0, 1.0))
        gamma_mild = float(np.clip(effective_gamma * treatment_effectiveness + treatment_boost, 0.0, 1.0))
        gamma_severe = float(np.clip(effective_gamma * 0.5 * treatment_effectiveness + treatment_boost * 0.5, 0.0, 1.0))
        gamma_icu = float(np.clip(effective_gamma * 0.25 * treatment_effectiveness, 0.0, 1.0))

        rec_asymptomatic = int(np.clip(rng.binomial(I_asymptomatic, gamma_asymptomatic), 0, I_asymptomatic)) if I_asymptomatic > 0 else 0
        rec_mild = int(np.clip(rng.binomial(I_mild, gamma_mild), 0, I_mild)) if I_mild > 0 else 0
        rec_severe = int(np.clip(rng.binomial(I_severe, gamma_severe), 0, I_severe)) if I_severe > 0 else 0
        rec_icu = int(np.clip(rng.binomial(I_icu, gamma_icu), 0, I_icu)) if I_icu > 0 else 0
        new_recoveries = rec_asymptomatic + rec_mild + rec_severe + rec_icu

        # ICU deaths with capacity overflow and budget scaling
        icu_capacity_base = state.get("icu_capacity", max(int(0.001 * N), 1))
        icu_capacity = int(np.clip(icu_capacity_base * new_budget, 1, icu_capacity_base * 3))
        icu_overflow = max(0, I_icu - icu_capacity)
        base_icu_death_rate = 0.02
        if icu_overflow > 0:
            overflow_factor = icu_overflow / max(I_icu, 1)
            icu_death_rate = float(np.clip(base_icu_death_rate + 0.20 * overflow_factor, 0.0, 0.7))
        else:
            icu_death_rate = base_icu_death_rate
        icu_deaths = int(np.clip(rng.binomial(I_icu, icu_death_rate), 0, I_icu)) if I_icu > 0 else 0

        # Severe deaths
        severe_death_rate = float(np.clip(state.get("severe_death_rate", 0.005), 0.0, 0.2))
        severe_deaths = int(np.clip(rng.binomial(I_severe, severe_death_rate), 0, I_severe)) if I_severe > 0 else 0

        # ------------------------------------------------------------------ #
        # 10. Age-stratified mortality                                         #
        # ------------------------------------------------------------------ #
        age_deaths_I = {}
        for g in age_groups:
            g_I = age_I[g]
            g_mort = age_mortality.get(g, 0.001)
            age_deaths_I[g] = int(np.clip(rng.binomial(g_I, g_mort), 0, g_I)) if g_I > 0 else 0

        age_excess_deaths = sum(age_deaths_I.values())

        # ------------------------------------------------------------------ #
        # 11. Waning immunity with reinfection tracking                        #
        # ------------------------------------------------------------------ #
        waning_rate = float(np.clip(state.get("waning_rate", 0.002), 0.0, 1.0))
        booster_coverage = float(np.clip(state.get("booster_coverage", 0.0), 0.0, 1.0))

        # Boosters reduce waning
        effective_waning = waning_rate * (1.0 - 0.7 * booster_coverage)
        effective_waning = float(np.clip(effective_waning, 0.0, 1.0))

        # Immune escape from dominant strain
        if dominant_strain.get("immune_escape", 0.0) > 0.2:
            escape_acceleration = float(np.clip(dominant_strain["immune_escape"] * 1.5, 0.0, 2.0))
            effective_waning = float(np.clip(effective_waning * (1.0 + escape_acceleration), 0.0, 1.0))
        elif dominant_strain.get("beta_mult", 1.0) > 1.5:
            immune_escape_factor = float(np.clip((dominant_strain["beta_mult"] - 1.0) * 0.5, 0.0, 0.8))
            effective_waning = float(np.clip(effective_waning * (1.0 + immune_escape_factor), 0.0, 1.0))

        newly_susceptible = 0
        if effective_waning > 0.0 and R > 0:
            # Track partial immunity: some waned-immune become partially protected
            waned_total = int(np.clip(rng.binomial(R, effective_waning), 0, R))
            # Reinfection shield reduces full susceptibility
            newly_susceptible = int(np.clip(waned_total * (1.0 - reinfection_shield), 0, waned_total))
            next_state["reinfection_shield"] = float(np.clip(reinfection_shield - 0.001, 0.0, 0.8))

        # ------------------------------------------------------------------ #
        # 12. Demographic dynamics with age structure                          #
        # ------------------------------------------------------------------ #
        birth_rate = float(np.clip(state.get("birth_rate", 0.00005), 0.0, 0.01))
        base_death_rate = float(np.clip(state.get("death_rate", 0.00003), 0.0, 0.01))

        # Birth rate slightly reduced by epidemic severity
        effective_birth_rate = birth_rate * (1.0 - prevalence_rate * 0.2)
        new_births = int(rng.poisson(effective_birth_rate * N)) if effective_birth_rate > 0.0 else 0

        deaths_S = int(np.clip(rng.binomial(S, base_death_rate), 0, S)) if S > 0 and base_death_rate > 0 else 0
        deaths_R = int(np.clip(rng.binomial(R, base_death_rate), 0, R)) if R > 0 and base_death_rate > 0 else 0
        deaths_I_natural = int(np.clip(rng.binomial(max(0, I - I_icu - I_severe), base_death_rate * 1.5), 0, max(0, I - I_icu - I_severe))) if I > 0 else 0
        deaths_I = deaths_I_natural + icu_deaths + severe_deaths
        # Age-excess deaths from I (add any that exceed natural deaths)
        extra_age_deaths = max(0, age_excess_deaths - deaths_I_natural)
        deaths_I = min(deaths_I + extra_age_deaths, I)

        # ------------------------------------------------------------------ #
        # 13. Healthcare system feedback on gamma with budget scaling          #
        # ------------------------------------------------------------------ #
        hospital_capacity_base = state.get("hospital_capacity", max(int(0.005 * N), 1))
        hospital_capacity = int(np.clip(hospital_capacity_base * new_budget, 1, hospital_capacity_base * 3))
        occupancy = I / max(hospital_capacity, 1)

        if occupancy > 4.0:
            capacity_modifier = 0.15
        elif occupancy > 3.0:
            capacity_modifier = 0.25
        elif occupancy > 2.0:
            capacity_modifier = 0.40
        elif occupancy > 1.5:
            capacity_modifier = 0.55
        elif occupancy > 1.0:
            capacity_modifier = 0.70
        elif occupancy > 0.75:
            capacity_modifier = 0.80
        elif occupancy > 0.5:
            capacity_modifier = 0.90
        else:
            capacity_modifier = 1.00

        # Budget-scaled modifier
        capacity_modifier = float(np.clip(capacity_modifier * min(new_budget, 1.5), 0.0, 1.0))
        effective_gamma = float(np.clip(effective_gamma * capacity_modifier, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # 14. Assemble new compartments                                        #
        # ------------------------------------------------------------------ #
        S_new = S + newly_susceptible + new_births - deaths_S
        I_new = I + total_new_infections - new_recoveries - deaths_I
        R_new = R + new_recoveries + newly_vaccinated - newly_susceptible - deaths_R

        S_new = max(0, S_new)
        I_new = max(0, I_new)
        R_new = max(0, R_new)

        # ------------------------------------------------------------------ #
        # 15. Policy learning with weighted rolling metrics                    #
        # ------------------------------------------------------------------ #
        intervention_history = list(state.get("intervention_history", []))
        if action is not None:
            intervention_history.append({
                "day": day,
                "action": action,
                "I_before": I,
                "I_after": I_new,
                "gdp": gdp_index,
                "trust": institutional_trust,
                "fatigue": social_fatigue,
            })
        if len(intervention_history) > 60:
            intervention_history = intervention_history[-60:]
        next_state["intervention_history"] = intervention_history

        # Compute weighted rolling effectiveness (recent entries weighted more)
        if len(intervention_history) >= 7:
            recent = intervention_history[-14:]
            weights = np.array([float(i + 1) for i in range(len(recent))])
            weights = weights / weights.sum()
            i_changes = np.array([entry["I_after"] - entry["I_before"] for entry in recent], dtype=float)
            weighted_avg = float(np.dot(weights, i_changes))
            next_state["rolling_avg_I_change"] = weighted_avg

            # Track action-specific effectiveness
            action_effects = {}
            for entry in recent:
                act = str(entry["action"])
                delta = entry["I_after"] - entry["I_before"]
                if act not in action_effects:
                    action_effects[act] = []
                action_effects[act].append(delta)
            next_state["action_effectiveness"] = {k: float(np.mean(v)) for k, v in action_effects.items()}
        else:
            next_state["rolling_avg_I_change"] = state.get("rolling_avg_I_change", 0.0)
            next_state["action_effectiveness"] = state.get("action_effectiveness", {})

        # ------------------------------------------------------------------ #
        # 16. Population-conservation correction                               #
        # ------------------------------------------------------------------ #
        total_deaths = deaths_S + deaths_I + deaths_R
        quarantined_prev = state.get("quarantined", 0)
        quarantined_curr = next_state.get("quarantined", quarantined_prev)
        quarantined_delta = quarantined_curr - quarantined_prev

        expected_total = N + new_births - total_deaths - quarantined_delta
        actual_total = S_new + I_new + R_new
        discrepancy = expected_total - actual_total

        if discrepancy > 0:
            S_new += discrepancy
        elif discrepancy < 0:
            abs_disc = abs(discrepancy)
            if R_new >= abs_disc:
                R_new = max(0, R_new - abs_disc)
            elif R_new + S_new >= abs_disc:
                remainder = abs_disc - R_new
                R_new = 0
                S_new = max(0, S_new - remainder)
            else:
                saved_R = R_new
                saved_S = S_new
                R_new = 0
                S_new = 0
                I_new = max(0, I_new - (abs_disc - saved_R - saved_S))

        # ------------------------------------------------------------------ #
        # 17. Cumulative statistics with additional metrics                    #
        # ------------------------------------------------------------------ #
        next_state["cumulative_infections"] = state.get("cumulative_infections", 0) + total_new_infections
        next_state["cumulative_deaths"] = state.get("cumulative_deaths", 0) + total_deaths
        next_state["cumulative_vaccinations"] = state.get("cumulative_vaccinations", 0) + newly_vaccinated
        next_state["cumulative_recoveries"] = state.get("cumulative_recoveries", 0) + new_recoveries
        next_state["cumulative_icu_days"] = state.get("cumulative_icu_days", 0) + I_icu
        next_state["cumulative_severe_days"] = state.get("cumulative_severe_days", 0) + I_severe
        next_state["cumulative_quarantine_days"] = state.get("cumulative_quarantine_days", 0) + quarantine_count
        next_state["peak_infections"] = max(int(state.get("peak_infections", 0)), I)
        next_state["peak_day"] = state.get("peak_day", day) if I < int(state.get("peak_infections", 0)) else day

        # Effective reproduction number estimate
        r_eff = float(effective_beta * S_new / max(N, 1) / max(effective_gamma, 1e-6))
        next_state["r_effective"] = float(np.clip(r_eff, 0.0, 20.0))

        # Herd immunity threshold tracking
        herd_immunity_threshold = 1.0 - 1.0 / max(r_eff, 1.0)
        next_state["herd_immunity_threshold"] = float(np.clip(herd_immunity_threshold, 0.0, 1.0))
        immune_fraction = R_new / max(S_new + I_new + R_new, 1)
        next_state["immune_fraction"] = float(immune_fraction)
        next_state["herd_immunity_achieved"] = bool(immune_fraction >= herd_immunity_threshold)

        # ------------------------------------------------------------------ #
        # 18. Write final state                                                #
        # ------------------------------------------------------------------ #
        next_state["S"] = int(S_new)
        next_state["I"] = int(I_new)
        next_state["R"] = int(R_new)

        return next_state


    def evaluate(self, x):
        return 0

