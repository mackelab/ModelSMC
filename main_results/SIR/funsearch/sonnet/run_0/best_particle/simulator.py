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

        S = state["S"]
        I = state["I"]
        R = state["R"]
        N = S + I + R

        if N == 0:
            return next_state

        beta, gamma = parameters[0], parameters[1]

        # --- Age-stratified population structure ---
        age_strata = {
            "child":   {"fraction": 0.18, "susceptibility": 0.85, "severity": 0.10, "compliance": 0.60, "contact_rate": 1.4},
            "young":   {"fraction": 0.25, "susceptibility": 1.00, "severity": 0.20, "compliance": 0.65, "contact_rate": 1.6},
            "adult":   {"fraction": 0.35, "susceptibility": 0.95, "severity": 0.40, "compliance": 0.80, "contact_rate": 1.2},
            "elderly": {"fraction": 0.15, "susceptibility": 1.30, "severity": 0.85, "compliance": 0.90, "contact_rate": 0.7},
            "senior":  {"fraction": 0.07, "susceptibility": 1.60, "severity": 0.98, "compliance": 0.92, "contact_rate": 0.5},
        }

        # --- Socioeconomic stratification ---
        socio_strata = {
            "low":    {"fraction": 0.30, "crowding_factor": 2.0, "healthcare_access": 0.4, "compliance": 0.5,  "nutrition_index": 0.60},
            "mid":    {"fraction": 0.50, "crowding_factor": 1.0, "healthcare_access": 0.8, "compliance": 0.75, "nutrition_index": 0.85},
            "high":   {"fraction": 0.20, "crowding_factor": 0.5, "healthcare_access": 1.2, "compliance": 0.95, "nutrition_index": 1.10},
        }

        # --- Co-morbidity burden ---
        comorbidity_profiles = {
            "none":        {"fraction": 0.55, "mortality_multiplier": 1.0,  "recovery_penalty": 0.0},
            "mild":        {"fraction": 0.25, "mortality_multiplier": 2.5,  "recovery_penalty": 0.10},
            "moderate":    {"fraction": 0.13, "mortality_multiplier": 5.0,  "recovery_penalty": 0.25},
            "severe":      {"fraction": 0.07, "mortality_multiplier": 12.0, "recovery_penalty": 0.50},
        }

        # --- Behavioral fatigue and compliance tracking ---
        time_in_intervention = 0
        compliance_fatigue_factor = 1.0
        if action is not None and action > 0:
            fatigue_base = rng.uniform(0.85, 1.0)
            compliance_fatigue_factor = fatigue_base

        # --- Prevalence and incidence estimation ---
        prevalence = I / N if N > 0 else 0.0
        behavioral_adaptation = 1.0

        if prevalence > 0.35:
            behavioral_adaptation = 0.45
        elif prevalence > 0.25:
            behavioral_adaptation = 0.55
        elif prevalence > 0.18:
            behavioral_adaptation = 0.65
        elif prevalence > 0.12:
            behavioral_adaptation = 0.75
        elif prevalence > 0.07:
            behavioral_adaptation = 0.84
        elif prevalence > 0.03:
            behavioral_adaptation = 0.92
        else:
            behavioral_adaptation = 1.0

        # --- Social trust and information quality ---
        misinformation_factor = rng.uniform(0.85, 1.15)
        trust_index = rng.uniform(0.60, 1.0)
        behavioral_adaptation = behavioral_adaptation * trust_index * misinformation_factor
        behavioral_adaptation = min(max(behavioral_adaptation, 0.30), 1.20)

        beta = beta * behavioral_adaptation

        # --- Multi-channel media and social network influence ---
        media_influence = 1.0
        if I > 0:
            media_saturation = min(I / (0.04 * N), 1.0)
            social_media_amplification = rng.uniform(0.9, 1.2)
            traditional_media_weight = 0.6
            social_media_weight = 0.4
            combined_media = (traditional_media_weight * (1.0 - 0.35 * media_saturation) +
                              social_media_weight * (1.0 - 0.20 * media_saturation * social_media_amplification))
            media_influence = combined_media
            beta = beta * media_influence

        # --- Action processing with nuanced effects ---
        quarantine_efficiency = 0.0
        testing_boost = 1.0
        healthcare_boost = 1.0
        contact_tracing_efficiency = 0.0
        vaccination_rate = 0.0
        mask_mandate_factor = 1.0
        school_closure_factor = 1.0
        travel_restriction_factor = 1.0

        if action is not None:
            if action == 0:
                pass
            elif action == 1:
                beta = beta * 0.82 * compliance_fatigue_factor
                quarantine_efficiency = 0.07
                contact_tracing_efficiency = 0.04
                mask_mandate_factor = 0.90
            elif action == 2:
                beta = beta * 0.62 * compliance_fatigue_factor
                quarantine_efficiency = 0.16
                testing_boost = 1.18
                contact_tracing_efficiency = 0.14
                mask_mandate_factor = 0.80
                school_closure_factor = 0.85
            elif action == 3:
                beta = beta * 0.38 * compliance_fatigue_factor
                gamma = gamma * 1.18
                quarantine_efficiency = 0.32
                testing_boost = 1.28
                healthcare_boost = 1.18
                contact_tracing_efficiency = 0.28
                vaccination_rate = 0.0022
                mask_mandate_factor = 0.70
                school_closure_factor = 0.70
                travel_restriction_factor = 0.80
            elif action == 4:
                beta = beta * 0.14 * compliance_fatigue_factor
                gamma = gamma * 1.38
                quarantine_efficiency = 0.52
                testing_boost = 1.48
                healthcare_boost = 1.38
                contact_tracing_efficiency = 0.48
                vaccination_rate = 0.0052
                mask_mandate_factor = 0.55
                school_closure_factor = 0.50
                travel_restriction_factor = 0.40
            elif action == 5:
                beta = beta * 0.47 * compliance_fatigue_factor
                gamma = gamma * 1.12
                healthcare_boost = 1.08
                testing_boost = 1.12
                vaccination_rate = 0.0032
                mask_mandate_factor = 0.75
            elif action == 6:
                beta = beta * 0.25 * compliance_fatigue_factor
                gamma = gamma * 1.30
                quarantine_efficiency = 0.45
                testing_boost = 1.40
                healthcare_boost = 1.30
                contact_tracing_efficiency = 0.40
                vaccination_rate = 0.0040
                mask_mandate_factor = 0.60
                school_closure_factor = 0.60
                travel_restriction_factor = 0.60
            else:
                beta = beta * 0.88 * compliance_fatigue_factor
                quarantine_efficiency = 0.02

        beta = beta * mask_mandate_factor * school_closure_factor * travel_restriction_factor

        # --- Healthcare worker protection and depletion ---
        hw_fraction = 0.03
        hw_count = int(N * hw_fraction)
        hw_infection_risk = 0.0
        hw_effectiveness_multiplier = 1.0

        if I > 0 and hw_count > 0:
            hw_exposure_prob = min(prevalence * 3.5, 0.25)
            hw_ppe_compliance = rng.uniform(0.70, 0.98)
            hw_infected = rng.binomial(int(hw_count * min(S / N, 1.0)), hw_exposure_prob * (1.0 - hw_ppe_compliance * 0.85))
            hw_depletion_rate = hw_infected / max(hw_count, 1)
            hw_effectiveness_multiplier = max(1.0 - hw_depletion_rate * 2.5, 0.50)
            healthcare_boost = healthcare_boost * hw_effectiveness_multiplier

        # --- Multi-dose vaccination dynamics ---
        vaccinated_this_step = 0
        booster_effectiveness_modifier = 1.0

        if vaccination_rate > 0 and S > 0:
            vaccine_efficacy_base = 0.90
            vaccine_hesitancy_base = rng.uniform(0.60, 0.95)
            supply_constraint = rng.uniform(0.70, 1.0)
            cold_chain_factor = rng.uniform(0.88, 1.0)

            # Age-stratified vaccination prioritization
            priority_boost = 1.0
            if prevalence > 0.10:
                priority_boost = 1.25
            elif prevalence > 0.05:
                priority_boost = 1.10

            effective_vaccination_rate = vaccination_rate * vaccine_hesitancy_base * supply_constraint * cold_chain_factor * priority_boost
            effective_vaccination_rate = min(effective_vaccination_rate, 1.0)

            first_dose = rng.binomial(S, effective_vaccination_rate)
            booster_fraction = rng.uniform(0.01, 0.08)
            booster_dose_count = rng.binomial(min(int(R * booster_fraction), S), min(effective_vaccination_rate * 0.5, 1.0))
            vaccinated_this_step = min(first_dose, S)
            booster_effectiveness_modifier = min(1.0 + booster_dose_count / max(N, 1) * 10, 1.25)

        # --- Multi-strain disease dynamics with cross-immunity ---
        num_strains = 4
        strain_betas = [beta]
        strain_gammas = [gamma]
        strain_weights = [1.0]
        strain_cross_immunity = [[1.0]]

        strain_emergence_prob = min(0.003 * prevalence * 100, 0.10)
        mutation_pressure = min(prevalence * 2.0, 1.0)

        for strain_idx in range(1, num_strains):
            if rng.random() < strain_emergence_prob * (1.0 + mutation_pressure * 0.5):
                strain_type = rng.integers(0, 5)
                cross_immunity_row = []

                if strain_type == 0:
                    sb = beta * rng.uniform(1.4, 2.2)
                    sg = gamma * rng.uniform(0.80, 1.05)
                    sw = rng.uniform(0.10, 0.45)
                    cross_vals = [rng.uniform(0.3, 0.7) for _ in range(strain_idx)]
                elif strain_type == 1:
                    sb = beta * rng.uniform(0.65, 0.92)
                    sg = gamma * rng.uniform(0.45, 0.72)
                    sw = rng.uniform(0.05, 0.22)
                    cross_vals = [rng.uniform(0.5, 0.9) for _ in range(strain_idx)]
                elif strain_type == 2:
                    sb = beta * rng.uniform(1.15, 1.75)
                    sg = gamma * rng.uniform(0.88, 1.12)
                    sw = rng.uniform(0.06, 0.32)
                    cross_vals = [rng.uniform(0.2, 0.6) for _ in range(strain_idx)]
                elif strain_type == 3:
                    sb = beta * rng.uniform(0.88, 1.25)
                    sg = gamma * rng.uniform(1.08, 1.30)
                    sw = rng.uniform(0.08, 0.22)
                    cross_vals = [rng.uniform(0.4, 0.8) for _ in range(strain_idx)]
                else:
                    sb = beta * rng.uniform(1.8, 3.0)
                    sg = gamma * rng.uniform(0.70, 0.95)
                    sw = rng.uniform(0.05, 0.20)
                    cross_vals = [rng.uniform(0.1, 0.4) for _ in range(strain_idx)]

                cross_immunity_row = cross_vals + [1.0]
                strain_betas.append(sb)
                strain_gammas.append(sg)
                strain_weights.append(sw)
                for i, row in enumerate(strain_cross_immunity):
                    row.append(cross_vals[i] if i < len(cross_vals) else rng.uniform(0.3, 0.8))
                strain_cross_immunity.append(cross_immunity_row)

        total_weight = sum(strain_weights)
        strain_weights = [w / total_weight for w in strain_weights]

        # --- Seasonal, climate, and environmental forcing ---
        primary_phase = rng.uniform(0, 2 * np.pi)
        secondary_phase = rng.uniform(0, 2 * np.pi)
        tertiary_phase = rng.uniform(0, 2 * np.pi)
        quaternary_phase = rng.uniform(0, 2 * np.pi)
        humidity = rng.uniform(0.75, 1.25)
        uv_index_effect = 1.0 - 0.10 * rng.uniform(0, 1)
        air_quality_index = rng.uniform(0.88, 1.12)
        pollen_factor = 1.0 + 0.05 * rng.uniform(0, 1)
        temperature_wave = (1.0 + 0.20 * np.sin(primary_phase)
                            + 0.10 * np.cos(secondary_phase)
                            + 0.05 * np.sin(tertiary_phase)
                            + 0.02 * np.cos(quaternary_phase))
        seasonal_modifier = temperature_wave * humidity * uv_index_effect * air_quality_index
        seasonal_modifier = max(seasonal_modifier, 0.40)

        # --- Spatial heterogeneity and clustering ---
        num_spatial_clusters = 5
        cluster_betas = []
        for cluster_idx in range(num_spatial_clusters):
            cluster_density = rng.uniform(0.5, 2.5)
            cluster_connectivity = rng.uniform(0.3, 1.0)
            cluster_beta = beta * cluster_density * cluster_connectivity
            cluster_betas.append(cluster_beta)
        spatial_effective_beta = np.mean(cluster_betas) * rng.uniform(0.85, 1.15)
        spatial_modifier = spatial_effective_beta / max(beta, 1e-9)
        spatial_modifier = min(max(spatial_modifier, 0.5), 3.0)

        # --- Hospital capacity and stress with surge capacity ---
        hospital_capacity_fraction = 0.012
        surge_capacity_fraction = 0.004
        hospital_capacity = max(int(N * hospital_capacity_fraction), 1)
        surge_capacity = int(N * surge_capacity_fraction)
        total_capacity = hospital_capacity + surge_capacity
        overcapacity_ratio = I / total_capacity if total_capacity > 0 else 1.0

        gamma_adjusted = gamma
        if overcapacity_ratio > 2.5:
            stress_penalty = min((overcapacity_ratio - 2.5) * 0.22 + 0.25, 0.70)
            gamma_adjusted = gamma * (1.0 - stress_penalty) * hw_effectiveness_multiplier
        elif overcapacity_ratio > 1.5:
            stress_penalty = min((overcapacity_ratio - 1.5) * 0.15, 0.25)
            gamma_adjusted = gamma * (1.0 - stress_penalty) * hw_effectiveness_multiplier
        elif overcapacity_ratio > 1.0:
            stress_penalty = min((overcapacity_ratio - 1.0) * 0.08, 0.12)
            gamma_adjusted = gamma * (1.0 - stress_penalty) * hw_effectiveness_multiplier
        else:
            gamma_adjusted = gamma * healthcare_boost * hw_effectiveness_multiplier

        gamma = gamma_adjusted

        # --- Network-based superspreader events with heterogeneous mixing ---
        network_types = {
            "household":   {"prob_base": 0.03, "prob_slope": 0.0005, "prob_max": 0.22, "amp_range": (1.2, 2.8)},
            "workplace":   {"prob_base": 0.02, "prob_slope": 0.0003, "prob_max": 0.17, "amp_range": (1.1, 2.2)},
            "social":      {"prob_base": 0.05, "prob_slope": 0.0008, "prob_max": 0.30, "amp_range": (1.5, 4.5)},
            "transport":   {"prob_base": 0.01, "prob_slope": 0.0002, "prob_max": 0.12, "amp_range": (1.05, 2.0)},
            "healthcare":  {"prob_base": 0.008, "prob_slope": 0.0004, "prob_max": 0.18, "amp_range": (1.3, 3.0)},
            "school":      {"prob_base": 0.025, "prob_slope": 0.0006, "prob_max": 0.20, "amp_range": (1.2, 2.5)},
        }
        network_amplification = 1.0
        event_count = 0

        if I > 0:
            for net_type, net_params in network_types.items():
                net_prob = min(net_params["prob_base"] + net_params["prob_slope"] * I, net_params["prob_max"])
                net_prob = net_prob * (1.0 - quarantine_efficiency)

                if net_type == "school":
                    net_prob = net_prob * school_closure_factor

                if net_type == "transport":
                    net_prob = net_prob * travel_restriction_factor

                if rng.random() < net_prob:
                    amp_range = net_params["amp_range"]
                    event_amp = rng.uniform(amp_range[0], amp_range[1])
                    network_amplification *= event_amp
                    event_count += 1

            # Diminishing returns for many simultaneous events
            if event_count > 3:
                diminishing_factor = 1.0 / (1.0 + 0.15 * (event_count - 3))
                network_amplification = network_amplification * diminishing_factor

            network_amplification = min(network_amplification, 15.0)

        # --- Age and socioeconomic stratified infection dynamics ---
        new_infections_total = 0
        stratum_infections_detail = {}

        weighted_nutrition = sum(
            sp["fraction"] * sp["nutrition_index"] for sp in socio_strata.values()
        )
        nutrition_immunity_effect = min(max(weighted_nutrition, 0.5), 1.5)

        for stratum_name, stratum_props in socio_strata.items():
            fraction = stratum_props["fraction"]
            crowding = stratum_props["crowding_factor"]
            compliance = stratum_props["compliance"] * compliance_fatigue_factor
            compliance = min(max(compliance, 0.0), 1.0)
            nutrition = stratum_props["nutrition_index"]

            S_stratum = int(round(S * fraction))
            if S_stratum <= 0 or I == 0:
                stratum_infections_detail[stratum_name] = 0
                continue

            stratum_total_infections = 0

            # Age-modulated infection within stratum
            age_susceptibility_weighted = sum(
                ap["fraction"] * ap["susceptibility"] for ap in age_strata.values()
            )

            for s_idx, (sb, sg, sw) in enumerate(zip(strain_betas, strain_gammas, strain_weights)):
                effective_strain_beta = sb * seasonal_modifier * crowding * network_amplification * spatial_modifier
                effective_strain_beta = effective_strain_beta * (1.0 - quarantine_efficiency * compliance)
                effective_strain_beta = effective_strain_beta * age_susceptibility_weighted
                effective_strain_beta = effective_strain_beta / max(nutrition_immunity_effect, 0.1)

                infection_prob = 1.0 - np.exp(-effective_strain_beta * sw * I / N)
                infection_prob = min(max(infection_prob, 0.0), 1.0)

                effective_S_stratum = int(S_stratum * (1.0 - quarantine_efficiency * compliance))
                quarantined_S_stratum = S_stratum - effective_S_stratum

                strain_infections = rng.binomial(max(effective_S_stratum, 0), infection_prob)

                if quarantined_S_stratum > 0:
                    leak_prob = infection_prob * (1.0 - compliance) * 0.08
                    leak_prob = min(max(leak_prob, 0.0), 1.0)
                    strain_infections += rng.binomial(quarantined_S_stratum, leak_prob)

                stratum_total_infections += strain_infections

            stratum_total_infections = min(stratum_total_infections, S_stratum)
            stratum_infections_detail[stratum_name] = stratum_total_infections
            new_infections_total += stratum_total_infections

        new_infections = min(new_infections_total, S)

        # --- Enhanced contact tracing with digital tools ---
        if contact_tracing_efficiency > 0 and new_infections > 0:
            digital_adoption = rng.uniform(0.30, 0.75)
            enhanced_tracing = contact_tracing_efficiency * (1.0 + digital_adoption * 0.30)
            enhanced_tracing = min(enhanced_tracing, 0.75)
            traced_and_isolated = rng.binomial(new_infections, enhanced_tracing)
            new_infections = max(new_infections - traced_and_isolated, 0)

        # --- Environmental reservoir transmission with pathogen decay ---
        env_transmission = 0
        if I > 0 and S > 0:
            shedding_rate = 0.0007 * I
            surface_decay = rng.uniform(0.20, 0.45)
            airborne_fraction = rng.uniform(0.30, 0.70)
            surface_fraction = 1.0 - airborne_fraction
            airborne_load = shedding_rate * airborne_fraction / max(surface_decay * 0.5, 1e-9)
            surface_load = shedding_rate * surface_fraction / max(surface_decay, 1e-9)
            env_prob_airborne = min(1.0 - np.exp(-airborne_load * 0.12), 0.030)
            env_prob_surface = min(1.0 - np.exp(-surface_load * 0.05), 0.012)
            env_prob = min(env_prob_airborne + env_prob_surface, 0.040)
            env_transmission = rng.binomial(max(S - new_infections, 0), env_prob)

        new_infections = min(new_infections + env_transmission, S)

        # --- Vector-borne component with seasonality ---
        vector_transmission = 0
        vector_season_prob = abs(np.sin(primary_phase))
        vector_population_size = rng.poisson(max(int(N * 0.001 * vector_season_prob), 0))

        if I > 0 and S > 0 and vector_population_size > 0:
            vector_beta = beta * 0.12 * seasonal_modifier
            vector_infection_rate = min(vector_population_size / max(N, 1) * vector_beta, 0.015)
            vector_prob = min(1.0 - np.exp(-vector_infection_rate * I / N), 0.012)
            vector_transmission = rng.binomial(max(S - new_infections, 0), vector_prob)

        new_infections = min(new_infections + vector_transmission, S)

        # --- Zoonotic spillover events ---
        zoonotic_transmission = 0
        zoonotic_base_prob = 0.0008
        if rng.random() < zoonotic_base_prob * seasonal_modifier:
            spillover_size = rng.poisson(max(int(N * 0.00005), 1))
            zoonotic_transmission = min(spillover_size, max(S - new_infections, 0))

        new_infections = min(new_infections + zoonotic_transmission, S)

        # --- Age and comorbidity-stratified multi-stage recovery ---
        recovery_stages = 6
        total_recoveries = 0
        I_remaining = I

        # Calculate comorbidity-weighted recovery penalty
        comorbidity_recovery_penalty = sum(
            cp["fraction"] * cp["recovery_penalty"] for cp in comorbidity_profiles.values()
        )

        for stage_idx in range(recovery_stages):
            if I_remaining <= 0:
                break

            if stage_idx < recovery_stages - 1:
                stage_fraction = 1.0 / (recovery_stages - stage_idx)
                stage_count = int(round(I_remaining * stage_fraction))
            else:
                stage_count = I_remaining

            stage_count = min(max(stage_count, 0), I_remaining)
            if stage_count <= 0:
                continue

            progressive_gamma = gamma * (0.30 + 0.70 * (stage_idx + 1) / recovery_stages)
            progressive_gamma = progressive_gamma * testing_boost * booster_effectiveness_modifier
            progressive_gamma = progressive_gamma * (1.0 - comorbidity_recovery_penalty)

            # Socioeconomic and age adjustment on recovery
            weighted_healthcare = sum(
                sp["fraction"] * sp["healthcare_access"] for sp in socio_strata.values()
            )
            age_severity_weighted = sum(
                ap["fraction"] * (1.0 - ap["severity"] * 0.3) for ap in age_strata.values()
            )
            progressive_gamma = progressive_gamma * weighted_healthcare * age_severity_weighted
            progressive_gamma = max(progressive_gamma, 0.001)

            recovery_prob = min(max(1.0 - np.exp(-progressive_gamma), 0.0), 1.0)
            stage_recoveries = rng.binomial(stage_count, recovery_prob)
            total_recoveries += stage_recoveries
            I_remaining -= stage_count

        new_recoveries = min(total_recoveries, I)

        # --- Long COVID and post-acute sequelae ---
        long_covid_fraction = rng.uniform(0.05, 0.20)
        long_covid_cases = rng.binomial(new_recoveries, long_covid_fraction)
        long_covid_recovery_delay = rng.poisson(int(long_covid_cases * 0.3))
        new_recoveries = max(new_recoveries - long_covid_recovery_delay, 0)

        # --- Multi-tier waning immunity with booster effects ---
        waning_tiers = [
            {"fraction": 0.15, "rate": 0.002, "reinfection_multiplier": 0.15, "booster_protection": 0.90},
            {"fraction": 0.30, "rate": 0.006, "reinfection_multiplier": 0.40, "booster_protection": 0.75},
            {"fraction": 0.30, "rate": 0.012, "reinfection_multiplier": 0.65, "booster_protection": 0.55},
            {"fraction": 0.15, "rate": 0.020, "reinfection_multiplier": 0.82, "booster_protection": 0.35},
            {"fraction": 0.10, "rate": 0.035, "reinfection_multiplier": 0.98, "booster_protection": 0.15},
        ]

        reinfection_from_R = 0
        for tier in waning_tiers:
            tier_R = int(R * tier["fraction"])
            if tier_R <= 0:
                continue

            waning_prob = min(1.0 - np.exp(-tier["rate"]), 1.0)
            waned = rng.binomial(tier_R, waning_prob)

            if waned > 0 and I > 0:
                booster_protection = tier["booster_protection"] * booster_effectiveness_modifier
                effective_reinfection_mult = tier["reinfection_multiplier"] * (1.0 - min(booster_protection, 0.95))
                reinfection_prob = min(
                    beta * effective_reinfection_mult * seasonal_modifier * I / N * 0.30,
                    1.0
                )
                reinfection_prob = max(reinfection_prob, 0.0)
                tier_reinfections = rng.binomial(waned, reinfection_prob)
                reinfection_from_R += tier_reinfections

        reinfection_from_R = min(reinfection_from_R, R)

        # --- Partial immunity from prior infection and cross-strain protection ---
        cross_strain_protection = 0
        if len(strain_weights) > 1 and R > 0:
            for s_idx in range(1, len(strain_weights)):
                if s_idx < len(strain_cross_immunity) and len(strain_cross_immunity[s_idx]) > 0:
                    avg_cross = np.mean(strain_cross_immunity[s_idx][:-1]) if len(strain_cross_immunity[s_idx]) > 1 else 0.5
                    cross_protection_rate = min(avg_cross * 0.008, 0.02)
                    cross_strain_protection += rng.binomial(max(int(R * 0.1), 0), min(cross_protection_rate, 1.0))

        reinfection_from_R = max(reinfection_from_R - cross_strain_protection, 0)
        reinfection_from_R = min(reinfection_from_R, R)

        # --- Update core compartments ---
        new_infections = min(new_infections, S)
        new_recoveries = min(new_recoveries, I)
        reinfection_from_R = min(reinfection_from_R, R)

        effective_S_loss = new_infections + vaccinated_this_step
        effective_S_loss = min(effective_S_loss, S)

        next_S = S - effective_S_loss + reinfection_from_R
        next_I = I + new_infections - new_recoveries - reinfection_from_R
        next_R = R + new_recoveries + vaccinated_this_step - reinfection_from_R

        # --- Age-stratified demographic dynamics ---
        birth_rate_base = 0.00012
        natural_death_rate_base = 0.000080
        disease_death_base = 0.00022

        # Age-weighted birth and death rates
        age_weighted_birth = sum(
            ap["fraction"] * birth_rate_base * (1.2 if ag in ["young", "adult"] else 0.5)
            for ag, ap in age_strata.items()
        )
        age_weighted_death = sum(
            ap["fraction"] * natural_death_rate_base * (0.3 if ag == "child" else
                                                          0.7 if ag == "young" else
                                                          1.0 if ag == "adult" else
                                                          2.5 if ag == "elderly" else 5.0)
            for ag, ap in age_strata.items()
        )

        births = rng.poisson(age_weighted_birth * N) if N > 0 else 0
        deaths_S = rng.binomial(max(int(next_S), 0), age_weighted_death) if next_S > 0 else 0

        deaths_I = 0
        if next_I > 0:
            overcap_penalty = 1.0 + max(overcapacity_ratio - 1.0, 0.0) * 0.60
            comorbidity_mortality = sum(
                cp["fraction"] * cp["mortality_multiplier"] for cp in comorbidity_profiles.values()
            )
            age_mortality_weight = sum(
                ap["fraction"] * ap["severity"] for ap in age_strata.values()
            )
            effective_disease_death = min(
                disease_death_base * overcap_penalty * comorbidity_mortality * age_mortality_weight,
                0.08
            )
            deaths_I_natural = rng.binomial(max(int(next_I), 0), age_weighted_death)
            deaths_I_disease = rng.binomial(max(int(next_I), 0), effective_disease_death)
            deaths_I = deaths_I_natural + deaths_I_disease

        deaths_R = rng.binomial(max(int(next_R), 0), age_weighted_death) if next_R > 0 else 0

        # --- Multi-corridor immigration/emigration dynamics ---
        immigration_corridors = rng.integers(2, 6)
        total_immigrants = 0
        total_immigrant_infected = 0
        total_emigrants_S = 0

        for corridor_idx in range(immigration_corridors):
            corridor_rate = rng.uniform(0.00003, 0.00020) * travel_restriction_factor
            corridor_immigrants = rng.poisson(corridor_rate * N) if N > 0 else 0
            corridor_infection_rate = rng.uniform(0.003, 0.05)
            corridor_infected = rng.binomial(corridor_immigrants, corridor_infection_rate)
            total_immigrants += corridor_immigrants
            total_immigrant_infected += corridor_infected

            corridor_emigration = rng.uniform(0.00002, 0.00012) * travel_restriction_factor
            corridor_emigrants = rng.binomial(max(int(next_S), 0), corridor_emigration) if next_S > 0 else 0
            total_emigrants_S += corridor_emigrants

        immigrant_susceptible = total_immigrants - total_immigrant_infected
        total_emigrants_S = min(total_emigrants_S, max(int(next_S), 0))

        # --- Refugee and displacement flows ---
        displacement_event_prob = 0.005
        displacement_inflow = 0
        if rng.random() < displacement_event_prob:
            displacement_size = rng.poisson(max(int(N * rng.uniform(0.001, 0.005)), 1))
            displacement_infected_frac = rng.uniform(0.02, 0.15)
            disp_infected = rng.binomial(displacement_size, displacement_infected_frac)
            displacement_inflow = displacement_size
            total_immigrant_infected += disp_infected
            immigrant_susceptible += (displacement_size - disp_infected)

        # --- Final compartment update ---
        next_S = next_S + births - deaths_S - total_emigrants_S + immigrant_susceptible
        next_I = next_I - deaths_I + total_immigrant_infected
        next_R = next_R - deaths_R

        # --- Catastrophic multi-type event system ---
        catastrophe_prob = 0.003
        num_catastrophe_checks = 3

        for cat_check in range(num_catastrophe_checks):
            if rng.random() < catastrophe_prob / num_catastrophe_checks:
                catastrophe_type = rng.integers(0, 6)

                if catastrophe_type == 0:
                    # Mass outbreak event
                    shock_fraction = rng.uniform(0.03, 0.10)
                    shock_count = int(min(next_S * shock_fraction, next_S))
                    shock_count = max(shock_count, 0)
                    next_S = max(int(next_S) - shock_count, 0)
                    next_I = int(next_I) + shock_count

                elif catastrophe_type == 1:
                    # Displacement/conflict event
                    displacement_fraction = rng.uniform(0.01, 0.06)
                    displaced = int(N * displacement_fraction)
                    s_displaced = int(displaced * 0.55)
                    r_displaced = int(displaced * 0.35)
                    i_displaced = displaced - s_displaced - r_displaced
                    next_S = max(int(next_S) - s_displaced, 0)
                    next_R = max(int(next_R) - r_displaced, 0)
                    next_I = max(int(next_I) - max(i_displaced, 0), 0)

                elif catastrophe_type == 2:
                    # Population surge (refugees arriving)
                    surge_fraction = rng.uniform(0.005, 0.025)
                    surge = int(N * surge_fraction)
                    surge_infected = rng.binomial(surge, rng.uniform(0.05, 0.20))
                    next_S = int(next_S) + (surge - surge_infected)
                    next_I = int(next_I) + surge_infected

                elif catastrophe_type == 3:
                    # Healthcare system collapse
                    collapse_factor = rng.uniform(0.10, 0.35)
                    gamma_collapse_penalty = gamma * collapse_factor
                    extra_deaths = rng.binomial(max(int(next_I), 0), min(gamma_collapse_penalty * 0.5, 0.10))
                    next_I = max(int(next_I) - extra_deaths, 0)

                elif catastrophe_type == 4:
                    # Natural disaster (floods, earthquakes)
                    disaster_mortality = rng.uniform(0.001, 0.008)
                    disaster_deaths_S = rng.binomial(max(int(next_S), 0), disaster_mortality)
                    disaster_deaths_I = rng.binomial(max(int(next_I), 0), disaster_mortality * 1.5)
                    disaster_deaths_R = rng.binomial(max(int(next_R), 0), disaster_mortality)
                    next_S = max(int(next_S) - disaster_deaths_S, 0)
                    next_I = max(int(next_I) - disaster_deaths_I, 0)
                    next_R = max(int(next_R) - disaster_deaths_R, 0)
                    # Post-disaster crowding increases transmission
                    post_disaster_infections = rng.binomial(max(int(next_S), 0), min(beta * 0.05, 0.05))
                    next_S = max(int(next_S) - post_disaster_infections, 0)
                    next_I = int(next_I) + post_disaster_infections

                else:
                    # Vaccine cold chain failure
                    if vaccinated_this_step > 0:
                        failed_fraction = rng.uniform(0.10, 0.40)
                        failed_vaccinations = int(vaccinated_this_step * failed_fraction)
                        next_R = max(int(next_R) - failed_vaccinations, 0)
                        next_S = int(next_S) + failed_vaccinations

        # --- Policy effectiveness feedback loop ---
        if action is not None and action > 0:
            policy_effectiveness_decay = rng.uniform(0.95, 1.0)
            next_I = int(next_I * policy_effectiveness_decay + (1 - policy_effectiveness_decay) * next_I)

        # --- Genetic drift and pathogen evolution effect on immune escape ---
        immune_escape_event_prob = 0.001
        if rng.random() < immune_escape_event_prob and R > 0:
            escape_fraction = rng.uniform(0.01, 0.05)
            immune_escaped = rng.binomial(max(int(next_R), 0), escape_fraction)
            immune_escaped = min(immune_escaped, max(int(next_R), 0))
            next_R = max(int(next_R) - immune_escaped, 0)
            next_I = int(next_I) + immune_escaped

        # --- Final safety clamp with conservation check ---
        next_S_final = max(int(next_S), 0)
        next_I_final = max(int(next_I), 0)
        next_R_final = max(int(next_R), 0)

        next_state["S"] = next_S_final
        next_state["I"] = next_I_final
        next_state["R"] = next_R_final

        return next_state


    def evaluate(self, x):
        return 0

