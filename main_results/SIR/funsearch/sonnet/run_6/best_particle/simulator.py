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
        """Improved version of `forward_v0`."""
        next_state = state.copy()

        S = state["S"]
        I = state["I"]
        R = state["R"]
        N = S + I + R

        if N == 0 or (S == 0 and I == 0):
            return next_state

        beta = np.clip(parameters[0], 0.0, 1.0)
        gamma = np.clip(parameters[1], 0.0, 1.0)

        # Extended age-structured cohorts: 7 age groups
        age_fractions = [0.10, 0.12, 0.18, 0.25, 0.18, 0.12, 0.05]
        age_susceptibility = [0.55, 0.75, 0.90, 1.0, 1.15, 1.45, 1.85]
        age_recovery_modifier = [1.5, 1.3, 1.1, 1.0, 0.85, 0.65, 0.40]
        age_mortality_risk = [0.00005, 0.0001, 0.0003, 0.0007, 0.002, 0.008, 0.025]
        age_comorbidity_factor = [1.0, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7]
        NUM_AGE = 7

        age_S = [max(0, int(S * f)) for f in age_fractions]
        age_I = [max(0, int(I * f)) for f in age_fractions]
        age_R = [max(0, int(R * f)) for f in age_fractions]

        # Correct rounding discrepancies
        for total, age_list in [(S, age_S), (I, age_I), (R, age_R)]:
            diff = total - sum(age_list)
            if diff > 0:
                age_list[3] += diff
            elif diff < 0:
                for idx in [3, 2, 4, 1, 5, 0, 6]:
                    reduction = min(age_list[idx], abs(diff))
                    age_list[idx] -= reduction
                    diff += reduction
                    if diff == 0:
                        break

        # Multi-strain disease model: 3 strains with different characteristics
        num_strains = 3
        strain_prevalence = [0.70, 0.20, 0.10]
        strain_beta_multiplier = [1.0, 1.35, 1.80]
        strain_gamma_multiplier = [1.0, 0.85, 0.65]
        strain_mortality_multiplier = [1.0, 1.20, 1.60]
        strain_vaccine_escape = [0.05, 0.20, 0.45]
        strain_cross_immunity = [[1.0, 0.7, 0.4], [0.7, 1.0, 0.6], [0.4, 0.6, 1.0]]

        # Recompute strain prevalence with drift
        for s in range(num_strains):
            drift = rng.uniform(-0.02, 0.02)
            strain_prevalence[s] = max(0.01, strain_prevalence[s] + drift)
        total_prev = sum(strain_prevalence)
        strain_prevalence = [p / total_prev for p in strain_prevalence]

        # Asymptomatic compartments per age group
        asymptomatic_base_fraction = 0.35
        age_asymptomatic_fraction = [
            asymptomatic_base_fraction * (1.2 - 0.05 * i) for i in range(NUM_AGE)
        ]
        age_asymptomatic = []
        age_symptomatic = []
        age_critical = []
        for age_idx in range(NUM_AGE):
            async_frac = np.clip(age_asymptomatic_fraction[age_idx], 0.1, 0.6)
            critical_frac = np.clip(age_mortality_risk[age_idx] * 15 * age_comorbidity_factor[age_idx], 0.0, 0.25)
            async_count = int(age_I[age_idx] * async_frac)
            crit_count = int(age_I[age_idx] * critical_frac)
            symp_count = max(0, age_I[age_idx] - async_count - crit_count)
            age_asymptomatic.append(async_count)
            age_symptomatic.append(symp_count)
            age_critical.append(crit_count)

        # Environmental contamination with decay and spatial spread
        base_env_contamination = np.clip((I / max(N, 1)) * 0.20 + rng.uniform(0.0, 0.03), 0.0, 0.35)
        environmental_layers = {
            "aerosol": base_env_contamination * 0.5,
            "surface": base_env_contamination * 0.3,
            "water": base_env_contamination * 0.2,
        }
        if I > N * 0.25:
            environmental_layers["aerosol"] *= 1.6
            environmental_layers["surface"] *= 1.3
        if I > N * 0.5:
            for key in environmental_layers:
                environmental_layers[key] = np.clip(environmental_layers[key] * 1.4, 0.0, 0.35)
        env_contamination_total = np.clip(sum(environmental_layers.values()), 0.0, 0.40)

        # Multi-factor seasonal and climate effects
        season_phase = rng.uniform(0, 2 * np.pi)
        secondary_phase = rng.uniform(0, np.pi)
        season_amplitude_primary = 0.18
        season_amplitude_secondary = 0.07
        seasonal_factor = (
            1.0
            + season_amplitude_primary * np.sin(season_phase)
            + season_amplitude_secondary * np.cos(secondary_phase)
        )
        temperature_effect = rng.normal(0, 0.04)
        humidity_effect = rng.normal(0, 0.025)
        uv_index_effect = rng.uniform(-0.02, 0.01)
        air_quality_effect = rng.uniform(-0.01, 0.03)
        beta_climate = np.clip(
            beta * seasonal_factor + temperature_effect + humidity_effect + uv_index_effect + air_quality_effect,
            0.0, 1.0
        )

        # Healthcare system with tiered capacity and resource constraints
        icu_capacity = max(1, int(N * 0.002))
        general_hospital_capacity = max(1, int(N * 0.05))
        medical_workers = max(1, int(N * 0.03))
        ppe_availability = np.clip(1.0 - rng.uniform(0.0, 0.15), 0.5, 1.0)

        critical_patients = sum(age_critical)
        icu_strain = min(1.0, critical_patients / icu_capacity)
        hospital_strain = min(1.0, I / general_hospital_capacity)
        worker_strain = min(1.0, I / max(medical_workers * 10, 1))

        supply_disruption = 0.0
        if I > general_hospital_capacity * 1.2:
            supply_disruption = np.clip(0.08 * ((I / general_hospital_capacity) - 1.2), 0.0, 0.35)
        if icu_strain > 0.8:
            supply_disruption = np.clip(supply_disruption + 0.05 * icu_strain, 0.0, 0.40)

        gamma_base_adjusted = gamma * (1.0 - 0.35 * hospital_strain) * (1.0 - supply_disruption) * ppe_availability
        icu_mortality_amplifier = 1.0 + 2.5 * icu_strain

        # Behavioral dynamics: social compliance, fatigue, and adaptation
        social_compliance = 1.0
        behavioral_adaptation = 0.0
        pandemic_fatigue = 0.0
        voluntary_distancing = 0.0

        if action is not None:
            if action > 2:
                pandemic_fatigue = np.clip(0.04 * (action - 2), 0.0, 0.30)
                compliance_noise = rng.uniform(-0.03, 0.02)
                social_compliance = np.clip(1.0 - pandemic_fatigue + compliance_noise, 0.35, 1.0)
            if I > N * 0.1:
                voluntary_distancing = np.clip(0.10 * (I / N) * 10, 0.0, 0.25)
                behavioral_adaptation = np.clip(0.05 * (I / N) * 20, 0.0, 0.15)
        else:
            if I > N * 0.05:
                voluntary_distancing = np.clip(0.08 * (I / N) * 20, 0.0, 0.20)
                behavioral_adaptation = np.clip(0.04 * (I / N) * 25, 0.0, 0.10)

        # Multi-tier intervention system
        effective_beta = beta_climate * (1.0 - voluntary_distancing)
        intervention_boost_gamma = 0.0
        testing_capacity_boost = 0.0
        contact_tracing_efficiency = 0.0
        border_control_factor = 1.0

        if action is not None:
            if action < 0:
                escalation_levels = abs(action)
                amplify = 1.0
                for lvl in range(min(escalation_levels, 8)):
                    amplify += 0.10 * np.exp(-0.12 * lvl) * rng.uniform(0.9, 1.1)
                amplify = min(amplify, 2.5)
                effective_beta = np.clip(beta_climate * amplify, 0.0, 1.0)
            elif action == 0:
                effective_beta = np.clip(beta_climate * (1.0 - voluntary_distancing), 0.0, 1.0)
            else:
                cumulative_reduction = 0.0
                remaining = action
                policy_tiers = [
                    ("hygiene_campaign", 0.05, 1, 0.005, 0.01, 0.0),
                    ("mask_mandate", 0.09, 1, 0.012, 0.02, 0.0),
                    ("social_distancing", 0.11, 1, 0.008, 0.01, 0.0),
                    ("school_closure", 0.13, 1, 0.0, 0.0, 0.0),
                    ("workplace_restrictions", 0.10, 1, 0.006, 0.01, 0.0),
                    ("partial_lockdown", 0.16, 1, 0.010, 0.03, 0.05),
                    ("contact_tracing_basic", 0.07, 1, 0.018, 0.05, 0.0),
                    ("mass_testing", 0.08, 1, 0.020, 0.08, 0.0),
                    ("full_lockdown", 0.19, 1, 0.012, 0.04, 0.10),
                    ("travel_ban", 0.08, 1, 0.0, 0.0, 0.20),
                    ("curfew", 0.07, 1, 0.004, 0.01, 0.0),
                    ("contact_tracing_advanced", 0.10, 1, 0.025, 0.15, 0.0),
                    ("emergency_healthcare", 0.04, 1, 0.035, 0.02, 0.0),
                    ("border_closure", 0.06, 1, 0.0, 0.0, 0.40),
                ]
                for tier_name, reduction, cost, gamma_boost, tracing_boost, border_boost in policy_tiers:
                    if remaining <= 0:
                        break
                    if remaining >= cost:
                        compliance_factor = np.exp(-0.06 * (action - remaining)) * social_compliance
                        adaptation_bonus = behavioral_adaptation * 0.5
                        cumulative_reduction += (reduction + adaptation_bonus) * compliance_factor
                        intervention_boost_gamma += gamma_boost * compliance_factor
                        testing_capacity_boost += tracing_boost * compliance_factor
                        border_control_factor = max(0.0, border_control_factor - border_boost * compliance_factor)
                        remaining -= cost
                    else:
                        compliance_factor = np.exp(-0.06 * (action - remaining)) * social_compliance
                        partial_ratio = remaining / cost
                        adaptation_bonus = behavioral_adaptation * 0.5 * partial_ratio
                        cumulative_reduction += (reduction + adaptation_bonus) * partial_ratio * compliance_factor
                        intervention_boost_gamma += gamma_boost * partial_ratio * compliance_factor
                        testing_capacity_boost += tracing_boost * partial_ratio * compliance_factor
                        border_control_factor = max(0.0, border_control_factor - border_boost * partial_ratio * compliance_factor)
                        remaining = 0

                fatigue_penalty = 0.0
                if action > 5:
                    fatigue_penalty = 0.025 * (action - 5) * (1.0 - social_compliance)
                if action > 9:
                    fatigue_penalty += 0.015 * (action - 9) * pandemic_fatigue
                net_reduction = max(0.0, cumulative_reduction - fatigue_penalty)
                effective_beta = np.clip(
                    beta_climate * (1.0 - net_reduction) * (1.0 - voluntary_distancing),
                    0.0, 1.0
                )
                contact_tracing_efficiency = np.clip(testing_capacity_boost, 0.0, 0.55)

        gamma_effective_base = np.clip(gamma_base_adjusted + intervention_boost_gamma, 0.0, 1.0)

        # Multi-dose vaccination with waning immunity and age prioritization
        vaccination_rate_base = 0.0015
        booster_rate = 0.0005
        if action is not None and action > 1:
            vaccination_rate_base += 0.0012 * min(action - 1, 7)
            booster_rate += 0.0003 * min(action - 1, 7)

        age_vaccination_priority = [0.40, 0.50, 0.70, 0.90, 1.30, 1.60, 1.90]
        age_vaccine_hesitancy = [0.15, 0.20, 0.18, 0.12, 0.08, 0.05, 0.04]
        age_vaccine_efficacy = [0.92, 0.90, 0.88, 0.85, 0.80, 0.72, 0.62]

        vaccinated_total = 0
        for age_idx in range(NUM_AGE):
            if age_S[age_idx] > 0:
                priority = age_vaccination_priority[age_idx]
                hesitancy = age_vaccine_hesitancy[age_idx]
                efficacy = age_vaccine_efficacy[age_idx]
                v_prob = np.clip(
                    vaccination_rate_base * priority * (1.0 - hesitancy) * age_susceptibility[age_idx],
                    0.0, 1.0
                )
                vaccinated = rng.binomial(age_S[age_idx], v_prob)
                effective_vaccinated = rng.binomial(min(vaccinated, age_S[age_idx]), efficacy)
                partial_vaccinated = min(vaccinated, age_S[age_idx]) - effective_vaccinated
                age_S[age_idx] -= min(vaccinated, age_S[age_idx])
                age_R[age_idx] += effective_vaccinated
                age_S[age_idx] += partial_vaccinated
                vaccinated_total += effective_vaccinated

        # Booster doses for recovered individuals
        for age_idx in range(NUM_AGE):
            if age_R[age_idx] > 0:
                b_prob = np.clip(booster_rate * age_vaccination_priority[age_idx], 0.0, 0.05)
                boosted = rng.binomial(age_R[age_idx], b_prob)

        # Multi-variant mutation system
        mutation_prob_base = 0.012
        if I > N * 0.2:
            mutation_prob_base += 0.008
        mutation_factor = 1.0
        gamma_mutation_modifier = 1.0
        variant_emerged = False
        variant_severity = 0
        num_simultaneous_mutations = 0

        mutation_roll = rng.random()
        if mutation_roll < mutation_prob_base and I > 5:
            num_simultaneous_mutations = rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10])
            for mut_event in range(num_simultaneous_mutations):
                mutation_severity = rng.choice([0, 1, 2, 3, 4], p=[0.35, 0.28, 0.18, 0.12, 0.07])
                if mutation_severity == 0:
                    pass
                elif mutation_severity == 1:
                    mf = rng.uniform(1.05, 1.25)
                    gm = rng.uniform(0.93, 1.05)
                    mutation_factor *= mf
                    gamma_mutation_modifier *= gm
                    variant_emerged = True
                    variant_severity = max(variant_severity, 1)
                elif mutation_severity == 2:
                    mf = rng.uniform(1.25, 1.65)
                    gm = rng.uniform(0.80, 0.95)
                    mutation_factor *= mf
                    gamma_mutation_modifier *= gm
                    variant_emerged = True
                    variant_severity = max(variant_severity, 2)
                elif mutation_severity == 3:
                    mf = rng.uniform(1.65, 2.20)
                    gm = rng.uniform(0.65, 0.85)
                    mutation_factor *= mf
                    gamma_mutation_modifier *= gm
                    variant_emerged = True
                    variant_severity = max(variant_severity, 3)
                else:
                    mf = rng.uniform(2.20, 3.50)
                    gm = rng.uniform(0.45, 0.70)
                    mutation_factor *= mf
                    gamma_mutation_modifier *= gm
                    variant_emerged = True
                    variant_severity = max(variant_severity, 4)

            mutation_factor = np.clip(mutation_factor, 0.8, 4.0)
            gamma_mutation_modifier = np.clip(gamma_mutation_modifier, 0.3, 1.2)
            effective_beta = np.clip(effective_beta * mutation_factor, 0.0, 1.0)
            gamma_effective_base = np.clip(gamma_effective_base * gamma_mutation_modifier, 0.0, 1.0)

        # Quarantine and isolation with multi-tier contact tracing
        quarantine_rate = 0.0
        isolation_compliance = 0.85
        if action is not None and action >= 1:
            quarantine_rate = np.clip(0.04 * action, 0.0, 0.45)
            if action >= 3:
                isolation_compliance = np.clip(0.85 - 0.02 * (action - 3) * (1.0 - social_compliance), 0.50, 0.95)

        quarantined_I_per_age = []
        traced_quarantine_extra = []
        isolated_critical_per_age = []

        for age_idx in range(NUM_AGE):
            effective_quarantine_rate = quarantine_rate * isolation_compliance
            if age_symptomatic[age_idx] > 0:
                q = rng.binomial(age_symptomatic[age_idx], np.clip(effective_quarantine_rate, 0.0, 1.0))
                quarantined_I_per_age.append(q)
            else:
                quarantined_I_per_age.append(0)

            if age_asymptomatic[age_idx] > 0 and contact_tracing_efficiency > 0:
                tq_eff = contact_tracing_efficiency * social_compliance
                tq = rng.binomial(age_asymptomatic[age_idx], np.clip(tq_eff, 0.0, 1.0))
                traced_quarantine_extra.append(tq)
            else:
                traced_quarantine_extra.append(0)

            if age_critical[age_idx] > 0:
                iso_rate = np.clip(0.70 + 0.05 * (action if action is not None else 0), 0.0, 0.95)
                iso_crit = rng.binomial(age_critical[age_idx], iso_rate)
                isolated_critical_per_age.append(iso_crit)
            else:
                isolated_critical_per_age.append(0)

        for age_idx in range(NUM_AGE):
            tq = traced_quarantine_extra[age_idx]
            age_asymptomatic[age_idx] = max(0, age_asymptomatic[age_idx] - tq)
            quarantined_I_per_age[age_idx] += tq

        active_symptomatic_per_age = [
            max(0, age_symptomatic[i] - (quarantined_I_per_age[i] - traced_quarantine_extra[i]))
            for i in range(NUM_AGE)
        ]
        active_critical_per_age = [
            max(0, age_critical[i] - isolated_critical_per_age[i])
            for i in range(NUM_AGE)
        ]
        active_I_per_age = [
            active_symptomatic_per_age[i] + age_asymptomatic[i] + active_critical_per_age[i]
            for i in range(NUM_AGE)
        ]

        # Network-based mixing: superspreader events and cluster outbreaks
        hub_probability = 0.07
        hub_amplification = 1.0
        cluster_event = False
        cluster_multiplier = 1.0

        if rng.random() < hub_probability and I > 3:
            hub_size = rng.integers(2, min(15, I + 1))
            hub_amplification = 1.0 + 0.12 * hub_size * rng.uniform(0.8, 1.2)
            if hub_amplification > 2.5:
                cluster_event = True
                cluster_multiplier = rng.uniform(1.3, 2.0)
            effective_beta = np.clip(effective_beta * hub_amplification, 0.0, 1.0)

        # Spatial heterogeneity: urban, suburban, rural zones
        urban_fraction = 0.55
        suburban_fraction = 0.30
        rural_fraction = 0.15
        urban_density_factor = 1.45
        suburban_density_factor = 1.10
        rural_density_factor = 0.65

        total_active_I = sum(active_I_per_age)
        total_active_S = sum(age_S)

        urban_S = int(total_active_S * urban_fraction)
        suburban_S = int(total_active_S * suburban_fraction)
        rural_S = total_active_S - urban_S - suburban_S

        urban_I = int(total_active_I * urban_fraction)
        suburban_I = int(total_active_I * suburban_fraction)
        rural_I = total_active_I - urban_I - suburban_I

        urban_N = max(1, int(N * urban_fraction))
        suburban_N = max(1, int(N * suburban_fraction))
        rural_N = max(1, N - urban_N - suburban_N)

        prevalence = I / max(N, 1)
        mixing_exponent = 1.0 + 0.25 * prevalence + 0.12 * hospital_strain + 0.05 * icu_strain
        noise_scale = 0.003 + 0.010 * prevalence + 0.006 * hospital_strain + 0.004 * icu_strain

        urban_force = 0.0
        if urban_N > 0 and urban_I > 0:
            urban_force = effective_beta * urban_density_factor * (urban_I / urban_N) ** mixing_exponent
        suburban_force = 0.0
        if suburban_N > 0 and suburban_I > 0:
            suburban_force = effective_beta * suburban_density_factor * (suburban_I / suburban_N) ** mixing_exponent
        rural_force = 0.0
        if rural_N > 0 and rural_I > 0:
            rural_force = effective_beta * rural_density_factor * (rural_I / rural_N) ** mixing_exponent

        cross_transmission_urban_suburban = 0.08 * (urban_I / max(urban_N, 1))
        cross_transmission_suburban_rural = 0.05 * (suburban_I / max(suburban_N, 1))
        cross_transmission_urban_rural = border_control_factor * 0.03 * (urban_I / max(urban_N, 1))

        env_cross = env_contamination_total * 0.4

        infection_force_urban = np.clip(
            urban_force + rng.normal(0, noise_scale) + env_cross + cross_transmission_urban_suburban * 0.5,
            0.0, 1.0
        )
        infection_force_suburban = np.clip(
            suburban_force + rng.normal(0, noise_scale * 0.8) + env_cross
            + cross_transmission_urban_suburban * 0.5
            + cross_transmission_suburban_rural * 0.5,
            0.0, 1.0
        )
        infection_force_rural = np.clip(
            rural_force + rng.normal(0, noise_scale * 0.5) + env_cross * 0.5
            + cross_transmission_suburban_rural * 0.5
            + cross_transmission_urban_rural,
            0.0, 1.0
        )

        infection_force = (
            infection_force_urban * urban_fraction
            + infection_force_suburban * suburban_fraction
            + infection_force_rural * rural_fraction
        )

        dynamic_rate = infection_force + gamma_effective_base + icu_strain * 0.1
        if dynamic_rate > 0.75:
            sub_steps = 8
        elif dynamic_rate > 0.62:
            sub_steps = 7
        elif dynamic_rate > 0.50:
            sub_steps = 6
        elif dynamic_rate > 0.38:
            sub_steps = 5
        elif dynamic_rate > 0.25:
            sub_steps = 4
        else:
            sub_steps = 3

        prev_infection_force = infection_force
        total_deaths = 0
        cumulative_hospitalizations = 0

        # Multi-step simulation with advanced dynamics
        for step in range(sub_steps):
            total_temp_I = sum(active_I_per_age)
            total_temp_S = sum(age_S)
            total_temp_R = sum(age_R)
            total_temp_N = total_temp_S + total_temp_I + sum(age_R) + sum(quarantined_I_per_age) + sum(isolated_critical_per_age)

            if total_temp_N == 0:
                break

            interpolation_weight = (step + 1) / sub_steps
            smoothed_force = (1.0 - interpolation_weight) * prev_infection_force + interpolation_weight * infection_force

            if cluster_event and step < 2:
                smoothed_force = np.clip(smoothed_force * cluster_multiplier, 0.0, 1.0)

            sub_recovery_base = 1.0 - (1.0 - gamma_effective_base) ** (1.0 / sub_steps)
            sub_infection_base = 1.0 - (1.0 - smoothed_force) ** (1.0 / sub_steps)

            # Immunity waning with variant escape and cross-immunity
            waning_prob_base = 0.0008 / sub_steps
            variant_escape_factor = 1.0
            if variant_emerged:
                if variant_severity >= 3:
                    variant_escape_factor = rng.uniform(1.8, 3.0)
                elif variant_severity == 2:
                    variant_escape_factor = rng.uniform(1.4, 2.0)
                else:
                    variant_escape_factor = rng.uniform(1.1, 1.5)

            for age_idx in range(NUM_AGE):
                if age_R[age_idx] > 0:
                    effective_waning = np.clip(
                        waning_prob_base * age_susceptibility[age_idx] * variant_escape_factor * age_comorbidity_factor[age_idx],
                        0.0, 1.0
                    )
                    waned = rng.binomial(age_R[age_idx], effective_waning)
                    partial_immunity_prob = 0.45 + 0.05 * (variant_severity > 1)
                    waned_partial = int(waned * partial_immunity_prob)
                    waned_full = waned - waned_partial
                    age_R[age_idx] = max(0, age_R[age_idx] - waned)
                    partial_susceptibility_factor = np.clip(0.4 * (1.0 + 0.2 * variant_severity), 0.0, 1.0)
                    effective_partial_waned = int(waned_partial * partial_susceptibility_factor)
                    age_S[age_idx] += waned_full + effective_partial_waned

            # Super-spreader events with network effects
            super_spreader_factor = 1.0
            if rng.random() < 0.04 and total_temp_I > 0:
                ss_base = rng.uniform(1.3, 3.5)
                if mutation_factor > 1.3:
                    ss_base *= rng.uniform(1.1, 1.6)
                if hub_amplification > 1.2:
                    ss_base *= rng.uniform(1.05, 1.25)
                if icu_strain > 0.7:
                    ss_base *= rng.uniform(1.1, 1.3)
                super_spreader_factor = min(ss_base, 5.0)

            # Multi-strain reinfection dynamics
            reinfection_rate_base = 0.0008 / sub_steps
            strain_reinfection_multipliers = [1.0, 1.5, 2.8]
            effective_reinfection_rate = reinfection_rate_base
            for s_idx in range(num_strains):
                effective_reinfection_rate += (
                    reinfection_rate_base * strain_prevalence[s_idx] * strain_reinfection_multipliers[min(s_idx, 2)]
                )
            if variant_emerged:
                effective_reinfection_rate *= rng.uniform(1.5, 4.0)

            for age_idx in range(NUM_AGE):
                # Reinfection
                if age_R[age_idx] > 0:
                    reinf_prob = np.clip(effective_reinfection_rate * age_susceptibility[age_idx], 0.0, 1.0)
                    reinfected = rng.binomial(age_R[age_idx], reinf_prob)
                    reinfected = min(reinfected, age_R[age_idx])
                    age_R[age_idx] -= reinfected
                    active_I_per_age[age_idx] += reinfected

                # New infections
                age_infection_prob = np.clip(
                    sub_infection_base * age_susceptibility[age_idx] * super_spreader_factor * age_comorbidity_factor[age_idx],
                    0.0, 1.0
                )
                if age_S[age_idx] > 0:
                    new_inf = rng.binomial(age_S[age_idx], age_infection_prob)
                    new_inf = min(new_inf, age_S[age_idx])
                    age_S[age_idx] -= new_inf
                    active_I_per_age[age_idx] += new_inf
                    new_critical_fraction = np.clip(age_mortality_risk[age_idx] * 12 * age_comorbidity_factor[age_idx], 0.0, 0.30)
                    new_critical = int(new_inf * new_critical_fraction)
                    active_critical_per_age[age_idx] += new_critical
                    cumulative_hospitalizations += new_critical

                # Mortality with healthcare capacity
                if active_I_per_age[age_idx] > 0:
                    base_mort = age_mortality_risk[age_idx] * age_comorbidity_factor[age_idx] / sub_steps
                    capacity_mort_multiplier = 1.0 + icu_mortality_amplifier * icu_strain
                    if variant_emerged and mutation_factor > 1.5:
                        base_mort *= (1.0 + 0.4 * (variant_severity - 1))
                    mort_prob = np.clip(base_mort * capacity_mort_multiplier, 0.0, 1.0)
                    deaths = rng.binomial(active_I_per_age[age_idx], mort_prob)
                    deaths = min(deaths, active_I_per_age[age_idx])
                    active_I_per_age[age_idx] -= deaths
                    active_critical_per_age[age_idx] = max(0, active_critical_per_age[age_idx] - int(deaths * 0.6))
                    total_deaths += deaths

                # Recovery with heterogeneous rates and comorbidity effects
                if active_I_per_age[age_idx] > 0:
                    age_rec_prob = np.clip(
                        sub_recovery_base * age_recovery_modifier[age_idx] / age_comorbidity_factor[age_idx],
                        0.0, 1.0
                    )
                    fast_count = int(active_I_per_age[age_idx] * 0.15)
                    moderate_count = int(active_I_per_age[age_idx] * 0.50)
                    slow_count = int(active_I_per_age[age_idx] * 0.25)
                    critical_rec_count = max(0, active_I_per_age[age_idx] - fast_count - moderate_count - slow_count)

                    fast_rec = rng.binomial(fast_count, np.clip(age_rec_prob * 1.9, 0.0, 1.0)) if fast_count > 0 else 0
                    mod_rec = rng.binomial(moderate_count, np.clip(age_rec_prob, 0.0, 1.0)) if moderate_count > 0 else 0
                    slow_rec = rng.binomial(slow_count, np.clip(age_rec_prob * 0.40, 0.0, 1.0)) if slow_count > 0 else 0
                    crit_rec = rng.binomial(critical_rec_count, np.clip(age_rec_prob * 0.20, 0.0, 1.0)) if critical_rec_count > 0 else 0

                    total_rec = min(fast_rec + mod_rec + slow_rec + crit_rec, active_I_per_age[age_idx])
                    active_I_per_age[age_idx] -= total_rec
                    age_R[age_idx] += total_rec
                    active_critical_per_age[age_idx] = max(0, active_critical_per_age[age_idx] - int(total_rec * 0.3))

                # Quarantine recovery and mortality
                if quarantined_I_per_age[age_idx] > 0:
                    q_rec_prob = np.clip(
                        sub_recovery_base * age_recovery_modifier[age_idx] * 1.20 / age_comorbidity_factor[age_idx],
                        0.0, 1.0
                    )
                    q_mort_prob = np.clip(
                        age_mortality_risk[age_idx] * age_comorbidity_factor[age_idx] * 0.35 / sub_steps,
                        0.0, 1.0
                    )
                    if icu_strain > 0.9:
                        q_mort_prob = np.clip(q_mort_prob * 1.5, 0.0, 1.0)
                    q_deaths = rng.binomial(quarantined_I_per_age[age_idx], q_mort_prob)
                    q_deaths = min(q_deaths, quarantined_I_per_age[age_idx])
                    quarantined_I_per_age[age_idx] -= q_deaths
                    total_deaths += q_deaths
                    if quarantined_I_per_age[age_idx] > 0:
                        q_rec = rng.binomial(quarantined_I_per_age[age_idx], q_rec_prob)
                        q_rec = min(q_rec, quarantined_I_per_age[age_idx])
                        quarantined_I_per_age[age_idx] -= q_rec
                        age_R[age_idx] += q_rec

                # Isolated critical patients
                if isolated_critical_per_age[age_idx] > 0:
                    icu_rec_prob = np.clip(
                        sub_recovery_base * age_recovery_modifier[age_idx] * 0.85 / age_comorbidity_factor[age_idx],
                        0.0, 1.0
                    )
                    icu_mort_prob = np.clip(
                        age_mortality_risk[age_idx] * age_comorbidity_factor[age_idx] * 0.8 * (1.0 + icu_strain * 0.5) / sub_steps,
                        0.0, 1.0
                    )
                    icu_deaths = rng.binomial(isolated_critical_per_age[age_idx], icu_mort_prob)
                    icu_deaths = min(icu_deaths, isolated_critical_per_age[age_idx])
                    isolated_critical_per_age[age_idx] -= icu_deaths
                    total_deaths += icu_deaths
                    if isolated_critical_per_age[age_idx] > 0:
                        icu_rec = rng.binomial(isolated_critical_per_age[age_idx], icu_rec_prob)
                        icu_rec = min(icu_rec, isolated_critical_per_age[age_idx])
                        isolated_critical_per_age[age_idx] -= icu_rec
                        age_R[age_idx] += icu_rec

            # Environmental contamination decay with weather effects
            weather_decay = rng.uniform(0.80, 0.92)
            for key in environmental_layers:
                environmental_layers[key] = np.clip(environmental_layers[key] * weather_decay, 0.0, 0.35)
            env_contamination_total = np.clip(sum(environmental_layers.values()), 0.0, 0.40)

            # Update icu strain dynamically
            current_critical = sum(active_critical_per_age) + sum(isolated_critical_per_age)
            icu_strain = min(1.0, current_critical / max(icu_capacity, 1))
            icu_mortality_amplifier = 1.0 + 2.5 * icu_strain

            # Recompute infection force for next sub-step
            if step < sub_steps - 1:
                next_total_I = sum(active_I_per_age)
                next_N = sum(age_S) + next_total_I + sum(age_R) + sum(quarantined_I_per_age) + sum(isolated_critical_per_age)
                if next_N > 0 and next_total_I > 0:
                    prev_infection_force = infection_force
                    next_prev = next_total_I / next_N
                    next_mix_exp = 1.0 + 0.25 * next_prev + 0.08 * hospital_strain
                    step_noise = rng.normal(0, noise_scale * 0.30)
                    env_boost = env_contamination_total * 0.25
                    recomputed_force = effective_beta * (next_total_I / next_N) ** next_mix_exp + step_noise + env_boost
                    if cluster_event and step < 2:
                        recomputed_force *= cluster_multiplier * 0.5
                    infection_force = np.clip(recomputed_force, 0.0, 1.0)
                elif next_total_I == 0:
                    infection_force = 0.0
                    prev_infection_force = 0.0

        # Aggregate all compartments
        final_S = sum(age_S)
        final_I = sum(active_I_per_age) + sum(quarantined_I_per_age) + sum(isolated_critical_per_age)
        final_R = sum(age_R)

        effective_N = max(0, N - total_deaths)

        final_S = int(max(0, final_S))
        final_I = int(max(0, final_I))
        final_R = int(max(0, final_R))

        # Population conservation with deaths
        current_total = final_S + final_I + final_R
        if current_total != effective_N:
            diff = effective_N - current_total
            if diff > 0:
                final_R += diff
            else:
                excess = abs(diff)
                for comp_name, comp_val in [("R", final_R), ("S", final_S), ("I", final_I)]:
                    if excess <= 0:
                        break
                    reduction = min(comp_val, excess)
                    if comp_name == "R":
                        final_R -= reduction
                    elif comp_name == "S":
                        final_S -= reduction
                    else:
                        final_I -= reduction
                    excess -= reduction

        # Reconcile with original N
        current_total = final_S + final_I + final_R
        if current_total != N:
            diff = N - current_total
            if diff > 0:
                final_R += diff
            elif diff < 0:
                excess = abs(diff)
                for comp_name in ["R", "S", "I"]:
                    if excess <= 0:
                        break
                    if comp_name == "R":
                        reduction = min(final_R, excess)
                        final_R -= reduction
                    elif comp_name == "S":
                        reduction = min(final_S, excess)
                        final_S -= reduction
                    else:
                        reduction = min(final_I, excess)
                        final_I -= reduction
                    excess -= reduction

        if final_S + final_I + final_R != N:
            final_R = max(0, N - final_S - final_I)

        # Boundary checks
        if final_I < 0:
            final_S += abs(final_I)
            final_I = 0
        if final_S < 0:
            final_R += abs(final_S)
            final_S = 0
        if final_R < 0:
            final_S += abs(final_R)
            final_R = 0

        next_state["S"] = int(final_S)
        next_state["I"] = int(final_I)
        next_state["R"] = int(final_R)

        return next_state


    def evaluate(self, x):
        return 0

