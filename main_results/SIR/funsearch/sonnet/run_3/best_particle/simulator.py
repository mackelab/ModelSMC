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

        if N <= 0:
            return next_state

        beta, gamma = parameters[0], parameters[1]

        # --- Extended age-cohort stratification (7 tiers) ---
        age_fractions = [0.08, 0.12, 0.15, 0.25, 0.18, 0.14, 0.08]
        age_labels = ["infant", "child", "teen", "adult", "middle_aged", "senior", "elder"]
        age_susceptibility = [0.45, 0.60, 0.78, 1.00, 1.15, 1.35, 1.65]
        age_recovery_mod = [1.20, 1.45, 1.30, 1.00, 0.88, 0.72, 0.55]
        age_hospitalization_risk = [0.005, 0.01, 0.015, 0.05, 0.10, 0.22, 0.38]
        age_mortality_risk = [0.0001, 0.0002, 0.0004, 0.0018, 0.008, 0.018, 0.055]
        age_compliance = [1.00, 0.72, 0.58, 0.76, 0.82, 0.90, 0.88]
        age_mobility = [0.30, 0.65, 0.90, 1.00, 0.95, 0.75, 0.50]
        age_contact_rate = [3.2, 8.5, 12.0, 9.0, 7.5, 5.0, 3.0]

        num_cohorts = len(age_fractions)

        age_S = [int(S * f) for f in age_fractions]
        age_I = [int(I * f) for f in age_fractions]
        age_R = [int(R * f) for f in age_fractions]

        age_S[-1] += S - sum(age_S)
        age_I[-1] += I - sum(age_I)
        age_R[-1] += R - sum(age_R)

        # --- Multi-dose vaccination tracking ---
        vax_coverage_dose1 = [0.0] * num_cohorts
        vax_coverage_dose2 = [0.0] * num_cohorts
        vax_coverage_booster = [0.0] * num_cohorts
        vax_efficacy_dose1 = [0.65, 0.68, 0.70, 0.72, 0.70, 0.65, 0.58]
        vax_efficacy_dose2 = [0.88, 0.90, 0.91, 0.92, 0.90, 0.85, 0.78]
        vax_efficacy_booster = [0.94, 0.95, 0.95, 0.96, 0.94, 0.91, 0.85]

        # --- Comorbidity burden per cohort ---
        comorbidity_prevalence = [0.02, 0.04, 0.06, 0.15, 0.28, 0.42, 0.58]
        comorbidity_mortality_multiplier = [2.0, 2.5, 2.8, 3.2, 3.5, 3.8, 4.2]
        comorbidity_hospitalization_multiplier = [1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]

        # --- Social mixing matrix (simplified NGM approach) ---
        contact_matrix = np.zeros((num_cohorts, num_cohorts))
        for i in range(num_cohorts):
            for j in range(num_cohorts):
                age_diff = abs(i - j)
                base_contact = age_contact_rate[i] * age_contact_rate[j] / max(sum(age_contact_rate), 1)
                distance_decay = np.exp(-0.35 * age_diff)
                household_boost = 1.40 if age_diff <= 1 else 1.0
                contact_matrix[i][j] = base_contact * distance_decay * household_boost

        # --- Multi-layer action handling ---
        active_policies = []
        policy_memory = {}
        policy_durations = {}
        policy_fatigue_factor = 1.0
        policy_interaction_matrix = {
            ("mild_distancing", "mask_mandate"): 0.94,
            ("strict_distancing", "closure"): 0.88,
            ("quarantine", "test_and_trace"): 0.85,
            ("targeted_isolation", "healthcare_surge"): 0.90,
            ("combined_response", "travel_restriction"): 0.91,
            ("vaccination", "mask_mandate"): 0.97,
            ("closure", "travel_restriction"): 0.86,
            ("healthcare_surge", "quarantine"): 0.87,
            ("school_closure", "remote_work"): 0.89,
            ("targeted_isolation", "test_and_trace"): 0.83,
            ("mask_mandate", "ventilation_upgrade"): 0.92,
            ("vaccination", "healthcare_surge"): 0.95,
        }

        beta_original = beta
        gamma_original = gamma

        if action is not None:
            if action == 1:
                effective_compliance = sum(age_compliance[i] * age_mobility[i] * age_fractions[i] for i in range(num_cohorts))
                beta *= (0.78 * effective_compliance + 0.22)
                beta = max(0.0, min(beta, 1.0))
                active_policies.append("mild_distancing")
                policy_memory["mild_distancing"] = {"beta_scale": 0.78, "gamma_scale": 1.0, "compliance": effective_compliance}
            elif action == 2:
                effective_compliance = sum(age_compliance[i] * age_fractions[i] for i in range(num_cohorts)) * 0.82
                beta *= (0.42 * effective_compliance + 0.08)
                beta = max(0.0, min(beta, 1.0))
                active_policies.append("strict_distancing")
                policy_memory["strict_distancing"] = {"beta_scale": 0.42, "gamma_scale": 1.0, "compliance": effective_compliance}
            elif action == 3:
                gamma = min(gamma * 1.5, 1.0)
                beta *= 0.85
                active_policies.append("quarantine")
                policy_memory["quarantine"] = {"beta_scale": 0.85, "gamma_scale": 1.5, "compliance": 0.78}
            elif action == 4:
                beta *= 0.52
                gamma = min(gamma * 1.30, 1.0)
                active_policies.append("combined_response")
                policy_memory["combined_response"] = {"beta_scale": 0.52, "gamma_scale": 1.30, "compliance": 0.70}
            elif action == 5:
                # Multi-dose age-priority vaccination
                base_vax_rate_d1 = 0.030
                base_vax_rate_d2 = 0.020
                base_vax_rate_boost = 0.015

                if S > 8000:
                    base_vax_rate_d1 = 0.040
                elif S > 5000:
                    base_vax_rate_d1 = 0.035
                elif S > 2000:
                    base_vax_rate_d1 = 0.030
                elif S > 500:
                    base_vax_rate_d1 = 0.022
                elif S < 200:
                    base_vax_rate_d1 = 0.010

                vaccinated_total = 0
                priority_order = [6, 5, 4, 3, 2, 1, 0]
                for cohort_idx in priority_order:
                    # Dose 1
                    d1_rate = min(base_vax_rate_d1 * (1.0 - vax_coverage_dose1[cohort_idx]), 1.0)
                    if d1_rate > 0:
                        d1_count = int(rng.binomial(n=int(age_S[cohort_idx]), p=d1_rate))
                        d1_count = max(0, min(d1_count, age_S[cohort_idx]))
                        partial_immune_d1 = int(d1_count * vax_efficacy_dose1[cohort_idx])
                        partial_not_immune = d1_count - partial_immune_d1
                        age_S[cohort_idx] -= d1_count
                        age_R[cohort_idx] += partial_immune_d1
                        age_S[cohort_idx] += partial_not_immune
                        vax_coverage_dose1[cohort_idx] = min(vax_coverage_dose1[cohort_idx] + d1_rate * 0.6, 1.0)
                        vaccinated_total += partial_immune_d1

                    # Dose 2 (for those already dose-1 vaccinated)
                    d2_eligible = int(age_S[cohort_idx] * vax_coverage_dose1[cohort_idx] * 0.5)
                    d2_rate = min(base_vax_rate_d2 * (1.0 - vax_coverage_dose2[cohort_idx]), 1.0)
                    if d2_eligible > 0 and d2_rate > 0:
                        d2_count = int(rng.binomial(n=d2_eligible, p=d2_rate))
                        additional_immunity = int(d2_count * (vax_efficacy_dose2[cohort_idx] - vax_efficacy_dose1[cohort_idx]))
                        additional_immunity = max(0, min(additional_immunity, age_S[cohort_idx]))
                        age_S[cohort_idx] -= additional_immunity
                        age_R[cohort_idx] += additional_immunity
                        vax_coverage_dose2[cohort_idx] = min(vax_coverage_dose2[cohort_idx] + d2_rate * 0.4, 1.0)
                        vaccinated_total += additional_immunity

                    # Booster (for R compartment, waned immunity)
                    boost_eligible = int(age_R[cohort_idx] * 0.3)
                    boost_rate = min(base_vax_rate_boost * (1.0 - vax_coverage_booster[cohort_idx]), 1.0)
                    if boost_eligible > 0 and boost_rate > 0:
                        boost_count = int(rng.binomial(n=boost_eligible, p=boost_rate))
                        vax_coverage_booster[cohort_idx] = min(vax_coverage_booster[cohort_idx] + boost_rate * 0.3, 1.0)

                S = sum(age_S)
                R = sum(age_R)
                next_state["S"] = S
                next_state["R"] = R
                active_policies.append("vaccination")
                policy_memory["vaccination"] = {"vaccinated": vaccinated_total, "d1_coverage": vax_coverage_dose1[:], "d2_coverage": vax_coverage_dose2[:]}
            elif action == 6:
                beta *= 0.28
                gamma = min(gamma * 2.0, 1.0)
                active_policies.append("targeted_isolation")
                policy_memory["targeted_isolation"] = {"beta_scale": 0.28, "gamma_scale": 2.0, "compliance": 0.82}
            elif action == 7:
                beta *= 0.58
                gamma = min(gamma * 1.35, 1.0)
                active_policies.append("test_and_trace")
                policy_memory["test_and_trace"] = {"beta_scale": 0.58, "gamma_scale": 1.35, "compliance": 0.80}
            elif action == 8:
                beta *= 0.48
                active_policies.append("closure")
                policy_memory["closure"] = {"beta_scale": 0.48, "gamma_scale": 1.0, "compliance": 0.75}
            elif action == 9:
                gamma = min(gamma * 2.2, 1.0)
                beta *= 0.95
                active_policies.append("healthcare_surge")
                policy_memory["healthcare_surge"] = {"beta_scale": 0.95, "gamma_scale": 2.2, "compliance": 0.97}
            elif action == 10:
                beta *= 0.62
                active_policies.append("travel_restriction")
                policy_memory["travel_restriction"] = {"beta_scale": 0.62, "gamma_scale": 1.0, "compliance": 0.90}
            elif action == 11:
                beta *= 0.68
                gamma = min(gamma * 1.06, 1.0)
                active_policies.append("mask_mandate")
                policy_memory["mask_mandate"] = {"beta_scale": 0.68, "gamma_scale": 1.06, "compliance": 0.84}
            elif action == 12:
                # School closure with age-specific effects
                for cohort_idx in [0, 1, 2]:
                    age_susceptibility[cohort_idx] *= 0.50
                    age_contact_rate[cohort_idx] *= 0.55
                beta *= 0.82
                active_policies.append("school_closure")
                policy_memory["school_closure"] = {"beta_scale": 0.82, "gamma_scale": 1.0, "compliance": 0.97}
            elif action == 13:
                # Remote work mandate
                age_susceptibility[3] *= 0.62
                age_susceptibility[4] *= 0.68
                age_contact_rate[3] *= 0.65
                beta *= 0.75
                active_policies.append("remote_work")
                policy_memory["remote_work"] = {"beta_scale": 0.75, "gamma_scale": 1.0, "compliance": 0.76}
            elif action == 14:
                # Ventilation upgrade (reduces airborne transmission)
                beta *= 0.72
                active_policies.append("ventilation_upgrade")
                policy_memory["ventilation_upgrade"] = {"beta_scale": 0.72, "gamma_scale": 1.0, "compliance": 1.0}
            elif action == 15:
                # Antiviral treatment rollout
                gamma = min(gamma * 1.6, 1.0)
                for cohort_idx in [4, 5, 6]:
                    age_recovery_mod[cohort_idx] = min(age_recovery_mod[cohort_idx] * 1.3, 1.5)
                active_policies.append("antiviral_treatment")
                policy_memory["antiviral_treatment"] = {"beta_scale": 1.0, "gamma_scale": 1.6, "compliance": 0.88}

        # --- Policy interaction synergy/fatigue with non-linear scaling ---
        if len(active_policies) > 1:
            fatigue_curve = {2: 0.97, 3: 0.92, 4: 0.87, 5: 0.80, 6: 0.73, 7: 0.67}
            num_policies = min(len(active_policies), 7)
            policy_fatigue_factor = fatigue_curve.get(num_policies, 0.67)

            synergy_bonus = 1.0
            synergy_count = 0
            for i in range(len(active_policies)):
                for j in range(i + 1, len(active_policies)):
                    pair = (active_policies[i], active_policies[j])
                    rev_pair = (active_policies[j], active_policies[i])
                    if pair in policy_interaction_matrix:
                        synergy_bonus *= policy_interaction_matrix[pair]
                        synergy_count += 1
                    elif rev_pair in policy_interaction_matrix:
                        synergy_bonus *= policy_interaction_matrix[rev_pair]
                        synergy_count += 1

            # Diminishing returns on synergy
            if synergy_count > 3:
                synergy_bonus = max(synergy_bonus, 0.65)
            else:
                synergy_bonus = max(synergy_bonus, 0.72)

            synergy_bonus = min(synergy_bonus, 1.0)
            combined_scale = policy_fatigue_factor * synergy_bonus
            beta = min(beta / max(combined_scale, 0.38), 1.0)

        beta = max(0.0, min(beta, 1.0))
        gamma = max(0.0, min(gamma, 1.0))

        # --- Epidemic metrics ---
        R0 = beta / max(gamma, 1e-9)
        susceptible_fraction = S / max(N, 1)
        Rt = R0 * susceptible_fraction
        infected_fraction = I / max(N, 1)
        recovered_fraction = R / max(N, 1)

        herd_immunity_threshold = 1.0 - (1.0 / max(R0, 1e-9))
        herd_immunity_threshold = max(0.0, min(herd_immunity_threshold, 1.0))

        if Rt > 1.2:
            epidemic_phase = "rapid_growth"
        elif Rt > 1.0:
            epidemic_phase = "growth"
        elif Rt > 0.85:
            epidemic_phase = "endemic"
        elif Rt > 0.6:
            epidemic_phase = "slow_decline"
        else:
            epidemic_phase = "decline"

        # --- Multi-layer waning immunity with cross-immunity ---
        waning_base_rate = 0.003
        memory_bcell_protection = 0.0
        tcell_protection = 0.0
        cross_immunity_factor = 0.0

        if R > 0:
            # Waning rate based on epidemic pressure and phase
            if epidemic_phase == "rapid_growth":
                waning_base_rate = 0.018
            elif epidemic_phase == "growth":
                waning_base_rate = 0.013
            elif epidemic_phase == "endemic":
                waning_base_rate = 0.008
            elif epidemic_phase == "slow_decline":
                waning_base_rate = 0.004
            else:
                waning_base_rate = 0.002

            # Infection pressure modification
            if infected_fraction > 0.40:
                waning_base_rate *= 1.55
            elif infected_fraction > 0.30:
                waning_base_rate *= 1.38
            elif infected_fraction > 0.20:
                waning_base_rate *= 1.22
            elif infected_fraction > 0.10:
                waning_base_rate *= 1.10
            elif infected_fraction > 0.05:
                waning_base_rate *= 1.04

            # B-cell memory (humoral immunity)
            if recovered_fraction > 0.65:
                memory_bcell_protection = 0.48
            elif recovered_fraction > 0.50:
                memory_bcell_protection = 0.40
            elif recovered_fraction > 0.35:
                memory_bcell_protection = 0.31
            elif recovered_fraction > 0.20:
                memory_bcell_protection = 0.22
            elif recovered_fraction > 0.10:
                memory_bcell_protection = 0.13
            elif recovered_fraction > 0.05:
                memory_bcell_protection = 0.07
            else:
                memory_bcell_protection = 0.03

            # T-cell immunity
            if recovered_fraction > 0.45:
                tcell_protection = 0.24
            elif recovered_fraction > 0.25:
                tcell_protection = 0.17
            elif recovered_fraction > 0.12:
                tcell_protection = 0.11
            elif recovered_fraction > 0.05:
                tcell_protection = 0.06
            else:
                tcell_protection = 0.02

            # Cross-immunity from prior variants/infections
            if len(active_policies) > 0 and "vaccination" in active_policies:
                cross_immunity_factor = 0.12
            elif recovered_fraction > 0.30:
                cross_immunity_factor = 0.08
            else:
                cross_immunity_factor = 0.03

            combined_immune_protection = 1.0 - (1.0 - memory_bcell_protection) * (1.0 - tcell_protection) * (1.0 - cross_immunity_factor)
            combined_immune_protection = max(0.0, min(combined_immune_protection, 0.80))

            waning_prob = 1.0 - np.exp(-waning_base_rate)
            waning_prob = max(0.0, min(waning_prob, 1.0))

            # Booster effect reduces waning
            avg_booster_coverage = sum(vax_coverage_booster) / num_cohorts
            if avg_booster_coverage > 0.5:
                waning_prob *= 0.60
            elif avg_booster_coverage > 0.3:
                waning_prob *= 0.72
            elif avg_booster_coverage > 0.1:
                waning_prob *= 0.85

            total_waned = 0
            for cohort_idx in range(num_cohorts):
                # Older cohorts wane faster
                cohort_waning_multiplier = 1.0 + 0.10 * cohort_idx
                # High comorbidity burden increases waning
                comorbidity_waning_boost = 1.0 + comorbidity_prevalence[cohort_idx] * 0.5
                cohort_waning = min(waning_prob * cohort_waning_multiplier * comorbidity_waning_boost, 1.0)

                waned_cohort = int(rng.binomial(n=age_R[cohort_idx], p=cohort_waning))
                waned_cohort = max(0, min(waned_cohort, age_R[cohort_idx]))

                truly_waned = int(waned_cohort * (1.0 - combined_immune_protection))
                partially_waned = waned_cohort - truly_waned

                age_S[cohort_idx] += truly_waned
                partial_susceptible = int(partially_waned * 0.28)
                age_S[cohort_idx] += partial_susceptible
                age_R[cohort_idx] -= truly_waned + partial_susceptible
                age_R[cohort_idx] = max(0, age_R[cohort_idx])

                total_waned += truly_waned

            S = sum(age_S)
            R = sum(age_R)

        # --- Socioeconomic stratification effects ---
        ses_layers = ["low", "middle", "high"]
        ses_fractions = [0.30, 0.50, 0.20]
        ses_crowding = [1.45, 1.00, 0.72]
        ses_healthcare_access = [0.55, 0.85, 0.98]
        ses_compliance = [0.62, 0.78, 0.88]

        avg_ses_crowding = sum(ses_fractions[i] * ses_crowding[i] for i in range(3))
        avg_ses_compliance = sum(ses_fractions[i] * ses_compliance[i] for i in range(3))
        avg_ses_healthcare = sum(ses_fractions[i] * ses_healthcare_access[i] for i in range(3))

        if len(active_policies) > 0:
            # Policies differentially affect SES groups
            policy_ses_equity = 1.0
            if "targeted_isolation" in active_policies or "strict_distancing" in active_policies:
                policy_ses_equity = 0.88  # harder on low-SES
            elif "remote_work" in active_policies:
                policy_ses_equity = 0.92
            avg_ses_crowding *= policy_ses_equity

        # --- Network-aware transmission with SES and mobility ---
        network_clustering_coeff = 0.0
        if N > 0:
            if N < 300:
                network_clustering_coeff = 0.72
            elif N < 1000:
                network_clustering_coeff = 0.60
            elif N < 3000:
                network_clustering_coeff = 0.50
            elif N < 10000:
                network_clustering_coeff = 0.40
            elif N < 50000:
                network_clustering_coeff = 0.28
            elif N < 200000:
                network_clustering_coeff = 0.18
            elif N < 1000000:
                network_clustering_coeff = 0.12
            else:
                network_clustering_coeff = 0.07

        # Mobility-adjusted network modifier
        avg_mobility = sum(age_fractions[i] * age_mobility[i] for i in range(num_cohorts))
        if "closure" in active_policies:
            avg_mobility *= 0.55
        elif "strict_distancing" in active_policies:
            avg_mobility *= 0.68
        elif "mild_distancing" in active_policies:
            avg_mobility *= 0.82

        network_beta_modifier = 1.0 + 0.30 * network_clustering_coeff * infected_fraction * avg_mobility * avg_ses_crowding
        network_beta_modifier = max(0.75, min(network_beta_modifier, 1.65))

        # --- Multi-pathway environmental contamination ---
        environmental_beta = 0.0
        if I > 0:
            contamination_level = I / max(N, 1)
            base_env_factor = 0.0

            # Aerosol contamination tiers
            if contamination_level > 0.55:
                base_env_factor = 0.14
            elif contamination_level > 0.45:
                base_env_factor = 0.12
            elif contamination_level > 0.35:
                base_env_factor = 0.10
            elif contamination_level > 0.25:
                base_env_factor = 0.08
            elif contamination_level > 0.15:
                base_env_factor = 0.065
            elif contamination_level > 0.10:
                base_env_factor = 0.05
            elif contamination_level > 0.06:
                base_env_factor = 0.035
            elif contamination_level > 0.03:
                base_env_factor = 0.020
            elif contamination_level > 0.01:
                base_env_factor = 0.012
            else:
                base_env_factor = 0.006

            # Ventilation quality affects contamination
            ventilation_factor = 1.0
            if "ventilation_upgrade" in active_policies:
                ventilation_factor = 0.45
            elif N > 100000:
                ventilation_factor = 0.85  # larger cities tend to have better ventilation
            elif N < 1000:
                ventilation_factor = 1.25  # small communities often have poor ventilation

            env_policy_reduction = 1.0
            for policy in active_policies:
                if policy in ["strict_distancing", "targeted_isolation", "closure"]:
                    env_policy_reduction *= 0.55
                elif policy in ["mild_distancing", "combined_response", "mask_mandate"]:
                    env_policy_reduction *= 0.78
                elif policy in ["test_and_trace", "remote_work"]:
                    env_policy_reduction *= 0.85
                elif policy in ["school_closure"]:
                    env_policy_reduction *= 0.91
                elif policy in ["ventilation_upgrade"]:
                    env_policy_reduction *= 0.52

            base_env_factor *= env_policy_reduction * ventilation_factor
            environmental_beta = beta * base_env_factor * contamination_level * network_beta_modifier

        effective_beta = min(beta * network_beta_modifier + environmental_beta, 1.0)

        # --- Seasonal and climate proxy ---
        season_modulator = 1.0
        if N > 0:
            if infected_fraction < 0.008:
                season_modulator = 1.15
            elif infected_fraction < 0.02:
                season_modulator = 1.10
            elif infected_fraction < 0.04:
                season_modulator = 1.06
            elif infected_fraction < 0.07:
                season_modulator = 1.03
            elif infected_fraction < 0.12:
                season_modulator = 1.01
            elif infected_fraction > 0.65:
                season_modulator = 0.86
            elif infected_fraction > 0.55:
                season_modulator = 0.89
            elif infected_fraction > 0.45:
                season_modulator = 0.91
            elif infected_fraction > 0.35:
                season_modulator = 0.93
            elif infected_fraction > 0.22:
                season_modulator = 0.96
            elif infected_fraction > 0.15:
                season_modulator = 0.98

            if epidemic_phase in ["rapid_growth"]:
                season_modulator *= 1.04
            elif epidemic_phase == "decline":
                season_modulator *= 0.96
            elif epidemic_phase == "slow_decline":
                season_modulator *= 0.98

            # SES crowding amplifies seasonal effect
            if avg_ses_crowding > 1.2:
                season_modulator *= 1.03
            elif avg_ses_crowding < 0.85:
                season_modulator *= 0.97

        effective_beta = min(effective_beta * season_modulator, 1.0)

        # --- Spatial heterogeneity with geography-aware hotspots ---
        hotspot_factor = 1.0
        hotspot_active = False
        hotspot_tier = 0
        hotspot_location = None

        if N > 0:
            hotspot_threshold = 0.055
            if infected_fraction > hotspot_threshold:
                hotspot_scale = infected_fraction / hotspot_threshold
                base_hotspot_prob = min(0.07 * hotspot_scale, 0.35)
                base_hotspot_prob = min(base_hotspot_prob * (1.0 + network_clustering_coeff * 0.6) * avg_ses_crowding, 0.45)

                rand_hotspot = rng.random()
                # Mega hotspot (e.g., large event, densely crowded venue)
                if rand_hotspot < base_hotspot_prob * 0.08:
                    hotspot_factor = rng.uniform(1.55, 2.10)
                    hotspot_active = True
                    hotspot_tier = 4
                    hotspot_location = "mega_venue"
                # Large hotspot
                elif rand_hotspot < base_hotspot_prob * 0.18:
                    hotspot_factor = rng.uniform(1.38, 1.68)
                    hotspot_active = True
                    hotspot_tier = 3
                    hotspot_location = "large_venue"
                # Medium hotspot
                elif rand_hotspot < base_hotspot_prob * 0.38:
                    hotspot_factor = rng.uniform(1.20, 1.45)
                    hotspot_active = True
                    hotspot_tier = 2
                    hotspot_location = "medium_venue"
                # Minor hotspot
                elif rand_hotspot < base_hotspot_prob:
                    hotspot_factor = rng.uniform(1.06, 1.25)
                    hotspot_active = True
                    hotspot_tier = 1
                    hotspot_location = "minor_venue"

                if hotspot_active:
                    for policy in active_policies:
                        if policy in ["closure", "strict_distancing", "targeted_isolation"]:
                            hotspot_factor *= 0.70
                        elif policy in ["combined_response", "test_and_trace", "quarantine"]:
                            hotspot_factor *= 0.82
                        elif policy in ["mild_distancing", "mask_mandate", "ventilation_upgrade"]:
                            hotspot_factor *= 0.90

                    # High SES communities can better control hotspots
                    if avg_ses_healthcare > 0.90:
                        hotspot_factor *= 0.88
                    elif avg_ses_healthcare < 0.65:
                        hotspot_factor *= 1.12

        effective_beta = min(effective_beta * hotspot_factor, 1.0)

        # --- Healthcare system capacity with SES-adjusted access ---
        hospital_capacity_fraction = 0.10
        icu_capacity_fraction = 0.035
        step_down_capacity_fraction = 0.18
        healthcare_overflow = False
        icu_overflow = False
        step_down_overflow = False
        healthcare_overflow_severity = 0.0
        icu_overflow_severity = 0.0
        step_down_overflow_severity = 0.0

        adjusted_hospital_capacity = hospital_capacity_fraction * avg_ses_healthcare
        adjusted_icu_capacity = icu_capacity_fraction * avg_ses_healthcare

        if infected_fraction > adjusted_hospital_capacity:
            healthcare_overflow = True
            healthcare_overflow_severity = (infected_fraction - adjusted_hospital_capacity) / max(adjusted_hospital_capacity, 1e-9)
            healthcare_overflow_severity = min(healthcare_overflow_severity, 5.0)

        if infected_fraction > adjusted_icu_capacity:
            icu_overflow = True
            icu_overflow_severity = (infected_fraction - adjusted_icu_capacity) / max(adjusted_icu_capacity, 1e-9)
            icu_overflow_severity = min(icu_overflow_severity, 6.0)

        if infected_fraction > step_down_capacity_fraction:
            step_down_overflow = True
            step_down_overflow_severity = (infected_fraction - step_down_capacity_fraction) / max(step_down_capacity_fraction, 1e-9)
            step_down_overflow_severity = min(step_down_overflow_severity, 3.0)

        # Healthcare surge specifically addresses capacity
        if "healthcare_surge" in active_policies:
            adjusted_hospital_capacity *= 1.40
            adjusted_icu_capacity *= 1.35
            if infected_fraction <= adjusted_hospital_capacity:
                healthcare_overflow = False
                healthcare_overflow_severity = 0.0
            if infected_fraction <= adjusted_icu_capacity:
                icu_overflow = False
                icu_overflow_severity = 0.0

        # --- Primary stochastic SIR transitions with contact matrix ---
        new_infections_by_age = [0] * num_cohorts
        new_recoveries_by_age = [0] * num_cohorts
        mortality_by_age = [0] * num_cohorts
        hospitalized_by_age = [0] * num_cohorts
        icu_by_age = [0] * num_cohorts

        for cohort_idx in range(num_cohorts):
            cohort_S = age_S[cohort_idx]
            cohort_I = age_I[cohort_idx]
            cohort_susc = age_susceptibility[cohort_idx]
            cohort_rec_mod = age_recovery_mod[cohort_idx]
            cohort_comorbidity = comorbidity_prevalence[cohort_idx]

            # Multi-dose vaccination reduces susceptibility
            vax_protection = (
                vax_coverage_dose1[cohort_idx] * vax_efficacy_dose1[cohort_idx] * 0.4 +
                vax_coverage_dose2[cohort_idx] * vax_efficacy_dose2[cohort_idx] * 0.45 +
                vax_coverage_booster[cohort_idx] * vax_efficacy_booster[cohort_idx] * 0.15
            )
            vax_protection = min(vax_protection, 0.95)
            effective_susc = cohort_susc * (1.0 - vax_protection)

            if I > 0 and cohort_S > 0:
                # Use contact matrix to compute force of infection
                force_of_infection = 0.0
                for source_cohort in range(num_cohorts):
                    contact_weight = contact_matrix[cohort_idx][source_cohort]
                    source_I_fraction = age_I[source_cohort] / max(N, 1)
                    force_of_infection += contact_weight * source_I_fraction

                cohort_beta = min(effective_beta * effective_susc, 1.0)
                infection_prob = 1.0 - np.exp(-cohort_beta * force_of_infection * N / max(N, 1))
                infection_prob = max(0.0, min(infection_prob, 1.0))

                # Population density with 10 tiers
                density_factor = 1.0
                if N > 1000000:
                    density_factor = 1.55
                elif N > 500000:
                    density_factor = 1.48
                elif N > 200000:
                    density_factor = 1.40
                elif N > 100000:
                    density_factor = 1.32
                elif N > 50000:
                    density_factor = 1.24
                elif N > 20000:
                    density_factor = 1.16
                elif N > 10000:
                    density_factor = 1.10
                elif N > 5000:
                    density_factor = 1.05
                elif N > 2000:
                    density_factor = 1.00
                elif N > 1000:
                    density_factor = 0.95
                elif N > 500:
                    density_factor = 0.90
                elif N > 200:
                    density_factor = 0.84
                elif N > 100:
                    density_factor = 0.80
                else:
                    density_factor = 0.75

                # Age-specific density and policy interactions
                if cohort_idx in [0, 1]:  # Infants/children
                    if "school_closure" in active_policies:
                        density_factor *= 0.62
                    else:
                        density_factor *= 1.08
                elif cohort_idx == 2:  # Teens
                    if "school_closure" in active_policies:
                        density_factor *= 0.68
                    else:
                        density_factor *= 1.12
                elif cohort_idx in [3, 4]:  # Adults/middle-aged
                    if "remote_work" in active_policies:
                        density_factor *= 0.75
                    elif "closure" in active_policies:
                        density_factor *= 0.80
                elif cohort_idx in [5, 6]:  # Seniors/elders
                    if "targeted_isolation" in active_policies:
                        density_factor *= 0.55
                    elif "quarantine" in active_policies:
                        density_factor *= 0.65

                # SES-based crowding effect
                ses_density_adjustment = 0.85 + 0.30 * avg_ses_crowding
                density_factor *= ses_density_adjustment
                density_factor = max(0.5, min(density_factor, 2.0))

                infection_prob = max(0.0, min(infection_prob * density_factor, 1.0))
                cohort_new_inf = int(rng.binomial(n=cohort_S, p=infection_prob))
                new_infections_by_age[cohort_idx] = max(0, min(cohort_new_inf, cohort_S))

            if cohort_I > 0:
                # Recovery with SES healthcare access
                recovery_variation = rng.uniform(0.65, 1.35)
                healthcare_access_mod = avg_ses_healthcare * 0.4 + 0.6
                adjusted_gamma = min(gamma * recovery_variation * cohort_rec_mod * healthcare_access_mod, 1.0)

                # Policy boosts to recovery
                for policy in active_policies:
                    if policy == "quarantine":
                        adjusted_gamma = min(adjusted_gamma * 1.15, 1.0)
                    elif policy == "targeted_isolation":
                        adjusted_gamma = min(adjusted_gamma * 1.18, 1.0)
                    elif policy == "healthcare_surge":
                        adjusted_gamma = min(adjusted_gamma * 1.28, 1.0)
                    elif policy == "test_and_trace":
                        adjusted_gamma = min(adjusted_gamma * 1.12, 1.0)
                    elif policy == "combined_response":
                        adjusted_gamma = min(adjusted_gamma * 1.10, 1.0)
                    elif policy == "antiviral_treatment":
                        adjusted_gamma = min(adjusted_gamma * 1.20, 1.0)

                # Healthcare overflow penalty
                if icu_overflow:
                    icu_penalty = max(0.40, 1.0 - 0.20 * icu_overflow_severity)
                    adjusted_gamma *= icu_penalty
                elif healthcare_overflow:
                    overflow_penalty = max(0.50, 1.0 - 0.15 * healthcare_overflow_severity)
                    adjusted_gamma *= overflow_penalty
                elif step_down_overflow:
                    adjusted_gamma *= max(0.85, 1.0 - 0.06 * step_down_overflow_severity)
                elif infected_fraction > 0.45:
                    adjusted_gamma *= 0.76
                elif infected_fraction > 0.35:
                    adjusted_gamma *= 0.84
                elif infected_fraction > 0.25:
                    adjusted_gamma *= 0.90
                elif infected_fraction > 0.15:
                    adjusted_gamma *= 0.94
                elif infected_fraction > 0.08:
                    adjusted_gamma *= 0.97

                # Age cohort penalties (older = slower recovery without surge)
                if cohort_idx == 6 and "healthcare_surge" not in active_policies:
                    adjusted_gamma *= 0.70
                elif cohort_idx == 5 and "healthcare_surge" not in active_policies:
                    adjusted_gamma *= 0.80
                elif cohort_idx == 4 and "healthcare_surge" not in active_policies:
                    adjusted_gamma *= 0.88

                # Comorbidity slows recovery
                comorbidity_recovery_penalty = 1.0 - cohort_comorbidity * 0.30
                adjusted_gamma *= max(comorbidity_recovery_penalty, 0.50)

                adjusted_gamma = max(0.0, min(adjusted_gamma, 1.0))
                recovery_prob = 1.0 - np.exp(-adjusted_gamma)
                recovery_prob = max(0.0, min(recovery_prob, 1.0))
                cohort_new_rec = int(rng.binomial(n=cohort_I, p=recovery_prob))
                new_recoveries_by_age[cohort_idx] = max(0, min(cohort_new_rec, cohort_I))

                # --- Mortality with comorbidity and SES ---
                mort_prob = age_mortality_risk[cohort_idx]

                # Comorbidity multiplies mortality
                effective_mort_prob = mort_prob * (
                    (1.0 - cohort_comorbidity) +
                    cohort_comorbidity * comorbidity_mortality_multiplier[cohort_idx]
                )

                if icu_overflow:
                    effective_mort_prob = min(effective_mort_prob * (1.0 + 0.55 * icu_overflow_severity), 0.92)
                elif healthcare_overflow:
                    effective_mort_prob = min(effective_mort_prob * (1.0 + 0.28 * healthcare_overflow_severity), 0.65)

                # SES healthcare access reduces mortality
                effective_mort_prob *= (1.0 - avg_ses_healthcare * 0.35)

                if "healthcare_surge" in active_policies:
                    effective_mort_prob *= 0.65
                if "antiviral_treatment" in active_policies:
                    effective_mort_prob *= 0.72

                effective_mort_prob = max(0.0, min(effective_mort_prob, 1.0))
                cohort_deaths = int(rng.binomial(n=cohort_I, p=effective_mort_prob))
                cohort_deaths = max(0, min(cohort_deaths, cohort_I - new_recoveries_by_age[cohort_idx]))
                mortality_by_age[cohort_idx] = cohort_deaths

                # --- Hospitalization with comorbidity ---
                hosp_prob = age_hospitalization_risk[cohort_idx]
                effective_hosp_prob = hosp_prob * (
                    (1.0 - cohort_comorbidity) +
                    cohort_comorbidity * comorbidity_hospitalization_multiplier[cohort_idx]
                )

                if icu_overflow:
                    effective_hosp_prob = min(effective_hosp_prob * (1.0 + 0.30 * icu_overflow_severity), 1.0)
                elif healthcare_overflow:
                    effective_hosp_prob = min(effective_hosp_prob * (1.0 + 0.20 * healthcare_overflow_severity), 1.0)

                if "healthcare_surge" in active_policies:
                    effective_hosp_prob *= 0.80

                effective_hosp_prob = max(0.0, min(effective_hosp_prob * avg_ses_healthcare, 1.0))
                hosp_count = int(rng.binomial(n=cohort_I, p=effective_hosp_prob))
                hospitalized_by_age[cohort_idx] = max(0, hosp_count)

                icu_ratio = [0.08, 0.12, 0.18, 0.25, 0.32, 0.42, 0.55][cohort_idx]
                icu_prob = min(icu_ratio * effective_hosp_prob, 1.0)
                icu_count_cohort = int(rng.binomial(n=cohort_I, p=icu_prob))
                icu_by_age[cohort_idx] = max(0, icu_count_cohort)

        new_infections = sum(new_infections_by_age)
        new_recoveries = sum(new_recoveries_by_age)
        total_deaths = sum(mortality_by_age)
        total_hospitalized = sum(hospitalized_by_age)
        total_icu = sum(icu_by_age)

        new_infections = max(0, min(new_infections, S))
        new_recoveries = max(0, min(new_recoveries, I))
        total_deaths = max(0, min(total_deaths, I - new_recoveries))

        # --- Multi-tier superspreader model with hotspot and SES ---
        if I > 0 and S > 0:
            mega_superspreader_prob = 0.0025
            major_superspreader_prob = 0.010
            moderate_superspreader_prob = 0.028
            minor_superspreader_prob = 0.060

            # Hotspot amplification
            if hotspot_tier >= 4:
                mega_superspreader_prob *= 3.0
                major_superspreader_prob *= 2.0
                moderate_superspreader_prob *= 1.5
            elif hotspot_tier == 3:
                mega_superspreader_prob *= 2.0
                major_superspreader_prob *= 1.6
            elif hotspot_tier == 2:
                mega_superspreader_prob *= 1.5
                major_superspreader_prob *= 1.3
            elif hotspot_tier == 1:
                mega_superspreader_prob *= 1.2
                minor_superspreader_prob *= 1.3

            # Low SES areas have higher superspreader risk
            if avg_ses_crowding > 1.3:
                mega_superspreader_prob *= 1.3
                major_superspreader_prob *= 1.2

            rand_val = rng.random()
            if rand_val < mega_superspreader_prob:
                lam_val = max(0.1, beta * (18 if hotspot_active else 13) * avg_ses_crowding)
                extra = int(rng.poisson(lam=lam_val))
                extra = max(0, min(extra, S - new_infections))
                new_infections += extra
            elif rand_val < major_superspreader_prob:
                lam_val = max(0.1, beta * (10 if hotspot_active else 7.5) * avg_ses_crowding)
                extra = int(rng.poisson(lam=lam_val))
                extra = max(0, min(extra, S - new_infections))
                new_infections += extra
            elif rand_val < moderate_superspreader_prob:
                lam_val = max(0.1, beta * (5.5 if hotspot_active else 4.0))
                extra = int(rng.poisson(lam=lam_val))
                extra = max(0, min(extra, S - new_infections))
                new_infections += extra
            elif rand_val < minor_superspreader_prob:
                lam_val = max(0.1, beta * (2.5 if hotspot_active else 1.8))
                extra = int(rng.poisson(lam=lam_val))
                extra = max(0, min(extra, S - new_infections))
                new_infections += extra

            # Cluster-based transmission
            cluster_thresholds = [
                (300, 0.10, 0.20),
                (700, 0.08, 0.17),
                (1500, 0.06, 0.14),
                (3000, 0.04, 0.11),
                (7000, 0.028, 0.08),
                (15000, 0.016, 0.055),
                (30000, 0.009, 0.038),
                (70000, 0.005, 0.022),
            ]

            for (threshold, cluster_prob, lam_scale) in cluster_thresholds:
                if N < threshold and I > 0:
                    adjusted_cluster_prob = cluster_prob * avg_ses_crowding
                    if rng.random() < adjusted_cluster_prob:
                        cluster_size = int(rng.poisson(lam=max(1, I * lam_scale)))
                        cluster_infections = max(0, min(cluster_size, S - new_infections))
                        new_infections += cluster_infections
                    break

        new_infections = max(0, min(new_infections, S))
        new_recoveries = max(0, min(new_recoveries, max(0, I - total_deaths)))

        # --- Adaptive multi-stage sub-stepping ---
        if Rt > 6.0:
            sub_steps = 14
        elif Rt > 5.0:
            sub_steps = 12
        elif Rt > 4.0:
            sub_steps = 10
        elif Rt > 3.0:
            sub_steps = 8
        elif Rt > 2.5:
            sub_steps = 7
        elif Rt > 2.0:
            sub_steps = 6
        elif Rt > 1.5:
            sub_steps = 5
        elif Rt > 1.0:
            sub_steps = 4
        else:
            sub_steps = 3

        if hotspot_tier >= 3:
            sub_steps = min(sub_steps + 3, 15)
        elif hotspot_tier == 2:
            sub_steps = min(sub_steps + 2, 15)
        elif hotspot_tier == 1:
            sub_steps = min(sub_steps + 1, 15)

        # High SES compliance enables more refined stepping
        if avg_ses_compliance > 0.85:
            sub_steps = min(sub_steps + 1, 15)

        sub_infections = 0
        sub_recoveries = 0
        sub_deaths = 0

        remaining_S = max(0, S - new_infections)
        remaining_I = max(0, I + new_infections - new_recoveries - total_deaths)
        remaining_R = R + new_recoveries

        extinction_step = -1
        resurgence_detected = False
        peak_detected = False
        post_peak_steps = 0
        peak_I = remaining_I
        behavioral_fatigue = 1.0

        for step in range(sub_steps):
            if remaining_S <= 0 or remaining_I <= 0:
                extinction_step = step
                break

            infected_frac_now = remaining_I / max(N, 1)
            recovered_frac_now = remaining_R / max(N, 1)

            if remaining_I > peak_I:
                peak_I = remaining_I
                post_peak_steps = 0
            elif remaining_I < peak_I * 0.88 and not peak_detected:
                peak_detected = True
                post_peak_steps = 0
            elif peak_detected:
                post_peak_steps += 1

            # Behavioral fatigue increases over time post-peak
            if peak_detected and post_peak_steps > 0:
                behavioral_fatigue = min(1.0 + 0.04 * post_peak_steps, 1.30)

            # Multi-tier dynamic behavioral response
            sub_beta_scale = 1.0
            if infected_frac_now > 0.70:
                sub_beta_scale = 0.52
            elif infected_frac_now > 0.60:
                sub_beta_scale = 0.58
            elif infected_frac_now > 0.50:
                sub_beta_scale = 0.64
            elif infected_frac_now > 0.40:
                sub_beta_scale = 0.70
            elif infected_frac_now > 0.32:
                sub_beta_scale = 0.76
            elif infected_frac_now > 0.24:
                sub_beta_scale = 0.82
            elif infected_frac_now > 0.16:
                sub_beta_scale = 0.87
            elif infected_frac_now > 0.10:
                sub_beta_scale = 0.92
            elif infected_frac_now > 0.05:
                sub_beta_scale = 0.96
            elif infected_frac_now > 0.02:
                sub_beta_scale = 0.98

            # Post-peak behavioral relaxation with fatigue
            if peak_detected:
                sub_beta_scale = min(sub_beta_scale * behavioral_fatigue, 1.0)

            # SES compliance modulates behavioral response
            sub_beta_scale = sub_beta_scale * (1.0 - avg_ses_compliance * 0.08) + avg_ses_compliance * 0.08

            # Resurgence detection
            if step > 1 and infected_frac_now > 0.035 and not resurgence_detected:
                if remaining_I > (I + new_infections) * 1.22:
                    resurgence_detected = True
                    sub_beta_scale *= 0.82

            for policy in active_policies:
                if policy == "strict_distancing":
                    sub_beta_scale *= 0.88
                elif policy == "mild_distancing":
                    sub_beta_scale *= 0.95
                elif policy == "closure":
                    sub_beta_scale *= 0.90
                elif policy == "mask_mandate":
                    sub_beta_scale *= 0.93
                elif policy == "travel_restriction":
                    sub_beta_scale *= 0.96
                elif policy == "school_closure":
                    sub_beta_scale *= 0.94
                elif policy == "remote_work":
                    sub_beta_scale *= 0.95
                elif policy == "ventilation_upgrade":
                    sub_beta_scale *= 0.92
                elif policy == "targeted_isolation":
                    sub_beta_scale *= 0.86

            # Herd immunity proximity effect
            if recovered_frac_now > herd_immunity_threshold * 0.95:
                sub_beta_scale *= 0.80
            elif recovered_frac_now > herd_immunity_threshold * 0.85:
                sub_beta_scale *= 0.88
            elif recovered_frac_now > herd_immunity_threshold * 0.70:
                sub_beta_scale *= 0.94

            sub_beta = (effective_beta * sub_beta_scale) / sub_steps
            sub_gamma = gamma / sub_steps

            # Healthcare capacity in sub-steps
            if icu_overflow:
                sub_gamma *= max(0.50, 1.0 - 0.17 * icu_overflow_severity)
            elif healthcare_overflow:
                sub_gamma *= max(0.62, 1.0 - 0.12 * healthcare_overflow_severity)

            # SES healthcare access modifies sub-step gamma
            sub_gamma *= (0.7 + 0.3 * avg_ses_healthcare)

            sub_infection_prob = 1.0 - np.exp(-sub_beta * remaining_I / max(N, 1))
            sub_infection_prob = max(0.0, min(sub_infection_prob, 1.0))

            sub_recovery_prob = 1.0 - np.exp(-sub_gamma)
            sub_recovery_prob = max(0.0, min(sub_recovery_prob, 1.0))

            step_infections = int(rng.binomial(n=remaining_S, p=sub_infection_prob))
            step_recoveries = int(rng.binomial(n=remaining_I, p=sub_recovery_prob))

            step_infections = max(0, min(step_infections, remaining_S))
            step_recoveries = max(0, min(step_recoveries, remaining_I))

            # Step-dependent micro-noise with tightening variance
            if step == 0:
                noise_lo, noise_hi = 0.88, 1.12
                rec_noise_lo, rec_noise_hi = 0.90, 1.10
            elif step < 2:
                noise_lo, noise_hi = 0.92, 1.08
                rec_noise_lo, rec_noise_hi = 0.93, 1.07
            elif step < 5:
                noise_lo, noise_hi = 0.95, 1.05
                rec_noise_lo, rec_noise_hi = 0.96, 1.04
            else:
                noise_lo, noise_hi = 0.97, 1.03
                rec_noise_lo, rec_noise_hi = 0.98, 1.02

            noise_factor = rng.uniform(noise_lo, noise_hi)
            step_infections = max(0, min(int(step_infections * noise_factor), remaining_S))

            recovery_noise = rng.uniform(rec_noise_lo, rec_noise_hi)
            step_recoveries = max(0, min(int(step_recoveries * recovery_noise), remaining_I))

            # Sub-step mortality
            if icu_overflow:
                sub_mort_rate = 0.0018
            elif healthcare_overflow:
                sub_mort_rate = 0.0010
            else:
                sub_mort_rate = 0.0004

            sub_mort_rate *= (1.0 - avg_ses_healthcare * 0.40)
            step_deaths = int(rng.binomial(n=remaining_I, p=sub_mort_rate))
            step_deaths = max(0, min(step_deaths, remaining_I - step_recoveries))

            sub_infections += step_infections
            sub_recoveries += step_recoveries
            sub_deaths += step_deaths

            remaining_S -= step_infections
            remaining_I = max(0, remaining_I + step_infections - step_recoveries - step_deaths)
            remaining_R += step_recoveries

            if remaining_I == 0:
                extinction_step = step
                break

        if extinction_step >= 0:
            sub_recoveries = min(sub_recoveries, I + new_infections - total_deaths)

        # --- Dynamic blending with phase and SES awareness ---
        if Rt > 5.5:
            blend = 0.25
        elif Rt > 4.5:
            blend = 0.32
        elif Rt > 3.5:
            blend = 0.40
        elif Rt > 2.5:
            blend = 0.48
        elif Rt > 2.0:
            blend = 0.55
        elif Rt > 1.5:
            blend = 0.62
        elif Rt > 1.0:
            blend = 0.68
        elif Rt > 0.5:
            blend = 0.78
        else:
            blend = 0.88

        if peak_detected:
            blend = min(blend * (1.0 + 0.05 * min(post_peak_steps, 3)), 0.92)

        # SES compliance improves blending accuracy
        blend = min(blend + avg_ses_compliance * 0.03, 0.95)

        total_new_infections = int(blend * new_infections + (1 - blend) * sub_infections)
        total_new_recoveries = int(blend * new_recoveries + (1 - blend) * sub_recoveries)
        combined_deaths = int(blend * total_deaths + (1 - blend) * sub_deaths)

        total_new_infections = max(0, min(total_new_infections, S))
        total_new_recoveries = max(0, min(total_new_recoveries, I + total_new_infections))
        combined_deaths = max(0, min(combined_deaths, I + total_new_infections - total_new_recoveries))

        if I + total_new_infections - total_new_recoveries - combined_deaths < 0:
            combined_deaths = 0
            total_new_recoveries = I + total_new_infections

        # --- Multi-variant mutation emergence with SES and immune escape ---
        variant_infection_bonus = 0
        active_variants = []
        variant_severity_bonus_mortality = 0.0

        if I > 12:
            base_variant_prob = 0.0030 * (I / max(N, 1))

            if infected_fraction > 0.35:
                base_variant_prob *= 1.90
            elif infected_fraction > 0.25:
                base_variant_prob *= 1.62
            elif infected_fraction > 0.15:
                base_variant_prob *= 1.35
            elif infected_fraction > 0.08:
                base_variant_prob *= 1.15
            elif infected_fraction > 0.04:
                base_variant_prob *= 1.05

            # High SES increases surveillance (catches variants earlier, less spread)
            base_variant_prob *= (1.0 - avg_ses_healthcare * 0.12)

            immune_escape_multiplier = 1.0 + recovered_fraction * 1.0 + (len(active_policies) > 0) * 0.18

            # Mild variant
            if rng.random() < base_variant_prob * immune_escape_multiplier:
                mild_boost = rng.uniform(0.020, 0.12)
                v_infections = int(rng.binomial(
                    n=max(0, S - total_new_infections),
                    p=min(mild_boost * I / max(N, 1), 1.0)
                ))
                variant_infection_bonus += max(0, v_infections)
                active_variants.append("mild")

            # Moderate variant
            if I > 50 and rng.random() < base_variant_prob * 0.42 * immune_escape_multiplier:
                moderate_boost = rng.uniform(0.055, 0.18)
                v_infections = int(rng.binomial(
                    n=max(0, S - total_new_infections - variant_infection_bonus),
                    p=min(moderate_boost * I / max(N, 1), 1.0)
                ))
                variant_infection_bonus += max(0, v_infections)
                active_variants.append("moderate")

            # Severe immune escape variant
            if I > 100 and rng.random() < base_variant_prob * 0.20 * immune_escape_multiplier:
                severe_boost = rng.uniform(0.09, 0.30)
                r_escape_fraction = 0.12 if "vaccination" in active_policies else 0.22
                # Low SES has higher escape rate
                r_escape_fraction *= (1.0 + (1.0 - avg_ses_healthcare) * 0.25)
                escaped_recovered = int(R * r_escape_fraction * rng.uniform(0.4, 1.0))
                escaped_recovered = max(0, min(escaped_recovered, R))
                v_infections = int(rng.binomial(
                    n=max(0, S - total_new_infections - variant_infection_bonus),
                    p=min(severe_boost * I / max(N, 1), 1.0)
                ))
                v_infections += escaped_recovered
                variant_infection_bonus += max(0, v_infections)
                active_variants.append("severe_escape")
                variant_severity_bonus_mortality = 0.002

            # Hyper-transmissible variant
            if I > 250 and rng.random() < base_variant_prob * 0.08 * immune_escape_multiplier:
                hyper_boost = rng.uniform(0.16, 0.50)
                v_infections = int(rng.binomial(
                    n=max(0, S - total_new_infections - variant_infection_bonus),
                    p=min(hyper_boost * I / max(N, 1), 1.0)
                ))
                variant_infection_bonus += max(0, v_infections)
                active_variants.append("hyper_transmissible")

            # Novel virulent variant (very rare, high mortality)
            if I > 500 and rng.random() < base_variant_prob * 0.025 * immune_escape_multiplier:
                virulent_boost = rng.uniform(0.08, 0.22)
                v_infections = int(rng.binomial(
                    n=max(0, S - total_new_infections - variant_infection_bonus),
                    p=min(virulent_boost * I / max(N, 1), 1.0)
                ))
                variant_infection_bonus += max(0, v_infections)
                active_variants.append("virulent")
                variant_severity_bonus_mortality += 0.005

        total_new_infections += variant_infection_bonus
        total_new_infections = max(0, min(total_new_infections, S))

        # Apply variant mortality bonus
        if variant_severity_bonus_mortality > 0 and I > 0:
            extra_deaths_from_variant = int(rng.binomial(n=I, p=min(variant_severity_bonus_mortality, 1.0)))
            extra_deaths_from_variant = max(0, min(extra_deaths_from_variant, I - total_new_recoveries - combined_deaths))
            combined_deaths += extra_deaths_from_variant

        total_new_recoveries = max(0, min(total_new_recoveries, I + total_new_infections - combined_deaths))

        # --- Spontaneous recovery floor ---
        if I > 0 and total_new_recoveries == 0:
            if icu_overflow:
                floor_recovery_prob = 0.004
            elif healthcare_overflow:
                floor_recovery_prob = 0.006
            elif "healthcare_surge" in active_policies:
                floor_recovery_prob = 0.018
            elif "antiviral_treatment" in active_policies:
                floor_recovery_prob = 0.022
            else:
                floor_recovery_prob = 0.012
            floor_recovery_prob *= avg_ses_healthcare
            floor_recoveries = int(rng.binomial(n=I, p=min(floor_recovery_prob, 1.0)))
            total_new_recoveries = max(0, min(floor_recoveries, I - combined_deaths))

        # --- Multi-pathway reinfection ---
        reinfection_count = 0
        if R > 0 and I > 0:
            base_reinfection_prob = 0.0008 * (I / max(N, 1))

            if Rt > 4.0:
                base_reinfection_prob *= 2.0
            elif Rt > 3.0:
                base_reinfection_prob *= 1.85
            elif Rt > 2.0:
                base_reinfection_prob *= 1.55
            elif Rt > 1.5:
                base_reinfection_prob *= 1.32

            if "severe_escape" in active_variants:
                base_reinfection_prob *= 1.70
            elif "virulent" in active_variants:
                base_reinfection_prob *= 1.55
            elif "hyper_transmissible" in active_variants:
                base_reinfection_prob *= 1.40
            elif "moderate" in active_variants:
                base_reinfection_prob *= 1.28
            elif "mild" in active_variants:
                base_reinfection_prob *= 1.12

            if variant_infection_bonus > 0:
                base_reinfection_prob *= 1.20

            if recovered_fraction > 0.60:
                base_reinfection_prob *= 1.30
            elif recovered_fraction > 0.45:
                base_reinfection_prob *= 1.20
            elif recovered_fraction > 0.30:
                base_reinfection_prob *= 1.12
            elif recovered_fraction > 0.15:
                base_reinfection_prob *= 1.06

            # Booster vaccination reduces reinfection
            avg_booster_coverage = sum(vax_coverage_booster) / num_cohorts
            base_reinfection_prob *= (1.0 - avg_booster_coverage * 0.50)

            # SES healthcare access reduces reinfection risk
            base_reinfection_prob *= (1.0 - avg_ses_healthcare * 0.20)

            base_reinfection_prob = max(0.0, min(base_reinfection_prob, 1.0))
            reinfections = int(rng.binomial(n=R, p=base_reinfection_prob))
            reinfection_count = max(0, min(reinfections, R))
            R = max(0, R - reinfection_count)

        # --- Update state with population conservation ---
        new_S = S - total_new_infections
        effective_I_start = max(0, I - combined_deaths)
        new_I = effective_I_start + total_new_infections - total_new_recoveries + reinfection_count
        new_R = R + total_new_recoveries

        new_S = max(0, new_S)
        new_I = max(0, new_I)
        new_R = max(0, new_R)

        next_state["S"] = new_S
        next_state["I"] = new_I
        next_state["R"] = new_R

        effective_N = max(0, N - combined_deaths)

        # --- Population conservation with multi-pass correction ---
        for correction_pass in range(6):
            total_pop = next_state["S"] + next_state["I"] + next_state["R"]
            if total_pop == effective_N:
                break

            diff = effective_N - total_pop
            if diff > 0:
                next_state["R"] += diff
            else:
                excess = abs(diff)
                if next_state["R"] >= excess:
                    next_state["R"] -= excess
                elif next_state["R"] + next_state["I"] >= excess:
                    remaining_excess = excess - next_state["R"]
                    next_state["R"] = 0
                    next_state["I"] = max(0, next_state["I"] - remaining_excess)
                else:
                    remaining_excess = excess - next_state["R"] - next_state["I"]
                    next_state["R"] = 0
                    next_state["I"] = 0
                    next_state["S"] = max(0, next_state["S"] - remaining_excess)

        for key in ["S", "I", "R"]:
            if next_state[key] < 0:
                next_state[key] = 0

        # Maintain original N for compatibility
        total_final = next_state["S"] + next_state["I"] + next_state["R"]
        if total_final != N:
            adjustment = N - total_final
            if adjustment > 0:
                next_state["R"] = max(0, next_state["R"] + adjustment)
            else:
                excess = abs(adjustment)
                if next_state["R"] >= excess:
                    next_state["R"] -= excess
                elif next_state["I"] >= excess - next_state["R"]:
                    excess -= next_state["R"]
                    next_state["R"] = 0
                    next_state["I"] = max(0, next_state["I"] - excess)
                else:
                    excess -= next_state["R"] + next_state["I"]
                    next_state["R"] = 0
                    next_state["I"] = 0
                    next_state["S"] = max(0, next_state["S"] - excess)

        for key in ["S", "I", "R"]:
            if next_state[key] < 0:
                next_state[key] = 0

        total_check = next_state["S"] + next_state["I"] + next_state["R"]
        if total_check != N:
            next_state["R"] = max(0, next_state["R"] + (N - total_check))

        return next_state


    def evaluate(self, x):
        return 0

