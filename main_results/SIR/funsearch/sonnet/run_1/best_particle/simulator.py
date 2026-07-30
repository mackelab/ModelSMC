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

        if N == 0:
            return next_state

        beta  = parameters[0]
        gamma = parameters[1]
        seasonal_factor          = parameters[2]  if len(parameters) > 2  else 1.0
        waning_immunity_rate     = parameters[3]  if len(parameters) > 3  else 0.0
        mutation_rate            = parameters[4]  if len(parameters) > 4  else 0.0
        hospital_capacity_frac   = parameters[5]  if len(parameters) > 5  else 1.0
        vaccination_rate         = parameters[6]  if len(parameters) > 6  else 0.0
        contact_tracing_eff      = parameters[7]  if len(parameters) > 7  else 0.0

        # ------------------------------------------------------------------ #
        # SECTION 1 – Latent / exposed sub-compartment (SEIR extension)       #
        # ------------------------------------------------------------------ #
        exposed_frac      = 0.18          # fraction of I assumed still in latent window
        exposed_I         = max(0, int(I * exposed_frac))
        infectious_I      = max(0, I - exposed_I)
        incubation_rate   = 0.20          # rate at which E -> I per step

        presymptomatic_transmission_factor = 0.35
        effective_infectious = infectious_I + int(exposed_I * presymptomatic_transmission_factor)

        # ------------------------------------------------------------------ #
        # SECTION 2 – Heterogeneous contact matrix (7 age bands)              #
        # ------------------------------------------------------------------ #
        age_bands        = ["infant", "child", "teen", "young_adult", "adult", "senior", "elderly"]
        age_fracs        = [0.05,     0.12,    0.11,   0.18,          0.30,    0.16,     0.08]
        age_susc         = [1.10,     0.70,    0.90,   1.00,          1.00,    1.40,     2.00]
        age_gamma_mod    = [1.50,     1.40,    1.30,   1.10,          1.00,    0.80,     0.55]
        age_mortality    = [0.0003,   0.0001,  0.0002, 0.0003,        0.0008,  0.0040,   0.0150]
        age_vax_priority = [0.80,     0.60,    0.65,   0.70,          0.75,    0.90,     0.95]

        # Contact matrix (relative mixing rates between age bands)
        contact_matrix = np.array([
            [1.5, 0.8, 0.3, 0.2, 0.6, 0.4, 0.3],
            [0.8, 3.0, 1.2, 0.5, 0.8, 0.3, 0.2],
            [0.3, 1.2, 4.5, 2.0, 0.9, 0.4, 0.2],
            [0.2, 0.5, 2.0, 3.5, 2.0, 0.6, 0.3],
            [0.6, 0.8, 0.9, 2.0, 2.8, 1.0, 0.5],
            [0.4, 0.3, 0.4, 0.6, 1.0, 2.5, 1.2],
            [0.3, 0.2, 0.2, 0.3, 0.5, 1.2, 2.0],
        ])
        n_age = len(age_bands)

        def split_pop(total, fracs):
            groups = [max(0, int(total * f)) for f in fracs]
            diff   = total - sum(groups)
            if diff != 0:
                groups[-1] = max(0, groups[-1] + diff)
            return groups

        age_S_vec = split_pop(S, age_fracs)
        age_I_vec = split_pop(I, age_fracs)
        age_R_vec = split_pop(R, age_fracs)

        # ------------------------------------------------------------------ #
        # SECTION 3 – Intervention decoding with synergy effects              #
        # ------------------------------------------------------------------ #
        base_beta_mult     = 1.0
        quarantine_rate    = 0.0
        mask_eff           = 0.0
        tracing_red        = 0.0
        school_factor      = 1.0
        dist_factor        = 1.0
        travel_ban         = 0.0
        lockdown_intensity = 0.0

        if action is None:
            pass
        elif action == 0:
            pass
        elif action == 1:
            base_beta_mult  = 0.82
            mask_eff        = 0.04
            quarantine_rate = 0.008
        elif action == 2:
            base_beta_mult  = 0.68
            mask_eff        = 0.09
            quarantine_rate = 0.030
            tracing_red     = contact_tracing_eff * 0.10
        elif action == 3:
            base_beta_mult     = 0.52
            mask_eff           = 0.14
            quarantine_rate    = 0.060
            tracing_red        = contact_tracing_eff * 0.22
            school_factor      = 0.80
            dist_factor        = 0.90
        elif action == 4:
            base_beta_mult     = 0.38
            mask_eff           = 0.20
            quarantine_rate    = 0.100
            tracing_red        = contact_tracing_eff * 0.38
            school_factor      = 0.65
            dist_factor        = 0.75
            travel_ban         = 0.20
        elif action == 5:
            base_beta_mult     = 0.25
            mask_eff           = 0.28
            quarantine_rate    = 0.160
            tracing_red        = contact_tracing_eff * 0.52
            school_factor      = 0.50
            dist_factor        = 0.58
            travel_ban         = 0.40
            lockdown_intensity = 0.30
        elif action == 6:
            base_beta_mult     = 0.14
            mask_eff           = 0.38
            quarantine_rate    = 0.240
            tracing_red        = contact_tracing_eff * 0.68
            school_factor      = 0.35
            dist_factor        = 0.40
            travel_ban         = 0.60
            lockdown_intensity = 0.55
        elif action == 7:
            base_beta_mult     = 0.06
            mask_eff           = 0.48
            quarantine_rate    = 0.340
            tracing_red        = contact_tracing_eff * 0.80
            school_factor      = 0.20
            dist_factor        = 0.25
            travel_ban         = 0.80
            lockdown_intensity = 0.75
        else:
            strength           = min(action / 10.0, 1.0)
            base_beta_mult     = max(0.04, 1.0 - 0.96 * strength)
            mask_eff           = min(0.55, strength * 0.55)
            quarantine_rate    = min(0.45, strength * 0.45)
            tracing_red        = min(0.85, contact_tracing_eff * strength * 0.85)
            school_factor      = max(0.15, 1.0 - 0.85 * strength)
            dist_factor        = max(0.18, 1.0 - 0.82 * strength)
            travel_ban         = min(0.90, strength * 0.90)
            lockdown_intensity = min(0.90, strength * 0.90)

        # Synergy bonus: combining multiple strong interventions
        active_interventions = sum([
            1 if mask_eff > 0.1 else 0,
            1 if quarantine_rate > 0.05 else 0,
            1 if tracing_red > 0.1 else 0,
            1 if school_factor < 0.8 else 0,
            1 if dist_factor < 0.8 else 0,
        ])
        synergy_bonus = 0.0
        if active_interventions >= 3:
            synergy_bonus = 0.04 * (active_interventions - 2)
        if active_interventions >= 5:
            synergy_bonus += 0.08

        composite_beta_mult = (base_beta_mult * school_factor * dist_factor
                               * (1.0 - mask_eff) * (1.0 - synergy_bonus))
        composite_beta_mult  = max(0.0, composite_beta_mult)
        effective_beta_base  = beta * composite_beta_mult * seasonal_factor

        # ------------------------------------------------------------------ #
        # SECTION 4 – Dynamic network topology (4 settings)                   #
        # ------------------------------------------------------------------ #
        settings = {
            "household":  {"frac": 0.30, "density": 2.50, "lockdown_sens": 0.20},
            "workplace":  {"frac": 0.35, "density": 1.40, "lockdown_sens": 0.80},
            "school":     {"frac": 0.15, "density": 1.80, "lockdown_sens": 0.95},
            "community":  {"frac": 0.20, "density": 0.90, "lockdown_sens": 0.60},
        }

        setting_new_infections = {}
        for setting_name, cfg in settings.items():
            frac        = cfg["frac"]
            density     = cfg["density"]
            ld_sens     = cfg["lockdown_sens"]
            s_s         = max(0, int(S * frac))
            i_s         = max(1, int(effective_infectious * frac))
            n_s         = max(1, int(N * frac))

            setting_reduction = 1.0 - lockdown_intensity * ld_sens
            if setting_name == "school":
                setting_reduction *= school_factor
            setting_reduction = max(0.05, setting_reduction)

            local_beta    = effective_beta_base * density * setting_reduction
            local_beta    = max(0.0, local_beta - tracing_red)
            inf_rate      = local_beta * i_s / n_s
            inf_prob      = float(np.clip(1.0 - np.exp(-inf_rate), 0.0, 1.0))

            new_inf_s = 0
            if s_s > 0 and inf_prob > 0.0:
                new_inf_s = int(rng.binomial(s_s, inf_prob))
            setting_new_infections[setting_name] = new_inf_s

        total_setting_infections = sum(setting_new_infections.values())

        # ------------------------------------------------------------------ #
        # SECTION 5 – Age-structured infection via contact matrix             #
        # ------------------------------------------------------------------ #
        age_force_of_infection = np.zeros(n_age)
        for g in range(n_age):
            foi_g = 0.0
            for h in range(n_age):
                i_h_frac = age_I_vec[h] / N if N > 0 else 0.0
                foi_g   += contact_matrix[g, h] * i_h_frac * age_susc[g]
            age_force_of_infection[g] = foi_g * effective_beta_base

        age_new_infections = []
        age_new_recoveries = []
        age_new_deaths     = []

        for g in range(n_age):
            s_g   = age_S_vec[g]
            i_g   = age_I_vec[g]
            foi_g = max(0.0, age_force_of_infection[g] - tracing_red)
            inf_p = float(np.clip(1.0 - np.exp(-foi_g), 0.0, 1.0))

            new_inf_g = 0
            if s_g > 0 and inf_p > 0.0:
                new_inf_g = int(rng.binomial(s_g, inf_p))

            rec_p = float(np.clip(gamma * age_gamma_mod[g], 0.0, 1.0))
            new_rec_g = 0
            if i_g > 0 and rec_p > 0.0:
                new_rec_g = int(rng.binomial(i_g, rec_p))

            mort_p  = float(np.clip(age_mortality[g], 0.0, 1.0))
            remaining_i = max(0, i_g - new_rec_g)
            new_death_g = 0
            if remaining_i > 0 and mort_p > 0.0:
                new_death_g = int(rng.binomial(remaining_i, mort_p))

            age_new_infections.append(new_inf_g)
            age_new_recoveries.append(new_rec_g)
            age_new_deaths.append(new_death_g)

        total_age_infections = sum(age_new_infections)
        total_new_recoveries = sum(age_new_recoveries)
        total_new_deaths     = sum(age_new_deaths)

        # Blend setting-based and age-based infections
        alpha_blend          = 0.55
        total_new_infections = int(
            alpha_blend * total_age_infections + (1.0 - alpha_blend) * total_setting_infections
        )

        # ------------------------------------------------------------------ #
        # SECTION 6 – Hospital saturation with multi-tier overflow penalties  #
        # ------------------------------------------------------------------ #
        hospital_cap   = max(1, int(N * hospital_capacity_frac * 0.04))
        icu_cap        = max(1, int(hospital_cap * 0.12))
        step_down_cap  = max(1, int(hospital_cap * 0.30))

        recovery_penalty  = 1.0
        mortality_penalty = 1.0

        if I <= hospital_cap:
            recovery_penalty  = 1.00
            mortality_penalty = 1.00
        elif I <= 2 * hospital_cap:
            ratio              = (I - hospital_cap) / hospital_cap
            recovery_penalty   = max(0.75, 1.0 - 0.15 * ratio)
            mortality_penalty  = 1.0 + 0.25 * ratio
        elif I <= 4 * hospital_cap:
            ratio              = (I - 2 * hospital_cap) / (2 * hospital_cap)
            recovery_penalty   = max(0.55, 0.85 - 0.20 * ratio)
            mortality_penalty  = 1.25 + 0.50 * ratio
        else:
            ratio              = min((I - 4 * hospital_cap) / hospital_cap, 3.0)
            recovery_penalty   = max(0.35, 0.55 - 0.10 * ratio)
            mortality_penalty  = 1.75 + 0.40 * ratio

        critical_I = max(0, int(I * 0.06))
        if critical_I > icu_cap:
            icu_ratio          = critical_I / icu_cap
            icu_mortality_mult = max(1.0, 1.0 + 0.35 * np.log(icu_ratio))
            mortality_penalty *= icu_mortality_mult

        total_new_recoveries = int(total_new_recoveries * recovery_penalty)
        total_new_deaths     = int(total_new_deaths * mortality_penalty)
        total_new_recoveries = min(total_new_recoveries, I)
        total_new_deaths     = min(total_new_deaths, max(0, I - total_new_recoveries))

        # ------------------------------------------------------------------ #
        # SECTION 7 – Quarantine with stochastic compliance tiers             #
        # ------------------------------------------------------------------ #
        quarantine_extra_recoveries = 0
        if quarantine_rate > 0 and I > 0:
            compliance_dist = [(0.25, 1.00), (0.45, 0.65), (0.20, 0.25), (0.10, 0.05)]
            effective_qrate = sum(w * c * quarantine_rate for w, c in compliance_dist)
            effective_qrate = float(np.clip(effective_qrate, 0.0, 1.0))
            quarantined     = int(rng.binomial(I, effective_qrate))

            if quarantined > 0:
                q_rec_prob = float(np.clip(gamma * 1.20 * recovery_penalty, 0.0, 1.0))
                quarantine_extra_recoveries = int(rng.binomial(quarantined, q_rec_prob))
                total_new_recoveries = min(I, total_new_recoveries + quarantine_extra_recoveries)

        # ------------------------------------------------------------------ #
        # SECTION 8 – Fomite and airborne environmental transmission          #
        # ------------------------------------------------------------------ #
        airborne_contamination = 0.0
        fomite_contamination   = 0.0

        if I > 0:
            airborne_shedding    = 0.0006 * effective_infectious / N
            fomite_shedding      = 0.0003 * I / N
            airborne_decay       = 0.25
            fomite_decay         = 0.10
            airborne_contamination = min(1.0, airborne_shedding) * (1.0 - airborne_decay)
            fomite_contamination   = min(1.0, fomite_shedding)   * (1.0 - fomite_decay)
            if lockdown_intensity > 0.0:
                airborne_contamination *= max(0.2, 1.0 - lockdown_intensity * 0.6)
                fomite_contamination   *= max(0.3, 1.0 - lockdown_intensity * 0.4)

        env_inf_prob = float(np.clip(
            1.0 - np.exp(-(airborne_contamination + fomite_contamination)), 0.0, 0.06
        ))
        env_infections = 0
        if S > 0 and env_inf_prob > 0.0:
            env_infections = int(rng.binomial(S, env_inf_prob))
        total_new_infections = min(S, total_new_infections + env_infections)

        # ------------------------------------------------------------------ #
        # SECTION 9 – Multi-dose vaccination with breakthrough infections     #
        # ------------------------------------------------------------------ #
        total_vaccinated = 0
        breakthrough_from_vax = 0
        if vaccination_rate > 0 and S > 0:
            for g in range(n_age):
                s_g      = age_S_vec[g]
                priority = age_vax_priority[g]
                vax_prob = float(np.clip(vaccination_rate * priority * 1.15, 0.0, 1.0))
                vaxed_g  = int(rng.binomial(s_g, vax_prob)) if s_g > 0 else 0

                # Two-dose efficacy model
                first_dose_eff  = 0.65
                second_dose_eff = 0.92
                dose_ratio      = 0.60   # fraction completing second dose
                effective_eff   = (1 - dose_ratio) * first_dose_eff + dose_ratio * second_dose_eff
                effective_eff  *= (1.0 - 0.03 * g)  # slight waning for older cohorts

                protected_g = int(rng.binomial(vaxed_g, float(np.clip(effective_eff, 0.0, 1.0)))) if vaxed_g > 0 else 0
                total_vaccinated += protected_g

                # Breakthrough infections (rare)
                not_protected = max(0, vaxed_g - protected_g)
                if not_protected > 0 and I > 0:
                    bt_prob = float(np.clip(effective_beta_base * I / N * 0.25, 0.0, 0.15))
                    bt_inf  = int(rng.binomial(not_protected, bt_prob))
                    breakthrough_from_vax += bt_inf

            total_vaccinated = min(S, total_vaccinated)

        total_new_infections = min(S - total_vaccinated, total_new_infections + breakthrough_from_vax)
        total_new_infections = max(0, total_new_infections)

        # ------------------------------------------------------------------ #
        # SECTION 10 – Waning immunity with partial protection tiers          #
        # ------------------------------------------------------------------ #
        natural_wane_prob = float(np.clip(1.0 - np.exp(-waning_immunity_rate), 0.0, 1.0))
        vax_wane_prob     = float(np.clip(natural_wane_prob * 1.80, 0.0, 1.0))

        full_immune_R    = max(0, int(R * 0.65))
        partial_immune_R = max(0, int(R * 0.25))
        vax_immune_R     = max(0, R - full_immune_R - partial_immune_R)

        waned_full = 0
        if full_immune_R > 0 and natural_wane_prob > 0:
            waned_full = int(rng.binomial(full_immune_R, natural_wane_prob))

        waned_partial = 0
        partial_wane_prob = float(np.clip(natural_wane_prob * 2.2, 0.0, 1.0))
        if partial_immune_R > 0 and partial_wane_prob > 0:
            waned_partial = int(rng.binomial(partial_immune_R, partial_wane_prob))

        waned_vax = 0
        if vax_immune_R > 0 and vax_wane_prob > 0:
            waned_vax = int(rng.binomial(vax_immune_R, vax_wane_prob))

        total_waned = waned_full + waned_partial + waned_vax
        total_waned = min(total_waned, R)

        # ------------------------------------------------------------------ #
        # SECTION 11 – Mutation and variant emergence (probabilistic)         #
        # ------------------------------------------------------------------ #
        mutation_escaped = 0
        if mutation_rate > 0 and R > 0 and I > 0:
            variant_stages = [
                (mutation_rate * 0.40, 0.03, 0.08,  "minor_drift"),
                (mutation_rate * 0.30, 0.07, 0.18,  "moderate_drift"),
                (mutation_rate * 0.20, 0.12, 0.30,  "immune_escape"),
                (mutation_rate * 0.08, 0.25, 0.50,  "major_variant"),
                (mutation_rate * 0.02, 0.40, 0.75,  "pandemic_variant"),
            ]
            for stage_prob, lo_frac, hi_frac, label in variant_stages:
                if rng.random() < stage_prob:
                    escape_frac = rng.uniform(lo_frac, hi_frac)
                    escaped     = min(int(R * escape_frac), R - mutation_escaped)
                    mutation_escaped += max(0, escaped)

            # Reinfection boost from cross-immune erosion
            if mutation_rate > 0.01:
                erosion_prob   = float(np.clip(mutation_rate * 5.0, 0.0, 0.25))
                eroded         = int(rng.binomial(R, erosion_prob))
                mutation_escaped = min(R, mutation_escaped + eroded)

        mutation_escaped = min(mutation_escaped, R)

        # ------------------------------------------------------------------ #
        # SECTION 12 – Reinfection dynamics                                   #
        # ------------------------------------------------------------------ #
        reinfection_prob = 0.0
        if N > 0 and I > 0:
            reinfection_base = 0.004 * effective_beta_base * effective_infectious / N
            reinfection_prob = float(np.clip(1.0 - np.exp(-reinfection_base), 0.0, 0.04))
        reinfections = 0
        available_R  = max(0, R - mutation_escaped)
        if available_R > 0 and reinfection_prob > 0.0:
            reinfections = int(rng.binomial(available_R, reinfection_prob))

        # ------------------------------------------------------------------ #
        # SECTION 13 – Behavioral adaptation (prevalence-driven, 6-tier)     #
        # ------------------------------------------------------------------ #
        prevalence = I / N if N > 0 else 0.0
        beh_thresholds = [0.01, 0.03, 0.07, 0.12, 0.22, 0.38]
        beh_reductions = [0.02, 0.10, 0.22, 0.38, 0.55, 0.70]

        behavioral_reduction = 0.0
        for ti in range(len(beh_thresholds) - 1):
            lo = beh_thresholds[ti]
            hi = beh_thresholds[ti + 1]
            if lo < prevalence <= hi:
                t  = (prevalence - lo) / (hi - lo)
                behavioral_reduction = beh_reductions[ti] + t * (beh_reductions[ti + 1] - beh_reductions[ti])
                break
        else:
            if prevalence > beh_thresholds[-1]:
                behavioral_reduction = beh_reductions[-1]
            elif prevalence <= beh_thresholds[0]:
                behavioral_reduction = 0.0

        # Fatigue: behavioral adherence decays if prevalence sustained high
        fatigue_factor = 1.0
        if prevalence > 0.15:
            fatigue_factor = max(0.50, 1.0 - 0.35 * ((prevalence - 0.15) / 0.15))
        behavioral_reduction *= fatigue_factor

        averted_by_behavior  = int(total_new_infections * behavioral_reduction)
        total_new_infections = max(0, total_new_infections - averted_by_behavior)
        total_new_infections = min(total_new_infections, S - total_vaccinated)

        # ------------------------------------------------------------------ #
        # SECTION 14 – Travel and importation                                 #
        # ------------------------------------------------------------------ #
        import_lam = max(0.0, 0.00025 * N * (1.0 - travel_ban))
        export_lam = max(0.0, 0.00015 * I * (1.0 - travel_ban))

        imported  = 0
        if S > 0 and import_lam > 0:
            imported = min(int(rng.poisson(lam=import_lam)), S - total_vaccinated)
        exported  = 0
        if I > 0 and export_lam > 0:
            exported = min(int(rng.poisson(lam=export_lam)), I)

        total_new_infections = min(max(0, S - total_vaccinated), total_new_infections + imported)

        # ------------------------------------------------------------------ #
        # SECTION 15 – HCW exposure sub-model                                 #
        # ------------------------------------------------------------------ #
        hcw_frac     = 0.018
        hcw_S        = max(0, int(S * hcw_frac))
        ppe_reduction = 0.55
        hcw_beta      = effective_beta_base * 2.00 * (1.0 - ppe_reduction)
        hcw_inf_prob  = float(np.clip(1.0 - np.exp(-hcw_beta * effective_infectious / N), 0.0, 1.0)) if N > 0 else 0.0
        hcw_infections = 0
        if hcw_S > 0 and hcw_inf_prob > 0.0:
            hcw_infections = int(rng.binomial(hcw_S, hcw_inf_prob))
        total_new_infections = min(S - total_vaccinated, total_new_infections + hcw_infections)

        # ------------------------------------------------------------------ #
        # SECTION 16 – Update compartments                                    #
        # ------------------------------------------------------------------ #
        new_S = int(S
                    - total_new_infections
                    + total_waned
                    + mutation_escaped
                    - total_vaccinated)
        new_I = int(I
                    + total_new_infections
                    - total_new_recoveries
                    - total_new_deaths
                    - exported
                    + reinfections)
        new_R = int(R
                    + total_new_recoveries
                    - total_waned
                    - mutation_escaped
                    + total_vaccinated
                    - reinfections)

        new_S = max(0, new_S)
        new_I = max(0, new_I)
        new_R = max(0, new_R)

        # ------------------------------------------------------------------ #
        # SECTION 17 – Conservation (account for deaths + exports)            #
        # ------------------------------------------------------------------ #
        N_adjusted = N - total_new_deaths - exported
        N_adjusted = max(0, N_adjusted)

        total_current = new_S + new_I + new_R
        discrepancy   = N_adjusted - total_current

        if discrepancy != 0:
            keys_sorted = sorted(["S", "I", "R"],
                                  key=lambda k: {"S": new_S, "I": new_I, "R": new_R}[k],
                                  reverse=True)
            remaining = discrepancy
            compartments = {"S": new_S, "I": new_I, "R": new_R}
            for key in keys_sorted:
                if remaining == 0:
                    break
                proposed = compartments[key] + remaining
                compartments[key] = max(0, proposed)
                remaining = N_adjusted - (compartments["S"] + compartments["I"] + compartments["R"])
            new_S, new_I, new_R = compartments["S"], compartments["I"], compartments["R"]

        # ------------------------------------------------------------------ #
        # SECTION 18 – Super-spreader events (multi-tier)                     #
        # ------------------------------------------------------------------ #
        super_spreader_events = [
            (0.015, 0.03,  2.5),
            (0.007, 0.06,  5.5),
            (0.003, 0.12, 12.0),
            (0.001, 0.22, 25.0),
            (0.0002, 0.35, 60.0),
        ]

        if new_I > 0 and new_S > 0:
            for tier_p, scale, lam_mult in super_spreader_events:
                if rng.random() < tier_p:
                    base_lam  = max(1, int(scale * new_S))
                    extra_inf = int(np.clip(rng.poisson(lam=base_lam * lam_mult), 0, new_S))
                    if extra_inf > 0:
                        new_S -= extra_inf
                        new_I += extra_inf

        # ------------------------------------------------------------------ #
        # SECTION 19 – Cluster outbreak events                                #
        # ------------------------------------------------------------------ #
        cluster_event_prob  = 0.010
        cluster_size_config = [
            (0.55,  2,  15),
            (0.30, 15,  75),
            (0.12, 75, 300),
            (0.03, 300, 800),
        ]

        if new_S > 0 and new_I > 0 and rng.random() < cluster_event_prob:
            roll       = rng.random()
            cum        = 0.0
            c_lo, c_hi = 2, 10
            for c_prob, lo, hi in cluster_size_config:
                cum += c_prob
                if roll < cum:
                    c_lo, c_hi = lo, hi
                    break
            c_hi_clamp  = min(c_hi, new_S)
            c_lo_clamp  = min(c_lo, c_hi_clamp)
            if c_hi_clamp > c_lo_clamp:
                cluster_sz  = int(rng.integers(low=c_lo_clamp, high=c_hi_clamp + 1))
                cluster_inf = min(cluster_sz, new_S)
                new_S -= cluster_inf
                new_I += cluster_inf

        # ------------------------------------------------------------------ #
        # SECTION 20 – Spontaneous recovery surge (medical breakthrough)      #
        # ------------------------------------------------------------------ #
        breakthrough_prob = 0.0025
        if new_I > 40 and rng.random() < breakthrough_prob:
            surge_frac       = rng.uniform(0.025, 0.10)
            surge_rec        = min(int(new_I * surge_frac), new_I)
            new_I           -= surge_rec
            new_R           += surge_rec

        # ------------------------------------------------------------------ #
        # SECTION 21 – Partial recovered relapse (edge case)                  #
        # ------------------------------------------------------------------ #
        relapse_prob = 0.0008
        if new_R > 0 and new_I > 0 and rng.random() < relapse_prob:
            relapse_frac = rng.uniform(0.005, 0.02)
            relapse_cnt  = min(int(new_R * relapse_frac), new_R)
            new_R -= relapse_cnt
            new_I += relapse_cnt

        # ------------------------------------------------------------------ #
        # SECTION 22 – Final conservation enforcement                         #
        # ------------------------------------------------------------------ #
        new_S = max(0, new_S)
        new_I = max(0, new_I)
        new_R = max(0, new_R)

        final_total       = new_S + new_I + new_R
        final_discrepancy = N_adjusted - final_total

        if final_discrepancy != 0:
            largest = max([("S", new_S), ("I", new_I), ("R", new_R)], key=lambda x: x[1])[0]
            if largest == "S":
                new_S = max(0, new_S + final_discrepancy)
            elif largest == "I":
                new_I = max(0, new_I + final_discrepancy)
            else:
                new_R = max(0, new_R + final_discrepancy)

        next_state["S"] = int(new_S)
        next_state["I"] = int(new_I)
        next_state["R"] = int(new_R)

        return next_state


    def evaluate(self, x):
        return 0

