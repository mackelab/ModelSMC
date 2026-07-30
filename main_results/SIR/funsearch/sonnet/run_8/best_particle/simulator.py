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

        beta = parameters[0]
        gamma = parameters[1]

        beta = np.clip(beta, 0.0, 1.0)
        gamma = np.clip(gamma, 0.0, 1.0)

        R0 = beta / (gamma + 1e-9)
        prevalence = I / N if N > 0 else 0.0
        immunity_ratio = R / N if N > 0 else 0.0
        susceptibility_ratio = S / N if N > 0 else 0.0

        # Epidemic phase with finer granularity
        if prevalence == 0.0:
            epidemic_phase = "extinct"
        elif prevalence < 0.005:
            epidemic_phase = "seed"
        elif prevalence < 0.01:
            epidemic_phase = "early"
        elif prevalence < 0.05:
            epidemic_phase = "growth"
        elif prevalence < 0.15:
            epidemic_phase = "acceleration"
        elif prevalence < 0.3:
            epidemic_phase = "peak"
        elif prevalence < 0.5:
            epidemic_phase = "decline"
        else:
            epidemic_phase = "collapse"

        # Seasonal forcing: sinusoidal variation in transmission
        season_amplitude = 0.15
        # Use a proxy for season based on immunity_ratio as a stand-in for time
        season_phase = np.pi * immunity_ratio * 4.0
        seasonal_factor = 1.0 + season_amplitude * np.sin(season_phase)
        seasonal_factor = np.clip(seasonal_factor, 0.8, 1.2)

        # Age-stratified risk weights (virtual strata: young, adult, elderly)
        age_strata_weights = [0.25, 0.50, 0.25]
        age_susceptibility = [0.6, 1.0, 1.4]
        age_severity = [0.3, 1.0, 2.5]
        weighted_susceptibility = sum(
            age_strata_weights[k] * age_susceptibility[k] for k in range(3)
        )
        weighted_severity = sum(
            age_strata_weights[k] * age_severity[k] for k in range(3)
        )

        # Waning immunity: a fraction of R slowly returns to S
        waning_rate = 0.002
        if epidemic_phase in ("decline", "collapse", "extinct"):
            waning_rate = 0.005
        waning_return = min(int(round(R * waning_rate)), R)
        S_waned = S + waning_return
        R_waned = R - waning_return

        # Recompute prevalence after waning
        prevalence_adj = I / N if N > 0 else 0.0

        # Action-based intervention with multi-tier compliance and fatigue
        intervention_fatigue = 0.0
        if action is not None:
            fatigue_lookup = {0: 0.0, 1: 0.05, 2: 0.12, 3: 0.25}
            intervention_fatigue = fatigue_lookup.get(action, 0.0)

            if epidemic_phase in ("peak", "acceleration"):
                intervention_fatigue *= 1.5
            elif epidemic_phase in ("decline", "collapse"):
                intervention_fatigue *= 0.5

            intervention_fatigue = np.clip(intervention_fatigue, 0.0, 0.4)

            if action == 0:
                effective_beta = beta * seasonal_factor
                compliance_factor = 1.0
                intervention_label = "none"
            elif action == 1:
                base_reduction = 0.75
                if epidemic_phase == "peak":
                    base_reduction = 0.80
                elif epidemic_phase == "seed":
                    base_reduction = 0.70
                effective_beta = beta * base_reduction * seasonal_factor
                compliance_factor = max(0.5, 0.95 - intervention_fatigue)
                intervention_label = "mild"
            elif action == 2:
                base_reduction = 0.50
                if epidemic_phase in ("growth", "acceleration", "peak"):
                    base_reduction = 0.45
                elif epidemic_phase == "decline":
                    base_reduction = 0.55
                elif epidemic_phase == "seed":
                    base_reduction = 0.40
                effective_beta = beta * base_reduction * seasonal_factor
                compliance_factor = max(0.4, 0.85 - intervention_fatigue)
                intervention_label = "moderate"
            elif action == 3:
                base_reduction = 0.20
                if epidemic_phase == "early":
                    base_reduction = 0.15
                elif epidemic_phase == "peak":
                    base_reduction = 0.25
                elif epidemic_phase == "collapse":
                    base_reduction = 0.30
                effective_beta = beta * base_reduction * seasonal_factor
                compliance_factor = max(0.3, 0.70 - intervention_fatigue)
                intervention_label = "strict"
            else:
                effective_beta = beta * seasonal_factor
                compliance_factor = 1.0
                intervention_label = "unknown"
        else:
            effective_beta = beta * seasonal_factor
            compliance_factor = 1.0
            intervention_label = "none"

        # Apply weighted susceptibility from age strata
        effective_beta *= weighted_susceptibility

        # Herd immunity dampening with non-linear threshold
        herd_immunity_threshold = 1.0 - (1.0 / (R0 + 1e-9))
        herd_immunity_threshold = np.clip(herd_immunity_threshold, 0.0, 0.99)

        if immunity_ratio > herd_immunity_threshold * 0.4:
            excess = immunity_ratio - herd_immunity_threshold * 0.4
            herd_factor = np.exp(-3.0 * excess)
            herd_factor = np.clip(herd_factor, 0.05, 1.0)
            effective_beta *= herd_factor
        
        # Network heterogeneity: scale-free contact distribution
        if R0 > 2.5:
            network_amplifier = 1.0 + 0.1 * np.log(R0)
        else:
            network_amplifier = 1.0
        effective_beta = np.clip(effective_beta * network_amplifier, 0.0, 1.0)

        # Adaptive substep count: more substeps for faster dynamics
        substep_map = {
            "extinct": 1,
            "seed": 2,
            "early": 4,
            "growth": 8,
            "acceleration": 10,
            "peak": 12,
            "decline": 6,
            "collapse": 4,
        }
        num_substeps = substep_map.get(epidemic_phase, 5)

        if R0 > 4.0:
            num_substeps = max(num_substeps, 14)
        elif R0 > 3.0:
            num_substeps = max(num_substeps, 10)
        elif R0 > 2.0:
            num_substeps = max(num_substeps, 7)

        # Large populations benefit from more substeps
        if N > 100000:
            num_substeps = max(num_substeps, 10)
        elif N > 10000:
            num_substeps = max(num_substeps, 7)

        S_curr = float(S_waned)
        I_curr = float(I)
        R_curr = float(R_waned)

        total_new_infections = 0
        total_new_recoveries = 0
        substep_infection_history = []
        substep_recovery_history = []
        peak_substep_I = I_curr

        # Variant emergence probability
        variant_emerged = False
        variant_boost = 1.0

        for step in range(num_substeps):
            N_curr = S_curr + I_curr + R_curr
            if N_curr <= 0 or I_curr <= 0:
                break

            sub_beta = effective_beta / num_substeps
            sub_gamma = gamma / num_substeps

            # Track peak within substeps
            if I_curr > peak_substep_I:
                peak_substep_I = I_curr

            # Variant emergence: random mutation event
            if not variant_emerged and epidemic_phase in ("growth", "acceleration", "peak"):
                mutation_prob = 0.002 * (I_curr / N_curr) * (step + 1)
                if rng.random() < mutation_prob:
                    variant_emerged = True
                    variant_boost = rng.uniform(1.1, 1.5)

            if variant_emerged:
                sub_beta = np.clip(sub_beta * variant_boost, 0.0, 1.0 / num_substeps)

            # Super-spreading events: heavy-tailed contact distribution
            super_spread_prob = 0.0
            if epidemic_phase in ("acceleration", "peak") and prevalence_adj > 0.10:
                super_spread_prob = 0.06
            elif epidemic_phase == "growth" and prevalence_adj > 0.04:
                super_spread_prob = 0.03

            if super_spread_prob > 0 and rng.random() < super_spread_prob:
                spread_multiplier = rng.uniform(2.5, 6.0)
                sub_beta = np.clip(sub_beta * spread_multiplier, 0.0, 1.0 / num_substeps)

            # Behavioral feedback: multi-layer response
            behavioral_response = 1.0
            if prevalence_adj > 0.10:
                behavioral_response = np.exp(-3.5 * prevalence_adj)
            elif prevalence_adj > 0.05:
                behavioral_response = np.exp(-2.0 * prevalence_adj)
            elif prevalence_adj > 0.02:
                behavioral_response = 1.0 - prevalence_adj * 4.0
            elif prevalence_adj > 0.005:
                behavioral_response = 1.0 - prevalence_adj * 2.0
            behavioral_response = np.clip(behavioral_response, 0.05, 1.0)

            # Media amplification: high-profile outbreaks trigger extra caution
            if epidemic_phase in ("acceleration", "peak") and prevalence_adj > 0.08:
                media_factor = 0.85
            elif epidemic_phase == "growth" and prevalence_adj > 0.03:
                media_factor = 0.92
            else:
                media_factor = 1.0

            effective_sub_beta = sub_beta * behavioral_response * compliance_factor * media_factor

            # Healthcare saturation: recovery slows as hospitals fill
            saturation_factor = 1.0
            if prevalence_adj > 0.25:
                saturation_factor = max(0.4, 1.0 - (prevalence_adj - 0.25) * 1.5)
            elif prevalence_adj > 0.15:
                saturation_factor = max(0.6, 1.0 - (prevalence_adj - 0.15) * 0.8)
            elif prevalence_adj > 0.10:
                saturation_factor = max(0.75, 1.0 - (prevalence_adj - 0.10) * 0.5)

            # Age-severity weighted recovery
            effective_sub_gamma = sub_gamma * saturation_factor / weighted_severity

            infection_prob = np.clip(1.0 - np.exp(-effective_sub_beta * I_curr / N_curr), 0.0, 1.0)
            recovery_prob = np.clip(1.0 - np.exp(-effective_sub_gamma), 0.0, 1.0)

            s_int = max(0, int(round(S_curr)))
            i_int = max(0, int(round(I_curr)))

            new_infections = rng.binomial(s_int, infection_prob) if s_int > 0 and infection_prob > 0 else 0
            new_recoveries = rng.binomial(i_int, recovery_prob) if i_int > 0 and recovery_prob > 0 else 0

            new_infections = min(new_infections, int(S_curr))
            new_recoveries = min(new_recoveries, int(I_curr))

            # Secondary attack clusters: small additional infections in dense areas
            cluster_infections = 0
            if epidemic_phase in ("growth", "acceleration", "peak") and N > 500:
                cluster_prob = 0.01 * prevalence_adj
                if rng.random() < cluster_prob:
                    cluster_size = rng.integers(1, max(2, int(0.005 * N)))
                    cluster_infections = min(int(cluster_size), int(S_curr) - new_infections)
                    cluster_infections = max(0, cluster_infections)

            new_infections = min(new_infections + cluster_infections, int(S_curr))

            S_curr -= new_infections
            I_curr += new_infections - new_recoveries
            R_curr += new_recoveries

            total_new_infections += new_infections
            total_new_recoveries += new_recoveries
            substep_infection_history.append(new_infections)
            substep_recovery_history.append(new_recoveries)

            S_curr = max(0.0, S_curr)
            I_curr = max(0.0, I_curr)
            R_curr = max(0.0, R_curr)

            # Dynamic early-exit: if both infection and recovery are negligible
            if I_curr < 0.5 and new_infections == 0 and new_recoveries == 0:
                break

            # Update local prevalence for next substep behavioral response
            N_curr_new = S_curr + I_curr + R_curr
            if N_curr_new > 0:
                prevalence_adj = I_curr / N_curr_new

        # Noise computation: multi-factor scale
        base_noise_scale = max(1, int(0.008 * N))

        phase_noise_map = {
            "extinct": 0.1,
            "seed": 1.5,
            "early": 2.0,
            "growth": 1.8,
            "acceleration": 1.3,
            "peak": 1.0,
            "decline": 0.6,
            "collapse": 0.3,
        }
        noise_multiplier = phase_noise_map.get(epidemic_phase, 1.0)

        # Variant noise boost
        if variant_emerged:
            noise_multiplier *= 1.2

        # Action-based noise reduction
        action_noise_reduction = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.45}
        if action is not None:
            noise_multiplier *= action_noise_reduction.get(action, 1.0)

        # Population size scaling: larger populations have relatively smaller noise
        if N > 50000:
            noise_multiplier *= 0.5
        elif N > 10000:
            noise_multiplier *= 0.75

        noise_scale = max(1, int(base_noise_scale * noise_multiplier))

        # Differentiated noise by phase
        if epidemic_phase in ("extinct", "seed", "early", "decline", "collapse") and N > 50:
            s_noise_raw = rng.poisson(lam=max(1, noise_scale // 3))
            s_noise = int(s_noise_raw) if rng.random() > 0.5 else -int(s_noise_raw)
            s_noise = int(np.clip(s_noise, -noise_scale, noise_scale))

            i_noise_raw = rng.poisson(lam=max(1, noise_scale // 2))
            i_noise = int(i_noise_raw) if rng.random() > 0.5 else -int(i_noise_raw)
            i_noise = int(np.clip(i_noise, -noise_scale, noise_scale))
        elif epidemic_phase in ("growth", "acceleration"):
            s_noise = int(rng.integers(-noise_scale, noise_scale + 1))
            i_noise = int(rng.normal(0, noise_scale * 0.5))
            i_noise = int(np.clip(i_noise, -noise_scale, noise_scale))
        else:
            s_noise = int(rng.integers(-noise_scale, noise_scale + 1))
            i_noise = int(rng.integers(-noise_scale, noise_scale + 1))

        S_final = int(round(S_curr))
        I_final = int(round(I_curr))
        R_final = int(round(R_curr))

        # Apply noise
        S_final = max(0, S_final + s_noise)
        I_final = max(0, I_final + i_noise)
        R_final = max(0, R_final - s_noise - i_noise)

        # Multi-pass conservation correction with priority ordering
        priority_order = ["R", "S", "I"]
        if epidemic_phase in ("growth", "acceleration", "peak"):
            priority_order = ["S", "R", "I"]
        elif epidemic_phase in ("decline", "collapse"):
            priority_order = ["I", "R", "S"]

        for correction_pass in range(5):
            total_after = S_final + I_final + R_final
            diff = N - total_after

            if diff == 0:
                break

            compartments_vals = {"S": S_final, "I": I_final, "R": R_final}
            total_compartment_sum = sum(compartments_vals.values())

            if total_compartment_sum > 0:
                if abs(diff) >= 4:
                    s_share = int(round(diff * S_final / total_compartment_sum))
                    i_share = int(round(diff * I_final / total_compartment_sum))
                    r_share = diff - s_share - i_share
                    S_final = max(0, S_final + s_share)
                    I_final = max(0, I_final + i_share)
                    R_final = max(0, R_final + r_share)
                elif abs(diff) >= 2:
                    sorted_compartments = sorted(
                        compartments_vals.items(), key=lambda x: x[1], reverse=(diff > 0)
                    )
                    half = diff // 2
                    remainder = diff - half
                    for idx, (key, _) in enumerate(sorted_compartments[:2]):
                        portion = half if idx == 0 else remainder
                        if key == "S":
                            S_final = max(0, S_final + portion)
                        elif key == "I":
                            I_final = max(0, I_final + portion)
                        else:
                            R_final = max(0, R_final + portion)
                else:
                    for key in priority_order:
                        if key == "S" and S_final + diff >= 0:
                            S_final += diff
                            break
                        elif key == "I" and I_final + diff >= 0:
                            I_final += diff
                            break
                        elif key == "R" and R_final + diff >= 0:
                            R_final += diff
                            break
            else:
                R_final = max(0, diff)

            S_final = max(0, S_final)
            I_final = max(0, I_final)
            R_final = max(0, R_final)

        # Final cascade conservation fix
        total_final = S_final + I_final + R_final
        remaining_diff = N - total_final
        if remaining_diff != 0:
            if R_final + remaining_diff >= 0:
                R_final = R_final + remaining_diff
            elif I_final + (remaining_diff + R_final) >= 0:
                tmp = remaining_diff + R_final
                R_final = 0
                I_final = max(0, I_final + tmp)
            else:
                R_final = 0
                I_final = 0
                S_final = max(0, N)

            if S_final + I_final + R_final != N:
                I_final = max(0, N - S_final - R_final)
            if S_final + I_final + R_final != N:
                S_final = max(0, N - I_final - R_final)

        next_state["S"] = S_final
        next_state["I"] = I_final
        next_state["R"] = R_final

        return next_state


    def evaluate(self, x):
        return 0

