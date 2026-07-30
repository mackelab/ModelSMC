import torch
import torch.nn as nn


class DiscoveredSimulator(nn.Module):
    def __init__(self):
        super(DiscoveredSimulator, self).__init__()
        return


    def forward(self, init_voltage: float, input_current: torch.Tensor, dt: float, t: torch.Tensor, params: torch.Tensor, seed=None):
        """
        Simulates a Hodgkin-Huxley neuron for a specified time duration.

        Args:
            init_voltage: torch.Tensor: (batch_size,) # initial voltage
            input_current: torch.Tensor: (batch_size, time_steps) # input current
            dt: float # time step size
            t: torch.Tensor: (time_steps,) # time array
            params: torch.Tensor: (batch_size, n_params) # parameters
            seed: optional random seed

        Returns:
            V: torch.Tensor: (batch_size, time_steps) # voltage traces
        """
        device = params.device

        # Set up random generator
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = torch.Generator(device=device)

        batch_size = params.shape[0]
        time_steps = t.shape[0]

        # Extract parameters
        gbar_Na = params[:, 0].float()  # mS/cm2
        gbar_K = params[:, 1].float() # mS/cm2
        g_leak = params[:, 2].float() # mS/cm2
        E_leak = -params[:, 3].float() # mV
        Vt = -params[:, 4].float() # mV
        nois_fact = params[:, 5].float() # unitless
        # TWO POSSIBLE ADDITIONAL CHANNELS (X1, X2)
        # Each channel has one tunable parameter: conductance gbar_Xi
        # Then there are two additional parameters available: param_i and param_j.
        # ONLY ADD ONE CHANNEL IF NECESSARY. Keep the model as simple as possible.
        gbar_X1 = params[:, 6].float() # mS/cm2 # you can rename X1 to anything you want # in range [1e-4, 10]
        gbar_X2 = params[:, 7].float() # mS/cm2 # you can rename X2 to anything you want # in range [1e-4, 120]
        param_i = -params[:, 8].float() # (param are positive values in range [1e-4, 150])
        param_j = -params[:, 9].float() # (param are positive values in range [1e-4, 3000])

        tstep = float(dt)

        # Parameters
        nois_fact_obs = 0.0
        C = 1.0  # uF/cm²
        E_Na = 53.0 # mV
        E_K = -107.0

        ####################################
        # kinetics
        def Exp(z):
            return torch.where(z < -5e2, torch.exp(torch.full_like(z, -5e2)), torch.exp(z))

        def efun(z):
            return torch.where(torch.abs(z) < 1e-4, 1 - z / 2, z / (Exp(z) - 1))

        # Channel kinetics
        def alpha_m(x):
            v1 = x - Vt - 13.0
            return 0.32 * efun(-0.25 * v1) / 0.25

        def beta_m(x):
            v1 = x - Vt - 40
            return 0.28 * efun(0.2 * v1) / 0.2

        def alpha_h(x):
            v1 = x - Vt - 17.0
            return 0.128 * Exp(-v1 / 18.0)

        def beta_h(x):
            v1 = x - Vt - 40.0
            return 4.0 / (1 + Exp(-0.2 * v1))

        def alpha_n(x):
            v1 = x - Vt - 15.0
            return 0.032 * efun(-0.2 * v1) / 0.2

        def beta_n(x):
            v1 = x - Vt - 10.0
            return 0.5 * Exp(-v1 / 40)

        def tau_x(alpha, beta):
            return 1.0 / (alpha + beta)

        def inf_x(alpha, beta):
            return alpha / (alpha + beta)

        # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
        # TODO: add the missing kinetics equations for the Hodgkin-Huxley neuron similar to the ones above; ONLY ADD IF NECESSARY
        # ===== END EDITABLE SECTION =====

        ####################################

        # simulation from initial point
        V = torch.zeros((batch_size, time_steps), device=device)  # baseline voltage
        n = torch.zeros((batch_size, time_steps), device=device)
        m = torch.zeros((batch_size, time_steps), device=device)
        h = torch.zeros((batch_size, time_steps), device=device)
        # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
        # TODO: add the missing state variables for the Hodgkin-Huxley neuron similar to the ones above; ONLY ADD IF NECESSARY
        # ===== END EDITABLE SECTION =====

        # Initialization
        V_init = init_voltage.to(device)
        V[:, 0] = V_init
        n[:, 0] = inf_x(alpha_n(V[:, 0]), beta_n(V[:, 0]))
        m[:, 0] = inf_x(alpha_m(V[:, 0]), beta_m(V[:, 0]))
        h[:, 0] = inf_x(alpha_h(V[:, 0]), beta_h(V[:, 0]))
        # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
        # TODO: add the missing state variable initialization for the Hodgkin-Huxley neuron similar to the ones above; ONLY ADD IF NECESSARY
        # ===== END EDITABLE SECTION =====

        # Simulation loop
        for i in range(1, time_steps):
            # All operations now work on batched tensors (batch_size,)
            a_m, b_m = alpha_m(V[:, i - 1]), beta_m(V[:, i - 1])
            a_h, b_h = alpha_h(V[:, i - 1]), beta_h(V[:, i - 1])
            a_n, b_n = alpha_n(V[:, i - 1]), beta_n(V[:, i - 1])
            # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
            # TODO: add the missing kinetics equations for the Hodgkin-Huxley neuron similar to the ones above; ONLY ADD IF NECESSARY
            # ===== END EDITABLE SECTION =====

            tau_V_inv = (
                (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1]
                + (n[:, i - 1] ** 4) * gbar_K
                + g_leak
                # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
                # TODO: add the missing terms for the effective membrane time constant inverse; ONLY ADD IF NECESSARY
                # ===== END EDITABLE SECTION =====
            ) / C

            V_inf = (
                (m[:, i - 1] ** 3) * gbar_Na * h[:, i - 1] * E_Na
                + (n[:, i - 1] ** 4) * gbar_K * E_K
                + g_leak * E_leak
                # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
                # TODO: add the missing terms for the voltage steady state; ONLY ADD IF NECESSARY
                # ===== END EDITABLE SECTION =====
                + input_current[:,i - 1]
                + nois_fact * torch.randn(batch_size, generator=generator, device=device) / (tstep**0.5)
            ) / (tau_V_inv * C)

            V[:, i] = V_inf + (V[:, i - 1] - V_inf) * Exp(-tstep * tau_V_inv)
            n[:, i] = inf_x(a_n, b_n) + (n[:, i - 1] - inf_x(a_n, b_n)) * Exp(-tstep / tau_x(a_n, b_n))
            m[:, i] = inf_x(a_m, b_m) + (m[:, i - 1] - inf_x(a_m, b_m)) * Exp(-tstep / tau_x(a_m, b_m))
            h[:, i] = inf_x(a_h, b_h) + (h[:, i - 1] - inf_x(a_h, b_h)) * Exp(-tstep / tau_x(a_h, b_h))
            # ===== BEGIN EDITABLE SECTION (only modify within this block) =====
            # TODO: add the missing state variable updates for the Hodgkin-Huxley neuron similar to the ones above; ONLY ADD IF NECESSARY
            # ===== END EDITABLE SECTION =====

        # Return voltage with optional observation noise
        return V + nois_fact_obs * torch.randn(
            batch_size, time_steps, generator=generator, device=device
        )


    def evaluate(self, x):
        return 0

