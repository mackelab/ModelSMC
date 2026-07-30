########################################################################################
#
# Portions of this file (the simulation rollout in `simulation_wrapper` and the
# Wasserstein/MMD evaluation in `eval_function`; see the respective docstrings for
# the exact source lines) are based on and adapted from
#
# https://github.com/samholt/generative-simulations/blob/72b5d51a7790b8b2ebc87973ee5e9d21aa818ece/libs/SIR/env.py
#
# which was written by Samuel Holt and released under the MIT license:
#
# MIT License
#
# Copyright (c) 2025 Samuel Holt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################

import logging
import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from sbi.utils import BoxUniform
from torch import Tensor
from torch.distributions import Distribution

from modelsmc.tasks.base_embedding_net import FixedEmbeddingHandler
from modelsmc.tasks.base_task import BaseTask
from modelsmc.tasks.SIR.external_code.env import (
    SimulatorStep,
    compute_mmd,
    trajectories_to_numpy,
    wasserstein_distance_nd,
)

logger = logging.getLogger("ModelSMC")


class NormalizeContextByPopulationEmbedder(nn.Module):
    """Embedder that normalizes the SIR context (initial state) by population size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the context [S0, I0, R0] by the total population size.

        Args:
            x: Context tensor. Shape: (batch_size, 3).

        Returns:
            x: Population-normalized context. Shape: (batch_size, 3).
        """
        assert x.shape[1] == 3, "Context for SIR task must have shape [batch_size, 3]."

        # Get the population size
        population_size = x.sum(dim=-1, keepdim=True)
        assert population_size.shape == torch.Size([x.shape[0], 1])

        # Normalize the context by the population size
        x = x / population_size

        assert x.shape[1] == 3, "Context for SIR task must have shape [batch_size, 3]."

        return x


class NormalizeObservationsByPopulationEmbedder(nn.Module):
    """Embedder that normalizes flat SIR observations by population size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each flattened [S, I, R] trajectory by its population size.

        Args:
            x: Flat observation tensor. Shape: (batch_size, 3 * time_steps).

        Returns:
            x: Population-normalized observations of the same shape as the input.
        """
        assert len(x.shape) == 2

        c = x.reshape(x.shape[0], -1, 3)

        # Get the population size for each instance
        population_size = c[:, 0, :].sum(dim=-1, keepdim=True)
        assert population_size.shape == torch.Size([x.shape[0], 1])

        # Normalize the data by the population size
        x = x / population_size

        assert x.shape == torch.Size([x.shape[0], c.shape[1] * c.shape[2]])

        return x


class SIRFixedEmbeddingHandler(FixedEmbeddingHandler):
    """Fixed (non-learnable) embedding handler for the SIR task.

    Uses population-size normalization as the summary statistic for both the
    observations and the context.
    """

    def __init__(self, split_x_ctxt: Callable) -> None:
        """Initialize the fixed SIR embedding handler.

        Args:
            split_x_ctxt: Callable that splits a combined tensor into data and
                context (typically the task's ``split_context_data``).
        """
        super().__init__(split_x_ctxt=split_x_ctxt)

        self.obs_embedding_function = NormalizeObservationsByPopulationEmbedder()
        self.context_embedding_function = NormalizeContextByPopulationEmbedder()

        self.obs_embedding_function.eval()
        self.context_embedding_function.eval()


class SIRBaseTask(BaseTask):
    """Base task for the SIR (Susceptible-Infected-Recovered) epidemic model.

    Data loading is level-specific, so subclasses must implement ``get_data``.
    Any ``get_data`` implementation must populate the following attributes, which
    are read by ``get_context`` and ``simulation_wrapper``:

    - ``context_size``: Dimensionality of the context (3 for [S0, I0, R0]).
    - ``_context_validation``: Validation contexts. Shape: (num_valid, context_size).
    - ``_context_training``: Training context(s). Shape: (1, context_size).
    - ``_num_time_steps``: Number of time steps per trajectory.
    """

    def __init__(
        self,
        config: DictConfig,
        prompts_path: str | None = None,
        base_simulator_path: str | None = None,
    ) -> None:
        """Initialize the SIR task.

        Args:
            config: Hydra configuration containing task settings.
            prompts_path: Path to the prompts.yaml file for this task.
            base_simulator_path: Path to the base_simulator.py file for this task.
        """
        super().__init__(config, prompts_path, base_simulator_path)

        # Prior distribution
        self.prior_dist = self.get_prior_distribution()

        # Load prompts
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
        self.system_description = prompts["system_description"]
        self.signature_description = prompts["signature_description"]
        self.task_description = prompts["task_description"]

        # Load base simulator code from file
        with open(base_simulator_path, "r") as f:
            self.base_simulator = f.read()

        # Initialize the embedding network
        self.init_embedding_net()

    def init_embedding_net(self) -> None:
        """Initialize the embedding network used to embed observations and context.

        The SIR task only supports the fixed, population-normalizing embedder;
        learnable summary statistics are not implemented.
        """
        # Only fixed summary statistics are implemented for the SIR task.
        if self.config.learnable_summary_statistics:
            raise NotImplementedError(
                "Learnable summary statistics are not supported for the SIR task. "
                "Set learnable_summary_statistics=False in the task config."
            )

        logger.info("Using fixed summary statistics for SIR task.")
        self.embedding_net = SIRFixedEmbeddingHandler(
            split_x_ctxt=self.split_context_data,
        )

    @staticmethod
    def get_prior_distribution() -> Distribution:
        """
        Get the prior distribution. In https://arxiv.org/pdf/2506.09272 the upper and
        lower bounds are predicted by the LLM, here we hardcode them as an adaptive
        selection of the prior bounds is not supported yet. The bounds of the prior are:
            - param 1 (beta):  [0.0, 2.0]
            - param 2 (gamma): [0.0, 1.0]
        as described on page 23 of the supplementary material of
        https://arxiv.org/pdf/2506.09272 for beta and gamma respectively.

        Returns:
            A 2D BoxUniform prior over (beta, gamma).
        """
        return BoxUniform(low=torch.tensor([0.0, 0.0]), high=torch.tensor([2.0, 1.0]))

    @staticmethod
    def _plot_observation(
        valid_data_raw: torch.Tensor,
        sim_data_raw: torch.Tensor | None = None,
        path: str | None = None,
        fig_name: str = "observation.png",
        max_observation: int = 10,
    ) -> None:
        """Plot raw SIR trajectories (S, I, R channels) and save the figure.

        Args:
            valid_data_raw: Raw validation trajectories.
                Shape: (batch_size, 3 * time_steps).
            sim_data_raw: Raw simulated trajectories to overlay (optional), same shape
                as ``valid_data_raw``.
            path: Directory to save the plot in.
            fig_name: Filename for the saved plot.
            max_observation: Maximum number of trajectories to plot (capped at 10).
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        for i in range(min(max_observation, valid_data_raw.shape[0])):
            ax = axes.flatten()[i]

            data_i = valid_data_raw[i].cpu().numpy().reshape(-1, 3)
            ax.plot(data_i[:, 0], label="S", color="blue", ls="--")
            ax.plot(data_i[:, 1], label="I", color="red", ls="--")
            ax.plot(data_i[:, 2], label="R", color="yellow", ls="--")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Population")

            if sim_data_raw is not None:
                data_sim_i = sim_data_raw[i].cpu().numpy().reshape(-1, 3)
                ax.plot(data_sim_i[:, 0], label="sim - S", color="blue")
                ax.plot(data_sim_i[:, 1], label="sim - I", color="red")
                ax.plot(data_sim_i[:, 2], label="sim - R", color="yellow")
                ax.set_title(f"Observation {i + 1}")
            ax.legend()

        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, fig_name))
        plt.close()

    def eval_function(
        self,
        sim_data: torch.Tensor,
        sim_data_raw: torch.Tensor,
        valid_data: torch.Tensor,
        valid_data_raw: torch.Tensor,
        parameter_estimates: torch.Tensor,
        simulator: nn.Module,
    ) -> dict[str, float]:
        """Evaluate the simulator against the validation data.

        Computes the mean squared error on the raw trajectories as well as the
        Wasserstein distance and the maximum mean discrepancy (MMD) between
        samples drawn from the fitted simulator and the ground-truth simulator.

        The evaluation in this method is adapted from the evaluation code in

        https://github.com/samholt/generative-simulations/blob/72b5d51a7790b8b2ebc87973ee5e9d21aa818ece/libs/SIR/env.py

        mainly lines 100 - 121

        published by Samuel Holt under the MIT license.

        Args:
            sim_data: Summary statistics of the simulated data.
                Shape: (batch_size, n_summary_stats).
            sim_data_raw: Raw simulated trajectories.
                Shape: (batch_size, 3 * time_steps).
            valid_data: Summary statistics of the validation data.
                Shape: (batch_size, n_summary_stats).
            valid_data_raw: Raw validation trajectories.
                Shape: (batch_size, 3 * time_steps).
            parameter_estimates: Estimated (beta, gamma) parameters.
                Shape: (batch_size, 2) or (2,).
            simulator: The fitted simulator used to generate the simulated data.

        Returns:
            Dictionary with the keys 'mse', 'wasserstein_distance' and 'mmd_distance'.
        """

        # Compute the mean squared error between the simulated data and the
        # validation data. The data is in raw format here, i.e. the flattened data.
        mse = torch.mean(torch.square(sim_data_raw - valid_data_raw)).item()

        results_dict = {
            "mse": mse,
        }

        # Add batch dimension if necessary
        if parameter_estimates.ndim == 1:
            parameter_estimates = parameter_estimates.unsqueeze(0)

        # If a single parameter set is provided, repeat it for all contexts
        if parameter_estimates.shape[0] == 1:
            parameter_estimates = parameter_estimates.repeat(valid_data_raw.shape[0], 1)

        assert parameter_estimates.shape[0] == valid_data_raw.shape[0], (
            "Parameter estimates and validation data must have the same batch size."
        )

        assert parameter_estimates.shape[1] == 2, (
            "Parameter estimates must have shape [batch_size, 2]."
        )

        # Reconstruct the three-channel structure of the data for Wasserstein distance
        # and MMD
        valid_data_channeled = valid_data_raw.reshape(valid_data_raw.shape[0], -1, 3)

        # Get the ground truth simulator
        gt_simulator = SimulatorStep()

        wdist_total, mmdist_total = [], []

        # Compute the Wasserstein distance and the MMD for each validation trajectory
        for i, data_traj in enumerate(valid_data_channeled):
            # set the parameters of the simulator to the estimated parameters
            simulator.set_parameters(parameters=parameter_estimates[i].cpu().numpy())

            for traj_state in data_traj:
                traj_state_dict = {
                    "S": traj_state[0].item(),
                    "I": traj_state[1].item(),
                    "R": traj_state[2].item(),
                }

                samples = [
                    simulator.step(
                        state=traj_state_dict, action=0, rng=np.random.default_rng()
                    )
                    for _ in range(self.config.n_samples_wsd_mmd)
                ]

                gt_samples = [
                    gt_simulator.step(
                        state=traj_state_dict, action=0, rng=np.random.default_rng()
                    )
                    for _ in range(self.config.n_samples_wsd_mmd)
                ]

                samples_np = trajectories_to_numpy(samples)
                gt_samples_np = trajectories_to_numpy(gt_samples)

                wdist_total.append(wasserstein_distance_nd(samples_np, gt_samples_np))
                mmdist_total.append(
                    compute_mmd(samples_np, gt_samples_np, self.config.mmd_sigma)
                )

        mean_wdist = np.mean(wdist_total)
        mean_mmdist = np.mean(mmdist_total)

        results_dict["wasserstein_distance"] = mean_wdist
        results_dict["mmd_distance"] = mean_mmdist

        return results_dict

    def get_context(self, mode: str, id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Get the context information for given observation ids.

        Args:
            mode: 'training' or 'validation' to select the context for training or
            validation data.
            id: Tensor of shape [batch_size] containing the observation ids. If None,
            the context for all stored observations is returned.

        Returns:
            selected_context: Tensor of shape [batch_size, context_size] containing the
            context information.
        """

        if mode == "validation":
            if id is None:
                id = torch.arange(len(self._context_validation))

            selected_context = self._context_validation[id]

        elif mode == "training":
            if id is None:
                id = torch.arange(len(self._context_training))

            selected_context = self._context_training[id]

        else:
            raise ValueError("Mode must be 'training' or 'validation'.")

        assert selected_context.ndim == 2
        assert selected_context.shape[1] == self.context_size

        return selected_context

    def sample_context(self, n_samples: int) -> torch.Tensor:
        """
        Randomly sample SIR initial states to be used as context.

        Each context is an initial state [S0, I0, R0] sampled as
        S0 in [50, 100], I0 in [1, 5] and R0 = 0.

        Args:
            n_samples: Number of contexts to sample.

        Returns:
            sampled_context: Tensor of shape [n_samples, context_size] containing the
            sampled initial states.
        """

        sampled_S = torch.randint(50, 101, (n_samples, 1))
        sampled_I = torch.randint(1, 6, (n_samples, 1))
        sampled_R = torch.zeros((n_samples, 1), dtype=torch.int64)

        sampled_context = torch.cat((sampled_S, sampled_I, sampled_R), dim=-1)

        return sampled_context

    def simulation_wrapper(
        self,
        simulator: Callable,
        context: Any = None,
        params: torch.Tensor | None = None,
    ) -> Tensor:
        """
        Generate trajectories following the likelihood defined by a simulator.

        The simulation in this method is adapted from the code in

        https://github.com/samholt/generative-simulations/blob/72b5d51a7790b8b2ebc87973ee5e9d21aa818ece/libs/SIR/env.py

        mainly lines 87 - 98

        published by Samuel Holt under the MIT license.

        Args:
            simulator: Simulator used to generate data.
            context: Initial states [S0, I0, R0] used to seed each trajectory.
                Shape: (batch_size, 3).
            params: Tensor of shape [batch_size, 2] containing the parameters
                of the simulator.

        Returns:
            Tensor of shape [batch_size, 3 * time_steps] containing the generated
            raw trajectories in flattened form.
        """

        # Add batch dimension if necessary
        if params.ndim == 1:
            params = params.unsqueeze(0)

        # If a single parameter set is provided, repeat it for all contexts
        if params.shape[0] == 1:
            params = params.repeat(context.shape[0], 1)

        assert params.shape[0] == context.shape[0], (
            "Params and context must have the same batch size."
        )

        assert params.shape[1] == 2, "Parameters must have shape [batch_size, 2]."

        trajectories = torch.zeros((params.shape[0], self._num_time_steps, 3))

        for i in range(params.shape[0]):
            context_i = context[i].cpu()

            trajectories_i = torch.zeros((self._num_time_steps, 3))
            trajectories_i[0, :] = context_i

            simulator.set_parameters(parameters=params[i].cpu().numpy())

            for t in range(1, self._num_time_steps):
                state_i = {
                    "S": trajectories_i[t - 1][0].item(),
                    "I": trajectories_i[t - 1][1].item(),
                    "R": trajectories_i[t - 1][2].item(),
                }

                new_state = simulator.step(
                    state=state_i, action=0, rng=np.random.default_rng()
                )

                # Store the generated state
                trajectories_i[t][0] = new_state["S"]
                trajectories_i[t][1] = new_state["I"]
                trajectories_i[t][2] = new_state["R"]

            trajectories[i] = trajectories_i

        # Flatten the trajectories to match the shape of the stored observations
        trajectories_flat = trajectories.reshape(params.shape[0], -1)

        return trajectories_flat

    def plot_observation(
        self,
        valid_data: torch.Tensor | None = None,
        valid_data_raw: torch.Tensor | None = None,
        sim_data: torch.Tensor | None = None,
        sim_data_raw: torch.Tensor | None = None,
        path: str | None = None,
        fig_name: str = "observation.png",
        max_observation: int = 10,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot the validation observations and optionally overlay simulated data.

        Only the raw trajectories are plotted; the summary-statistic arguments
        (``valid_data``, ``sim_data``) are accepted for interface compatibility
        but not used. Delegates to :meth:`_plot_observation`.

        Args:
            valid_data: Validation summary statistics (unused).
            valid_data_raw: Raw validation trajectories.
                Shape: (batch_size, 3 * time_steps).
            sim_data: Simulated summary statistics (unused).
            sim_data_raw: Raw simulated trajectories to overlay (optional).
            path: Directory to save the plot in.
            fig_name: Filename for the saved plot.
            max_observation: Maximum number of trajectories to plot.
            kwargs: Additional task-specific plotting options (unused).
        """
        self._plot_observation(
            valid_data_raw=valid_data_raw,
            sim_data_raw=sim_data_raw,
            path=path,
            fig_name=fig_name,
            max_observation=max_observation,
        )

    def split_context_data(
        self,
        combined: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the combined context and data tensor back into data and context
        tensors. I.e. it is the inverse of the concatenate_context_data function.

        Args:
            combined: Combined data and context tensor. Shape: (batch_size,
                data_dim + context_size).

        Returns:
            data: Extracted data tensor. Shape: (batch_size, data_dim).
            context: Extracted context tensor. Shape: (batch_size, context_size).
        """

        assert combined.ndim == 2

        # The context is stored in the last context_size (three) dimensions of the
        # combined tensor.
        data = combined[:, : -self.context_size]
        context = combined[:, -self.context_size :]

        return data, context

    def concatenate_context_data(
        self,
        context: torch.Tensor | None,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate context and data into one tensor. The task has a context of size
        three. The context is contained in the final three dimensions of the
        concatenated tensor.

        Args:
            context: The context information. Shape: (batch_size, context_size).
            data: Data tensor. Shape: (batch_size, data_dim).

        Returns:
            combined: Combined data and context tensor.
        """

        assert data.ndim == context.ndim
        assert data.ndim == 2
        assert data.shape[0] == context.shape[0]

        combined = torch.cat((data, context), dim=-1)

        return combined

    def concatenate_context_params(
        self,
        context: torch.Tensor | None,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate context and parameters into one tensor. The task has a context of
        size three. The context is contained in the final three dimensions of the
        concatenated tensor.

        Args:
            context: The embedded context information. Shape: (batch_size, context_size)
            params: Parameters tensor. Shape: (batch_size, param_dim) with param_dim = 2

        Returns:
            combined: Combined parameters and context tensor.
        """

        # Add batch dimension if necessary
        if params.ndim == 1:
            params = params.unsqueeze(0)

        # If a single parameter set is provided, repeat it for all contexts
        if params.shape[0] == 1:
            params = params.repeat(context.shape[0], 1)

        assert params.shape[1] == 2, "Parameters must have shape [batch_size, 2]."
        assert params.ndim == context.ndim
        assert params.ndim == 2
        assert params.shape[0] == context.shape[0]

        combined = torch.cat((params, context), dim=-1)

        return combined
