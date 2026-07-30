########################################################################################
#
# This file wraps external code (see `modelsmc/tasks/allen/external_code/`) adapted
# from https://github.com/mackelab/IdentifyMechanisticModels_2020/tree/master/6_allen
# (MIT-licensed, see that directory's file headers for the full license text)
# implementing a Hodgkin-Huxley neuron simulator and its summary statistics.
#
# The observations loaded via `get_data`/`get_data_single` are electrophysiology
# recordings from the Allen Cell Types Database (Allen Institute for Brain
# Science, 2015), available from celltypes.brain-map.org.
#
########################################################################################

import logging
import os
from copy import deepcopy
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from torch import Tensor

import modelsmc.tasks.allen.external_code.allen_utils as allen_utils
from modelsmc.tasks.allen.external_code.allen_summary_stats import (
    HodgkinHuxleyStatsMoments,
)
from modelsmc.tasks.base_embedding_net import (
    FixedEmbeddingHandler,
)
from modelsmc.tasks.base_task import BaseTask

logger = logging.getLogger("ModelSMC")


class FixedSummaryStatisticsAllen(nn.Identity):
    """Fixed (non-learnable) summary statistics for Allen data set observations.

    Wraps the hand-crafted Hodgkin-Huxley summary statistics so they can be used
    as a fixed embedding function for the observations. It has no learnable
    parameters.
    """

    def __init__(
        self,
        dt: float,
        t: Tensor,
        t_on: float,
        t_off: float,
        n_xcorr: int,
        n_mom: int,
        n_summary: int,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the fixed summary-statistics embedding.

        Args:
            dt: Time-step size of the recordings.
            t: Time axis of the recordings.
            t_on: Time at which the input current turns on.
            t_off: Time at which the input current turns off.
            n_xcorr: Number of auto-correlation summary statistics.
            n_mom: Number of moment summary statistics.
            n_summary: Total number of summary statistics returned.
        """
        self.dt = dt
        self.t = t
        logger.debug("Initialize a fixed embedding model for the Allen model.")
        super().__init__(*args, **kwargs)
        self.stats = HodgkinHuxleyStatsMoments(
            t_on=t_on,
            t_off=t_off,
            n_xcorr=n_xcorr,
            n_mom=n_mom,
            n_summary=n_summary,
        )

    def forward(
        self, obs: Tensor, dt: float | None = None, t: Tensor | None = None
    ) -> Tensor:
        """Calculate summary statistics for observations."""
        obs = obs.detach().cpu().numpy()
        obs_list = []
        for i in range(obs.shape[0]):
            obs_list.append(
                {
                    "data": obs[i].reshape(
                        -1,
                    ),
                    "dt": self.dt,
                    "time": self.t,
                }
            )

        summary_stats = self.stats.calc(obs_list)
        return torch.tensor(summary_stats, dtype=torch.float32)


class AllenEmbeddingHandler(FixedEmbeddingHandler):
    """Context-aware embedding handler for the Allen task.

    Uses fixed summary statistics to embed the observations and the identity
    function to embed the context.
    """

    def __init__(
        self,
        split_x_ctxt: Callable,
        dt: float,
        t: Tensor,
        t_on: float,
        t_off: float,
        n_xcorr: int,
        n_mom: int,
        n_summary: int,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the context-aware embedding handler.

        Args:
            split_x_ctxt: Callable that splits a combined tensor into data and
                context.
            dt: Time-step size of the recordings.
            t: Time axis of the recordings.
            t_on: Time at which the input current turns on.
            t_off: Time at which the input current turns off.
            n_xcorr: Number of auto-correlation summary statistics.
            n_mom: Number of moment summary statistics.
            n_summary: Total number of summary statistics returned.
        """

        logger.debug(
            "Initialize a context aware embedding model for periodic time series."
        )
        super().__init__(split_x_ctxt=split_x_ctxt)

        # Update the observation embedding network to use fixed summary statistics
        # Embedding function for embedding the observations
        self.obs_embedding_function = FixedSummaryStatisticsAllen(
            dt=dt,
            t=t,
            t_on=t_on,
            t_off=t_off,
            n_xcorr=n_xcorr,
            n_mom=n_mom,
            n_summary=n_summary,
        )

        # Embedding function for embedding the context
        self.context_embedding_function = nn.Identity()


class AllenBase(BaseTask):
    """Base task for the Allen (Hodgkin-Huxley) cell simulator.

    Loads recordings for up to 10 Allen cells, derives the simulation context
    (input current and initial voltage) from them, and embeds observations with
    fixed Hodgkin-Huxley summary statistics. Concrete levels subclass this and
    define the prior over the simulator parameters.
    See the module docstring above for the source of the model code and data.
    """

    def __init__(
        self,
        config: DictConfig,
        prompts_path: str | None = None,
        base_simulator_path: str | None = None,
    ) -> None:
        """Initialize the Allen task.

        Args:
            config: Hydra configuration containing task settings.
            prompts_path: Path to the prompts.yaml file for this task (required).
            base_simulator_path: Path to the base_simulator.py file for this task
                (required).
        """
        assert prompts_path is not None, "prompts_path is required"
        assert base_simulator_path is not None, "base_simulator_path is required"

        self.n_xcorr = config.get("n_xcorr", 0)
        self.n_mom = config.get("n_mom", 4)
        self.n_summary = config.get("n_summary", 7)

        # Data location
        self.dir_data = os.path.join(os.path.dirname(__file__), "data")

        super().__init__(config)

        self.num_obs_train = config["num_obs_train"]
        self.num_obs_valid = config["num_obs_valid"]
        assert self.num_obs_train + self.num_obs_valid <= 10, (
            "num_obs_train + num_obs_valid must be less than or equal to 10"
        )

        # load prompts
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
        self.system_description = prompts["system_description"]
        self.signature_description = prompts["signature_description"]
        self.task_description = prompts["task_description"]

        # load base simulator code from file
        with open(base_simulator_path, "r") as f:
            self.base_simulator = f.read()

        # Initialize the embedding network for observations
        self.init_embedding_net()

    def init_embedding_net(self) -> None:
        """Initialize the network used to embed observations.

        The Allen task always uses fixed (non-learnable) Hodgkin-Huxley summary
        statistics; requesting learnable summary statistics is not supported.
        """
        logger.debug("Reset embedding net")

        _, I, dt, t_on, t_off = self.get_data_single(1)  # noqa E741
        t = torch.tensor(np.arange(0, len(I), 1) * dt, dtype=torch.float32)

        # Learnable summary statistics are not implemented for the Allen task.
        if self.config.get("learnable_summary_statistics", False):
            raise NotImplementedError(
                "Learnable embedding net for Allen data not implemented"
            )

        logger.debug("Initialize a fixed embedding function")
        self.embedding_net = AllenEmbeddingHandler(
            dt=dt,
            t=t,
            t_on=t_on,
            t_off=t_off,
            n_xcorr=self.n_xcorr,
            n_mom=self.n_mom,
            n_summary=self.n_summary,
            split_x_ctxt=self.split_context_data,
        )

    def eval_function(
        self,
        sim_data: Tensor,
        sim_data_raw: Tensor,
        valid_data: Tensor,
        valid_data_raw: Tensor,
        parameter_estimates: Tensor,
        simulator: torch.nn.Module,
    ) -> dict[str, float | list[float]]:
        """Compute the (variance-normalized) mean squared error between the
        simulated and validation summary statistics.

        Returns:
            Dictionary with the scalar ``mse`` and the per-dimension
            ``mse_per_dim`` list.
        """
        metric_dict = {}

        num_dims = sim_data.shape[-1]
        weights = torch.ones(num_dims)

        # Weighted squared error
        squared_error = (sim_data - valid_data) ** 2
        weighted_squared_error = squared_error * weights

        # Normalize the MSE for each dimension to account for different scales
        # Normalize by the value of the validation data to get a relative error
        eps = 1e-10
        variance = torch.var(valid_data, dim=0) + eps
        assert variance.shape[0] == weighted_squared_error.shape[1]

        weighted_squared_error = weighted_squared_error / variance

        mse = torch.mean(weighted_squared_error)
        mse_per_dim = torch.mean(weighted_squared_error, dim=0)

        metric_dict["mse"] = mse.item()
        metric_dict["mse_per_dim"] = mse_per_dim.tolist()

        return metric_dict

    def get_data(self) -> tuple[Tensor, Tensor]:
        """Load the Allen cell recordings and derive the simulation context.

        Populates the per-cell context attributes (input current and initial
        voltage) used by :meth:`get_context`, :meth:`sample_context` and
        :meth:`simulation_wrapper`, and returns the raw training and validation
        observations. As individual parameters are inferred per observation, the
        training data is a copy of the validation data.

        Returns:
            Tuple of (train_data_raw, valid_data_raw), each of shape
            (num_obs, time_steps).
        """

        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = []

        # The input currents all have the same start and end time and the same values.
        # Only store the max value.

        self.__context_I_val_high = torch.zeros(len(ids))
        self.__context_initial_voltage = torch.zeros(len(ids))

        dt_list = []
        step_off_list = []
        step_on_list = []
        min_val_list = []
        n_timesteps_list = []

        for id in ids:
            obs, I, dt, t_on, t_off = self.get_data_single(id)  # noqa E741
            data.append(torch.tensor(obs))

            # Store the maximum value of the input current for each observation
            self.__context_I_val_high[id - 1] = I.max().item()

            # Store the initial voltage for each observation
            self.__context_initial_voltage[id - 1] = obs[0].item()

            # Get the size of a time step
            dt_list.append(dt)

            # Get the minimum value of the input current
            min_val_list.append(I.min().item())

            # Get the number of time steps
            n_timesteps_list.append(len(I))

            # Get the index for t_on and t_off

            # Get the index where the step current turns on
            step_on_idx = I.nonzero()[0][0]
            step_on_list.append(step_on_idx)

            # Get the index where the step current turns off
            step_off_idx = I.nonzero()[0][-1]
            step_off_list.append(step_off_idx)

        unique_dt = set(dt_list)
        unique_t_on = set(step_on_list)
        unique_t_off = set(step_off_list)
        unique_n_timesteps = set(n_timesteps_list)
        unique_min_val = set(min_val_list)

        assert len(unique_dt) == 1, "All dt values must be the same"
        assert len(unique_t_on) == 1, "All t_on values must be the same"
        assert len(unique_t_off) == 1, "All t_off values must be the same"
        assert len(unique_min_val) == 1, "All min_val values must be the same"
        assert len(unique_n_timesteps) == 1, "All n_timesteps values must be the same"

        self.__context_dt = unique_dt.pop()
        self.__context_I_val_low = unique_min_val.pop()
        self.__context_t_on = unique_t_on.pop()
        self.__context_t_off = unique_t_off.pop()
        self.__context_n_timesteps = unique_n_timesteps.pop()

        logger.debug(f"Context dt: {self.__context_dt}")
        logger.debug(f"Context I_val_low: {self.__context_I_val_low}")
        logger.debug(f"Context t_on: {self.__context_t_on}")
        logger.debug(f"Context t_off: {self.__context_t_off}")
        logger.debug(f"Context n_timesteps: {self.__context_n_timesteps}")
        logger.debug(f"Context I_val_high: {self.__context_I_val_high}")

        # Store the context size
        self.context_size = 2  # [I_val_high, initial_voltage]

        data = torch.stack(data)
        # Use the cells in a fixed order (no shuffling) so the split is deterministic.
        indices = torch.arange(self.num_obs_train + self.num_obs_valid)
        valid_indices = indices[self.num_obs_train :]
        valid_data_raw = data[valid_indices]

        if self.config.plotting:
            self._plot_observation(
                valid_data_raw=valid_data_raw,
                path=self.save_dir,
                fig_name="observation.png",
            )

        # For this task, training data (i.e. the data used to obtain a parameter
        # estimate) is the same as the validation data (i.e. the data used to
        # evaluate the parameter estimate) since we want to infer an individual
        # parameter set for each observation.
        train_data_raw = deepcopy(valid_data_raw)

        return train_data_raw, valid_data_raw

    def get_data_single(self, id: int) -> tuple:
        """Load the recording and stimulus for a single Allen cell.

        Returns:
            Tuple of (data, I, dt, t_on, t_off): the membrane-potential trace,
            the input current, the time-step size, and the on/off times of the
            step current.
        """
        obs_idx, sweep_num, A_soma = allen_utils.get_allen_task_parameters(id)
        obs = allen_utils.allen_obs_data(
            obs_idx, sweep_num, A_soma, dir_data=self.dir_data
        )
        self.JUNCTION_POTENTIAL = -14
        obs["data"] = obs["data"] + self.JUNCTION_POTENTIAL
        data = obs["data"]
        I = obs["I"]  # noqa E741
        dt = obs["dt"]
        t_on = obs["t_on"]
        t_off = obs["t_off"]
        return data, I, dt, t_on, t_off

    def simulation_wrapper(
        self,
        simulator: Callable,
        context: Tensor | None = None,
        params: Tensor | None = None,
    ) -> Tensor:
        """Simulate the Allen model for a batch of parameters.

        Args:
            simulator: Simulator to run.
            context: Per-observation context of shape (batch_size, 2), holding
                the max input current value and the initial voltage.
            params: Parameter tensor. Shape: (batch_size, n_params).

        Returns:
            The simulated membrane-potential traces.
        """
        n_batches = params.shape[0]

        # Split the context into initial voltage and input current max value
        initial_voltage = context[:, 1]  # initial voltage
        I_val_high = context[:, 0]  # max input current value

        # Get the time steps
        t = torch.tensor(
            np.arange(0, self.__context_n_timesteps, 1) * self.__context_dt,
            dtype=torch.float32,
        )

        traces = torch.zeros(n_batches, self.__context_n_timesteps, dtype=torch.float32)

        I = torch.zeros(n_batches, self.__context_n_timesteps, dtype=torch.float32)  # noqa E741
        I[:, self.__context_t_on : self.__context_t_off] = I_val_high.unsqueeze(1)

        traces = simulator(initial_voltage, I, self.__context_dt, t, params)

        assert traces.shape[0] == n_batches, (
            "traces must have the same batch size as params"
        )
        return traces

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
        """Plot the validation observations and optionally the simulated traces."""
        self._plot_observation(
            sim_data_raw=sim_data_raw,
            valid_data_raw=valid_data_raw,
            path=path,
            fig_name=fig_name,
            max_observation=max_observation,
        )

    def get_context(self, mode: str, id: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return any task-specific context needed for simulations. The returned context
        needs to contain numerical values only. Since the training data is just a
        copy of the validation data, the context is the same for both modes, hence the
        'mode' argument is not used here.

        Args:
            mode: Mode for which to get the context ('training' or 'validation').
            id: The identifier for the context to retrieve.

        Returns:
            context: The context corresponding to the provided id(s).
        """

        # If no ids are provided, return the context for all validation observations
        if id is None:
            id = torch.arange(self.config.num_obs_valid)

        I_val_high = self.__context_I_val_high[id].unsqueeze(1)
        initial_voltage = self.__context_initial_voltage[id].unsqueeze(1)

        # Merge the two context variables into a single tensor
        # [I_val_high, initial_voltage]
        context = torch.cat([I_val_high, initial_voltage], dim=1)
        return context

    def sample_context(self, n_samples: int) -> torch.Tensor:
        """Randomly sample contexts from the available context variables.

        The first value is the maximum input current value, the second value is
        the initial voltage. The two components are sampled independently to
        increase the diversity of the sampled contexts.

        Args:
            n_samples: Number of contexts to sample.

        Returns:
            context: Sampled contexts of shape (n_samples, 2).
        """

        idx = torch.randint(0, self.__context_I_val_high.shape[0], (n_samples,))
        I_val_high = self.__context_I_val_high[idx].unsqueeze(1)

        idx = torch.randint(0, self.__context_I_val_high.shape[0], (n_samples,))
        initial_voltage = self.__context_initial_voltage[idx].unsqueeze(1)

        # Merge the two context variables into a single tensor
        # [I_val_high, initial_voltage]
        context = torch.cat([I_val_high, initial_voltage], dim=1)

        logger.debug(f"Sampled context shape: {context.shape}")

        return context

    def concatenate_context_data(
        self,
        context: torch.Tensor | None,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate context and data into one tensor. The task has a context of size
        two. The context is contained in the final two dimensions of the concatenated
        tensor.

        Args:
            context: The context information.
            data: Data tensor. Shape: (batch_size, n_params)
        Returns:
            combined: Combined context and data tensor.
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
        size two. The context is contained in the final two dimensions of the
        concatenated tensor.

        Args:
            context: The context information.
            params: Parameters tensor. Shape: (batch_size, n_params)

        Returns:
            combined: Combined context and parameters tensor.
        """

        assert params.ndim == context.ndim
        assert params.ndim == 2
        assert params.shape[0] == context.shape[0]

        combined = torch.cat((params, context), dim=-1)

        return combined

    def split_context_data(
        self,
        combined: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the combined data and context tensor back into its parts. This is the
        inverse of the concatenate_context_data function.

        Args:
            combined: Combined data and context tensor.

        Returns:
            data: Extracted data tensor.
            context: Extracted context tensor (the final ``context_size``
                dimensions).
        """

        assert combined.ndim == 2

        # The context is stored in the last two dimensions of the combined tensor
        data = combined[:, : -self.context_size]
        context = combined[:, -self.context_size :]

        return data, context

    @staticmethod
    def _plot_observation(
        valid_data_raw: Tensor,
        sim_data_raw: Tensor | None = None,
        path: str | None = None,
        fig_name: str = "observation.png",
        max_observation: int = 10,
    ) -> None:
        """Plot validation (and optional simulated) membrane-potential traces."""
        os.makedirs(path, exist_ok=True)

        # Calculate subplot layout
        max_rows = 5
        num_cols = (max_observation + max_rows - 1) // max_rows  # Ceiling division
        num_rows = min(max_observation, max_rows)

        # Extract voltage data from the first tuple element (obs, I)
        if isinstance(valid_data_raw[0], tuple):
            voltage_data = [item[0] for item in valid_data_raw]
        else:
            voltage_data = valid_data_raw

        # Create time axis based on the length of the first observation
        t = torch.arange(0, len(voltage_data[0]), 1)

        if len(voltage_data) == 1:
            # Handle single observation case
            plt.figure(figsize=(10, 5))
            plt.plot(
                t.reshape(-1),
                voltage_data[0].reshape(-1),
                label="True",
                color="black",
                linestyle="--",
            )
            if sim_data_raw is not None:
                sim_voltage = (
                    sim_data_raw[0]
                    if isinstance(sim_data_raw[0], tuple)
                    else sim_data_raw
                )
                if isinstance(sim_voltage, tuple):
                    sim_voltage = sim_voltage[0]
                plt.plot(
                    t.reshape(-1),
                    sim_voltage.reshape(-1),
                    label="Simulated",
                    color="blue",
                )
            plt.xlabel("Time steps")
            plt.ylabel("Membrane Potential (mV)")
            plt.title("Allen Cell Observation")
            plt.legend()
            plt.savefig(os.path.join(path, fig_name))
            plt.close()
            return
        else:
            # Handle batch observations case
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True
            )

            # Flatten axes for easy indexing, regardless of shape
            if num_rows == 1 and num_cols == 1:
                axes = [axes]
            elif num_rows == 1 or num_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i in range(min(max_observation, len(voltage_data))):
                ax = axes[i]
                ax.plot(
                    t.reshape(-1),
                    voltage_data[i].reshape(-1),
                    label="True",
                    color="black",
                    linestyle="--",
                )
                if sim_data_raw is not None and i < len(sim_data_raw):
                    sim_voltage = (
                        sim_data_raw[i]
                        if isinstance(sim_data_raw[i], tuple)
                        else sim_data_raw[i]
                    )
                    if isinstance(sim_voltage, tuple):
                        sim_voltage = sim_voltage[0]
                    ax.plot(
                        t.reshape(-1),
                        sim_voltage.reshape(-1),
                        label="Simulated",
                        color="blue",
                    )
                if i == 0:
                    ax.legend()
                ax.set_title(f"Cell {i + 1}")

            fig.supylabel("Membrane Potential (mV)")
            fig.supxlabel("Time steps")
            fig.suptitle("Allen Cell Data")

            # Hide unused subplots if any
            for j in range(min(max_observation, len(voltage_data)), len(axes)):
                fig.delaxes(axes[j])

            # Save figure
            plt.savefig(os.path.join(path, fig_name))
            plt.close()
            return
