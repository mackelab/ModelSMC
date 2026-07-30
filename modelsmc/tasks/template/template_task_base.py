import logging
from typing import Any, Callable

import torch
import yaml
from omegaconf import DictConfig
from sbi.utils.user_input_checks_utils import MultipleIndependent

from modelsmc.tasks.base_task import BaseTask

logger = logging.getLogger("ModelSMC")


class TemplateBase(BaseTask):
    """Template task illustrating the structure required to implement a new task.

    Copy this file as a starting point and fill in the ``# TODO`` sections. The
    abstract methods (``eval_function``, ``get_data``, ``simulation_wrapper`` and
    ``plot_observation``) must be implemented; the remaining task interface is
    inherited from :class:`BaseTask`. See an existing task (e.g. the SIR task) for
    a complete reference implementation.
    """

    def __init__(
        self,
        config: DictConfig,
        prompts_path: str | None = None,
        base_simulator_path: str | None = None,
    ) -> None:
        """Initialize the template task.

        Args:
            config: Hydra configuration containing task settings.
            prompts_path: Path to the prompts.yaml file for this task.
            base_simulator_path: Path to the base_simulator.py file for this task.
        """
        super().__init__(config, prompts_path, base_simulator_path)

        # Required: define the prior over the parameters to be inferred. Replace the
        # bounds below with the task-specific parameter ranges.
        self.prior_dist = MultipleIndependent(
            [
                # BoxUniform(low=torch.tensor([...]), high=torch.tensor([...])),
            ]
        )

        # Load the prompts and base simulator from the provided paths.
        if prompts_path is not None:
            with open(prompts_path, "r") as f:
                prompts = yaml.safe_load(f)
            self.system_description = prompts["system_description"]
            self.signature_description = prompts["signature_description"]
            self.task_description = prompts["task_description"]

        if base_simulator_path is not None:
            with open(base_simulator_path, "r") as f:
                self.base_simulator = f.read()

    def eval_function(
        self,
        sim_data: torch.Tensor,
        sim_data_raw: torch.Tensor,
        valid_data: torch.Tensor,
        valid_data_raw: torch.Tensor,
        parameter_estimates: torch.Tensor,
        simulator: torch.nn.Module,
    ) -> dict[str, float]:
        """Evaluate the simulator against the validation data.

        Args:
            sim_data: Summary statistics of the simulated data.
                Shape: (batch_size, n_summary_stats).
            sim_data_raw: Raw simulation outputs. Shape: (batch_size, time_steps).
            valid_data: Summary statistics of the validation data.
                Shape: (batch_size, n_summary_stats).
            valid_data_raw: Raw validation data. Shape: (batch_size, time_steps).
            parameter_estimates: Parameters estimated by the inference method.
                Shape: (batch_size, n_params).
            simulator: The simulator module used to generate the simulated data.

        Returns:
            Dictionary with metric names as keys and their scalar metric values.
        """
        # TODO: Compute evaluation metrics (e.g. MSE, MAE) and return them, e.g.:
        #   mse = torch.mean((sim_data_raw - valid_data_raw) ** 2)
        #   return {"mse": mse.item()}
        pass

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the training and validation data.

        Returns:
            Tuple of (train_data_raw, valid_data_raw), each a tensor of raw
            observations with shape (n_samples, time_steps) or task-specific.
        """
        # TODO:
        # 1. Run the ground-truth simulator to generate observations, or load
        #    pre-generated data from disk.
        # 2. Split the observations into training and validation sets.
        # 3. Store any task-specific context attributes needed by get_context()
        #    and simulation_wrapper().
        # 4. Return the raw training and validation tensors.
        pass

    def plot_observation(
        self,
        valid_data: torch.Tensor | None = None,
        valid_data_raw: torch.Tensor | None = None,
        sim_data: torch.Tensor | None = None,
        sim_data_raw: torch.Tensor | None = None,
        path: str | None = None,
        fig_name: str = "observation.png",
        max_observation: int = 5,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot the validation observations and optionally overlay simulated data.

        Args:
            valid_data: Validation summary statistics (optional).
            valid_data_raw: Raw validation data for plotting (optional).
            sim_data: Simulated summary statistics (optional).
            sim_data_raw: Raw simulated data for plotting (optional).
            path: Directory to save the plot in.
            fig_name: Filename for the saved plot.
            max_observation: Maximum number of traces to plot.
            kwargs: Additional task-specific plotting options (optional).
        """
        # TODO: Create matplotlib plots showing:
        # - the raw validation traces (up to max_observation),
        # - the simulated traces overlaid when sim_data_raw is provided,
        # - and save the figure to os.path.join(path, fig_name).
        pass

    def simulation_wrapper(
        self,
        simulator: Callable,
        context: Any = None,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the simulator with the task-specific interface.

        Args:
            simulator: Simulator function or class to run.
            context: Additional context for the simulation, such as initial
                conditions (if applicable).
            params: Parameter tensor. Shape: (batch_size, n_params).

        Returns:
            Tensor of raw simulation traces. Shape: (batch_size, time_steps).
        """
        # TODO:
        # 1. Prepare the simulator inputs (time grid, initial conditions, etc.).
        # 2. Call the simulator with the parameters (and context if applicable).
        # 3. Return the raw traces.
        pass
