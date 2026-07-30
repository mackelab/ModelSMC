import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks_utils import MultipleIndependent
from torch import Tensor
from torch.distributions import Distribution

from modelsmc.tasks.base_task import BaseTask, register_task
from modelsmc.tasks.minimal_example_n_dim.generate_models import GMMConfigs
from modelsmc.tasks.minimal_example_n_dim.GMM import GMM

logger = logging.getLogger("ModelSMC")

# Load GMM configurations
try:
    BASE_DIR = Path(__file__).resolve().parent
    GMM_configs = GMMConfigs.model_validate_json(
        open(f"{BASE_DIR}/gmm_configs.json").read()
    )
except FileNotFoundError:
    logger.error(
        "GMM configurations not found. Please run "
        "'python modelsmc/tasks/minimal_example_n_dim/generate_models.py --new_config' "
        "to generate new configurations."
    )

# Get the number of dimensions to shift
num_dims_to_shift = len(GMM_configs.configs[0].dims_to_shift)
logger.debug(f"Number of dimensions to shift: {num_dims_to_shift}")


@register_task("minimal_example_n_dim")
class MinimalExampleNDim(BaseTask):
    """Minimal n-dimensional GMM example task (used for the LLM-free experiment)."""

    def __init__(
        self,
        config: DictConfig,
        prompts_path: str | None = None,
        base_simulator_path: str | None = None,
    ) -> None:
        super().__init__(config, prompts_path, base_simulator_path)

        # Prior distribution
        self.prior_dist = self.get_prior_distribution()

        # This task is LLM-free, so the prompt/description fields are unused.
        self.task_description = ""
        self.signature_description = ""
        self.system_description = ""
        self.base_simulator = ""

        # Set the number of training instances and validation instances
        self.num_obs_train = config["num_obs_train"]
        self.num_obs_valid = config["num_obs_valid"]

        # Get the identifier for the ground truth GMM configuration
        self.gt_GMM_configuration_index = config.gt_GMM_configuration_index

        self.num_models = len(GMM_configs.configs)

        dir_path = os.path.dirname(__file__)
        base_simulator_path = os.path.join(dir_path, "base_simulator.py")

        with open(base_simulator_path, "r") as file:
            self.__skeleton_implementation = file.read()

    def get_skeleton_implementation(self) -> str:
        """Return the base simulator skeleton code for the LLM to complete."""
        return deepcopy(self.__skeleton_implementation)

    @staticmethod
    def get_prior_distribution() -> Distribution:
        """
        Get the prior distribution over the per-dimension shifts and the global scale.

        Returns:
            A MultipleIndependent prior with a BoxUniform(-2, 2) over each shifted
            dimension and a BoxUniform(0.1, 2) over the global scale.
        """
        # Get the prior distribution for the shift in each dimension
        dist_list = [
            BoxUniform(low=torch.tensor([-2.0]), high=torch.tensor([2.0]))
            for _i in range(num_dims_to_shift)
        ]

        # Get the prior distribution for the global scale
        dist_list.append(BoxUniform(low=torch.tensor([0.1]), high=torch.tensor([2.0])))

        prior_distribution = MultipleIndependent(dist_list)

        return prior_distribution

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

        No task-specific metrics are computed for the minimal example.
        """
        metric_dict = {}
        return metric_dict

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training and validation data following the true likelihood.

        For this task the training and validation data are identical, since an
        individual parameter set is inferred for each observation.

        Returns:
            Tuple of (train_data_raw, valid_data_raw), each of shape
            (num_obs_valid, dim).
        """

        logger.info(
            f"Obtaining true observations: {self.num_obs_train} training, "
            f"{self.num_obs_valid} validation"
        )

        # Ground-truth parameters: zero shift on every dimension and unit global scale
        theta = [0.0 for _ in range(num_dims_to_shift)] + [1.0]
        theta_gt = torch.Tensor(theta).unsqueeze(0)
        assert theta_gt.shape == torch.Size([1, 1 + num_dims_to_shift])

        # Get the ground truth GMM
        gt_GMM = self.get_model(self.gt_GMM_configuration_index)

        samples = gt_GMM.sample(
            num_samples=self.num_obs_train + self.num_obs_valid, theta=theta_gt
        )

        # Simple random split
        indices = torch.randperm(self.num_obs_train + self.num_obs_valid)
        valid_indices = indices[self.num_obs_train :]

        # Train and test split
        valid_data_raw = samples[valid_indices]

        # For this task, training data (i.e. the data used to obtain a parameter
        # estimate) is the same as the validation data (i.e. the data used to
        # evaluate the parameter estimate) since we want to infer an individual
        # parameter set for each observation.
        train_data_raw = deepcopy(valid_data_raw)

        return train_data_raw, valid_data_raw

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
        """Plotting is not implemented for n-dimensional GMMs (no-op)."""
        logger.debug("Plotting observations not implemented for n-dimensional GMMs.")
        return  # Plotting disabled for n-dimensional case

    @staticmethod
    def get_model(model_idx: int) -> GMM:
        """Build the GMM for the given configuration index.

        Args:
            model_idx: Index into the loaded GMM configurations.

        Returns:
            The GMM instance for the selected configuration.
        """
        # Collect parameters of the GMM
        config_idx = GMM_configs.configs[model_idx]
        covs_list = [torch.tensor(cov) for cov in config_idx.covs]
        means_list = [torch.tensor(mean) for mean in config_idx.means]
        weights_list = config_idx.weights
        dims_to_shift = config_idx.dims_to_shift

        # Initialize the GMM
        gmm = GMM(
            covs_list=covs_list,
            means_list=means_list,
            dims_to_shift=dims_to_shift,
            weights_list=weights_list,
        )

        return gmm

    def simulation_wrapper(
        self,
        simulator: Callable,
        context: Any = None,
        params: torch.Tensor | None = None,
    ) -> Tensor:
        """Run the simulator with the given parameters.

        Args:
            simulator: Simulator to run.
            context: Unused for this task.
            params: Parameter tensor. Shape: (batch_size, n_params).

        Returns:
            The simulated data.
        """
        return simulator(theta=params)
