import os

import torch
from omegaconf import DictConfig
from torch import Tensor

from modelsmc.tasks.base_task import register_task
from modelsmc.tasks.SIR.SIR_task_base import SIRBaseTask


@register_task("SIR_level3")
class SIRLevel3(SIRBaseTask):
    """SIR Level 3: loads pre-generated data from disk instead of simulating at runtime.

    The training and validation data must be generated once beforehand by running:
        python -m modelsmc.tasks.SIR.level3.generate_data --seed 42
    """

    def __init__(self, config: DictConfig) -> None:
        # Load the prompts and base simulator from this level's directory
        level_dir = os.path.dirname(__file__)
        prompts_path = os.path.join(level_dir, "prompts.yaml")
        base_simulator_path = os.path.join(level_dir, "base_simulator.py")
        super().__init__(config, prompts_path, base_simulator_path)

    def get_data(self) -> tuple[Tensor, Tensor]:
        """Load the pre-generated training and validation data from ``level3/data/``.

        Populates the context attributes required by ``SIRBaseTask`` (see its
        docstring) and returns the raw, flattened trajectories.

        Returns:
            Tuple of (train_data_raw, valid_data_raw), each of shape
            (num_instances, 3 * time_steps).

        Raises:
            FileNotFoundError: If the pre-generated data files do not exist yet.
        """

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        train_path = os.path.join(data_dir, "train_data.pt")
        valid_path = os.path.join(data_dir, "valid_data.pt")

        if not os.path.exists(train_path) or not os.path.exists(valid_path):
            raise FileNotFoundError(
                f"Pre-generated data not found in {data_dir}. "
                "Run: python -m modelsmc.tasks.SIR.level3.generate_data --seed 42"
            )

        # Load data: shape [num_instances, time_steps, 3]
        train_data = torch.load(train_path, weights_only=True)
        val_data = torch.load(valid_path, weights_only=True)

        # Flatten to [num_instances, time_steps * 3]
        train_data_raw = train_data.reshape(train_data.shape[0], -1)
        valid_data_raw = val_data.reshape(val_data.shape[0], -1)

        # Plot the validation data
        if self.config.plotting:
            self._plot_observation(
                valid_data_raw=valid_data_raw,
                path=self.save_dir,
                fig_name="observation.png",
            )

        # Store context (initial state [S0, I0, R0]). These attributes are read by
        # get_context() and simulation_wrapper() in SIRBaseTask (see the SIRBaseTask
        # docstring).
        self.context_size = 3
        self._context_validation = val_data[:, 0, :]
        self._context_training = train_data[0, 0, :].unsqueeze(0)

        assert self._context_validation.shape[1] == self._context_training.shape[1]
        assert self._context_validation.shape[1] == self.context_size
        assert self._context_validation.shape[0] == valid_data_raw.shape[0]
        assert self._context_training.shape[0] == 1
        assert self._context_training.shape[1] == self.context_size

        # Store the number of time steps
        self._num_time_steps = val_data.shape[1]

        assert train_data_raw.shape[0] == 1, (
            "Only a single training instance is supported."
        )

        return train_data_raw, valid_data_raw
