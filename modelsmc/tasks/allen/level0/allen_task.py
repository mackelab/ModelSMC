import os

import numpy as np
import torch
from omegaconf import DictConfig
from sbi.utils import BoxUniform

from modelsmc.tasks.allen.allen_task_base import AllenBase
from modelsmc.tasks.allen.external_code.allen_utils import param_transform
from modelsmc.tasks.base_task import register_task


@register_task("allen_level0")
class AllenLevel0(AllenBase):
    """Allen (Hodgkin-Huxley) task, level 0: defines the full prior over the
    simulator parameters and loads this level's prompts and base simulator."""

    def __init__(self, config: DictConfig) -> None:
        dir_path = os.path.dirname(__file__)
        prompts_path = os.path.join(dir_path, "prompts.yaml")
        base_simulator_path = os.path.join(dir_path, "base_simulator.py")

        super().__init__(config, prompts_path, base_simulator_path)

        self.prior_dist = self.prior()

    def prior(self, prior_log: bool = False) -> BoxUniform:
        """Build the uniform prior over the Hodgkin-Huxley simulator parameters.

        Args:
            prior_log: If True, the bounds are transformed to log space via
                ``param_transform``.

        Returns:
            A BoxUniform prior over the simulator parameters.
        """
        range_lower = param_transform(
            prior_log,
            np.array(
                [
                    0.5,  # gbar_Na: minimum sodium channel conductance
                    1e-4,  # gbar_K: minimum potassium channel conductance
                    1e-4,  # g_leak: minimum leak channel conductance
                    35.0,  # E_leak: minimum leak reversal potential
                    40.0,  # Vt: minimum threshold voltage
                    1e-4,  # nois_fact: minimum noise factor
                    # additional channel: conductances
                    1e-4,  # gbar_X1: minimum additional conductance
                    1e-4,  # gbar_X2: minimum additional conductance
                    # additional channel: parameters
                    1e-4,  # param_i
                    1e-4,  # param_j
                ]
            ),
        )

        range_upper = param_transform(
            prior_log,
            np.array(
                [
                    80.0,  # gbar_Na: maximum sodium channel conductance
                    15.0,  # gbar_K: maximum potassium channel conductance
                    0.6,  # g_leak: maximum leak channel conductance
                    100.0,  # E_leak: maximum leak reversal potential
                    90.0,  # Vt: maximum threshold voltage
                    0.15,  # nois_fact: maximum noise factor
                    # additional channel: conductances
                    10,  # gbar_X1: maximum additional conductance
                    120,  # gbar_X2: maximum additional conductance
                    # additional channel: parameters
                    150,  # param_i
                    3000,  # param_j
                ]
            ),
        )

        prior_min = torch.tensor(range_lower.astype(np.float32))
        prior_max = torch.tensor(range_upper.astype(np.float32))
        return BoxUniform(prior_min, prior_max)
