import os

from omegaconf import DictConfig

from modelsmc.tasks.allen.allen_task_base import AllenBase
from modelsmc.tasks.allen.level0.allen_task import AllenLevel0
from modelsmc.tasks.base_task import register_task


@register_task("allen_level_ablation_minprompt")
class AllenLevelAblationMinprompt(AllenLevel0):
    """Allen (Hodgkin-Huxley) task, minimal-prompt ablation: reuses
    AllenLevel0's prior over the simulator parameters and base simulator, but
    loads this level's own (minimal) prompts."""

    def __init__(self, config: DictConfig) -> None:
        dir_path = os.path.dirname(__file__)
        prompts_path = os.path.join(dir_path, "prompts.yaml")

        # The base simulator is identical to AllenLevel0's, so load it from
        # there instead of keeping a duplicate copy in this directory.
        level0_dir_path = os.path.join(dir_path, "..", "level0")
        base_simulator_path = os.path.join(level0_dir_path, "base_simulator.py")

        # AllenLevel0.__init__ hardcodes its own prompts/simulator paths, so
        # go through AllenBase directly to load this level's paths instead.
        AllenBase.__init__(self, config, prompts_path, base_simulator_path)

        self.prior_dist = self.prior()
