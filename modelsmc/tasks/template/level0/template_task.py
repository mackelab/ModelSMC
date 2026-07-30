import os

from omegaconf import DictConfig

from modelsmc.tasks.base_task import register_task
from modelsmc.tasks.template.template_task_base import TemplateBase


@register_task("template_level0")
class TemplateLevel0(TemplateBase):
    """Minimal level-0 template task wiring up this level's prompts and simulator."""

    def __init__(self, config: DictConfig) -> None:
        level_dir = os.path.dirname(__file__)
        prompts_path = os.path.join(level_dir, "prompts.yaml")
        base_simulator_path = os.path.join(level_dir, "base_simulator.py")
        super().__init__(config, prompts_path, base_simulator_path)
