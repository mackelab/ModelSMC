from typing import TYPE_CHECKING

from omegaconf import DictConfig

from modelsmc.method.modules.optimizer.abc import OptimizerABCModule
from modelsmc.method.modules.optimizer.base import OptimizerBase
from modelsmc.method.modules.optimizer.npe import OptimizerNPEModule
from modelsmc.method.modules.optimizer.npe_pfn import OptimizerNPEPFNModule

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

__all__ = [
    "OptimizerBase",
    "OptimizerABCModule",
    "OptimizerNPEModule",
    "OptimizerNPEPFNModule",
    "create_optimizer",
]


def create_optimizer(
    config: DictConfig, task: "BaseTask", verbose: bool
) -> OptimizerBase:
    """Factory function to create the appropriate optimizer based on config.

    Supported optimizer types: ``"abc"``, ``"npe"``, ``"npe_pfn"``.

    Args:
        config:  Optimizer-level Hydra config; must contain a ``name`` field selecting
                 the backend (``"abc"``, ``"npe"``, or ``"npe_pfn"``).
        task:    Task object providing the prior, embedding net, and simulation wrapper.
        verbose: Passed to the optimizer constructor; controls progress logging.

    Raises:
        ValueError: If ``config.name`` is not one of the supported optimizer types.
    """
    if config.name == "abc":
        return OptimizerABCModule(config, task, verbose)
    elif config.name == "npe":
        return OptimizerNPEModule(config, task, verbose)
    elif config.name == "npe_pfn":
        return OptimizerNPEPFNModule(config, task, verbose)
    else:
        raise ValueError(
            f"Optimizer '{config.name}' not supported. Available options: "
            "'abc', 'npe', 'npe_pfn'"
        )
