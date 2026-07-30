import logging
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from modelsmc.method.modules.likelihood_estimator.nle import LikelihoodEstimatorNLE
from modelsmc.method.modules.likelihood_estimator.nle_pfn import (
    LikelihoodEstimatorNLEPFN,
)

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

logger = logging.getLogger("ModelSMC")

__all__ = [
    "LikelihoodEstimatorNLE",
    "LikelihoodEstimatorNLEPFN",
    "create_likelihood_estimator",
]


def create_likelihood_estimator(
    task: "BaseTask", config: DictConfig
) -> LikelihoodEstimatorNLE | LikelihoodEstimatorNLEPFN:
    """Factory function to create the configured likelihood estimator.

    Args:
        task:   Task object providing the prior and likelihood-estimator data
                preparation. Also supplies ``custom_likelihood_estimator_class``
                for the debug-only ``"task-specific"`` option.
        config: Evaluator-level Hydra config; must contain a
                ``likelihood_estimator`` field selecting the backend
                (``"NLE"``, ``"NLE_PFN"``, or ``"task-specific"``).

    Returns:
        An instantiated likelihood estimator.

    Raises:
        ValueError: If ``config.likelihood_estimator`` is not supported.
    """
    if config.likelihood_estimator == "NLE":
        return LikelihoodEstimatorNLE(task, config)
    elif config.likelihood_estimator == "NLE_PFN":
        return LikelihoodEstimatorNLEPFN(task, config)

    # This is a special case to allow the usage of the ground truth likelihood for
    # debugging.
    elif config.likelihood_estimator == "task-specific":
        logger.warning("A custom, possibly not trainable likelihood estimator is used!")
        return task.custom_likelihood_estimator_class(task, config)
    else:
        raise ValueError(
            f"Likelihood estimator {config.likelihood_estimator} not supported"
        )
