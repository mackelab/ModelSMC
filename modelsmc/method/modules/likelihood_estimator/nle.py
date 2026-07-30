import logging
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig
from sbi.inference import NLE
from torch import Tensor

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

logger = logging.getLogger("ModelSMC")


class LikelihoodEstimatorNLE:
    def __init__(
        self, task: "BaseTask", config: DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        """
        Args:
            task:   Task object; used to access the prior distribution.
            config: Evaluator-level Hydra config; must contain
                    ``NLE_max_num_epochs``.
        """
        self.config = config
        self.inference = NLE(
            prior=task.prior_dist,
            show_progress_bars=False,
        )
        self.likelihood_estimator = None
        self.task = task

    def fit_likelihood_estimator(self, theta_train: Tensor, x_train: Tensor) -> None:
        """Fit the NLE neural network on simulated (theta, x) pairs.

        Args:
            theta_train: Parameter samples used as conditions. Shape: [N, dim_theta].
            x_train:     Corresponding (embedded) simulation outputs. Shape: [N, dim_x].
        """
        self.inference.append_simulations(theta=theta_train, x=x_train)
        self.inference.train(max_num_epochs=self.config.NLE_max_num_epochs)

        self.likelihood_estimator = self.inference._neural_net
        self.likelihood_estimator.train(False)

        logger.debug(
            f"NLE training completed: epochs_trained={self.inference.epoch}, "
            f"loss_train={self.inference.summary['training_loss'][-1]:.5g}, "
            f"loss_val={self.inference.summary['validation_loss'][-1]:.5g}"
        )

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        """Evaluate the negative log-likelihood for (x, theta) pairs.

        Args:
            input:     Observation tensor (embedded). Shape: [N, dim_x].
            condition: Parameter tensor. Shape: [N, dim_theta].

        Returns:
            Per-sample negative log-likelihood. Shape: [N].
        """
        return self.likelihood_estimator.loss(input=input, condition=condition)
