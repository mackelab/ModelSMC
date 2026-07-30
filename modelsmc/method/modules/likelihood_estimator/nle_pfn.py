########################################################################################
#
# This file is adapted from the autoregressive log-probability routine of
#
# https://github.com/mackelab/npe-pfn/blob/6100e60b43f0437601c3377013909ee6114aa737/npe_pf/npe_pf.py
#
# Authored by: J. Vetter, M. Gloeckler, D. Gedon, J. Macke
#
# under the Prior Labs License, Version 1.1, May 2025. A copy of the License is
# distributed at ../posterior_estimator/npe_pfn/LICENSE.
#
# The factorisation is applied to the likelihood p(x | theta) rather than the
# posterior p(theta | x), and this file otherwise differs substantially from the
# original.
#
########################################################################################

import logging
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from sbi.utils import handle_invalid_x
from tabpfn import TabPFNRegressor
from torch import Tensor

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

logger = logging.getLogger("ModelSMC")


class LikelihoodEstimatorNLEPFN:
    def __init__(
        self, task: "BaseTask", config: DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        """
        Args:
            task:   Task object; used by ``loss`` to call
                    ``prepare_likelihood_estimator_data``.
            config: Evaluator-level Hydra config.
        """
        self.config = config
        self.task = task
        self.inf_method_all = []

    @staticmethod
    def check_constant_columns(data_joint: Tensor, eps: float = 1e-5) -> Tensor:
        """since TabPFN estimates the number of bars for their bar distribution, a
        constant column would cause the model to crash. To avoid this, we add a small
        amount of noise to the constant columns.
        """
        for col in range(data_joint.shape[1]):
            if torch.all(data_joint[:, col] == data_joint[:, col][0]):
                data_joint[:, col] += torch.randn(data_joint.shape[0]) * eps
        return data_joint

    def fit_likelihood_estimator(self, theta_train: Tensor, x_train: Tensor) -> None:
        """
        Fit one TabPFN regressor per output dimension via autoregressive factorisation.

        Args:
            theta_train: Parameter samples. Shape: [N, dim_theta].
            x_train:     Corresponding (embedded) simulation outputs. Shape: [N, dim_x].
        """
        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x_train, exclude_invalid_x=True
        )
        x_train = x_train[is_valid_x]
        theta_train = theta_train[is_valid_x]
        logger.debug(
            f"Number of invalid x: {num_nans}, {num_infs}"
        ) if num_nans > 0 or num_infs > 0 else None

        data_joint = torch.cat([theta_train, x_train], dim=1)
        data_joint = LikelihoodEstimatorNLEPFN.check_constant_columns(data_joint)

        dim_theta = theta_train.shape[1]
        dim_x = x_train.shape[1]

        # sequential prediction for each x dimension
        for x_idx in range(dim_x):
            idx = dim_theta + x_idx

            inf_method = TabPFNRegressor()
            inf_method.fit(data_joint[:, :idx], data_joint[:, idx])

            # append the trained model to the list
            self.inf_method_all.append(inf_method)

            # Clear cache to avoid memory issues, otherwise can segmentation fault
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def loss(
        self, input: Tensor, condition: Tensor, logits_clip_abs: float = 50
    ) -> Tensor:
        """Evaluate the negative log-likelihood for (x, theta) pairs.

        The likelihood log p(x | theta) is factorised autoregressively
        across the ``dim_x`` output dimensions (one fitted TabPFN regressor per
        dimension, see ``fit_likelihood_estimator``); the per-dimension negative
        log-likelihoods are summed.

        Sign convention: like ``LikelihoodEstimatorNLE.loss``, this returns the
        *negative* log-likelihood. Callers therefore recover the log-likelihood
        log p(x | theta) by negating the result, e.g.
        ``-likelihood_estimator.loss(...)`` (as done in the evaluator).

        Args:
            input:           Observation tensor (embedded), i.e. x_val.
                             Shape: [N, dim_x].
            condition:       Parameter tensor, i.e. theta_est.
                             Shape: [N, dim_theta].
            logits_clip_abs: Symmetric clamp applied to the predicted
                             bar-distribution logits before scoring, guarding
                             against +/-inf values.

        Returns:
            Per-sample negative log-likelihood, summed over the ``dim_x`` output
            dimensions. Shape: [N].
        """
        # check dimension
        input = input.unsqueeze(0) if input.ndim == 1 else input
        condition = condition.unsqueeze(0) if condition.ndim == 1 else condition
        data_joint = torch.cat([condition, input], dim=1)

        dim_condition = condition.shape[1]

        nll_all = torch.zeros(input.shape[0], input.shape[1])
        for idx, inf_method in enumerate(self.inf_method_all):
            features_end = dim_condition + idx
            target_idx = dim_condition + idx

            x_pred_dist = inf_method.predict(
                data_joint[:, :features_end], output_type="full", quantiles=[]
            )
            # clip logits -inf and inf values
            x_pred_dist["logits"] = torch.clamp(
                x_pred_dist["logits"], min=-logits_clip_abs, max=logits_clip_abs
            )
            logits = x_pred_dist["logits"]

            nll_all[:, idx] = (
                x_pred_dist["criterion"](
                    logits, data_joint[:, target_idx].to(logits.device)
                )
                .detach()
                .cpu()
            )

        # Clear cache to avoid memory issues, otherwise can segmentation fault
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return nll_all.sum(dim=1)
