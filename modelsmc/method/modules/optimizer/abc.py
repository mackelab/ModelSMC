import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Distribution

from modelsmc.method.modules.optimizer.base import OptimizerBase

logger = logging.getLogger("ModelSMC")


class OptimizerABCModule(OptimizerBase):
    """ABC (Approximate Bayesian Computation) optimizer."""

    def __init__(self, config: DictConfig, task: "BaseTask", verbose: bool) -> None:
        """
        Args:
            config:  Optimizer-level Hydra config (``config.method.optimizer``).
            task:    Task object providing the prior, embedding net, and simulation
                     wrapper.
            verbose: Passed to the base class; controls progress logging.

        Raises:
            ValueError: If the task uses learnable summary statistics, which ABC
                        does not support.
        """
        super().__init__(config, task, verbose)
        self.optimizer_type = "abc"

        flag_1 = (
            ("learnable_summary_statistics" in task.config)
            and (task.config.learnable_summary_statistics)
            and "train_sss" not in self.config
        )
        flag_2 = (
            ("learnable_summary_statistics" in task.config)
            and (task.config.learnable_summary_statistics)
            and ("train_sss" in self.config)
            and not (self.config.train_sss)
        )

        if flag_1 or flag_2:
            raise ValueError(
                "ABC optimization does not support learnable "
                "embedding networks for observations. Set "
                "'learnable_summary_statistics' to 'false' if you want to use this "
                "optimizer"
            )

    def optimize(
        self,
        error_dict: dict | None,
        simulator: Callable,
        training_data_raw: Tensor,
        save_dir: str,
        valid_data_raw: Tensor | None = None,
    ) -> tuple[dict, Tensor | None]:
        """Run ABC optimization. Delegates to ``OptimizerBase.optimize``.

        Args:
            error_dict:        Upstream error dict; if not None optimization is skipped.
            simulator:         Callable simulator module to optimise.
            training_data_raw: Raw observed training data used for parameter estimation.
            save_dir:          Directory where outputs are saved.
            valid_data_raw:    Optional validation data; currently unused.
        """
        logger.info("Optimizing simulator parameters: ABC")
        return super().optimize(
            error_dict=error_dict,
            simulator=simulator,
            training_data_raw=training_data_raw,
            save_dir=save_dir,
            valid_data_raw=valid_data_raw,
        )

    def _optimize_specific(
        self,
        prior_samples: Tensor,
        context_samples: Tensor | None,
        sim_data_raw: Tensor,
        prior: Distribution,
        output_dict: dict,
    ) -> dict:
        """
        Store simulation results in output_dict; no further processing needed for ABC.

        Args:
            prior_samples:   Parameter samples drawn from the prior.
                             Shape: [N, dim_theta].
            context_samples: Optional context samples. Shape: [N, dim_context] or None.
            sim_data_raw:    Simulated data. Shape: [N, dim_x].
            prior:           Prior distribution object (unused in ABC, required by
                             interface).
            output_dict:     Base output dictionary to populate and return.
        """
        output_dict["posterior_obj"] = None
        output_dict["sim_data_raw"] = sim_data_raw
        output_dict["prior_samples"] = prior_samples
        output_dict["context_samples"] = context_samples
        return output_dict

    def get_parameter_estimates(
        self, training_data_raw: Tensor, task: "BaseTask"
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """Get parameter estimates for ABC using nearest neighbor approach.

        Args:
            training_data_raw: Raw training observations tensor.
            task:              Task object with embedding net and utilities.

        Returns:
            tuple: (parameter_estimates, optimizer_data, error_msg)
        """

        try:
            # Get the stored simulation data from the optimization
            sim_data_raw = self.optimize_output["sim_data_raw"]
            prior_samples = self.optimize_output["prior_samples"]

            # Ensure tensors are on the same device and have the same dtype
            training_data_raw = training_data_raw.float()
            sim_data_raw = sim_data_raw.float()
            prior_samples = prior_samples.float()

            # Note: Here only the (embedded) data is used for distance computation but
            # not the context.

            # For each training data point, find the closest simulation data point
            num_train_samples = training_data_raw.shape[0]
            parameter_est = torch.zeros(
                (num_train_samples, prior_samples.shape[1]),
                dtype=prior_samples.dtype,
                device=prior_samples.device,
            )

            # Get the summary statistics for the training data and the simulated data
            # Important: Only use the embedded data for distance computation, i.e. only
            # a part of the full embedding function.

            summary_fn = self.task.embedding_net.embed_x

            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                current_device = "cuda:" + str(current_device)
            else:
                current_device = "cpu"

            # Compute summary statistics for all data
            sim_summary_stats = summary_fn(sim_data_raw.to(current_device)).cpu()
            training_summary_stats = summary_fn(
                training_data_raw.to(current_device)
            ).cpu()

            # Normalize summary statistics to handle different scales
            # Compute normalization parameters from simulated data only (avoid data
            # leakage). Normalize across features (dim=0) since examples are independent
            mean = torch.mean(sim_summary_stats, dim=0)
            std = torch.std(sim_summary_stats, dim=0)
            std = torch.where(
                std < 1e-8, torch.ones_like(std), std
            )  # Avoid division by zero

            sim_summary_normalized = (sim_summary_stats - mean) / std
            training_summary_normalized = (training_summary_stats - mean) / std

            for i in range(num_train_samples):
                # Calculate Euclidean distances between this training point and all
                # simulation data
                distances = torch.norm(
                    sim_summary_normalized - training_summary_normalized[i], dim=1
                )

                # Find the index of the closest simulation data point
                closest_idx = torch.argmin(distances)

                # Get the corresponding parameter estimate
                parameter_est[i] = prior_samples[closest_idx]

            # Return the parameter estimates and empty optimizer data (no posterior
            # samples for abc)
            optimizer_data = {}
            return parameter_est, optimizer_data, None

        except Exception as e:
            error_msg = f"ABC parameter estimation failed: {type(e).__name__}: {e}"
            return None, None, error_msg
