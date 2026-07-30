import logging
import signal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

import torch
from omegaconf import DictConfig
from sbi.utils import handle_invalid_x
from torch import Tensor
from torch.distributions import Distribution

from modelsmc.method.modules.likelihood_estimator.nle_pfn import (
    LikelihoodEstimatorNLEPFN,
)
from modelsmc.method.modules.optimizer.base import OptimizerBase
from modelsmc.method.modules.posterior_estimator.npe_pfn.npe_pfn import (
    TabPFN_Based_NPE_PFN,
)

logger = logging.getLogger("ModelSMC")


class OptimizerNPEPFNModule(OptimizerBase):
    """
    Neural Posterior Estimation with Pre-trained Foundation Networks (NPE-PFN)
    optimizer.
    """

    def __init__(self, config: DictConfig, task: "BaseTask", verbose: bool) -> None:
        """
        Args:
            config:  Optimizer-level Hydra config (``config.method.optimizer``).
            task:    Task object providing the prior, embedding net, and simulation
                     wrapper.
            verbose: Passed to the base class; controls progress logging.

        Raises:
            ValueError: If the task uses learnable summary statistics without
                        ``train_sss`` enabled, which NPE-PFN does not support.
        """
        super().__init__(config, task, verbose)
        self.optimizer_type = "npe_pfn"

        # Validate that NPE-PFN doesn't use learnable embedding networks for
        # observations if it is not explicitly allowed via 'train_sss' config.
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
                "NPE-PFN optimization does not support learnable embedding networks "
                "for observations. Set 'learnable_summary_statistics' to 'false' if "
                "you want to use this optimizer"
            )

    def optimize(
        self,
        error_dict: dict | None,
        simulator: Callable,
        training_data_raw: Tensor,
        save_dir: str,
        valid_data_raw: Tensor | None = None,
    ) -> tuple[dict, Tensor | None]:
        """Run NPE-PFN optimization. Delegates to ``OptimizerBase.optimize``.

        Args:
            error_dict:        Upstream error dict; if not None optimization is skipped.
            simulator:         Callable simulator module to optimise.
            training_data_raw: Raw observed training data used for parameter estimation.
            save_dir:          Directory where outputs are saved.
            valid_data_raw:    Optional validation data; currently unused.
        """
        logger.info("Optimizing simulator parameters: NPE-PFN")
        return super().optimize(
            error_dict, simulator, training_data_raw, save_dir, valid_data_raw
        )

    def _optimize_specific(
        self,
        prior_samples: Tensor,
        context_samples: Tensor | None,
        sim_data_raw: Tensor,
        prior: Distribution,
        output_dict: dict,
    ) -> dict:
        """Set up the TabPFN-based NPE-PFN inference object on prior samples.

        Args:
            prior_samples:   Parameter samples drawn from the prior.
                             Shape: [N, dim_theta].
            context_samples: Optional context samples concatenated with observations
                             before embedding. Shape: [N, dim_context] or None.
            sim_data_raw:    Simulated data. Shape: [N, dim_x].
            prior:           Prior distribution passed to ``TabPFN_Based_NPE_PFN``.
            output_dict:     Base output dictionary to populate and return.
        """

        logger.debug("Using NPE_PFN posterior estimator")

        # Combine the additional context samples with the simulation data if applicable.
        # This is necessary because sbi does not allow additional context as input to
        # the embedding network. The context has to be concatenated with the data and is
        # then passed as input to the embedding network. Internally the two properties
        # are separated again and for each part the respective embedding networks are
        # applied.
        sim_data_raw_ctxt = self.task.concatenate_context_data(
            context=context_samples, data=sim_data_raw
        )

        # Initialize the TabPFN-based NPE-PFN inference object. Here the full embedding
        # network of the task is used which internally separates context and data again.
        # Both are then embedded using their respective embedding networks. The final
        # embedding is obtained by concatenating the two embeddings.
        inference = TabPFN_Based_NPE_PFN(
            prior=prior,
            embedding_net=self.task.get_embedding_net(),
            x_shape=[sim_data_raw_ctxt.shape[1]],
        )

        # Check the simulated data for invalid entries (NaNs, Infs)
        is_valid_x, num_nans, num_infs = handle_invalid_x(
            sim_data_raw_ctxt, exclude_invalid_x=True
        )
        sim_data_raw_ctxt_filtered = sim_data_raw_ctxt[is_valid_x]
        prior_samples_filtered = prior_samples[is_valid_x]
        if num_nans > 0 or num_infs > 0:
            logger.debug(f"Number of invalid x: {num_nans}, {num_infs}")

        # Check for constant columns and handle them
        sim_data_raw_ctxt_checked = LikelihoodEstimatorNLEPFN.check_constant_columns(
            sim_data_raw_ctxt_filtered
        )
        prior_samples_checked = LikelihoodEstimatorNLEPFN.check_constant_columns(
            prior_samples_filtered
        )

        # Add the training data to the inference object
        inference.append_simulations(prior_samples_checked, sim_data_raw_ctxt_checked)

        logger.debug("NPE-PFN setup successful")

        # Store results in output_dict (using filtered data)
        output_dict["posterior_obj"] = inference
        output_dict["sim_data_raw"] = sim_data_raw[is_valid_x]
        output_dict["context_samples"] = (
            context_samples[is_valid_x] if context_samples is not None else None
        )
        output_dict["prior_samples"] = prior_samples_filtered
        return output_dict

    def get_parameter_estimates(
        self, training_data_raw: Tensor, task: "BaseTask"
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """Get parameter estimates for NPE-PFN using posterior sampling.

        Args:
            training_data_raw: Raw training observations tensor.
            task:              Task object with context sampler and utilities.

        Returns:
            tuple: (parameter_estimates, optimizer_data, error_msg)
        """
        try:
            posterior_obj = self.optimize_output["posterior_obj"]

            if self.config.evaluation_mode == "max_log_posterior":
                return self._get_max_log_posterior_estimates(
                    posterior_obj, training_data_raw
                )
            elif self.config.evaluation_mode == "power_scaling":
                return self._get_power_scaling_estimates(
                    posterior_obj, training_data_raw
                )
            else:
                error_msg = (
                    f"Evaluation mode {self.config.evaluation_mode} not supported"
                )
                logger.debug(error_msg)
                return None, None, error_msg

        except Exception as e:
            error_msg = f"NPE-PFN parameter estimation failed: {type(e).__name__}: {e}"
            logger.debug(error_msg)
            return None, None, error_msg

    def _get_max_log_posterior_estimates(
        self, posterior_obj: TabPFN_Based_NPE_PFN, training_data_raw: Tensor
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """
        Get parameter estimates by selecting the posterior sample with the highest log
        probability.

        Args:
            posterior_obj:     Fitted posterior object (``TabPFN_Based_NPE_PFN``
                               instance).
            training_data_raw: Raw observed training data. Shape: [N, dim_x].
        """
        try:
            num_samples = (
                training_data_raw.shape[0] if hasattr(training_data_raw, "shape") else 1
            )
            timeout = int(max(1, self.config.sampling_timeout * max(1, num_samples)))

            # Set up timeout, preserving any pre-existing SIGALRM handler so it
            # can be restored once sampling is done.
            old_sigalrm_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(timeout)

            try:
                # Get the context for the training data
                ctxt_train_data = self.task.get_context(mode="training")

                # Concatenate context and training data
                data_ctxt_training = self.task.concatenate_context_data(
                    context=ctxt_train_data, data=training_data_raw
                )

                posterior_samples_batched = posterior_obj.sample_batched(
                    sample_shape=(self.config.num_samples_posterior,),
                    x=data_ctxt_training.float(),
                    show_progress_bars=False,
                )
            finally:
                # Always cancel the alarm and restore the previous handler,
                # including on unexpected exceptions and on normal return paths.
                signal.alarm(0)
                signal.signal(
                    signal.SIGALRM,
                    old_sigalrm_handler
                    if old_sigalrm_handler is not None
                    else signal.SIG_DFL,
                )

            # Compute log probs for all posterior samples and all training data
            log_prob_batched = posterior_obj.log_prob_batched(
                posterior_samples_batched,
                x=data_ctxt_training,
            )

            # Get posterior values for max log_prob
            max_log_prob_idx = torch.argmax(log_prob_batched, dim=0)
            obs_indices = torch.arange(log_prob_batched.shape[1])
            parameter_est = posterior_samples_batched[max_log_prob_idx, obs_indices, :]

            # Return the posterior samples for optimizer metrics
            optimizer_data = {"posterior_samples_batched": posterior_samples_batched}
            return parameter_est, optimizer_data, None

        except TimeoutError:
            warning_msg = (
                "Posterior sampling timed out after "
                f"{self.config.sampling_timeout * num_samples} seconds. Falling back "
                "to ABC-style nearest neighbor estimation. This is expected if e.g. "
                "the posterior lies outside the support of the prior, which is a sign "
                "of model misspecification."
            )
            logger.warning(warning_msg)
            return self._fallback_to_abc_estimates(
                training_data_raw, self.task, warning_msg
            )

        except Exception as e:
            error_msg = f"Failed to get MAP estimate: {type(e).__name__}: {e}"
            logger.debug(error_msg)
            return None, None, error_msg

    def _get_power_scaling_estimates(
        self, posterior_obj: TabPFN_Based_NPE_PFN, training_data_raw: Tensor
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """
        Get parameter estimates by drawing a single sample from a temperature-scaled
        posterior.

        Args:
            posterior_obj:     Fitted posterior object (``TabPFN_Based_NPE_PFN``
                               instance).
            training_data_raw: Raw observed training data. Shape: [N, dim_x].
        """
        try:
            num_samples = (
                training_data_raw.shape[0] if hasattr(training_data_raw, "shape") else 1
            )
            timeout = int(max(1, self.config.sampling_timeout * max(1, num_samples)))
            logger.debug(f"Set sampling timeout: {timeout} seconds")

            # Set up timeout, preserving any pre-existing SIGALRM handler so it
            # can be restored once sampling is done.
            old_sigalrm_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(timeout)

            try:
                # Get the context for the training data
                ctxt_train_data = self.task.get_context(mode="training")

                # Concatenate context and training data
                data_ctxt_training = self.task.concatenate_context_data(
                    context=ctxt_train_data, data=training_data_raw
                )

                parameter_est = posterior_obj.sample_batched(
                    sample_shape=(1,),
                    x=data_ctxt_training.float(),
                    show_progress_bars=False,
                    temperature=self.config.temperature,
                ).squeeze()
            finally:
                # Always cancel the alarm and restore the previous handler,
                # including on unexpected exceptions and on normal return paths.
                signal.alarm(0)
                signal.signal(
                    signal.SIGALRM,
                    old_sigalrm_handler
                    if old_sigalrm_handler is not None
                    else signal.SIG_DFL,
                )

            optimizer_data = None
            return parameter_est, optimizer_data, None

        except TimeoutError:
            warning_msg = (
                "Posterior sampling timed out after "
                f"{self.config.sampling_timeout * num_samples} seconds. Falling back "
                "ABC-style nearest neighbor estimation. This is expected if e.g. the "
                "posterior lies outside the support of the prior, which is a sign of "
                "model misspecification."
            )
            logger.warning(warning_msg)
            return self._fallback_to_abc_estimates(
                training_data_raw, self.task, warning_msg
            )

        except Exception as e:
            error_msg = f"Failed to get MAP estimate: {type(e).__name__}: {e}"
            logger.error(error_msg)
            return None, None, error_msg
