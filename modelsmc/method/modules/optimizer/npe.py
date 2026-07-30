import logging
import signal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

import torch
from omegaconf import DictConfig
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from torch import Tensor
from torch.distributions import Distribution

from modelsmc.method.modules.optimizer.base import OptimizerBase
from modelsmc.method.modules.posterior_estimator import NPEPosteriorWrapper

logger = logging.getLogger("ModelSMC")


class OptimizerNPEModule(OptimizerBase):
    """Neural Posterior Estimation (NPE) optimizer."""

    def __init__(self, config: DictConfig, task: "BaseTask", verbose: bool) -> None:
        """
        Args:
            config:  Optimizer-level Hydra config (``config.method.optimizer``).
            task:    Task object providing the prior, embedding net, and simulation
                     wrapper.
            verbose: Passed to the base class; controls progress logging.
        """
        super().__init__(config, task, verbose)
        self.optimizer_type = "npe"

        # Set z_score_x to false all the time. Previously this was configurable but due
        # to the introduction of summary statistics for observations and context,
        # adding a standardization layer internally in sbi's embedding network on top
        # of the task-specific embedding network causes issues when using the jointly
        # trained embedding networks as argument and condition in the evaluation step.
        z_score_x = "none"
        self.config.z_score_x = z_score_x

    def optimize(
        self,
        error_dict: dict | None,
        simulator: Callable,
        training_data_raw: Tensor,
        save_dir: str,
        valid_data_raw: Tensor | None = None,
    ) -> tuple[dict, Tensor | None]:
        """Run NPE optimization. Delegates to ``OptimizerBase.optimize``.

        Args:
            error_dict:        Upstream error dict; if not None optimization is skipped.
            simulator:         Callable simulator module to optimise.
            training_data_raw: Raw observed training data used for parameter estimation.
            save_dir:          Directory where outputs are saved.
            valid_data_raw:    Optional validation data; currently unused.
        """
        logger.info("Optimizing simulator parameters: NPE")
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
        """Train a neural posterior on prior samples.

        Args:
            prior_samples:   Parameter samples drawn from the prior.
                             Shape: [N, dim_theta].
            context_samples: Optional context samples concatenated with observations
                             before embedding. Shape: [N, dim_context] or None.
            sim_data_raw:    Simulated data. Shape: [N, dim_x].
            prior:           Prior distribution passed to ``sbi.NPE``.
            output_dict:     Base output dictionary to populate and return.
        """

        logger.debug("Using NPE posterior estimator")

        # Initialize the neural posterior model with the task specific
        # embedding network. Make sure that fixed (handcrafted) summary
        # statistics work as intended, disable the z-scoring of the observations
        # in case of fixed summary statistics.

        # Combine the additional context samples with the simulation data if applicable.
        # This is necessary because sbi does not allow additional context as input to
        # the embedding network. The context has to be concatenated with the data and is
        # then passed as input to the embedding network. Internally the two properties
        # are separated again and for each part the respective embedding networks are
        # applied.
        sim_data_raw_ctxt = self.task.concatenate_context_data(
            data=sim_data_raw, context=context_samples
        )

        # Delegate embedding strategy to the handler: fixed handlers pre-embed once and
        # return nn.Identity(); learnable handlers return raw data and self for joint
        # training with sbi.
        original_embedding = self.task.get_embedding_net()
        x_for_sbi, embedding_for_sbi = original_embedding.prepare_for_sbi(
            sim_data_raw_ctxt
        )

        # Here the full embedding network of the task is used which internally separates
        # context and data again. Both are then embedded using their respective
        # embedding networks. The final embedding is obtained by concatenating the two
        # embeddings.
        neural_posterior = posterior_nn(
            model="maf",
            embedding_net=embedding_for_sbi,
            z_score_x=self.config.z_score_x,
        )

        inference = NPE(
            prior=prior,
            show_progress_bars=False,
            density_estimator=neural_posterior,
        )

        inference.append_simulations(prior_samples, x_for_sbi)
        inference.train(max_num_epochs=self.config.max_num_epochs)

        # Restore the correct embedding into the trained net BEFORE build_posterior so
        # the posterior object is self-contained: all callers (optimizer, evaluator) can
        # pass raw concatenated data and the posterior applies the embedding internally.
        #
        # Learnable: get_trained_embedding_net returns sbi's net (with z-score wrapper)
        #   — no-op reassignment, behaviour unchanged.
        # Fixed: get_trained_embedding_net returns the original fixed handler, replacing
        #   the nn.Identity() that sbi trained with — posterior now embeds correctly.
        correct_embedding = original_embedding.get_trained_embedding_net(
            inference._neural_net.embedding_net
        )
        inference._neural_net.embedding_net = correct_embedding
        self.task.embedding_net = correct_embedding

        posterior_obj = inference.build_posterior()

        # Plot loss if requested
        if self.config.plotting:
            summary = inference.summary
            self._plot_loss(
                summary["training_loss"],
                summary["validation_loss"],
            )

        logger.debug("NPE training successful")

        # Store results in output_dict
        output_dict["posterior_obj"] = NPEPosteriorWrapper(posterior_obj)
        output_dict["sim_data_raw"] = sim_data_raw
        output_dict["prior_samples"] = prior_samples
        output_dict["context_samples"] = context_samples
        return output_dict

    def get_parameter_estimates(
        self, training_data_raw: Tensor, task: "BaseTask"
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """Get parameter estimates for NPE using posterior sampling.

        Args:
            training_data_raw: Raw training observations tensor.
            task:              Task object with context sampler and utilities.

        Returns:
            tuple: (parameter_estimates, optimizer_data, error_msg)
        """
        try:
            posterior_obj = self.optimize_output["posterior_obj"]

            num_samples = training_data_raw.shape[0]
            timeout = int(max(1, self.config.sampling_timeout * max(1, num_samples)))

            # Sample from posterior with timeout
            try:
                # Set up timeout, preserving any pre-existing SIGALRM handler so it
                # can be restored once sampling is done.
                old_sigalrm_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(timeout)

                try:
                    # Get the task specific context for the training data
                    ctxt_train_data = task.get_context(mode="training")

                    # Combine the additional context samples with the simulation data if
                    # applicable. This is necessary because sbi does not allow
                    # additional context as input to the embedding network. The context
                    # has to be concatenated with the data and is then passed as input
                    # to the embedding network. Internally the two properties are
                    # separated again and for each part the respective embedding
                    # networks are applied.
                    data_ctxt_training = task.concatenate_context_data(
                        context=ctxt_train_data, data=training_data_raw
                    )

                    # Sample from the posterior p(\theta|s_x(x),s_c(context))
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

                # Compute log probs log(p(\theta|s_x(x),s_c(context)))for all posterior
                # samples and all training data
                log_prob_batched = posterior_obj.log_prob_batched(
                    posterior_samples_batched,
                    x=data_ctxt_training,
                )

                # Get posterior values for max log_prob
                max_log_prob_idx = torch.argmax(log_prob_batched, dim=0)
                obs_indices = torch.arange(log_prob_batched.shape[1])
                parameter_est = posterior_samples_batched[
                    max_log_prob_idx, obs_indices, :
                ]

                # Return the posterior samples for optimizer metrics
                optimizer_data = {
                    "posterior_samples_batched": posterior_samples_batched
                }

            except TimeoutError:
                warning_msg = (
                    f"Posterior sampling timed out after "
                    f"{self.config.sampling_timeout * num_samples} seconds. Falling "
                    "back to ABC-style nearest neighbor estimation. This is expected "
                    "if e.g. the posterior lies outside the support of the prior, "
                    "which is a sign of model misspecification."
                )
                logger.warning(warning_msg)
                return self._fallback_to_abc_estimates(
                    training_data_raw, task, warning_msg
                )

            return parameter_est, optimizer_data, None

        except Exception as e:
            error_msg = f"NPE parameter estimation failed: {type(e).__name__}: {e}"
            return None, None, error_msg
