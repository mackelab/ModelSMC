import logging
import os
import signal
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from sbi.analysis import pairplot
from torch import Tensor

from modelsmc.method.modules.likelihood_estimator import (
    LikelihoodEstimatorNLE,
    LikelihoodEstimatorNLEPFN,
    create_likelihood_estimator,
)

logger = logging.getLogger("ModelSMC")


class EvaluatorBase:
    def __init__(self, config: DictConfig, task: "BaseTask", verbose: bool) -> None:
        """
        Args:
            config:  Evaluator-level Hydra config (``config.method.evaluator``).
            task:    Task object providing the evaluation function, embedding net,
                     context sampler, and likelihood-estimator data preparation.
            verbose: If True, log progress messages during evaluation.
        """
        self.config = config
        self.verbose = verbose
        self.save_dir = None
        self.task = task

    def evaluate(
        self,
        parameter_estimates: Tensor | None,
        optimizer_output: dict,
        simulator: torch.nn.Module,
        valid_data_raw: Tensor,
        save_dir: str,
    ) -> dict:
        """Evaluate a fitted simulator on validation data.

        Trains a neural likelihood estimators on the simulation
        data produced by the optimizer, then computes evidence-based metrics
        (negative log marginal likelihood) for all validation observations and,
        when parameter estimates are available, simulation-based metrics
        (task-specific MSE / MMD, negative log-likelihood at the MAP estimate)
        and posterior inter-quantile ranges.

        Args:
            parameter_estimates: MAP parameter estimate tensor of shape
                ``[n_obs, dim_theta]``, or ``None`` when parameter estimation
                was skipped or failed.
            optimizer_output:    Dict returned by the optimizer containing at
                minimum ``errors``, ``error_type``, ``sim_data_raw``,
                ``prior_samples``, ``context_samples``, and ``posterior_obj``.
            simulator:           Compiled simulator module (may be ``None`` if
                compilation failed — errors are handled internally).
            valid_data_raw:      Validation observations tensor.
            save_dir:            Output directory for plots and diagnostic files.

        Returns:
            Dictionary with keys ``errors``, ``error_type``, and (on success)
            ``eval_metrics`` containing all computed scalar metrics.
        """
        self.save_dir = save_dir
        logger.info(
            "Evaluating simulator performance: " + self.config.likelihood_estimator
        )

        # initialize output dict
        output_dict = {}
        output_dict["errors"] = optimizer_output["errors"]
        output_dict["error_type"] = optimizer_output["error_type"]

        # default return if there are errors or no parameter estimates
        if (
            output_dict["errors"] is not None  # Some error in the past
            or (
                parameter_estimates is None and self.config.estimate_parameters
            )  # Parameters should be estimated but are None
        ):
            return output_dict

        # ==============================================================================
        # Evaluations that do NOT need the parameter estimates
        # ==============================================================================

        # get simulation wrapper
        simulation_wrapper = self.task.simulation_wrapper
        simulation_wrapper = partial(
            simulation_wrapper,
            simulator,
        )

        # Perform NLE training
        try:
            # Get training data for the likelihood estimator
            x_train, theta_train = self.task.prepare_likelihood_estimator_data(
                x_raw=optimizer_output["sim_data_raw"],
                params=optimizer_output["prior_samples"],
                context_raw=optimizer_output["context_samples"],
            )

            logger.debug(
                "Fitting likelihood estimator: " + self.config.likelihood_estimator
            )

            # Create the likelihood estimator
            likelihood_estimator = create_likelihood_estimator(
                task=self.task, config=self.config
            )

            # Fit the likelihood estimator
            likelihood_estimator.fit_likelihood_estimator(
                theta_train=theta_train, x_train=x_train
            )

        except Exception as e:
            logger.error(f"Failed to run NLE training: {type(e).__name__}: {e}")
            output_dict["errors"] = f"{type(e).__name__}: {e}"
            output_dict["error_type"] = "Failed to run NLE training"
            return output_dict

        # Get the context for the validation data
        context_validation = self.task.get_context(mode="validation")

        # Compute the evidence-based metrics that do not require the parameter
        # estimate for the validation data
        try:
            log_marginals_est = self._compute_evidence(
                valid_data_raw=valid_data_raw,
                context_validation=context_validation,
                optimizer_output=optimizer_output,
                likelihood_estimator=likelihood_estimator,
            )

            # Get the average log marginal
            # i.e. frac{1}{N} \sum_{i = 1}^N log p(x_i)
            avg_log_marginal_NLE = log_marginals_est.mean().item()
            neg_avg_log_marginal_NLE = -avg_log_marginal_NLE

            # Get the log marginal
            # i.e. log \left(\prod_{i = 1}^N p(x_i)\right)
            log_marginal_NLE = log_marginals_est.sum().item()
            neg_log_marginal_NLE = -log_marginal_NLE

            if "eval_metrics" not in output_dict:
                output_dict["eval_metrics"] = {}

            output_dict["eval_metrics"]["neg_avg_log_marginal_NLE"] = (
                neg_avg_log_marginal_NLE
            )
            output_dict["eval_metrics"]["neg_log_marginal_NLE"] = neg_log_marginal_NLE
            output_dict["eval_metrics"]["timeout_evidence_estimation"] = (
                self._flag_timeout_evidence
            )

        except Exception as e:
            logger.error(
                f"Failed to compute evidence-based metrics: {type(e).__name__}: {e}"
            )
            output_dict["errors"] = f"{type(e).__name__}: {e}"
            output_dict["error_type"] = "Failed to compute evidence-based metrics"

        # ==============================================================================
        # Evaluations that DO need the parameter estimates
        # ==============================================================================

        if parameter_estimates is not None:
            # Get simulate using the parameter estimates and the validation context
            sim_data_raw = simulation_wrapper(
                params=parameter_estimates,
                context=context_validation,
            )

            # Embed the simulation data
            sim_data = self.task.embedding_net.embed_x(x=sim_data_raw)

            sim_data = torch.as_tensor(sim_data, dtype=torch.float32)

            # Embed the validation data
            valid_data = self.task.embedding_net.embed_x(x=valid_data_raw)
            valid_data = torch.as_tensor(valid_data, dtype=torch.float32)

            # Plot the simulations and the validation data
            try:
                if self.config.plotting:
                    self.task.plot_observation(
                        valid_data,
                        valid_data_raw,
                        sim_data,
                        sim_data_raw,
                        path=self.save_dir,
                        fig_name="simulated_data.png",
                        kwargs={
                            "parameter_estimate": parameter_estimates,
                            "simulator": simulator,
                        },
                    )

            except Exception as e:
                logger.info(f"Failed to plot simulated data: {type(e).__name__}: {e}")

            # Compute task-specific metrics
            task_metrics = self.task.eval_function(
                sim_data_raw=sim_data_raw,
                valid_data_raw=valid_data_raw,
                sim_data=sim_data,
                valid_data=valid_data,
                parameter_estimates=parameter_estimates,
                simulator=simulator,
            )

            # Initialize the metric dict
            if "eval_metrics" not in output_dict:
                output_dict["eval_metrics"] = task_metrics
            # Merge with existing metric dict
            else:
                output_dict["eval_metrics"] = output_dict["eval_metrics"] | task_metrics

            # Compute the likelihood-based metrics
            try:
                x_validation, theta_validation = (
                    self.task.prepare_likelihood_estimator_data(
                        x_raw=valid_data_raw,
                        context_raw=context_validation,
                        params=parameter_estimates,
                    )
                )

                # Evaluate the log likelihoods of the validation data under the
                # optimal parameters
                log_likelihood_est_all = -likelihood_estimator.loss(
                    input=x_validation,  # shape: (n_valid, n_dim_x)
                    condition=theta_validation,
                )  # shape: (n_valid)

                # Compute the average log likelihood under the optimal parameters
                # i.e. \frac{1}{N}\sum_{i = 1}^N \log(p(s(x_i)|\hat\theta_i))
                avg_log_likelihood_NLE = log_likelihood_est_all.mean().item()
                neg_avg_log_likelihood_NLE = -avg_log_likelihood_NLE

                # Compute the log likelihood under the optimal parameters
                # i.e. \log \left(\prod_{i = 1}^N p(s(x_i)|\hat\theta_i)\right)
                log_likelihood_NLE = log_likelihood_est_all.sum().item()
                neg_log_likelihood_NLE = -log_likelihood_NLE

                output_dict["eval_metrics"]["neg_avg_log_likelihood_NLE"] = (
                    neg_avg_log_likelihood_NLE
                )
                output_dict["eval_metrics"]["neg_log_likelihood_NLE"] = (
                    neg_log_likelihood_NLE
                )

            except Exception as e:
                logger.error(
                    "Failed to compute likelihood-based metrics: "
                    f"{type(e).__name__}: {e}"
                )
                output_dict["errors"] = f"{type(e).__name__}: {e}"
                output_dict["error_type"] = "Failed to compute likelihood-based metrics"

            # Compute optimizer-specific metrics
            try:
                output_dict = self._get_optimizer_metrics(
                    output_dict=output_dict,
                    optimizer_output=optimizer_output,
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute optimizer-specific metrics: "
                    f"{type(e).__name__}: {e}"
                )
                output_dict["errors"] = f"{type(e).__name__}: {e}"
                output_dict["error_type"] = (
                    "Failed to compute optimizer-specific metrics"
                )

        return output_dict

    def _get_optimizer_metrics(
        self,
        output_dict: dict,
        optimizer_output: dict,
    ) -> dict:
        """Compute optimizer-specific metrics based on available data."""

        if optimizer_output:
            # Check if we have posterior samples for SBI-based metrics
            if (
                ("optimizer_data" in optimizer_output)
                and (optimizer_output["optimizer_data"] is not None)
                and ("posterior_samples_batched" in optimizer_output["optimizer_data"])
            ):
                posterior_samples_batched = optimizer_output["optimizer_data"][
                    "posterior_samples_batched"
                ]

                # Get posterior quantile ranges (if configured)
                if (
                    hasattr(self.config, "inter_quantile_range")
                    and self.config.inter_quantile_range
                ):
                    inter_quantile_mean, inter_quantile_std = [], []
                    for inter_quantile_range in self.config.inter_quantile_range:
                        # Create symmetric quantile range
                        quantile_range = torch.tensor(
                            [
                                0.5 - inter_quantile_range / 2,
                                0.5 + inter_quantile_range / 2,
                            ]
                        )
                        posterior_quantiles = torch.quantile(
                            posterior_samples_batched, quantile_range, dim=0
                        )
                        inter_quantiles = (
                            posterior_quantiles[1] - posterior_quantiles[0]
                        )

                        inter_quantile_mean.append(torch.mean(inter_quantiles, dim=0))
                        inter_quantile_std.append(torch.std(inter_quantiles, dim=0))

                    inter_quantile_mean = torch.stack(inter_quantile_mean)
                    inter_quantile_std = torch.stack(inter_quantile_std)

                    # Save the posterior quantiles for each dimension
                    output_dict["posterior_inter_quantiles"] = {}
                    for i in range(inter_quantiles.shape[1]):
                        output_dict["posterior_inter_quantiles"][f"dim_{i}"] = {}
                        output_dict["posterior_inter_quantiles"][f"dim_{i}"]["mean"] = (
                            inter_quantile_mean[:, i].tolist()
                        )
                        output_dict["posterior_inter_quantiles"][f"dim_{i}"]["std"] = (
                            inter_quantile_std[:, i].tolist()
                        )

                    # Save the values where the quantiles are taken from
                    output_dict["inter_quantile_range"] = (
                        self.config.inter_quantile_range
                    )

                # Save posterior plot if requested
                if self.config.plotting:
                    self._plot_posterior(posterior_samples_batched)

            # Save ABC fallback information if used
            optimizer_data = optimizer_output.get("optimizer_data")
            if optimizer_data is not None:
                if (
                    "used_abc_fallback" in optimizer_data
                    and optimizer_data["used_abc_fallback"]
                ):
                    output_dict["warnings"] = optimizer_data.get("warning_msg")

        return output_dict

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        raise TimeoutError("Posterior sampling in evidence computation took too long")

    def _prior_sampling_fallback(
        self, n_obs: int, num_samples: int
    ) -> tuple[Tensor, Tensor]:
        """Sample from the prior for use as IS proposal with unit weights.

        Used both as the main prior-based evidence estimator and as a fallback
        when posterior sampling times out.

        Args:
            n_obs:       Number of validation observations.
            num_samples: Number of IS samples to draw per observation (B).

        Returns:
            theta_samples: [n_obs, num_samples, dim_θ] — same B prior samples
                broadcast across all observations.
            log_weights: [n_obs, num_samples] — all zeros (q = p(θ)).
        """
        prior_samples = self.task.prior_dist.sample((num_samples,))  # [B, dim_θ]
        theta_samples = prior_samples.unsqueeze(0).expand(
            n_obs, -1, -1
        )  # [n_obs, B, dim_θ]
        log_weights = torch.zeros(n_obs, num_samples)  # [n_obs, B]
        return theta_samples, log_weights

    def _plot_posterior(self, posterior_samples: Tensor, num_samples: int = 3) -> None:
        """Save a pairplot visualization of the posterior samples for one random
        observation.
        The pairplot is saved in the save_dir.

        Args:
            posterior_samples: Samples drawn from the posterior distribution
        """
        # Select num_samples random samples
        num_samples = min(num_samples, posterior_samples.shape[1])
        random_indices = torch.randperm(posterior_samples.shape[1])[:num_samples]
        selected_outputs = posterior_samples[:, random_indices, :]

        for i in range(num_samples):
            pairplot(selected_outputs[:, i, :])
            save_dir = os.path.join(self.save_dir, "posterior_samples")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"sample{i}.png"))
            plt.close()

    def _compute_evidence(
        self,
        valid_data_raw: Tensor,
        context_validation: Tensor | None,
        optimizer_output: dict,
        likelihood_estimator: LikelihoodEstimatorNLE | LikelihoodEstimatorNLEPFN,
    ) -> Tensor:
        """
        Estimate the log marginal likelihood (evidence) for each validation observation.

        The evidence for model m given observation x is the marginal likelihood
        obtained by integrating out the parameters:

            p(x | m) = ∫ p(x | θ, m) p(θ) dθ

        This integral is intractable in general and is approximated via importance
        sampling (IS) using a proposal distribution q(θ):

            p(x | m) ≈ (1/B) Σ_j  p(x | θ_j, m) · w_j,   θ_j ~ q(θ),
                        where w_j = p(θ_j) / q(θ_j).

        In log-space, the IS estimate for observation i is:

            log p(x_i | m) ≈ logsumexp_j [ log p(x_i | θ_ij, m) + log w_ij ] − log B

        Two proposal choices are supported (configured via
        config.evidence_estimation_mode):
          - "prior_samples":     q(θ) = p(θ).  All IS weights equal 1 (log w = 0),
                                 reducing to a standard Monte Carlo estimate.
          - "posterior_samples": q(θ) = p(θ|x).  IS weights correct for the
                                 prior/posterior mismatch, yielding a lower-variance
                                 estimate at the cost of requiring a trained posterior.

        Log-likelihoods are evaluated by the already-fitted neural likelihood estimator
        (NLE or NLE-PFN).  To guard against memory errors the full [n_obs × B] batch is
        processed in chunks of size config.bs_marginals.

        Args:
            valid_data_raw: Raw (un-embedded) validation observations.
                Shape: [n_obs, dim_obs].
            context_validation: Optional conditioning context for each validation
                observation (e.g. initial conditions, covariates).
                Shape: [n_obs, dim_context], or None.
            optimizer_output: Dictionary returned by the optimizer. Must contain
                the key "posterior_obj" (the trained posterior object) when
                evidence_estimation_mode is "posterior_samples"; may be None when
                using prior-based estimation or when the optimizer is ABC.
            likelihood_estimator: Fitted neural likelihood estimator used to
                evaluate log p(x | θ).

        Returns:
            log_marginals_est: Tensor of per-observation log marginal likelihood
                estimates. Shape: [n_obs].
        """
        # --- Step 1: Draw proposal samples and compute IS log-weights ---
        # theta_samples: [n_obs, B, dim_θ], log_weights: [n_obs, B].
        # For prior mode log_weights are all zero; for posterior mode they correct
        # for the mismatch between prior and posterior
        # (see _prepare_evidence_computation for details).
        theta_samples, log_weights = self._prepare_evidence_computation(
            x=valid_data_raw,
            posterior_object=optimizer_output.get("posterior_obj"),
            num_samples=self.config.n_samples_evidence_estimation,
            mode=self.config.evidence_estimation_mode,
            context=context_validation,
        )

        n_obs = valid_data_raw.shape[0]
        B = theta_samples.shape[1]  # number of IS samples per observation

        # Chunk size for batched likelihood evaluation (None = process all at once).
        if "bs_marginals" in self.config:
            max_batch_size = self.config.bs_marginals
        else:
            max_batch_size = None

        logger.debug(f"Max batch size for marginal computation: {max_batch_size}")

        # --- Step 2: Flatten observations and samples for batched NLE evaluation ---
        # Pair each of the n_obs observations with each of its B theta samples,
        # producing a flat batch of size n_obs*B. This works identically for both
        # prior mode (same B samples broadcast across all observations before
        # flattening) and posterior mode (per-observation samples).
        x_flat = valid_data_raw.repeat_interleave(B, dim=0)  # [n_obs*B, dim_obs]
        theta_flat = theta_samples.reshape(n_obs * B, -1)  # [n_obs*B, dim_θ]
        context_flat = (
            context_validation.repeat_interleave(B, dim=0)
            if context_validation is not None
            else None
        )  # [n_obs*B, dim_ctxt] or None

        # --- Step 3: Evaluate log p(x_i | θ_j) in chunks to avoid OOM ---
        total = n_obs * B
        chunk_size = max_batch_size if max_batch_size is not None else total
        log_ll_flat = torch.zeros(total, dtype=valid_data_raw.dtype)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            x_prep, theta_prep = self.task.prepare_likelihood_estimator_data(
                x_raw=x_flat[start:end],
                params=theta_flat[start:end],
                context_raw=context_flat[start:end]
                if context_flat is not None
                else None,
            )
            log_ll_flat[start:end] = -likelihood_estimator.loss(
                input=x_prep,
                condition=theta_prep,
            )

        # --- Step 4: Compute the IS estimate of the log marginal in log-space ---
        # log p(x_i) ≈ logsumexp_j [ log p(x_i|θ_ij) + log w_ij ] − log B
        # Using logsumexp ensures numerical stability when individual likelihoods
        # are very small. For prior-based estimation log_weights = 0, so this
        # reduces to the standard Monte Carlo estimate of the evidence.
        log_likelihoods = log_ll_flat.reshape(n_obs, B)
        log_marginals_est = torch.logsumexp(
            log_likelihoods + log_weights, dim=-1
        ) - np.log(B)  # shape: (n_obs,)
        assert log_marginals_est.shape == torch.Size([valid_data_raw.shape[0]])

        return log_marginals_est

    def _prepare_evidence_computation(
        self,
        x: Tensor,
        posterior_object: Any,
        num_samples: int,
        mode: str = "prior_samples",
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        The evidence is defined as

            p(x|m) = \\int p(x|m,\theta)p(\theta) d\theta.

        In general, using a proposal q(\theta) the computation of the evidence can
        be formulated as

            p(x|m) = \\EX_{\theta\\sim q(\theta)}\\left[p(x|m,\theta) p(\theta)/ q(\theta)\right].

        which in practice is approximated as

            p(x|m) \approx \frac{1}{B}\\sum_{i=1}^B p(x|m,\theta_i)\\cdot w_i

        with

            w_i = \frac{p(\theta_i)}{q(\theta_i)} and \theta_i \\sim q(\theta)

        In this function two different choices of the proposal q are implemented:

        1) q(\theta) = p(\theta) i.e. using the prior as proposal.
        2) q(\theta) = p(\theta|x) i.e. using the posterior as proposal.

        The output of this function are the theta samples used in the approximation of
        the evidence and the corresponding weights w. If the prior distribution is
        needed, it is accessed via the task stored in the evaluator instance.

        Args:
            x: Tensor containing the observations. Has shape [n_obs, dim_obs]
            posterior_object: Trained posterior object as returned by the optimizer.
                If the optimizer uses ABC, fall back to the implementation using option
                1) i.e. use the prior as posterior. If NPE or NPE-PFN is used in the
                optimizer, use the specified version of the evidence estimation.
            num_samples: Number of theta samples used for each observation.
            mode: String indicating which mode for estimating the evidence is used,
                i.e. based on the prior or the posterior.
            context: Optional context tensor for the observations. Has shape
                [n_obs, dim_context]. If provided, it is concatenated with x before
                passing to the posterior for sampling and log-prob evaluation.

        Returns:
            theta_samples: Tensor containing the parameter samples drawn from q for the
                given observations. Has shape [n_obs, num_samples, dim_parameters].
            log_weights: Tensor containing the logarithm of the weights w_i for the
                different theta samples and observations. Has shape [n_obs, num_samples]
        """  # noqa: E501
        n_obs = x.shape[0]

        # ── Mode "prior_samples" or ABC fallback ──────────────────────────────
        # When the proposal is the prior (q = p(θ)), log importance weights are all
        # zero (log w = log(p(θ)/q(θ)) = log 1 = 0).
        use_prior = (mode == "prior_samples") or (posterior_object is None)

        if use_prior:
            if posterior_object is None and mode != "prior_samples":
                logger.debug(
                    "posterior_object is None (ABC optimizer): "
                    "falling back to prior-based evidence estimation."
                )
            self._flag_timeout_evidence = False
            return self._prior_sampling_fallback(n_obs, num_samples)

        # ── Mode "posterior_samples" (q = p(θ|x)) ────────────────────────────
        # Concatenate context with x if provided, matching the convention used
        # by the optimizer when calling sample_batched / log_prob_batched.
        if context is not None:
            x_ctxt = self.task.concatenate_context_data(data=x, context=context)
        else:
            x_ctxt = x

        # Sample B posterior samples per observation with a timeout.
        # sample_batched(with_log_prob=True) returns (samples, log_probs) in one
        # pass, avoiding a separate log_prob_batched call. For NPE-PFN this also
        # uses the exact autoregressive log-density computed during sampling rather
        # than the ratio-based estimator. Shapes: samples [B, n_obs, dim_θ],
        # log_probs [B, n_obs].
        old_sigalrm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._timeout_handler)
        timeout = int(max(1, self.config.sampling_timeout * max(1, n_obs)))
        signal.alarm(timeout)
        try:
            try:
                theta_samples_raw, log_posterior_raw = posterior_object.sample_batched(
                    sample_shape=(num_samples,),
                    x=x_ctxt.float(),
                    show_progress_bars=False,
                    with_log_prob=True,
                )  # [B, n_obs, dim_θ], [B, n_obs]
            except TimeoutError:
                logger.info(
                    "Evidence estimation timed out. Falling back to prior-based "
                    "evidence estimation."
                )
                logger.debug(
                    f"Posterior sampling timed out after {timeout}s during evidence "
                    "computation. Falling back to prior-based evidence estimation."
                )

                # Evaluation ran into a timeout
                self._flag_timeout_evidence = True

                # fallback with more samples to ensure we have enough since
                # posterior sampling will likely use much less samples
                return self._prior_sampling_fallback(
                    n_obs, num_samples * self.config.prior_sampling_fallback_factor
                )
        finally:
            # Ensure alarm and original signal handler are always restored,
            # including on unexpected exceptions and on normal return paths.
            signal.alarm(0)
            signal.signal(
                signal.SIGALRM,
                old_sigalrm_handler
                if old_sigalrm_handler is not None
                else signal.SIG_DFL,
            )

        # Reshape to [n_obs, B, dim_θ] for the output convention.
        theta_samples = theta_samples_raw.permute(1, 0, 2)  # [n_obs, B, dim_θ]

        # ── Compute importance weights: log w = log p(θ) − log p(θ|x) ────────

        # log p(θ): evaluate prior on flattened samples, reshape back.
        theta_flat = theta_samples.reshape(n_obs * num_samples, -1)  # [n_obs*B, dim_θ]
        log_prior = self.task.prior_dist.log_prob(theta_flat).reshape(
            n_obs, num_samples
        )

        log_posterior = log_posterior_raw.permute(1, 0)  # [n_obs, B]

        log_weights = log_prior - log_posterior  # [n_obs, B]

        # Evaluation did not run into a timeout
        self._flag_timeout_evidence = False

        return theta_samples, log_weights
