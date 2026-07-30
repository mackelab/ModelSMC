"""Compute evaluation metrics for all candidate × ground-truth model pairs using the
analytic (ground-truth) conditional likelihood in place of a learned NLE.

The script iterates over every combination of ground-truth data-generating model and
candidate simulator drawn from the fixed set of GMM configurations stored in
``gmm_configs.json``.

This produces the full N × N metric matrix needed to analyse the expected resampling
weights in the LLM-free SMC baseline (see accompanying notebook).  Each
(ground-truth, candidate) pair is evaluated with the same random seed, making
individual cells independently reproducible.

Usage (run from Fig_G3_appendix_ModelSMC_validation_no_LLM/)
-----
python scripts/compute_metrics_with_gt_likelihood.py partition=a100_single +experiment=minimal_example_no_LLM_ABC_NLE ++hydra.sweeper.params=null ++method.evaluator.likelihood_estimator=task-specific seed=0,1,2,3,4 hydra.sweep.dir=scripts/hydra_outputs

or

python scripts/compute_metrics_with_gt_likelihood.py partition=none launcher=local +experiment=minimal_example_no_LLM_ABC_NLE ++hydra.sweeper.params=null ++method.evaluator.likelihood_estimator=task-specific seed=0,1,2,3,4 hydra.sweep.dir=scripts/hydra_outputs --multirun

for local evaluation.
"""  # noqa

import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

from modelsmc.tasks.minimal_example_n_dim import GMM_configs
from modelsmc.tasks.minimal_example_n_dim.GMM import GMM
from modelsmc.tasks.minimal_example_n_dim.minimal_example_task_n_dim import (
    MinimalExampleNDim,
)
from modelsmc.utils import utils

os.environ["DSPY_CACHEDIR"] = ".dspy_cache_eval"

logger = logging.getLogger("ModelSMC")


class GMM_extended(GMM):
    """GMM subclass that adds an analytic conditional log-likelihood.

    Inherits sampling from :class:`~llm_discovery.tasks.minimal_example_n_dim.GMM.GMM`
    and adds :meth:`log_prob_conditional`, which evaluates the density of observations
    x under the affine transformation

        x = c · z + shift,   z ~ GMM(μ_j, Σ_j, w_j)

    induced by parameters θ = (s_1, …, s_k, c), where s_1, …, s_k are additive
    shifts along the ``dims_to_shift`` dimensions and c > 0 is a global scale factor.
    Evaluation uses the change-of-variables formula rather than constructing a new
    transformed GMM per observation:

        log p(x | θ) = log p_z((x − shift) / c) − d · log(c)

    where p_z is the base GMM density evaluated via the pre-built components in the
    parent class.  This class is used as the ground-truth likelihood estimator in the
    LLM-free SMC evaluation.

    Args:
        covs_list:     List of base covariance matrices, one per component.
                       Each element has shape (dim, dim).
        means_list:    List of base mean vectors, one per component.
                       Each element has shape (dim,).
        dims_to_shift: Indices of the dimensions that are shifted by θ.
        weights_list:  Mixture weights.  Defaults to uniform if None.
    """

    def __init__(self, covs_list, means_list, dims_to_shift, weights_list=None):
        super().__init__(covs_list, means_list, dims_to_shift, weights_list)

    def log_prob_conditional(self, x: Tensor, theta: Tensor):
        """Compute the analytic log-likelihood log p(x | θ, model) for a batch of
        observations and parameters.

        Uses the change-of-variables formula for the affine transformation
        x = c · z + shift.  Inverting gives z = (x − shift) / c, with Jacobian
        ∂z/∂x = (1/c) · I_d, so:

            log p(x | θ) = log p_z((x − shift) / c) − d · log(c)

        where p_z is the *base* (unparameterized) GMM density, evaluated in one
        batched call to :meth:`log_prob_uncond` using the pre-built components stored
        in the parent class.  This avoids constructing N new GMM objects.

        Args:
            x:     Observations. Shape: (N, 10).
            theta: Parameters. Shape: (1, 5) or (N, 5).
                   Each row contains (s_1, …, s_k, c) where s_1, …, s_k are shifts
                   along the ``dims_to_shift`` dimensions and c is the global scale.
                   If shape is (1, 5), the same θ is broadcast to all observations.

        Returns:
            log_probs: Per-observation log-likelihoods log p(x_i | θ_i). Shape: (N,).
        """

        assert theta.shape[0] in [1, x.shape[0]]
        assert x.shape[1] == 10
        assert theta.shape[1] == 5

        # Broadcast a single parameter vector to all observations if needed.
        if theta.shape[0] == 1:
            theta = theta.repeat(x.shape[0], 1)
            assert theta.shape[0] == x.shape[0]

        # Build the full-dimensional shift matrix for all N observations at once.
        # Rows are zero except at dims_to_shift, which receive s_1, …, s_k.
        shift = torch.zeros(x.shape[0], self.dim)
        shift[:, self.dims_to_shift] = theta[:, : self.num_dims_to_shift]

        # Extract the global scale factor c for each observation.  Shape: (N,).
        scale = theta[:, -1]

        # Invert the affine transformation: z = (x − shift) / c.
        # scale is unsqueezed to (N, 1) to broadcast across the feature dimension.
        z = (x - shift) / scale.unsqueeze(1)  # shape: (N, d)
        assert z.shape == x.shape

        # Evaluate the base GMM log-density for all N back-transformed points in one
        # batched call.  log_prob_uncond uses the MultivariateNormal components that
        # were built once at construction time — no new GMM objects are created.
        log_p_z = self.log_prob_uncond(x=z)  # shape: (N,)
        assert log_p_z.shape == torch.Size((x.shape[0],))

        # Apply the log-Jacobian correction −d · log(c).
        # The Jacobian of z = x/c is (1/c)^d, so log|det J| = −d · log(c).
        log_probs = log_p_z - self.dim * torch.log(scale)  # shape: (N,)
        assert log_probs.shape == torch.Size((x.shape[0],))

        return log_probs


class GroundTruthLikelihoodEstimator:
    """
    Training-free likelihood estimator that uses the analytic GMM conditional density.

    Implements the same interface as the learned NLE (``fit_likelihood_estimator`` and
    ``loss``) so it can serve as a drop-in replacement inside
    :class:`~llm_discovery.method.modules.evaluator.EvaluatorBase`.

    The estimator evaluates the analytic conditional log-likelihood of the *candidate*
    model (identified by ``task.evaluated_GMM_idx``).

    Note that the candidate model (``evaluated_GMM_idx``) is distinct from the
    ground-truth data-generating model (``gt_GMM_configuration_index``).  The evaluator
    measures how well the candidate model explains data generated by the ground-truth
    model.
    """

    def __init__(self, task, config, *args, **kwargs):
        """Instantiate the estimator for the candidate model specified by the task.

        Loads the GMM configuration of the *candidate* model (not the ground-truth
        data-generating model) and constructs a :class:`GMM_extended` instance to
        evaluate the analytic likelihood.

        Args:
            task:   Task object.  ``task.evaluated_GMM_idx`` must be set to the index
                    of the candidate model before this constructor is called.
            config: Hydra evaluator config (unused; kept for interface compatibility).
        """
        # Load the candidate model's GMM configuration.  This is the model whose
        # likelihood we evaluate, NOT the model that generated the observed data.
        config_idx = GMM_configs.configs[task.evaluated_GMM_idx]

        covs_list = [torch.tensor(cov) for cov in config_idx.covs]
        means_list = [torch.tensor(mean) for mean in config_idx.means]
        weights_list = config_idx.weights
        dims_to_shift = config_idx.dims_to_shift

        # Initialize the GMM with the analytic conditional likelihood.
        self.__gmm = GMM_extended(
            covs_list=covs_list,
            means_list=means_list,
            dims_to_shift=dims_to_shift,
            weights_list=weights_list,
        )

        logger.info(
            "initialize GMM with ground-truth configuration"
            f"idx = {task.evaluated_GMM_idx}"
        )

    def fit_likelihood_estimator(self, theta_train: Tensor, x_train: Tensor):
        """No-op: the analytic likelihood requires no training.

        This method exists only to satisfy the NLE interface expected by
        :class:`~llm_discovery.method.modules.evaluator.EvaluatorBase`.

        Args:
            theta_train: Unused training parameters.
            x_train:     Unused training observations.
        """
        return

    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        """Return the negative analytic log-likelihood of observations given parameters.

        Computes −log p(x | θ, model_m) element-wise using the ground-truth conditional
        density of the candidate model.  The sign convention (lower = better) matches
        the NLE interface used throughout the evaluator.

        Args:
            input:     Observations x. Shape: (N, dim_obs).
            condition: Parameters θ. Shape: (1, dim_θ) or (N, dim_θ).

        Returns:
            neg_log_probs: Negative log-likelihoods −log p(x_i | θ_i). Shape: (N,).
        """
        return -self.__gmm.log_prob_conditional(x=input, theta=condition)


class MinimalExampleNDim_extended(MinimalExampleNDim):
    """Extended task class that wires in the ground-truth likelihood estimator.

    Extends :class:`MinimalExampleNDim` with two additions required for the
    LLM-free evaluation:

    * ``evaluated_GMM_idx`` — index (into ``GMM_configs.configs``) of the *candidate*
      model whose likelihood is being evaluated.  This is distinct from
      ``gt_GMM_configuration_index``, which identifies the data-generating model.
    * ``custom_likelihood_estimator_class`` — set to
      :class:`GroundTruthLikelihoodEstimator` so that
      :class:`~llm_discovery.method.modules.evaluator.EvaluatorBase` uses the analytic
      likelihood instead of training a neural estimator.

    Args:
        evaluated_GMM_idx:   Index of the candidate GMM.  May be ``None`` at
                             construction time and assigned later before evaluation.
        config:              Hydra task config.
        prompts_path:        Passed through to the parent class (unused here).
        base_simulator_path: Passed through to the parent class (unused here).
    """

    def __init__(
        self,
        evaluated_GMM_idx: int,
        config,
        prompts_path=None,
        base_simulator_path=None,
    ):
        super().__init__(config, prompts_path, base_simulator_path)

        # Index of the candidate model to be evaluated.  This is NOT the model from
        # which the validation data is drawn (that is gt_GMM_configuration_index).
        self.evaluated_GMM_idx = evaluated_GMM_idx

        # Override the likelihood estimator class so EvaluatorBase instantiates
        # GroundTruthLikelihoodEstimator instead of a neural NLE.
        self.custom_likelihood_estimator_class = GroundTruthLikelihoodEstimator


def set_seeds(seed):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="../../../config", config_name="config.yaml")
def main(config: DictConfig):
    """Hydra entry point: evaluate all candidate × ground-truth model pairs.

    Iterates over the full N × N matrix of (ground-truth model, candidate model) index
    pairs.  For each pair the optimizer infers parameters for the candidate model on
    data generated by the ground-truth model, and the evaluator scores the fit using
    the analytic conditional likelihood of the candidate model.  The seed is reset at
    the start of each pair so that every (ground-truth, candidate) cell is
    independently reproducible regardless of evaluation order.

    Results are appended incrementally to a shared CSV file using a file lock to allow
    safe concurrent writes when running as a Hydra multi-run sweep (e.g. over seeds).

    Args:
        config: Hydra DictConfig assembled from the config directory.  Expected to
                contain ``method``, ``task``, ``seed``, and ``logging`` sub-configs.
    """

    script_dir = Path(__file__).parent.parent
    csv_path = os.path.join(script_dir, "compute_metrics_with_gt_likelihood.csv")

    logger.info(OmegaConf.to_yaml(config))

    level = getattr(logging, config.logging.level)
    logger.setLevel(level)

    # Propagate the parameter-estimation flag from the task config to both the
    # optimizer and evaluator so they stay in sync.
    with open_dict(config):
        config.method.evaluator.estimate_parameters = (
            config.task.compute_parameter_estimate
        )
        config.method.optimizer.estimate_parameters = (
            config.task.compute_parameter_estimate
        )

    # Communicate hardware resource requirements to Ray workers via
    # environment variables.
    os.environ["REQ_NUM_CPUS_PER_RAY_THREAD"] = (
        f"{config.method.req_num_cpus_per_ray_thread}"
    )
    os.environ["REQ_NUM_GPUS_PER_RAY_THREAD"] = (
        f"{config.method.req_num_gpus_per_ray_thread}"
    )

    # Deferred import to ensure environment variables are set
    from modelsmc.method.modules.evaluator import EvaluatorBase
    from modelsmc.method.modules.optimizer import create_optimizer

    logger.info(OmegaConf.to_yaml(config))

    num_models = len(GMM_configs.configs)
    logger.info(f"Found {num_models} candidate models")

    # ==============================================================================
    # Main evaluation loop: iterate over all N × N (ground-truth, candidate) pairs
    # ==============================================================================

    for idx_ground_truth_model in range(num_models):
        for idx_evaluated_model in range(num_models):
            # Reset the seed at the start of each pair so that each (ground-truth,
            # candidate) cell is independently reproducible.
            set_seeds(config.seed)

            # --- Task setup ---
            # Initialise the task with the candidate model index; the ground-truth
            # index is set separately below since data generation only depends on
            # gt_GMM_configuration_index.
            task = MinimalExampleNDim_extended(
                config=config.task, evaluated_GMM_idx=idx_evaluated_model
            )

            # Set which GMM generates the observed data.
            task.gt_GMM_configuration_index = idx_ground_truth_model

            # Draw observed data from the ground-truth model at the identity parameter
            # θ = (0, …, 0, 1) — no shift, unit scale.  Training and validation sets
            # are identical for this task (see MinimalExampleNDim.get_data).
            training_data_raw, valid_data_raw = task.get_data()

            logger.info(
                f"Evaluate idx_ground_truth_model = {task.gt_GMM_configuration_index}, "
                f"idx_evaluated_model = {task.evaluated_GMM_idx}"
            )

            # --- Simulator initialisation ---
            # Instantiate the candidate model's simulator by filling the GMM config
            # index placeholder in the base simulator template.
            implementation_idx = task.get_skeleton_implementation().format(
                GMM_CONFIG_IDX=task.evaluated_GMM_idx
            )
            simulator, error_dict = utils.convert_model_string(implementation_idx)

            # --- Parameter inference ---
            # Run the SBI optimizer
            optimizer = create_optimizer(
                config=config.method.optimizer, task=task, verbose=False
            )

            optim_metrics, parameter_estimates = optimizer.optimize(
                error_dict=error_dict,
                simulator=simulator,
                training_data_raw=training_data_raw,
                save_dir="./outputs",
                valid_data_raw=valid_data_raw,
            )

            # --- Evaluation ---
            # Score the candidate model using the analytic conditional likelihood
            evaluator = EvaluatorBase(
                config=config.method.evaluator, task=task, verbose=False
            )

            eval_metrics = evaluator.evaluate(
                parameter_estimates=parameter_estimates,
                optimizer_output=optim_metrics,
                simulator=simulator,
                valid_data_raw=valid_data_raw,
                save_dir="./outputs",
            )

            # --- Results persistence ---
            # Collect scalar metrics into a single-row DataFrame and append it to the
            # shared CSV file.
            row_dict = {
                "seed": [config.seed],
                "idx_ground_truth_model": [task.gt_GMM_configuration_index],
                "evaluated_GMM_idx": [task.evaluated_GMM_idx],
            }

            for key in eval_metrics["eval_metrics"]:
                row_dict[key] = [eval_metrics["eval_metrics"][key]]

            row_dict["config"] = [str(OmegaConf.to_container(config))]

            new_row = pd.DataFrame(row_dict)

            lock_file = csv_path + ".lock"
            lock = FileLock(lock_file)

            with lock:
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path)

                    # Check if either DataFrame is empty to avoid FutureWarning
                    if existing_df.empty:
                        df = new_row
                    else:
                        df = pd.concat(
                            [existing_df, new_row], ignore_index=True, sort=False
                        )

                else:
                    df = new_row

                df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
