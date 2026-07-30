import copy
import json
import logging
import math
import numbers
import os
import shutil
import time
from typing import Callable

import dspy
import numpy as np
import pandas as pd
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from modelsmc.method.modules.codingsimulator import CodingSimulatorModule
from modelsmc.method.modules.dataclasses import (
    Ancestor,
    DiscoveryContext,
    Particle,
    ParticlePool,
)
from modelsmc.method.modules.evaluator import EvaluatorBase
from modelsmc.method.modules.feedback import FeedbackModule
from modelsmc.method.modules.optimizer import OptimizerBase, create_optimizer
from modelsmc.tasks.base_embedding_net import BaseEmbeddingHandler
from modelsmc.tasks.base_task import BaseTask
from modelsmc.utils import utils
from modelsmc.utils.plotting import plot_pool, plot_trajectories
from modelsmc.utils.utils import (
    detect_code_type,
    register_omegaconf_resolvers,
    setup_worker_logging,
)

logger = logging.getLogger("ModelSMC")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────


def _sum_token_usage(history: list) -> dict:
    """Aggregate LLM token counts from a DSPy LM call history.

    Args:
        history: List of history entries returned by ``dspy.LM.history``,
                 each containing a nested ``usage`` dict.

    Returns:
        Dict with keys ``total_tokens``, ``prompt_tokens``,
        ``completion_tokens`` summed over all entries.
    """
    total = prompt = completion = 0
    for h in history:
        usage = h.get("usage", {})
        total += usage.get("total_tokens", 0)
        prompt += usage.get("prompt_tokens", 0)
        completion += usage.get("completion_tokens", 0)
    return {
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


#########################################################
# SMC Discovery Engine
#########################################################


class SMCEngine:
    """Core engine for Sequential Monte Carlo (SMC) model discovery.

    Maintains a population (pool) of N particles, where each particle
    represents a candidate simulator — a piece of code that defines a
    stochastic data-generating model m with free parameters theta.
    The engine iterates three steps to refine the population:

        1. Resample  – draw N particles from the pool with replacement,
                       weighting each by its fitness score.
        2. Propagate – mutate each particle via an LLM, producing a new
                       simulator code variant.
        3. Weight    – evaluate each particle: fit parameters with SBI,
                       compute metrics on validation data, and assign a
                       scalar log-weight.

    Propagation and weighting are parallelised across particles using Ray.

    Args:
        config:     Method-level Hydra config (``config.method``).
        task:       Task object providing data, evaluation, and task
                    description.
        output_dir: Root directory for per-particle result files.
    """

    def __init__(self, config: DictConfig, task: BaseTask, output_dir: str):
        self.config = config
        self.output_dir = output_dir

        self.simcoder = CodingSimulatorModule(
            task,
            verbose=config.verbose,
            use_chain_of_thought=config.use_chain_of_thought,
        )
        self.optimizer = create_optimizer(config.optimizer, task, config.verbose)
        self.evaluator = EvaluatorBase(config.evaluator, task, config.verbose)
        self.feedback = FeedbackModule(
            task, config.verbose, feedback_mode=config.feedback_mode
        )
        self.compiled_gen_simulator = None

        # Global particle counter; incremented after each weighting step.
        self.__particle_count = 0
        self.itr = 0

        # ── Ray initialisation ────────────────────────────────────────────
        # Ray parallelises propagation and weighting across particles.
        # Hardware limits are read from the Hydra launcher config when
        # available; otherwise sensible defaults are used.
        # See https://docs.ray.io/en/latest/ray-core/configure.html for
        # advanced configuration options.

        hc = HydraConfig.get()

        try:
            num_cpus = hc.launcher.cpus_per_task
            # SlurmQueueConf uses gpus_per_task; LocalQueueConf uses gpus_per_node.
            # Use explicit None checks so that explicit `0` values are respected.
            _gpus_val = OmegaConf.select(hc.launcher, "gpus_per_task")
            if _gpus_val is None:
                _gpus_val = OmegaConf.select(hc.launcher, "gpus_per_node")
            num_gpus = _gpus_val if _gpus_val is not None else 0
        except Exception as e:
            logger.warning(f"{e}: Failed to read hardware config. Using defaults.")
            num_cpus = 5
            num_gpus = 0

        try:
            total_mem_gb = hc.launcher.mem_gb
        except Exception as e:
            logger.warning(f"{e}: Failed to read memory config. Using defaults.")
            total_mem_gb = 8

        # Reserve 20 % of total memory for the Ray object store.
        object_storage_gb = int(0.2 * total_mem_gb)
        logger.debug(f"Ray object store memory (GB): {object_storage_gb}")

        # Always forward worker stdout to the driver so that logging statements
        # configured by setup_worker_logging() inside each remote function reach
        # the main process output.  Ray's own internal logging is kept quiet at
        # ERROR level regardless of the user-configured level.
        ray.init(
            ignore_reinit_error=False,
            num_cpus=int(num_cpus),
            num_gpus=int(num_gpus),
            object_store_memory=object_storage_gb * 1024**3,
            logging_level=logging.ERROR,
            log_to_driver=True,
        )

        tot = ray.cluster_resources()
        avail = ray.available_resources()

        logger.debug("=== Ray resources (cluster-wide) ===")
        logger.debug(
            f"CPUs:  total={self.__get_info(tot, 'CPU'):.0f}  in_use="
            f"{self.__get_info(tot, 'CPU') - self.__get_info(avail, 'CPU'):.0f}  free="
            f"{self.__get_info(avail, 'CPU'):.0f}"
        )
        logger.debug(
            f"GPUs:  total={self.__get_info(tot, 'GPU'):.0f}  in_use="
            f"{self.__get_info(tot, 'GPU') - self.__get_info(avail, 'GPU'):.0f}  free="
            f"{self.__get_info(avail, 'GPU'):.0f}"
        )
        logger.debug(
            f"Object store: {self.__format_bytes(tot.get('object_store_memory', 0))}"
        )
        logger.debug(f"Memory:       {self.__format_bytes(tot.get('memory', 0))}")
        logger.debug("Ray remote resource env-vars:")
        logger.debug(
            "  REQ_NUM_GPUS_PER_RAY_THREAD: "
            f"{int(os.getenv('REQ_NUM_GPUS_PER_RAY_THREAD'))}"
        )
        logger.debug(
            "  REQ_NUM_CPUS_PER_RAY_THREAD: "
            f"{int(os.getenv('REQ_NUM_CPUS_PER_RAY_THREAD'))}"
        )

        # Cumulative LLM token usage across the full experiment.
        self.llm_tokens = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def __get_info(self, res: dict, k: str) -> float:
        """Return a Ray resource value as float, defaulting to 0."""
        return float(res.get(k, 0))

    def __format_bytes(self, n: float) -> str:
        """Format a byte count as a human-readable string (e.g. '3.2 GB')."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024

    def _get_save_dir(self, particle_index: int) -> str:
        """Create and return the output directory for a single particle.

        Args:
            particle_index: Global particle index used to name the directory.

        Returns:
            Absolute path to the newly created directory.
        """
        save_dir = os.path.join(
            self.output_dir,
            f"iter-{self.itr}_p{particle_index}",
        )
        os.makedirs(save_dir, exist_ok=False)
        return save_dir

    @staticmethod
    def _save_llm_history(
        particle: Particle, save_llm_history: bool, save_dir: str, history: list[dict]
    ) -> None:
        """Persist the propagation and feedback LLM histories to disk.

        Args:
            particle:         Particle whose propagation history is written (if
                              present on ``particle._llm_history_propagation``).
            save_llm_history: If False, this is a no-op.
            save_dir:         Directory in which the JSON history files are written.
            history:          Feedback-step LLM history to serialise.
        """
        if save_llm_history:
            with open(os.path.join(save_dir, "llm_history_feedback.json"), "w") as f:
                json.dump(history, f, default=str)
            propagation_history = getattr(particle, "_llm_history_propagation", None)
            if propagation_history is not None:
                with open(
                    os.path.join(save_dir, "llm_history_propagation.json"), "w"
                ) as f:
                    json.dump(propagation_history, f, default=str)

    # ─────────────────────────────────────────────────────────────────────
    # Main interfaces
    # ─────────────────────────────────────────────────────────────────────

    def init_particle_pool(
        self, context: DiscoveryContext
    ) -> tuple[ParticlePool, None, dict]:
        """Evaluate the base simulator and initialise the particle pool.

        Loads the hand-coded base simulator provided by the task, evaluates
        it once (parameter optimisation + fitness scoring), and fills the pool
        with ``particle_pool_size`` identical copies of the evaluated particle.
        All particles therefore start from the same prior hypothesis, which the
        SMC iterations then diversify via LLM-guided mutations.

        Args:
            context: Discovery context carrying task, data, and save path.

        Returns:
            pool:       Initialised ``ParticlePool`` of size
                        ``config.particle_pool_size``.
            None:       Placeholder for the ESS (not computed at init).
            llm_tokens: Cumulative token usage dict for this call.
        """
        implementation = context.task.get_base_simulator()

        particle = Particle(implementation=implementation)
        pool = ParticlePool(particles=[particle])

        # Evaluate the single seed particle.
        pool = self.weight(pool=pool, context=context)

        completed_particle = pool.get_particles()[0]

        # Replicate the evaluated seed to fill the pool.
        particles = [
            copy.deepcopy(completed_particle)
            for _ in range(self.config.particle_pool_size)
        ]
        pool = ParticlePool(particles=particles)

        logger.debug(
            f"Initialised particle pool with {len(pool)} copies of the base simulator"
        )

        # Return the cumulative token usage populated by self.weight(), consistent
        # with smc_step(). The feedback LLM runs inside Ray workers (their own LM
        # clients), so dspy.settings.lm.history on the driver is empty and must not
        # be used here. Note: released summary files predate this correction, which
        # only affects the iteration-0 token value; cumulative totals are unaffected.
        return pool, None, self.llm_tokens

    def smc_step(
        self, pool: ParticlePool, context: DiscoveryContext
    ) -> tuple[ParticlePool, float, dict]:
        """Execute one full SMC iteration: resample → propagate → weight.

        Args:
            pool:    Current particle pool.
            context: Discovery context carrying task, data, and save path.

        Returns:
            pool:       Updated particle pool after one iteration.
            ess:        Effective Sample Size before resampling, as computed by
                        ``resample`` (a float in [1, N]).
            llm_tokens: Cumulative LLM token usage dict.
        """
        time_start = time.time()

        pool, ess = self.resample(pool)
        pool = self.propagate(pool=pool, context=context)
        pool = self.weight(pool, context)

        logger.info(f"SMC step duration: {time.time() - time_start:.4g} s")

        return pool, ess, self.llm_tokens

    # ─────────────────────────────────────────────────────────────────────
    # Resampling
    # ─────────────────────────────────────────────────────────────────────

    def resample(self, pool: ParticlePool) -> tuple[ParticlePool, float]:
        """Resample the particle pool with replacement using systematic resampling.

        Draws N particles from the current pool of N particles with
        replacement, where the sampling probability of particle i is
        proportional to its temperature-scaled fitness weight W_i (see
        ``_get_resampling_weights``).  High-fitness simulators are therefore
        selected as parents more often, driving the population towards better
        models.

        Resampling is skipped when the relative ESS = 1/(N*sum_i W_i^2)
        exceeds ``config.ess_threshold``, avoiding diversity loss when the
        weight distribution is still close to uniform.

        Systematic resampling following the suggested implementation in:
        # Naesseth et al. (2019), Elements of Sequential Monte Carlo, FnTML 12(3)

        Args:
            pool: Current particle pool of size N.

        Returns:
            new_pool: Resampled pool of the same size N (or unchanged if
                      resampling was skipped).
            ess:      Effective Sample Size in [1, N].
        """
        weights = self._get_resampling_weights(pool)

        if np.sum(weights) != 1.0:
            weights = weights / np.sum(weights)

        # Compute ESS
        N = len(pool)
        ess = 1.0 / np.sum(weights**2)  # ESS = 1/sum(W_i^2), in [1, N]
        ess_rel = ess / N

        logger.debug(
            f"ESS = {ess:.4g} / {N}  (ESS/N = {ess_rel:.4g}, threshold = "
            f"{self.config.ess_threshold})"
        )

        # Skip resampling if ESS/N >= threshold
        if ess_rel >= self.config.ess_threshold:
            logger.debug("ESS/N >= threshold — skipping resampling.")
            return pool, ess

        # Systematic resampling
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0  # clamp for floating-point drift
        u0 = np.random.rand() / N
        u = u0 + np.arange(N) / N  # N evenly spaced points in [0, 1)
        idx = np.searchsorted(cdf, u, side="right")

        particles = pool.get_particles()
        selected_particles = [particles[i] for i in idx]
        new_pool = ParticlePool(selected_particles)

        assert len(pool) == len(new_pool)

        return new_pool, ess

    def _get_resampling_weights(self, pool: ParticlePool) -> np.ndarray:
        """Compute normalised resampling probabilities from particle log-weights.

        Applies temperature scaling followed by a softmax (via the log-sum-exp
        trick) to convert raw log-weights into a valid probability distribution:

            W_i = exp(log w_i / T) / sum_j exp(log w_j / T)

        where T is ``config.temperature``.  A temperature T > 1 flattens the
        distribution (reduced selection pressure); T < 1 sharpens it towards
        winner-take-all selection.  Weights that are NaN (evaluation failure)
        are treated as probability zero.

        Args:
            pool: Particle pool whose log-weights are used.

        Returns:
            weights: Array of shape (N,) summing to 1.
        """
        log_weights = pool.get_log_weights()

        logger.debug(f"Particles considered for resampling: {len(log_weights)}")

        # Degenerate case: all evaluations failed.
        if np.all(np.isnan(log_weights)):
            logger.info("All log-weights are NaN — falling back to uniform resampling.")
            n = len(log_weights)
            return np.ones(n) / n

        # If all log weights are -inf, the resampling fails. To avoid this, uniform
        # weights are used for resampling.
        if np.all(np.isinf(log_weights)):
            logger.info(
                "All log-weights are -inf — falling back to uniform resampling."
            )
            n = len(log_weights)
            return np.ones(n) / n

        # Treat failed evaluations as having probability zero.
        log_weights = np.nan_to_num(log_weights, nan=-np.inf)
        log_weights = torch.from_numpy(log_weights).float()

        # Temperature scaling + log-sum-exp normalisation.
        log_weights_tempered = log_weights / self.config.temperature
        log_Z = torch.logsumexp(log_weights_tempered, dim=0)
        normalized_log_weights = log_weights_tempered - log_Z

        logger.debug(f"Log partition function log Z: {log_Z.item():.4g}")
        logger.debug(f"Normalised log-weights: {list(normalized_log_weights)}")

        weights = torch.exp(normalized_log_weights).numpy()

        # Numerical guard: if all weights collapsed to zero, use uniform.
        if np.all(np.isclose(weights, 0.0)):
            logger.debug("All weights are numerically zero — using uniform resampling.")
            weights = np.ones_like(weights) / len(weights)

        # Renormalise to correct for floating-point rounding errors.
        weights = weights / np.sum(weights)

        assert np.isclose(np.sum(weights), 1.0, atol=1e-10), (
            f"Weights do not sum to 1 (sum={np.sum(weights):.16f}); "
            "this may indicate a genuine numerical problem upstream."
        )

        logger.debug(f"Resampling weights: {list(weights)}")
        logger.debug(f"Weight sum deviation from 1: {np.sum(weights) - 1:.3g}")

        return weights

    # ─────────────────────────────────────────────────────────────────────
    # Propagation
    # ─────────────────────────────────────────────────────────────────────

    def propagate(
        self,
        pool: ParticlePool,
        context: DiscoveryContext,
    ) -> ParticlePool:
        """Propagate all particles by sampling from the LLM proposal.

        Dispatches one ``sample_proposal`` Ray task per particle.  Each task
        independently decides (via a Bernoulli draw) whether to copy the
        particle unchanged or to call the LLM to generate a mutated simulator
        code conditioned on the particle's ancestry and feedback.

        Args:
            pool:    Current particle pool (typically after resampling).
            context: Discovery context with task description and data.

        Returns:
            Updated pool containing the propagated particles.
        """
        llm_kwargs = dict(
            dspy.settings.lm.kwargs
        )  # shallow copy — avoid mutating global LM state
        llm_model = dspy.settings.lm.model

        # GPT-5 uses 'max_completion_tokens'; normalise to 'max_tokens'.
        if "max_completion_tokens" in llm_kwargs:
            llm_kwargs["max_tokens"] = llm_kwargs["max_completion_tokens"]
            del llm_kwargs["max_completion_tokens"]

        # Check if the full histories are saved for each particle.
        # By default, do not save.
        save_llm_history = getattr(self.config, "save_llm_history", False)

        logger.debug(
            f"CUDA_VISIBLE_DEVICES before propagation: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        futures = [
            self.sample_proposal.options(retry_exceptions=True, max_retries=1).remote(
                particle_i,
                copy.deepcopy(self.simcoder),
                copy.deepcopy(self.config.few_shot_examples),
                copy.deepcopy(context),
                copy.deepcopy(self.config.alpha),
                copy.deepcopy(llm_model),
                copy.deepcopy(llm_kwargs),
                save_llm_history,
            )
            for particle_i in pool.get_particles()
        ]

        results = ray.get(futures)
        new_particles, llm_tokens = zip(*results, strict=False)
        new_particles = list(new_particles)
        llm_tokens = list(llm_tokens)

        for key in self.llm_tokens:
            self.llm_tokens[key] += sum(t.get(key, 0) for t in llm_tokens)

        assert isinstance(new_particles, list)
        assert all(isinstance(p, Particle) for p in new_particles)
        assert len(pool.get_particles()) == len(new_particles)

        return ParticlePool(new_particles)

    @staticmethod
    @ray.remote(
        num_cpus=int(os.getenv("REQ_NUM_CPUS_PER_RAY_THREAD")),
        num_gpus=int(os.getenv("REQ_NUM_GPUS_PER_RAY_THREAD")),
    )
    def sample_proposal(
        particle: Particle,
        simcoder: CodingSimulatorModule,
        few_shot_examples: int,
        context: DiscoveryContext,
        alpha: float,
        model: str,
        llm_kwargs: dict,
        save_llm_history: bool = False,
    ) -> tuple[Particle, dict]:
        """Sample a new particle from the proposal distribution (Ray remote).

        The proposal is a mixture of an identity (copy) kernel and an
        LLM-based mutation kernel:

            q(m' | m) = alpha * delta_{m'=m}
                      + (1-alpha) * q_LLM(m' | m, ancestors, feedback)

        With probability alpha the input particle is returned unchanged.
        Otherwise the LLM generates a new simulator code, conditioned on the
        particle's ancestry (up to ``few_shot_examples`` previous versions)
        and the diagnostic feedback from the last evaluation.  The ancestry
        serves as few-shot demonstrations for the LLM, allowing it to see the
        trajectory of refinements that led to the current code.

        This function runs in a Ray worker and creates its own LLM client to
        avoid shared-state issues between parallel workers.

        Args:
            particle:          Input particle to propagate.
            simcoder:          DSPy-based code-generation module.
            few_shot_examples: Maximum number of ancestors included as LLM
                               context (most recent first).
            context:           Discovery context with task description.
            alpha:             Copy probability in [0, 1].
            model:             LiteLLM model identifier string.
            llm_kwargs:        Additional keyword arguments forwarded to the
                               LLM.
            save_llm_history:  If True, store the worker's LLM call history on
                               the propagated particle for later persistence.

        Returns:
            new_particle: Propagated (or copied) particle.
            llm_tokens:   Token usage dict for this call.
        """
        # Configure the logger for this worker process so that log records are
        # forwarded to the driver via Ray's stdout piping.
        logger = setup_worker_logging()

        # Log GPU availability in the worker for debugging.
        available_gpu_ids_ray = [int(i) for i in ray.get_gpu_ids()]
        try:
            logger.debug(f"Ray GPU ids: {available_gpu_ids_ray}")
            logger.debug(
                "CUDA_VISIBLE_DEVICES in worker: "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.debug(
                "Device name: "
                f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
        except Exception as e:
            logger.debug(f"GPU info unavailable: {e}")

        # Each Ray worker initialises its own LLM client with a clean history.
        lm_temporary = dspy.LM(model=model, cache=False, **llm_kwargs)
        if len(lm_temporary.history) != 0:
            lm_temporary.history = []

        simcoder.set_lm(lm_temporary)

        z = np.random.rand()
        assert (0.0 <= alpha) and (1.0 >= alpha)

        if z <= alpha:
            # Identity move: return the particle unchanged.
            logger.debug(f"Copying particle uuid={particle.uuid}")
            new_particle = particle

        else:
            # LLM move: generate a mutated simulator code.
            logger.debug(f"Propagating particle uuid={particle.uuid}")

            t0 = time.time()

            # Wrap the current particle as an ancestor for the LLM context.
            ancestor = Ancestor(
                implementation=particle.implementation,
                feedback=particle.feedback,
                scm_definition=particle.scm_definition,
                uuid=particle.uuid,
                log_weight=particle.log_weight,
            )

            # Append the current particle and keep the most recent ancestors.
            # DSPy preserves order for the few shot examples (oldest first).
            # Guard against few_shot_examples == 0, for which the slice
            # ``[-0:]`` would return the full list instead of an empty one.
            all_ancestors = particle.ancestors + [ancestor]
            ancestors = (
                all_ancestors[-few_shot_examples:] if few_shot_examples > 0 else []
            )

            logger.debug(f"Using {len(ancestors)} ancestor(s) as LLM context")

            # Build few-shot training examples from the ancestry chain.
            # Input fields must be present for DSPy's format_demos() to
            # include the examples (it requires has_input AND has_output).
            # The actual values are identical across all demos and repeated
            # in the final user message, so we use short placeholders to
            # avoid massive token overhead.
            _input_placeholder = "(see current input)"
            trainset_sim = [
                dspy.Example(
                    signature_description=_input_placeholder,
                    task_description=_input_placeholder,
                    base_simulator=_input_placeholder,
                    simulator_code=ancestor_i.implementation,
                    feedback=ancestor_i.feedback,
                    scm_definition=ancestor_i.scm_definition,
                )
                for ancestor_i in ancestors
            ]

            dspy_optimizer = dspy.LabeledFewShot(k=few_shot_examples)
            compiled_simcoder = dspy_optimizer.compile(
                simcoder, trainset=trainset_sim, sample=False
            )
            compiled_simcoder.set_lm(lm=lm_temporary)

            simcoder_results = compiled_simcoder()

            dt_propagation = time.time() - t0

            new_particle = Particle(
                implementation=simcoder_results.code,
                scm_definition=simcoder_results.scm_definition,
                ancestors=ancestors,
                times={"dt_propagation_sec": dt_propagation},
            )

            logger.debug(f"Created new particle uuid={new_particle.uuid}")

        llm_tokens = _sum_token_usage(lm_temporary.history)

        # Save the llm history of the propagation step into the particle for saving
        # once the results folder for this particle is created.
        if save_llm_history and lm_temporary.history:
            new_particle._llm_history_propagation = list(lm_temporary.history)

        return new_particle, llm_tokens

    # ─────────────────────────────────────────────────────────────────────
    # Weighting
    # ─────────────────────────────────────────────────────────────────────

    def weight(
        self,
        pool: ParticlePool,
        context: DiscoveryContext,
    ) -> ParticlePool:
        """Evaluate and assign log-weights to all un-evaluated particles.

        Particles that already carry a ``log_weight`` and ``feedback`` are
        skipped (e.g. copied particles from the propagation step).  New
        particles are evaluated in parallel via Ray: each worker compiles the
        simulator code, runs parameter optimisation (SBI), computes evaluation
        metrics on validation data, generates LLM diagnostic feedback, and
        assigns a scalar log-weight derived from the task's optimisation metric.

        Args:
            pool:    Particle pool, possibly containing un-evaluated particles.
            context: Discovery context with task, data, and output directory.

        Returns:
            Updated pool where every particle has ``log_weight`` and
            ``feedback`` set.
        """
        particles = pool.get_particles()
        completed_particles = []
        new_particles = []

        for particle in particles:
            if particle.evaluated:
                logger.debug(
                    f"Skipping already-evaluated particle uuid={particle.uuid}"
                )
                completed_particles.append(particle)
            else:
                logger.debug(f"Queuing particle uuid={particle.uuid} for evaluation")
                new_particles.append(particle)

        # Fast path: all particles were already evaluated (e.g. pure copy step).
        if not new_particles:
            completed_pool = ParticlePool(particles=completed_particles)
            assert len(completed_pool) == len(pool)
            return completed_pool

        llm_kwargs = dict(
            dspy.settings.lm.kwargs
        )  # shallow copy — avoid mutating global LM state
        llm_model = dspy.settings.lm.model

        # GPT-5 uses 'max_completion_tokens'; normalise to 'max_tokens'.
        if "max_completion_tokens" in llm_kwargs:
            llm_kwargs["max_tokens"] = llm_kwargs["max_completion_tokens"]
            del llm_kwargs["max_completion_tokens"]

        # Check if the full histories are saved for each particle.
        # By default, do not save.
        save_llm_history = getattr(self.config, "save_llm_history", False)

        logger.debug(
            "CUDA_VISIBLE_DEVICES before weighting: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        # Assign a unique output directory to each new particle.
        context_list = []
        for idx in range(len(new_particles)):
            c_i = copy.deepcopy(context)
            c_i.save_dir = self._get_save_dir(
                particle_index=self.__particle_count + idx
            )
            context_list.append(c_i)

        futures = [
            self.weight_particle.options(retry_exceptions=True, max_retries=1).remote(
                self.__class__,
                particle_i,
                context_list[i],
                copy.deepcopy(self.evaluator),
                copy.deepcopy(self.optimizer),
                copy.deepcopy(self.feedback),
                copy.deepcopy(llm_model),
                copy.deepcopy(llm_kwargs),
                self.__particle_count + i,
                save_llm_history,
            )
            for i, particle_i in enumerate(new_particles)
        ]

        results = ray.get(futures)
        new_particles, llm_tokens = zip(*results, strict=False)
        new_particles = list(new_particles)
        llm_tokens = list(llm_tokens)

        assert isinstance(new_particles, list)
        assert all(isinstance(p, Particle) for p in new_particles)

        completed_particles = completed_particles + new_particles
        completed_pool = ParticlePool(particles=completed_particles)

        assert len(completed_pool) == len(pool)

        self.__particle_count += len(new_particles)

        for key in self.llm_tokens:
            self.llm_tokens[key] += sum(t.get(key, 0) for t in llm_tokens)

        return completed_pool

    @staticmethod
    @ray.remote(
        num_cpus=int(os.getenv("REQ_NUM_CPUS_PER_RAY_THREAD")),
        num_gpus=int(os.getenv("REQ_NUM_GPUS_PER_RAY_THREAD")),
    )
    def weight_particle(
        class_type,
        particle: Particle,
        context: DiscoveryContext,
        evaluator: EvaluatorBase,
        optimizer: OptimizerBase,
        feedback_model: FeedbackModule,
        model: str,
        llm_kwargs: dict,
        particle_index: int,
        save_llm_history: bool = False,
    ) -> tuple[Particle, dict]:
        """Evaluate a single particle and assign its log-weight (Ray remote).

        Full evaluation pipeline for one particle:

            1. Compile the simulator code string to an executable Python module.
            2. Fit parameters theta to training data via SBI:
                   theta* = argmax_theta  p(x_train | f, theta)
            3. Evaluate the fitted simulator on validation data to compute fitness
               metrics (e.g. MSE, average log-likelihood, ...).
            4. Call the LLM feedback module to produce a diagnostic string
               that guides the next propagation step.
            5. Derive the scalar log-weight from the task's optimisation
               metric:
                   log w = -metric   (minimisation task)
                   log w = +metric   (maximisation task)
               Failed evaluations receive log w = -inf (zero probability).

        All intermediate results are persisted to ``context.save_dir``.

        This function runs in a Ray worker and creates its own LLM client for
        the feedback step.

        Args:
            class_type:      Reference to ``SMCEngine``, used to call the
                             static helper methods ``_weight_evaluation`` and
                             ``_weight_feedback``.
            particle:        Particle to evaluate.
            context:         Discovery context providing data and save path.
            evaluator:       Evaluator module for fitness computation.
            optimizer:       SBI-based parameter optimisation module.
            feedback_model:  LLM-based diagnostic feedback module.
            model:           LiteLLM model identifier string.
            llm_kwargs:      Additional keyword arguments forwarded to the LLM.
            particle_index:  Global integer index for this particle, used to
                             populate ``particle.particle_index`` without
                             parsing the directory name.
            save_llm_history:If True, persist the worker's propagation and
                             feedback LLM histories to ``context.save_dir``.

        Returns:
            particle:   Evaluated particle with ``log_weight``, ``feedback``,
                        ``metrics``, ``times``, and ``evaluated`` populated.
            llm_tokens: Token usage dict for this call.
        """
        # Configure the logger for this worker process so that log records are
        # forwarded to the driver via Ray's stdout piping.
        logger = setup_worker_logging()

        # Log GPU availability in the worker for debugging.
        available_gpu_ids_ray = [int(i) for i in ray.get_gpu_ids()]
        try:
            logger.debug(f"Ray GPU ids: {available_gpu_ids_ray}")
            logger.debug(
                "CUDA_VISIBLE_DEVICES in worker: "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.debug(
                "Device name: "
                f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
        except Exception as e:
            logger.debug(f"GPU info unavailable: {e}")

        # Each Ray worker initialises its own LLM client with a clean history.
        lm_temporary = dspy.LM(model=model, cache=False, **llm_kwargs)
        if len(lm_temporary.history) != 0:
            lm_temporary.history = []

        feedback_model.set_lm(lm_temporary)

        # Persist the raw simulator code to disk before attempting compilation.
        code_type = detect_code_type(particle.implementation)
        file_ext = utils.code_type_to_file_ext(code_type)
        with open(os.path.join(context.save_dir, f"simulator.{file_ext}"), "w") as f:
            f.write(particle.implementation)

        if particle.scm_definition is not None:
            with open(os.path.join(context.save_dir, "scm.txt"), "w") as f:
                f.write(particle.scm_definition)

        # Compile the code string to an executable module (returns None on failure).
        simulator, error_dict = utils.convert_model_string(particle.implementation)

        dt_optimization, dt_evaluation, eval_metrics = class_type._weight_evaluation(
            simulator=simulator,
            error_dict=error_dict,
            context=context,
            evaluator=evaluator,
            optimizer=optimizer,
        )

        dt_feedback, feedback = class_type._weight_feedback(
            implementation=particle.implementation,
            context=context,
            feedback_model=feedback_model,
            eval_metrics=eval_metrics,
        )

        # Derive scalar log-weight from the designated optimisation metric.
        metric_to_optimize = context.task.config.metric_to_optimize
        optimization_direction = context.task.config.optimization_direction

        try:
            selected_metric = eval_metrics["eval_metrics"][metric_to_optimize]
            is_valid = isinstance(selected_metric, numbers.Number) and not math.isnan(
                selected_metric
            )
            if is_valid:
                log_weight = (
                    -selected_metric
                    if optimization_direction == "min"
                    else selected_metric
                )
            else:
                log_weight = -np.inf
        except Exception as e:
            logger.warning(e)
            log_weight = -np.inf

        logger.debug(
            f"Assigned log-weight {log_weight} to particle uuid={particle.uuid}"
        )

        particle.feedback = feedback
        particle.log_weight = log_weight

        if particle.times is None:
            particle.times = {}
        particle.times["dt_optimization_sec"] = dt_optimization
        particle.times["dt_evaluation_sec"] = dt_evaluation
        particle.times["dt_feedback_sec"] = dt_feedback

        particle.metrics = eval_metrics
        particle.results_folder = context.save_dir.split("/")[-1]
        particle.particle_index = particle_index
        particle.evaluated = True

        llm_tokens = _sum_token_usage(lm_temporary.history)

        # Write the saved histories from the propagation and the weighting step to the
        # results folder.
        class_type._save_llm_history(
            particle=particle,
            save_llm_history=save_llm_history,
            save_dir=context.save_dir,
            history=lm_temporary.history,
        )

        return particle, llm_tokens

    @staticmethod
    def _weight_evaluation(
        simulator: Callable,
        error_dict: dict,
        context: DiscoveryContext,
        evaluator: EvaluatorBase,
        optimizer: OptimizerBase,
    ) -> tuple[float, float, dict]:
        """Run parameter optimisation and evaluate the fitted simulator.

        Args:
            simulator:  Callable simulator compiled from the code string, or
                        ``None`` if compilation failed.
            error_dict: Compilation error information (empty dict if no errors).
            context:    Discovery context with training and validation data.
            evaluator:  Evaluator module for computing fitness metrics.
            optimizer:  SBI-based parameter optimisation module.

        Returns:
            dt_optimization: Wall-clock time for the optimisation step (s).
            dt_evaluation:   Wall-clock time for the evaluation step (s).
            eval_metrics:    Dictionary of computed evaluation metrics.
        """
        t0 = time.time()
        optim_metrics, parameter_est = optimizer.optimize(
            error_dict=error_dict,
            simulator=simulator,
            training_data_raw=context.train_data,
            save_dir=context.save_dir,
            valid_data_raw=context.valid_data,
        )
        dt_optimization = time.time() - t0

        # If the task uses learnable summary statistics, copy the embedding
        # network trained by the optimizer to the evaluator so that both
        # modules use the same fitted feature extractor.
        if (
            "learnable_summary_statistics" in optimizer.task.config
            and optimizer.task.config.learnable_summary_statistics
        ):
            logger.debug("Copying trained embedding net from optimizer to evaluator.")
            optim_embedding_net = copy.deepcopy(optimizer.task.embedding_net)
            assert isinstance(optim_embedding_net, BaseEmbeddingHandler), (
                "Optimizer embedding net is not a BaseEmbeddingHandler instance."
            )
            del evaluator.task.embedding_net
            evaluator.task.embedding_net = optim_embedding_net

        t1 = time.time()
        eval_metrics = evaluator.evaluate(
            parameter_est,
            optim_metrics,
            simulator,
            context.valid_data,
            context.save_dir,
        )
        dt_evaluation = time.time() - t1

        return dt_optimization, dt_evaluation, eval_metrics

    @staticmethod
    def _weight_feedback(
        implementation: str,
        context: DiscoveryContext,
        feedback_model: FeedbackModule,
        eval_metrics: dict,
    ) -> tuple[float, str]:
        """Generate LLM diagnostic feedback for a particle.

        Sends the simulator code and evaluation metrics to the LLM feedback
        module, which returns a natural-language diagnosis to guide the next
        propagation step.

        Args:
            implementation: Simulator source code string.
            context:        Discovery context with save path.
            feedback_model: LLM-based feedback module.
            eval_metrics:   Evaluation metrics dict from ``_weight_evaluation``.

        Returns:
            dt_feedback: Wall-clock time for the feedback step (s).
            feedback:    Natural-language diagnostic string from the LLM.
        """
        t0 = time.time()
        feedback = feedback_model.retrieve(
            eval_metrics, implementation, context.save_dir
        )
        dt_feedback = time.time() - t0
        return dt_feedback, feedback


#########################################################
# SMC Orchestrator
#########################################################


class SMCOrchestrator:
    """Top-level orchestrator that drives the SMC discovery loop.

    Wraps ``SMCEngine`` and manages the experiment lifecycle: loading data,
    initialising the particle pool, iterating SMC steps, logging results, and
    optionally applying early-stopping criteria.

    Args:
        config:     Full Hydra config object (``config_full``).
        task:       Task object providing data and evaluation.
        output_dir: Per-run output directory created by Hydra.
    """

    def __init__(self, config: DictConfig, task: BaseTask, output_dir: str):
        self.config_full = config
        self.config = config.method

        self.task = task
        self.output_dir = output_dir

        # Walk up the directory tree to locate the experiment-level directory
        # (whose basename equals config.name).  This directory collects results
        # across multiple seeds or sweep runs.
        output_super_dir = os.path.dirname(self.output_dir)
        while os.path.basename(output_super_dir) != self.config_full.name:
            output_super_dir = os.path.dirname(output_super_dir)
        self.output_super_dir = output_super_dir

        self.smc_engine = SMCEngine(self.config, task, output_dir)

        # Per-run early-stopping state; reset at the start of run_discovery.
        self._itr_no_improvement = 0
        self._prev_best_log_weight = None

    def _discovery_iteration(
        self,
        pool: ParticlePool | None,
        context: DiscoveryContext,
        best_particle: Particle | None,
    ) -> tuple[ParticlePool, Particle | None]:
        """Run a single discovery iteration.

        At iteration 0, the pool is initialised from the task's base simulator.
        Subsequent iterations execute a full SMC step (resample → propagate →
        weight).

        Args:
            pool:          Current particle pool (``None`` before init).
            context:       Discovery context with task and data.
            best_particle: Best particle seen so far (currently unused; kept
                           for API compatibility with a future best-particle
                           tracker).

        Returns:
            pool:          Updated particle pool.
            best_particle: Unchanged (tracking currently disabled).
        """
        if self.smc_engine.itr == 0:
            pool, ess, llm_tokens = self.smc_engine.init_particle_pool(context)
        else:
            pool, ess, llm_tokens = self.smc_engine.smc_step(
                pool=pool,
                context=context,
            )

        self._log_performance(
            itr=self.smc_engine.itr,
            pool=pool,
            ess=ess,
            llm_tokens=llm_tokens,
            output_dir=self.output_super_dir,
            config=self.config_full,
        )

        best_particle = self._update_best_particle(
            pool=pool, current_best_particle=best_particle, context=context
        )

        # Plot the composition of the pool
        plot_pool(
            pool_csv_path=os.path.join(self.output_super_dir, "pool_composition.csv"),
            summary_csv_path=os.path.join(self.output_super_dir, "summary.csv"),
            run_id=self.config_full.run_id,
            cmap_name="viridis",
            save_dir=self.output_dir,
        )

        # Plot the trajectory
        plot_trajectories(
            pool_csv_path=os.path.join(self.output_super_dir, "pool_composition.csv"),
            summary_csv_path=os.path.join(self.output_super_dir, "summary.csv"),
            run_id=self.config_full.run_id,
            save_dir=self.output_dir,
            image_name="trajectories.png",
            metric_name="log_weight",
            yscale="linear",
        )

        self.smc_engine.itr += 1

        return pool, best_particle

    def run_discovery(self):
        """Main entry point: run the full SMC discovery loop.

        Iterates for ``config.max_iter + 1`` steps (step 0 is pool
        initialisation from the base simulator).

        Returns:
            None
        """
        train_data, valid_data = self.task.get_data()
        best_particle = None
        pool = None

        context = DiscoveryContext(
            task=self.task,
            train_data=train_data,
            valid_data=valid_data,
        )

        self._itr_no_improvement = 0
        self._prev_best_log_weight = None

        for _itr in range(self.config.max_iter + 1):
            logger.info("-" * 50)
            logger.info(f"Iteration {self.smc_engine.itr} / {self.config.max_iter}")
            logger.info("-" * 50)

            pool, best_particle = self._discovery_iteration(
                pool=pool,
                context=context,
                best_particle=best_particle,
            )

            if self._check_termination(best_particle, itr=self.smc_engine.itr - 1):
                break

        logger.info("Discovery completed.")

        try:
            SMCOrchestrator.posthoc_parameter_estimation(
                folder=os.path.join(self.output_dir, "best_particle")
            )
        except Exception as e:
            logger.warning(
                f"Post-hoc parameter estimation for best particle failed: {e}"
            )

        best_particle_dir = os.path.join(self.output_dir, "best_particle")
        try:
            SMCOrchestrator.posthoc_parameter_estimation(
                folder=os.path.join(self.output_dir, "iter-0_p0"),
                config_path=os.path.join(best_particle_dir, ".hydra", "config.yaml"),
                train_data_path=os.path.join(best_particle_dir, "training_data.pt"),
                valid_data_path=os.path.join(best_particle_dir, "validation_data.pt"),
            )
        except Exception as e:
            logger.warning(f"Post-hoc parameter estimation for baseline failed: {e}")

        return None

    def _log_performance(
        self,
        itr: int,
        pool: ParticlePool,
        ess: float,
        llm_tokens: dict,
        output_dir: str,
        config: DictConfig,
    ) -> None:
        """Log per-particle metrics and write summary CSV rows.

        For each particle in the pool, logs the optimisation metric and writes
        a row to the run-level summary CSV via ``utils.save_summary``.  After
        processing all particles, saves the pool composition (including
        duplicates) via ``utils.save_pool`` so that the genealogy can be
        reconstructed from the CSV.

        Args:
            itr:        Current iteration index.
            pool:       Evaluated particle pool.
            ess:        Effective Sample Size for this iteration (a float in
                        [1, N] for SMC steps; ``None`` at iteration 0, where no
                        resampling occurs).
            llm_tokens: Cumulative token usage dict.
            output_dir: Experiment-level output directory for CSV files.
            config:     Full Hydra config.
        """
        particles = pool.get_particles()

        for particle in particles:
            particle_index = particle.particle_index
            metric_to_optimize = getattr(self.task.config, "metric_to_optimize", None)

            if (
                "eval_metrics" in particle.metrics
                and metric_to_optimize is not None
                and metric_to_optimize in particle.metrics["eval_metrics"]
            ):
                logger.info(
                    f"Particle {particle_index}: "
                    f"{metric_to_optimize} = "
                    f"{particle.metrics['eval_metrics'][metric_to_optimize]:.4g}"
                )
            else:
                logger.info(f"Particle {particle_index}: no metrics available")

            utils.save_summary(
                run_id=config.run_id,
                particle_id=particle.uuid,
                parent_id=particle.ancestors[-1].uuid
                if len(particle.ancestors) > 0
                else None,
                itr=itr,
                metrics=particle.metrics,
                save_dir=output_dir,
                config=config,
                total_tokens=llm_tokens["total_tokens"],
                prompt_tokens=llm_tokens["prompt_tokens"],
                completion_tokens=llm_tokens["completion_tokens"],
                log_weight=particle.log_weight,
                particle_index=particle.particle_index,
                ess=ess,
                times=particle.times,
            )

        # The summary CSV stores unique particles only; save_pool records which
        # particles (and how many copies) are present after each iteration.
        utils.save_pool(pool=pool, itr=itr, save_dir=output_dir, run_id=config.run_id)

    def _update_best_particle(
        self,
        current_best_particle: Particle | None,
        pool: ParticlePool,
        context: DiscoveryContext,
    ) -> Particle | None:
        """Update and persist the best particle found so far.

        Compares the best valid particle in ``pool`` against
        ``current_best_particle`` (the global best from prior iterations).
        A particle is *valid* when it has been evaluated, carries a finite
        log-weight, and that log-weight is not NaN.  If a new best is found
        the ``best_particle/`` output directory is atomically replaced with
        the winner's artifacts.

        The following are written to ``<output_dir>/best_particle/``:

        1. ``training_data.pt``    — ``context.train_data``
        2. ``validation_data.pt``  — ``context.valid_data``
        3. All files from the particle's per-iteration results folder
           (located via ``particle.results_folder``).
        4. ``best_particle_info.csv`` — index, log-weight, folder, and any
           ``eval_metrics`` stored on the particle.
        5. ``.hydra/`` — Hydra config snapshot (copied only when present).

        Args:
            current_best_particle: Best particle found in previous iterations,
                or ``None`` at the start of the run.
            pool:    The current ``ParticlePool`` (after the weight step).
            context: ``DiscoveryContext`` supplying ``train_data`` and
                ``valid_data``.

        Returns:
            The new global best particle, or ``current_best_particle`` when
            no improvement is found.
        """

        def is_valid(p):
            return (
                p is not None
                and p.evaluated
                and p.log_weight is not None
                and not np.isnan(p.log_weight)
                and not np.isinf(p.log_weight)
            )

        valid_from_pool = [p for p in pool.get_particles() if is_valid(p)]
        best_from_pool = max(valid_from_pool, key=lambda p: p.log_weight, default=None)

        candidates = [p for p in [current_best_particle, best_from_pool] if is_valid(p)]
        if not candidates:
            return current_best_particle

        best_particle = max(candidates, key=lambda p: p.log_weight)

        if (
            current_best_particle is not None
            and best_particle.log_weight <= current_best_particle.log_weight
        ):
            return current_best_particle

        logger.info(
            f"New best particle: index={best_particle.particle_index}, "
            f"log_weight={best_particle.log_weight:.4g}"
        )

        best_particle_save_dir = os.path.join(self.output_dir, "best_particle")
        if os.path.exists(best_particle_save_dir):
            shutil.rmtree(best_particle_save_dir)
        os.makedirs(best_particle_save_dir, exist_ok=True)

        torch.save(
            context.train_data, os.path.join(best_particle_save_dir, "training_data.pt")
        )
        torch.save(
            context.valid_data,
            os.path.join(best_particle_save_dir, "validation_data.pt"),
        )

        if best_particle.results_folder is None:
            logger.warning(
                "Best particle has no results_folder; cannot copy artifacts."
            )
            return best_particle

        src = os.path.join(self.output_dir, best_particle.results_folder)
        if not os.path.exists(src):
            logger.warning(f"Results folder not found: {src}; cannot copy artifacts.")
            return best_particle

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(best_particle_save_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        best_particle_info = {
            "particle_index": best_particle.particle_index,
            "log_weight": best_particle.log_weight,
            "folder": best_particle.results_folder,
            "particle_id": best_particle.uuid,
        }
        eval_metrics = best_particle.metrics.get("eval_metrics", {})
        if eval_metrics:
            best_particle_info.update(eval_metrics)
        else:
            logger.warning("Best particle has no eval_metrics.")

        pd.DataFrame([best_particle_info]).to_csv(
            os.path.join(best_particle_save_dir, "best_particle_info.csv"),
            index=False,
        )

        hydra_src = os.path.join(self.output_dir, ".hydra")
        if os.path.exists(hydra_src):
            shutil.copytree(
                hydra_src,
                os.path.join(best_particle_save_dir, ".hydra"),
                dirs_exist_ok=True,
            )

        return best_particle

    @staticmethod
    def posthoc_parameter_estimation(
        folder: str,
        config_path: str | None = None,
        train_data_path: str | None = None,
        valid_data_path: str | None = None,
    ) -> None:
        """Compute parameter estimates for a particle after discovery.

        Loads the saved simulator code and data from ``folder``,
        forces parameter estimation on, runs the optimizer and evaluator, and
        updates ``best_particle_info.csv`` with any newly computed metrics (only
        if that file already exists in the folder).

        Can be called automatically at the end of ``run_discovery()`` and also
        standalone from an external script (e.g. after a cluster timeout):

            SMCOrchestrator.posthoc_parameter_estimation("/path/to/best_particle")

        Args:
            folder: Path to the particle directory.  Must contain
                a ``simulator.py`` (or ``simulator.R``).
            config_path: Optional explicit path to the Hydra ``config.yaml`` to
                use.  When omitted, the config is loaded from
                ``{folder}/.hydra/config.yaml`` (the default for
                the ``best_particle/`` directory).  Pass this for folders that do
                not carry their own Hydra snapshot (e.g. ``iter-0_p0/``).
            train_data_path: Optional explicit path to ``training_data.pt``.
                When omitted, ``{folder}/training_data.pt`` is
                used.  Raises ``FileNotFoundError`` if the resolved path does
                not exist.
            valid_data_path: Optional explicit path to ``validation_data.pt``.
                When omitted, ``{folder}/validation_data.pt`` is
                used.  Raises ``FileNotFoundError`` if the resolved path does
                not exist.
        """

        if os.path.exists(os.path.join(folder, "parameter_estimates.pt")):
            logger.warning(
                f"Post-hoc parameter estimation skipped: parameter estimates already "
                f"exist in {folder}"
            )
            return

        # 1. Register OmegaConf resolvers idempotently.
        register_omegaconf_resolvers()

        # 2. Resolve config path: explicit argument takes priority, then fall
        #    back to the .hydra snapshot inside folder.
        if config_path is None:
            config_path = os.path.join(folder, ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            logger.warning(
                f"Post-hoc parameter estimation skipped: config not found at "
                f"{config_path}"
            )
            return
        config = OmegaConf.load(config_path)

        # 3. Set env vars required by the Ray remote decorator (evaluated at
        #    module-import time, so must be present before further imports).
        os.environ.setdefault(
            "REQ_NUM_CPUS_PER_RAY_THREAD",
            str(config.method.req_num_cpus_per_ray_thread),
        )
        os.environ.setdefault(
            "REQ_NUM_GPUS_PER_RAY_THREAD",
            str(config.method.req_num_gpus_per_ray_thread),
        )

        # 4. Force parameter estimation on.
        with open_dict(config):
            config.method.optimizer.estimate_parameters = True
            config.method.evaluator.estimate_parameters = True
            config.method.optimizer.save_parameter_estimates = True
            config.task.plotting = False

        # 5. Load task (deferred import to ensure env vars are set first).
        from modelsmc.tasks import create_task

        task = create_task(config.task)

        # This is needed to internally initialize the task object.
        _, _ = task.get_data()

        # 6. Load simulator code (tries .py, .R).
        simulator_path = utils.find_simulator_file(folder)
        if simulator_path is None:
            logger.warning(
                f"Post-hoc parameter estimation skipped: no simulator file found in "
                f"{folder}"
            )
            return

        with open(simulator_path, "r") as f:
            implementation = f.read()

        simulator, error_dict = utils.convert_model_string(implementation)

        # 7. Resolve and load training and validation data.  Explicit path
        #    arguments take priority; otherwise default to files in the folder.
        if train_data_path is None:
            train_data_path = os.path.join(folder, "training_data.pt")
        if valid_data_path is None:
            valid_data_path = os.path.join(folder, "validation_data.pt")
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data not found at {train_data_path}")
        if not os.path.exists(valid_data_path):
            raise FileNotFoundError(f"Validation data not found at {valid_data_path}")
        train_data = torch.load(train_data_path, weights_only=False)
        valid_data = torch.load(valid_data_path, weights_only=False)

        # 8. Instantiate optimizer and evaluator with parameter estimation forced on.
        optimizer = create_optimizer(
            config.method.optimizer, task, config.method.verbose
        )
        evaluator = EvaluatorBase(config.method.evaluator, task, config.method.verbose)

        # 9. Build DiscoveryContext pointing back to folder so
        #    that all artifacts (plots, parameter_estimates.pt) are written there.
        context = DiscoveryContext(
            task=task,
            train_data=train_data,
            valid_data=valid_data,
            save_dir=folder,
        )

        # 10. Run the optimizer → evaluator pipeline (no Ray, no feedback LLM).
        logger.info(f"Running post-hoc parameter estimation in {folder}.")
        _, _, eval_metrics = SMCEngine._weight_evaluation(
            simulator=simulator,
            error_dict=error_dict,
            context=context,
            evaluator=evaluator,
            optimizer=optimizer,
        )

        # 11. Write posthoc metrics to best_particle_info.csv.
        #     If that file already exists, rename it to original_best_particle_info.csv
        #     first (preserving it unchanged), then write a fresh file with only
        #     the posthoc eval metrics.
        info_path = os.path.join(folder, "best_particle_info.csv")
        if os.path.exists(info_path):
            original_info_path = os.path.join(folder, "original_best_particle_info.csv")
            os.rename(info_path, original_info_path)
        posthoc_info = eval_metrics.get("eval_metrics", {})
        pd.DataFrame([posthoc_info]).to_csv(info_path, index=False)

        logger.info("Post-hoc parameter estimation completed.")

    def _check_termination(self, best_particle: Particle, itr: int) -> bool:
        """Return True if an early-stopping criterion is met.

        Skips all checks for ``itr == 0`` (pool initialisation step) and
        instead records the baseline log-weight.

        Checks (in order):
          1. **Stagnation** — best ``log_weight`` has not strictly improved for
             ``config.max_num_itr_no_improvement`` consecutive iterations.
             Skipped when the config field is absent.
          2. **Threshold** — best ``log_weight`` has reached or crossed the
             value implied by ``task.config.termination_val``.  Skipped when
             ``termination_val`` is ``None``.

        Note: ``log_weight`` encodes optimisation direction — it equals
        ``-metric`` for minimisation tasks and ``+metric`` for maximisation
        tasks — so **higher is always better** and comparisons are
        direction-agnostic.

        Args:
            best_particle: Global best particle after the current iteration,
                or ``None`` if no valid particle exists yet.
            itr:           Index of the iteration just completed (0 = init).

        Returns:
            ``True`` if the loop should stop, ``False`` otherwise.
        """

        if itr == 0:
            if best_particle is not None:
                self._prev_best_log_weight = best_particle.log_weight
            return False

        max_no_improve = getattr(self.config, "max_num_itr_no_improvement", None)
        termination_val = getattr(self.task.config, "termination_val", None)
        cur_log_weight = best_particle.log_weight if best_particle is not None else None

        if cur_log_weight is None:
            return False

        # --- Stagnation check ---
        if max_no_improve is not None:
            improved = (
                self._prev_best_log_weight is None
                or cur_log_weight > self._prev_best_log_weight
            )
            self._itr_no_improvement = 0 if improved else self._itr_no_improvement + 1
            self._prev_best_log_weight = cur_log_weight

            if self._itr_no_improvement >= max_no_improve:
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self._itr_no_improvement} consecutive iterations."
                )
                return True

        # --- Threshold check ---
        if termination_val is not None:
            direction = getattr(self.task.config, "optimization_direction", "min")
            threshold_log_weight = (
                -termination_val if direction == "min" else termination_val
            )
            if cur_log_weight >= threshold_log_weight:
                logger.info(
                    f"Early stopping: termination threshold reached "
                    f"(log_weight={cur_log_weight:.4g} >= {threshold_log_weight:.4g})."
                )
                return True

        return False


def smc_method(config: DictConfig, task: BaseTask, output_dir: str) -> None:
    """Entry point: instantiate ``SMCOrchestrator`` and run discovery.

    Args:
        config:     Full Hydra config object.
        task:       Task object providing data, evaluation, and description.
        output_dir: Per-run output directory created by Hydra.

    Returns:
        None
    """
    orchestrator = SMCOrchestrator(config, task, output_dir)
    return orchestrator.run_discovery()
