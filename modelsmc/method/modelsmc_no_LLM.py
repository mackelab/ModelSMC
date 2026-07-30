import logging
import os
import time
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from modelsmc.method.modelsmc import SMCEngine, SMCOrchestrator
from modelsmc.method.modules.codingsimulator import CodingSimulatorModule
from modelsmc.method.modules.dataclasses import (
    Ancestor,
    DiscoveryContext,
    Particle,
    ParticlePool,
)
from modelsmc.method.modules.feedback import FeedbackModule
from modelsmc.tasks.base_task import BaseTask
from modelsmc.utils.utils import save_GMM_idx_distribution, setup_worker_logging

logger = logging.getLogger("ModelSMC")


#########################################################
# SMC Discovery Engine — Minimal variant
#########################################################


class SMCEngineMinimal(SMCEngine):
    """SMC engine specialised for the minimal (GMM) tasks.

    Overrides ``SMCEngine`` to replace the LLM-based proposal with a simple
    random-index kernel that selects one of the task's pre-defined GMM
    configurations uniformly at random.  This makes the method fully
    deterministic in its model structure (no code generation) and is
    primarily used for validating the SMC scaffolding without
    using an LLM.

    The pool is initialised by evaluating every available GMM configuration
    once and then replicating the evaluated particles to reach the configured
    pool size (``config.particle_pool_size``).  The number of models must
    divide the pool size exactly.

    Args:
        config:     Method-level Hydra config (``config.method``).
        task:       Task object; must expose ``num_models`` and
                    ``get_skeleton_implementation()``.
        output_dir: Root directory for per-particle result files.
    """

    def __init__(self, config: DictConfig, task: BaseTask, output_dir: str):
        super().__init__(config, task, output_dir)
        self.task = task

    def init_particle_pool(self, context):
        """Evaluate every GMM configuration and initialise the particle pool.

        Creates one particle per available GMM configuration, evaluates all of
        them (parameter optimisation + fitness scoring), and then replicates
        the full evaluated set uniformly to reach ``config.particle_pool_size``.
        The pool size must be an exact multiple of the number of GMM models.

        Args:
            context: Discovery context carrying task, data, and save path.

        Returns:
            pool:       Initialised ``ParticlePool`` of size
                        ``config.particle_pool_size``.
            None:       Placeholder for ESS (not computed at init).
            llm_tokens: Cumulative token usage dict (empty here — no LLM calls
                        are made during pool initialisation for this variant).
        """

        # Determine how many times each model must appear to fill the pool.
        pool_size = self.config.particle_pool_size
        num_models = self.task.num_models

        multiplicity = int(pool_size / num_models)
        assert (pool_size % num_models) == 0, (
            f"particle_pool_size ({pool_size}) must be divisible by "
            f"num_models ({num_models})."
        )

        logger.debug(
            f"Pool init: {num_models} GMM configs × {multiplicity} copies = "
            f"{pool_size} particles"
        )

        # Build one particle per GMM configuration index.
        particles = []
        for idx in range(num_models):
            implementation_idx = self.task.get_skeleton_implementation().format(
                GMM_CONFIG_IDX=idx
            )
            particle_idx = Particle(implementation=implementation_idx)
            particles.append(particle_idx)

        pool = ParticlePool(particles=particles)

        # Evaluate all seed particles (parameter optimisation + fitness scoring).
        pool = self.weight(pool=pool, context=context)

        # Replicate each evaluated particle to reach the target pool size.
        particles = pool.get_particles()
        particles_duplicated = []
        for p in particles:
            for _i in range(multiplicity):
                particles_duplicated.append(deepcopy(p))

        pool = ParticlePool(particles_duplicated)
        assert len(pool) == pool_size

        logger.debug(
            f"Initialised particle pool with {len(pool)} particles "
            f"({num_models} unique GMM configs × {multiplicity} copies)"
        )

        # No llm usage
        llm_tokens = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        return pool, None, llm_tokens

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
        """Propagate a single particle according to the minimal mixture kernel.

        This is a Ray remote function; it runs in an isolated worker process.
        The kernel is a two-component mixture:

        * With probability ``alpha``  — identity move: return the particle
          unchanged (no new evaluation required downstream).
        * With probability ``1-alpha`` — random move: draw a GMM configuration
          index uniformly at random and create a fresh particle from that
          skeleton, recording the current particle as its ancestor.

        Unlike ``SMCEngine.sample_proposal``, this method never calls the LLM.
        The ``simcoder``, ``few_shot_examples``, and ``model`` / ``llm_kwargs``
        arguments are accepted only to satisfy the shared interface but are
        unused here.

        Args:
            particle:          Current particle to propagate.
            simcoder:          LLM coding module (unused in this variant).
            few_shot_examples: Number of ancestry examples for the LLM prompt
                               (unused in this variant).
            context:           Discovery context; ``context.task.num_models``
                               determines the number of available GMM configs.
            alpha:             Copy probability in [0, 1].  Higher values mean
                               fewer random moves per iteration.
            model:             LLM model identifier (unused in this variant).
            llm_kwargs:        Extra kwargs for the LLM client (unused).

        Returns:
            new_particle: Propagated particle (either the original or a new one
                          drawn from the skeleton library).
            llm_tokens:   Token usage dict (always zero for this variant).
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

        z = np.random.rand()
        assert 0.0 <= alpha <= 1.0

        if z <= alpha:
            # Identity move: return the particle unchanged.
            logger.debug(f"Identity move — keeping particle uuid={particle.uuid}")
            new_particle = particle

        else:
            # Random move: sample a GMM configuration index uniformly at
            # random and construct a fresh particle from that skeleton.
            idx = np.random.randint(0, context.task.num_models)
            logger.debug(
                f"Random move — particle uuid={particle.uuid} → GMM config index {idx}"
            )

            implementation_idx = context.task.get_skeleton_implementation().format(
                GMM_CONFIG_IDX=idx
            )

            # Record the current particle as an ancestor for traceability.
            parent = Ancestor(
                implementation=particle.implementation,
                feedback=particle.feedback,
                scm_definition=particle.scm_definition,
                uuid=particle.uuid,
                log_weight=particle.log_weight,
            )

            new_particle = Particle(
                implementation=implementation_idx,
                scm_definition="",
                ancestors=[parent],
            )

            logger.debug(f"Created new particle uuid={new_particle.uuid}")

        # No llm usage
        llm_tokens = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        return new_particle, llm_tokens

    @staticmethod
    def _weight_feedback(
        implementation: str,
        context: DiscoveryContext,
        feedback_model: FeedbackModule,
        eval_metrics: dict,
    ) -> tuple[float, str]:
        """Skip LLM-based feedback generation for the minimal variant.

        In the full ``SMCEngine``, this method calls an LLM to produce
        natural-language diagnostic feedback for the particle.  In this
        minimal variant no LLM is involved, so empty feedback
        and a zero elapsed time is returned.

        Args:
            implementation: Simulator source code string (unused).
            context:        Discovery context (unused).
            feedback_model: LLM feedback module (unused).
            eval_metrics:   Dictionary of evaluation metrics (unused).

        Returns:
            dt_feedback: Elapsed time for feedback generation (0.0).
            feedback:    Empty string — no feedback generated.
        """
        return 0.0, ""


#########################################################
# SMC Orchestrator — Minimal variant
#########################################################


class SMCOrchestratorMinimal(SMCOrchestrator):
    """SMC orchestrator that substitutes ``SMCEngineMinimal`` for the default engine.

    Inherits the full experiment loop (logging, pool serialisation, genealogy
    tracking) from ``SMCOrchestrator`` but replaces the core SMC engine with
    ``SMCEngineMinimal``.  Because the parent ``__init__`` already starts a
    Ray cluster, this class shuts Ray down and restarts it through
    ``SMCEngineMinimal`` so that resource limits are applied consistently.

    Args:
        config:     Full Hydra config object.
        task:       Task object (must be a GMM-based minimal-example task).
        output_dir: Per-run output directory created by Hydra.
    """

    def __init__(self, config, task, output_dir):
        super().__init__(config, task, output_dir)

        logger.info("Initialising SMCOrchestratorMinimal...")
        logger.info("Replacing default SMCEngine with SMCEngineMinimal...")
        del self.smc_engine

        # The parent __init__ already started a Ray cluster.  Shut it down
        # before SMCEngineMinimal re-initialises Ray with the correct settings.
        if ray.is_initialized():
            logger.debug("Ray is already initialised — shutting down before reinit")
            ray.shutdown()
            time.sleep(5.0)

        self.smc_engine = SMCEngineMinimal(
            config=self.config,
            task=task,
            output_dir=output_dir,
        )

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

        pool, best_particle = super()._discovery_iteration(
            pool=pool, context=context, best_particle=best_particle
        )

        # Save how often each model index is present in the current pool of particles
        save_GMM_idx_distribution(
            pool=pool,
            save_dir=self.output_super_dir,
            run_id=self.config_full.run_id,
            itr=self.smc_engine.itr - 1,
            num_GMM_configs=self.task.num_models,
        )

        return pool, best_particle


def smc_method_minimal(config: DictConfig, task: BaseTask, output_dir: str) -> None:
    """Entry point: instantiate ``SMCOrchestratorMinimal`` and run discovery.

    Args:
        config:     Full Hydra config object.
        task:       Task object providing data, evaluation, and description.
        output_dir: Per-run output directory created by Hydra.

    Returns:
        None
    """
    orchestrator = SMCOrchestratorMinimal(config, task, output_dir)
    return orchestrator.run_discovery()
