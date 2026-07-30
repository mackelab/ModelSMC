import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import uuid

import dotenv
import hydra
import numpy as np
import pyfiglet
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from modelsmc.utils.utils import register_omegaconf_resolvers

logger = logging.getLogger("ModelSMC")

register_omegaconf_resolvers()

logo = pyfiglet.figlet_format("ModelSMC")


def main_wrapper():
    """CLI entry point for the ``modelsmc`` command (defined in ``pyproject.toml``).

    Re-invokes the current script via ``dotenvx run`` so that ``.env``-managed
    API keys and other secrets are injected into the environment before any
    project code runs.  All command-line arguments passed to ``modelsmc`` are
    forwarded unchanged to the subprocess.

    This indirection is necessary because ``dotenvx`` must wrap the process
    that actually uses the environment variables; calling ``dotenv.load_dotenv``
    after the fact is insufficient when Ray workers or subprocesses inherit
    the environment at spawn time.

    Returns:
        None.  Raises ``subprocess.CalledProcessError`` if the child process
        exits with a non-zero status.
    """
    cmd = ["dotenvx", "run", "--", "python", "-m", "modelsmc.main"] + sys.argv[1:]
    subprocess.run(cmd, check=True)


def main():
    """Print the ASCII logo and hand off to the Hydra-managed entry point.

    Exists as a thin wrapper so that the ``modelsmc`` CLI command (which calls
    this function) sees the logo before Hydra prints its own configuration
    output.  The 1.5-second sleep gives the user time to read the banner
    before log output starts scrolling.

    Returns:
        None
    """
    print(logo)
    time.sleep(1.5)
    _main()


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def _main(config: DictConfig):
    """Hydra-managed experiment entry point.

    Sets up the run and dispatches to the configured discovery method, in order:

        1. Configure logging first, so that every subsequent message is
           emitted through the project handler.
        2. Set the Ray resource env-vars (must precede any import of project
           code that reads them at module level).
        3. Resolve the per-run and experiment-level output directories.
        4. Inject a fresh ``run_id``, resolve a random seed if requested, and
           persist the updated config.
        5. Toggle ``estimate_parameters`` (optimizer + evaluator) based on the
           task metric, controlling whether SBI parameter inference runs.
        6. Seed ``random`` / ``numpy`` / ``torch``.
        7. Initialise the LLM (DSPy ``LM``, or ``DummyLLM`` for offline tests).
        8. Build the task, run the discovery method, and optionally clean up
           the DSPy cache.

    Args:
        config: Merged Hydra ``DictConfig`` assembled from
                ``config/config.yaml`` and all active config groups
                (task, method, llm, launcher, sweeper, experiment).

    Returns:
        None
    """
    # Configure logging as early as possible so that every subsequent
    # INFO/WARNING message is emitted through the project handler.
    logging.basicConfig(level=logging.WARNING)
    # Set logging level for external libraries to WARNING to suppress most messages
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(
        logging.WARNING
    )  # Added this as LiteLLM uses both formats
    logging.getLogger("dspy").setLevel(logging.WARNING)
    # Configure the logger used for this project
    level = getattr(logging, config.logging.level)
    logger.setLevel(level)
    logger.propagate = False
    # Export the level so Ray worker processes can inherit it via setup_worker_logging()
    os.environ["MODELSMC_LOG_LEVEL"] = config.logging.level
    # Create a console handler
    ch = logging.StreamHandler()
    # Define a custom format for this logger
    formatter = logging.Formatter(
        fmt=(
            "[%(asctime)s][%(name)s][%(filename)s: "
            "%(lineno)d][%(levelname)s] - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Attach the formatter to the handler
    ch.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(ch)

    # Set the environment variables for Ray remote function resource allocation
    os.environ["REQ_NUM_CPUS_PER_RAY_THREAD"] = str(
        config.method.req_num_cpus_per_ray_thread
    )
    os.environ["REQ_NUM_GPUS_PER_RAY_THREAD"] = str(
        config.method.req_num_gpus_per_ray_thread
    )

    # Import here to ensure the environment variables are set before importing custom
    # code using the variables
    from modelsmc.tasks import create_task

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    # Go back to the folder named "config.name"
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != config.name:
        output_super_dir = os.path.dirname(output_super_dir)

    # Temporarily disable struct mode to add run_id
    with open_dict(config):
        config.run_id = str(uuid.uuid4())

    # Overwrite the seed specified in the config with a randomly generated seed
    # if desired.
    if config.seed == "random":
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        config.seed = seed
        logger.warning(f"Using a randomly selected seed. Set seed to {config.seed}")

    # Persist the updated config (now including run_id) to the Hydra output
    # directory so that individual experiment folders contain the run_id.
    hydra_config_path = os.path.join(output_dir, ".hydra", "config.yaml")
    if os.path.exists(hydra_config_path):
        OmegaConf.save(config, hydra_config_path)

    # Collect metrics for which no parameter estimate is required
    metrics_no_parameter_estimate = ["neg_avg_log_marginal_NLE", "neg_log_marginal_NLE"]

    # Temporarily disable struct mode to add flags
    with open_dict(config):
        if (
            config.task.metric_to_optimize in metrics_no_parameter_estimate
            and not config.task.compute_parameter_estimate
        ):
            logger.info(
                "No parameter estimates will be computed during the discovery run."
            )
            config.method.optimizer.estimate_parameters = False
            config.method.evaluator.estimate_parameters = False

        else:
            logger.info(
                "Parameter estimates will be computed during the discovery run."
            )
            config.method.optimizer.estimate_parameters = True
            config.method.evaluator.estimate_parameters = True

    # Dump the fully resolved config after all mutations (run_id, resolved seed,
    # parameter-estimation flags)
    logger.info(OmegaConf.to_yaml(config))

    # Log runtime diagnostics
    logger.debug(
        f"REQ_NUM_CPUS_PER_RAY_THREAD: {int(os.environ['REQ_NUM_CPUS_PER_RAY_THREAD'])}"
    )
    logger.debug(
        f"REQ_NUM_GPUS_PER_RAY_THREAD: {int(os.environ['REQ_NUM_GPUS_PER_RAY_THREAD'])}"
    )
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory  : {output_dir}")
    logger.info(f"Output super directory: {output_super_dir}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Torch devices: {torch.cuda.device_count()}")
    logger.info(f"Task: {config.task.name}")
    logger.info(f"Method: {config.method.name}")
    logger.info(
        "Optimizer: "
        f"{config.method.optimizer.name if 'optimizer' in config.method else 'None'}"
    )
    logger.info(f"LLM: {config.llm.model}")

    # seed everything
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Initialize the LLM.
    # Deferred import: dspy must be imported after the Ray env-vars are set
    # above to avoid multiprocessing issues.
    import dspy

    if config.llm.name == "dummy":
        logger.debug("Using dummy LLM")
        from modelsmc.method.modules.llm import DummyLLM

        lm = DummyLLM()

    else:
        logger.debug("Using LLM")
        # Load API key from .env as a fallback for direct invocation
        # (``python -m modelsmc.main``).  When invoked via ``modelsmc``,
        # dotenvx has already injected the variables into the environment.
        dotenv.load_dotenv()
        api_key = os.getenv(config.llm.api_key_name)
        if api_key is None:
            raise ValueError(f"API key not found for {config.llm.api_key_name}")

        # configure dspy and llm
        lm_kwargs = dict(
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            api_key=api_key,
            temperature=config.llm.temperature,
            cache=config.llm.cache,
        )

        # Provider-specific reasoning kwargs (passed through to litellm).
        if getattr(config.llm, "reasoning_effort", None) is not None:
            lm_kwargs["reasoning_effort"] = config.llm.reasoning_effort
        if getattr(config.llm, "thinking", None) is not None:
            thinking = dict(OmegaConf.to_container(config.llm.thinking))
            effort = thinking.pop("effort", None)
            lm_kwargs["thinking"] = thinking
            if effort is not None:
                # LiteLLM maps reasoning_effort → output_config.effort for Claude 4.6
                lm_kwargs["reasoning_effort"] = effort

        lm = dspy.LM(**lm_kwargs)

    dspy.configure(lm=lm)

    # get task using factory function
    task = create_task(config.task)

    # run method
    from modelsmc.method import get_method

    method = get_method(config.method.name)
    method(config, task, output_dir)
    logger.info("Script completed successfully")

    # Remove the cache folder of dspy
    if config.delete_dspy_cache:
        cache_dir = os.environ.get("DSPY_CACHEDIR", None)
        if cache_dir is not None and os.path.exists(cache_dir):
            logger.info(f"Removing dspy cache folder {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            logger.info("No dspy cache folder to remove")


# Entry point for direct module execution (``python -m modelsmc.main``).
# Calls ``main()`` rather than ``_main()`` directly so that the ASCII banner
# is printed before Hydra takes over.  The ``modelsmc`` CLI entry point is
# ``main_wrapper()``, which re-invokes this module via a ``dotenvx`` subprocess.
if __name__ == "__main__":
    main()
