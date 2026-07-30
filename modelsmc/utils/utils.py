import logging
import os
import traceback
import types

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("ModelSMC")


def register_omegaconf_resolvers():
    """Register custom OmegaConf resolvers used across the project.

    Idempotent: safe to call multiple times (e.g. from both main.py and
    post-hoc evaluation paths that reload configs independently).
    """
    resolvers = {
        "div": lambda a, b: float(a) / float(b),
        "int": lambda a: int(a),
        "round": lambda a: round(float(a)),
        "min": lambda a, b: min(int(a), int(b)),
        "max": lambda a, b: max(int(a), int(b)),
        "mul": lambda a, b: float(a) * float(b),
        "sub": lambda a, b: float(a) - float(b),
    }
    for name, fn in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception as e:
            logger.debug(
                f"Resolver '{name}' not registered (likely already registered): {e}"
            )


def setup_worker_logging() -> logging.Logger:
    """Configure the ModelSMC logger for use inside a Ray worker process.

    Ray workers run in separate Python processes that do not inherit the
    logging configuration (handlers, level) of the driver process.  This
    function reads the desired log level from the ``MODELSMC_LOG_LEVEL``
    environment variable (set by ``main.py`` before workers are spawned),
    attaches a ``StreamHandler`` pointing to stdout, and returns the
    configured logger.

    Because Ray's ``log_to_driver=True`` pipes each worker's stdout back to
    the driver, log records written through this handler will appear in the
    main process output at exactly the level configured in ``config.logging``.

    The handler is only added once (guarded by ``not worker_logger.handlers``)
    so that re-importing the module inside the same worker does not duplicate
    output.

    Returns:
        The configured ``logging.Logger`` instance for the ``"ModelSMC"``
        namespace.
    """
    level_name = os.environ.get("MODELSMC_LOG_LEVEL", "WARNING")
    level = getattr(logging, level_name, logging.WARNING)
    worker_logger = logging.getLogger("ModelSMC")
    if not worker_logger.handlers:
        worker_logger.setLevel(level)
        worker_logger.propagate = False
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter(
                fmt=(
                    "[%(asctime)s][%(name)s][%(filename)s: %(lineno)d][%(levelname)s] "
                    "- %(message)s"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        worker_logger.addHandler(ch)
    return worker_logger


def detect_code_type(code_string: str) -> str:
    """
    Simple function to detect PyTorch modules vs NumPy functions.

    Detection order matters: NumPy ODE functions (def equation + numpy/np usage) are
    checked before the generic 'python' PyTorch check so that LLM-SR-Bench equations
    are not mis-classified as PyTorch modules.
    """
    code = code_string.lower()
    # Check for NumPy ODE equation function (LLM-SR-Bench style).
    # Must have a bare `def equation(` signature AND reference numpy/np – no nn.Module.
    if (
        "def equation(" in code
        and ("numpy" in code or "np." in code or "import numpy" in code)
        and "nn.module" not in code
    ):
        return "numpy_func"
    # Check for PyTorch indicators
    if any(
        indicator in code
        for indicator in ["python", "torch", "nn.module", "nn.linear", "forward"]
    ):
        return "py"
    return "unknown"


class NumPyEquationWrapper:
    """Thin callable wrapper around a plain NumPy ``equation(t, P, params)`` function.

    The LLM-SR-Bench bio task expects the LLM to generate::

        import numpy as np

        def equation(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:
            ...

    This wrapper holds the extracted function so that ``simulation_wrapper``
    (defined in ``SRBenchBioTaskBase``) can call it as::

        wrapper.equation(t, P, params)

    It is intentionally *not* a ``torch.nn.Module`` – the SRBench bio task's
    ``simulation_wrapper`` handles all tensor↔numpy conversions.
    """

    def __init__(self, equation_fn):
        self._equation_fn = equation_fn

    def equation(self, t, P, params):
        """Delegate to the LLM-generated equation function."""
        return self._equation_fn(t, P, params)


def strip_markdown_fences(code_string: str) -> str:
    """Strip leading/trailing markdown code fences that LLMs sometimes add.

    Handles both ```python and plain ``` fences.  Uses removeprefix/removesuffix
    rather than str.strip(chars) to avoid accidentally removing individual
    characters that appear in the fence tokens (e.g. 't', 'n', 'p', 'y').
    """
    code_string = code_string.strip()
    if code_string.startswith("```python"):
        code_string = code_string.removeprefix("```python")
    elif code_string.startswith("```"):
        code_string = code_string.removeprefix("```")
    if code_string.endswith("```"):
        code_string = code_string.removesuffix("```")
    return code_string.strip()


def convert_numpy_to_callable(code_string: str):
    """Exec a NumPy ``equation`` function and wrap it in a ``NumPyEquationWrapper``.

    Handles LLM-SR-Bench style code of the form::

        import numpy as np

        def equation(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:
            dP_dt = params[0] * P + params[1]
            return dP_dt

    The code is stripped of leading/trailing fences, executed in an isolated
    module namespace, and the ``equation`` function is extracted and wrapped.

    Returns:
        (NumPyEquationWrapper, None) on success, or (None, error_dict) on failure.
    """

    try:
        code_string = strip_markdown_fences(code_string)

        user_code_module = types.ModuleType("user_code")
        exec(code_string, user_code_module.__dict__)

        # Find the top-level function named 'equation'
        if not hasattr(user_code_module, "equation"):
            raise ValueError(
                "No top-level function named 'equation' found in generated code"
            )

        equation_fn = user_code_module.equation
        if not callable(equation_fn):
            raise ValueError("'equation' is not callable in generated code")

        return NumPyEquationWrapper(equation_fn), None

    except Exception as e:
        logger.error(f"Error converting NumPy equation to callable: {e}")
        tb = traceback.extract_tb(e.__traceback__)
        formatted_tb = "\n".join(traceback.format_list(tb))
        error = f"{type(e).__name__}: {formatted_tb}\n{e}"
        error_type = "numpy equation conversion failed"
        output_dict = {
            "errors": error,
            "error_type": error_type,
        }
        return None, output_dict


def code_type_to_file_ext(code_type: str) -> str:
    """Map an internal code-type string to a proper file extension.

    ``detect_code_type`` returns ``"numpy_func"`` for LLM-SR-Bench style
    equations, but those are plain Python files and should be saved with a
    ``.py`` extension.  Unknown types are returned unchanged so callers
    always get a usable extension.
    """
    _EXT_MAP = {"numpy_func": "py", "py": "py"}
    return _EXT_MAP.get(code_type, code_type)


def find_simulator_file(folder: str) -> "str | None":
    """Return the path of the simulator source file inside *folder*, or None."""
    path = os.path.join(folder, "simulator.py")
    return path if os.path.exists(path) else None


def convert_model_string(code_string: str):
    """Convert a model string to a module."""
    logger.debug("Converting model string to (compiled) module.")
    code_type = detect_code_type(code_string)
    if code_type == "numpy_func":
        return convert_numpy_to_callable(code_string)
    elif code_type == "py":
        return convert_python_to_module(code_string)
    else:
        raise ValueError(f"Unknown code type: {code_type}")


def convert_python_to_module(code_string: str):
    """Convert a model string to a module."""
    try:
        code_string = strip_markdown_fences(code_string)

        user_code_module = types.ModuleType("user_code")
        exec(code_string, user_code_module.__dict__)

        # Find the model class by inspecting the module (in case LLM renamed the class)
        model_classes = [
            name
            for name, obj in user_code_module.__dict__.items()
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module)
        ]
        if not model_classes:
            logger.error("No PyTorch model class found in generated code")
            raise ValueError("No PyTorch model class found in generated code")

        # Make the discovered model class callable directly from the module
        model_class = getattr(user_code_module, model_classes[0])
    except Exception as e:
        logger.error(f"Error converting model string to module: {e}")
        tb = traceback.extract_tb(e.__traceback__)
        formatted_tb = "\n".join(traceback.format_list(tb))
        error = f"{type(e).__name__}: {formatted_tb}\n{e}"
        error_type = "simulator code conversion failed"
        output_dict = {
            "errors": error,
            "error_type": error_type,
        }
        return None, output_dict

    return model_class(), None


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary, using dot notation for nested keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _prepare_summary_data(
    run_id: str,
    particle_id: str,
    parent_id: str,
    itr: int,
    metrics: dict,
    config: DictConfig = None,
    **kwargs,
) -> dict:
    """Prepare and flatten summary data for CSV storage."""
    flat_metrics = {
        "run_id": run_id,
        "particle_id": particle_id,
        "parent_id": parent_id,
        "method": config.method.name,
        "task": config.task.name,
        "metric_to_optimize": config.task.metric_to_optimize,
        "llm": config.llm.model,
        "iteration": itr,
    }

    # Add any additional kwargs to flat_metrics
    flat_metrics.update(kwargs)

    if metrics is not None:
        # Add eval_metrics directly to top level
        if "eval_metrics" in metrics and isinstance(metrics["eval_metrics"], dict):
            flat_metrics.update(metrics["eval_metrics"])

        if "errors" in metrics and isinstance(metrics["errors"], str):
            flat_metrics["errors"] = metrics["errors"].replace("\n", " | ")

        # Add other keys, flattening nested dicts (skip errors to avoid clutter)
        for k, v in metrics.items():
            if k not in ["eval_metrics", "errors", "error_type"]:
                if isinstance(v, dict):
                    flat_metrics.update(flatten_dict(v, parent_key=k))
                else:
                    flat_metrics[k] = v
    # add config at the end
    flat_metrics["config"] = str(OmegaConf.to_container(config))

    return flat_metrics


def _append_to_csv(csv_path: str, new_data: dict):
    """
    Append new data to CSV, handling column alignment automatically with
    thread-safe file locking.
    """
    lock_file = csv_path + ".lock"
    lock = FileLock(lock_file)

    with lock:
        new_df = pd.DataFrame([new_data])

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)

            # Check if either DataFrame is empty to avoid FutureWarning
            if existing_df.empty:
                df = new_df
            else:
                # Check if a particle with this id is already in the summary for this
                # run, if that is the case, do not add it.

                df_run = existing_df[existing_df["run_id"] == new_data["run_id"]]
                existing_particle_ids = df_run["particle_id"].unique()
                if new_data["particle_id"] in existing_particle_ids:
                    logger.debug(
                        f"particle with uuid = {new_data['particle_id']} is already "
                        "in the summary."
                    )
                    return

                # Pandas concat automatically aligns columns and fills missing values
                # with NaN. Filter out columns that are entirely NA in new_df to avoid
                # FutureWarning
                new_df_filtered = new_df.dropna(axis=1, how="all")
                if new_df_filtered.empty:
                    # If all columns were dropped, use existing_df
                    df = existing_df
                else:
                    df = pd.concat(
                        [existing_df, new_df_filtered], ignore_index=True, sort=False
                    )
        else:
            df = new_df

        df.to_csv(csv_path, index=False)


def save_summary(
    run_id: str,
    particle_id: str,
    parent_id: str,
    itr: int,
    metrics: dict,
    save_dir: str = None,
    config: DictConfig = None,
    **kwargs,
):
    """Save summary to a CSV file with thread-safe file locking."""
    summary_csv_path = os.path.join(save_dir, "summary.csv")
    flat_metrics = _prepare_summary_data(
        run_id=run_id,
        particle_id=particle_id,
        parent_id=parent_id,
        itr=itr,
        metrics=metrics,
        config=config,
        **kwargs,
    )
    _append_to_csv(summary_csv_path, flat_metrics)


def save_pool(pool, itr, save_dir, run_id):
    """Save pool composition to a CSV file with thread-safe file locking."""
    csv_path = os.path.join(save_dir, "pool_composition.csv")

    lock_file = csv_path + ".lock"
    lock = FileLock(lock_file)

    with lock:
        particles = pool.get_particles()

        new_row_dict = {
            "run_id": [run_id],
            "itr": [itr],
            "pool_members_uuids": [[p.uuid for p in particles]],
        }
        new_df = pd.DataFrame(new_row_dict)

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)

            # Check if either DataFrame is empty to avoid FutureWarning
            if existing_df.empty:
                df = new_df
            else:
                df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        else:
            df = new_df

        df.to_csv(csv_path, index=False)


def save_GMM_idx_distribution(
    pool, itr: int, save_dir: str, run_id: str, num_GMM_configs: int
):
    """Record the distribution of GMM configuration indices across the particle pool.

    Inspects each particle's implementation code to extract its GMM configuration
    index (via the ``__GMM_config_idx =`` assignment line), counts how many
    particles use each configuration at the current iteration, and appends a
    one-row summary to ``<save_dir>/GMM_idx_distribution.csv``.

    The resulting CSV has one row per call with the following columns:

    - ``run_id``          — identifier of the current experiment run.
    - ``itr``             — SMC iteration number.
    - ``counts(idx=k)``   — number of particles currently assigned to GMM
                            configuration ``k``, for ``k`` in
                            ``range(num_GMM_configs)``.  Configurations with
                            zero particles are still recorded as ``0``.

    File writes are protected by a ``FileLock`` so that concurrent Ray workers
    do not corrupt the shared CSV.

    The GMM index is parsed directly from each particle's source code by
    searching for a line that starts with ``"__GMM_config_idx ="``.  Every
    particle must contain exactly one such line; the function will raise an
    ``AssertionError`` otherwise.

    Args:
        pool:            Current particle pool whose GMM index distribution
                         is to be recorded.
        itr:             Current SMC iteration index (used as a CSV row label).
        save_dir:        Directory in which ``GMM_idx_distribution.csv`` is
                         written (created by the caller if it does not exist).
        run_id:          Identifier for the current experiment run (used as a
                         CSV row label).
        num_GMM_configs: Total number of available GMM configurations.
                         Determines which ``counts(idx=k)`` columns are
                         written.
    """
    csv_path = os.path.join(save_dir, "GMM_idx_distribution.csv")

    # Extract the GMM configuration index from each particle's source code.
    particles = pool.get_particles()

    GMM_idx_list = []

    for p in particles:
        lines = [line.strip() for line in p.implementation.splitlines() if line.strip()]

        prefix = "__GMM_config_idx"
        matching = [line for line in lines if line.startswith(prefix)]

        assert len(matching) == 1, (
            f"Expected exactly one '__GMM_config_idx' line in particle "
            f"implementation, found {len(matching)}."
        )

        # Parse the integer on the right-hand side of the assignment.
        idx_val = int(matching[0].split("=")[1].strip())

        GMM_idx_list.append(idx_val)

    values, counts = np.unique(GMM_idx_list, return_counts=True)

    new_row_dict = {"run_id": [run_id], "itr": [itr]}

    for idx in range(num_GMM_configs):
        if idx in values:
            match_positions = np.where(values == idx)[0]
            assert len(match_positions) == 1, (
                f"Expected exactly one occurrence of GMM config idx={idx} "
                f"in unique values array, found {len(match_positions)}."
            )
            new_row_dict[f"counts(idx={idx})"] = [int(counts[match_positions[0]])]
        else:
            new_row_dict[f"counts(idx={idx})"] = [0]

    new_df = pd.DataFrame(new_row_dict)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)

        # Check if either DataFrame is empty to avoid FutureWarning.
        if existing_df.empty:
            df = new_df
        else:
            # Guard against duplicate rows for the same (run_id, itr),
            already_logged = (
                (existing_df["run_id"] == run_id) & (existing_df["itr"] == itr)
            ).any()
            if already_logged:
                logger.debug(
                    f"GMM idx distribution for run_id={run_id}, itr={itr} "
                    "is already recorded — skipping."
                )
                return
            df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df

    lock_file = csv_path + ".lock"
    lock = FileLock(lock_file)

    with lock:
        df.to_csv(csv_path, index=False)
