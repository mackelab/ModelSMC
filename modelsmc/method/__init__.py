# Fix the problem of different tasks accessing the same dspy cache
# leading to errors. We create a unique cache folder for each run
# and delete it afterwards. Not using the cache can only be set after
# importing dspy, so we have to do it this way. May be resolved in future
# dspy versions.

import os
import uuid
from pathlib import Path
from typing import Callable

from hydra.core.hydra_config import HydraConfig

# Try to get the hydra config
try:
    cfg = HydraConfig.get()

except Exception as e:
    print(f"Error getting hydra config: {e}")
    cfg = None


# If the hydra config is available, try to get the cache folder from the output dir
if cfg is not None:
    # Base folder containing all dspy caches
    output_dir = Path(cfg.runtime.output_dir)
    cache_folder_base = os.path.join(output_dir, ".dspy_cache")

    # Create the base folder for dspy caches if it does not exist
    if not os.path.exists(cache_folder_base):
        os.mkdir(cache_folder_base)

    # Check if the environment variable DSPY_CACHEDIR is already set. If this is the
    # case, do not set it again to avoid having multiple caches for the same run.
    if "DSPY_CACHEDIR" in os.environ:
        print(
            f"DSPY_CACHEDIR is already set to {os.environ['DSPY_CACHEDIR']}. "
            "Skipping setting it again."
        )

    # If not, set it to a unique folder based on the job id if available
    else:
        print("DSPY_CACHEDIR is not set yet, setting it now.")

        # Try to get the job id from hydra config
        try:
            new_idx = f"job_{cfg.job.id}"

        # If not available, use a random index
        except Exception as e:
            print(f"Error getting job id: {e}")

            # Use uuid for the index
            new_idx = str(uuid.uuid4())[:8]

        # Store the cache folder as env variable
        os.environ["DSPY_CACHEDIR"] = f"{cache_folder_base}/.dspy_cache_{new_idx}"
        print(f"Using dspy cache folder {os.environ['DSPY_CACHEDIR']}")

else:
    # Check if the environment variable DSPY_CACHEDIR is already set.
    # If this is the case, everything is ok, otherwise something went wrong
    if "DSPY_CACHEDIR" not in os.environ:
        raise ValueError(
            "DSPY_CACHEDIR is not set and hydra config is not available. "
            "Cannot set dspy cache folder."
        )

# Configure dspy to not use disk or memory cache
import dspy

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Import methods using dspy
from modelsmc.method.modelsmc import smc_method  # noqa: E402
from modelsmc.method.modelsmc_no_LLM import smc_method_minimal  # noqa: E402


def get_method(name: str) -> Callable:
    """Return the entry-point function for the requested method.

    Args:
        name: Method identifier (``"smc_method"``, ``"smc_method_minimal"``).

    Returns:
        The callable implementing the requested method.

    Raises:
        ValueError: If ``name`` is not a known method.
    """
    if name == "smc_method":
        return smc_method
    elif name == "smc_method_minimal":
        return smc_method_minimal
    else:
        raise ValueError(f"Unknown method {name}")
