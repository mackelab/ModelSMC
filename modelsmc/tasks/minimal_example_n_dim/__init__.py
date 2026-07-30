import logging
from pathlib import Path

from .generate_models import GMMConfigs

logger = logging.getLogger("ModelSMC")

# Load GMM configurations
try:
    BASE_DIR = Path(__file__).resolve().parent
    GMM_configs = GMMConfigs.model_validate_json(
        open(f"{BASE_DIR}/gmm_configs.json").read()
    )
except FileNotFoundError:
    logger.error(
        "GMM configurations not found. Please run"
        "'python modelsmc/tasks/minimal_example_n_dim/generate_models.py --new_config'"
        "to generate new configurations."
    )
