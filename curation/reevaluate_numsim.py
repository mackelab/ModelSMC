"""
Post-hoc parameter re-evaluation at 5000 simulations.

For each run in main_results/allen_ablation/{numsim_200,numsim_500,numsim_1000}/<llm>/
run_{0,1,2,3,4}/, this script:

  1. Creates run_{seed}/reevaluated/
  2. Copies simulator.py (and scm.txt if present) into it
  3. Writes a .hydra/config.yaml with num_simulations overridden to 5000
  4. Calls SMCOrchestrator.posthoc_parameter_estimation pointing to the
     training/validation data that live in best_particle/

The rename of best_particle/ to original_best_particle/ is idempotent (skipped
if already renamed), but the posthoc parameter estimation itself is not: it
re-runs every time.

--llm selects a single LLM per invocation (default: sonnet); run the script
again with e.g. --llm gptmini to process another one.

Usage (from the repo root):
    dotenvx run -- python curation/reevaluate_numsim.py
    dotenvx run -- python curation/reevaluate_numsim.py --numsim numsim_1000 --run run_2
    dotenvx run -- python curation/reevaluate_numsim.py --llm gptmini
"""

import argparse
import logging
import os
import shutil
import tempfile

# ── Environment variables that must be set before any modelsmc imports ────────

# Ray remote decorators read these at smc_method module-import time.
os.environ.setdefault("REQ_NUM_CPUS_PER_RAY_THREAD", "1")
os.environ.setdefault("REQ_NUM_GPUS_PER_RAY_THREAD", "1")

# modelsmc/method/__init__.py raises if DSPY_CACHEDIR is unset and Hydra
# is not active.  Disk caching is disabled by dspy.configure_cache() in that
# same __init__, so the directory is never actually written to; any writable
# path satisfies the check.
os.environ.setdefault(
    "DSPY_CACHEDIR",
    os.path.join(tempfile.gettempdir(), "dspy_cache_reevaluate_numsim"),
)

from omegaconf import OmegaConf, open_dict  # noqa: E402

from modelsmc.method.modelsmc import SMCOrchestrator  # noqa: E402
from modelsmc.utils.utils import register_omegaconf_resolvers  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reevaluate_numsim")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO_ROOT, "main_results", "allen_ablation")
ALL_NUMSIM_DIRS = sorted(
    d
    for d in os.listdir(BASE_DIR)
    if d.startswith("numsim_") and os.path.isdir(os.path.join(BASE_DIR, d))
)
TARGET_NUM_SIMULATIONS = 5000

# Register OmegaConf resolvers once before any config loading.
register_omegaconf_resolvers()

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Post-hoc parameter re-evaluation.")
parser.add_argument(
    "--numsim",
    default=None,
    choices=ALL_NUMSIM_DIRS,
    help="Run only this numsim folder (e.g. numsim_1000). Default: all.",
)
parser.add_argument(
    "--run",
    default=None,
    help="Run only this run id (e.g. run_2). Default: all.",
)
parser.add_argument(
    "--llm",
    default="sonnet",
    help="Run the evaluation for this llm",
)
args = parser.parse_args()

NUMSIM_DIRS = [args.numsim] if args.numsim is not None else ALL_NUMSIM_DIRS

# ── Main loop ─────────────────────────────────────────────────────────────────

for numsim_dir in NUMSIM_DIRS:
    numsim_path = os.path.join(BASE_DIR, numsim_dir)
    all_seeds = sorted(
        int(d.removeprefix("run_"))
        for d in os.listdir(os.path.join(numsim_path, args.llm))
        if d.startswith("run_")
        and os.path.isdir(os.path.join(numsim_path, args.llm, d))
    )
    seeds = [int(args.run.removeprefix("run_"))] if args.run is not None else all_seeds
    for seed in seeds:
        run_dir = os.path.join(BASE_DIR, numsim_dir, args.llm, f"run_{seed}")

        # Rename the best particle folder. If the renamed version is already there,
        # do not perform the renaming
        NEW_NAME_ORIGINAL_BEST_PARTICLE_FOLDER = "original_best_particle"

        content = os.listdir(run_dir)
        if NEW_NAME_ORIGINAL_BEST_PARTICLE_FOLDER in content:
            logger.info("The original best particle folder was already renamed")
        else:
            # Check if the folder to rename is actually present
            assert "best_particle" in content

            # Rename
            os.rename(
                os.path.join(run_dir, "best_particle"),
                os.path.join(run_dir, NEW_NAME_ORIGINAL_BEST_PARTICLE_FOLDER),
            )

            logger.info(
                f"Rename {os.path.join(run_dir, 'best_particle')} to "
                f"{os.path.join(run_dir, NEW_NAME_ORIGINAL_BEST_PARTICLE_FOLDER)}"
            )

        best_particle_dir = os.path.join(
            run_dir, NEW_NAME_ORIGINAL_BEST_PARTICLE_FOLDER
        )
        reevaluated_dir = os.path.join(run_dir, "best_particle")

        # ── Verify source exists ───────────────────────────────────────────
        if not os.path.isdir(best_particle_dir):
            logger.warning(
                "best_particle dir not found, skipping: %s", best_particle_dir
            )
            continue

        src_config_path = os.path.join(best_particle_dir, ".hydra", "config.yaml")
        if not os.path.exists(src_config_path):
            logger.warning(
                "No .hydra/config.yaml found in %s, skipping.", best_particle_dir
            )
            continue

        # ── Create reevaluated directory ───────────────────────────────────
        os.makedirs(reevaluated_dir, exist_ok=True)
        logger.info("Preparing %s", reevaluated_dir)

        # ── Copy simulator file ────────────────────────────────────────────
        copied_simulator = False
        for ext in ("py", "R"):
            sim_src = os.path.join(best_particle_dir, f"simulator.{ext}")
            if os.path.exists(sim_src):
                shutil.copy2(sim_src, os.path.join(reevaluated_dir, f"simulator.{ext}"))
                logger.info("  Copied simulator.%s", ext)
                copied_simulator = True
                break
        if not copied_simulator:
            logger.warning(
                "  No simulator file found in %s, skipping.", best_particle_dir
            )
            continue

        # ── Copy scm.txt if present ────────────────────────────────────────
        scm_src = os.path.join(best_particle_dir, "scm.txt")
        if os.path.exists(scm_src):
            shutil.copy2(scm_src, os.path.join(reevaluated_dir, "scm.txt"))
            logger.info("  Copied scm.txt")

        # ── Write modified .hydra/config.yaml with num_simulations=5000 ───
        hydra_dst_dir = os.path.join(reevaluated_dir, ".hydra")
        os.makedirs(hydra_dst_dir, exist_ok=True)
        dst_config_path = os.path.join(hydra_dst_dir, "config.yaml")

        cfg = OmegaConf.load(src_config_path)
        with open_dict(cfg):
            # Store the number of simulations used during the discovery run as well
            cfg.method.optimizer.num_simulations_discovery = (
                cfg.method.optimizer.num_simulations
            )

            cfg.method.optimizer.num_simulations = TARGET_NUM_SIMULATIONS
        OmegaConf.save(cfg, dst_config_path)
        logger.info(
            "  Wrote config with num_simulations=%d to %s",
            TARGET_NUM_SIMULATIONS,
            dst_config_path,
        )

        # ── Run post-hoc parameter estimation ─────────────────────────────
        train_data_path = os.path.join(best_particle_dir, "training_data.pt")
        valid_data_path = os.path.join(best_particle_dir, "validation_data.pt")

        logger.info("  Running posthoc_parameter_estimation ...")
        SMCOrchestrator.posthoc_parameter_estimation(
            folder=reevaluated_dir,
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
        )
        logger.info("  Done.")

logger.info("All re-evaluations complete.")
