"""
Copy results CSV files (and optionally best-particle folders) from an input
folder into a target folder.

Tracked CSV files:
  - summary.csv
  - pool_composition.csv
  - GMM_idx_distribution.csv

By default the ``_filtered`` variants are copied and the ``_filtered`` suffix
is stripped so the target folder receives the canonical filenames.
Pass ``--raw`` to copy the original (unfiltered) files instead.

If ``--copy-best-particle`` is set (the default), the best_particle/ folder is
also copied for every unique run_id present in the summary file.  Run folders
are located by scanning subfolders of input_folder for ``.hydra/config.yaml``
files whose ``run_id`` field matches; any config path with a ``best_particle``
component is skipped so that a run's own top-level Hydra config is matched
instead of the (also present) config snapshot nested inside its
``best_particle/`` folder.  This means ``input_folder`` must be a raw Hydra
run/sweep output directory (each run folder carrying its own top-level
``.hydra/config.yaml``, e.g. ``results/<name>/<timestamp>/``) — not an
already-curated folder produced by a previous run of this script, since that
only retains the config nested inside ``best_particle/``.  The hydra config
is cross-checked against the config stored in the summary row (yaml keys
only).  Copied particle folders are placed in target_folder/run_0/, run_1/,
run_2/, …

By default a ValueError is raised if any target file or folder already exists.
Pass ``--overwrite`` to replace existing targets instead.

Usage:
    python curation/copy_results.py <input_folder> <target_folder>
    python curation/copy_results.py <input_folder> <target_folder> --raw
    python curation/copy_results.py <input_folder> <target_folder> --no-copy-best-particle
    python curation/copy_results.py <input_folder> <target_folder> --overwrite
    python curation/copy_results.py <input_folder> <target_folder> --copy-baseline
    python curation/copy_results.py <input_folder> <target_folder> --ensure-parameter-estimates
"""  # noqa

import argparse
import ast
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "diagnostics"))
    from utils import LINE_WIDTH
except ImportError:
    LINE_WIDTH = 76

TRACKED_FILES = [
    "summary.csv",
    "pool_composition.csv",
    "GMM_idx_distribution.csv",
]
FILTERED_SUFFIX = "_filtered"


# ---------------------------------------------------------------------------
# Config comparison helpers
# ---------------------------------------------------------------------------


def _deep_mismatches(yaml_val: Any, summary_val: Any, path: str = "") -> list[str]:
    """Return mismatch descriptions for keys present in yaml_val vs summary_val."""
    issues: list[str] = []
    if isinstance(yaml_val, dict):
        for key, val in yaml_val.items():
            child = f"{path}.{key}" if path else key
            if not isinstance(summary_val, dict) or key not in summary_val:
                issues.append(f"{child}: present in yaml but missing in summary config")
            else:
                issues.extend(_deep_mismatches(val, summary_val[key], child))
    else:
        if yaml_val != summary_val:
            issues.append(f"{path}: yaml={yaml_val!r}  summary={summary_val!r}")
    return issues


def _configs_match(yaml_cfg: dict, summary_cfg: dict) -> tuple[bool, list[str]]:
    mismatches = _deep_mismatches(yaml_cfg, summary_cfg)
    return len(mismatches) == 0, mismatches


# ---------------------------------------------------------------------------
# Run-folder discovery
# ---------------------------------------------------------------------------


def find_run_folders(input_folder: Path) -> dict[str, list[tuple[Path, dict]]]:
    """Scan input_folder recursively for .hydra/config.yaml files.

    Skips any path that contains a ``best_particle`` component.
    Returns {run_id: [(run_folder, yaml_cfg), ...]} (multiple entries indicate
    duplicates).
    """
    result: dict[str, list[tuple[Path, dict]]] = {}
    for config_file in sorted(input_folder.rglob(".hydra/config.yaml")):
        if "best_particle" in config_file.parts:
            continue
        run_folder = config_file.parent.parent
        try:
            cfg = yaml.safe_load(config_file.read_text())
        except Exception as exc:
            warnings.warn(
                f"Could not parse {config_file}: {exc}",
                stacklevel=2,
            )
            continue
        run_id = cfg.get("run_id")
        if run_id:
            result.setdefault(str(run_id), []).append((run_folder, cfg))
    return result


# ---------------------------------------------------------------------------
# Source-file resolution
# ---------------------------------------------------------------------------


def _resolve_src(
    input_folder: Path, filename: str, filtered: bool
) -> tuple[Path, bool]:
    """Return (src_path, used_fallback).

    If *filtered* is True and the filtered variant does not exist, fall back to
    the raw file and set *used_fallback* to True.  If neither file exists,
    *src_path* will point to the (missing) preferred file and *used_fallback*
    will be False.
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    if filtered:
        filtered_path = input_folder / f"{stem}{FILTERED_SUFFIX}{suffix}"
        if filtered_path.exists():
            return filtered_path, False
        raw_path = input_folder / filename
        return raw_path, raw_path.exists()
    return input_folder / filename, False


# ---------------------------------------------------------------------------
# CSV copying
# ---------------------------------------------------------------------------


def copy_csv_files(
    input_folder: Path,
    target_folder: Path,
    filtered: bool,
    overwrite: bool,
) -> None:
    target_folder.mkdir(parents=True, exist_ok=True)

    for filename in TRACKED_FILES:
        src, used_fallback = _resolve_src(input_folder, filename, filtered)
        dst = target_folder / filename

        if not src.exists():
            print(f"\n  [skip] {filename} — file not found.")
            continue

        if used_fallback:
            print(
                f"\n  [fallback] {filename} — no filtered variant found, using "
                "raw file."
            )

        if dst.exists():
            if overwrite:
                pass  # fall through to copy
            else:
                raise ValueError(
                    f"Target already exists: {dst}\n"
                    "Pass --overwrite to replace existing files."
                )

        shutil.copy2(src, dst)
        print(f"\n  {src.name}")
        print(f"    {'Copied to':<20}: {dst}")


# ---------------------------------------------------------------------------
# Posthoc parameter estimation helper
# ---------------------------------------------------------------------------


def _run_posthoc_if_needed(
    folder: Path,
    config_path: Path | None = None,
    train_data_path: Path | None = None,
    valid_data_path: Path | None = None,
) -> None:
    """Run posthoc parameter estimation for *folder* if not already present."""
    import os

    if (folder / "parameter_estimates.pt").exists():
        print(f"    Parameter estimates present in {folder.name}, skipping.")
        return

    # Resolve which config file to load.
    resolved_config = (
        config_path if config_path is not None else folder / ".hydra" / "config.yaml"
    )

    # Set environment variables required by modelsmc.method before any
    # import from that package.  DSPY_CACHEDIR must be present when
    # method/__init__.py runs; the Ray resource vars are read by @ray.remote
    # decorators at smc_method module-import time.
    os.environ.setdefault("DSPY_CACHEDIR", ".dspy_cache_eval")
    if resolved_config.exists():
        raw_cfg = yaml.safe_load(resolved_config.read_text()) or {}
        method_cfg = raw_cfg.get("method", {})
        os.environ.setdefault(
            "REQ_NUM_CPUS_PER_RAY_THREAD",
            str(method_cfg.get("req_num_cpus_per_ray_thread", 1)),
        )
        os.environ.setdefault(
            "REQ_NUM_GPUS_PER_RAY_THREAD",
            str(method_cfg.get("req_num_gpus_per_ray_thread", 0)),
        )

    print(f"    Running posthoc parameter estimation for {folder.name}...")
    from modelsmc.method.modelsmc import SMCOrchestrator

    SMCOrchestrator.posthoc_parameter_estimation(
        str(folder),
        config_path=str(resolved_config),
        train_data_path=str(train_data_path) if train_data_path is not None else None,
        valid_data_path=str(valid_data_path) if valid_data_path is not None else None,
    )


# ---------------------------------------------------------------------------
# Best-particle copying
# ---------------------------------------------------------------------------


def copy_best_particles(
    input_folder: Path,
    target_folder: Path,
    filtered: bool,
    overwrite: bool,
    copy_baseline: bool = False,
    ensure_parameter_estimates: bool = False,
) -> None:
    # Determine which summary file to read run_ids from
    summary_src, used_fallback = _resolve_src(input_folder, "summary.csv", filtered)
    if not summary_src.exists():
        print("\n  [skip best_particle] summary.csv — file not found.")
        return
    if used_fallback:
        print("\n  [fallback] summary.csv — no filtered variant found, using raw file.")

    df = pd.read_csv(summary_src)
    if "run_id" not in df.columns:
        warnings.warn(
            "summary.csv has no 'run_id' column; skipping best-particle copy.",
            stacklevel=2,
        )
        return

    unique_run_ids = df["run_id"].astype(str).unique().tolist()
    print(f"\n  Found {len(unique_run_ids)} unique run_id(s) in {summary_src.name}.")

    # Build map from run_id → [(run_folder, yaml_cfg), ...]
    all_run_folders = find_run_folders(input_folder)

    run_counter = 0
    for run_id in unique_run_ids:
        print(f"\n  {'─' * (LINE_WIDTH - 2)}")
        print(f"  run_id: {run_id}")

        # --- locate the run folder ---
        matches = all_run_folders.get(run_id, [])
        if len(matches) == 0:
            warnings.warn(f"No run folder found for run_id={run_id}", stacklevel=2)
            continue
        if len(matches) > 1:
            warnings.warn(
                f"Multiple run folders found for run_id={run_id}: "
                + ", ".join(str(p) for p, _ in matches)
                + " — using the first one.",
                stacklevel=2,
            )
        run_folder, yaml_cfg = matches[0]

        # --- cross-check config ---
        summary_rows = df[df["run_id"].astype(str) == run_id]
        # All rows for the same run_id should share the same config; take the first.
        config_str = (
            summary_rows.iloc[0].get("config", None) if "config" in df.columns else None
        )
        if config_str is not None:
            try:
                summary_cfg = ast.literal_eval(str(config_str))
                ok, mismatches = _configs_match(yaml_cfg, summary_cfg)
                if not ok:
                    warnings.warn(
                        f"Config mismatch for run_id={run_id} "
                        f"({len(mismatches)} key(s)):\n"
                        + "\n".join(f"    {m}" for m in mismatches),
                        stacklevel=2,
                    )
                else:
                    print(
                        f"    Config check     : OK ({len(yaml_cfg)} top-level keys "
                        "match)"
                    )
            except Exception as exc:
                warnings.warn(
                    f"Could not parse config for run_id={run_id}: {exc}",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"No 'config' column in summary; skipping config check for {run_id}.",
                stacklevel=2,
            )

        # --- locate best_particle subfolder ---
        best_particle_src = run_folder / "best_particle"
        if not best_particle_src.exists():
            warnings.warn(
                f"No best_particle/ folder in {run_folder}",
                stacklevel=2,
            )
            continue

        # --- determine target ---
        run_dir_name = f"run_{run_counter}"
        run_dir_dst = target_folder / run_dir_name
        best_particle_dst = run_dir_dst / "best_particle"

        if best_particle_dst.exists():
            if overwrite:
                shutil.rmtree(best_particle_dst)
            else:
                raise ValueError(
                    f"Target already exists: {best_particle_dst}\n"
                    "Pass --overwrite to replace existing folders."
                )

        run_dir_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(best_particle_src, best_particle_dst)
        print(f"    Run folder       : {run_folder}")
        print(f"    Copied to        : {best_particle_dst}")

        if ensure_parameter_estimates:
            _run_posthoc_if_needed(best_particle_dst)

        if copy_baseline:
            baseline_src = run_folder / "iter-0_p0"
            if not baseline_src.exists():
                warnings.warn(
                    f"No iter-0_p0/ folder in {run_folder}",
                    stacklevel=2,
                )
            else:
                baseline_dst = run_dir_dst / "baseline"
                if baseline_dst.exists():
                    if overwrite:
                        shutil.rmtree(baseline_dst)
                    else:
                        raise ValueError(
                            f"Target already exists: {baseline_dst}\n"
                            "Pass --overwrite to replace existing folders."
                        )
                shutil.copytree(baseline_src, baseline_dst)
                print(f"    Baseline copied  : {baseline_dst}")

                if ensure_parameter_estimates:
                    _run_posthoc_if_needed(
                        baseline_dst,
                        config_path=best_particle_dst / ".hydra" / "config.yaml",
                        train_data_path=best_particle_dst / "training_data.pt",
                        valid_data_path=best_particle_dst / "validation_data.pt",
                    )

        run_counter += 1


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def copy_results(
    input_folder: Path,
    target_folder: Path,
    filtered: bool,
    overwrite: bool,
    copy_best_particle: bool,
    copy_baseline: bool = False,
    ensure_parameter_estimates: bool = False,
) -> None:
    print("=" * LINE_WIDTH)
    print(f"  Input folder             : {input_folder}")
    print(f"  Target folder            : {target_folder}")
    print(f"  Use filtered             : {filtered}")
    print(f"  Overwrite                : {overwrite}")
    print(f"  Copy best particle       : {copy_best_particle}")
    print(f"  Copy baseline            : {copy_baseline}")
    print(f"  Ensure param. estimates  : {ensure_parameter_estimates}")
    print("=" * LINE_WIDTH)

    copy_csv_files(input_folder, target_folder, filtered, overwrite)

    if copy_best_particle:
        copy_best_particles(
            input_folder,
            target_folder,
            filtered,
            overwrite,
            copy_baseline,
            ensure_parameter_estimates,
        )

    print("\n" + "=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Copy results CSV files (and best-particle folders) into a target folder."
        )
    )
    parser.add_argument(
        "input_folder", type=Path, help="Folder containing the source CSV files."
    )
    parser.add_argument("target_folder", type=Path, help="Destination folder.")
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Copy the original (unfiltered) files instead of the filtered variants.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files/folders that already exist in the target folder.",
    )
    parser.add_argument(
        "--copy-best-particle",
        dest="copy_best_particle",
        action="store_true",
        default=True,
        help="Also copy the best_particle/ folder for each run_id (default: on).",
    )
    parser.add_argument(
        "--no-copy-best-particle",
        dest="copy_best_particle",
        action="store_false",
        help="Skip copying best_particle/ folders.",
    )
    parser.add_argument(
        "--copy-baseline",
        dest="copy_baseline",
        action="store_true",
        default=False,
        help=(
            "Also copy the iter-0_p0/ baseline folder for each run_id "
            "(requires --copy-best-particle)."
        ),
    )
    parser.add_argument(
        "--no-copy-baseline",
        dest="copy_baseline",
        action="store_false",
        help="Skip copying iter-0_p0/ baseline folders (default).",
    )
    parser.add_argument(
        "--ensure-parameter-estimates",
        dest="ensure_parameter_estimates",
        action="store_true",
        default=False,
        help=(
            "Ensure parameter_estimates.pt exists in each copied folder. "
            "If absent, posthoc parameter estimation is run automatically."
        ),
    )
    args = parser.parse_args()

    input_folder = args.input_folder.resolve()
    if not input_folder.is_dir():
        raise SystemExit(f"ERROR: '{input_folder}' is not a directory.")

    copy_results(
        input_folder,
        args.target_folder.resolve(),
        filtered=not args.raw,
        overwrite=args.overwrite,
        copy_best_particle=args.copy_best_particle,
        copy_baseline=args.copy_baseline,
        ensure_parameter_estimates=args.ensure_parameter_estimates,
    )
