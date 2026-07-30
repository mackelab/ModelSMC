"""
Compare Hydra configs stored in summary.csv files across a two-level folder
hierarchy (method / llm).

For every ``<root>/<method>/<llm>/summary.csv``:

  1. Verify that all configs within that file differ *only* in ``seed`` and
     ``run_id`` (flag a warning if they differ in anything else).
  2. Extract one representative (stripped) config for that run group.

Then compare representative configs across all discovered run groups and print
a table whose rows are nested dot-notation config keys and whose columns are
the run groups labelled ``METHOD.LLM``.

By default only keys whose values differ across groups are shown.  Pass
``--full`` to show every key; rows that differ are then marked with a leading
``*``.  Rows are sub-grouped by their top-level config key with a blank line
between groups.

Usage
-----
    python diagnostics/compare_configs_across_runs.py <root_folder>
    python diagnostics/compare_configs_across_runs.py <root_folder> --full
    python diagnostics/compare_configs_across_runs.py <root_folder> --no-check
    python diagnostics/compare_configs_across_runs.py <root_folder> --full --no-check
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    LINE_WIDTH,
    find_differing_paths,
    get_nested,
    make_hashable,
    ok,
    strip_config,
    warn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_representative_config(csv_path: Path, sanity_check: bool) -> dict | None:
    """Load *csv_path*, optionally warn if intra-file configs differ beyond
    seed/run_id, and return a single stripped representative config.

    Returns ``None`` if the file cannot be loaded or has no config column.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(warn(f"Could not read {csv_path}: {exc}"))
        return None

    if "config" not in df.columns:
        print(warn(f"No 'config' column in {csv_path} — skipping."))
        return None

    configs: list[dict] = []
    for raw in df["config"]:
        try:
            configs.append(ast.literal_eval(str(raw)))
        except Exception as e:
            print(warn(f"Problems parsing config: {e}"))

    if not configs:
        print(warn(f"Could not parse any config in {csv_path} — skipping."))
        return None

    stripped = [strip_config(c) for c in configs]

    if sanity_check:
        # Check that all rows share the same config (ignoring seed/run_id)
        unique_keys = {make_hashable(s) for s in stripped}
        if len(unique_keys) > 1:
            rep_configs = [stripped[0]]
            seen = {make_hashable(stripped[0])}
            for s in stripped[1:]:
                if make_hashable(s) not in seen:
                    rep_configs.append(s)
                    seen.add(make_hashable(s))
            diff_paths = find_differing_paths(rep_configs)
            print(
                warn(
                    f"{csv_path}: configs differ in keys other than seed/run_id: "
                    + ", ".join(diff_paths)
                )
            )

    return stripped[0]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def _all_paths(configs: list[dict], _prefix: str = "") -> list[str]:
    """Return sorted dot-notation paths of *all* leaf values across *configs*."""
    all_keys: set[str] = set().union(*configs)
    paths: list[str] = []
    for k in sorted(all_keys):
        full_path = f"{_prefix}.{k}" if _prefix else k
        vals = [cfg.get(k) for cfg in configs]
        non_none = [v for v in vals if v is not None]
        if non_none and all(isinstance(v, dict) for v in non_none):
            sub_configs = [v if isinstance(v, dict) else {} for v in vals]
            paths.extend(_all_paths(sub_configs, _prefix=full_path))
        else:
            paths.append(full_path)
    return paths


def compare(root: Path, sanity_check: bool = True, full: bool = False) -> None:
    if not root.is_dir():
        raise SystemExit(f"ERROR: {root} is not a directory.")

    # Discover method/llm subfolders that contain summary.csv
    entries: list[tuple[str, str, Path]] = []  # (method, llm, csv_path)
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        for llm_dir in sorted(method_dir.iterdir()):
            if not llm_dir.is_dir():
                continue
            csv_path = llm_dir / "summary.csv"
            if csv_path.exists():
                entries.append((method_dir.name, llm_dir.name, csv_path))

    if not entries:
        raise SystemExit(
            f"ERROR: No summary.csv files found under two-level subfolders of {root}."
        )

    print("=" * LINE_WIDTH)
    print(f"  Root : {root}")
    print(f"  Found {len(entries)} summary.csv file(s):")
    for method, llm, _csv_path in entries:
        print(f"    {method}/{llm}/summary.csv")
    print("=" * LINE_WIDTH)

    # Load one representative config per entry
    col_labels: list[str] = []
    rep_configs: list[dict] = []
    for method, llm, csv_path in entries:
        label = f"{method}.{llm}"
        rep = _load_representative_config(csv_path, sanity_check=sanity_check)
        if rep is not None:
            col_labels.append(label)
            rep_configs.append(rep)

    if len(rep_configs) < 2:
        print("\nNeed at least two loadable summary.csv files to compare. Exiting.")
        return

    # Find all keys that differ across representative configs
    diff_paths = find_differing_paths(rep_configs)

    print()
    if not diff_paths and not full:
        print(ok("All configs are identical (ignoring seed and run_id)."))
        print()
        return

    if full:
        table_paths = _all_paths(rep_configs)
    else:
        table_paths = diff_paths

    diff_set = set(diff_paths)

    if diff_paths:
        print(f"  {len(diff_paths)} config key(s) differ across run groups.\n")
    else:
        print(ok("All configs are identical (ignoring seed and run_id).\n"))

    # ── Build and print the table ────────────────────────────────────────────
    # Column widths
    key_col_w = max(len("config key"), max(len(p) for p in table_paths))
    val_col_ws = [
        max(len(lbl), max(len(repr(get_nested(cfg, p))) for p in table_paths))
        for lbl, cfg in zip(col_labels, rep_configs, strict=False)
    ]

    def _row(key: str, vals: list[str], prefix: str = "  ") -> str:
        cells = [f"{key:<{key_col_w}}"]
        for v, w in zip(vals, val_col_ws, strict=False):
            cells.append(f"  {v:<{w}}")
        return prefix + "  ".join(cells[0:1]) + "".join(cells[1:])

    # Header
    print(_row("config key", col_labels))
    print(_row("─" * key_col_w, ["─" * w for w in val_col_ws]))

    # Group paths by their top-level key and print with blank lines between groups
    current_top: str | None = None
    for path in table_paths:
        top = path.split(".")[0]
        if current_top is not None and top != current_top:
            print()
        current_top = top
        vals = [repr(get_nested(cfg, path)) for cfg in rep_configs]
        prefix = "* " if full and path in diff_set else "  "
        print(_row(path, vals, prefix=prefix))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hydra configs across summary.csv files found in a "
            "two-level <method>/<llm> folder hierarchy."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder containing <method>/<llm>/summary.csv files.",
    )
    parser.add_argument(
        "--no-check",
        dest="sanity_check",
        action="store_false",
        default=True,
        help=(
            "Skip the intra-file check that verifies configs within each "
            "summary.csv only differ in seed and run_id."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help=(
            "Print all config keys, not just the ones that differ. "
            "Rows that differ are marked with a leading '*'."
        ),
    )
    args = parser.parse_args()
    compare(args.root.resolve(), sanity_check=args.sanity_check, full=args.full)


if __name__ == "__main__":
    main()
