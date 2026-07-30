"""
Cross-check consistency between summary.csv and pool_composition.csv.

Three checks are performed for every run_id:

  1. Run-ID presence  — flag run_ids that appear in only one of the two files.
  2. Iteration match  — for each run_id present in both files, verify that the
                        set of iterations recorded in summary.csv (column
                        ``iteration``) matches those in pool_composition.csv
                        (column ``itr``).
  3. Pool-size check  — for every (run_id, itr) row in pool_composition.csv,
                        verify that the number of entries in
                        ``pool_members_uuids`` equals the ``particle_pool_size``
                        declared in the Hydra config stored in summary.csv.

Usage:
    python diagnostics/check_pool_summary_consistency.py <folder>
    python diagnostics/check_pool_summary_consistency.py <folder> --verbose
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    LINE_WIDTH,
    check_iteration_match,
    check_run_id_presence,
    get_pool_size_from_config,
    ok,
    parse_uuid_list,
    print_section,
    warn,
)

OTHER_FILENAME = "pool_composition.csv"


# ---------------------------------------------------------------------------
# Pool-specific check
# ---------------------------------------------------------------------------


def check_pool_sizes(
    summary: pd.DataFrame,
    pool: pd.DataFrame,
    common_run_ids: set[str],
    verbose: bool,
) -> bool:
    """Check that len(pool_members_uuids) == particle_pool_size for every row."""
    run_id_to_expected: dict[str, int | None] = {}
    for rid in common_run_ids:
        row = summary.loc[summary["run_id"] == rid].iloc[0]
        try:
            config = ast.literal_eval(row["config"])
            run_id_to_expected[rid] = get_pool_size_from_config(config)
        except Exception:
            run_id_to_expected[rid] = None

    all_ok = True
    for rid in sorted(common_run_ids):
        expected = run_id_to_expected.get(rid)
        if expected is None:
            print(
                warn(
                    f"{rid}  —  could not read particle_pool_size from config; "
                    "skipping."
                )
            )
            all_ok = False
            continue

        run_pool_rows = pool.loc[pool["run_id"] == rid]
        bad_rows: list[tuple[int, int]] = []

        for _, row in run_pool_rows.iterrows():
            itr = int(row["itr"])
            try:
                actual = len(parse_uuid_list(row["pool_members_uuids"]))
            except Exception:
                actual = -1
            if actual != expected:
                bad_rows.append((itr, actual))

        if bad_rows:
            all_ok = False
            print(
                warn(
                    f"{rid}  —  expected pool_size={expected}, "
                    f"but {len(bad_rows)} iteration(s) deviate:"
                )
            )
            for itr, actual in sorted(bad_rows):
                print(f"       itr={itr}  actual={actual}")
        elif verbose:
            print(
                ok(
                    f"{rid}  —  pool_size={expected} correct at all "
                    f"{len(run_pool_rows)} iteration(s)."
                )
            )

    if all_ok and not verbose:
        print(ok(f"Pool size matches config for all {len(common_run_ids)} run_id(s)."))

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_consistency(folder: Path, verbose: bool = False) -> None:
    summary_path = folder / "summary.csv"
    pool_path = folder / OTHER_FILENAME
    for p in (summary_path, pool_path):
        if not p.exists():
            raise SystemExit(f"ERROR: {p} not found.")

    summary = pd.read_csv(summary_path)
    pool = pd.read_csv(pool_path)

    for col in ("run_id", "iteration", "config"):
        if col not in summary.columns:
            raise SystemExit(f"ERROR: column '{col}' not found in summary.csv.")
    for col in ("run_id", "itr", "pool_members_uuids"):
        if col not in pool.columns:
            raise SystemExit(f"ERROR: column '{col}' not found in {OTHER_FILENAME}.")

    summary["run_id"] = summary["run_id"].astype(str)
    pool["run_id"] = pool["run_id"].astype(str)

    summary_run_ids = set(summary["run_id"].unique())
    pool_run_ids = set(pool["run_id"].unique())
    common_run_ids = summary_run_ids & pool_run_ids

    print("=" * LINE_WIDTH)
    print(f"  Folder : {folder}")
    print(
        f"  summary.csv          : {len(summary)} rows, {len(summary_run_ids)} "
        "run_id(s)"
    )
    print(f"  {OTHER_FILENAME} : {len(pool)} rows, {len(pool_run_ids)} run_id(s)")
    print(f"  Common run_ids       : {len(common_run_ids)}")
    print("=" * LINE_WIDTH)

    print_section("CHECK 1 / 3  —  Run-ID presence")
    ok1 = check_run_id_presence(summary_run_ids, pool_run_ids, OTHER_FILENAME)

    print_section("CHECK 2 / 3  —  Iteration match")
    if common_run_ids:
        ok2 = check_iteration_match(
            summary, pool, common_run_ids, verbose, OTHER_FILENAME
        )
    else:
        print("  (skipped — no common run_ids)")
        ok2 = False

    print_section("CHECK 3 / 3  —  Pool size vs config")
    if common_run_ids:
        ok3 = check_pool_sizes(summary, pool, common_run_ids, verbose)
    else:
        print("  (skipped — no common run_ids)")
        ok3 = False

    print(f"\n{'=' * LINE_WIDTH}")
    results = [
        ("Run-ID presence", ok1),
        ("Iteration match", ok2),
        ("Pool-size check", ok3),
    ]
    for label, passed in results:
        print(f"  {'[ok]' if passed else '[!] '}  {label}")
    print()
    if all(p for _, p in results):
        print("  All checks passed.")
    else:
        print("  One or more checks FAILED — see details above.")
    print("=" * LINE_WIDTH + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Check consistency between summary.csv and pool_composition.csv "
            "in a results folder."
        )
    )
    parser.add_argument(
        "folder", type=Path, help="Folder containing the two CSV files."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print a per-run-id [ok] line even when checks pass.",
    )
    args = parser.parse_args()
    check_consistency(args.folder.resolve(), verbose=args.verbose)
