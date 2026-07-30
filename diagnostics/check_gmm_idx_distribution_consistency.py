"""
Cross-check consistency between summary.csv and GMM_idx_distribution.csv.

Four checks are performed for every run_id:

  1. Run-ID presence      — flag run_ids that appear in only one of the two files.
  2. Duplicate rows       — check for duplicate (run_id, itr) pairs in
                            GMM_idx_distribution.csv.
  3. Iteration match      — for each run_id present in both files, verify that
                            the set of iterations in summary.csv (column
                            ``iteration``) matches those in GMM_idx_distribution.csv
                            (column ``itr``).
  4. Count-sum vs config  — for every (run_id, itr) row in GMM_idx_distribution.csv,
                            verify that the sum of all ``counts(idx=*)`` columns
                            equals the ``particle_pool_size`` declared in the Hydra
                            config stored in summary.csv.

Usage:
    python diagnostics/check_gmm_idx_distribution_consistency.py <folder>
    python diagnostics/check_gmm_idx_distribution_consistency.py <folder> --verbose
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
    print_section,
    warn,
)

OTHER_FILENAME = "GMM_idx_distribution.csv"


# ---------------------------------------------------------------------------
# GMM-specific checks
# ---------------------------------------------------------------------------


def check_duplicate_rows(gmm: pd.DataFrame) -> bool:
    """Check for duplicate (run_id, itr) pairs in *gmm*."""
    dups = gmm[gmm.duplicated(subset=["run_id", "itr"], keep=False)]
    if dups.empty:
        print(ok(f"No duplicate (run_id, itr) pairs in {OTHER_FILENAME}."))
        return True

    print(warn(f"{len(dups)} duplicate (run_id, itr) rows found in {OTHER_FILENAME}:"))
    for _, row in dups.sort_values(["run_id", "itr"]).iterrows():
        print(f"       run_id={row['run_id']}  itr={row['itr']}")
    return False


def check_count_sums(
    summary: pd.DataFrame,
    gmm: pd.DataFrame,
    common_run_ids: set[str],
    verbose: bool,
) -> bool:
    """Check that sum of counts(idx=*) columns equals particle_pool_size per row."""
    count_cols = [c for c in gmm.columns if c.startswith("counts(idx=")]
    if not count_cols:
        print(
            warn(f"No 'counts(idx=...)' columns found in {OTHER_FILENAME}; skipping.")
        )
        return False

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

        run_gmm_rows = gmm.loc[gmm["run_id"] == rid]
        bad_rows: list[tuple[int, int]] = []

        for _, row in run_gmm_rows.iterrows():
            itr = int(row["itr"])
            actual = int(row[count_cols].sum())
            if actual != expected:
                bad_rows.append((itr, actual))

        if bad_rows:
            all_ok = False
            print(
                warn(
                    f"{rid}  —  expected count_sum={expected}, "
                    f"but {len(bad_rows)} iteration(s) deviate:"
                )
            )
            for itr, actual in sorted(bad_rows):
                print(f"       itr={itr}  actual_sum={actual}")
        elif verbose:
            print(
                ok(
                    f"{rid}  —  count_sum={expected} correct at all {len(run_gmm_rows)}"
                    "iteration(s)."
                )
            )

    if all_ok and not verbose:
        print(
            ok(f"Count sums match pool_size for all {len(common_run_ids)} run_id(s).")
        )

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_gmm_consistency(folder: Path, verbose: bool = False) -> None:
    summary_path = folder / "summary.csv"
    gmm_path = folder / OTHER_FILENAME
    for p in (summary_path, gmm_path):
        if not p.exists():
            raise SystemExit(f"ERROR: {p} not found.")

    summary = pd.read_csv(summary_path)
    gmm = pd.read_csv(gmm_path)

    for col in ("run_id", "iteration", "config"):
        if col not in summary.columns:
            raise SystemExit(f"ERROR: column '{col}' not found in summary.csv.")
    for col in ("run_id", "itr"):
        if col not in gmm.columns:
            raise SystemExit(f"ERROR: column '{col}' not found in {OTHER_FILENAME}.")

    summary["run_id"] = summary["run_id"].astype(str)
    gmm["run_id"] = gmm["run_id"].astype(str)

    summary_run_ids = set(summary["run_id"].unique())
    gmm_run_ids = set(gmm["run_id"].unique())
    common_run_ids = summary_run_ids & gmm_run_ids
    count_cols = [c for c in gmm.columns if c.startswith("counts(idx=")]

    print("=" * LINE_WIDTH)
    print(f"  Folder : {folder}")
    print(
        f"  summary.csv              : {len(summary)} rows, {len(summary_run_ids)} "
        "run_id(s)"
    )
    print(
        f"  {OTHER_FILENAME} : {len(gmm)} rows, {len(gmm_run_ids)} run_id(s), "
        f"{len(count_cols)} GMM index column(s)"
    )
    print(f"  Common run_ids           : {len(common_run_ids)}")
    print("=" * LINE_WIDTH)

    print_section("CHECK 1 / 4  —  Run-ID presence")
    ok1 = check_run_id_presence(summary_run_ids, gmm_run_ids, OTHER_FILENAME)

    print_section("CHECK 2 / 4  —  Duplicate (run_id, itr) rows")
    ok2 = check_duplicate_rows(gmm)

    print_section("CHECK 3 / 4  —  Iteration match")
    if common_run_ids:
        ok3 = check_iteration_match(
            summary, gmm, common_run_ids, verbose, OTHER_FILENAME
        )
    else:
        print("  (skipped — no common run_ids)")
        ok3 = False

    print_section("CHECK 4 / 4  —  Count sum vs pool_size")
    if common_run_ids:
        ok4 = check_count_sums(summary, gmm, common_run_ids, verbose)
    else:
        print("  (skipped — no common run_ids)")
        ok4 = False

    print(f"\n{'=' * LINE_WIDTH}")
    results = [
        ("Run-ID presence", ok1),
        ("Duplicate rows", ok2),
        ("Iteration match", ok3),
        ("Count sum vs pool_size", ok4),
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
            "Check consistency between summary.csv and GMM_idx_distribution.csv "
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
    check_gmm_consistency(args.folder.resolve(), verbose=args.verbose)
