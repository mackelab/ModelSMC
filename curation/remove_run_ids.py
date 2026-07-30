"""
Remove specified run_ids from the recorded CSV files in a results folder.

Affected files (if present):
  - summary.csv
  - pool_composition.csv
  - GMM_idx_distribution.csv   (only produced for the minimal-example task)

The originals are never modified.  Each cleaned file is written next to the
original with a ``_filtered`` suffix, e.g. ``summary_filtered.csv``.

By default, if a ``_filtered`` file already exists it is used as the input
instead of the original, so removals can be applied incrementally across
multiple invocations.  Pass ``--no-incremental`` to always read from the
originals regardless.

Usage:
    python curation/remove_run_ids.py <folder> --run-ids <id1> <id2> ...

Example — two incremental passes:
    python curation/remove_run_ids.py results/my_exp --run-ids abc123 def456
    python curation/remove_run_ids.py results/my_exp --run-ids ghi789
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Re-use LINE_WIDTH from the diagnostics utilities if available; fall back to
# a local constant so this script can also be run standalone.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "diagnostics"))
    from utils import LINE_WIDTH
except ImportError:
    LINE_WIDTH = 76

OUTPUT_SUFFIX = "_filtered"

# Each entry: (filename, run_id column name)
TRACKED_FILES: list[tuple[str, str]] = [
    ("summary.csv", "run_id"),
    ("pool_composition.csv", "run_id"),
    ("GMM_idx_distribution.csv", "run_id"),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def remove_run_ids(
    folder: Path,
    run_ids_to_remove: list[str],
    incremental: bool = True,
) -> None:
    target_ids: set[str] = set(run_ids_to_remove)

    print("=" * LINE_WIDTH)
    print(f"  Folder          : {folder}")
    print(f"  Incremental     : {incremental}")
    print(f"  Run IDs to remove ({len(target_ids)}):")
    for rid in sorted(target_ids):
        print(f"      {rid}")
    print("=" * LINE_WIDTH)

    # Track which requested IDs were found in at least one file
    ids_seen_anywhere: set[str] = set()

    for filename, id_col in TRACKED_FILES:
        original_path = folder / filename
        out_path = (
            folder / f"{Path(filename).stem}{OUTPUT_SUFFIX}{Path(filename).suffix}"
        )

        # Decide which file to read from
        if incremental and out_path.exists():
            read_path = out_path
            read_label = out_path.name
        elif original_path.exists():
            read_path = original_path
            read_label = original_path.name
        else:
            print(f"\n  [skip] {filename} — file not found.")
            continue

        df = pd.read_csv(read_path)
        df[id_col] = df[id_col].astype(str)

        all_ids_in_file: set[str] = set(df[id_col].unique())
        found_in_file: set[str] = target_ids & all_ids_in_file
        not_found_in_file: set[str] = target_ids - all_ids_in_file
        ids_seen_anywhere |= found_in_file

        n_before = len(df)
        df_clean = df[~df[id_col].isin(target_ids)].copy()
        n_removed = n_before - len(df_clean)
        n_after = len(df_clean)

        df_clean.to_csv(out_path, index=False)

        print(f"\n  {filename}")
        print(f"    {'Read from':<28}: {read_label}")
        print(
            f"    {'Run IDs found in file':<28}: {len(found_in_file)} / {len(target_ids)}" # noqa
        )
        if not_found_in_file:
            print(f"    {'Run IDs not in file':<28}: {sorted(not_found_in_file)}")
        print(f"    {'Rows before':<28}: {n_before}")
        print(f"    {'Rows removed':<28}: {n_removed}")
        print(f"    {'Rows remaining':<28}: {n_after}")
        print(f"    {'Saved to':<28}: {out_path.name}")

    # Warn about IDs not found in any file
    ids_not_found_anywhere = target_ids - ids_seen_anywhere
    print(f"\n{'─' * LINE_WIDTH}")
    if ids_not_found_anywhere:
        print(
            f"  [!]  {len(ids_not_found_anywhere)} run_id(s) were not found in any file:" # noqa
        )
        for rid in sorted(ids_not_found_anywhere):
            print(f"       {rid}")
    else:
        print(
            f"  [ok] All {len(target_ids)} requested run_id(s) were found in at least "
            "one file."
        )
    print("=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Remove run_ids from summary.csv, pool_composition.csv, and "
            "GMM_idx_distribution.csv, writing cleaned copies with a "
            f"'{OUTPUT_SUFFIX}' suffix."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Results folder containing the CSV files.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        metavar="RUN_ID",
        help="One or more run_ids to remove.",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        default=True,
        help=(
            "Always read from the original files, ignoring any existing "
            f"'{OUTPUT_SUFFIX}' files (default: read from filtered files if present)."
        ),
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"ERROR: '{folder}' is not a directory.")

    remove_run_ids(folder, args.run_ids, incremental=args.incremental)
