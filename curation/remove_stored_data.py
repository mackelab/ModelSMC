"""
Remove the stored ``training_data.pt`` / ``validation_data.pt`` from a results folder.

Every saved particle folder carries a verbatim copy of the data the run was fitted
against, so these files dominate the size of the repository.

Only ``posthoc_parameter_estimation`` and ``reevaluate_numsim.py`` read them, and both
need the task's data anyway; no figure, table or analysis does.  ``parameter_estimates.pt``
holds inferred parameters, not observations, and is left untouched.

Nothing is deleted unless ``--apply`` is passed.

Usage:
    python curation/remove_stored_data.py                     # dry run, main_results/
    python curation/remove_stored_data.py --apply             # delete
    python curation/remove_stored_data.py <folder> --apply    # a different folder
"""

import argparse
import sys
from pathlib import Path

# Re-use LINE_WIDTH from the diagnostics utilities if available; fall back to
# a local constant so this script can also be run standalone.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "diagnostics"))
    from utils import LINE_WIDTH
except ImportError:
    LINE_WIDTH = 76

DEFAULT_FOLDER = Path(__file__).resolve().parent.parent / "main_results"
DEFAULT_NAMES = ["training_data.pt", "validation_data.pt"]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def remove_stored_data(
    folder: Path,
    names: "list[str] | None" = None,
    apply_changes: bool = False,
) -> None:
    names = list(DEFAULT_NAMES if names is None else names)

    files: list[Path] = []
    for name in names:
        files.extend(folder.rglob(name))
    files = sorted(set(files))

    print("=" * LINE_WIDTH)
    print(f"  Folder          : {folder}")
    print(f"  File names      : {', '.join(names)}")
    print(f"  Mode            : {'APPLY (deletes)' if apply_changes else 'dry run'}")
    print(f"  Files found     : {len(files)}")
    print("=" * LINE_WIDTH)

    bytes_freed = 0
    per_group: dict[str, list[int]] = {}

    for path in files:
        size = path.stat().st_size
        bytes_freed += size
        rel = path.relative_to(folder)

        # Group by the top-level folder under the results root, for the summary.
        group = rel.parts[0] if len(rel.parts) > 1 else "."
        per_group.setdefault(group, []).append(size)

        if apply_changes:
            path.unlink()

        verb = "deleted" if apply_changes else "would delete"
        print(f"  [{verb}] {rel}  ({size / 1e6:.1f} MB)")

    if per_group:
        print(f"\n{'─' * LINE_WIDTH}")
        print("  Per top-level folder:")
        for group in sorted(per_group):
            sizes = per_group[group]
            print(
                f"    {group:<24} {len(sizes):>4} file(s)"
                f"   {sum(sizes) / 1e6:>8.1f} MB"
            )

    print(f"\n{'─' * LINE_WIDTH}")
    if not files:
        print(f"  [ok] No {' / '.join(names)} found under {folder} — nothing to do.")
    elif apply_changes:
        print(f"  [ok] Deleted {len(files)} file(s), freeing {bytes_freed / 1e6:.1f} MB.")
    else:
        print(f"  [!]  {len(files)} file(s) would be deleted", end="")
        print(f" ({bytes_freed / 1e6:.1f} MB).")
        print("       Nothing was deleted. Re-run with --apply to remove them.")
    print("=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f"Delete every {' and '.join(DEFAULT_NAMES)} found recursively under a "
            "results folder. Pass --apply to actually delete."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=DEFAULT_FOLDER,
        help=f"Results folder to walk (default: {DEFAULT_FOLDER}).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_NAMES,
        metavar="NAME",
        help=f"File names to remove (default: {' '.join(DEFAULT_NAMES)}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the files (default: dry run, nothing is deleted).",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"ERROR: '{folder}' is not a directory.")

    remove_stored_data(folder, names=args.names, apply_changes=args.apply)
