"""
Remove the ``.hydra/hydra.yaml`` snapshots from a results folder.

``hydra.yaml`` is Hydra's runtime record — ``cwd``, ``output_dir`` and the job
environment — so it embeds absolute paths that expose local user names.
Nothing reads it, so it is deleted rather than scrubbed.  ``config.yaml`` and
``overrides.yaml`` are kept: the first is needed for posthoc parameter
estimation, the second records how the run was launched, and neither contains
paths.

A ``.hydra/`` folder left empty by the deletion is removed too.  Nothing is
deleted unless ``--apply`` is passed.

Usage:
    python curation/remove_hydra_yaml.py                    # dry run, main_results/
    python curation/remove_hydra_yaml.py --apply            # delete
    python curation/remove_hydra_yaml.py <folder> --apply   # a different folder
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
FILE_PATTERN = ".hydra/hydra.yaml"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def remove_hydra_yaml(folder: Path, apply_changes: bool = False) -> None:
    files = sorted(folder.rglob(FILE_PATTERN))

    print("=" * LINE_WIDTH)
    print(f"  Folder          : {folder}")
    print(f"  Pattern         : {FILE_PATTERN}")
    print(f"  Mode            : {'APPLY (deletes)' if apply_changes else 'dry run'}")
    print(f"  Files found     : {len(files)}")
    print("=" * LINE_WIDTH)

    bytes_freed = 0
    n_dirs_removed = 0

    for path in files:
        size = path.stat().st_size
        bytes_freed += size
        rel = path.relative_to(folder)

        note = ""
        if apply_changes:
            path.unlink()
            # Drop the .hydra folder too if hydra.yaml was its only content.
            parent = path.parent
            if not any(parent.iterdir()):
                parent.rmdir()
                n_dirs_removed += 1
                note = "  (.hydra removed, was empty)"

        verb = "deleted" if apply_changes else "would delete"
        print(f"  [{verb}] {rel}  ({size / 1e3:.1f} kB){note}")

    print(f"\n{'─' * LINE_WIDTH}")
    if not files:
        print(f"  [ok] No '{FILE_PATTERN}' found under {folder} — nothing to do.")
    elif apply_changes:
        print(f"  [ok] Deleted {len(files)} file(s), freeing {bytes_freed / 1e6:.2f} MB.")
        if n_dirs_removed:
            print(f"       Also removed {n_dirs_removed} empty '.hydra' folder(s).")
    else:
        print(f"  [!]  {len(files)} file(s) would be deleted", end="")
        print(f" ({bytes_freed / 1e6:.2f} MB).")
        print("       Nothing was deleted. Re-run with --apply to remove them.")
    print("=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f"Delete every '{FILE_PATTERN}' found recursively under a results "
            "folder, keeping config.yaml and overrides.yaml. Pass --apply to "
            "actually delete."
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
        "--apply",
        action="store_true",
        help="Actually delete the files (default: dry run, nothing is deleted).",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"ERROR: '{folder}' is not a directory.")

    remove_hydra_yaml(folder, apply_changes=args.apply)
