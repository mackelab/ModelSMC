"""
Remove the ``errors`` and ``warnings`` columns from the recorded ``summary.csv``
files.

The recorded errors are tracebacks from the cluster the experiments were run
on.  They embed absolute paths that expose local user names, so the column is stripped
before publication.  ``warnings`` holds runtime messages and is dropped along
with it for consistency.

Walks the folder recursively and rewrites every ``summary*.csv`` carrying one
of the columns **in place** — unlike the other curation scripts there is no
``_filtered`` variant, since leaving the original behind would defeat the
purpose.  Nothing is written unless ``--apply`` is passed.

Usage:
    python curation/remove_error_column.py                    # dry run, main_results/
    python curation/remove_error_column.py --apply            # rewrite in place
    python curation/remove_error_column.py <folder> --apply   # a different folder
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

# Re-use LINE_WIDTH from the diagnostics utilities if available; fall back to
# a local constant so this script can also be run standalone.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "diagnostics"))
    from utils import LINE_WIDTH
except ImportError:
    LINE_WIDTH = 76

# The ``config`` column holds a serialised Hydra config and easily exceeds the
# default 128 KiB field limit, as do the tracebacks themselves.
csv.field_size_limit(10**9)

DEFAULT_FOLDER = Path(__file__).resolve().parent.parent / "main_results"
FILE_PATTERN = "summary*.csv"
DEFAULT_COLUMNS = ["errors", "warnings"]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def strip_columns(
    path: Path, columns: list[str], apply_changes: bool
) -> "dict | None":
    """Rewrite ``path`` without ``columns``.

    Returns a stats dict, or ``None`` if the file carries none of them.
    With ``apply_changes=False`` the file is only inspected, not written.
    """
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)

    if header is None:
        return None

    present = [c for c in columns if c in header]
    if not present:
        return None

    idx_of = {c: header.index(c) for c in present}
    drop = set(idx_of.values())
    size_before = path.stat().st_size

    out = None
    tmp_path = None
    if apply_changes:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        tmp_path = Path(tmp_name)
        out = os.fdopen(fd, "w", newline="", encoding="utf-8")

    n_rows = 0
    n_nonempty = {c: 0 for c in present}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header, already read above

            writer = csv.writer(out) if out is not None else None
            if writer is not None:
                writer.writerow([c for i, c in enumerate(header) if i not in drop])

            for row in reader:
                n_rows += 1
                for c, i in idx_of.items():
                    if i < len(row) and row[i].strip():
                        n_nonempty[c] += 1
                if writer is not None:
                    writer.writerow([v for i, v in enumerate(row) if i not in drop])
    except Exception:
        if out is not None:
            out.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise

    size_after = size_before
    if apply_changes:
        out.close()
        size_after = tmp_path.stat().st_size
        tmp_path.replace(path)  # atomic on the same filesystem

    return {
        "removed": present,
        "rows": n_rows,
        "nonempty": n_nonempty,
        "size_before": size_before,
        "size_after": size_after,
    }


def remove_error_column(
    folder: Path,
    columns: "list[str] | None" = None,
    apply_changes: bool = False,
) -> None:
    columns = list(DEFAULT_COLUMNS if columns is None else columns)
    files = sorted(folder.rglob(FILE_PATTERN))

    print("=" * LINE_WIDTH)
    print(f"  Folder          : {folder}")
    print(f"  Pattern         : {FILE_PATTERN}")
    print(f"  Columns         : {', '.join(columns)}")
    print(f"  Mode            : {'APPLY (in place)' if apply_changes else 'dry run'}")
    print(f"  Files found     : {len(files)}")
    print("=" * LINE_WIDTH)

    n_stripped = 0
    n_clean = 0
    bytes_saved = 0

    for path in files:
        rel = path.relative_to(folder)
        stats = strip_columns(path, columns, apply_changes)

        if stats is None:
            n_clean += 1
            print(f"\n  [skip] {rel}")
            print(f"    {'Columns present':<28}: none")
            continue

        n_stripped += 1
        bytes_saved += stats["size_before"] - stats["size_after"]

        print(f"\n  {rel}")
        print(f"    {'Columns present':<28}: {', '.join(stats['removed'])}")
        print(f"    {'Rows':<28}: {stats['rows']}")
        for col, n in stats["nonempty"].items():
            print(f"    {'Rows with a ' + col + ' entry':<28}: {n}")
        if apply_changes:
            print(
                f"    {'Size':<28}: {stats['size_before'] / 1e6:.2f} MB"
                f" -> {stats['size_after'] / 1e6:.2f} MB"
            )
            print(f"    {'Status':<28}: removed")
        else:
            print(f"    {'Size':<28}: {stats['size_before'] / 1e6:.2f} MB")
            print(f"    {'Status':<28}: would remove")

    label = "/".join(f"'{c}'" for c in columns)
    print(f"\n{'─' * LINE_WIDTH}")
    if not files:
        print(f"  [!]  No files matching '{FILE_PATTERN}' found under {folder}.")
    elif n_stripped == 0:
        print(f"  [ok] No file carries a {label} column — nothing to do.")
    elif apply_changes:
        print(f"  [ok] Cleaned {n_stripped} file(s); {n_clean} already clean.")
        print(f"       Freed {bytes_saved / 1e6:.2f} MB.")
    else:
        print(f"  [!]  {n_stripped} file(s) carry a {label} column; ", end="")
        print(f"{n_clean} already clean.")
        print("       Nothing was written. Re-run with --apply to rewrite them.")
    print("=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            f"Remove the {'/'.join(DEFAULT_COLUMNS)} columns from every "
            f"'{FILE_PATTERN}' found recursively under a results folder. "
            "The files are rewritten in place; pass --apply to do so."
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
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        metavar="NAME",
        help=f"Columns to remove (default: {' '.join(DEFAULT_COLUMNS)}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite the files (default: dry run, nothing is written).",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"ERROR: '{folder}' is not a directory.")

    remove_error_column(folder, columns=args.columns, apply_changes=args.apply)
