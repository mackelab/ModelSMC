"""
Shared utilities for the diagnostics scripts.
"""

import ast

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_WIDTH = 76

# Top-level config keys excluded when grouping runs (they are expected to
# differ across runs within the same experimental configuration).
EXCLUDED_CONFIG_KEYS: frozenset[str] = frozenset({"seed", "run_id"})


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------


def ok(msg: str) -> str:
    """Return a formatted [ok] status line."""
    return f"  [ok] {msg}"


def warn(msg: str) -> str:
    """Return a formatted [!] warning line."""
    return f"  [!]  {msg}"


def print_section(title: str) -> None:
    """Print a section separator with *title*."""
    print(f"\n{'─' * LINE_WIDTH}")
    print(f"  {title}")
    print(f"{'─' * LINE_WIDTH}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_pool_size_from_config(config: dict) -> int | None:
    """Extract ``particle_pool_size`` from a parsed Hydra config dict.

    Returns ``None`` if the key cannot be found.
    """
    method = config.get("method")
    if isinstance(method, dict):
        val = method.get("particle_pool_size")
        if val is not None:
            return int(val)
    return None


def parse_uuid_list(raw: str) -> list[str]:
    """Parse a ``pool_members_uuids`` cell back to a Python list."""
    return ast.literal_eval(raw)


def strip_config(config: dict) -> dict:
    """Return a copy of *config* without the :data:`EXCLUDED_CONFIG_KEYS`."""
    return {k: v for k, v in config.items() if k not in EXCLUDED_CONFIG_KEYS}


def make_hashable(obj) -> object:
    """Recursively convert dicts / lists to a hashable representation."""
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(v) for v in obj)
    return obj


def find_differing_paths(configs: list[dict], _prefix: str = "") -> list[str]:
    """Return sorted dot-notation paths of values that differ across *configs*.

    Recurses into nested dicts so that e.g. ``task.name`` is reported instead
    of the entire ``task`` sub-dict.
    """
    if len(configs) <= 1:
        return []
    all_keys: set[str] = set().union(*configs)
    paths: list[str] = []
    for k in sorted(all_keys):
        full_path = f"{_prefix}.{k}" if _prefix else k
        vals = [cfg.get(k) for cfg in configs]
        non_none = [v for v in vals if v is not None]
        if non_none and all(isinstance(v, dict) for v in non_none):
            sub_configs = [v if isinstance(v, dict) else {} for v in vals]
            sub_paths = find_differing_paths(sub_configs, _prefix=full_path)
            if sub_paths:
                paths.extend(sub_paths)
            elif len({make_hashable(v) for v in vals}) > 1:
                paths.append(full_path)
        elif len({make_hashable(v) for v in vals}) > 1:
            paths.append(full_path)
    return paths


def get_nested(config: dict, path: str) -> object:
    """Return the value at dot-notation *path* inside *config*."""
    val = config
    for k in path.split("."):
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return "<missing>"
    return val


# ---------------------------------------------------------------------------
# Shared consistency checks
# ---------------------------------------------------------------------------


def check_run_id_presence(
    summary_run_ids: set[str],
    other_run_ids: set[str],
    other_filename: str,
) -> bool:
    """Check that run_ids are consistent between summary.csv and *other_filename*.

    Prints a warning for run_ids that appear in only one of the two sets and
    returns ``True`` if both sets are identical.
    """
    only_summary = summary_run_ids - other_run_ids
    only_other = other_run_ids - summary_run_ids

    all_present = not only_summary and not only_other
    if all_present:
        print(ok("All run_ids are present in both files."))
    else:
        if only_summary:
            print(
                warn(
                    f"{len(only_summary)} run_id(s) in summary.csv but NOT in "
                    f"{other_filename}:"
                )
            )
            for rid in sorted(only_summary):
                print(f"       {rid}")
        if only_other:
            print(
                warn(
                    f"{len(only_other)} run_id(s) in {other_filename} but NOT in "
                    "summary.csv:"
                )
            )
            for rid in sorted(only_other):
                print(f"       {rid}")
    return all_present


def check_iteration_match(
    summary: pd.DataFrame,
    other_df: pd.DataFrame,
    common_run_ids: set[str],
    verbose: bool,
    other_label: str = "other file",
) -> bool:
    """Check that the iteration sets in summary.csv and *other_df* match per run_id.

    summary.csv is expected to use an ``iteration`` column; *other_df* an ``itr``
    column.  *other_label* is used in mismatch messages to identify the second
    file (e.g. ``"pool_composition.csv"`` or ``"GMM_idx_distribution.csv"``).

    Returns ``True`` if all run_ids have matching iteration sets.
    """
    all_ok = True
    mismatches: list[str] = []

    for rid in sorted(common_run_ids):
        s_itrs = set(summary.loc[summary["run_id"] == rid, "iteration"].unique())
        o_itrs = set(other_df.loc[other_df["run_id"] == rid, "itr"].unique())

        only_s = s_itrs - o_itrs
        only_o = o_itrs - s_itrs

        if only_s or only_o:
            all_ok = False
            parts = []
            if only_s:
                parts.append(f"only in summary: {sorted(only_s)}")
            if only_o:
                parts.append(f"only in {other_label}: {sorted(only_o)}")
            mismatches.append(f"{rid}  →  " + ";  ".join(parts))
        elif verbose:
            n = len(s_itrs)
            print(ok(f"{rid}  —  {n} iteration(s), max={max(s_itrs)}"))

    if all_ok:
        if not verbose:
            print(ok(f"Iteration sets match for all {len(common_run_ids)} run_id(s)."))
    else:
        print(warn(f"Iteration mismatch in {len(mismatches)} run_id(s):"))
        for line in mismatches:
            print(f"       {line}")

    return all_ok
