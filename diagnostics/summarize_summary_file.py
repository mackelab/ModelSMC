"""
Summarize a summary.csv file by grouping runs that share the same Hydra
configuration (modulo ``seed`` and ``run_id``) and printing per-group
diagnostics to the console.

For each group the following is reported:
  - The value of every config key that differs between groups
  - A table of run_ids with their random seed, iteration counts, unique
    particle count, and percentage of valid (finite log-weight) particles
    (plus, when present in summary.csv, the percentage of particles that
    timed out during evidence estimation and the total/prompt/completion
    token usage at the final iteration)
  - A sanity check flagging any duplicate seeds within a group

Usage:
    python diagnostics/summarize_summary_file.py <folder>
    python diagnostics/summarize_summary_file.py <folder> --no-check
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    LINE_WIDTH,
    find_differing_paths,
    get_nested,
    make_hashable,
    strip_config,
)

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def summarize(folder: Path, sanity_check: bool = True) -> None:
    csv_path = folder / "summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"ERROR: summary.csv not found in {folder}")

    df = pd.read_csv(csv_path)

    for col in ("config", "run_id", "iteration"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: column '{col}' not found in summary.csv.")

    # Parse config strings once
    configs: list[dict] = [ast.literal_eval(row) for row in df["config"]]
    stripped: list[dict] = [strip_config(c) for c in configs]

    # ── Group rows by stripped config ────────────────────────────────────────
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, s in enumerate(stripped):
        groups[make_hashable(s)].append(idx)
    groups = dict(groups)
    n_groups = len(groups)

    # Representative stripped config per group (from the first row)
    group_rep_configs = [stripped[idxs[0]] for idxs in groups.values()]
    differing_paths = find_differing_paths(group_rep_configs)

    # ── Overall header ───────────────────────────────────────────────────────
    print("=" * LINE_WIDTH)
    print(f"  File   : {csv_path}")
    print(f"  Rows   : {len(df)}     Groups: {n_groups}")
    if differing_paths:
        print(f"  Groups differ in: {', '.join(differing_paths)}")
    else:
        print("  All groups share an identical config (only seed / run_id differ).")
    print("=" * LINE_WIDTH)

    has_log_weight = "log_weight" in df.columns
    has_particle_id = "particle_id" in df.columns
    has_timeout = "timeout_evidence_estimation" in df.columns
    token_cols = [
        c
        for c in ("total_tokens", "prompt_tokens", "completion_tokens")
        if c in df.columns
    ]
    has_tokens = len(token_cols) > 0
    any_sanity_failure = False

    for g_num, (_key, indices) in enumerate(groups.items(), start=1):
        rows = df.iloc[indices]
        rep_stripped = stripped[indices[0]]

        print(f"\n{'─' * LINE_WIDTH}")
        print(f"  GROUP {g_num} / {n_groups}")

        # ── Config values for differing paths ────────────────────────────────
        if differing_paths:
            print("\n  Config (differing entries):")
            for path in differing_paths:
                val = get_nested(rep_stripped, path)
                print(f"    {path} = {val!r}")

        # ── Build per-run-id info ─────────────────────────────────────────────
        run_id_to_seed: dict[str, object] = {}
        for i in indices:
            rid = str(df.iloc[i]["run_id"])
            if rid not in run_id_to_seed:
                run_id_to_seed[rid] = configs[i].get("seed", "<missing>")

        run_id_to_iters: dict[str, list[int]] = defaultdict(list)
        for _, row in rows.iterrows():
            run_id_to_iters[str(row["run_id"])].append(int(row["iteration"]))

        # ── Run table ─────────────────────────────────────────────────────────
        print(f"\n  Runs ({len(run_id_to_seed)}):")
        col_w = max((len(str(r)) for r in run_id_to_seed), default=6)
        col_w = max(col_w, len("run_id"))
        header = (
            f"    {'run_id':<{col_w}}  {'seed':>10}  {'n_itr':>6}"
            f"  {'max_itr':>8}  {'n_particles':>11}"
        )
        if has_log_weight:
            header += f"  {'valid_%':>7}"
        if has_timeout:
            header += f"  {'timeout_%':>9}"
        if has_tokens:
            header += f"  {'total_tok':>10}  {'prompt_tok':>10}  {'compl_tok':>10}"
        print(header)
        sep = f"    {'─' * col_w}  {'─' * 10}  {'─' * 6}  {'─' * 8}  {'─' * 11}"
        if has_log_weight:
            sep += f"  {'─' * 7}"
        if has_timeout:
            sep += f"  {'─' * 9}"
        if has_tokens:
            sep += f"  {'─' * 10}  {'─' * 10}  {'─' * 10}"
        print(sep)

        for run_id in sorted(run_id_to_seed, key=lambda r: run_id_to_seed[r]):
            run_rows = rows[rows["run_id"].astype(str) == run_id]
            iters = run_id_to_iters[run_id]
            n_itr = len(set(iters))
            max_itr = max(iters)
            seed = run_id_to_seed[run_id]
            n_unique = run_rows["particle_id"].nunique() if has_particle_id else "n/a"
            line = (
                f"    {run_id:<{col_w}}  {str(seed):>10}  {n_itr:>6}"
                f"  {max_itr:>8}  {str(n_unique):>11}"
            )
            if has_log_weight:
                lw = pd.to_numeric(run_rows["log_weight"], errors="coerce").to_numpy(
                    dtype=float
                )
                n_total = len(lw)
                n_valid = int(np.isfinite(lw).sum())
                pct = 100.0 * n_valid / n_total if n_total > 0 else float("nan")
                line += f"  {pct:>6.1f}%"
            if has_timeout:
                t = run_rows["timeout_evidence_estimation"]
                n_total_t = len(t)
                n_timeout = int(t.astype(str).str.lower().eq("true").sum())
                pct_t = 100.0 * n_timeout / n_total_t if n_total_t > 0 else float("nan")
                line += f"  {pct_t:>8.1f}%"
            if has_tokens:
                # Use the unique value across all rows at the final iteration
                final_rows = run_rows[run_rows["iteration"] == max_itr]

                def _unique_token_val(col: str) -> str:
                    if col not in token_cols:
                        return "n/a"
                    vals = (
                        pd.to_numeric(final_rows[col], errors="coerce")
                        .dropna()
                        .unique()
                    )
                    if len(vals) == 1:
                        return f"{int(vals[0]):,}"
                    return "n/u"  # not unique

                tok_total_str = _unique_token_val("total_tokens")
                tok_prompt_str = _unique_token_val("prompt_tokens")
                tok_compl_str = _unique_token_val("completion_tokens")
                line += (
                    f"  {tok_total_str:>10}  {tok_prompt_str:>10}  {tok_compl_str:>10}"
                )

            print(line)

        # ── Sanity check: duplicate seeds within group ────────────────────────
        if sanity_check:
            seed_to_rids: dict[object, list[str]] = defaultdict(list)
            for rid, seed in run_id_to_seed.items():
                seed_to_rids[seed].append(rid)

            dups = {s: rids for s, rids in seed_to_rids.items() if len(rids) > 1}
            if dups:
                any_sanity_failure = True
                print("\n  [!] SANITY CHECK FAILED: duplicate seeds within this group!")
                for seed in sorted(dups, key=str):
                    rids = dups[seed]
                    print(f"      seed={seed}  →  run_ids: {rids}")
            else:
                print("\n  [ok] All seeds are unique within this group.")

    print(f"\n{'=' * LINE_WIDTH}")
    if sanity_check and any_sanity_failure:
        print("  [!] One or more groups failed the seed-uniqueness sanity check.")
    elif sanity_check:
        print("  [ok] All sanity checks passed.")
    print("=" * LINE_WIDTH + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group runs in summary.csv by config (excl. seed & run_id) "
            "and print per-group diagnostics."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing summary.csv.")
    parser.add_argument(
        "--no-check",
        dest="sanity_check",
        action="store_false",
        default=True,
        help="Skip the seed-uniqueness sanity check.",
    )
    args = parser.parse_args()
    summarize(args.folder.resolve(), sanity_check=args.sanity_check)


if __name__ == "__main__":
    main()
