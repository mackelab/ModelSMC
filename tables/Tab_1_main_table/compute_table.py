"""
Build the main comparison table from a results folder.

The results folder is expected to have the structure:
    <folder>/<task_name>/<method_name>/<llm_name>/run_<i>/best_particle/best_particle_info.csv

Each run contributes one row to its cell via the metric value stored in
best_particle_info.csv. Cell statistics (median and optional bootstrap CI)
are aggregated across all runs of an experiment.

Usage (run from inside Tab_1_main_table/):
    python compute_table.py
    python compute_table.py --folder <path>
    python compute_table.py --folder <path> --ignore LLM-free_example other_folder
    python compute_table.py --folder <path> --with-ci
    python compute_table.py --folder <path> --with-ci --ci 95 --n-bootstrap 1000
    python compute_table.py --folder <path> --metric neg_log_marginal_NLE
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

metric_header = {
    "neg_log_marginal_NLE": "-\\log p(x_o|m) (\\downarrow)",
    "neg_log_likelihood_NLE": "-\\log p(x_o|\\hat\\theta, m) (\\downarrow)",
    "neg_avg_log_marginal_NLE": "-\\frac{1}{N}\\log p(x_o|m) (\\downarrow)",
    "neg_avg_log_likelihood_NLE": "-\\frac{1}{N}\\log p(x_o|\\hat\\theta, m) (\\downarrow)",  # noqa
    "mse": "\\text{MSE} (\\downarrow)",
}

method_labels = {
    "funsearch": "FunSearch+",
    "modelsmc": "ModelSMC",
    "modelsmcN1": "ModelSMC $N=1$",
}

llm_labels = {
    "gptmini": "GPT-5-mini",
    "sonnet": "Sonnet 4.6",
}

task_labels = {"allen": "HH", "SIR": "SIR", "kidney": "Kidney"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the main comparison table from a results folder structured as "
            "<folder>/<task>/<method>/<llm>/run_<i>/best_particle/best_particle_info.csv."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default="../../main_results",
        help="Root folder containing all results.",
    )
    parser.add_argument(
        "--ignore",
        dest="subfolder_to_ignore",
        nargs="*",
        default=[
            "LLM-free_example",
            "allen_ablation",
            "allen_LLM_history_recording",
            "allen_posterior_mass_analysis",
        ],
        metavar="SUBFOLDER",
        help=(
            "Top-level subfolders of <folder> to skip during evaluation. "
            "Defaults to ['LLM-free_example', 'allen_ablation', "
            "'allen_LLM_history_recording', 'allen_posterior_mass_analysis']."
        ),
    )
    parser.add_argument(
        "--with-ci",
        dest="with_ci",
        action="store_true",
        default=False,
        help="Include bootstrap confidence intervals in the table.",
    )
    parser.add_argument(
        "--ci",
        dest="ci",
        type=float,
        default=90.0,
        metavar="PERCENT",
        help="Confidence interval level in percent (default: 90).",
    )
    parser.add_argument(
        "--n-bootstrap",
        dest="n_bootstrap",
        type=int,
        default=10000,
        metavar="N",
        help=(
            "Number of bootstrap samples used to estimate CI of medians "
            "(default: 10000)."
        ),
    )
    parser.add_argument(
        "--metric",
        dest="metrics_to_optimize",
        nargs="+",
        default=["neg_log_marginal_NLE"],
        metavar="COLUMN",
        help=(
            "One or more column names in best_particle_info.csv to report. "
            "Each metric becomes a sub-table block. "
            "Only the first block shows the method and LLM headers "
            "(default: neg_log_marginal_NLE)."
        ),
    )
    parser.add_argument(
        "--direction",
        dest="optimization_direction",
        type=str,
        choices=["min", "max"],
        default="min",
        help=(
            "Optimization direction: 'min' means lower is better, 'max' means higher "
            "is better (default: min)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _best_per_run(experiment_folder: Path, metric: str) -> list[float] | None:
    """
    Read best_particle_info.csv from each run subfolder and return the metric
    value for that run's best particle.

    Existence and metric-column presence are pre-validated in main(), so this
    function reads without additional checks.

    Returns None if no run folders are found.
    """

    # Collect all run_* subdirectories within the experiment folder
    content = os.listdir(experiment_folder)
    run_folders = [c for c in content if c.startswith("run_")]

    values = []

    for run_name in run_folders:
        run_folder = os.path.join(experiment_folder, run_name)
        best_particle_info_file = os.path.join(
            run_folder, "best_particle", "best_particle_info.csv"
        )

        df_run = pd.read_csv(best_particle_info_file)
        values_run = df_run[metric].values

        # best_particle_info.csv contains exactly one row per run
        assert len(values_run) == 1

        value_run = values_run[0]
        values.append(value_run)

    assert len(values) == len(run_folders)

    return values if len(values) > 0 else None


def compute_cell_stats(
    experiment_folder: Path,
    metric: str,
    n_bootstrap: int,
    ci: float,
) -> tuple[float, float, float] | None:
    """
    Return (median, ci_low, ci_high) for a cell.

    ci_low / ci_high are the alpha/2 and 1-alpha/2 quantiles of the
    distribution of bootstrap medians, drawn with replacement from the
    per-run best-particle values.  n_bootstrap sets of size n_runs are drawn.
    """
    values = _best_per_run(experiment_folder, metric)
    if values is None:
        return None

    median = float(np.median(values))

    alpha = 1.0 - ci / 100.0

    rng = np.random.default_rng()
    bootstrap_medians = np.median(
        rng.choice(values, size=(n_bootstrap, len(values)), replace=True),
        axis=1,
    )
    ci_low = float(np.quantile(bootstrap_medians, alpha / 2))
    ci_high = float(np.quantile(bootstrap_medians, 1.0 - alpha / 2))

    return median, ci_low, ci_high


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a median value for display."""
    return f"{value:.2f}"


def build_latex_table(
    metrics: list[str],
    all_cell_values: list[dict[tuple[str, str, str], float]],
    tasks: list[str],
    methods: list[str],
    llms: list[str],
    direction: str,
    all_cell_ci: list[dict[tuple[str, str, str], tuple[float, float]] | None],
) -> str:
    r"""
    Build a LaTeX booktabs table with one block per metric.

    Rows = tasks, columns = (method, llm) combinations.
    The method and LLM sub-headers are emitted only in the first metric block.
    Each subsequent block starts with a \midrule followed by its metric header row.
    If an entry in all_cell_ci is not None, cells also show [ci_low, ci_high].
    """
    # Column keys: all (method, llm) pairs present in any metric's cell_values
    col_keys: list[tuple[str, str]] = [
        (m, l)
        for m in methods
        for l in llms
        if any(any((t, m, l) in cv for t in tasks) for cv in all_cell_values)
    ]

    col_spec = ">{\\centering\\arraybackslash}p{1.0cm} *{" + f"{len(col_keys)}" + "}{Y}"
    col_spec = col_spec + "c" * len(col_keys)
    n_data_cols = len(col_keys)

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{}")
    lines.append(r"\setlength{\tabcolsep}{0pt}")
    lines.append(r"\begin{tabularx}{\textwidth}{" + col_spec + "}")
    lines.append(r"\toprule")

    # ── Method header row (spanning LLMs) — first block only ────────────────
    method_spans: dict[str, list[int]] = {}
    for col_idx, (m, _l) in enumerate(col_keys):
        method_spans.setdefault(m, []).append(col_idx)

    method_cells: list[str] = [""]  # empty cell for the Task column
    for m in methods:
        if m not in method_spans:
            continue
        span = len(method_spans[m])
        m_label = method_labels.get(m, m.replace("_", r"\_"))
        method_cells.append(r"\multicolumn{" + str(span) + r"}{c}{" + m_label + "}")
    lines.append(" & ".join(method_cells) + r" \\")

    # ── LLM sub-header row — first block only ───────────────────────────────
    llm_cells: list[str] = [""]
    for _m, l in col_keys:
        llm_cells.append(llm_labels.get(l, l.replace("_", r"\_")))
    lines.append(" & ".join(llm_cells) + r" \\")

    # ── One block per metric ─────────────────────────────────────────────────
    for metric, cell_values, cell_ci in zip(
        metrics, all_cell_values, all_cell_ci, strict=False
    ):  # noqa
        lines.append(r"\midrule")

        # Metric header row spanning all data columns
        metric_label = metric_header.get(metric, metric.replace("_", r"\_"))
        lines.append(
            r" & \multicolumn{" + str(n_data_cols) + r"}{c}{$" + metric_label + r"$} \\"
        )
        lines.append(r"\midrule")

        # ── Data rows ───────────────────────────────────────────────────────
        for task in tasks:
            row_vals: list[float | None] = [
                cell_values.get((task, m, l)) for (m, l) in col_keys
            ]

            # Pre-compute winning indices to avoid float equality comparison in the
            # loop. All tied best values are bolded.
            task_label = task_labels.get(task, task.replace("_", r"\_"))
            cells: list[str] = [task_label]
            for _i, ((m, l), val) in enumerate(zip(col_keys, row_vals, strict=False)):
                if val is None:
                    cells.append("N/A")
                    continue
                s = _fmt(val)
                if cell_ci is not None and (task, m, l) in cell_ci:
                    lo, hi = cell_ci[(task, m, l)]
                    content = s + r" {\scriptsize [" + _fmt(lo) + ", " + _fmt(hi) + "]}"
                else:
                    content = s
                cells.append(content)
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()

    if not folder.exists():
        raise SystemExit(f"ERROR: folder not found: {folder}")
    if not folder.is_dir():
        raise SystemExit(f"ERROR: not a directory: {folder}")

    ignore = set(args.subfolder_to_ignore)
    metrics = args.metrics_to_optimize

    # Discover all experiment folders structured as <task>/<method>/<llm>/
    entries: list[tuple[str, str, str, Path]] = []
    for task_dir in sorted(folder.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ignore:
            continue
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for llm_dir in sorted(method_dir.iterdir()):
                if not llm_dir.is_dir():
                    continue
                experiment_folder = llm_dir
                if experiment_folder.exists():
                    entries.append(
                        (
                            task_dir.name,
                            method_dir.name,
                            llm_dir.name,
                            experiment_folder,
                        )
                    )

    if not entries:
        raise SystemExit(f"ERROR: no experiment folders found under {folder}")

    # Validate that every run within each experiment has a best_particle folder
    # and that its best_particle_info.csv contains each requested metric column.
    # missing_per_metric tracks, per metric, which experiments are excluded.
    missing_per_metric: dict[str, set[tuple[str, str, str]]] = {
        m: set() for m in metrics
    }
    for task, method, llm, experiment_folder in entries:
        content = os.listdir(experiment_folder)
        run_folders = [c for c in content if c.startswith("run_")]

        # If the folder does not contain any run_* subdirectories, mark as missing for
        # all metrics
        if not run_folders:
            for m in metrics:
                missing_per_metric[m].add((task, method, llm))

        # Otherwise check each run for best_particle/best_particle_info.csv
        # and that each metric column is present
        else:
            for run_folder in run_folders:
                f = os.path.join(experiment_folder, run_folder)
                bp_path = os.path.join(f, "best_particle")
                csv_path = os.path.join(bp_path, "best_particle_info.csv")

                if "best_particle" in os.listdir(
                    f
                ) and "best_particle_info.csv" in os.listdir(bp_path):
                    df_f = pd.read_csv(csv_path)
                    for m in metrics:
                        if m not in df_f.columns:
                            missing_per_metric[m].add((task, method, llm))
                            print(f"best_particle_info.csv does not contain column {m}")
                else:
                    for m in metrics:
                        missing_per_metric[m].add((task, method, llm))
                    print(
                        f"{f} is missing best_particle or "
                        "best_particle/best_particle_info.csv"
                    )
                    break

    # An experiment is flagged in the summary if it is missing for at least one metric
    any_missing: set[tuple[str, str, str]] = set().union(*missing_per_metric.values())

    print(f"Found {len(entries)} experiment(s):")
    for task, method, llm, _experiment_folder in entries:
        flag = "  [MISSING METRIC]" if (task, method, llm) in any_missing else ""
        print(f"  task={task}  method={method}  llm={llm}{flag}")

    if any_missing:
        print(
            f"\n  [!] {len(any_missing)} experiment(s) are missing at least one "
            "requested metric and will be excluded from their respective sub-table(s)."
        )

    print("\nSettings:")
    print(f"  metrics     = {metrics}  (direction: {args.optimization_direction})")
    print(f"  with_ci     = {args.with_ci}")
    if args.with_ci:
        print(f"  ci          = {args.ci}%")
        print(f"  n_bootstrap = {args.n_bootstrap}")

    # ── Compute cell values for each metric ──────────────────────────────────
    all_cell_values: list[dict[tuple[str, str, str], float]] = []
    all_cell_ci: list[dict[tuple[str, str, str], tuple[float, float]] | None] = []

    for metric in metrics:
        missing = missing_per_metric[metric]
        cell_values: dict[tuple[str, str, str], float] = {}
        cell_ci: dict[tuple[str, str, str], tuple[float, float]] = {}
        for task, method, llm, experiment_folder in entries:
            if (task, method, llm) in missing:
                continue
            stats = compute_cell_stats(
                experiment_folder, metric, args.n_bootstrap, args.ci
            )
            if stats is not None:
                median, ci_low, ci_high = stats
                cell_values[(task, method, llm)] = median
                cell_ci[(task, method, llm)] = (ci_low, ci_high)
        all_cell_values.append(cell_values)
        all_cell_ci.append(cell_ci if args.with_ci else None)

    # Preserve discovery order, deduplicated
    seen: set[str] = set()
    tasks: list[str] = []
    methods: list[str] = []
    llms: list[str] = []
    for task, _method, _llm, _ in entries:
        if task not in seen:
            tasks.append(task)
        seen.add(task)
    seen.clear()
    for _, method, _llm, _ in entries:
        if method not in seen:
            methods.append(method)
        seen.add(method)
    seen.clear()
    for _, _, llm, _ in entries:
        if llm not in seen:
            llms.append(llm)
        seen.add(llm)

    # ── Build and print LaTeX table ──────────────────────────────────────────
    latex = build_latex_table(
        metrics=metrics,
        all_cell_values=all_cell_values,
        tasks=tasks,
        methods=methods,
        llms=llms,
        direction=args.optimization_direction,
        all_cell_ci=all_cell_ci,
    )

    # Add settings to the string before saving
    if args.with_ci:
        latex += "\n\n%Settings for table computation:"
        latex += f"\n%ci          = {args.ci} percent"
        latex += f"\n%n_bootstrap = {args.n_bootstrap}"
    latex += f"\n%metrics     = {metrics}  (direction: {args.optimization_direction})"
    latex += f"\n%with_ci     = {args.with_ci}"
    latex += f"\n%folder     = {args.folder}"

    # ── Save table to file ───────────────────────────────────────────────────
    ci_suffix = f"_ci{int(args.ci)}" if args.with_ci else ""
    out_path = Path(f"main_table{ci_suffix}.tex")
    out_path.write_text(latex)
    print(f"\nTable saved to {out_path}")


if __name__ == "__main__":
    main()
