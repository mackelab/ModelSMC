"""Plot performance vs. token usage for Fig G4.

Usage (run from Fig_G4_token_usage_vs_performance/)::

    python scripts/plotting.py

Outputs SVG and PDF files to ``fig/performance_vs_token_usage.{svg,pdf}``.
Results are read from ``../../../main_results/{task}/{method}/{llm}/summary.csv``.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tol_colors as tc

# ── Settings ──────────────────────────────────────────────────────────────────

# Base folder containing the results (relative to this script)
BASE_FOLDER = os.path.join(os.path.dirname(__file__), "../../../main_results")

# Number of top-k runs to consider (selected per iteration based on
# running-minimum metric)
TOP_K_RUNS = 10

# Bootstrap CI settings
N_BOOTSTRAP = 10000
CI_LEVEL = 0.90

# Matplotlib style file
STYLE_FILE = os.path.join(os.path.dirname(__file__), "../../.matplotlibrc")

# Metric to plot (also used for top-k selection)
METRIC = "neg_log_marginal_NLE"
METRIC_YLABEL = r"$-\log p(x_o \mid m)$"

# LLMs — each becomes one column
llms = ["sonnet"]
llm_to_label = {
    "sonnet": "Claude Sonnet 4.6",
}

# Tasks — each becomes one row
tasks = ["SIR", "allen", "kidney"]
task_to_label = {
    "SIR": "SIR",
    "allen": "HH",
    "kidney": "Kidney",
}

# Y-axis limits per task. Set to None to use matplotlib's auto-scaling.
task_to_ylim = {
    "SIR": (-62000, -30000),
    "allen": None,
    "kidney": (37, 50),
}

# Methods — each becomes one line per subplot
methods = ["modelsmc", "modelsmcN1", "funsearch"]

# Get the colors
cset = tc.muted

method_to_color = {
    "modelsmc": cset.rose,
    "modelsmcN1": cset.indigo,
    "funsearch": cset.sand,
}
method_to_label = {
    "modelsmc": "ModelSMC",
    "modelsmcN1": "ModelSMC (N=1)",
    "funsearch": "FunSearch+",
}

# ── Helper functions ───────────────────────────────────────────────────────────


def get_run_metric_and_tokens(
    summary: pd.DataFrame, run_id: str, metric: str, token_col: str = "total_tokens"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (iterations, min_metric_per_iter, mean_tokens_per_iter) for one run.

    Args:
        summary: DataFrame containing all runs, with columns ``run_id``,
            ``iteration``, the requested metric, and ``token_col``.
        run_id: Identifier of the single run to extract.
        metric: Name of the metric column to aggregate (minimum per iteration).
        token_col: Name of the token-count column to aggregate (mean per
            iteration).

    Returns:
        Tuple of three arrays ``(iterations, metric_values, token_usages)``:
        - ``iterations`` — unique iteration indices for this run.
        - ``metric_values`` — per-iteration minimum of ``metric``; NaN rows
          become ``np.inf``.
        - ``token_usages`` — per-iteration mean of ``token_col``.
    """

    # Get the entries in the summary file for the specified run
    run = summary[summary["run_id"] == run_id]

    # Group the reduced summary by the iterations
    grouped = run.groupby("iteration")

    # Get the iteration counters
    iterations = np.array(list(grouped.groups.keys()))

    # For each group i.e. iteration get the minimal value of the metric for all
    # particles NEWLY generated at this iteration
    min_metric_values = grouped[metric].min().values
    min_metric_values = np.array(
        [np.inf if pd.isna(v) else v for v in min_metric_values]
    )

    # Get the number of tokens that were used up to (including the current iteration)
    assert (grouped[token_col].nunique() == 1).all(), (
        f"Expected exactly one unique {token_col!r} value per iteration, "
        f"but found: {grouped[token_col].nunique().to_dict()}"
    )
    token_usages = grouped[token_col].mean().values

    assert len(token_usages) == len(min_metric_values)
    assert len(token_usages) == len(iterations)

    return iterations, min_metric_values, token_usages


def _top_k_median_curve(
    run_list: list[dict], top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the top-k median metric and token curves from a list of run dicts.

    Args:
        run_list: List of dicts, each with keys ``running_min`` and ``tokens``
            (1-D arrays of the same length).  Runs may appear more than once
            (e.g. bootstrap resamples).
        top_k: Number of best runs (lowest running-min) to include in the
            median at each iteration.

    Returns:
        Tuple ``(avg_metric, avg_token)``, both 1-D arrays of length equal to
        the number of iterations in the longest run in ``run_list``.
    """

    # Get the maximum number if iterations over all runs
    max_iters = max(len(d["running_min"]) for d in run_list)

    # At each iteration collect the median value of the metric over all runs and the
    # median value of the token usage
    median_metric, median_token = [], []

    for t in range(max_iters):
        values_at_t = [
            (d["running_min"][t], d["tokens"][t])
            for d in run_list
            if t < len(d["running_min"])
        ]
        if not values_at_t:
            break

        # Sort the tuples of metric values and tokens at the current time step by the
        # value of the metric in ascending order
        values_at_t.sort(key=lambda x: x[0])

        # Only keep the best performing runs
        top_k_values = values_at_t[:top_k]

        # Compute the median token usage at the current iteration and the median
        # metric value over all top_k runs
        median_metric.append(np.median([v[0] for v in top_k_values]))
        median_token.append(np.median([v[1] for v in top_k_values]))

    assert len(median_metric) == max_iters
    assert len(median_token) == max_iters

    return np.array(median_metric), np.array(median_token)


def average_over_top_k_runs_per_iteration(
    summary: pd.DataFrame,
    metric: str,
    top_k: int,
    token_col: str = "total_tokens",
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    At each iteration, select the top-k runs by their running-minimum metric up
    to that iteration, then compute the median metric and token usage over those
    runs.  Bootstrapped confidence intervals are computed by resampling all runs
    with replacement.

    Args:
        summary: DataFrame containing all runs, with columns ``run_id``,
            ``iteration``, the requested metric, and ``token_col``.
        metric: Name of the metric column used for both selection and
            aggregation (lower is better).
        top_k: Number of best runs to keep at each iteration.
        token_col: Name of the token-count column to aggregate (median over
            the top-k runs per iteration).
        n_bootstrap: Number of bootstrap resamples for the CI.
        ci_level: Coverage of the confidence interval (e.g. 0.95 for 95 %).
        label: Identifier for the experimental setting, i.e. combination of task and
            method name.

    Returns:
        Tuple ``(avg_metric, avg_token, ci_low, ci_high)`` where all four are
        1-D arrays of the same length (one entry per iteration).
        ``avg_metric`` and ``avg_token`` are the median curves over the
        observed runs; ``ci_low`` and ``ci_high`` are the bootstrapped
        ``ci_level`` confidence bounds on ``avg_metric``.
    """
    all_run_ids = summary["run_id"].unique().tolist()

    # Precompute per-run running-min metric and token usage, i.e. get the best value
    # observed so far at each iteration of the run.
    run_data: dict[str, dict] = {}

    for run_id in all_run_ids:
        iters, mv, tv = get_run_metric_and_tokens(summary, run_id, metric, token_col)
        run_data[run_id] = {
            "iterations": iters,
            "running_min": np.minimum.accumulate(mv),
            "tokens": tv,
        }

    # Warn if some runs have fewer iterations than the longest one
    iter_lengths = {rid: len(d["iterations"]) for rid, d in run_data.items()}
    max_len = max(iter_lengths.values())
    short_runs = {rid: n for rid, n in iter_lengths.items() if n < max_len}
    if short_runs:
        prefix = f"[{label}] " if label else ""
        print(
            f"Warning {prefix}{len(short_runs)} of {len(run_data)} runs terminated"
            f" early (max iterations: {max_len}). Short runs: "
            + ", ".join(f"{rid}={n}" for rid, n in sorted(short_runs.items()))
        )

    # Collect the trajectories for the individual runs
    run_list = list(run_data.values())

    # Compute the median metric and token usage
    median_metric, median_token = _top_k_median_curve(run_list, top_k)

    # Bootstrap CI over the metric curve (x-axis token values stay fixed).
    # Runs may have different iteration counts; bootstrap resamples that only
    # contain short runs will produce shorter arrays, so pad with NaN and use
    # nanquantile.
    global_max_iters = len(median_metric)
    rng = np.random.default_rng(0)
    n_runs = len(run_list)
    boot_metrics = []
    for _ in range(n_bootstrap):
        # Resample which runs to use for the median
        indices = rng.integers(0, n_runs, size=n_runs)

        # Compute the median based on the selection
        boot_metric, _ = _top_k_median_curve([run_list[i] for i in indices], top_k)

        # Pad to global length so all bootstrap arrays are the same shape
        if len(boot_metric) < global_max_iters:
            boot_metric = np.concatenate(
                [
                    boot_metric,
                    np.full(global_max_iters - len(boot_metric), np.nan),
                ]
            )

        boot_metrics.append(boot_metric)

    boot_metrics = np.array(boot_metrics)
    assert boot_metrics.shape == (n_bootstrap, global_max_iters)

    # Compute the confidence interval of the median. Ignore nan values in the
    # quantile computation.
    alpha = (1 - ci_level) / 2
    ci_low = np.nanquantile(boot_metrics, alpha, axis=0)
    ci_high = np.nanquantile(boot_metrics, 1 - alpha, axis=0)

    return median_metric, median_token, ci_low, ci_high


# ── Plot ───────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "../fig"), exist_ok=True)

    with mpl.rc_context(fname=STYLE_FILE):
        fig, axes = plt.subplots(
            len(tasks), len(llms), figsize=(6.75, 2.2 * len(tasks)), squeeze=False
        )

        for row, task in enumerate(tasks):
            for col, llm in enumerate(llms):
                ax = axes[row][col]

                for method in methods:
                    summary_path = os.path.join(
                        BASE_FOLDER, task, method, llm, "summary.csv"
                    )
                    if not os.path.exists(summary_path):
                        continue

                    summary = pd.read_csv(summary_path)
                    if METRIC not in summary.columns:
                        print(
                            f"Metric '{METRIC}' not found in {summary_path}, skipping."
                        )
                        continue

                    # Compute the median properties and the ci of the metric median
                    median_metric, median_token, ci_low, ci_high = (
                        average_over_top_k_runs_per_iteration(
                            summary, METRIC, TOP_K_RUNS, label=f"{task}/{method}"
                        )
                    )

                    color = method_to_color[method]

                    # Plot the ci of the metric
                    ax.fill_between(
                        median_token,
                        ci_low,
                        ci_high,
                        color=color,
                        alpha=0.2,
                        rasterized=True,
                    )

                    # Plot the median curve
                    ax.plot(
                        median_token,
                        median_metric,
                        label=method_to_label[method],
                        color=color,
                    )

                # Row label (task) on the left-most column
                if col == 0:
                    ax.set_ylabel(f"{task_to_label[task]}\n{METRIC_YLABEL}")
                else:
                    ax.set_ylabel(METRIC_YLABEL)

                # Column header (LLM) on the top row, only when there are
                # multiple columns
                if row == 0 and len(llms) > 1:
                    ax.set_title(llm_to_label[llm])

                ax.set_xlabel("Total Tokens")

                if task_to_ylim[task] is not None:
                    ax.set_ylim(task_to_ylim[task])

        # Single shared legend below all subplots
        handles, labels = [], []
        for method in methods:
            color = method_to_color[method]
            line = mpl.lines.Line2D([], [], color=color)
            patch = mpl.patches.Patch(color=color, alpha=0.2)
            handles.append((patch, line))
            labels.append(method_to_label[method])

        # Add a single entry explaining the shaded band
        handles.append(mpl.patches.Patch(color="gray", alpha=0.2))
        labels.append(f"{int(CI_LEVEL * 100)}% CI")

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(methods) + 1,
            bbox_to_anchor=(0.5, -0.04),
            handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
        )

        fig.tight_layout(rect=[0, 0.06, 1, 1])

        out_base = os.path.join(
            os.path.dirname(__file__), "../fig/performance_vs_token_usage"
        )
        fig.savefig(out_base + ".svg", format="svg", transparent=True)
        fig.savefig(out_base + ".pdf", format="pdf", transparent=False)
        print(f"Saved to {out_base}.{{svg,pdf}}")


if __name__ == "__main__":
    main()
