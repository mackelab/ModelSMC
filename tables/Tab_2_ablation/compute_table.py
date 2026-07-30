"""
Build the ablation table for the Allen (HH) task.

Reads per-run best_particle_info.csv files from the result folders defined in
the `folders` dict, computes median and bootstrap CI for each ablation variant,
and writes a LaTeX booktabs table to ablation_allen.tex.

Each row in `folders` maps an ablation name to its result directory relative to
this script's location (../../main_results/...). Variants that improve over the
baseline (lower metric value) are underlined.

Usage (run from inside Tab_2_ablation/):
    python compute_table.py
    python compute_table.py --metric neg_log_marginal_NLE
    python compute_table.py --metric neg_log_marginal_NLE mse
    python compute_table.py --ci 95 --n-bootstrap 5000
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Tab_1_main_table.compute_table import _fmt, compute_cell_stats, metric_header

folders = {
    "baseline": "../../main_results/allen/modelsmc/sonnet",
    "MSE_weights": "../../main_results/allen_ablation/mse/sonnet",
    "numsim_200": "../../main_results/allen_ablation/numsim_200/sonnet",
    "numsim_500": "../../main_results/allen_ablation/numsim_500/sonnet",
    "numsim_1000": "../../main_results/allen_ablation/numsim_1000/sonnet",
    "miniprompt": "../../main_results/allen_ablation/miniprompt/sonnet",
    "feedbackmode_llm_and_metrics": "../../main_results/allen_ablation/feedbackmode_llm_and_metrics/sonnet",  # noqa
    "feedbackmode_metrics_only": "../../main_results/allen_ablation/feedbackmode_metrics_only/sonnet",  # noqa
    "pool_size_150": "../../main_results/allen_ablation/pool_size_150/sonnet",
    "pool_size_10": "../../main_results/allen_ablation/pool_size_10/sonnet",
    "pool_size_5": "../../main_results/allen_ablation/pool_size_5/sonnet",
    "gptmini": "../../main_results/allen_ablation/gptmini/gptmini",
}

labels = {
    "MSE_weights": "MSE weights",
    "numsim_200": "200",
    "numsim_500": "500",
    "numsim_1000": "1000",
    "miniprompt": "Reduced",
    "feedbackmode_llm_and_metrics": "LLM + metrics",
    "feedbackmode_metrics_only": "Metrics only ",
    "pool_size_150": "$N=150,K=5$",
    "pool_size_10": "$N=10,K=75$",
    "pool_size_5": "$N=5,K=150$",
    "gptmini": "GPT-5 mini",
}


def get_line(name, results, args, axis_annotation="-"):
    results[name] = {}

    if name == "baseline":
        l = ""
    else:
        l = labels[name]

    line = axis_annotation + "&" + l

    for metric_name in args.metric_names:
        median, ci_low, ci_high = compute_cell_stats(
            experiment_folder=folders[name],
            metric=metric_name,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
        )

        # Check if the value is better than the baseline
        if (
            name != "baseline"
            and "baseline" in results.keys()
            and results["baseline"][metric_name][0] > median
        ):
            m = r"\underline{" + _fmt(median) + "}"
        else:
            m = _fmt(median)

        content = m + r" {\scriptsize [" + _fmt(ci_low) + ", " + _fmt(ci_high) + "]}"

        line += "&"
        line += content

        results[name][metric_name] = (median, ci_low, ci_high)

    line += r"\\"

    return line, results


def table_for_LLM(args) -> str:
    # ----------------------------------------------------------------------------------
    # Initialize the table
    # ----------------------------------------------------------------------------------

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{}")
    lines.append(r"\setlength{\tabcolsep}{0pt}")
    lines.append(r"\begin{tabular}{llr}")
    lines.append(r"\toprule")

    header = "Axis & Variant"
    for metric_name in args.metric_names:
        header += r"&$" + metric_header[metric_name] + "$"
    header += r"\\"

    lines.append(header)
    lines.append(r"\midrule")

    results = {}

    # ----------------------------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------------------------

    line_baseline, results = get_line(
        name="baseline", results=results, args=args, axis_annotation="Default"
    )

    lines.append(line_baseline)
    lines.append(r"\midrule")

    # ----------------------------------------------------------------------------------
    # Use MSE weights to weight the particles
    # ----------------------------------------------------------------------------------

    line_MSE_weights, results = get_line(
        name="MSE_weights",
        results=results,
        axis_annotation="Weighting",
        args=args,
    )

    lines.append(line_MSE_weights)
    lines.append(r"\midrule")

    # ----------------------------------------------------------------------------------
    # Different number of simulations
    # ----------------------------------------------------------------------------------

    line_numsim_1000, results = get_line(
        name="numsim_1000",
        results=results,
        axis_annotation="Simulations",
        args=args,
    )
    lines.append(line_numsim_1000)

    line_numsim_500, results = get_line(
        name="numsim_500",
        results=results,
        axis_annotation="",
        args=args,
    )
    lines.append(line_numsim_500)

    line_numsim_200, results = get_line(
        name="numsim_200",
        results=results,
        axis_annotation="",
        args=args,
    )
    lines.append(line_numsim_200)
    lines.append(r"\midrule")

    # ----------------------------------------------------------------------------------
    # Smaller prompt
    # ----------------------------------------------------------------------------------

    line_miniprompt, results = get_line(
        name="miniprompt",
        results=results,
        axis_annotation="Prompt",
        args=args,
    )
    lines.append(line_miniprompt)
    lines.append(r"\midrule")

    # ----------------------------------------------------------------------------------
    # Feedback modes
    # ----------------------------------------------------------------------------------

    line_llm_and_metrics, results = get_line(
        name="feedbackmode_llm_and_metrics",
        results=results,
        axis_annotation="Feedback",
        args=args,
    )
    lines.append(line_llm_and_metrics)

    line_metrics_only, results = get_line(
        name="feedbackmode_metrics_only",
        results=results,
        axis_annotation="",
        args=args,
    )
    lines.append(line_metrics_only)

    # ----------------------------------------------------------------------------------
    # Different pool size
    # ----------------------------------------------------------------------------------

    lines.append(r"\midrule")

    for pool_size_name in ["pool_size_150", "pool_size_10", "pool_size_5"]:
        line_i, results = get_line(
            name=pool_size_name,
            results=results,
            axis_annotation="Pool size" if pool_size_name == "pool_size_150" else "",
            args=args,
        )
        lines.append(line_i)

    # ----------------------------------------------------------------------------------
    # LLM
    # ----------------------------------------------------------------------------------

    lines.append(r"\midrule")

    line_i, results = get_line(
        name="gptmini",
        results=results,
        axis_annotation="LLM",
        args=args,
    )
    lines.append(line_i)

    # ----------------------------------------------------------------------------------
    # Finish the table
    # ----------------------------------------------------------------------------------

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    return latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        dest="metric_names",
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

    args = parser.parse_args()

    table_llm = table_for_LLM(args=args)
    out_path_llm = "ablation_allen.tex"

    with open(out_path_llm, "w") as f:
        f.write(table_llm)
