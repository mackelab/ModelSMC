"""
Plot the relative occurrence of the correct GMM configuration index over iterations
for a single run, identified by run ID.

Imports plotting utilities from Fig_2_ModelSMC_validation_no_LLM/scripts/plotting.py.
Reads panel sizes from notebooks/panel_sizes_cm.yaml.
Saves the figure to panels/gmm_correct_idx_single_run_<run_id>.svg.

Usage (run from inside Fig_G3_appendix_ModelSMC_validation_no_LLM/):
    python scripts/correct_idx_distribution_single_run.py
    python scripts/correct_idx_distribution_single_run.py --folder <path>
    python scripts/correct_idx_distribution_single_run.py --run-id <uuid>

Arguments:
    --folder    Path to folder containing summary.csv and GMM_idx_distribution.csv
                (default: ../../main_results/LLM-free_example).
    --run-id    UUID of the single run to plot
                (default: 35e85ba1-0b04-43e8-b95d-cd57e275ef26).
    --max-iter  Maximum iteration to include (default: all).
    --linewidth Line width of the central curve (default: 2.0).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent.parent.parent
        / "figures"
        / "Fig_2_ModelSMC_validation_no_LLM"
        / "scripts"
    ),
)
from plotting import (
    build_run_id_to_gt_idx,
    check_configs_same_except_seed_and_run_id,
    load_data,
    plot_gmm_correct_idx_distribution,
    sanity_check_gmm_dist,
)

# Get information about the scaling of the figure
with open("notebooks/panel_sizes_cm.yaml", "r") as file:
    panel_sizes = yaml.safe_load(file)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot relative occurrence of the correct GMM index over iterations."
    )
    parser.add_argument(
        "--folder",
        default="../../main_results/LLM-free_example",
        type=Path,
        help="Folder containing summary.csv and GMM_idx_distribution.csv.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum iteration to include.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width of central curve.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="35e85ba1-0b04-43e8-b95d-cd57e275ef26",
        help="Id of the run to plot",
    )

    args = parser.parse_args()

    # Load the full experimental results
    folder = args.folder.resolve()
    summary, gmm_dist = load_data(folder)

    # Select the entries belonging to the specified run id
    summary = summary[summary["run_id"] == args.run_id]
    gmm_dist = gmm_dist[gmm_dist["run_id"] == args.run_id]

    # Get the style configuration
    f = Path(__file__).parent.parent.parent
    style_file = f / ".matplotlibrc"

    run_id_to_gt_idx = build_run_id_to_gt_idx(summary)
    unique_gt_indices = sorted(set(run_id_to_gt_idx.values()))
    print(f"Found {len(run_id_to_gt_idx)} unique run_id(s).")
    print(f"Unique gt_GMM_configuration_index values: {unique_gt_indices}")

    check_configs_same_except_seed_and_run_id(summary, run_id_to_gt_idx)
    sanity_check_gmm_dist(gmm_dist, run_id_to_gt_idx)

    with mpl.rc_context(fname=style_file):
        fig, ax = plt.subplots(
            figsize=(
                panel_sizes["panel_trajectory"]["width_cm"] / 2.54,
                panel_sizes["panel_trajectory"]["height_cm"] / 2.54,
            )
        )
        plot_gmm_correct_idx_distribution(
            ax,
            gmm_dist,
            run_id_to_gt_idx,
            linewidth=args.linewidth,
            max_iter=args.max_iter,
            num_runs_per_gt_model=1,
            label_prefix="Target model",
            colors=["darkblue"],
            show_legend=False,
        )
        ax.set_xlim(right=21, left=-1)
        fig.tight_layout()

        script_dir = Path(__file__).parent
        os.makedirs(script_dir / ".." / "panels", exist_ok=True)

        save_path = (
            script_dir
            / ".."
            / "panels"
            / f"gmm_correct_idx_single_run_{args.run_id}.svg"
        )
        fig.savefig(save_path, bbox_inches="tight", format="svg", transparent=True)
        print(f"Figure saved to {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
