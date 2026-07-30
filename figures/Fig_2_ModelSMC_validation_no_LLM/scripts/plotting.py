"""
Plot the relative occurrence of the correct GMM configuration index over iterations,
grouped by gt_GMM_configuration_index across runs.

Usage (run from inside Fig_2_ModelSMC_validation_no_LLM/):
    python scripts/plotting.py
    python scripts/plotting.py --folder <path>
    python scripts/plotting.py --folder <path> --no-save --show
    python scripts/plotting.py --aggregation median_bootstrap --bootstrap-n 10000 --bootstrap-alpha 0.0 --bootstrap-seed 0
    python scripts/plotting.py --aggregation mean_std

Arguments:
    --folder            Path to folder containing summary.csv and
                        GMM_idx_distribution.csv
                        (default: ../../main_results/LLM-free_example).
    --save / --no-save  Save the figure to fig/ inside Fig_2_ModelSMC_validation_no_LLM/
                        (default: save).
    --show              Display the figure interactively (default: False).
    --max-iter          Maximum iteration to include (default: all).
    --linewidth         Line width of the central curve (default: 2.0).
    --alpha-shade       Transparency of the shaded band (default: 0.2).
    --num-runs-per-gt-model
                        Expected number of runs per gt_GMM_configuration_index
                        (default: 10).
    --aggregation       Aggregation mode: 'mean_std' or 'median_bootstrap' (default:
                        median_bootstrap).
    --bootstrap-n       Bootstrap resamples for median_bootstrap (default: 10000).
    --bootstrap-alpha   Significance level for bootstrap CI; 0.0 gives 100%% CI
                        (default: 0.0).
    --bootstrap-seed    Random seed for bootstrap resampling (default: 0).
"""  # noqa

import argparse
import ast
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tol_colors as tc

# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------


def _bootstrap_median_ci(
    values: np.ndarray,
    n: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (median, lower_ci, upper_ci) via bootstrap resampling.

    Parameters
    ----------
    values:
        1-D array of observations to resample from.
    n:
        Number of bootstrap resamples.
    alpha:
        Significance level; the CI spans the [alpha/2, 1-alpha/2] quantiles
        of the bootstrap median distribution.
    rng:
        NumPy random Generator used for resampling.

    Returns
    -------
    Tuple of (median, lower CI bound, upper CI bound).
    """
    k = len(values)
    bootstrap_medians = np.median(rng.choice(values, size=(n, k), replace=True), axis=1)
    assert bootstrap_medians.shape == (n,)

    lo = float(np.quantile(bootstrap_medians, alpha / 2))
    hi = float(np.quantile(bootstrap_medians, 1.0 - alpha / 2))
    return float(np.median(values)), lo, hi


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------


def plot_gmm_correct_idx_distribution(
    ax: plt.Axes,
    gmm_dist: pd.DataFrame,
    run_id_to_gt_idx: dict,
    *,
    linewidth: float = 2.0,
    marker: str = "o",
    marker_incomplete: str = "s",
    markersize: float = 4.0,
    colors=None,
    max_iter: int | None = None,
    alpha_shade: float = 0.2,
    label_prefix: str = "gt idx",
    markevery: int = 1,
    num_runs_per_gt_model: int = 10,
    aggregation: str = "mean_std",
    bootstrap_n: int = 1000,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int | None = None,
    show_legend: bool = True,
) -> dict:
    """Plot relative occurrence of the correct GMM index per iteration.

    For each unique ``gt_GMM_configuration_index`` found in *run_id_to_gt_idx*,
    collects all matching runs from *gmm_dist*, computes the relative occurrence
    of that index at every iteration, and plots a summary curve ± band on *ax*.

    Two aggregation modes are supported (``aggregation`` parameter):

    * ``"mean_std"`` — plots the mean across runs with a ± 1 std shaded band.
    * ``"median_bootstrap"`` — plots the median across runs.  The shaded band is
      a ``(1 - bootstrap_alpha)`` confidence interval obtained by bootstrap
      resampling: *bootstrap_n* bootstrap samples are drawn (with replacement)
      from the runs at each iteration, the median is computed for each sample,
      and the ``bootstrap_alpha/2`` and ``1 - bootstrap_alpha/2`` quantiles of
      that distribution of bootstrap medians form the lower and upper band edges.

    Iterations where the number of unique random seeds does not equal
    ``num_runs_per_gt_model`` (e.g. due to crashed or re-run experiments) are
    drawn with ``marker_incomplete`` instead of ``marker`` to make deviations
    immediately visible.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    gmm_dist:
        DataFrame loaded from ``GMM_idx_distribution.csv``.  Expected columns:
        ``run_id``, ``itr``, and ``counts(idx=<i>)`` for each GMM component.
    run_id_to_gt_idx:
        Mapping from run_id to its ``gt_GMM_configuration_index`` (extracted
        from ``summary.csv``).
    linewidth:
        Line width of the central curve.
    marker:
        Marker style used at iterations where exactly ``num_runs_per_gt_model``
        unique random seeds are present.
    marker_incomplete:
        Marker style used at iterations where the number of unique random seeds
        differs from ``num_runs_per_gt_model`` (e.g. crashed / extra runs).
    markersize:
        Marker size for both marker styles.
    colors:
        Colour specification for each gt index.  Can be a dict mapping
        ``gt_GMM_configuration_index`` → colour string, or a list indexed by
        the sorted unique gt indices.  If ``None``, uses the current
        matplotlib colour cycle.
    max_iter:
        If given, only iterations ``<= max_iter`` are plotted.
    alpha_shade:
        Transparency of the shaded band (0 = invisible, 1 = opaque).
    label_prefix:
        Prefix used in the legend label: ``"{label_prefix} {gt_idx}"``.
    markevery:
        Plot a marker only every *markevery* data points (applied to the
        sorted iteration index, counting from 0).
    num_runs_per_gt_model:
        Expected number of runs (unique random seeds) per
        ``gt_GMM_configuration_index`` at each iteration.  Used both for
        sanity-check warnings and to decide which marker to draw.
    aggregation:
        Aggregation mode.  One of ``"mean_std"`` or ``"median_bootstrap"``.
    bootstrap_n:
        Number of bootstrap resamples used when ``aggregation="median_bootstrap"``.
    bootstrap_alpha:
        Significance level for the bootstrap confidence interval
        (``aggregation="median_bootstrap"``).  The plotted band spans the
        ``[bootstrap_alpha/2, 1 - bootstrap_alpha/2]`` quantiles of the
        bootstrap median distribution.  Default ``0.05`` gives a 95 % CI.
    bootstrap_seed:
        Optional integer seed for the bootstrap RNG to make results
        reproducible (``aggregation="median_bootstrap"``).
    show_legend:
        Visualize the legend

    Returns
    -------
    dict mapping gt_GMM_configuration_index → Line2D handle (useful for legends).
    """

    if aggregation not in {"mean_std", "median_bootstrap"}:
        raise ValueError(
            f"aggregation must be 'mean_std' or 'median_bootstrap', got \
            '{aggregation}'."
        )

    count_cols = [c for c in gmm_dist.columns if c.startswith("counts(idx=")]
    if not count_cols:
        raise ValueError(
            "No 'counts(idx=...)' columns found in GMM distribution DataFrame."
        )

    total_counts = gmm_dist[count_cols].sum(axis=1)

    if (total_counts == 0).any():
        warnings.warn(
            "Some rows have zero total counts; relative occurrence will be NaN for \
            hose rows.",
            stacklevel=2,
        )

    rng = (
        np.random.default_rng(bootstrap_seed)
        if aggregation == "median_bootstrap"
        else None
    )

    # Group run_ids by gt_GMM_configuration_index
    gt_idx_to_run_ids: dict = {}
    for run_id, gt_idx in run_id_to_gt_idx.items():
        gt_idx_to_run_ids.setdefault(gt_idx, []).append(run_id)

    unique_gt_indices = sorted(gt_idx_to_run_ids.keys())

    # Resolve colours
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {
            gt_idx: prop_cycle[i % len(prop_cycle)]
            for i, gt_idx in enumerate(unique_gt_indices)
        }
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = {
            gt_idx: colors[i % len(colors)]
            for i, gt_idx in enumerate(unique_gt_indices)
        }

    handles: dict = {}

    for gt_idx in unique_gt_indices:
        # Get all the run ids for the evaluated gt index
        run_ids = gt_idx_to_run_ids[gt_idx]

        # Construct the name of the column containing the counts of the gt index
        col_name = f"counts(idx={gt_idx})"

        if col_name not in gmm_dist.columns:
            warnings.warn(
                f"Column '{col_name}' not found; skipping gt_idx={gt_idx}.",
                stacklevel=2,
            )
            continue

        # Select the entries that correspond to a run with the selected gt index
        mask = gmm_dist["run_id"].isin(run_ids)
        subset = gmm_dist[mask].copy()

        # Select the initial max_iter iterations
        if max_iter is not None:
            subset = subset[subset["itr"] <= max_iter]

        # Compute the relative occurrence of the gt index at each iteration
        subset["rel_occurrence"] = subset[col_name] / total_counts[subset.index]

        # Sanity check: expected number of runs at every iteration
        runs_per_iter = subset.groupby("itr")["run_id"].nunique()
        bad_iters = runs_per_iter[runs_per_iter != num_runs_per_gt_model]
        if not bad_iters.empty:
            warnings.warn(
                f"gt_idx={gt_idx}: not all iterations have {num_runs_per_gt_model}"
                f"runs. Iterations with deviating counts:\n{bad_iters.to_string()}",
                stacklevel=2,
            )

        # Aggregate across runs per iteration
        sorted_itrs = sorted(subset["itr"].unique())

        central_vals = []
        lo_vals = []
        hi_vals = []

        for itr_val in sorted_itrs:
            # Select the relative occurrences of the gt index at the selected iteration
            values = subset.loc[subset["itr"] == itr_val, "rel_occurrence"].values

            # Mean and standard deviation
            if aggregation == "mean_std":
                c = float(np.mean(values))
                s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                central_vals.append(c)
                lo_vals.append(c - s)
                hi_vals.append(c + s)

            # Median and confidence intervals of the bootstrap distribution
            # of the median
            else:  # median_bootstrap
                med, lo, hi = _bootstrap_median_ci(
                    values, bootstrap_n, bootstrap_alpha, rng
                )
                central_vals.append(med)
                lo_vals.append(lo)
                hi_vals.append(hi)

        itr = np.array(sorted_itrs)
        central = np.array(central_vals)
        lo = np.array(lo_vals)
        hi = np.array(hi_vals)

        # Determine which iterations have the expected seed count
        complete_mask = runs_per_iter.reindex(itr).values == num_runs_per_gt_model

        color = color_map.get(gt_idx)
        label = f"{label_prefix} {gt_idx}"

        # Draw line without markers, then overlay markers per-iteration
        (line,) = ax.plot(
            itr,
            central,
            color=color,
            linewidth=linewidth,
            marker="",
            label=label,
        )

        # Plot the dispersion as shaded region
        ax.fill_between(itr, lo, hi, color=color, alpha=alpha_shade, linewidth=0)

        # Apply markevery by selecting every markevery-th index
        mark_indices = np.arange(0, len(itr), markevery)
        for mi in mark_indices:
            m = marker if complete_mask[mi] else marker_incomplete
            ax.plot(
                itr[mi],
                central[mi],
                color=color,
                marker=m,
                markersize=markersize,
                linestyle="",
            )

        handles[gt_idx] = line

    # Add labels
    ax.set_xlabel("iteration")
    ax.set_ylabel("prop. target model")
    ax.set_ylim(bottom=0.0)
    if handles and show_legend:
        ax.legend(
            handles=list(handles.values()),
            labels=[h.get_label() for h in handles.values()],
            ncol=2,
        )

    return handles


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _parse_gt_idx_from_config(config_str: str) -> int:
    """Parse gt_GMM_configuration_index from a Hydra config dict string."""
    config = ast.literal_eval(config_str)
    return int(config["task"]["gt_GMM_configuration_index"])


def load_data(folder: Path):
    """Load summary.csv and GMM_idx_distribution.csv from *folder*."""
    summary_path = folder / "summary.csv"
    gmm_path = folder / "GMM_idx_distribution.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in {folder}")
    if not gmm_path.exists():
        raise FileNotFoundError(f"GMM_idx_distribution.csv not found in {folder}")

    summary = pd.read_csv(summary_path)
    gmm_dist = pd.read_csv(gmm_path)
    return summary, gmm_dist


def build_run_id_to_gt_idx(summary: pd.DataFrame) -> dict:
    """Build mapping run_id → gt_GMM_configuration_index from summary.

    Raises if a run_id maps to more than one gt_GMM_configuration_index.
    """
    run_id_to_gt: dict = {}
    for _, row in summary.iterrows():
        run_id = row["run_id"]
        gt_idx = _parse_gt_idx_from_config(row["config"])

        # If the run id is found more than once, check for consistency, i.e. if the
        # rows with the same run_id have the same gt index
        if run_id in run_id_to_gt and run_id_to_gt[run_id] != gt_idx:
            raise ValueError(
                f"run_id '{run_id}' has conflicting gt_GMM_configuration_index values: "
                f"{run_id_to_gt[run_id]} vs {gt_idx}"
            )

        # Add to the mapping
        run_id_to_gt[run_id] = gt_idx
    return run_id_to_gt


def check_configs_same_except_seed_and_run_id(
    summary: pd.DataFrame, run_id_to_gt_idx: dict
) -> None:
    """Check that runs sharing the same gt_GMM_configuration_index have identical
    configs except for the top-level 'seed' and 'run_id' fields.

    Raises ValueError if any group contains configs that differ in other entries.
    """
    IGNORED_KEYS = {"seed", "run_id"}

    def _strip(config: dict) -> dict:
        """Remove ignored top level keys from the dictionary"""
        return {k: v for k, v in config.items() if k not in IGNORED_KEYS}

    # Build mapping: run_id -> stripped config (use first row per run_id)
    run_id_to_stripped: dict = {}
    for _, row in summary.iterrows():
        run_id = row["run_id"]
        if run_id not in run_id_to_stripped:
            config = ast.literal_eval(row["config"])
            run_id_to_stripped[run_id] = _strip(config)

    # Group run_ids by gt_GMM_configuration_index
    gt_idx_to_run_ids: dict = {}
    for run_id, gt_idx in run_id_to_gt_idx.items():
        gt_idx_to_run_ids.setdefault(gt_idx, []).append(run_id)

    for gt_idx, run_ids in gt_idx_to_run_ids.items():
        # No need to check consistency if there are less than two runs per gt index
        if len(run_ids) < 2:
            continue

        # Reference against which the other runs are compared
        reference_run_id = run_ids[0]
        reference_config = run_id_to_stripped[reference_run_id]

        # Compare the remaining runs for this gt index to the reference
        for run_id in run_ids[1:]:
            # Get the config
            other_config = run_id_to_stripped[run_id]

            # Compare
            if other_config != reference_config:
                # Find differing keys for a helpful error message
                all_keys = set(reference_config) | set(other_config)
                diffs = {
                    k
                    for k in all_keys
                    if reference_config.get(k) != other_config.get(k)
                }
                raise ValueError(
                    f"gt_idx={gt_idx}: run '{run_id}' differs from "
                    f"'{reference_run_id}' in config keys other than "
                    f"seed/run_id: {diffs}"
                )

    print(
        "Config sanity check passed: all runs per gt_idx only differ in 'seed' and "
        "'run_id'."
    )


def sanity_check_gmm_dist(gmm_dist: pd.DataFrame, run_id_to_gt_idx: dict) -> None:
    """Run basic sanity checks on the GMM distribution DataFrame."""

    # Check if the two data files share the same run ids
    known_run_ids = set(run_id_to_gt_idx.keys())
    gmm_run_ids = set(gmm_dist["run_id"].unique())

    only_in_gmm = gmm_run_ids - known_run_ids
    only_in_summary = known_run_ids - gmm_run_ids

    if only_in_gmm:
        warnings.warn(
            f"{len(only_in_gmm)} run_id(s) appear in GMM distribution but not in "
            f"summary.csv: {only_in_gmm}",
            stacklevel=2,
        )
    if only_in_summary:
        warnings.warn(
            f"{len(only_in_summary)} run_id(s) appear in summary.csv but not in GMM "
            f"distribution: {only_in_summary}",
            stacklevel=2,
        )

    # Check all runs share the same iteration set
    iters_per_run = gmm_dist.groupby("run_id")["itr"].apply(frozenset)
    unique_iter_sets = iters_per_run.unique()

    if len(unique_iter_sets) > 1:
        warnings.warn(
            "Not all runs share the same set of iterations. "
            "The summary statistic at each iteration only includes runs that have data "
            "for that iteration.",
            stacklevel=2,
        )

    # Check no duplicate (run_id, itr) pairs
    dups = gmm_dist.duplicated(subset=["run_id", "itr"])
    if dups.any():
        raise ValueError(
            f"Duplicate (run_id, itr) pairs found in GMM distribution:\n"
            f"{gmm_dist[dups][['run_id', 'itr']]}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot relative occurrence of the correct GMM index over iterations."
    )

    parser.add_argument(
        "--folder",
        type=Path,
        default="../../main_results/LLM-free_example",
        help="Folder containing summary.csv and GMM_idx_distribution.csv.",
    )

    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=True,
        help="Save figure to <folder>/gmm_correct_idx_distribution.pdf (default).",
    )

    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save the figure.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the figure interactively (default: False).",
    )

    parser.add_argument(
        "--max-iter", type=int, default=None, help="Maximum iteration to include."
    )

    parser.add_argument(
        "--linewidth", type=float, default=2.0, help="Line width of central curve."
    )

    parser.add_argument(
        "--alpha-shade", type=float, default=0.2, help="Transparency of shaded band."
    )

    parser.add_argument(
        "--num-runs-per-gt-model",
        type=int,
        default=10,
        help="Expected number of runs (unique random seeds) per gt_GMM_configuration_index (default: 10).",  # noqa
    )

    parser.add_argument(
        "--aggregation",
        choices=["mean_std", "median_bootstrap"],
        default="median_bootstrap",
        help="Aggregation mode: 'mean_std' (mean ± 1 std) or 'median_bootstrap' (median + bootstrap CI).",  # noqa
    )

    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (used with --aggregation median_bootstrap, default: 10000).",  # noqa
    )

    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.0,
        help="Significance level for bootstrap CI (default: 0.0 → 100 %% CI).",
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling (default: 0).",
    )

    args = parser.parse_args()

    folder = args.folder.resolve()
    summary, gmm_dist = load_data(folder)

    f = Path(__file__).parent.parent.parent
    style_file = f / ".matplotlibrc"

    run_id_to_gt_idx = build_run_id_to_gt_idx(summary)
    unique_gt_indices = sorted(set(run_id_to_gt_idx.values()))

    print(f"Found {len(run_id_to_gt_idx)} unique run_id(s).")
    print(f"Unique gt_GMM_configuration_index values: {unique_gt_indices}")

    check_configs_same_except_seed_and_run_id(summary, run_id_to_gt_idx)
    sanity_check_gmm_dist(gmm_dist, run_id_to_gt_idx)

    with mpl.rc_context(fname=style_file):
        # Get the colors
        cset = tc.muted

        # Plot the trajectories
        fig, ax = plt.subplots(figsize=(3.3, 1.65))
        plot_gmm_correct_idx_distribution(
            ax,
            gmm_dist,
            run_id_to_gt_idx,
            linewidth=args.linewidth,
            max_iter=args.max_iter,
            alpha_shade=args.alpha_shade,
            num_runs_per_gt_model=args.num_runs_per_gt_model,
            aggregation=args.aggregation,
            bootstrap_n=args.bootstrap_n,
            bootstrap_alpha=args.bootstrap_alpha,
            bootstrap_seed=args.bootstrap_seed,
            label_prefix="Target model",
            colors=[cset.rose, cset.indigo, cset.sand, cset.green, cset.cyan],
        )
        fig.tight_layout()

        # Create a suffix to identify the configurations used to compute the plot
        if args.aggregation == "median_bootstrap":
            info_str = (
                args.aggregation
                + f"_{100 * (1 - args.bootstrap_alpha)}%-CI_{args.bootstrap_n}-samples_seed-{args.bootstrap_seed}"  # noqa
            )
        else:
            info_str = args.aggregation

        # Save the figure
        if args.save:
            script_dir = Path(__file__).parent.parent

            if not os.path.exists(script_dir / "fig"):
                os.mkdir(script_dir / "fig")

            # SVG
            save_path = (
                script_dir / "fig" / f"gmm_correct_idx_distribution_{info_str}.svg"
            )
            fig.savefig(save_path, format="svg", transparent=True)
            print(f"Figure saved to {save_path}")

            # PDF
            save_path = (
                script_dir / "fig" / f"gmm_correct_idx_distribution_{info_str}.pdf"
            )
            fig.savefig(save_path, format="pdf")
            print(f"Figure saved to {save_path}")

        # Show the plot
        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
