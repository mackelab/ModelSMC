import ast
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelsmc.utils.plot_utils import use_style

logger = logging.getLogger("ModelSMC")


def _setup_trajectory_plot(
    summary_csv_path: str,
    pool_csv_path: str,
    run_id: str,
    cmap_name: str | None,
    single_color: str,
    fig,
    ax,
    figsize: tuple[float, float],
    max_time_step: int | None = None,
) -> tuple:
    """Set up common elements for trajectory plotting.

    Returns:
        fig, ax, df_filtered_summary, df_filtered_pool, colors, n_iterations, or
        fig, ax, None, None, None, None if loading fails.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    try:
        if os.path.exists(summary_csv_path):
            df_raw_summary = pd.read_csv(summary_csv_path)
            df_filtered_summary = df_raw_summary[df_raw_summary["run_id"] == run_id]
        else:
            raise ValueError(
                f"The specified summary file {summary_csv_path} does not exist."
            )
    except KeyError as e:
        logger.error(e)
        logger.error("Failed to load recorded summary.")
        return fig, ax, None, None, None

    try:
        if os.path.exists(pool_csv_path):
            df_raw_pool = pd.read_csv(pool_csv_path)
            df_filtered_pool = df_raw_pool[df_raw_pool["run_id"] == run_id]
        else:
            raise ValueError(f"The specified pool file {pool_csv_path} does not exist.")
    except KeyError as e:
        logger.error(e)
        logger.error("Failed to load recorded pool.")
        return fig, ax, None, None, None

    if max_time_step is not None:
        n_iterations = min(
            max(df_filtered_summary["iteration"].values), max_time_step + 1
        )
    else:
        n_iterations = max(df_filtered_summary["iteration"].values) + 1

    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in np.linspace(0, 1.0, n_iterations)]
    else:
        colors = [single_color] * n_iterations

    return fig, ax, df_filtered_summary, df_filtered_pool, colors, n_iterations


def plot_trajectories_raw(
    summary_csv_path: str,
    pool_csv_path: str,
    run_id: str,
    metric_name: str | None = None,
    cmap_name: str | None = "plasma",
    single_color: str = "C0",
    lw_full: int = 2,
    s_new: float = 5.0,
    lw_particle_exists: float = 1.0,
    alpha: float = 1.0,
    fig=None,
    ax=None,
    figsize: tuple[float, float] = (10, 5),
    max_time_step: int | None = None,
    show_failed_particles: bool = True,
    show_copied_lines: bool = True,
    deduplicate_markers: bool = False,
) -> tuple:
    """Core plotting logic for particle trajectories without styling.

    Args:
        summary_csv_path: Path to the per-particle summary CSV.
        pool_csv_path: Path to the pool composition CSV.
        run_id: Run identifier used to filter both CSVs.
        metric_name: Column name of the metric to plot on the y-axis.
        cmap_name: Colormap name for iteration colours. Uses single_color if None.
        single_color: Fallback colour when cmap_name is None.
        lw_full: Line width for lines connecting a new particle to its ancestor.
        s_new: Marker size for particles; scaled by pool multiplicity.
        lw_particle_exists: Line width for dashed lines showing copied-particle lifespan.
        alpha: Opacity for all plotted elements.
        fig: Existing figure to draw on; a new one is created if None.
        ax: Existing axes to draw on; new axes are created if None.
        figsize: Figure size when creating a new figure.
        max_time_step: Truncate the plot at this iteration. Plots all if None.
        show_failed_particles: If True, mark failed (non-finite) particles with a square.
        show_copied_lines: If True, draw dashed lines for the lifespan of copied particles.
        deduplicate_markers: If True, each particle is drawn only once per iteration
            (avoiding opacity accumulation from overdrawing duplicates).

    Returns:
        fig, ax: The figure and axes with plotted data.
    """  # noqa
    fig, ax, summary, pool, colors, n_iterations = _setup_trajectory_plot(
        summary_csv_path=summary_csv_path,
        pool_csv_path=pool_csv_path,
        run_id=run_id,
        cmap_name=cmap_name,
        single_color=single_color,
        fig=fig,
        ax=ax,
        figsize=figsize,
        max_time_step=max_time_step,
    )

    if summary is None or pool is None:
        return fig, ax

    for iteration in range(n_iterations):
        pool_iteration = pool[pool["itr"] == iteration]
        uuids = ast.literal_eval(pool_iteration["pool_members_uuids"].values[0])

        for uuid in dict.fromkeys(uuids) if deduplicate_markers else uuids:
            metric_values = summary[summary["particle_id"] == uuid][metric_name].values
            assert len(metric_values) == 1
            metric_val = metric_values[0]

            if np.isfinite(metric_val):
                ax.plot(
                    [iteration],
                    [metric_val],
                    c=colors[iteration],
                    ms=s_new,
                    marker="o",
                    alpha=alpha,
                    zorder=10,
                )

            # Draw connection to parent particle
            if iteration > 0:
                previous_pool = pool[pool["itr"] == (iteration - 1)]

                uuids_previous = ast.literal_eval(
                    previous_pool["pool_members_uuids"].values[0]
                )

                if uuid in uuids_previous:
                    # Copied particle: draw a short dashed line at constant value
                    if not np.isfinite(metric_val):
                        continue

                    metric_val_previous = metric_val
                    lw = lw_particle_exists
                    ls = ":" if show_copied_lines else ""
                    color = "k"
                    start = iteration - 1
                    end = iteration
                    x_pos = None

                else:
                    # New particle: connect to its ancestor
                    row = summary[summary["particle_id"] == uuid]
                    parent_id = row["parent_id"].values[0]

                    metric_values_previous = summary[
                        summary["particle_id"] == parent_id
                    ][metric_name].values
                    assert len(metric_values_previous) == 1
                    metric_val_previous = metric_values_previous[0]

                    lw = lw_full
                    ls = "-"
                    color = colors[iteration - 1]

                    if not np.isfinite(metric_val_previous) and not np.isfinite(
                        metric_val
                    ):
                        continue
                    elif not np.isfinite(metric_val_previous) and np.isfinite(
                        metric_val
                    ):
                        # Parent failed, child succeeded
                        start = iteration - 0.5
                        end = iteration if show_failed_particles else start
                        x_pos = [start, metric_val] if show_failed_particles else None
                        metric_val_previous = metric_val
                    elif np.isfinite(metric_val_previous) and not np.isfinite(
                        metric_val
                    ):
                        # Parent succeeded, child failed
                        start = iteration - 1
                        end = iteration - 0.5 if show_failed_particles else start
                        x_pos = (
                            [end, metric_val_previous]
                            if show_failed_particles
                            else None
                        )
                        metric_val = metric_val_previous
                    else:
                        start = iteration - 1
                        end = iteration
                        x_pos = None

                ax.plot(
                    [start, end],
                    [metric_val_previous, metric_val],
                    c=color,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                    zorder=9,
                )

                if x_pos:
                    ax.plot(
                        [x_pos[0]],
                        [x_pos[1]],
                        marker="s",
                        ms=s_new,
                        alpha=alpha,
                        zorder=9,
                        c=colors[iteration - 1],
                    )

    return fig, ax


def plot_trajectories(
    summary_csv_path: str,
    pool_csv_path: str,
    run_id: str,
    save_dir: str,
    image_name: str,
    metric_name: str | None = None,
    cmap_name: str | None = "plasma",
    single_color: str = "C0",
    lw_full: int = 2,
    axes=None,
    s_new: float = 5.0,
    lw_particle_exists: float = 1.0,
    figsize: tuple[float, float] = (10, 5),
    save: bool = True,
    show: bool = False,
    yscale: str = "log",
    ylim: tuple[float, float] = None,
    y_label: str | None = None,
    fs: int = 14,
) -> None:
    """Plot a prerecorded particle history from an SMC experiment.

    Args:
        summary_csv_path: Path to the per-particle summary CSV.
        pool_csv_path: Path to the pool composition CSV.
        run_id: Run identifier used to filter both CSVs.
        save_dir: Directory where the plot is saved.
        image_name: Filename for the saved plot.
        metric_name: Column name of the metric to plot. Defaults to the weighting metric.
        cmap_name: Colormap name for iteration colours. Uses single_color if None.
        single_color: Fallback colour when cmap_name is None.
        lw_full: Line width for lines connecting a new particle to its ancestor.
        axes: Existing axes to draw on; a new figure is created if None.
        s_new: Marker size for particles.
        lw_particle_exists: Line width for dashed lines showing copied-particle lifespan.
        figsize: Figure size when creating a new figure.
        save: If True, save the figure to folder/image_name and close it.
        show: If True and save is False, call plt.show().
        yscale: Scale for the y-axis (e.g. "log", "linear").
        ylim: (bottom, top) y-axis limits. No limit applied if None.
        y_label: Y-axis label. Defaults to metric_name if None.
        fs: Font size for axis labels and tick labels.
    """  # noqa

    with use_style("pyloric"):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = axes.get_figure()

        fig, axes = plot_trajectories_raw(
            summary_csv_path=summary_csv_path,
            pool_csv_path=pool_csv_path,
            run_id=run_id,
            metric_name=metric_name,
            cmap_name=cmap_name,
            single_color=single_color,
            lw_full=lw_full,
            s_new=s_new,
            lw_particle_exists=lw_particle_exists,
            fig=fig,
            ax=axes,
            figsize=figsize,
        )

        axes.set_xlabel("Iteration", fontsize=fs)
        axes.set_ylabel(y_label if y_label is not None else metric_name, fontsize=fs)
        axes.tick_params(axis="both", which="major", labelsize=fs)
        axes.set_yscale(yscale)

        plt.tight_layout()
        if ylim is not None:
            axes.set_ylim(bottom=ylim[0], top=ylim[1])

    if save:
        plt.savefig(os.path.join(save_dir, image_name), bbox_inches="tight")
        plt.close(fig)
    else:
        if show:
            plt.show()
