import ast
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelsmc.utils.plot_utils import use_style

logger = logging.getLogger("ModelSMC")


def plot_pool_raw(
    pool_csv_path: str,
    summary_csv_path: str,
    run_id: str,
    cmap_name: str = "viridis",
    fs: int = 14,
    fig=None,
    ax=None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple:
    """Core plotting logic for pool composition heatmap from pool_composition.csv.

    Reads the pool composition CSV and summary CSV to build a 2D count matrix
    where each cell encodes how many times a given particle appears in the pool
    at a given iteration.  The matrix is rendered as a heatmap via
    ``ax.pcolormesh``.

    Args:
        pool_csv_path:    Path to ``pool_composition.csv`` produced by the run.
        summary_csv_path: Path to ``summary.csv`` produced by the run.
        run_id:           Run identifier used to filter rows in both CSVs.
        cmap_name:        Matplotlib colormap name for the heatmap.
        fs:               Font size applied to all text elements.
        fig:              Existing figure; a new one is created when ``None``.
        ax:               Existing axes; a new one is created when ``None``.
        figsize:          Figure size used when creating a new figure.

    Returns:
        fig, ax: The figure and axes objects with the plotted heatmap.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Load pool composition CSV
    try:
        pool_df = pd.read_csv(pool_csv_path)
    except FileNotFoundError:
        logger.error(f"pool_composition.csv not found at: {pool_csv_path}")
        return fig, ax

    pool_run = pool_df[pool_df["run_id"] == run_id].copy()
    if pool_run.empty:
        logger.warning(f"No pool data found for run_id={run_id!r} in {pool_csv_path}")
        return fig, ax

    # Parse pool_members_uuids from string repr of Python list
    pool_run["pool_members_uuids"] = pool_run["pool_members_uuids"].apply(
        ast.literal_eval
    )

    # Load summary CSV for uuid -> particle_index mapping
    try:
        summary_df = pd.read_csv(summary_csv_path)
    except FileNotFoundError:
        logger.error(f"summary.csv not found at: {summary_csv_path}")
        return fig, ax

    summary_run = summary_df[summary_df["run_id"] == run_id].copy()
    if summary_run.empty:
        logger.warning(
            f"No summary data found for run_id={run_id!r} in {summary_csv_path}"
        )
        return fig, ax

    # Build uuid -> particle_index lookup (use first occurrence for deduplication)
    uuid_to_idx = (
        summary_run.drop_duplicates(subset="particle_id")
        .set_index("particle_id")["particle_index"]
        .to_dict()
    )

    # Build count records: (pool_itr, particle_index, normalized_count)
    records = []
    for _, row in pool_run.iterrows():
        pool_itr = int(row["itr"])
        uuids = row["pool_members_uuids"]
        pool_size = len(uuids)
        # Count multiplicity of each UUID at this iteration
        uuid_counts: dict[str, int] = {}
        for uuid in uuids:
            uuid_counts[uuid] = uuid_counts.get(uuid, 0) + 1
        for uuid, count in uuid_counts.items():
            pidx = uuid_to_idx.get(uuid)
            if pidx is None:
                logger.debug(f"UUID {uuid!r} not found in summary.csv; skipping.")
                continue
            records.append(
                {
                    "pool_itr": pool_itr,
                    "particle_index": int(pidx),
                    "count": count / pool_size,
                }
            )

    if not records:
        logger.warning("No valid pool records found after UUID resolution.")
        return fig, ax

    counts_df = pd.DataFrame(records)

    # Pivot to 2D array: rows = particle_index, cols = pool_itr
    pool_iterations = sorted(counts_df["pool_itr"].unique())
    particle_indices = sorted(counts_df["particle_index"].unique())

    grid = np.zeros((len(particle_indices), len(pool_iterations)), dtype=float)
    pidx_pos = {pidx: i for i, pidx in enumerate(particle_indices)}
    itr_pos = {itr: j for j, itr in enumerate(pool_iterations)}

    for _, row in counts_df.iterrows():
        i = pidx_pos[row["particle_index"]]
        j = itr_pos[row["pool_itr"]]
        grid[i, j] = row["count"]

    # Mask zero-count cells so they render as white regardless of colormap
    masked_grid = np.ma.masked_where(grid == 0, grid)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    # Plot heatmap
    mesh = ax.pcolormesh(
        np.arange(len(pool_iterations) + 1),
        np.arange(len(particle_indices) + 1),
        masked_grid,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Fraction of pool", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)

    # Set x-ticks at cell centres (one per iteration)
    ax.set_xticks(np.arange(len(pool_iterations)) + 0.5)
    ax.set_xticklabels(pool_iterations, fontsize=fs)

    # Show ~10% of y-ticks, evenly spaced, to avoid label overlap
    n_yticks = max(1, len(particle_indices) // 10)
    step = max(1, len(particle_indices) // n_yticks)
    shown_positions = np.arange(0, len(particle_indices), step)
    ax.set_yticks(shown_positions + 0.5)
    ax.set_yticklabels([particle_indices[i] for i in shown_positions], fontsize=fs)

    return fig, ax


def plot_pool(
    pool_csv_path: str,
    summary_csv_path: str,
    run_id: str,
    image_name: str = "pool.png",
    save_dir: str = None,
    cmap_name: str = "YlOrRd",
    show: bool = False,
    fs: int = 14,
    fig=None,
    ax=None,
    figsize: tuple[float, float] = (10, 5),
) -> tuple:
    """Plot a heatmap of pool composition over iterations for a given run.

    Reads ``pool_composition.csv`` and ``summary.csv``, filters by ``run_id``,
    and renders a heatmap where the x-axis is the pool iteration, the y-axis is
    the particle index, and each cell's colour encodes how many copies of that
    particle exist in the pool at that iteration.  Absent particles have count 0
    and receive the lowest colour value.

    The figure is saved to ``save_dir`` when provided, or displayed when
    ``show=True``.

    Args:
        pool_csv_path:    Path to ``pool_composition.csv``.
        summary_csv_path: Path to ``summary.csv``.
        run_id:           Run identifier to filter.
        image_name:       Filename for the saved figure.
        save_dir:         Directory in which to save the figure.
        cmap_name:        Matplotlib colormap for the heatmap cells.
        show:             Call ``plt.show()`` when ``save_dir`` is ``None``.
        fs:               Font size for axis labels and tick labels.
        fig:              Existing figure to draw on.
        ax:               Existing axes to draw on.
        figsize:          Figure size when creating a new figure.

    Returns:
        fig, ax: The figure and axes objects with the plotted heatmap.
    """
    with use_style("pyloric"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        fig, ax = plot_pool_raw(
            pool_csv_path=pool_csv_path,
            summary_csv_path=summary_csv_path,
            run_id=run_id,
            cmap_name=cmap_name,
            fs=fs,
            fig=fig,
            ax=ax,
            figsize=figsize,
        )

        ax.set_xlabel("Pool iteration", fontsize=fs)
        ax.set_ylabel("Particle index", fontsize=fs)

        plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, image_name), bbox_inches="tight")
        plt.close(fig)
    elif show:
        plt.show()

    return fig, ax
