"""Standalone script to (re)generate SIR training and validation data.

Reproducibility note:
    This script does NOT reproduce a previously generated dataset bit-for-bit. The
    ``--seed`` only seeds the legacy ``np.random`` state, which fixes the sampled
    initial states. The simulator dynamics (``SimulatorStep.step`` in external_code/
    env.py) draw from a fresh, unseeded ``np.random.default_rng()`` on every step, so
    the stochastic trajectories differ on each run. Re-running therefore produces
    *new* SIR data.

Usage:
    python -m modelsmc.tasks.SIR.level3.generate_data --seed 42
    python -m modelsmc.tasks.SIR.level3.generate_data --seed 42 --num_train 1 --num_valid 100
"""  # noqa E501

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from modelsmc.tasks.SIR.external_code.env import simulate, trajectories_to_numpy
from modelsmc.tasks.SIR.SIR_task_base import SIRBaseTask


def main() -> None:
    """Generate, save and plot the SIR train/validation data for level3."""
    parser = argparse.ArgumentParser(
        description="Pre-generate SIR data with a dedicated seed."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data generation."
    )
    parser.add_argument(
        "--num_train", type=int, default=1, help="Number of training trajectories."
    )
    parser.add_argument(
        "--num_valid", type=int, default=100, help="Number of validation trajectories."
    )
    args = parser.parse_args()

    # Set seeds before any data generation
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate data using the existing simulate function
    train_trajectories, _ = simulate(n=args.num_train)
    val_trajectories, _ = simulate(n=args.num_valid)

    # Convert to numpy arrays with shape [n, time_steps, 3]
    train_data_np = trajectories_to_numpy(train_trajectories)
    val_data_np = trajectories_to_numpy(val_trajectories)

    # Convert to tensors
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    val_data = torch.tensor(val_data_np, dtype=torch.float32)

    # Ensure both tensors are 3D [n, time_steps, 3]. trajectories_to_numpy squeezes the
    # instance dimension when n == 1, so add it back for downstream code that indexes it
    if train_data.ndim == 2:
        train_data = train_data.unsqueeze(0)
    if val_data.ndim == 2:
        val_data = val_data.unsqueeze(0)

    # Save to disk
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    torch.save(train_data, os.path.join(data_dir, "train_data.pt"))
    torch.save(val_data, os.path.join(data_dir, "valid_data.pt"))

    metadata = {
        "seed": args.seed,
        "num_train": args.num_train,
        "num_valid": args.num_valid,
        "train_shape": list(train_data.shape),
        "valid_shape": list(val_data.shape),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved train_data.pt {list(train_data.shape)} and valid_data.pt "
        f"{list(val_data.shape)} to {data_dir}"
    )

    # Plot validation data
    # Flatten to match _plot_observation's expected input format
    val_data_flat = val_data.reshape(val_data.shape[0], -1)
    SIRBaseTask._plot_observation(
        valid_data_raw=val_data_flat,
        path=data_dir,
        fig_name="observation_validation.png",
    )

    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    SIRBaseTask._plot_observation(
        valid_data_raw=train_data_flat,
        path=data_dir,
        fig_name="observation_training.png",
    )


if __name__ == "__main__":
    main()
