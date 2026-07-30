"""Standalone script to generate Gaussian-mixture-model (GMM) configurations.

The generated configurations define the candidate models for the
``minimal_example_n_dim`` task. They are written to ``gmm_configs.json`` in this
package directory, which is where the task loads them from.

Usage:
    python modelsmc/tasks/minimal_example_n_dim/generate_models.py --new_config
"""

import argparse
import os
import random

import numpy as np
import torch
from pydantic import BaseModel


class GMMConfig(BaseModel):
    means: list[list[float]]
    covs: list[list[list[float]]]
    weights: list[float]
    dims_to_shift: list[int]


class GMMConfigs(BaseModel):
    configs: list[GMMConfig]


def main() -> None:
    """Parse CLI arguments and generate the GMM configurations."""
    parser = argparse.ArgumentParser(
        description="Generate GMM configurations and save to JSON"
    )
    parser.add_argument(
        "--num_configs",
        type=int,
        default=20,
        help="Number of GMM configurations to generate",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--new_config",
        action="store_true",
        help="Generate new configurations and overwrite gmm_configs.json",
    )
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality of the GMM")
    parser.add_argument(
        "--max_num_modes",
        type=int,
        default=10,
        help="Maximum number of modes in the GMM",
    )
    parser.add_argument(
        "--min_num_modes",
        type=int,
        default=1,
        help="Minimum number of modes in the GMM",
    )
    parser.add_argument(
        "--num_dims_shifts", type=int, default=4, help="Number of dimensions to shift"
    )

    args = parser.parse_args()
    num_gmm_configs = args.num_configs
    seed = args.seed

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not args.new_config:
        print("Nothing to do: pass --new_config to (re)generate gmm_configs.json.")
        return

    print("Generating new GMM configurations...")

    # Min and max number of modes in the GMM
    min_num_modes = args.min_num_modes
    max_num_modes = args.max_num_modes
    dim = args.dim

    # Range for the means of each component
    means_range = [-5, 5]

    configs = []

    for i in range(num_gmm_configs):
        # Randomly choose the number of modes
        num_modes = torch.randint(
            low=min_num_modes, high=max_num_modes + 1, size=(1,)
        ).item()

        print(
            f"Generating configuration {i + 1}/{num_gmm_configs} with {num_modes} "
            "modes"
        )

        means_list = []
        covs_list = []

        # Get the weights of the components
        weight_values = np.random.rand(num_modes)
        weights_list = (weight_values / np.sum(weight_values)).tolist()
        weights_list[-1] = 1.0 - np.sum(weights_list[:-1])  # Ensure sum to 1
        assert np.isclose(np.sum(weights_list), 1.0), "Weights do not sum to 1"

        for _j in range(num_modes):
            # Get the mean of the component
            means_j = (
                np.random.rand(dim) * (means_range[1] - means_range[0])
                + means_range[0]
            )
            means_list.append(means_j.tolist())

            # Get the covariance of the component. Rescale the uniform samples to
            # cover the range [-2, 2] to get a wider variety of covariances.
            A = torch.rand(dim, dim) * 4 - 2
            cov_j = torch.mm(A, A.t()) * (torch.rand(1).item() + 1.0)
            covs_list.append(cov_j.tolist())

        # Randomly choose which dimensions can be shifted using parameters later in
        # the inference
        dims_to_shift = random.sample(range(dim), args.num_dims_shifts)

        # Create a GMMConfig instance
        config_i = GMMConfig(
            means=means_list,
            covs=covs_list,
            weights=weights_list,
            dims_to_shift=dims_to_shift,
        )
        configs.append(config_i)

    gmm_configs = GMMConfigs(configs=configs)

    # Save to a JSON file next to this module, where the task loads it from
    output_path = os.path.join(os.path.dirname(__file__), "gmm_configs.json")
    with open(output_path, "w") as f:
        f.write(gmm_configs.model_dump_json(indent=4))
    print(f"Saved {len(configs)} GMM configurations to {output_path}")


if __name__ == "__main__":
    main()
