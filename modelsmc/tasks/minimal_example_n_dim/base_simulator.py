import logging

import torch
import torch.nn as nn

from modelsmc.tasks.minimal_example_n_dim import GMM_configs
from modelsmc.tasks.minimal_example_n_dim.GMM import GMM

logger = logging.getLogger("ModelSMC")

class DiscoveredSimulator(nn.Module):
    def __init__(self):
        super().__init__()

        # Indicate which GMM config is used to initialize the GMM. Replace the place
        # holder by string operations in downstream tasks.
        __GMM_config_idx = {GMM_CONFIG_IDX}

        config_idx = GMM_configs.configs[__GMM_config_idx]
        covs_list = [torch.tensor(cov) for cov in config_idx.covs]
        means_list = [torch.tensor(mean) for mean in config_idx.means]
        weights_list = config_idx.weights
        dims_to_shift = config_idx.dims_to_shift

        # Initialize the GMM
        self.__gmm = GMM(
            covs_list=covs_list,
            means_list=means_list,
            dims_to_shift=dims_to_shift,
            weights_list=weights_list
        )

    def forward(self, theta):
        return self.__gmm.sample(num_samples = theta.shape[0], theta = theta)
