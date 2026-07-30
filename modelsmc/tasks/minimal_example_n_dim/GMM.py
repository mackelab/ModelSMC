import numpy as np
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal


class GMM:
    """Gaussian Mixture Model with optional per-dimension shift and global scaling.

    The mixture components are fixed multivariate normals. ``sample`` applies a
    parameterised transformation (a shift along selected dimensions and a global
    scale) to the drawn samples; ``log_prob_uncond`` evaluates the untransformed
    mixture density.
    """

    def __init__(
        self,
        covs_list: list[Tensor],
        means_list: list[Tensor],
        dims_to_shift: list[int],
        weights_list: list[float] | None = None,
    ):
        """Initialize the GMM.

        Args:
            covs_list: Covariance matrix of each component, each of shape (dim, dim).
            means_list: Mean of each component, each of shape (dim,).
            dims_to_shift: Indices of the dimensions that can be shifted via the
                transformation parameters.
            weights_list: Mixture weights. Defaults to uniform weights.
        """

        self.dim = means_list[0].shape[0]

        assert len(covs_list) == len(means_list), (
            "Covariance and mean lists must have the same length."
        )
        for cov in covs_list:
            assert cov.shape[0] == cov.shape[1], "Covariance matrices must be square."
            assert cov.shape[0] == means_list[0].shape[0], (
                "Covariance matrices and means must have compatible dimensions."
            )

        self.__covs_list = covs_list
        self.__means_list = means_list
        self.__weights_list = (
            weights_list
            if weights_list is not None
            else [1.0 / len(means_list)] * len(means_list)
        )
        self.__components = [
            MultivariateNormal(loc=mean.squeeze(), covariance_matrix=cov)
            for mean, cov in zip(self.__means_list, self.__covs_list, strict=True)
        ]
        self.dims_to_shift = dims_to_shift
        self.num_dims_to_shift = len(dims_to_shift)

    def sample(self, num_samples: int, theta: Tensor) -> Tensor:
        """
        Sample from the GMM with transformation parameters theta.

        Args:
            num_samples: Number of samples to draw.
            theta: Tensor of shape (num_samples, n_dims_to_shift + 1) or
                (1, n_dims_to_shift + 1), where each row contains the shifts along the
                selected dimensions and a global scale factor.

        Returns:
            samples: Tensor of shape (num_samples, dim) containing the sampled points.

        Note:
            The returned samples are grouped by mixture component rather than in draw
            order, so they should be treated as an unordered set. When ``theta`` rows
            differ, do not assume ``samples[i]`` was generated from ``theta[i]``.
        """

        assert theta.shape == torch.Size(
            [1, 1 + self.num_dims_to_shift]
        ) or theta.shape == torch.Size([num_samples, 1 + self.num_dims_to_shift])

        # Use the same set of parameters for all samples
        if theta.shape == torch.Size([1, 1 + self.num_dims_to_shift]):
            theta = theta.repeat(num_samples, 1)

        # Get the shift of the entire distribution and the global scale
        shift_reduced = theta[:, : self.num_dims_to_shift]
        shift = torch.zeros((num_samples, self.dim))
        shift[:, self.dims_to_shift] = shift_reduced

        global_scale = theta[:, self.num_dims_to_shift].unsqueeze(1)

        assert shift.shape == (num_samples, self.dim)
        assert global_scale.shape == (num_samples, 1)

        # Get the number of samples drawn from each component based on the weights
        num_components = len(self.__components)
        component_assignments = np.random.choice(
            num_components, size=num_samples, p=self.__weights_list
        )
        component_idx, samples_per_component = np.unique(
            component_assignments, return_counts=True
        )

        samples = []

        for idx in range(len(component_idx)):
            sample_i = self.__components[component_idx[idx]].sample(
                (samples_per_component[idx],)
            )
            samples.append(sample_i)

        samples_raw = torch.cat(samples, dim=0)
        assert samples_raw.shape == (num_samples, self.dim)

        # Apply the transformation to the samples
        samples_transformed = (samples_raw * global_scale) + shift
        assert samples_transformed.shape == (num_samples, self.dim)

        return samples_transformed

    def log_prob_uncond(self, x: Tensor) -> Tensor:
        """
        Compute the log probability of data points x under the (untransformed) GMM.

        Args:
            x: Tensor of shape (num_points, dim) containing the data points.

        Returns:
            log_probs: Tensor of shape (num_points,) containing the log probabilities.
        """

        num_points = x.shape[0]
        log_probs_components = torch.zeros((num_points, len(self.__components)))

        for idx, component in enumerate(self.__components):
            log_probs_components[:, idx] = component.log_prob(x) + torch.log(
                torch.tensor(self.__weights_list[idx])
            )

        # Use log-sum-exp trick for numerical stability
        log_probs = torch.logsumexp(log_probs_components, dim=1)

        assert log_probs.shape == (num_points,)

        return log_probs
