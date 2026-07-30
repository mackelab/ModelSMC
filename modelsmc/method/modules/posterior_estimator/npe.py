from typing import Any

import torch
from sbi.inference.posteriors import DirectPosterior
from torch import Tensor


class NPEPosteriorWrapper:
    """Wraps SBI's DirectPosterior to expose sample_batched(with_log_prob=True).

    This gives NPE the same combined sample+log_prob interface as NPE-PFN so
    that callers (e.g. the evaluator) can always call
    ``sample_batched(..., with_log_prob=True)`` and receive ``(samples, log_probs)``
    without knowing which backend is being used.
    """

    def __init__(self, posterior: DirectPosterior) -> None:
        self._posterior = posterior

    def sample_batched(
        self,
        sample_shape: torch.Size | tuple,
        x: Tensor,
        show_progress_bars: bool = False,
        with_log_prob: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from the posterior for a batch of observations.

        Args:
            sample_shape:       Desired number of samples (1-D).
            x:                  Batch of observations. Shape: [n_obs, dim_x].
            show_progress_bars: Forwarded to SBI's sampler.
            with_log_prob:      If True, also return log-probabilities of shape
                                [B, n_obs].

        Returns:
            samples of shape [B, n_obs, dim_theta], or a (samples, log_probs)
            tuple when ``with_log_prob=True``.
        """
        samples = self._posterior.sample_batched(
            sample_shape, x=x, show_progress_bars=show_progress_bars, **kwargs
        )
        if not with_log_prob:
            return samples
        log_probs = self._posterior.log_prob_batched(samples, x=x)  # [B, n_obs]
        return samples, log_probs

    def log_prob_batched(self, theta: Tensor, x: Tensor, **kwargs: Any) -> Tensor:
        """Evaluate log p(theta | x) for a batch of observations.

        Args:
            theta: Posterior samples. Shape: [B, n_obs, dim_theta].
            x:     Observations. Shape: [n_obs, dim_x].

        Returns:
            Log-probabilities of shape [B, n_obs].
        """
        return self._posterior.log_prob_batched(theta, x=x, **kwargs)
