########################################################################################
#
# This file has been modified from its original version, available at
#
# https://github.com/mackelab/npe-pfn/blob/1119261fa9c0edc43f742c1e689e22658e253fdd/npe_pfn/support_posterior.py
#
# Authored by: J. Vetter, M. Gloeckler, D. Gedon, J. Macke
#
# under the Prior Labs License, Version 1.1, May 2025. A copy of the License is
# distributed in this directory (see ./LICENSE).
#
########################################################################################


from typing import Any, Mapping

import torch
from sbi.utils import BoxUniform
from torch import Tensor
from torch.distributions import Independent, Uniform
from tqdm.auto import tqdm
import logging

logger = logging.getLogger("ModelSMC")

class PosteriorSupport:
    def __init__(
        self,
        prior: Any,
        posterior: Any,
        obs: Tensor,
        num_samples_to_estimate_support: int = 10_000,
        batch_size_for_estimate_support: int = 10_000,
        allowed_false_negatives: float = 0.0,
        sampling_method: str = "rejection",
        max_iter_rejection: int = 1000,
        oversample_sir: int = 100,
        log_prob_kwargs: Mapping = {},  # optionally pass some stuff depending on the posterior object
    ) -> None:

        self._prior = prior
        self._posterior = posterior
        self._obs = obs
        self._posterior_thr = None

        self.sampling_method = sampling_method
        self.max_iter = max_iter_rejection
        self.oversample_sir = oversample_sir
        self.allowed_false_negatives = allowed_false_negatives

        self._log_prob_kwargs = log_prob_kwargs

        # NOTE: reuse samples to get quantile and constrained support to save time
        if sampling_method == "rejection":
            samples_to_estimate_support = self._posterior.sample(
                (num_samples_to_estimate_support,),
                self._obs,
                max_sampling_batch_size=batch_size_for_estimate_support,
            )

            self.thr = self.tune_threshold(
                samples_to_estimate_support,
                allowed_false_negatives,
                batch_size=batch_size_for_estimate_support,
            )

    def tune_threshold(
        self,
        samples: Tensor,
        allowed_false_negatives: float = 0.0,
        batch_size: int = 10_000,
    ) -> None:

        log_probs = self._posterior.log_prob(
            samples,
            self._obs,
            max_sampling_batch_size=batch_size,
            **self._log_prob_kwargs,
        )

        # any reason why not torch.quantile previously?
        return torch.quantile(log_probs, allowed_false_negatives)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,  # True for now because one cannot set it within `simulate_for_sbi`
        sampling_batch_size: int = 10_000,
        return_acceptance_rate: bool = False,  # TODO could be something like return diagnostic
        return_ess: bool = False,
    ) -> Tensor:

        if self.sampling_method == "rejection":
            return self.sample_rejection(
                sample_shape=sample_shape,
                show_progress_bars=show_progress_bars,
                sampling_batch_size=sampling_batch_size,
                return_acceptance_rate=return_acceptance_rate,
            )
        elif self.sampling_method == "sir":
            return self.sample_sir(
                sample_shape=sample_shape,
                show_progress_bars=show_progress_bars,
                sampling_batch_size=sampling_batch_size,
                return_ess=return_ess,
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def sample_rejection(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,  # True for now because one cannot set it within `simulate_for_sbi`
        sampling_batch_size: int = 10_000,
        return_acceptance_rate: bool = False,
    ) -> Tensor:
        """
        Return samples from the `RestrictedPrior`.
        Samples are obtained by sampling from the prior, evaluating them under the
        trained classifier (`RestrictionEstimator`) and using only those that were
        accepted.
        Args:
            sample_shape: Shape of the returned samples.
            show_progress_bars: Whether or not to show a progressbar during sampling.
            max_sampling_batch_size: Batch size for drawing samples from the posterior.
            return_acceptance_rate: Whether to return the acceptance rate.
        Returns:
            Samples from the `RestrictedPrior`.
        """

        # There is no reason to support other shapes here
        assert len(sample_size := torch.Size(sample_shape)) == 1
        num_samples = sample_size[0]

        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} restricted posterior samples",
        )

        # minimal supported acceptance is num_samples / (max_iter * sampling_batch_size)
        pre_acceptance_rate = 1.0  # overwriting continously is fine
        lower, upper = None, None
        num_sampled_total, num_remaining = 0, num_samples
        accepted = []
        for _ in range(self.max_iter):
            if num_remaining <= 0:
                break

            if lower is None or upper is None:
                candidates = self._prior.sample((sampling_batch_size,))
                log_probs = self._posterior.log_prob(
                    candidates, self._obs, **self._log_prob_kwargs
                )
                lower, upper = self._posterior._get_classifier_bounds()
            else:
                candidates, pre_acceptance_rate = prereject_with_bounds(
                    self._prior, lower, upper, sampling_batch_size
                )
                log_probs = self._posterior.log_prob(
                    candidates, self._obs, **self._log_prob_kwargs
                )
                sanity_lower, sanity_upper = self._posterior._get_classifier_bounds()
                assert torch.allclose(lower, sanity_lower)
                assert torch.allclose(upper, sanity_upper)

            are_accepted_by_classifier = log_probs > self.thr
            samples = candidates[are_accepted_by_classifier.bool()]
            accepted.append(samples)

            num_sampled_total += sampling_batch_size
            num_remaining -= samples.shape[0]
            pbar.update(samples.shape[0])

        pbar.close()

        acceptance_rate = (num_samples - num_remaining) / num_sampled_total

        logger.info(f"Pre-acceptance rate: {pre_acceptance_rate}")
        logger.info(f"Log prob acceptance rate: {acceptance_rate}")
        overall_acceptance_rate = pre_acceptance_rate * acceptance_rate
        logger.info(f"Overall acceptance rate: {overall_acceptance_rate}")

        if num_remaining > 0:
            remaining_samples = self._prior.sample((num_remaining,))
            accepted.append(remaining_samples)
            logger.info(f"Max iter exceeded. Added {remaining_samples} prior samples.")

        samples = torch.cat(accepted)[:num_samples]
        assert samples.shape[0] == num_samples

        if return_acceptance_rate:
            return samples, overall_acceptance_rate
        else:
            return samples

    def sample_sir(
        self,
        sample_shape: torch.Size = torch.Size(),
        show_progress_bars: bool = True,
        sampling_batch_size: int = 10_000,  # divide by oversampling
        return_ess: bool = False,
    ):
        assert len(sample_size := torch.Size(sample_shape)) == 1
        num_samples = sample_size[0]

        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} restricted posterior samples",
        )

        oversampling_factor = self.oversample_sir
        assert sampling_batch_size % oversampling_factor == 0
        sir_batch_size = sampling_batch_size // oversampling_factor

        num_remaining = num_samples
        all_samples = []
        all_ess = []
        while num_remaining > 0:

            # use "free" log probs
            posterior_samples, posterior_log_probs = self._posterior.sample(
                (sampling_batch_size,),
                self._obs,
                max_sampling_batch_size=sampling_batch_size,
                with_log_prob=True,
            )
            truncated_prior_log_probs = self._prior.log_prob(posterior_samples)

            # Adaptive threshold instead of pre-computed
            thr = torch.quantile(posterior_log_probs, self.allowed_false_negatives)
            truncated_prior_log_probs[posterior_log_probs < thr] = -float("inf")

            log_ratios = truncated_prior_log_probs - posterior_log_probs
            log_ratios = torch.nan_to_num(log_ratios, -float("inf"))
            reshaped_ratio = torch.reshape(
                log_ratios, (sir_batch_size, oversampling_factor)
            )
            # Save guard
            probs = torch.exp(
                reshaped_ratio - torch.logsumexp(reshaped_ratio, dim=1, keepdim=True)
            )

            all_ess.append(1.0 / torch.sum(probs**2, dim=1))

            cat_dist = torch.distributions.Categorical(logits=reshaped_ratio)
            categorical_samples = cat_dist.sample((1,))[0, :]
            reshaped_posterior_samples = torch.reshape(
                posterior_samples, (sir_batch_size, self.oversample_sir, -1)
            )
            selected_posterior_samples = reshaped_posterior_samples[
                torch.arange(sir_batch_size), categorical_samples
            ]

            all_samples.append(selected_posterior_samples)
            num_remaining -= sir_batch_size
            pbar.update(sir_batch_size)

        pbar.close()

        samples = torch.cat(all_samples)[:num_samples]
        assert samples.shape[0] == num_samples

        ess = torch.cat(all_ess)
        logger.info(f"Mean ESS: {ess.mean().item()}")
        logger.info(f"Min ESS: {ess.min().item()}")
        if return_ess:
            return samples, ess
        else:
            return samples


# Some utils for pre-rejection and filtering


def prereject_with_bounds(
    proposal: Any,  # proposal distribution with the typical sample method
    lower_bound: Tensor,
    upper_bound: Tensor,
    sampling_batch_size: int = 10_000,
    pre_sampling_batch_size: int = 1_000_000,  # hard coded atm
):
    """
    Pre-reject samples that are not in the support of the posterior.
    Args:
        lower_bound: Lower bound of the support of the posterior.
        upper_bound: Upper bound of the support of the posterior.
    Returns:
        Samples from the proposal that are in the support of the posterior.
    """
    is_uniform = check_for_uniform(proposal)

    num_pre_accepted = 0
    num_sampled_total = 0
    pre_samples = []
    while num_pre_accepted < sampling_batch_size:
        samples = proposal.sample((pre_sampling_batch_size,))
        within_bounds = torch.all(
            (samples >= lower_bound) & (samples <= upper_bound), dim=1
        )
        samples = samples[within_bounds.bool()]
        pre_samples.append(samples)

        num_pre_accepted += samples.shape[0]
        num_sampled_total += pre_sampling_batch_size

        if is_uniform:
            break  # perform one iteration to estimate pre acceptance rate

    pre_acceptance_rate = num_pre_accepted / num_sampled_total

    if is_uniform:
        prop_lower_bound, prop_upper_bound = get_uniform_bounds(proposal)
        max_lower = torch.max(lower_bound, prop_lower_bound)
        min_upper = torch.min(upper_bound, prop_upper_bound)
        return (
            BoxUniform(max_lower, min_upper).sample((sampling_batch_size,)),
            pre_acceptance_rate,
        )
    else:
        return torch.cat(pre_samples)[:sampling_batch_size], pre_acceptance_rate


def check_for_uniform(proposal: Any):
    if isinstance(proposal, BoxUniform):
        return True
    if isinstance(proposal, Independent) and isinstance(proposal.base_dist, Uniform):
        return True
    # anything else?
    return False


def get_uniform_bounds(proposal):
    # anything else?
    return proposal.base_dist.low, proposal.base_dist.high


# NOTE filter functions should always return (theta, x) in that order
def get_filtering_method(name: str):
    if name == "no_filtering":
        return no_filtering
    elif name == "latest_filtering":
        return latest_filtering
    elif name == "random_filtering":
        return random_filtering
    elif name == "standardized_euclidean_filtering":
        return standardized_euclidean_filtering
    elif callable(name):
        return name
    else:
        raise ValueError(f"Unknown filtering method: {name}")


def no_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    return theta, x


def latest_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    # assumes that the latest samples are at the end
    return theta[-context_size:], x[-context_size:]


def random_filtering(obs: Tensor, theta: Tensor, x: Tensor, context_size: int):
    num_samples = theta.shape[0]
    perm = torch.randperm(num_samples)
    return theta[perm[:context_size]], x[perm[:context_size]]


def standardized_euclidean_filtering(
    obs: Tensor, theta: Tensor, x: Tensor, context_size: int
):
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x_s = (x - x_mean) / x_std

    obs_s = (obs - x_mean) / x_std

    dists = torch.norm(x_s - obs_s, dim=1)

    _, idx = torch.topk(dists, min(context_size, dists.shape[0]), largest=False)
    return theta[idx], x[idx]
