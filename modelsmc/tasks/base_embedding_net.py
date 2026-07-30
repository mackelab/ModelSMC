import logging
from typing import Callable, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("ModelSMC")

class BaseEmbeddingHandler(nn.Identity):
    r"""Base class representing the structure of the used embedding functions.

    Its main feature is that it can embed both observations and additional context
    information. There are separate embedding functions for observations and context.
    These embedding functions can be customized in subclasses and even be learned
    during training of a NPE model.

    The reason for this is the following: During the parameter estimation with for
    example NPE the following density is approximated:

    $$p(\theta|x, c)$$

    In the next step, the computation of the weights for the resampling step, the
    following density is required:

    $$p(x| \theta, c)$$

    Since the context is in the conditioning of both densities, it is required that
    context and observations are embedded separately.

    Since sbi does not support context-aware embedding functions natively, the context
    and observations are concatenated and passed as a single tensor to the embedding
    function. Internally the two properties are separated again and for each part the
    respective embedding networks are applied.

    To correctly split the combined tensor into observations and context, a callable
    `split_x_ctxt` has to be provided that takes the combined tensor and returns the
    observations and context separately.
    """

    def __init__(
        self,
        split_x_ctxt: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor | None]],
    ) -> None:
        """
        Args:
            split_x_ctxt: A callable that takes the combined tensor and splits it into
                observations and context, returning them as a tuple. The context is
                ``None`` if the task has no context. An individual embedding function
                can be trained for the observations and the context.
        """

        super().__init__()

        # Callable to split the combined tensor into observations and context
        self.split_x_ctxt = split_x_ctxt

        # Embedding function for embedding the observations
        self.obs_embedding_function = nn.Identity()

        # Embedding function for embedding the context
        self.context_embedding_function = nn.Identity()

    def forward(self, x_ctxt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that embeds both observations and context. For learnable
        embeddings this is the path sbi calls during NPE training; for fixed
        embeddings it is used to pre-embed the data once (see
        :meth:`FixedEmbeddingHandler.prepare_for_sbi`).

        Args:
            x_ctxt: Combined tensor of observations and context.

        Returns:
            combined_embedding: Embedded representation combining both observations
                and context. If the task has no context (``split_x_ctxt`` returns
                ``None`` for the context), only the embedded observations are
                returned.
        """

        # Split the context and the observations using a task-specific function.
        x, context = self.split_x_ctxt(x_ctxt)

        # Embed the observations
        embedded_obs = self.embed_x(x)
        # ensure that x and embedded_obs are on the same device (in case embed_x is not nn.Module)
        if x.device != embedded_obs.device:
            embedded_obs = embedded_obs.to(x.device)

        # In case no context is provided, return only the embedded observations.
        if context is None:
            return embedded_obs

        # Embed the context
        embedded_context = self.embed_context(context)
        # ensure context and observations share a device before concatenation
        # (in case embed_context is not an nn.Module)
        if embedded_context.device != embedded_obs.device:
            embedded_context = embedded_context.to(embedded_obs.device)

        # Check if the two embeddings have matching dimensions
        assert embedded_obs.shape[0] == embedded_context.shape[0], \
            "Batch size of observations and context must match."

        assert embedded_obs.ndim == embedded_context.ndim, \
            "Embedded observations and context must have the same number of dimensions."

        # Combine the two embeddings by concatenation
        combined_embedding = torch.cat((embedded_obs, embedded_context), dim=-1)

        return combined_embedding

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """Embed only the observations.

        Args:
            x: Observations tensor.

        Returns:
            embedded_obs: Embedded observations.
        """

        embedded_obs = self.obs_embedding_function(x)
        return embedded_obs

    def embed_context(self, context: torch.Tensor | None) -> torch.Tensor | None:
        """Embed only the context.

        Args:
            context: Context tensor, or ``None`` if the task has no context.

        Returns:
            embedded_context: Embedded context, or ``None`` if ``context`` is ``None``.
        """

        if context is None:
            return None

        embedded_context = self.context_embedding_function(context)
        return embedded_context

    def prepare_for_sbi(self, x_ctxt_raw: torch.Tensor) -> Tuple[torch.Tensor, nn.Module]:
        """Prepare the data and embedding net handed to sbi for NPE training.

        Default (learnable): pass the raw data and ``self`` so sbi trains the
        embedding jointly with the density estimator.

        Args:
            x_ctxt_raw: Combined tensor of raw (un-embedded) observations and context.

        Returns:
            data_for_sbi: The data tensor passed to sbi. Here the raw data is returned
                unchanged.
            embedding_net_for_sbi: The embedding net passed to sbi (``self``), so the
                embedding is trained jointly.
        """

        logger.debug("Learnable embedding function: Do not modify x_ctxt_raw for NPE training.")

        return x_ctxt_raw, self

    def get_trained_embedding_net(self, sbi_embedding_net: nn.Module) -> nn.Module:
        """Return the embedding net to store on the task after NPE training.

        Default (learnable): use sbi's trained net, which may include a z-score
        wrapper added by sbi.

        Args:
            sbi_embedding_net: The embedding net returned by sbi after training.

        Returns:
            embedding_net: The embedding net to store on the task. Here sbi's trained
                net is returned unchanged.
        """

        logger.debug("Learnable embedding function: Do not modify the trained embedding function")

        return sbi_embedding_net

class FixedEmbeddingHandler(BaseEmbeddingHandler):
    """Embedding handler for fixed (non-learnable) summary statistics.

    Pre-embeds training data once before passing to sbi, avoiding redundant
    recomputation of the deterministic transform on every training mini-batch.

    The constructor is inherited from :class:`BaseEmbeddingHandler`; subclasses set
    the (non-learnable) ``obs_embedding_function`` and ``context_embedding_function``
    after calling ``super().__init__``.
    """

    def prepare_for_sbi(self, x_ctxt_raw: torch.Tensor) -> Tuple[torch.Tensor, nn.Module]:
        """Pre-embed the data once and pass ``nn.Identity()`` to sbi.

        Because the summary statistics are fixed (non-learnable), the deterministic
        embedding is applied a single time here instead of being recomputed on every
        training mini-batch.

        Args:
            x_ctxt_raw: Combined tensor of raw (un-embedded) observations and context.

        Returns:
            data_for_sbi: The pre-embedded data tensor passed to sbi.
            embedding_net_for_sbi: ``nn.Identity()``, so sbi does not recompute the
                embedding during training.
        """

        x_ctxt_embedded = self.forward(x_ctxt_raw).detach()

        logger.debug(f"Fixed embedding function: Prepare data for NPE training. Embed x_ctxt_raw of shape {x_ctxt_raw.shape}. Output tensor has shape {x_ctxt_embedded.shape}.")

        return x_ctxt_embedded, nn.Identity()

    def get_trained_embedding_net(self, sbi_embedding_net: nn.Module) -> nn.Module:
        """Restore the original fixed handler after NPE training.

        sbi's ``nn.Identity()`` (passed in :meth:`prepare_for_sbi`) is not useful
        downstream, so the original fixed embedding handler is restored instead.

        Args:
            sbi_embedding_net: The embedding net returned by sbi after training (an
                ``nn.Identity()`` for the fixed handler).

        Returns:
            embedding_net: ``self``, the original fixed embedding handler.
        """

        logger.debug(f"Fixed embedding function: Restore the embedding net from type {type(sbi_embedding_net)} to {type(self)}.")
        return self
