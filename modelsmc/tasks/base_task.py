import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from modelsmc.tasks.base_embedding_net import BaseEmbeddingHandler

logger = logging.getLogger("ModelSMC")

TASK_REGISTRY = {}


def register_task(name):
    """Class decorator that registers a task under the given name.

    The (case-insensitive) name is used by :func:`create_task` to look up the
    task class from the config.

    Args:
        name: Name under which the decorated task class is registered.

    Returns:
        The class decorator that registers and returns the task class unchanged.
    """

    def decorator(cls):
        TASK_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class BaseTask(ABC):
    """Abstract base class for all simulation tasks.

    This class defines the standard interface that all tasks must implement.
    It provides a consistent structure for task creation, data generation,
    evaluation, and simulation.
    """

    def __init__(
        self,
        config: DictConfig,
        prompts_path: str | None = None,
        base_simulator_path: str | None = None,
    ):
        """Initialize base task with configuration and optional paths.

        Args:
            config: Hydra configuration containing task settings
            prompts_path: Path to prompts.yaml file (optional, for subclass flexibility)
            base_simulator_path: Path to base_simulator.py file (optional,
                for subclass flexibility)
        """
        self.config = config

        try:
            self.save_dir = os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "task"
            )
        except ValueError:
            logger.warning("No save_dir set. Set manually if required.")
            self.save_dir = None

        # These should be set by subclasses
        self.task_description: str | None = None
        self.signature_description: str | None = None
        self.system_description: str | None = None
        self.base_simulator: str | None = None

        # Parameter distributions (should be set by subclasses)
        self.prior_dist = None  # Parameters to be inferred

        # Initialize the embedding network
        self.init_embedding_net()

    def get_context(
        self, mode: str, id: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """
        Return any task-specific context needed for simulations. The returned context
        needs to contain numerical values only.

        Args:
            mode: Mode for which to get the context ('training' or 'validation').
            id: The identifier for the context to retrieve.

        Returns:
            context: The context corresponding to the provided id(s).
        """

        # By default, there is no context for the task
        context = None

        return context

    def sample_context(self, n_samples: int) -> Any:
        """
        Randomly sample contexts from the task's context distribution.

        Args:
            n_samples: Number of contexts to sample.

        Returns:
            contexts: Sampled contexts.
        """

        # By default, there is no context for the task
        contexts = None

        return contexts

    def concatenate_context_data(
        self,
        context: torch.Tensor | None,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate context and data into one tensor. This is required as input for
        NPE. Since data and context may have different dimensionality, this method
        may need to be overridden by the specific task.

        Args:
            context: The context information.
            data: Data tensor. Shape: (batch_size, data_dim)
        Returns:
            combined: Combined context and data tensor.
        """

        # If the context is None, return the data as is
        if context is None:
            logger.debug("No context provided, using parameters only.")
            combined = data

        # If the context is provided, concatenate it with the data. By default
        # along the last dimension.
        else:
            logger.debug(
                f"Concatenating context of shape {context.shape} with data of shape"
                f" {data.shape}."
            )
            assert context.ndim == data.ndim, (
                "Context and data must have the same number of dimensions"
            )
            combined = torch.cat((context, data), dim=1)

        return combined

    def concatenate_context_params(
        self,
        context: torch.Tensor | None,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate context and parameters into one tensor. This is required as input
        for NLE. Since parameters and context may have different dimensionality, this
        method may need to be overridden by the specific task.

        Args:
            context: The context information.
            params: Parameters tensor. Shape: (batch_size, param_dim)

        Returns:
            combined: Combined context and parameters tensor.
        """

        # If the context is None, return the parameters as is
        if context is None:
            logger.debug("No context provided, using parameters only.")
            combined = params

        # If the context is provided, concatenate it with the parameters. By default
        # along the last dimension.
        else:
            logger.debug(
                f"Concatenating context of shape {context.shape} with parameters of"
                f"shape {params.shape}."
            )
            assert context.ndim == params.ndim, (
                "Context and parameters must have the same number of dimensions"
            )
            combined = torch.cat((context, params), dim=1)

        return combined

    def split_context_data(
        self,
        combined: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        This function splits the combined context and data tensor back into context and
        data tensors. I.e. is it the inverse of the concatenate_context_data function.

        Args:
            combined: Combined context and parameters tensor.

        Returns:
            data: Extracted parameters tensor.
            context: Extracted context tensor.
        """

        # By default, there is no context, so return None and the combined tensor
        context = None
        data = combined

        return data, context

    def init_embedding_net(self) -> None:
        """
        This function initializes a torch.nn.Module used to embed observations. The
        default implementation is the identity function, so no embedding is
        applied if not explicitly implemented
        """

        self.embedding_net = BaseEmbeddingHandler(split_x_ctxt=self.split_context_data)

    def get_embedding_net(self) -> Module:
        """
        This function returns the network used to embed the observations.
        """

        return self.embedding_net

    def get_task_description(self) -> str:
        """Return the task description for prompting."""
        if self.task_description is None:
            raise NotImplementedError("task_description must be set by subclass")
        return self.task_description

    def get_signature_description(self) -> str:
        """Return the signature description for prompting."""
        if self.signature_description is None:
            raise NotImplementedError("signature_description must be set by subclass")
        return self.signature_description

    def get_system_description(self) -> str:
        """Return the system description for prompting."""
        if self.system_description is None:
            raise NotImplementedError("system_description must be set by subclass")
        return self.system_description

    def get_base_simulator(self) -> str:
        """Return base simulator code for model implementation."""
        if self.base_simulator is None:
            raise NotImplementedError("base_simulator must be set by subclass")
        return self.base_simulator

    def prepare_likelihood_estimator_data(
        self,
        x_raw: torch.Tensor,
        params: torch.Tensor,
        context_raw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training the likelihood estimator by embedding raw
        observations and concatenating embedded context with parameters.

        Args:
            x_raw: Raw simulation data. Shape: (batch_size, data_dim) or task-specific
            params: Parameters used to generate the simulations.
                Shape: (batch_size, n_params)
            context_raw: Raw context data used in the simulations.
            Shape: (batch_size, context_dim) or None

        Returns:
            x_prepared: Embedded simulation data. Shape: (batch_size, n_summary_stats)
            theta_ctxt_prepared: Concatenation of parameters and embedded context.
            Shape: (batch_size, n_params + context_embedding_dim)
        """

        # Embed the simulations
        x_prepared = self.embedding_net.embed_x(x_raw).detach()

        # Embed the context used in the simulations
        context_prepared = self.embedding_net.embed_context(context_raw)

        # Combine the embedded context and the parameters used to generate the
        # data into one tensor
        theta_ctxt_prepared = self.concatenate_context_params(
            context=context_prepared,
            params=params,
        ).detach()
        return x_prepared, theta_ctxt_prepared

    @abstractmethod
    def eval_function(
        self,
        sim_data: torch.Tensor,
        sim_data_raw: torch.Tensor,
        valid_data: torch.Tensor,
        valid_data_raw: torch.Tensor,
        parameter_estimates: torch.Tensor,
        simulator: Module,
    ) -> dict[str, float]:
        """Evaluate simulator performance against validation data.

        Args:
            sim_data: Summary statistics from simulated data.
                Shape: (batch_size, n_summary_stats)
            sim_data_raw: Raw simulation outputs. Shape: (batch_size, time_steps)
                or task-specific
            valid_data: Summary statistics from validation data.
                Shape: (batch_size, n_summary_stats)
            valid_data_raw: Raw validation data. Shape: (batch_size, time_steps)
                or task-specific
            parameter_estimates: Parameters estimated by the inference method.
                Shape: (batch_size, n_params)
            simulator: The simulator module used to generate the simulated data.

        Returns:
            Dictionary with metric names as keys and their scalar metric values.
        """
        pass

    @abstractmethod
    def get_data(self) -> tuple[Tensor, Tensor]:
        """
        Generate training and validation data for the task. Only raw data is generated,
        as the embedding may not be available at the time of data generation.

        Returns:
            Tuple of (train_data, valid_data) where each is a Tensor containing
            raw observations, shape (n_samples, time_steps) or task-specific
        """
        pass

    @abstractmethod
    def simulation_wrapper(
        self,
        simulator: Callable,
        context: Any = None,
        params: torch.Tensor | None = None,
    ) -> Tensor:
        """Wrapper to run simulator with task-specific interface.

        Args:
            simulator: Simulator function or class to run
            context: Additional context for the simulation like initial conditions
                (if applicable)
            params: Parameters tensor. Shape: (batch_size, n_params)

        Returns:
            Raw simulation traces
        """
        pass

    @abstractmethod
    def plot_observation(
        self,
        valid_data: torch.Tensor | None = None,
        valid_data_raw: torch.Tensor | None = None,
        sim_data: torch.Tensor | None = None,
        sim_data_raw: torch.Tensor | None = None,
        path: str | None = None,
        fig_name: str = "observation.png",
        max_observation: int = 5,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Plot observations and optionally simulated data for visualization.

        Args:
            valid_data: Validation summary statistics (optional)
            valid_data_raw: Raw validation data (optional)
            sim_data: Simulated summary statistics (optional)
            sim_data_raw: Raw simulated data (optional)
            path: Directory to save plot
            fig_name: Filename for saved plot
            max_observation: Maximum number of traces to plot
            kwargs: Additional task-specific plotting options (optional)
        """
        pass


def create_task(config: DictConfig) -> BaseTask:
    """Factory function to create tasks based on config.

    Args:
        config: Hydra configuration containing task settings

    Returns:
        BaseTask: The appropriate task instance

    Raises:
        ValueError: If task name is not recognized
    """
    task_name = config.get("name", "").lower()
    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name](config)
    else:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}"
        )
