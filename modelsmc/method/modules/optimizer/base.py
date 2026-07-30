import logging
import os
import time
import traceback
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from modelsmc.tasks.base_task import BaseTask

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Distribution

logger = logging.getLogger("ModelSMC")


class OptimizerBase:
    """Base optimizer class with unified common functionality."""

    def __init__(self, config: DictConfig, task: "BaseTask", verbose: bool) -> None:
        """
        Args:
            config:  Optimizer-level Hydra config (``config.method.optimizer``).
            task:    Task object providing the prior, embedding net, and simulation
                     wrapper.
            verbose: Controls progress logging.
        """
        self.config = config
        self.task = task
        self.verbose = verbose
        self.save_dir = None
        self.num_simulations = config.num_simulations
        self.optimizer_type = None
        self.optimize_output = (
            None  # Store optimization results for parameter estimation
        )

    def optimize(
        self,
        error_dict: dict | None,
        simulator: Callable,
        training_data_raw: Tensor,
        save_dir: str,
        valid_data_raw: Tensor | None = None,
    ) -> tuple[dict, Tensor | None]:
        """
        Main optimization method with common setup, calls _optimize_specific for
        algorithm-specific logic.

        Args:
            error_dict:        Upstream error dict from code generation; if not None the
                               optimization is skipped and errors are forwarded.
            simulator:         Callable simulator module to optimise.
            training_data_raw: Raw observed training data used for parameter estimation.
            save_dir:          Directory where outputs (parameter estimates, plots) are
                               saved.
            valid_data_raw:    Optional validation data; currently unused, reserved for
                               future use.

        Returns:
            tuple[dict, Tensor | None]: output dict and MAP parameter estimates
                                        (or None).
        """
        self.save_dir = save_dir

        # Create base output dictionary
        output_dict = self._create_base_output_dict(self.optimizer_type)

        # Handle existing errors
        if self._handle_error_dict(output_dict, error_dict):
            return output_dict, None

        # Set up simulation wrapper
        simulation_wrapper = self._setup_simulation_wrapper(simulator)

        # Sample from prior
        prior_samples, prior = self._sample_from_prior()

        # Get additional context from the task specific distribution. This could for
        # example be initial states or other auxiliary information needed to run the
        # simulations.
        context_samples = self._sample_context()

        # Run simulations for the sampled parameters and contexts
        sim_data_raw, is_success = self._run_simulations(
            simulation_wrapper=simulation_wrapper,
            prior_samples=prior_samples,
            output_dict=output_dict,
            context_samples=context_samples,
        )
        if not is_success:
            return output_dict, None

        # Reset the embedding network to ensure that the training does not start
        # using the embedding network trained in a previous iteration as the task
        # object persists over the entire experiment. This resetting is moved here to
        # ensure that the embedding network is reset before any training is done and not
        # after wards which would overwrite a trained embedding network.
        self.task.init_embedding_net()

        # Train the embedding network if learnable summary statistics are used and a
        # specific train method is implemented. This can be relevant in cases where the
        # summary network can not trained jointly with the posterior estimator, e.g.
        # when using NPE-PFN. The specific training method is task specific.
        if (
            hasattr(self.task.embedding_net, "train_embedding")
            and hasattr(self.config, "train_sss")
            and self.config.train_sss
        ):
            logger.info("Training sufficient summary statistics...")
            train_loss, val_loss = self.task.embedding_net.train_embedding(
                x=sim_data_raw,
                theta=prior_samples,
                ctxt=context_samples,
                save_dir=self.save_dir,
                plotting=self.config.plotting,
            )

        # Call algorithm-specific optimization
        try:
            output_dict = self._optimize_specific(
                prior_samples=prior_samples,
                context_samples=context_samples,
                sim_data_raw=sim_data_raw,
                prior=prior,
                output_dict=output_dict,
            )

            # Store the output for parameter estimation
            self.optimize_output = output_dict

            # Compute parameter estimates if training data is provided
            self.task.embedding_net.eval()

            if (
                training_data_raw is not None  # Valid training data
                and self.task is not None  # Valid task
                and self.config.estimate_parameters  # Parameters are required for each particle # noqa: E501
            ):
                logger.debug("Estimate parameters")

                time_start = time.time()

                parameter_est, optimizer_data, error_msg = self.get_parameter_estimates(
                    training_data_raw, self.task
                )

                time_end = time.time()
                dt = time_end - time_start

                if parameter_est is not None:
                    output_dict["optimizer_data"] = optimizer_data
                    output_dict["param_estim_time"] = dt
                else:
                    output_dict["errors"] = error_msg
                    output_dict["error_type"] = "Parameter estimation failed"
                    return output_dict, None

                if (
                    parameter_est is not None
                    and isinstance(parameter_est, torch.Tensor)
                    and parameter_est.device.type == "cuda"
                ):
                    parameter_est = parameter_est.detach().cpu()

                # Save the parameter estimates for the best particle
                if self.config.save_parameter_estimates:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir, exist_ok=True)

                    torch.save(
                        parameter_est,
                        os.path.join(self.save_dir, "parameter_estimates.pt"),
                    )

                return output_dict, parameter_est

            # No parameters are estimated
            else:
                logger.debug("No parameters are estimated")
                output_dict["param_estim_time"] = 0.0
                return output_dict, None

        except Exception as e:
            logger.warning(
                f"Failed to run {self.optimizer_type} optimization: "
                f"{type(e).__name__}: {e}"
            )
            output_dict["errors"] = f"{type(e).__name__}: {e}"
            output_dict["error_type"] = f"{self.optimizer_type} optimization failed"

            return output_dict, None

    def _optimize_specific(
        self,
        prior_samples: Tensor,
        context_samples: Tensor | None,
        sim_data_raw: Tensor,
        prior: Distribution,
        output_dict: dict,
    ) -> dict:
        """Perform algorithm-specific optimization and return updated output_dict. To be
        implemented by subclasses.

        Args:
            prior_samples:   Parameter samples drawn from the prior.
                             Shape: [N, dim_theta].
            context_samples: Optional context samples for simulations.
                             Shape: [N, dim_context] or None.
            sim_data_raw:    Simulated data. Shape: [N, dim_x].
            prior:           Prior distribution object.
            output_dict:     Base output dictionary to populate and return.
        """

        raise NotImplementedError("Subclasses must implement this method")

    def get_parameter_estimates(
        self, training_data_raw: Tensor, task: "BaseTask"
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """Get parameter estimates for training data using the trained optimizer.

        Args:
            training_data_raw: Raw training data
            task: Task object with summary statistics and other utilities

        Returns:
            tuple: (parameter_estimates, optimizer_data, errors)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _create_base_output_dict(self, optimizer_type: str) -> dict:
        """Create the base output dictionary structure.

        Args:
            optimizer_type: String identifier of the optimizer
            (e.g. ``"abc"``, ``"npe"``).
        """
        return {
            "optimizer_type": optimizer_type,
            "errors": None,
            "error_type": None,
            "posterior_obj": None,
        }

    def _handle_error_dict(self, output_dict: dict, error_dict: dict | None) -> bool:
        """Handle existing error dictionary and return early if errors exist.

        Args:
            output_dict: Dictionary to write forwarded error fields into.
            error_dict:  Upstream error dict, or None if no prior errors exist.
        """
        if error_dict is not None:
            output_dict["errors"] = error_dict["errors"]
            output_dict["error_type"] = error_dict["error_type"]
            return True
        return False

    def _setup_simulation_wrapper(self, simulator: Callable) -> Callable:
        """Set up the simulation wrapper with the given simulator.

        Args:
            simulator: Simulator callable generated by the LLM.
        """
        simulation_wrapper = self.task.simulation_wrapper
        return partial(simulation_wrapper, simulator)

    def _sample_from_prior(self) -> tuple[Tensor, Any]:
        """Sample parameters from the prior distribution."""
        prior = self.task.prior_dist
        return prior.sample((self.num_simulations,)), prior

    def _sample_context(self) -> Tensor | None:
        """
        Sample any additional context needed for simulations from the task. The context
        is sampled randomly from a task-specific distribution.

        Returns:
            context: Sampled context tensor.
        """

        context = self.task.sample_context(self.num_simulations)
        return context

    def _run_simulations(
        self,
        simulation_wrapper: Callable,
        prior_samples: Tensor,
        output_dict: dict,
        context_samples: Tensor | None,
    ) -> tuple[Tensor | None, bool]:
        """
        Run simulations and handle any errors that occur.

        Args:
            simulation_wrapper: Function to run simulations. This function takes two
                arguments: params and context.
            prior_samples: Sampled parameters from the prior distribution.
            context_samples: Sampled context for simulations.
            output_dict: Dictionary to store output information.

        Returns:
            sim_data_raw: Raw simulation data tensor.
            success: Boolean indicating if simulations were successful.
        """

        try:
            sim_data_raw = simulation_wrapper(
                params=prior_samples, context=context_samples
            )

            sim_data_raw = torch.as_tensor(sim_data_raw, dtype=torch.float32)
            logger.debug("Model simulation successful")
            return sim_data_raw, True

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            formatted_tb = "\n".join(traceback.format_list(tb))
            logger.warning(f"Error optimizing model: {type(e).__name__}")
            output_dict["errors"] = f"{type(e).__name__}: {formatted_tb}\n{e}"
            output_dict["error_type"] = "forward simulation failed"
            return None, False

    def _plot_loss(
        self, training_loss_list: list[float], validation_loss_list: list[float]
    ) -> None:
        """Plot training and validation loss.

        Args:
            training_loss_list:   Per-epoch training losses.
            validation_loss_list: Per-epoch validation losses.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(training_loss_list, label="Training Loss")
        plt.plot(validation_loss_list, label="Validation Loss")
        plt.legend()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, "sbi_loss.png"))
        plt.close()

    def _fallback_to_abc_estimates(
        self, training_data_raw: Tensor, task: "BaseTask", warning_msg: str
    ) -> tuple[Tensor | None, dict | None, str | None]:
        """Fallback to ABC-style parameter estimation using the ABC optimizer.

        This method is used when posterior sampling times out or fails.
        It instantiates an ABC optimizer and uses it to get parameter estimates
        from the already-computed simulation data.

        Args:
            training_data_raw: Raw training data
            task: Task object with summary statistics and other utilities
            warning_msg: Warning message explaining why the fallback was triggered

        Returns:
            tuple: (parameter_estimates, optimizer_data_with_fallback_flag, error_msg)
        """
        # Local import breaks the circular dependency: abc.py imports OptimizerBase
        # from this module, so a top-level import here would be circular.
        from modelsmc.method.modules.optimizer.abc import OptimizerABCModule

        try:
            # Temporarily disable learnable_summary_statistics validation for fallback
            # Store original value if it exists
            original_learnable = None
            if hasattr(task, "config") and hasattr(
                task.config, "learnable_summary_statistics"
            ):
                original_learnable = task.config.learnable_summary_statistics
                task.config.learnable_summary_statistics = False

            try:
                # Create an ABC optimizer instance with the same config and task
                abc_optimizer = OptimizerABCModule(self.config, task, self.verbose)

                # Set the optimize_output to reuse the simulation data from this
                # optimizer
                abc_optimizer.optimize_output = self.optimize_output

                # Get parameter estimates using ABC's nearest neighbor approach
                parameter_est, optimizer_data, error_msg_abc = (
                    abc_optimizer.get_parameter_estimates(training_data_raw, task)
                )

                # If successful, add fallback flag to optimizer_data
                if parameter_est is not None:
                    if optimizer_data is None:
                        optimizer_data = {}
                    optimizer_data["used_abc_fallback"] = True
                    optimizer_data["warning_msg"] = warning_msg
                    return parameter_est, optimizer_data, None
                else:
                    return None, None, error_msg_abc
            finally:
                # Restore original learnable_summary_statistics setting
                if original_learnable is not None:
                    task.config.learnable_summary_statistics = original_learnable

        except Exception as e:
            error_msg = (
                f"ABC fallback parameter estimation failed: {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            return None, None, error_msg

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        """
        Signal handler that raises TimeoutError when posterior sampling exceeds the
        limit.

        Args:
            signum: Received signal number (unused).
            frame:  Current stack frame at interrupt time (unused).
        """
        logger.debug("Timeout handler called - raising TimeoutError")
        raise TimeoutError("Posterior sampling took too long")
