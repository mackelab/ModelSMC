import json
import logging
import os
from typing import Dict, List, Literal, Optional, Union

import dspy
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("ModelSMC")


#########################################################
# Metrics helper
#########################################################
class PerformanceMetrics(BaseModel):
    errors: Optional[str] = None
    error_type: Optional[str] = None
    warnings: Optional[str] = None
    eval_metrics: Optional[Dict[str, Union[float, List[float]]]] = None
    posterior_inter_quantiles: Optional[Dict[str, Dict[str, List[float]]]] = None
    inter_quantile_range: Optional[List[float]] = None

    def __str__(self) -> str:
        parts = []

        # 1. Errors and warnings
        if self.errors:
            parts.append(f"Error: {self.errors}")
            if self.error_type:
                parts.append(f"Type: {self.error_type}")
        if self.warnings:
            parts.append(f"Warning: {self.warnings}")

        # 2. Evaluation metrics
        if self.eval_metrics:
            metrics = []
            for name, val in self.eval_metrics.items():
                if isinstance(val, list):
                    metrics.append(f"{name}: {[f'{v:.3g}' for v in val]}")
                else:
                    metrics.append(f"{name}: {val:.3g}")
            parts.append("Metrics: " + ", ".join(metrics))

        # 3. Optimization metrics
        if self.posterior_inter_quantiles is not None:
            opt_metrics = []
            if self.inter_quantile_range is not None:
                range_def = (
                    "range def: "
                    f"{', '.join(f'{val:.3g}' for val in self.inter_quantile_range)}"
                )

                opt_metrics.append(range_def)

            quantile_items = []
            for dim, vals in self.posterior_inter_quantiles.items():
                means = ", ".join(f"{val:.3g}" for val in vals["mean"])
                stds = ", ".join(f"{val:.3g}" for val in vals["std"])
                quantile_items.append(f"{dim}: means [{means}], stds [{stds}]")

            if quantile_items:
                opt_metrics.append(f"quantiles: {', '.join(quantile_items)}")
            parts.append("Optimization: " + ", ".join(opt_metrics))

        return "; ".join(parts) if parts else "No metrics"

    def render(self) -> str:
        return self.__str__()


#########################################################
# Feedback part
#########################################################
class Issue(BaseModel):
    description: str = Field(..., description="Issue description")
    severity: Literal["critical", "major", "minor", "suggestion"] = Field(
        ..., description="Severity level"
    )
    location: Optional[str] = Field(None, description="Code location")
    suggestion: str = Field(..., description="Fix suggestion")


class ProvideFeedbackOutput(BaseModel):
    main_diagnosis: str = Field(..., description="Main issue and location")
    issues: List[Issue] = Field(..., max_items=2, description="Issues with fixes")
    # overall_assessment: str = Field(..., description="Quality assessment")

    @field_validator("issues")
    @classmethod
    def limit_issues(cls, v):
        return v[:2] if len(v) > 2 else v


class DiagnoseAndImprove(dspy.Signature):
    """Analyze evaluation metrics to diagnose failure modes, then provide code
    improvement feedback."""

    system_description: str = dspy.InputField(desc="System description")
    simulator_code: str = dspy.InputField(desc="Simulator code")
    performance_metrics: str = dspy.InputField(desc="Performance metrics")
    feedback: ProvideFeedbackOutput = dspy.OutputField(desc="Improvement feedback")


#########################################################
# Full module
#########################################################
class FeedbackModule(dspy.Module):
    """Feedback module with three operating modes:

    - ``llm_only``        (default): LLM-generated diagnosis JSON string.
                          Mirrors the original behaviour.
    - ``metrics_only``:   Return the plain performance metrics string with no
                          LLM call.  Saves computation and removes the LLM
                          interpretation layer.
    - ``llm_and_metrics``: LLM diagnosis JSON followed by the plain metrics
                          string, so the code-generation LLM sees both the
                          interpreted diagnosis and the raw numbers.
    """

    VALID_MODES = ("llm_only", "metrics_only", "llm_and_metrics")

    def __init__(
        self, task, verbose: bool = True, feedback_mode: str = "llm_only"
    ) -> None:
        """
        Args:
            task:          Task object; used to extract the system description passed
                           to the LLM as context.
            verbose:       If True, log a message each time feedback is retrieved.
            feedback_mode: Controls what is returned by ``retrieve``.  Must be one
                           of ``"llm_only"``, ``"metrics_only"``, or
                           ``"llm_and_metrics"``.
        """
        super().__init__()
        if feedback_mode not in self.VALID_MODES:
            raise ValueError(
                "feedback_mode must be one of "
                f"{self.VALID_MODES}, got '{feedback_mode}'"
            )
        self.save_dir = None
        self.verbose = verbose
        self.feedback_mode = feedback_mode
        self.system_description = task.get_system_description()

        self.feedback_module = dspy.Predict(DiagnoseAndImprove)

    def retrieve(self, eval_metrics: dict, simulator_code: str, save_dir: str) -> str:
        """Generate feedback for a particle.

        Args:
            eval_metrics:   Evaluation metrics dict from the weighting step.
            simulator_code: Simulator source code string.
            save_dir:       Directory in which to persist the feedback artefact.

        Returns:
            Feedback string that will be stored on the particle and later used
            as few-shot context for the code-generation LLM.
        """
        self.save_dir = save_dir
        performance_metrics_str = str(PerformanceMetrics(**eval_metrics))

        if self.feedback_mode == "metrics_only":
            logger.info("Retrieving feedback (metrics only)") if self.verbose else None
            self._save_metrics_feedback(performance_metrics_str)
            return performance_metrics_str

        # Both llm_only and llm_and_metrics require an LLM call.
        logger.info("Retrieving feedback") if self.verbose else None
        feedback = self.feedback_module(
            system_description=self.system_description,
            simulator_code=simulator_code,
            performance_metrics=performance_metrics_str,
        ).feedback

        self._save_feedback(feedback)
        llm_feedback_str = feedback.model_dump_json(indent=2)

        if self.feedback_mode == "llm_and_metrics":
            return (
                llm_feedback_str + "\n\nPerformance metrics: " + performance_metrics_str
            )

        # llm_only
        return llm_feedback_str

    def _save_feedback(self, feedback: ProvideFeedbackOutput) -> None:
        """Save LLM feedback as a JSON file.

        Args:
            feedback: Structured feedback object to serialise.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        feedback_json_path = os.path.join(self.save_dir, "feedback.json")
        with open(feedback_json_path, "w") as f:
            json.dump(feedback.model_dump(), f, indent=2)

    def _save_metrics_feedback(self, performance_metrics_str: str) -> None:
        """Save plain metrics feedback as a text file.

        Args:
            performance_metrics_str: Formatted metrics string to write.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        feedback_path = os.path.join(self.save_dir, "feedback.txt")
        with open(feedback_path, "w") as f:
            f.write(performance_metrics_str)
