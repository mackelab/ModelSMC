import inspect
import logging
from typing import Optional

import dspy

from modelsmc.utils.utils import strip_markdown_fences

logger = logging.getLogger("ModelSMC")


instruction_template = """
You are an expert in mechanistic modeling with differential equations and simulator optimization.

Core Requirements:
- Create a scientific simulator that captures system dynamics accurately
- Write fully functional, executable code with no placeholders
- Use tabs for indentation and mathematically correct formulations
- Add variable shapes in every line of code
- Achieve optimal validation metrics

Implementation Guidelines:
- Start with mechanistic components, add stochastic elements where needed
- Follow physical/biological constraints and include prior knowledge
- Use any operations (log, exp, power, etc.) and non-differentiable operations
- Ensure computational efficiency for many simulations
- Keep the code as simple as possible

Iterative Optimization Rules:
- You are part of an iterative process - make only 1-2 targeted improvements per iteration
- Never copy code verbatim - preserve working components while making precise, localized changes
- Make creative but sensible and simple modifications that are structurally different from prior examples
- Focus on changes that build incrementally toward better validation metrics
- Ensure executable code without syntax or shape errors

System description:
{system_description}
"""  # noqa: E501


class CodingSimulatorSignature(dspy.Signature):
    signature_description: str = dspy.InputField(desc="Simulator I/O description")
    task_description: str = dspy.InputField(desc="Task considerations")
    base_simulator: str = dspy.InputField(desc="Base code structure")

    scm_definition: str = dspy.OutputField(
        desc=(
            "Compact SCM summary as a bullet list (max 5 lines): key variables, "
            "their causal dependencies, and core assumptions. No equations or "
            "verbose descriptions."
        )
    )
    simulator_code: str = dspy.OutputField(
        desc=(
            "Improved mechanistic simulator with substantial structural "
            "differences from prior examples."
        )
    )
    feedback: Optional[str] = dspy.OutputField(
        desc="Optional feedback. Only generate if requested.", default=None
    )


# Capture the DSPy-generated default instructions before any runtime mutation.
# This preserves the field descriptions so they can be combined with the
# task-specific system prompt rather than overwritten.
_CODING_SIMULATOR_DEFAULT_INSTRUCTIONS = CodingSimulatorSignature.instructions


class CodingSimulatorModule(dspy.Module):
    def __init__(
        self,
        task,
        num_completions: int = 1,
        verbose: bool = True,
        use_chain_of_thought: bool = False,
    ):
        """
        Args:
            task:                 Task object; used to extract the system description,
                                  signature description, base simulator, and task
                                  description that are injected into the LLM prompt.
            num_completions:      Number of candidate completions to request per LLM
                                  call.
            verbose:              If True, log a message each time code generation runs.
            use_chain_of_thought: If True, use ``dspy.ChainOfThought`` instead of
                                  ``dspy.Predict`` for the code-generation predictor.
        """
        super().__init__()
        self.verbose = verbose
        self.system_description = task.get_system_description()
        self.signature_description = task.get_signature_description()
        self.base_simulator = task.get_base_simulator()
        self.task_description = task.get_task_description()

        # Dynamically set the __doc__ string of the signature class by combining the
        # default docstring and the custom instructions and system description.
        self._apply_instructions()

        predictor_cls = dspy.ChainOfThought if use_chain_of_thought else dspy.Predict
        self.coding_simulator = predictor_cls(
            CodingSimulatorSignature, n=num_completions
        )

    def _apply_instructions(self):
        CodingSimulatorSignature.__doc__ = (
            _CODING_SIMULATOR_DEFAULT_INSTRUCTIONS
            + "\n"
            + instruction_template.format(system_description=self.system_description)
        )

        # Check that the expected instructions are contained in the actual instructions
        expected = inspect.cleandoc(
            instruction_template.format(system_description=self.system_description)
        )
        if expected not in CodingSimulatorSignature.instructions:
            raise ValueError(
                "CodingSimulatorSignature.instructions does not contain "
                "the instruction template. "
                f"Got: {CodingSimulatorSignature.instructions!r}"
            )

    def __setstate__(self, state):
        super().__setstate__(state)
        # Re-apply the custom instructions lost during pickling.
        # Pickle stores classes by (module, qualname) reference, so Ray workers
        # import CodingSimulatorSignature fresh with the default __doc__.
        self._apply_instructions()

    def forward(self) -> dspy.Prediction:
        logger.info("Generating simulator code") if self.verbose else None

        result = self.coding_simulator(
            signature_description=self.signature_description,
            task_description=self.task_description,
            base_simulator=self.base_simulator,
        )

        # save code
        code = result.simulator_code
        code = strip_markdown_fences(code)
        code = code.replace("\\t", "\t")

        # save scm
        scm_definition = result.scm_definition

        return dspy.Prediction(code=code, scm_definition=scm_definition)
