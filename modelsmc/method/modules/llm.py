import dspy


class DummyLLM(dspy.LM):
    """A dummy LM that does nothing, just to satisfy the interface requirements."""

    def __init__(self, response: str = "This is a dummy response.") -> None:
        """
        Args:
            response: Fixed string returned for every call.
        """
        super().__init__(model="Dummy-LLM")
        self.response = response

    def chat_completion(self, *args, **kwargs) -> dict:
        """Return a fixed chat-completion response without calling any API."""
        return {
            "choices": [{"message": {"content": self.response}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
