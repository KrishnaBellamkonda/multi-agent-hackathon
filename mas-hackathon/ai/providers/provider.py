from abc import ABC, abstractmethod
from typing import Any, Optional

from acua.providers.util import APIPayload, LLMOutput


class AbstractProvider(ABC):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__()

    def __call__(self, payload: APIPayload) -> LLMOutput:
        """Make a call to the model and return the parsed response."""
        response = self._call_api(payload)
        return self._parse_response(response)

    @abstractmethod
    def _call_api(self, payload: APIPayload) -> Any:
        """Make a call to the model API and return the raw response."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMOutput:
        """Parse the raw response from the model API and return a structured response."""
        pass
