from acua.providers.provider import AbstractProvider
from acua.providers.util import (
    APIPayload,
    LLMOutput,
    MessageBuilder,
    ToolCall,
    ToolCalls,
)

# Provider imports are optional — degrade gracefully when SDK is not installed.
try:
    from acua.providers.openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore[assignment,misc]

try:
    from acua.providers.anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore[assignment,misc]

try:
    from acua.providers.openrouter_provider import OpenRouterProvider
except ImportError:
    OpenRouterProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "AbstractProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "APIPayload",
    "LLMOutput",
    "MessageBuilder",
    "ToolCall",
    "ToolCalls",
]
