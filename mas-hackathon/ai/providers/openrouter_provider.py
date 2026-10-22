"""OpenRouter provider — uses OpenRouter's Responses API (beta).

OpenRouter supports the same Responses API format as OpenAI, so this
subclasses OpenAIProvider and just sets the base_url and adds
OpenRouter-specific extras (provider routing, reasoning flag).

For models that don't support tool calling (Kimi, Qwen), set
code_gen_mode=True. The provider will skip sending tools, inject
a code-gen system prompt, parse ## Code blocks from responses,
and return the pyautogui code as LLMOutput.action.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from acua.providers.openai_provider import OpenAIProvider, call_openai_api_with_retries
from acua.providers.util import APIPayload, LLMOutput, MessageBuilder, ToolCalls

logger = logging.getLogger("acua.providers.openrouter")

# Fixed provider routing for specific models
_MODEL_PROVIDERS = {
    "moonshotai/kimi-k2.5": {"order": ["Chutes"], "allow_fallbacks": False},
    "minimax/minimax-01": {"order": ["MiniMax"], "allow_fallbacks": False},
    "xiaomi/mimo-v2-omni": {"order": ["Xiaomi"], "allow_fallbacks": False},
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider using the Responses API (beta).

    Inherits all message formatting and response parsing from OpenAIProvider.
    Adds OpenRouter-specific provider routing via extra_body.

    When code_gen_mode=True, the model is expected to output pyautogui
    code blocks instead of using tool calling. The provider handles
    parsing and coordinate scaling transparently.
    """

    # Models accessed via OpenRouter. See README.md "Models".
    MODELS = (
        "xiaomi/mimo-v2-omni",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.5-27b",
        "qwen/qwen3.5-flash-02-23",
        "moonshotai/kimi-k2.5",
        "minimax/minimax-01",
        "grok-4",
        "grok-4-1-fast-reasoning",
        "grok-4.20-beta-0309-reasoning",
    )

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        reasoning_effort: str | None = None,
        code_gen_mode: bool = False,
        screen_size: tuple[int, int] = (1920, 1080),
        password: str = "",
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        self.code_gen_mode = code_gen_mode
        self.screen_size = screen_size
        self.password = password
        super().__init__(
            model_name=model_name,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _format_message(self, msg: MessageBuilder) -> List[dict]:
        """Convert a MessageBuilder to OpenRouter Responses API format.

        OpenRouter's Responses API is stricter than OpenAI's — it rejects
        output_text content arrays in assistant messages. Use plain strings
        for assistant text content instead.
        """
        if msg.role == "tool":
            return self._format_tool_response(msg)

        role = "developer" if msg.role == "system" else msg.role
        content = [self._format_content_item(item, role) for item in msg.content]

        # OpenRouter rejects [{"type": "output_text", ...}] for assistant —
        # flatten to a plain string
        if role == "assistant" and all(c.get("type") == "output_text" for c in content):
            text = "\n".join(c["text"] for c in content)
            return [{"role": "assistant", "content": text}]

        return [{"role": role, "content": content}]

    def _build_api_params(self, payload: APIPayload) -> Dict[str, Any]:
        """Build API params via parent, then inject OpenRouter provider routing."""
        if self.code_gen_mode:
            return self._build_code_gen_params(payload)

        api_params = super()._build_api_params(payload)

        provider_config = _MODEL_PROVIDERS.get(self.model_name)
        if provider_config:
            api_params["extra_body"] = {"provider": provider_config}

        return api_params

    def _build_code_gen_params(self, payload: APIPayload) -> Dict[str, Any]:
        """Build API params for code-gen mode (no tools, inject code-gen prompt)."""
        from acua.providers.code_gen import KIMI_SYSTEM_PROMPT

        # Replace the system message with the code-gen prompt
        input_items = []
        for msg in payload.messages:
            if msg.role == "system":
                # Inject code-gen system prompt, preserving the task instruction
                system_text = KIMI_SYSTEM_PROMPT.format(password=self.password)
                # Append the original system text (which has the task instruction)
                for item in msg.content:
                    if "text" in item:
                        system_text += "\n\n" + item["text"]
                input_items.append({"role": "developer", "content": system_text})
            else:
                input_items.extend(self._format_message(msg))

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
        }

        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_output_tokens"] = self.max_tokens

        # No tools, no tool_choice — code-gen mode

        provider_config = _MODEL_PROVIDERS.get(self.model_name)
        if provider_config:
            api_params["extra_body"] = {"provider": provider_config}

        return api_params

    def _call_api(self, payload: APIPayload) -> Any:
        api_params = self._build_api_params(payload)
        return call_openai_api_with_retries(self.client.responses.create, api_params)

    def _parse_response(self, response: Any) -> LLMOutput:
        if not self.code_gen_mode:
            return super()._parse_response(response)

        return self._parse_code_gen_response(response)

    def _parse_code_gen_response(self, response: Any) -> LLMOutput:
        """Parse a code-gen response: extract ## Code block, scale coords."""
        from acua.providers.code_gen import (
            code_to_action,
            parse_code_response,
            scale_pyautogui_coordinates,
        )

        # Extract text and reasoning from response
        text_parts = []
        reasoning = ""
        for item in response.output:
            if item.type == "reasoning":
                if hasattr(item, "summary") and item.summary:
                    reasoning = item.summary[0].text
                elif hasattr(item, "content") and item.content:
                    reasoning = item.content[0].text
            elif item.type == "message" and item.content:
                for c in item.content:
                    if hasattr(c, "text"):
                        text_parts.append(c.text)

        text = "\n".join(text_parts)

        # Parse the code response
        parsed = parse_code_response(text, reasoning=reasoning)

        # Convert to action
        action = code_to_action(parsed)

        # Scale coordinates if it's pyautogui code
        if action not in ("WAIT", "DONE", "FAIL"):
            action = scale_pyautogui_coordinates(
                action, self.screen_size[0], self.screen_size[1]
            )

        think = parsed.thought or reasoning

        return LLMOutput(
            raw_response=response,
            think=think,
            action=action,
            tool_calls=ToolCalls(tool_calls=[], raw_response=response),
        )
