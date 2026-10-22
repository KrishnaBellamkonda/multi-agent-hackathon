"""Anthropic provider — uses the Messages API (client.messages.create).

Supports Claude Sonnet 4.6, Opus 4.6, Haiku 4.5, etc.
Extended thinking is controlled via the thinking parameter with a token budget.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import Anthropic

from acua.providers.provider import AbstractProvider
from acua.providers.util import (
    APIPayload,
    ContentItem,
    LLMOutput,
    MessageBuilder,
    PythonAction,
    ToolCall,
    ToolCalls,
    generic_call_api_with_retries,
)

logger = logging.getLogger("acua.providers.anthropic")


def call_anthropic_api_with_retries(client_function, api_params, max_retries=10):
    """Anthropic API call with retries for rate limits and transient failures."""

    def is_response_valid(response):
        if hasattr(response, "content") and response.content is not None:
            return True
        logging.warning("Anthropic response has no content.")
        return False

    def get_status_code(exception):
        return getattr(exception, "status_code", None)

    return generic_call_api_with_retries(
        client_function=client_function,
        api_params=api_params,
        is_response_valid_fn=is_response_valid,
        rate_limit_exceptions=(anthropic.RateLimitError,),
        api_error_exceptions=(anthropic.APIError,),
        get_status_code_fn=get_status_code,
        max_retries=max_retries,
    )


class AnthropicProvider(AbstractProvider):
    """Anthropic Messages API provider for Claude models."""

    # Models accessed via the Anthropic Messages API. See README.md "Models".
    MODELS = ("claude-sonnet-4-6",)

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        thinking_budget: int | None = None,
    ):
        self.action_space_as_tools = True
        self.thinking_budget = thinking_budget
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        client_args = {}
        if api_key is not None:
            client_args["api_key"] = api_key
        self.client = Anthropic(**client_args)

    # --- Message formatting ---

    def _format_content_item(self, item: ContentItem) -> dict:
        """Convert a canonical ContentItem to Anthropic format."""
        if "text" in item:
            return {"type": "text", "text": item["text"]}
        elif "image" in item:
            return self._convert_image(item["image"])
        else:
            raise ValueError(f"Unsupported content type: {item}")

    @staticmethod
    def _convert_image(image_url: str) -> dict:
        """Convert a data:image/... URL to Anthropic's base64 image format."""
        match = re.match(r"data:(image/\w+);base64,(.+)", image_url)
        if match:
            media_type = match.group(1)
            data = match.group(2)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        # If it's a URL, pass through
        return {
            "type": "image",
            "source": {"type": "url", "url": image_url},
        }

    def _format_messages(
        self, messages: List[MessageBuilder]
    ) -> tuple[str, List[dict]]:
        """Convert MessageBuilder list to Anthropic format.

        Returns (system_text, messages_list).
        System messages are extracted to the system parameter.
        Tool responses are formatted as user messages with tool_result blocks.
        """
        system_text = ""
        formatted = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic: system is a parameter, not a message
                parts = []
                for item in msg.content:
                    if "text" in item:
                        parts.append(item["text"])
                system_text = "\n\n".join(parts)
                continue

            if msg.role == "tool":
                # Tool responses → user message with tool_result blocks
                formatted.extend(self._format_tool_response(msg))
                continue

            content = [self._format_content_item(item) for item in msg.content]
            formatted.append({"role": msg.role, "content": content})

        return system_text, formatted

    def _format_tool_response(self, msg: MessageBuilder) -> List[dict]:
        """Convert tool response to Anthropic format.

        Unlike OpenAI, we don't echo the assistant turn — just send
        a user message with tool_result blocks.
        """
        if msg.responded_tool_calls is None:
            raise ValueError("No tool calls in responded_tool_calls")

        # First, echo the assistant turn with the tool_use blocks
        assistant_content = []
        for fn_call in msg.responded_tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": fn_call.raw_call.id,
                    "name": fn_call.raw_call.name,
                    "input": fn_call.raw_call.input,
                }
            )

        # Then the user turn with tool_result blocks
        tool_results = []
        for fn_call in msg.responded_tool_calls:
            result_content = fn_call.tool_response
            if "text" in result_content:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": fn_call.raw_call.id,
                        "content": result_content["text"],
                    }
                )
            elif "image" in result_content:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": fn_call.raw_call.id,
                        "content": [self._convert_image(result_content["image"])],
                    }
                )

        return [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

    def _convert_tools(self, tools: List[dict]) -> List[dict]:
        """Convert tool definitions to Anthropic format.

        Anthropic uses 'input_schema' instead of 'parameters'.
        Tools from to_tool_dict() use Responses API format which needs conversion.
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                converted.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            else:
                converted.append(tool)
        return converted

    # --- API call ---

    def _call_api(self, payload: APIPayload) -> Any:
        system_text, messages = self._format_messages(payload.messages)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens or 4096,
        }

        if system_text:
            api_params["system"] = system_text
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if payload.tools is not None:
            api_params["tools"] = self._convert_tools(payload.tools)
        if payload.tool_choice is not None and payload.force_call_tool is None:
            if payload.tool_choice == "required":
                api_params["tool_choice"] = {"type": "any"}
            elif payload.tool_choice != "none":
                api_params["tool_choice"] = {"type": payload.tool_choice}
        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {
                "type": "tool",
                "name": payload.force_call_tool,
            }
        if self.thinking_budget:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        return call_anthropic_api_with_retries(self.client.messages.create, api_params)

    # --- Response parsing ---

    def _parse_response(self, response: Any) -> LLMOutput:
        think_output = self._extract_thinking(response)
        toolcalls = self._extract_tool_calls(response)

        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(toolcalls)
        else:
            env_action = self._extract_text_content(response)

        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action,
            tool_calls=toolcalls if toolcalls else None,
        )

    def _extract_thinking(self, response: Any) -> str:
        """Extract thinking and text content from response."""
        parts = []
        for block in response.content:
            if block.type == "thinking":
                parts.append(block.thinking)
            elif block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _extract_tool_calls(self, response: Any) -> ToolCalls:
        """Extract tool calls from response content blocks."""
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,  # already a dict
                        raw_call=block,
                    )
                )
        return ToolCalls(tool_calls=tool_calls, raw_response=response)

    def _extract_env_actions_from_toolcalls(self, toolcalls: ToolCalls) -> str | None:
        if not toolcalls:
            return None
        actions = [
            PythonAction.convert_toolcall_to_python_action_format(call)
            for call in toolcalls
        ]
        return (
            PythonAction.convert_multiactions_to_python_action_format(actions)
            if len(actions) > 1
            else actions[0]
        )

    def _extract_text_content(self, response: Any) -> str | None:
        """Extract just text content (no thinking) for text-fallback mode."""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts) if parts else None
