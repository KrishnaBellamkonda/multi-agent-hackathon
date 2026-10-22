from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
from openai.types.responses import Response

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


def call_openai_api_with_retries(client_function, api_params, max_retries=10):
    """
    Makes an OpenAI API call with retries for transient failures,
    rate limiting, and invalid or error-containing responses.
    (This is now a wrapper around generic_call_api_with_retries for OpenAI)

    Args:
        client_function: The OpenAI API client function to call.
        api_params: Parameters to pass to the client function.
        max_retries: Maximum number of retry attempts.

    Returns:
        The OpenAI API response if successful.
    """

    def is_openai_response_valid(response):
        # Check for explicit error field in response object first
        if getattr(response, "error", None):
            logging.warning(
                f"OpenAI API response contains an error attribute: {response.error}"
            )
            return False  # Treat as invalid for retry purposes
        if hasattr(response, "choices") and response.choices:  # Chat Completion API
            return True
        if hasattr(response, "output") and response.output:  # Response API
            return True
        logging.warning(
            "OpenAI API response is missing 'choices' or 'output' is empty."
        )
        return False

    def get_openai_status_code(exception):
        return getattr(exception, "http_status", None)

    return generic_call_api_with_retries(
        client_function=client_function,
        api_params=api_params,
        is_response_valid_fn=is_openai_response_valid,
        rate_limit_exceptions=(openai.RateLimitError,),
        api_error_exceptions=(
            openai.APIError,
        ),  # openai.RateLimitError is caught first
        get_status_code_fn=get_openai_status_code,
        max_retries=max_retries,
        # You can also pass initial_retry_delay_seconds and max_retry_delay_seconds
        # if you want to customize them from their defaults in the generic function.
    )


class OpenAIProvider(AbstractProvider):
    # Models accessed via the OpenAI Responses API.
    # Dev/cheap: gpt-4o-mini  |  Production: gpt-4o, o3, o4-mini
    MODELS = (
        "gpt-4o-mini",   # dev — cheap, fast
        "gpt-4o",        # production baseline
        "o4-mini",       # production reasoning
        "o3",            # production top-tier
    )

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 2**14,
        reasoning_effort: str | None = None,
    ):
        self.action_space_as_tools = True
        self.reasoning_effort = reasoning_effort
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        client_args = {}
        if base_url is not None:
            client_args["base_url"] = base_url
        if api_key is not None:
            client_args["api_key"] = api_key
        self.client = OpenAI(**client_args)
        # self.init_pricing_tracker(pricing_api="openai")

    # --- Message formatting (moved from OpenAIResponseAPIMessageBuilder) ---

    def _format_content_item(self, item: ContentItem, role: str) -> dict:
        """Convert a canonical content item to OpenAI Response API format."""
        if "text" in item:
            content_type = "input_text" if role != "assistant" else "output_text"
            return {"type": content_type, "text": item["text"]}
        elif "image" in item:
            return {"type": "input_image", "image_url": item["image"]}
        else:
            raise ValueError(f"Unsupported content type: {item}")

    def _format_message(self, msg: MessageBuilder) -> List[dict]:
        """Convert a canonical MessageBuilder to OpenAI Response API messages."""
        if msg.role == "tool":
            return self._format_tool_response(msg)

        role = "developer" if msg.role == "system" else msg.role
        content = [self._format_content_item(item, role) for item in msg.content]
        return [{"role": role, "content": content}]

    def _format_tool_response(self, msg: MessageBuilder) -> List[dict]:
        """Convert a tool response MessageBuilder to OpenAI Response API format."""
        if msg.responded_tool_calls is None:
            raise ValueError("No tool calls found in responded_tool_calls")

        output = []
        # Echo the full assistant turn (reasoning + tool calls)
        output.extend(msg.responded_tool_calls.raw_response.output)

        for fn_call in msg.responded_tool_calls:
            call_type = fn_call.raw_call.type
            call_id = fn_call.raw_call.call_id
            call_response = fn_call.tool_response

            match call_type:
                case "function_call":
                    assert "image" not in call_response, (
                        "Image output is not supported in function calls response."
                    )
                    output.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": call_response["text"],
                        }
                    )
                case "computer_call":
                    assert "text" not in call_response, (
                        "Text output is not supported in computer calls response."
                    )
                    output.append(
                        {
                            "type": "computer_call_output",
                            "call_id": call_id,
                            "output": self._format_content_item(call_response, "tool"),
                        }
                    )
        return output

    # --- API call (formatting now happens here) ---

    def _build_api_params(self, payload: APIPayload) -> Dict[str, Any]:
        """Build the Responses API parameter dict from a payload."""
        input = []
        for msg in payload.messages:
            input.extend(self._format_message(msg))

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
        }
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_output_tokens"] = self.max_tokens
        if payload.tools is not None:
            api_params["tools"] = payload.tools
        if payload.tool_choice is not None and payload.force_call_tool is None:
            api_params["tool_choice"] = (
                "required"
                if payload.tool_choice in ("required", "any")
                else payload.tool_choice
            )
        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {
                "type": "function",
                "name": payload.force_call_tool,
            }

        if self.reasoning_effort:
            api_params["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "concise",
            }

        return api_params

    def _call_api(self, payload: APIPayload) -> Response:
        api_params = self._build_api_params(payload)
        return call_openai_api_with_retries(self.client.responses.create, api_params)

    # --- Response parsing (unchanged) ---

    def _parse_response(self, response: Response) -> LLMOutput:
        think_output = self._extract_thinking_content_from_response(response)
        toolcalls = self._extract_tool_calls_from_response(response)

        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(toolcalls)
        else:
            env_action = self._extract_env_actions_from_text_response(response)

        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action if env_action is not None else None,
            tool_calls=toolcalls if toolcalls is not None else None,
        )

    def _extract_tool_calls_from_response(self, response: Response) -> ToolCalls:
        tool_calls = []
        for output in response.output:
            if output.type == "function_call":
                tool_name = output.name
                tool_args = json.loads(output.arguments)
            elif output.type == "computer_call":
                tool_name, tool_args = self.cua_action_to_env_tool_name_and_args(
                    output.action
                )
            else:
                continue
            tool_calls.append(
                ToolCall(name=tool_name, arguments=tool_args, raw_call=output)
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

    def _extract_thinking_content_from_response(self, response: Response) -> str:
        thinking_content = ""
        for output in response.output:
            if output.type == "reasoning":
                if len(output.summary) > 0:
                    thinking_content += output.summary[0].text + "\n"
            elif output.type == "message" and output.content:
                thinking_content += output.content[0].text + "\n"
            elif hasattr(output, "output_text") and output.output_text:
                thinking_content += f"{output.output_text}\n"
        return thinking_content

    def cua_action_to_env_tool_name_and_args(
        self, action: str
    ) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _extract_env_actions_from_text_response(self, response) -> str | None:
        pass
