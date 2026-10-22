from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import numpy as np
from PIL import Image

"""This module contains utility classes for building input messages and interacting with LLM APIs. 
It includes:
    1. Message Builder for building input messages
    2. Base Reponse class for different LLM APIs (OpenAI, Anthropic, etc.)
    3. Factory classes (inherits from BaseModelArgs) for creating instances of LLM Response models.
"""

logger = logging.getLogger(__name__)

ContentItem = Dict[
    str, Any
]  # This is just a singleton Dict i.e. tuple. Singleton nature helps with turning into JSON.


def image_to_png_base64_url(image: np.ndarray | Image.Image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, "PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{image_base64}"


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM.
    Attributes:
    name: Name of the tool called.
    arguments: Arguments passed to the tool.
    raw_call: The raw call object from the LLM API.
    tool_response: Output of the tool call goes here. It can be only one content item.
    """

    name: str | None = field(default=None)
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_call: Any = field(default=None)
    tool_response: ContentItem | None = None

    @property
    def is_response_set(self) -> bool:
        """Check if the tool response is set."""
        return self.tool_response is not None

    def response_text(self, text: str) -> "ToolCall":
        self.tool_response = {"text": text}
        return self

    def response_image(self, image: str) -> "ToolCall":
        self.tool_response = {"image": image}
        return self

    def __repr__(self):
        return f"ToolCall(name={self.name}, arguments={self.arguments})"


@dataclass
class ToolCalls:
    """A collection of tool calls made by the LLM.

    Attributes:
    tool_calls: List of ToolCall objects.
    raw_response: Represents raw tool calls object returned by a LLM API, may contain one or more tool calls.

    When the LLM makes two tool calls in one turn, the API returns something like this (OpenAI Response API):
      {
        "output": [
          {"type": "reasoning", "summary": [{"text": "I need to click and then scroll"}]},
          {"type": "function_call", "call_id": "fc_1", "name": "click", "arguments": "{\"bid\": \"a1\"}"},
          {"type": "function_call", "call_id": "fc_2", "name": "scroll", "arguments": "{\"x\": 0, \"y\": 500}"}
        ]
      }

      This gets split into:

      ToolCalls:
        raw_response = <entire response object above>
        tool_calls = [
          ToolCall(name="click",  arguments={...}, raw_call=<the fc_1 object>),
          ToolCall(name="scroll", arguments={...}, raw_call=<the fc_2 object>),
        ]

      Now when handle_tool_call() builds the next turn's messages, it needs to send back:

      1. The full assistant turn — including the reasoning AND all tool calls, because the API expects you to echo the complete assistant message before sending tool responses
      2. Individual tool responses — each paired to its call_id

      output = []
      # Step 1: echo the ENTIRE assistant turn (reasoning + both tool calls)
      output.extend(self.responded_tool_calls.raw_response.output)  # needs raw_response

      # Step 2: append individual responses paired by ID
      for fn_call in self.responded_tool_calls:
          output.append({
              "type": "function_call_output",
              "call_id": fn_call.raw_call.call_id,  # needs individual raw_call
              "output": fn_call.tool_response["text"],
          })

      The individual raw_call objects alone aren't enough — you'd lose the reasoning block and any non-tool-call content in the assistant turn. That's why raw_response stores the whole response.
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Any = field(default=None)

    def add_tool_call(self, tool_call: ToolCall) -> "ToolCalls":
        self.tool_calls.append(tool_call)
        return self

    @property
    def all_responses_set(self) -> bool:
        """Check if all tool calls have responses set."""
        return all(call.is_response_set for call in self.tool_calls)

    def __len__(self) -> int:
        """Return the number of tool calls."""
        return len(self.tool_calls)

    def __iter__(self):
        """Make ToolCalls iterable."""
        return iter(self.tool_calls)

    def __bool__(self):
        """Check if there are any tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class APIPayload:
    messages: "List[MessageBuilder] | None" = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "any", "required"] | None = None
    force_call_tool: str | None = (
        None  # Name of the tool to call # If set, will force the LLM to call this tool.
    )
    use_cache_breakpoints: bool = False  # If True, will apply cache breakpoints to the messages. # applicable for Anthropic
    cache_tool_definition: bool = (
        False  # If True, will cache the tool definition in the last message.
    )
    cache_complete_prompt: bool = (
        False  # If True, will cache the complete prompt in the last message.
    )
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    def __post_init__(self):
        if self.tool_choice and self.force_call_tool:
            raise ValueError("tool_choice and force_call_tool are mutually exclusive")
        if self.reasoning_effort is not None:
            logger.info(
                "In agentlab reasoning_effort is used by LiteLLM API only. We will eventually shift to LiteLLM API for all LLMs."
            )


@dataclass
class LLMOutput:
    """Serializable object for the output of a response LLM."""

    raw_response: Any = field(default=None)
    think: str = field(default="")
    action: str | None = field(default=None)  # Default action if no tool call is made
    tool_calls: ToolCalls | None = field(
        default=None
    )  # This will hold the tool call response if any


class MessageBuilder:
    def __init__(self, role: str):
        self.role = role
        self.content: List[ContentItem] = []
        self.responded_tool_calls: ToolCalls | None = None

    @classmethod
    def system(cls) -> "MessageBuilder":
        return cls("system")

    @classmethod
    def user(cls) -> "MessageBuilder":
        return cls("user")

    @classmethod
    def assistant(cls) -> "MessageBuilder":
        return cls("assistant")

    def add_text(self, text: str) -> "MessageBuilder":
        self.content.append({"text": text})
        return self

    def add_image(self, image: str) -> "MessageBuilder":
        self.content.append({"image": image})
        return self

    def to_markdown(self) -> str:
        parts = []
        for item in self.content:
            if "text" in item:
                parts.append(f"\n```\n{item['text']}\n```\n")
            elif "image" in item:
                parts.append(f"![Image]({item['image']})")

        # Tool call markdown repr
        if self.responded_tool_calls is not None:
            for i, tool_call in enumerate(self.responded_tool_calls.tool_calls, 1):
                parts.append(
                    f"\n**Tool Call {i}**: {tool_call_to_python_code(tool_call.name, tool_call.arguments)}"
                )
                response = tool_call.tool_response
                if response is not None:
                    parts.append(f"\n**Tool Response {i}:**")
                    content = (
                        f"```\n{response['text']}\n```"
                        if "text" in response
                        else f"![Tool Response Image]({response['image']})"
                    )
                    parts.append(content)

        markdown = f"### {self.role.capitalize()}\n"
        markdown += "\n".join(parts)

        return markdown

    def add_image_url(self, image_url: str) -> "MessageBuilder":
        """Add an image URL to the message content."""
        self.content.append({"image": image_to_png_base64_url(image_url)})
        return self

    def mark_all_previous_msg_for_caching(self):
        """Insert a cache breakpoint in the message content."""
        # This is a placeholder for future implementation.
        raise NotImplementedError

    @classmethod
    def add_responded_tool_calls(
        cls, responded_tool_calls: ToolCalls
    ) -> "MessageBuilder":
        """Add tool calls to the message content."""
        assert responded_tool_calls.all_responses_set, (
            "All tool calls must have a response."
        )
        msg = cls("tool")
        msg.responded_tool_calls = responded_tool_calls
        return msg


def tool_call_to_python_code(func_name, kwargs):
    """Format a function name and kwargs dict into a Python function call string."""
    if kwargs is None:
        kwargs = {}

    if not kwargs:
        return f"{func_name}()"

    args_str = ", ".join(f"{key}={repr(value)}" for key, value in kwargs.items())
    return f"{func_name}({args_str})"


class PythonAction:
    """
    Collection of utility function to convert tool calls to Python action format.
    """

    @staticmethod
    def convert_toolcall_to_python_action_format(toolcall: ToolCall) -> str:
        """Convert a tool call to an Python environment action string.

        Args:
            toolcall: ToolCall object containing the name and arguments of the tool call.

        Returns:
            A string representing the action in Python format i.e. python function call string.
        """

        tool_name, tool_args = toolcall.name, toolcall.arguments
        return tool_call_to_python_code(tool_name, tool_args)

    @staticmethod
    def convert_multiactions_to_python_action_format(actions: list[str]) -> str | None:
        """Convert multiple actions list to a format that env supports.

        Args:
            actions: List of action strings to be joined.

        Returns:
            Joined actions separated by newlines, or None if empty.
        """
        return "\n".join(actions) if actions else None


def generic_call_api_with_retries(
    client_function,
    api_params,
    is_response_valid_fn,
    rate_limit_exceptions,
    api_error_exceptions,
    get_status_code_fn=None,
    max_retries=10,
    initial_retry_delay_seconds=20,
    max_retry_delay_seconds=60 * 5,
):
    """
    Makes an API call with retries for transient failures, rate limiting,
    and responses deemed invalid by a custom validation function.
    (Refactored for improved readability with helper functions)

    Args:
        client_function: The API client function to call.
        api_params: Parameters to pass to the client function.
        is_response_valid_fn: Function to validate if the response is valid.
        rate_limit_exceptions: Tuple of exception types for rate limiting.
        api_error_exceptions: Tuple of exception types for API errors.
        get_status_code_fn: Optional function to extract status code from exceptions.
        max_retries: Maximum number of retry attempts.
        initial_retry_delay_seconds: Initial delay between retries in seconds.
        max_retry_delay_seconds: Maximum delay between retries in seconds.

    Returns:
        The API response if successful.

    Raises:
        Exception: For unexpected errors that are immediately re-raised.
        RuntimeError: If API call fails after maximum retries.
    """

    def _calculate_delay(
        current_attempt, initial_delay, max_delay, is_first_attempt_for_type=False
    ):
        """Calculates exponential backoff delay."""
        # For invalid response content (not an exception), the first "attempt" at retrying this specific issue
        # might use a slightly different delay calculation if desired (e.g. attempt-1 for the exponent).
        # For exceptions, the attempt number directly applies.
        # Here, we use 'current_attempt' for exception-driven retries,
        # and 'current_attempt -1' for the first retry due to invalid content (is_first_attempt_for_type).
        if is_first_attempt_for_type:  # First retry due to invalid content
            # The first retry after an invalid response (attempt 1 for this *type* of failure)
            effective_attempt = current_attempt - 1  # Use 0 for the first exponent
        else:  # Retries due to exceptions or subsequent invalid content retries
            effective_attempt = current_attempt  # Use current_attempt for exponent

        # Ensure effective_attempt for exponent is at least 0
        exponent_attempt = max(
            0,
            effective_attempt if not is_first_attempt_for_type else current_attempt - 1,
        )

        return min(initial_delay * (2**exponent_attempt), max_delay)

    def _handle_invalid_response_content(attempt):
        logging.warning(
            f"[Attempt {attempt}/{max_retries}] API response deemed invalid by validation function. Retrying after delay..."
        )
        if attempt < max_retries:
            # For the first retry due to invalid content, use attempt-1 for exponent
            delay = _calculate_delay(
                attempt,
                initial_retry_delay_seconds,
                max_retry_delay_seconds,
                is_first_attempt_for_type=True,
            )
            logging.debug(
                f"Sleeping for {delay:.2f} seconds due to invalid response content."
            )
            time.sleep(delay)
            return True  # Indicate retry
        return False  # Max retries reached for this path

    def _handle_rate_limit_error(e, attempt):
        logging.warning(
            f"[Attempt {attempt}/{max_retries}] Rate limit error: {e}. Retrying after delay..."
        )
        if attempt < max_retries:
            delay = _calculate_delay(
                attempt, initial_retry_delay_seconds, max_retry_delay_seconds
            )
            logging.debug(f"Sleeping for {delay:.2f} seconds due to rate limit.")
            time.sleep(delay)
            return True  # Indicate retry
        return False  # Max retries reached for this path

    def _handle_api_error(e, attempt):
        logging.error(f"[Attempt {attempt}/{max_retries}] APIError: {e}")
        status_code = None
        if get_status_code_fn:
            try:
                status_code = get_status_code_fn(e)
            except Exception as ex_status_fn:
                logging.warning(
                    f"Could not get status code from exception {type(e)} using get_status_code_fn: {ex_status_fn}"
                )

        if status_code == 429 or (status_code and status_code >= 500):
            log_msg = (
                "Rate limit (429)"
                if status_code == 429
                else f"Server error ({status_code})"
            )
            logging.warning(
                f"{log_msg} indicated by status code. Retrying after delay..."
            )
            if attempt < max_retries:
                delay = _calculate_delay(
                    attempt, initial_retry_delay_seconds, max_retry_delay_seconds
                )
                logging.debug(
                    f"Sleeping for {delay:.2f} seconds due to API error status {status_code}."
                )
                time.sleep(delay)
                return True  # Indicate retry
            return False  # Max retries reached for this path
        else:
            logging.error(
                f"Non-retriable or unrecognized API error occurred (status: {status_code}). Raising."
            )
            raise e  # Re-raise non-retriable error

    # Main retry loop
    for attempt in range(1, max_retries + 1):
        try:
            response = client_function(**api_params)

            if is_response_valid_fn(response):
                logging.info(f"[Attempt {attempt}/{max_retries}] API call succeeded.")
                return response
            else:
                if _handle_invalid_response_content(attempt):
                    continue
                else:  # Max retries reached after invalid content
                    break

        except rate_limit_exceptions as e:
            if _handle_rate_limit_error(e, attempt):
                continue
            else:  # Max retries reached after rate limit
                break

        except api_error_exceptions as e:
            # _handle_api_error will raise if non-retriable, or return True to continue
            if _handle_api_error(e, attempt):
                continue
            else:  # Max retries reached for retriable API error
                break

        except Exception as e:  # Catch-all for truly unexpected errors
            logging.exception(
                f"[Attempt {attempt}/{max_retries}] Unexpected exception: {e}. Raising."
            )
            raise e  # Re-raise unexpected errors immediately

    logging.error(f"Exceeded maximum {max_retries} retry attempts. API call failed.")
    raise RuntimeError(f"API call failed after {max_retries} retries.")
