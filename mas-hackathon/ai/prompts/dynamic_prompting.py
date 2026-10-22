"""Dynamic prompt construction for browser-use agents.

Adapted from agentlab's dynamic_prompting to use the provider-agnostic
MessageBuilder class and the project's Observation / ActionSet abstractions.

All prompt elements produce either plain strings (folded into MessageBuilder
by the caller) or MessageBuilder instances directly.  Parsing helpers for
``<action>`` and ``<think>`` tags are preserved for backward compatibility.
"""

from __future__ import annotations

import abc
import logging
import platform
import time
from copy import copy
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Literal

from acua.envs.action_sets import ActionSet
from acua.envs.hints import (
    CandidateSummary,
    ErrorHints,
    ExperimentHints,
    FocusedElementHints,
    LastBatchHints,
    ScreenshotSOMHints,
    SpecialistViewHints,
    TabsHints,
    TargetMetricsHints,
)
from acua.envs.observation import Observation as AgentObservation
from acua.envs.utils import function_to_tool_schema
from acua.prompts.parsing import (
    ParseError,
    extract_code_blocks,
    parse_html_tags_raise,
)
from acua.providers.util import MessageBuilder, image_to_png_base64_url

logger = logging.getLogger(__name__)


# ── Flag dataclasses ────────────────────────────────────────────────


@dataclass
class ObsFlags:
    """Controls which observation components are included in the prompt."""

    use_html: bool = True
    use_ax_tree: bool = False
    use_tabs: bool = False
    use_focused_element: bool = False
    use_error_logs: bool = False
    use_history: bool = False
    use_past_error_logs: bool = False
    use_action_history: bool = False
    use_think_history: bool = False
    use_diff: bool = False
    html_type: str = "pruned_html"
    use_screenshot: bool = True
    use_som: bool = False
    extract_visible_tag: bool = False
    extract_clickable_tag: bool = False
    extract_coords: Literal["False", "center", "box"] = "False"
    filter_visible_elements_only: bool = False
    openai_vision_detail: Literal["low", "high", "auto"] = "auto"
    filter_with_bid_only: bool = False
    filter_som_only: bool = False

    # ── Biology experiment flags (ignored by web/OS prompts) ────────
    use_experiment_state: bool = False  # show cycle / budget summary
    use_candidate_pool: bool = False    # show ranked gene candidate table
    max_candidates: int = 50            # max rows in candidate table
    show_correlation: bool = False      # include cross-target correlation matrix
    show_last_batch: bool = False       # show last batch hit outcomes


@dataclass
class ActionFlags:
    """Controls how the action prompt is rendered."""

    action_set: ActionSet | None = None
    long_description: bool = True
    individual_examples: bool = False
    multi_actions: bool = False
    is_strict: bool = False


# ── Base prompt element ─────────────────────────────────────────────


class PromptElement:
    """Base class for all prompt elements.  Elements can be hidden via *visible*."""

    _prompt: str = ""
    _abstract_ex: str = ""
    _concrete_ex: str = ""

    def __init__(self, visible: bool = True) -> None:
        self._visible = visible

    @property
    def prompt(self) -> str:
        if self.is_visible:
            return self._prompt
        return ""

    @property
    def abstract_ex(self) -> str:
        if self.is_visible:
            return self._abstract_ex
        return ""

    @property
    def concrete_ex(self) -> str:
        if self.is_visible:
            return self._concrete_ex
        return ""

    @property
    def is_visible(self) -> bool:
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _parse_answer(self, text_answer: str) -> dict:
        return {}

    def parse_answer(self, text_answer: str) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        return {}

    # ── MessageBuilder helpers ──────────────────────────────────

    def to_message(self, role: str = "user") -> MessageBuilder:
        """Convert this element's prompt text into a MessageBuilder message."""
        text = self.prompt
        if not text:
            return MessageBuilder(role)
        return MessageBuilder(role).add_text(text)

    def to_system_message(self) -> MessageBuilder:
        return self.to_message("system")

    def to_user_message(self) -> MessageBuilder:
        return self.to_message("user")


# ── Shrinkable / Trunkater ──────────────────────────────────────────


class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Progressively reduce the size of this prompt element."""


class Trunkater(Shrinkable):
    """Shrinkable that truncates from the bottom after *start_trunkate_iteration*."""

    def __init__(
        self, visible, shrink_speed: float = 0.3, start_trunkate_iteration: int = 10
    ):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = "\n".join(lines[:new_line_count])
            self._prompt += (
                f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."
            )
        self.shrink_calls += 1


def fit_tokens(
    shrinkable: Shrinkable,
    max_prompt_tokens: int | None = None,
    max_iterations: int = 20,
    count_tokens_fn=None,
    additional_prompts: list[str] | None = None,
) -> str:
    """Shrink *shrinkable* until its prompt fits *max_prompt_tokens*.

    Args:
        shrinkable: The prompt element to shrink.
        max_prompt_tokens: Token budget.  ``None`` means unlimited.
        max_iterations: Max shrink rounds.
        count_tokens_fn: ``fn(text) -> int``.  If ``None``, a rough
            whitespace-split estimator is used.
        additional_prompts: Extra text whose tokens are subtracted from the budget.

    Returns:
        The (possibly shrunk) prompt string.
    """
    if max_prompt_tokens is None:
        return shrinkable.prompt

    if count_tokens_fn is None:
        # Rough token estimate: ~4 chars per token
        def count_tokens_fn(text: str) -> int:
            return len(text) // 4

    additional_prompts = additional_prompts or []
    for extra in additional_prompts:
        max_prompt_tokens -= count_tokens_fn(extra) + 1

    for _ in range(max_iterations):
        prompt = shrinkable.prompt
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        if count_tokens_fn(prompt_str) <= max_prompt_tokens:
            return prompt
        shrinkable.shrink()

    logger.info(
        dedent(f"""\
        After {max_iterations} shrink iterations the prompt is still too long. Returning as-is.""")
    )
    return shrinkable.prompt


# ── Concrete prompt elements ────────────────────────────────────────


class HTML(Trunkater):
    def __init__(
        self,
        html: str,
        visible_elements_only: bool,
        visible: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(visible=visible, start_trunkate_iteration=5)
        note = ""
        if visible_elements_only:
            note = (
                "Note: only elements that are visible in the viewport are presented. "
                "You might need to scroll the page, or open tabs or menus to see more.\n\n"
            )
        self._prompt = f"\n{prefix}HTML:\n{note}{html}\n"


class AXTree(Trunkater):
    def __init__(
        self,
        ax_tree: str,
        visible_elements_only: bool,
        visible: bool = True,
        coord_type: str | None = None,
        visible_tag: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(visible=visible, start_trunkate_iteration=10)
        bid_info = (
            "Note: [bid] is the unique alpha-numeric identifier at the beginning of lines "
            "for each element in the AXTree. Always use bid to refer to elements in your actions.\n\n"
        )
        coord_note = ""
        if coord_type == "center":
            coord_note = (
                "Note: center coordinates are provided in parenthesis and are "
                "relative to the top left corner of the page.\n\n"
            )
        elif coord_type == "box":
            coord_note = (
                "Note: bounding box of each object are provided in parenthesis and "
                "are relative to the top left corner of the page.\n\n"
            )
        visible_note = ""
        if visible_elements_only:
            visible_note = (
                "Note: only elements that are visible in the viewport are presented. "
                "You might need to scroll the page, or open tabs or menus to see more.\n\n"
            )
        tag_note = ""
        if visible_tag:
            tag_note = (
                'Note: You can only interact with visible elements. If the "visible" tag '
                "is not present, the element is not visible on the page.\n\n"
            )
        self._prompt = f"\n{prefix}AXTree:\n{bid_info}{coord_note}{visible_note}{tag_note}{ax_tree}\n"


class Error(PromptElement):
    def __init__(
        self,
        error: ErrorHints | str,
        visible: bool = True,
        prefix: str = "",
        limit_logs: bool = True,
    ) -> None:
        error_text = error.last_action_error if isinstance(error, ErrorHints) else error
        logs_separator = "Call log:"
        if limit_logs and logs_separator in error_text:
            error_text, logs = error_text.split(logs_separator, 1)
            logs = "\n".join(logs.split("\n")[:10])
            error_text = error_text + f"\n{logs_separator}\n{logs}"
        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error_text}\n"


class FocusedElement(PromptElement):
    def __init__(
        self,
        focused: FocusedElementHints | str | None,
        visible: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(visible=visible)
        bid = (
            focused.focused_element_bid
            if isinstance(focused, FocusedElementHints)
            else focused
        )
        self._prompt = f"\n{prefix}Focused element:\n"
        self._prompt += f"bid={repr(bid)}\n" if bid else "None\n"


class Tabs(PromptElement):
    """Render open browser tabs from :class:`TabsHints`."""

    def __init__(
        self, tabs_hints: TabsHints, visible: bool = True, prefix: str = ""
    ) -> None:
        super().__init__(visible=visible)
        self._tabs = tabs_hints
        self._prefix = prefix

    @property
    def _prompt(self) -> str:
        urls = self._tabs.open_pages_urls
        titles = self._tabs.open_pages_titles
        active = self._tabs.active_page_index

        pieces = [f"\n{self._prefix}Currently open tabs:"]
        for i, (url, title) in enumerate(zip(urls, titles)):
            active_str = " (active tab)" if i == active else ""
            pieces.append(f"Tab {i}{active_str}:\n    Title: {title}\n    URL: {url}")
        return "\n".join(pieces)


class ObservationPrompt(Shrinkable):
    """Builds the observation section of the prompt from an :class:`AgentObservation`.

    This is the provider-agnostic replacement for the original ``Observation``
    prompt element which operated on raw browsergym dicts.
    """

    def __init__(self, obs: AgentObservation, flags: ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs
        self._som_hints = obs.screenshot_som or ScreenshotSOMHints()

        self.tabs = Tabs(
            obs.tabs or TabsHints(),
            visible=lambda: flags.use_tabs,
            prefix="## ",
        )

        html_text = obs.html_dom or ""
        self.html = HTML(
            html_text,
            visible_elements_only=flags.filter_visible_elements_only,
            visible=lambda: flags.use_html and obs.has_html_dom,
            prefix="## ",
        )

        ax_text = obs.accessibility_tree or ""
        self.ax_tree = AXTree(
            ax_text,
            visible_elements_only=flags.filter_visible_elements_only,
            visible=lambda: flags.use_ax_tree and obs.has_accessibility_tree,
            coord_type=flags.extract_coords,
            visible_tag=flags.extract_visible_tag,
            prefix="## ",
        )

        error_hints = obs.error or ErrorHints()
        self.error = Error(
            error_hints,
            visible=lambda: (
                flags.use_error_logs and bool(error_hints.last_action_error)
            ),
            prefix="## ",
        )

        focused_hints = obs.focused_element or FocusedElementHints()
        self.focused_element = FocusedElement(
            focused_hints,
            visible=flags.use_focused_element,
            prefix="## ",
        )

    def shrink(self) -> None:
        self.ax_tree.shrink()
        self.html.shrink()

    @property
    def _step_info_prompt(self) -> str:
        if self.obs.step_info is None:
            return ""
        current, maximum = self.obs.step_info
        text = f"\n## Step {current} / {maximum}\n"
        if current == maximum:
            text += (
                "\nThis is your FINAL step. You must provide your best answer now "
                "using send_msg_to_user() or the equivalent action. "
                "Do not attempt further exploration — submit your answer immediately.\n"
            )
        elif maximum - current <= 3:
            text += (
                f"\nWarning: only {maximum - current} step(s) remaining. "
                "Start wrapping up and prepare to submit your answer soon.\n"
            )
        return text

    @property
    def _prompt(self) -> str:
        return (
            "\n# Observation of current step:\n"
            f"{self._step_info_prompt}"
            f"{self.tabs.prompt}{self.html.prompt}{self.ax_tree.prompt}"
            f"{self.focused_element.prompt}{self.error.prompt}\n\n"
        )

    def to_user_message(self) -> MessageBuilder:
        """Build a MessageBuilder with text + optional screenshot."""
        msg = MessageBuilder.user()
        text = self.prompt
        if text.strip():
            msg.add_text(text)
        if self.flags.use_screenshot and self.obs.has_screenshot:
            if self.flags.use_som:
                som_screenshot = self._som_hints.screenshot_som
                if som_screenshot is not None:
                    msg.add_text(
                        "\n## Screenshot:\nHere is a screenshot of the page, "
                        "it is annotated with bounding boxes and corresponding bids:"
                    )
                    msg.add_image(image_to_png_base64_url(som_screenshot))
                else:
                    # Fallback to regular screenshot if SOM not available
                    msg.add_text("\n## Screenshot:\nHere is a screenshot of the page:")
                    b64 = self.obs.screenshot_base64()
                    msg.add_image(f"data:image/png;base64,{b64}")
            else:
                msg.add_text("\n## Screenshot:\nHere is a screenshot of the page:")
                b64 = self.obs.screenshot_base64()
                msg.add_image(f"data:image/png;base64,{b64}")
        return msg


# ── Instruction elements ────────────────────────────────────────────


class MacNote(PromptElement):
    def __init__(self) -> None:
        super().__init__(visible=platform.system() == "Darwin")
        self._prompt = "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"


class BeCautious(PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = (
            "\nBe very cautious. Avoid submitting anything before verifying the effect of your "
            "actions. Take the time to explore the effect of safe actions first. For example "
            "you can fill a few elements of a form, but don't click submit before verifying "
            "that everything was filled correctly. Also, if asked to return counts, "
            "answer with a number. For example say `0 instances` instead of `no instances`. "
            "Also, if you are repeatedly clicking a button, and see that hardly anything is "
            "changing on the page, this maybe because you are downloading a file. Ignore "
            "this download, and instead navigate the page via scrolling to see the results. "
            "Take your time to explore pages, before you are confident that the information you need "
            "is not there and you decide to navigate elsewhere. \n"
        )


class GoalInstructions(PromptElement):
    """Renders the goal section of the prompt.

    Accepts either a ``MessageBuilder`` (from an Observation) or a
    raw browsergym ``goal_object`` list for backward compatibility.
    """

    def __init__(
        self,
        goal: MessageBuilder | list[dict] | None = None,
        visible: bool = True,
        extra_instructions: str | None = None,
    ) -> None:
        super().__init__(visible)
        self._goal = goal
        self._extra_instructions = extra_instructions

    def to_user_message(self) -> MessageBuilder:
        """Build a user message containing the goal instructions."""
        msg = MessageBuilder.user()
        msg.add_text(
            "# Instructions\n"
            "Review the current state of the page and all other information to find the best "
            "possible next action to accomplish your goal. Your answer will be interpreted "
            "and executed by a program, make sure to follow the formatting instructions.\n\n"
            "## Goal:\n"
        )

        if isinstance(self._goal, MessageBuilder):
            # Merge content from the goal MessageBuilder
            for item in self._goal.content:
                if "text" in item:
                    msg.add_text(item["text"])
                elif "image" in item:
                    msg.add_image(item["image"])
        elif isinstance(self._goal, list):
            # Raw browsergym goal_object (list of dicts with "type"/"text"/"image_url")
            for item in self._goal:
                if item.get("type") == "text":
                    msg.add_text(item["text"])
                elif item.get("type") == "image_url":
                    msg.add_image(item["image_url"]["url"])
        elif isinstance(self._goal, str):
            msg.add_text(self._goal)

        if self._extra_instructions:
            msg.add_text(f"\n## Extra instructions:\n\n{self._extra_instructions}\n")

        return msg

    @property
    def _prompt(self) -> str:
        # Text-only fallback
        parts = [
            "# Instructions\n"
            "Review the current state of the page and all other information to find the best "
            "possible next action to accomplish your goal.\n\n## Goal:\n"
        ]
        if isinstance(self._goal, MessageBuilder):
            for item in self._goal.content:
                if "text" in item:
                    parts.append(item["text"])
        elif isinstance(self._goal, str):
            parts.append(self._goal)
        if self._extra_instructions:
            parts.append(f"\n## Extra instructions:\n\n{self._extra_instructions}\n")
        return "\n".join(parts)


class ChatInstructions(PromptElement):
    def __init__(
        self,
        chat_messages: list[dict[str, Any]],
        visible: bool = True,
        extra_instructions: str | None = None,
    ) -> None:
        super().__init__(visible)
        lines = [
            "# Instructions\n\n"
            "You are a UI Assistant, your goal is to help the user perform tasks using a web browser. "
            "You can communicate with the user via a chat, in which the user gives you instructions "
            "and in which you can send back messages. You have access to a web browser that both you "
            "and the user can see, and with which only you can interact via specific commands.\n\n"
            "Review the instructions from the user, the current state of the page and all other "
            "information to find the best possible next action to accomplish your goal. Your answer "
            "will be interpreted and executed by a program, make sure to follow the formatting "
            "instructions.\n\n## Chat messages:\n\n"
        ]
        for msg in chat_messages:
            utc = time.asctime(time.gmtime(msg["timestamp"]))
            local = time.asctime(time.localtime(msg["timestamp"]))
            lines.append(
                f" - [{msg['role']}] UTC: {utc} - Local: {local} - {msg['message']}"
            )

        if extra_instructions:
            lines.append(f"\n\n## Extra instructions:\n\n{extra_instructions}\n")

        self._prompt = "\n".join(lines)


class SystemPrompt(PromptElement):
    _prompt = (
        "You are an agent trying to solve a web task based on the content of the page and "
        "user instructions. You can interact with the page and explore, and send messages to the user. "
        "Each time you submit an action it will be sent to the browser and you will receive a new page."
    )

    def to_system_message(self) -> MessageBuilder:
        return MessageBuilder.system().add_text(self.prompt)


# ── Action prompt with parsing ──────────────────────────────────────


class ActionPrompt(PromptElement):
    """Describes the available actions and parses ``<action>`` tags from LLM output."""

    _concrete_ex = "\n<action>\nclick('a324')\n</action>\n"

    def __init__(
        self, action_set: ActionSet, action_flags: ActionFlags | None = None
    ) -> None:
        super().__init__()
        self.action_set = action_set
        self.action_flags = action_flags or ActionFlags(action_set=action_set)

        # Build action descriptions from the ActionSet's function stubs
        descriptions: list[str] = []
        for func in action_set.value:
            schema = function_to_tool_schema(func)
            name = schema["name"]
            desc = schema.get("description", "")
            params = schema.get("parameters", {}).get("properties", {})
            sig_parts = []
            for pname, pschema in params.items():
                ptype = pschema.get("type", "any")
                if "default" in pschema:
                    sig_parts.append(f"{pname}: {ptype} = {pschema['default']!r}")
                else:
                    sig_parts.append(f"{pname}: {ptype}")
            sig = ", ".join(sig_parts)
            descriptions.append(f"  {name}({sig})\n    {desc}")

        action_list = "\n\n".join(descriptions)
        action_set_info = (
            "Note: This action set allows you to interact with your environment. Most of them "
            "are python functions executing playwright code. The primary way of referring to "
            "elements in the page is through bid which are specified in your observations.\n\n"
        )
        self._prompt = (
            f"# Action space:\n{action_set_info}{action_list}{MacNote().prompt}\n"
        )

        # Build abstract example from the first function in the set
        first_func = next(iter(action_set.value))
        self._abstract_ex = f"\n<action>\n{first_func.__name__}(...)\n</action>\n"

    def _parse_answer(self, text_answer: str) -> dict:
        """Parse an ``<action>`` tag from the LLM's text output."""
        try:
            ans_dict = parse_html_tags_raise(
                text_answer, keys=("action",), merge_multiple=True
            )
        except ParseError as e:
            if self.action_flags.is_strict:
                raise
            blocks = extract_code_blocks(text_answer)
            if not blocks:
                raise
            code = "\n".join(block for _, block in blocks)
            ans_dict = {"action": code, "parse_error": str(e)}

        if ans_dict["action"] == "None":
            ans_dict["action"] = None

        return ans_dict


# ── Think prompt with parsing ───────────────────────────────────────


class Think(PromptElement):
    _prompt = ""

    _abstract_ex = (
        "\n<think>\n"
        "Think step by step. If you need to make calculations such as coordinates, write them here. "
        "Describe the effect that your previous action had on the current content of the page.\n"
        "</think>\n"
    )
    _concrete_ex = (
        "\n<think>\n"
        'From previous action I tried to set the value of year to "2022", '
        "using select_option, but it doesn't appear to be in the form. It may be a "
        'dynamic dropdown, I will try using click with the bid "a324" and look at the '
        "response from the page.\n"
        "</think>\n"
    )

    def _parse_answer(self, text_answer: str) -> dict:
        try:
            return parse_html_tags_raise(
                text_answer, keys=("think",), merge_multiple=True
            )
        except ParseError:
            return {"think": text_answer, "parse_error": "Could not parse <think> tag."}


# ── History ─────────────────────────────────────────────────────────


class HistoryStep(Shrinkable):
    """One step in the action history."""

    def __init__(
        self,
        action: str,
        error: str = "",
        thought: str = "",
        memory: str | None = None,
        flags: ObsFlags | None = None,
    ) -> None:
        super().__init__()
        self._flags = flags or ObsFlags()
        self._action = action
        self._thought = thought
        self._memory = memory
        self.error_elem = Error(
            error,
            visible=(
                lambda: (
                    self._flags.use_error_logs
                    and bool(error)
                    and self._flags.use_past_error_logs
                )
            ),
            prefix="### ",
        )

    def shrink(self) -> None:
        pass

    @property
    def _prompt(self) -> str:
        prompt = ""
        if self._flags.use_think_history and self._thought:
            prompt += f"\n<think>\n{self._thought}\n</think>\n"
        if self._flags.use_action_history:
            prompt += f"\n<action>\n{self._action}\n</action>\n"
        prompt += self.error_elem.prompt
        if self._memory is not None:
            prompt += f"\n<memory>\n{self._memory}\n</memory>\n"
        return prompt


class History(Shrinkable):
    """Renders the complete action history."""

    def __init__(
        self,
        steps: list[HistoryStep] | None = None,
        visible: bool = True,
    ) -> None:
        super().__init__(visible=visible)
        self.steps: list[HistoryStep] = steps or []

    def add_step(self, step: HistoryStep) -> None:
        self.steps.append(step)

    def shrink(self) -> None:
        for step in self.steps:
            step.shrink()

    @property
    def _prompt(self) -> str:
        pieces = ["# History of interaction with the task:\n"]
        for i, step in enumerate(self.steps):
            pieces.append(f"## step {i}")
            pieces.append(step.prompt)
        return "\n".join(pieces) + "\n"


# ── Observation preprocessor ────────────────────────────────────────


def make_obs_preprocessor(flags: ObsFlags):
    """Return a function that preprocesses a raw browsergym obs dict.

    This enriches the dict with ``axtree_txt``, ``dom_txt``, ``pruned_html``,
    and ``screenshot_som`` fields based on *flags*.  It requires the optional
    ``browsergym`` package to be installed.

    The returned function is meant to be used *before* constructing an
    :class:`AgentObservation`, e.g. inside a custom ``to_agent_observation``
    override of :class:`~acua.envs.browsergym.env.BrowserGymEnv`.
    """
    try:
        from browsergym.utils.obs import (
            flatten_axtree_to_str,
            flatten_dom_to_str,
            overlay_som,
            prune_html,
        )
    except ImportError as e:
        raise ImportError(
            "make_obs_preprocessor requires the browsergym package. "
            "Install it with: pip install browsergym-webarena"
        ) from e

    def obs_mapping(obs: dict) -> dict:
        obs = copy(obs)
        extra = obs.get("extra_element_properties", {})
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            extra_properties=extra,
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=extra,
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        obs["screenshot_som"] = overlay_som(obs["screenshot"], extra_properties=extra)
        return obs

    return obs_mapping


# ── Experiment observation prompt (biology / gene-perturbation) ──────


def _fmt_float(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}"


def _ascii_table(
    headers: list[str],
    rows: list[list[str]],
    max_rows: int | None = None,
) -> str:
    """Render a compact ASCII table with fixed-width columns."""
    if max_rows is not None:
        rows = rows[:max_rows]
    all_rows = [headers] + rows
    col_widths = [max(len(r[c]) for r in all_rows) for c in range(len(headers))]
    sep = "  "

    def fmt_row(row: list[str]) -> str:
        return sep.join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    divider = "-" * (sum(col_widths) + len(sep) * (len(headers) - 1))
    lines = [fmt_row(headers), divider]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


class ExperimentObservationPrompt(Shrinkable):
    """Builds the observation prompt for biology gene-perturbation experiments.

    Replaces ObservationPrompt when the Observation carries experiment fields.
    """

    def __init__(self, obs: AgentObservation, flags: ObsFlags) -> None:
        super().__init__()
        self.obs = obs
        self.flags = flags
        self._n_candidates = flags.max_candidates

    def shrink(self) -> None:
        self._n_candidates = max(10, int(self._n_candidates * 0.7))

    @property
    def _prompt(self) -> str:
        parts: list[str] = []

        # ── Step info ───────────────────────────────────────────────
        if self.obs.step_info is not None:
            current, maximum = self.obs.step_info
            parts.append(f"\n## Cycle {current} / {maximum}")
            if current == maximum:
                parts.append(
                    "This is the FINAL cycle. Submit your best gene selection now."
                )
            elif maximum - current <= 2:
                parts.append(
                    f"Warning: only {maximum - current} cycle(s) remaining."
                )

        # ── Experiment state ────────────────────────────────────────
        exp = self.obs.experiment
        if self.flags.use_experiment_state and exp is not None:
            budget_total = exp.batch_size * exp.n_cycles
            parts.append(
                f"\n## Budget\n"
                f"- Remaining queries : {exp.budget_remaining} / {budget_total}\n"
                f"- Observed genes    : {exp.n_observed} / {exp.n_total}\n"
                f"- Batch size        : {exp.batch_size}\n"
                f"- Last cycle hit rate: {_fmt_float(exp.last_hit_rate)}"
            )

        # ── Target metrics ──────────────────────────────────────────
        tm = self.obs.target_metrics
        if self.flags.use_experiment_state and tm is not None and tm.target_names:
            K = len(tm.target_names)
            headers = ["Target", "Recall", "Hits", "Threshold"]
            rows = [
                [
                    tm.target_names[k],
                    _fmt_float(tm.per_target_recall[k] if k < len(tm.per_target_recall) else 0.0),
                    str(tm.per_target_hits[k] if k < len(tm.per_target_hits) else 0),
                    _fmt_float(tm.thresholds[k] if k < len(tm.thresholds) else 0.0),
                ]
                for k in range(K)
            ]
            parts.append(f"\n## Target Metrics\n{_ascii_table(headers, rows)}")
            if tm.mean_gp_sigma > 0:
                parts.append(f"Mean GP sigma (pool): {_fmt_float(tm.mean_gp_sigma)}")

            if self.flags.show_correlation and tm.correlation_matrix is not None:
                K = len(tm.correlation_matrix)
                corr_rows = [
                    [tm.target_names[i]] + [_fmt_float(tm.correlation_matrix[i][j], 2) for j in range(K)]
                    for i in range(K)
                ]
                corr_headers = [""] + list(tm.target_names)
                parts.append(f"\n### Cross-target correlation\n{_ascii_table(corr_headers, corr_rows)}")

        # ── Candidate pool ──────────────────────────────────────────
        candidates = self.obs.candidates
        if self.flags.use_candidate_pool and candidates:
            n_show = self._n_candidates
            shown = candidates[:n_show]

            tm = self.obs.target_metrics
            K = len(tm.target_names) if (tm and tm.target_names) else 0
            t_names = tm.target_names if tm else []

            headers = ["Gene", "Idx", "MaxPOH", "Sigma"] + [f"POH_{t}" for t in t_names]
            rows = [
                [
                    c.gene_name,
                    str(c.gene_index),
                    _fmt_float(c.max_poh),
                    _fmt_float(c.mean_sigma),
                ] + [_fmt_float(c.per_target_poh[k]) if k < len(c.per_target_poh) else "0.000" for k in range(K)]
                for c in shown
            ]
            hidden = max(0, len(candidates) - n_show)
            suffix = f"\n(showing top {n_show} of {len(candidates) + hidden} candidates)" if hidden else ""
            parts.append(
                f"\n## Candidate Pool — select exactly {self.obs.experiment.batch_size if self.obs.experiment else '?'} genes\n"
                f"{_ascii_table(headers, rows)}{suffix}"
            )

        # ── Last batch outcomes ─────────────────────────────────────
        lb = self.obs.last_batch
        if self.flags.show_last_batch and lb is not None and lb.selected_genes:
            tm = self.obs.target_metrics
            t_names = tm.target_names if tm else []
            hit_parts = ", ".join(
                f"{t}: {h}" for t, h in zip(t_names, lb.per_target_hits)
            )
            parts.append(
                f"\n## Last Batch\n"
                f"Selected : {', '.join(lb.selected_genes[:10])}"
                + (" ..." if len(lb.selected_genes) > 10 else "") + "\n"
                f"New hits : {hit_parts if hit_parts else 'none'}\n"
                f"Any hit  : {'yes' if lb.was_hit_any else 'no'}"
            )

        # ── Error ───────────────────────────────────────────────────
        err = self.obs.error
        if err and err.last_action_error:
            parts.append(
                f"\n## Error from previous action\n{err.last_action_error}"
            )

        return "\n# Experiment Observation\n" + "\n".join(parts) + "\n\n"

    def to_user_message(self) -> MessageBuilder:
        msg = MessageBuilder.user()
        text = self.prompt
        if text.strip():
            msg.add_text(text)
        return msg


# ── Single-agent observation prompt ──────────────────────────────────────────


class SingleAgentObservationPrompt(Shrinkable):
    """Observation prompt for the single-agent ablation.

    The agent sees ALL K targets' hit history in one prompt and calls
    select_genes() directly — no nomination/discussion protocol.

    This is the ablation counterpart of the multi-agent protocol:
    the same information is available but the reasoning is done by one
    LLM call instead of K specialist calls followed by a moderator call.

    Shrinking reduces the pool sample and confirmed-hit lists.
    """

    def __init__(
        self,
        obs: AgentObservation,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.obs = obs
        self.batch_size = batch_size
        self._max_pool_shown = 50
        self._max_hits_per_target = 30

    def shrink(self) -> None:
        self._max_pool_shown = max(10, int(self._max_pool_shown * 0.6))
        self._max_hits_per_target = max(5, int(self._max_hits_per_target * 0.7))

    @property
    def _prompt(self) -> str:
        parts: list[str] = []
        exp = self.obs.experiment
        tm = self.obs.target_metrics
        views = self.obs.specialist_views or []

        # ── Cycle / budget ──────────────────────────────────────────
        if exp is not None:
            budget_total = exp.batch_size * exp.n_cycles
            parts.append(
                f"## Cycle {exp.cycle} / {exp.n_cycles} | "
                f"Budget: {exp.budget_remaining} / {budget_total} queries remaining\n"
                f"Genes observed: {exp.n_observed} / {exp.n_total}"
            )

        # ── Overall progress summary ────────────────────────────────
        n_obs = exp.n_observed if exp else 1
        n_total_genes = exp.n_total if exp else 1
        if tm and tm.target_names:
            recall_lines = []
            for name, r, h in zip(tm.target_names, tm.per_target_recall, tm.per_target_hits):
                hit_rate = h / max(n_obs, 1)
                recall_lines.append(
                    f"  {name}: {h} hits / {n_obs} tested = {hit_rate:.1%} hit rate  "
                    f"| screen coverage {h}/{round(h/max(r,1e-9)):.0f} ({r:.1%} recall)"
                )
            if exp and exp.last_hit_rate > 0:
                recall_lines.append(
                    f"  Last batch overall: {exp.last_hit_rate:.1%} of queried genes were hits"
                )
            parts.append("\n## Progress\n" + "\n".join(recall_lines))

        # ── Per-target detail ───────────────────────────────────────
        parts.append("\n## Per-Target History")
        for v in views:
            target_label = v.phenotype or v.target_name
            cell_label = f" ({v.cell_type})" if v.cell_type else ""
            n_total_hits = v.n_total_hits
            background_rate = n_total_hits / max(n_total_genes, 1)
            parts.append(f"\n### {target_label}{cell_label}")

            if v.confirmed_hit_genes:
                shown = v.confirmed_hit_genes[-self._max_hits_per_target:]
                prefix = f"(latest {self._max_hits_per_target}) " if len(v.confirmed_hit_genes) > self._max_hits_per_target else ""
                parts.append(
                    f"Confirmed hits {prefix}({len(v.confirmed_hit_genes)} total): "
                    + ", ".join(shown)
                )
            else:
                parts.append("Confirmed hits: None yet.")

            if v.last_batch_hits or v.last_batch_misses:
                n_batch_hits = len(v.last_batch_hits)
                n_batch_total = n_batch_hits + len(v.last_batch_misses)
                batch_rate = n_batch_hits / max(n_batch_total, 1)
                enrichment = batch_rate / max(background_rate, 1e-9)
                hit_str = ", ".join(v.last_batch_hits) if v.last_batch_hits else "none"
                miss_count = len(v.last_batch_misses)
                misses_preview = (
                    ", ".join(v.last_batch_misses[:5]) + (" ..." if miss_count > 5 else "")
                ) if v.last_batch_misses else ""
                parts.append(
                    f"Last batch: {n_batch_hits}/{n_batch_total} hits "
                    f"= {batch_rate:.1%} ({enrichment:.1f}× background of {background_rate:.1%})\n"
                    f"  Hits: {hit_str}\n"
                    f"  Misses: {miss_count}" + (f" ({misses_preview})" if misses_preview else "")
                )

        # ── Pool sample (from first view — same pool shown to all specialists) ──
        if views and views[0].pool_sample:
            pool = views[0].pool_sample[:self._max_pool_shown]
            n_total = views[0].n_pool_total
            parts.append(
                f"\n## Available Gene Pool — SELECT ONLY FROM THIS LIST\n"
                f"({n_total:,} unobserved genes total; {len(pool)} shown)\n"
                + ", ".join(pool)
            )

        # ── Error ───────────────────────────────────────────────────
        err = self.obs.error
        if err and err.last_action_error:
            parts.append(f"\n## Error from previous action\n{err.last_action_error}")

        # ── Task instructions ───────────────────────────────────────
        parts.append(
            f"\n## Your Task\n"
            f"Select exactly {self.batch_size} genes for the next query batch.\n"
            "IMPORTANT: You MUST choose genes from the pool list above.\n"
            "Do NOT use gene names from general biology knowledge — only genes\n"
            "in the pool list exist in this screen.\n\n"
            "Strategy: consider all targets — prefer genes likely to help\n"
            "multiple lagging targets, or mechanistically relevant ones.\n\n"
            f'Call: select_genes(genes=["GENE1", "GENE2", ...])  '
            f"(exactly {self.batch_size} genes)"
        )

        return "\n# Multi-Target Observation\n" + "\n".join(parts) + "\n"

    def to_user_message(self) -> MessageBuilder:
        msg = MessageBuilder.user()
        text = self.prompt
        if text.strip():
            msg.add_text(text)
        return msg


# ── Specialist observation prompt ─────────────────────────────────────────────


class SpecialistObservationPrompt(Shrinkable):
    """Observation prompt shown to a single specialist agent.

    Contains only this specialist's target metrics and hit history — the
    specialist has no cross-target visibility.  A random pool sample is
    shown for reference; agents may nominate genes outside this sample.

    Shrinking reduces the pool sample shown.
    """

    def __init__(
        self,
        obs: AgentObservation,
        specialist_view: SpecialistViewHints,
        nominations_requested: int,
    ) -> None:
        super().__init__()
        self.obs = obs
        self.view = specialist_view
        self.nominations_requested = nominations_requested
        # Shrinkable: reduce pool sample on each shrink call
        self._max_pool_shown = len(specialist_view.pool_sample)

    def shrink(self) -> None:
        self._max_pool_shown = max(10, int(self._max_pool_shown * 0.6))

    @property
    def _prompt(self) -> str:
        parts: list[str] = []
        exp = self.obs.experiment
        v = self.view

        # ── Cycle / budget status ───────────────────────────────────
        if exp is not None:
            budget_total = exp.batch_size * exp.n_cycles
            parts.append(
                f"## Cycle {exp.cycle} / {exp.n_cycles} | "
                f"Budget: {exp.budget_remaining} / {budget_total} queries remaining\n"
                f"Genes observed so far: {exp.n_observed} / {exp.n_total}"
            )

        # ── This specialist's target ────────────────────────────────
        target_label = v.phenotype or v.target_name
        cell_label = f" in {v.cell_type}" if v.cell_type else ""
        n_obs = exp.n_observed if exp else 1
        # Hits / genes-tested  (empirical hit rate in our sample)
        hit_rate_cumul = v.n_hits_found / max(n_obs, 1)
        # Background hit rate: n_total_hits / n_total_genes
        n_total_genes = exp.n_total if exp else max(v.n_total_hits, 1)
        background_rate = v.n_total_hits / max(n_total_genes, 1)
        enrichment = hit_rate_cumul / max(background_rate, 1e-9)
        parts.append(
            f"\n## Your Target: {target_label}{cell_label}\n"
            f"Progress : {v.n_hits_found} hits found / {n_obs} genes tested "
            f"= {hit_rate_cumul:.1%} hit rate "
            f"({enrichment:.1f}× background of {background_rate:.1%})\n"
            f"Screen coverage: {v.n_hits_found}/{v.n_total_hits} hits discovered "
            f"({v.recall:.1%} recall)"
        )

        # ── Confirmed hits ──────────────────────────────────────────
        if v.confirmed_hit_genes:
            shown_hits = v.confirmed_hit_genes[-30:]
            prefix = "(latest 30) " if len(v.confirmed_hit_genes) > 30 else ""
            parts.append(
                f"\n## Confirmed Hits {prefix}({len(v.confirmed_hit_genes)} total)\n"
                + ", ".join(shown_hits)
            )
        else:
            parts.append("\n## Confirmed Hits\nNone yet — initial random seed only.")

        # ── Last batch outcomes ─────────────────────────────────────
        if v.last_batch_hits or v.last_batch_misses:
            n_batch_hits = len(v.last_batch_hits)
            n_batch_total = n_batch_hits + len(v.last_batch_misses)
            batch_rate = n_batch_hits / max(n_batch_total, 1)
            enrichment_batch = batch_rate / max(background_rate, 1e-9)
            hit_str = (", ".join(v.last_batch_hits)) if v.last_batch_hits else "none"
            miss_count = len(v.last_batch_misses)
            parts.append(
                f"\n## Last Batch Outcomes  "
                f"({n_batch_hits}/{n_batch_total} hits = {batch_rate:.1%}, "
                f"{enrichment_batch:.1f}× background)\n"
                f"  Hits   : {hit_str}\n"
                f"  Misses : {miss_count} gene(s)"
                + (f" ({', '.join(v.last_batch_misses[:5])}"
                   + (" ..." if miss_count > 5 else "") + ")"
                   if v.last_batch_misses else "")
            )

        # ── Pool sample ─────────────────────────────────────────────
        shown_pool = v.pool_sample[:self._max_pool_shown]
        parts.append(
            f"\n## Available Gene Pool — NOMINATE ONLY FROM THIS LIST\n"
            f"({v.n_pool_total:,} unobserved genes total; {len(shown_pool)} shown)\n"
            + ", ".join(shown_pool)
        )

        # ── Task instructions ───────────────────────────────────────
        parts.append(
            f"\n## Your Task\n"
            f"Nominate exactly {self.nominations_requested} genes to query next.\n"
            "IMPORTANT: You MUST choose genes from the pool list above.\n"
            "Do NOT invent gene names from general biology knowledge — only genes\n"
            "that appear in the pool list above exist in this screen.\n\n"
            "Call: nominate_genes(genes=[\"GENE1\", \"GENE2\", ...], reasoning=\"...\")\n\n"
            "Strategy:\n"
            "  1. Confirmed hits above → nominate other genes from the pool that\n"
            "     are in the same pathway or are known interaction partners\n"
            "  2. Avoid gene families that consistently produced misses\n"
            "  3. Pick genes with mechanistic relevance to this phenotype\n"
        )

        # ── Error passthrough ───────────────────────────────────────
        err = self.obs.error
        if err and err.last_action_error:
            parts.append(f"\n## Error from previous action\n{err.last_action_error}")

        return "\n# Specialist Observation\n" + "\n".join(parts) + "\n"

    def to_user_message(self) -> MessageBuilder:
        msg = MessageBuilder.user()
        text = self.prompt
        if text.strip():
            msg.add_text(text)
        return msg


# ── Orchestrator discussion prompt ────────────────────────────────────────────


class OrchestratorDiscussionPrompt(Shrinkable):
    """Prompt shown to the orchestrator after all specialists have nominated.

    Presents a written discussion transcript: each specialist's nominations
    and reasoning, overlap analysis, and budget.  The orchestrator's task is
    to resolve conflicts and emit select_genes([...]).

    Shrinking removes miss details from specialist reports.
    """

    def __init__(
        self,
        obs: AgentObservation,
        # nominations: {target_name: {"genes": [...], "reasoning": "..."}}
        nominations: dict[str, dict],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.obs = obs
        self.nominations = nominations
        self.batch_size = batch_size
        self._verbose = True  # set False by shrink()

    def shrink(self) -> None:
        self._verbose = False

    @property
    def _prompt(self) -> str:
        parts: list[str] = []
        exp = self.obs.experiment
        tm = self.obs.target_metrics

        # ── Header ─────────────────────────────────────────────────
        cycle_str = f"Cycle {exp.cycle}/{exp.n_cycles}" if exp else "Cycle ?"
        parts.append(f"## DISCUSSION TRANSCRIPT — {cycle_str}")
        if exp:
            parts.append(
                f"Budget remaining: {exp.budget_remaining} queries | "
                f"This batch: {self.batch_size} genes"
            )

        # ── Overall recall summary ──────────────────────────────────
        if tm and tm.target_names:
            recall_lines = [
                f"  {name}: recall={_fmt_float(r)}"
                for name, r in zip(tm.target_names, tm.per_target_recall)
            ]
            parts.append("\n### Current Recall\n" + "\n".join(recall_lines))

        # ── Specialist reports ──────────────────────────────────────
        parts.append("\n### Specialist Nominations")
        all_nominated: list[str] = []

        for target_name, nom in self.nominations.items():
            genes = nom.get("genes", [])
            reasoning = nom.get("reasoning", "(no reasoning provided)")
            all_nominated.extend(genes)

            # Recall for this target (for budget allocation hints)
            recall_k = 0.0
            if tm and target_name in tm.target_names:
                k = tm.target_names.index(target_name)
                recall_k = tm.per_target_recall[k] if k < len(tm.per_target_recall) else 0.0

            gene_str = ", ".join(genes) if genes else "(none)"
            parts.append(
                f"\n#### {target_name}  [recall={_fmt_float(recall_k)}]\n"
                f"Nominated {len(genes)} genes: {gene_str}\n"
                f"Reasoning: {reasoning[:400 if self._verbose else 150]}"
                + ("..." if len(reasoning) > (400 if self._verbose else 150) else "")
            )

        # ── Overlap analysis ────────────────────────────────────────
        from collections import Counter
        counts = Counter(all_nominated)
        shared = [g for g, c in counts.items() if c > 1]
        unique_total = len(set(all_nominated))
        parts.append(
            f"\n### Nomination Summary\n"
            f"Total unique nominations: {unique_total}\n"
            f"Shared across targets: {', '.join(shared) if shared else 'none'}"
        )

        # ── Moderator instructions ──────────────────────────────────
        parts.append(
            f"\n## Your Task as Moderator\n"
            f"Select exactly {self.batch_size} genes for the next query batch.\n\n"
            "IMPORTANT: You MUST only select genes that were nominated by the "
            "specialists above. Do NOT add genes from general biology knowledge.\n\n"
            "Guidelines:\n"
            "  1. Prefer genes nominated by MULTIPLE specialists (high cross-target value)\n"
            "  2. Allocate budget proportionally — targets with lower recall deserve more slots\n"
            "  3. Ensure diversity: avoid filling all slots with one gene family\n"
            "  4. If nominations total fewer than the batch size, repeat the strongest ones\n\n"
            "Respond ONLY with: select_genes(genes=[\"GENE1\", \"GENE2\", ...])"
        )

        return "\n# Orchestrator Discussion\n" + "\n".join(parts) + "\n"

    def to_user_message(self) -> MessageBuilder:
        msg = MessageBuilder.user()
        text = self.prompt
        if text.strip():
            msg.add_text(text)
        return msg
