"""Tool-calling agent that uses StructuredDiscussion to manage conversation history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from acua.envs.action_sets import ActionSet, to_tool_dict
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    ActionFlags,
    ActionPrompt,
    BeCautious,
    GoalInstructions,
    ObservationPrompt,
    ObsFlags,
    SystemPrompt,
    Think,
)
from acua.prompts.parsing import ParseError
from acua.providers.provider import AbstractProvider
from acua.providers.util import APIPayload, LLMOutput, MessageBuilder, PythonAction

from acua.agents.agent import AbstractAgent

logger = logging.getLogger(__name__)

# ── StructuredDiscussion ─────────────────────────────────────────────


@dataclass
class _MessageGroup:
    """A logical group of messages within a discussion (e.g. goal setup, one observation turn)."""

    label: str
    messages: list[MessageBuilder] = field(default_factory=list)


class StructuredDiscussion:
    """Manages a structured conversation as a sequence of message groups.

    Groups partition the conversation into logical sections (goal, observation
    turns, summaries).  When flattening, older observation groups can be
    compressed to keep prompt size bounded.

    Typical lifecycle per step:
        1. (first step only) new_group("goal") → append system/user msgs for goal
        2. new_group() → append observation messages
        3. flatten() → produce the message list sent to the provider
        4. new_group() → start next turn
    """

    def __init__(self, keep_last_n_obs: int | None = None):
        self._groups: list[_MessageGroup] = []
        self._keep_last_n_obs = keep_last_n_obs
        self._goal_set = False
        self._last_summary: MessageBuilder | None = None

    # ── Group management ──────────────────────────────────────────

    def new_group(self, label: str = "") -> None:
        """Start a new message group. Messages appended after this go into the new group."""
        self._groups.append(_MessageGroup(label=label))
        if label == "goal":
            self._goal_set = True

    def append(self, msg: MessageBuilder) -> None:
        """Append a message to the current (last) group."""
        if not self._groups:
            self.new_group()
        self._groups[-1].messages.append(msg)

    def is_goal_set(self) -> bool:
        return self._goal_set

    # ── Summary support ───────────────────────────────────────────

    def set_summary(self, summary: MessageBuilder) -> None:
        """Store a summary message (e.g. produced by a summarizer LLM)."""
        self._last_summary = summary

    def get_last_summary(self) -> MessageBuilder | None:
        return self._last_summary

    # ── Flattening ────────────────────────────────────────────────

    def get_groups_to_drop(self) -> list[_MessageGroup]:
        """Return observation groups that will be dropped on the next flatten().

        These are the groups that would benefit from being summarized before
        they are discarded.
        """
        if self._keep_last_n_obs is None:
            return []

        obs_groups = [g for g in self._groups if g.label != "goal"]
        if len(obs_groups) <= self._keep_last_n_obs:
            return []

        return obs_groups[: len(obs_groups) - self._keep_last_n_obs]

    def flatten(self) -> list[MessageBuilder]:
        """Flatten all groups into a single message list for the provider.

        If keep_last_n_obs is set, only the goal group and the last N
        observation groups are included.  A summary message (if present)
        is inserted between the goal and the retained observation groups.
        """
        if not self._groups:
            return []

        # Separate goal group(s) from observation groups
        goal_msgs: list[MessageBuilder] = []
        obs_groups: list[_MessageGroup] = []

        for group in self._groups:
            if group.label == "goal":
                goal_msgs.extend(group.messages)
            else:
                obs_groups.append(group)

        # Apply observation windowing
        if (
            self._keep_last_n_obs is not None
            and len(obs_groups) > self._keep_last_n_obs
        ):
            obs_groups = obs_groups[-self._keep_last_n_obs :]

        # Build final message list
        messages = list(goal_msgs)

        if self._last_summary is not None:
            messages.append(self._last_summary)

        for group in obs_groups:
            messages.extend(group.messages)

        return messages

    # ── Utilities ─────────────────────────────────────────────────

    @property
    def num_groups(self) -> int:
        return len(self._groups)

    def clear(self) -> None:
        """Reset the discussion state."""
        self._groups.clear()
        self._goal_set = False
        self._last_summary = None


# ── CUAAgent ─────────────────────────────────────────────────────────


class CUAAgent(AbstractAgent):
    """Provider- and environment-agnostic agent that uses tool calling.

    Uses StructuredDiscussion to manage conversation history,
    the dynamic prompting framework for prompt construction,
    and AbstractProvider + APIPayload/LLMOutput for LLM communication.
    """

    def __init__(
        self,
        provider: AbstractProvider,
        action_subsets: ActionSet,
        system_prompt: str | None = None,
        goal_txt: str | None = None,
        goal_img: str | None = None,
        task_hint: str | None = None,
        keep_last_n_obs: int | None = None,
        multiaction: bool = False,
        tool_choice: str = "any",
        obs_flags: ObsFlags | None = None,
        action_flags: ActionFlags | None = None,
        be_cautious: bool = False,
        extra_instructions: str | None = None,
    ):
        self.provider = provider
        self.action_subsets = action_subsets
        self.goal_txt = goal_txt
        self.goal_img = goal_img
        self.task_hint = task_hint
        self.keep_last_n_obs = keep_last_n_obs

        self.multiaction = multiaction
        self.tool_choice = tool_choice
        self.extra_instructions = extra_instructions

        # Prompt framework components
        self.obs_flags = obs_flags or ObsFlags(
            use_html=True,
            use_ax_tree=True,
            use_tabs=True,
            use_focused_element=True,
            use_error_logs=True,
            use_screenshot=True,
        )
        self.action_flags = action_flags or ActionFlags(
            action_set=action_subsets,
            multi_actions=multiaction,
        )

        # Build prompt elements
        if system_prompt is not None:
            self._system_prompt = SystemPrompt()
            self._system_prompt._prompt = system_prompt
        else:
            self._system_prompt = SystemPrompt()

        self._action_prompt = ActionPrompt(action_subsets, self.action_flags)
        self._think_prompt = Think()
        self._be_cautious = BeCautious(visible=be_cautious)

        self.tools = to_tool_dict(action_subsets)
        self.discussion = StructuredDiscussion(keep_last_n_obs)
        self.last_response = LLMOutput()
        self._responses: list[LLMOutput] = []

    # ── Core interface ────────────────────────────────────────────

    def get_python_action(self, obs: Observation) -> str:
        """Given an observation, return a Python action string for the environment."""
        if not self.discussion.is_goal_set():
            self._setup_goal(obs)

        self._append_observation(obs)

        # Summarize observation groups that are about to be dropped
        self._maybe_summarize()

        messages = self.discussion.flatten()
        response: LLMOutput = self.provider(
            APIPayload(
                messages=messages,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )
        )

        action = self._extract_action(response)

        # Echo tool calls back into the discussion so the provider
        # sees its own calls in context on subsequent turns.
        self._append_tool_calls(response, action)

        # Start a new group for the next turn
        self.discussion.new_group()

        self.last_response = response
        self._responses.append(response)

        return action

    # ── Goal setup ────────────────────────────────────────────────

    def _setup_goal(self, obs: Observation) -> None:
        """Build the initial goal group using prompt framework elements."""
        self.discussion.new_group("goal")

        # System message: base prompt + action space + think instructions + cautious note
        sys_msg = MessageBuilder.system()

        sys_text = self._system_prompt.prompt
        if self.multiaction:
            sys_text += "\nYou can take multiple actions in a single step, if needed."
        else:
            sys_text += "\nYou can only take one action at a time."
        sys_msg.add_text(sys_text)

        # Action space description
        action_text = self._action_prompt.prompt
        if action_text:
            sys_msg.add_text(action_text)

        # Think and example instructions
        if self._think_prompt.abstract_ex:
            sys_msg.add_text(
                "\nYour response should use the following format:\n"
                f"{self._think_prompt.abstract_ex}"
                f"{self._action_prompt.abstract_ex}"
            )

        # Be cautious note
        cautious_text = self._be_cautious.prompt
        if cautious_text:
            sys_msg.add_text(cautious_text)

        self.discussion.append(sys_msg)

        # Goal from observation or constructor args
        if obs.goal is not None or self.goal_txt or self.goal_img:
            goal_instructions = GoalInstructions(
                goal=obs.goal if obs.goal is not None else self.goal_txt,
                extra_instructions=self.extra_instructions,
            )
            self.discussion.append(goal_instructions.to_user_message())

            # Append goal image separately if provided via constructor
            if self.goal_img and obs.goal is None:
                self.discussion.append(MessageBuilder.user().add_image(self.goal_img))

        # Task hint
        if self.task_hint:
            self.discussion.append(
                MessageBuilder.user().add_text(f"[Hint] {self.task_hint}")
            )

        # Start first observation group
        self.discussion.new_group()

    # ── Observation handling ──────────────────────────────────────

    def _append_observation(self, obs: Observation) -> None:
        """Convert an Observation into messages using ObservationPrompt."""
        obs_prompt = ObservationPrompt(obs, self.obs_flags)
        self.discussion.append(obs_prompt.to_user_message())

    # ── Summarization ─────────────────────────────────────────────

    _SUMMARIZE_PROMPT = (
        "You are a concise summarizer. Below is a conversation history of actions "
        "taken so far during a computer-use task. Summarize what has been done, "
        "what state the task is in, and any important observations. "
        "Be brief — this summary will be used as context for future steps."
    )

    def _maybe_summarize(self) -> None:
        """If observation groups are about to be dropped, summarize them first.

        The summary is folded into the existing summary (if any) so that
        context accumulates across multiple summarization rounds.
        """
        groups_to_drop = self.discussion.get_groups_to_drop()
        if not groups_to_drop:
            return

        # Collect text content from groups that will be dropped
        text_parts: list[str] = []

        # Include the previous summary so it gets rolled forward
        prev_summary = self.discussion.get_last_summary()
        if prev_summary is not None:
            for item in prev_summary.content:
                if "text" in item:
                    text_parts.append(item["text"])

        for group in groups_to_drop:
            for msg in group.messages:
                for item in msg.content:
                    if "text" in item:
                        text_parts.append(f"[{msg.role}] {item['text']}")

        if not text_parts:
            return

        # Ask the provider to summarize
        messages = [
            MessageBuilder.system().add_text(self._SUMMARIZE_PROMPT),
            MessageBuilder.user().add_text("\n\n".join(text_parts)),
        ]
        response: LLMOutput = self.provider(APIPayload(messages=messages))

        summary_text = response.think or response.action or ""
        if summary_text:
            self.discussion.set_summary(
                MessageBuilder.user().add_text(
                    f"[Summary of previous steps]\n{summary_text}"
                )
            )

    # ── Tool call context management ─────────────────────────────

    def _append_tool_calls(self, response: LLMOutput, action: str) -> None:
        """Echo the LLM's tool calls and their responses back into the discussion.

        Provider APIs (e.g. OpenAI) expect the assistant's tool calls to appear
        in the message history followed by tool-response messages before the next
        user turn. Without this, the provider loses context of what it called.
        """
        if not response.tool_calls:
            return

        # Set a tool response on each call so the provider sees the result.
        # Since the agent doesn't execute actions itself (the env does),
        # we acknowledge each call with the python action string produced.
        for tc in response.tool_calls:
            if not tc.is_response_set:
                python_action = PythonAction.convert_toolcall_to_python_action_format(
                    tc
                )
                tc.response_text(f"Action dispatched: {python_action}")

        # Append the tool-response message (contains the raw_response echo
        # and individual call responses paired by call_id).
        self.discussion.append(
            MessageBuilder.add_responded_tool_calls(response.tool_calls)
        )

    # ── Action extraction ─────────────────────────────────────────

    def _extract_action(self, response: LLMOutput) -> str:
        """Extract a Python action string from the LLM response.

        Tries in order:
        1. Pre-parsed action from the provider (response.action)
        2. Tool calls converted to Python function call strings
        3. Text-based parsing of <action> tags from the raw response
        4. Falls back to noop()
        """
        # 1. Pre-parsed action from the provider
        if response.action is not None:
            return response.action

        # 2. Tool calls
        if response.tool_calls:
            actions = [
                PythonAction.convert_toolcall_to_python_action_format(tc)
                for tc in response.tool_calls
            ]
            if len(actions) == 1:
                return actions[0]
            combined = PythonAction.convert_multiactions_to_python_action_format(
                actions
            )
            return combined if combined is not None else "noop()"

        # 3. Text-based <action> tag parsing (for providers that return plain text)
        raw = getattr(response.raw_response, "output_text", None) or response.think
        if raw:
            try:
                parsed = self._action_prompt.parse_answer(raw)
                action = parsed.get("action")
                if action is not None:
                    return action
            except ParseError:
                logger.debug("Could not parse <action> tag from text response")

            # Also extract <think> if present
            think_parsed = self._think_prompt.parse_answer(raw)
            if "think" in think_parsed and not think_parsed.get("parse_error"):
                response.think = think_parsed["think"]

        return "noop()"

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.discussion.clear()
        self.last_response = LLMOutput()
        self._responses.clear()
