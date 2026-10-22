"""Orchestrator agent that delegates to sub-agent providers via tool calls.

The OrchestratorAgent has its own provider for decision-making. Its only tools
are ``call_{name}`` functions — one per sub-agent — whose descriptions guide the
orchestrator on when to use each specialist. When a tool is called, the
orchestrator dispatches the observation to the chosen sub-agent, which produces
the final environment action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from acua.agents.agent import AbstractAgent
from acua.agents.cua_agent import CUAAgent, StructuredDiscussion
from acua.envs.action_sets import ActionSet
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    BeCautious,
    GoalInstructions,
    ObservationPrompt,
    ObsFlags,
    Think,
)
from acua.providers.provider import AbstractProvider
from acua.providers.util import (
    APIPayload,
    LLMOutput,
    MessageBuilder,
)

logger = logging.getLogger(__name__)


# ── Sub-agent spec ───────────────────────────────────────────────────


def _model_name_to_tool_name(model_name: str) -> str:
    """Derive a tool-call-safe name from a provider's model_name.

    e.g. ``"Qwen/Qwen2.5-VL-32B-Instruct"`` → ``"call_qwen2_5_vl_32b_instruct"``
    """
    # Take the part after the last slash (or the whole string)
    short = model_name.rsplit("/", 1)[-1]
    # Replace non-alphanumeric chars with underscores, collapse runs, lowercase
    safe = ""
    for ch in short:
        safe += ch if ch.isalnum() else "_"
    safe = "_".join(part for part in safe.split("_") if part)
    return f"call_{safe.lower()}"


@dataclass
class SubAgentSpec:
    """Specification for a sub-agent available to the orchestrator.

    The sub-agent is uniquely identified by its provider's ``model_name``.
    Only one sub-agent per model is allowed.

    Attributes:
        provider: The LLM provider this sub-agent uses.  Its ``model_name``
            is used to derive the tool name (``call_{model}``).
        description: Guides the orchestrator on when to call this sub-agent.
            Appears as the tool description in the orchestrator's tool list.
        action_set: The action set the sub-agent operates with.
        obs_flags: Observation flags controlling what the sub-agent sees.
        be_cautious: Whether the sub-agent should use the BeCautious prompt.
        keep_last_n_obs: Observation window for the sub-agent's discussion.
    """

    provider: AbstractProvider
    description: str
    action_set: ActionSet = ActionSet.WEBARENA
    obs_flags: ObsFlags | None = None
    be_cautious: bool = False
    keep_last_n_obs: int | None = None


# ── Orchestrator agent ───────────────────────────────────────────────


class OrchestratorAgent(AbstractAgent):
    """Orchestrator that sees observations and delegates to specialist sub-agents.

    Each sub-agent is identified by its provider's ``model_name`` and exposed as
    a tool (``call_{model}(instruction)``). The orchestrator's own provider
    decides which sub-agent to call and what instruction to give it. The chosen
    sub-agent then produces the final environment action using its own provider.

    Only one sub-agent per provider model is allowed.

    Args:
        provider: The orchestrator's own LLM provider (for decision-making).
        sub_agents: The active sub-agent specifications.
        action_set: Action set for the environment (used by sub-agents).
        obs_flags: Observation flags for the orchestrator's prompt.
        keep_last_n_obs: Observation window for the orchestrator's discussion.
        be_cautious: Whether the orchestrator's prompt includes BeCautious.
        extra_instructions: Extra text appended to the goal instructions.
    """

    def __init__(
        self,
        provider: AbstractProvider,
        sub_agents: list[SubAgentSpec],
        action_set: ActionSet = ActionSet.WEBARENA,
        obs_flags: ObsFlags | None = None,
        keep_last_n_obs: int | None = None,
        be_cautious: bool = False,
        extra_instructions: str | None = None,
    ):
        self.provider = provider
        self.action_subsets = action_set
        self.keep_last_n_obs = keep_last_n_obs
        self.extra_instructions = extra_instructions

        self.obs_flags = obs_flags or ObsFlags(
            use_html=True,
            use_ax_tree=True,
            use_tabs=True,
            use_focused_element=True,
            use_error_logs=True,
            use_screenshot=True,
        )

        if not sub_agents:
            raise ValueError("OrchestratorAgent requires at least one sub-agent")

        # Enforce one sub-agent per provider model
        seen_models: dict[str, int] = {}
        for i, spec in enumerate(sub_agents):
            model = spec.provider.model_name
            if model in seen_models:
                raise ValueError(
                    f"Duplicate provider model_name {model!r}: "
                    f"sub_agents[{seen_models[model]}] and sub_agents[{i}]"
                )
            seen_models[model] = i

        # Build sub-agent CUAAgent instances and tool schemas
        self._sub_agents: dict[str, CUAAgent] = {}
        self.tools: list[dict] = []

        for spec in sub_agents:
            agent = CUAAgent(
                provider=spec.provider,
                action_subsets=spec.action_set or action_set,
                obs_flags=spec.obs_flags or self.obs_flags,
                keep_last_n_obs=spec.keep_last_n_obs,
                be_cautious=spec.be_cautious,
            )
            tool_name = _model_name_to_tool_name(spec.provider.model_name)
            self._sub_agents[tool_name] = agent

            self.tools.append(
                {
                    "type": "function",
                    "name": tool_name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": (
                                    "A precise instruction for this specialist describing "
                                    "what action to take and why, given the current observation."
                                ),
                            },
                        },
                        "required": ["instruction"],
                    },
                }
            )

        # Orchestrator prompt elements
        self._system_prompt_text = (
            "You are an orchestrator agent. You see the current state of the screen and "
            "must decide which specialist agent to delegate to by calling the appropriate tool. "
            "Each tool corresponds to a specialist with different strengths. "
            "Choose the most appropriate specialist and provide a clear, specific instruction "
            "describing what action they should take.\n\n"
            "You can only call one specialist per step."
        )
        self._think_prompt = Think()
        self._be_cautious = BeCautious(visible=be_cautious)

        # Orchestrator's own discussion state
        self.discussion = StructuredDiscussion(keep_last_n_obs)
        self.last_response = LLMOutput()
        self._responses: list[LLMOutput] = []

    # ── Core interface ────────────────────────────────────────────

    def get_python_action(self, obs: Observation) -> str:
        """Observe → orchestrator picks a sub-agent → sub-agent produces action."""
        if not self.discussion.is_goal_set():
            self._setup_goal(obs)

        self._append_observation(obs)

        messages = self.discussion.flatten()
        response: LLMOutput = self.provider(
            APIPayload(
                messages=messages,
                tools=self.tools,
                tool_choice="any",
            )
        )

        # Determine which sub-agent was called
        action = self._dispatch(response, obs)

        # Echo the orchestrator's tool call into the discussion
        self._append_tool_calls(response, action)
        self.discussion.new_group()

        self.last_response = response
        self._responses.append(response)

        return action

    # ── Dispatch to sub-agent ─────────────────────────────────────

    def _dispatch(self, response: LLMOutput, obs: Observation) -> str:
        """Route the orchestrator's tool call to the chosen sub-agent."""
        if not response.tool_calls:
            logger.warning(
                "Orchestrator produced no tool call — falling back to noop()"
            )
            return "noop()"

        # Use the first tool call
        tc = response.tool_calls.tool_calls[0]
        tool_name = tc.name
        instruction = tc.arguments.get("instruction", "")

        agent = self._sub_agents.get(tool_name)
        if agent is None:
            logger.warning(
                "Orchestrator called unknown tool %s — falling back to noop()",
                tool_name,
            )
            return "noop()"

        # Always log sub-agent routing — this is a key orchestrator decision
        # and must be visible even at the default WARNING log level.
        logger.warning(
            "[Orchestrator] → %s  instruction: %s",
            tool_name,
            instruction[:300] if instruction else "(none)",
        )

        # Inject the instruction as extra context for the sub-agent
        agent.extra_instructions = (
            f"{instruction}\n\n"
            "IMPORTANT: Do NOT call send_msg_to_user() unless you are submitting "
            "the final answer to the task. send_msg_to_user() is the STOP action "
            "and will immediately terminate the episode. Use browser actions "
            "(click, fill, scroll, etc.) to interact with the page."
        )

        # Sub-agent produces the environment action
        action = agent.get_python_action(obs)

        # Set the tool response so it can be echoed in the discussion
        tc.response_text(f"Sub-agent action: {action}")

        return action

    # ── Goal setup ────────────────────────────────────────────────

    def _setup_goal(self, obs: Observation) -> None:
        """Build the orchestrator's initial goal group."""
        self.discussion.new_group("goal")

        sys_msg = MessageBuilder.system()
        sys_msg.add_text(self._system_prompt_text)

        # List available specialists
        specialist_lines = []
        for tool in self.tools:
            specialist_lines.append(f"- **{tool['name']}**: {tool['description']}")
        sys_msg.add_text(
            "\n\n## Available specialists:\n" + "\n".join(specialist_lines)
        )

        # Think instructions
        if self._think_prompt.abstract_ex:
            sys_msg.add_text(
                "\nYour response should use the following format:\n"
                f"{self._think_prompt.abstract_ex}"
                "\nThen call exactly one specialist tool.\n"
            )

        cautious_text = self._be_cautious.prompt
        if cautious_text:
            sys_msg.add_text(cautious_text)

        self.discussion.append(sys_msg)

        # Goal from observation
        if obs.goal is not None:
            goal_instructions = GoalInstructions(
                goal=obs.goal,
                extra_instructions=self.extra_instructions,
            )
            self.discussion.append(goal_instructions.to_user_message())

        self.discussion.new_group()

    # ── Observation handling ──────────────────────────────────────

    def _append_observation(self, obs: Observation) -> None:
        obs_prompt = ObservationPrompt(obs, self.obs_flags)
        self.discussion.append(obs_prompt.to_user_message())

    # ── Tool call echo ────────────────────────────────────────────

    def _append_tool_calls(self, response: LLMOutput, action: str) -> None:
        """Echo tool calls into the discussion for context continuity."""
        if not response.tool_calls:
            return

        for tc in response.tool_calls:
            if not tc.is_response_set:
                tc.response_text(f"Action dispatched: {action}")

        self.discussion.append(
            MessageBuilder.add_responded_tool_calls(response.tool_calls)
        )

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset orchestrator and all sub-agent state."""
        self.discussion.clear()
        self.last_response = LLMOutput()
        self._responses.clear()
        for agent in self._sub_agents.values():
            agent.reset()
