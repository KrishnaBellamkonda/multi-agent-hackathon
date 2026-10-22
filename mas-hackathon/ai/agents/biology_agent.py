"""Biology gene-perturbation agent.

Subclasses CUAAgent with defaults suited for experiment observations:
- uses ExperimentObservationPrompt instead of the web/OS ObservationPrompt
- pre-configured ObsFlags that expose experiment state and candidate pool
- system prompt that instructs the LLM to select genes from the pool
"""

from __future__ import annotations

from acua.agents.cua_agent import CUAAgent
from acua.envs.action_sets import ActionSet
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    ActionFlags,
    ExperimentObservationPrompt,
    ObsFlags,
)
from acua.providers.provider import AbstractProvider

_BIOLOGY_SYSTEM_PROMPT = """\
You are a computational biology agent conducting a multi-target gene \
perturbation experiment. Your goal is to maximise the number of "hit" genes \
discovered across all targets within a fixed query budget.

At each cycle you receive:
  - A candidate pool table (genes ranked by max probability-of-hit across targets)
  - Per-target recall and hit counts so far
  - Budget and cycle status
  - Outcomes of the previous batch (if any)

Your task each cycle: call select_genes(genes=[...]) with exactly the number \
of genes specified as the batch size. Choose genes that maximise expected hits.

Strategy guidelines:
  1. Prefer genes with high MaxPOH — they are most likely to be hits.
  2. If a target has low recall, weight towards genes with high POH for that target.
  3. Avoid re-selecting already-observed genes (they are not in the candidate pool).
  4. Use the cross-target POH columns to balance exploration across targets.
  5. Gene names must match EXACTLY as shown in the Gene column of the table.
"""


class BiologyAgent(CUAAgent):
    """CUAAgent variant pre-configured for biology gene-selection experiments."""

    def __init__(
        self,
        provider: AbstractProvider,
        action_subsets: ActionSet = ActionSet.BIOLOGY,
        system_prompt: str | None = None,
        goal_txt: str | None = None,
        task_hint: str | None = None,
        keep_last_n_obs: int | None = None,
        tool_choice: str = "any",
        obs_flags: ObsFlags | None = None,
        extra_instructions: str | None = None,
        show_correlation: bool = False,
    ):
        default_obs_flags = ObsFlags(
            # disable web/OS fields
            use_html=False,
            use_ax_tree=False,
            use_tabs=False,
            use_focused_element=False,
            use_screenshot=False,
            use_som=False,
            # enable biology fields
            use_error_logs=True,
            use_experiment_state=True,
            use_candidate_pool=True,
            max_candidates=50,
            show_correlation=show_correlation,
            show_last_batch=True,
        )

        super().__init__(
            provider=provider,
            action_subsets=action_subsets,
            system_prompt=system_prompt or _BIOLOGY_SYSTEM_PROMPT,
            goal_txt=goal_txt,
            task_hint=task_hint,
            keep_last_n_obs=keep_last_n_obs,
            multiaction=False,
            tool_choice=tool_choice,
            obs_flags=obs_flags or default_obs_flags,
            action_flags=ActionFlags(
                action_set=action_subsets,
                multi_actions=False,
            ),
            be_cautious=False,
            extra_instructions=extra_instructions,
        )

    # ── Override observation rendering ────────────────────────────────

    def _append_observation(self, obs: Observation) -> None:
        """Use ExperimentObservationPrompt instead of the web ObservationPrompt."""
        prompt = ExperimentObservationPrompt(obs, self.obs_flags)
        self.discussion.append(prompt.to_user_message())
