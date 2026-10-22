"""Single-agent biology ablation.

BiologySingleAgent is the ablation counterpart of BiologyOrchestratorAgent.
The multi-agent protocol has K specialists that reason independently per target,
followed by a moderator LLM that resolves nominations into a final selection.

This agent collapses that protocol into a single LLM call:
  - One agent sees ALL K targets' hit history simultaneously.
  - It reasons over all targets and calls select_genes() directly.
  - No nomination round, no discussion, no moderator.

This ablation answers the question: does specialist decomposition help,
or does a single informed agent achieve the same result?

Both agents use use_gp=False BiologyEnv (no surrogate model).
"""

from __future__ import annotations

from acua.agents.biology_specialist import SCREEN_BIOLOGY
from acua.agents.cua_agent import CUAAgent
from acua.envs.action_sets import ActionSet
from acua.envs.hints import SpecialistViewHints
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    ActionFlags,
    ObsFlags,
    SingleAgentObservationPrompt,
)
from acua.providers.provider import AbstractProvider

try:
    from mygene_prompts.integration import enriched_phenotype_context
    from acua.agents.biology_specialist import SCREEN_ANCHOR_GENES
    _MYGENE_AVAILABLE = True
except ImportError:
    _MYGENE_AVAILABLE = False
    SCREEN_ANCHOR_GENES = {}


# ── System prompt ─────────────────────────────────────────────────────────────

_SINGLE_AGENT_SYSTEM_BASE = """\
You are a computational biology expert conducting a multi-target CRISPR \
perturbation experiment. Your goal is to discover as many hit genes as \
possible across ALL active phenotypes within a fixed query budget.

Active phenotypes in this experiment:
{phenotype_block}

Your role:
  - Each cycle you receive the current progress for ALL targets: confirmed
    hits, last-batch outcomes, and a pool sample of unobserved genes.
  - You must select exactly the requested number of genes for the next batch.
  - Querying a gene reveals its phenotypic effect for ALL targets at once.

Gene selection strategy:
  1. Cross-target hits: genes that are likely hits for MULTIPLE phenotypes
     are especially valuable — one query yields information for all targets.
  2. Balance allocation: if one target has substantially lower recall than
     others, devote proportionally more slots to improving it.
  3. Follow confirmed hits: nominate pathway neighbours, binding partners,
     and regulators of genes already confirmed as hits for any target.
  4. Mechanistic reasoning: use your knowledge of the biology above to
     prioritise genes with documented roles in the relevant pathways.
  5. Diversity: avoid selecting all genes from the same gene family.
  6. Use HUGO gene symbols exactly (JAK1 not jak1 or Jak1).

You are acting alone — reason carefully across all targets in a single step.
"""


def _build_system_prompt(target_names: list[str]) -> str:
    """Build system prompt with phenotype block for given target names."""
    lines: list[str] = []
    for name in target_names:
        bio = SCREEN_BIOLOGY.get(name)
        if bio:
            lines.append(
                f"  • {name}\n"
                f"    Phenotype : {bio['phenotype']}\n"
                f"    Cell type : {bio['cell_type']}\n"
                f"    Context   : {bio['context']}"
            )
        else:
            lines.append(f"  • {name}  (gene-perturbation screen)")
    phenotype_block = "\n\n".join(lines) if lines else "  (no biology context available)"
    prompt = _SINGLE_AGENT_SYSTEM_BASE.format(phenotype_block=phenotype_block)

    if _MYGENE_AVAILABLE:
        anchors = {
            n: SCREEN_ANCHOR_GENES[n]
            for n in target_names
            if n in SCREEN_ANCHOR_GENES
        }
        if anchors:
            try:
                addendum = enriched_phenotype_context(
                    screen_names=list(anchors),
                    base_context="",
                    cache=None,
                    exemplar_genes_per_screen=anchors,
                )
                if addendum.strip():
                    prompt = (
                        f"{prompt}\n\nKNOWN-BIOLOGY ANCHOR (MyGene.info "
                        f"summary of canonical regulators per screen — NOT "
                        f"this dataset's labels):\n{addendum}"
                    )
            except Exception:
                pass

    return prompt


# ── Agent class ───────────────────────────────────────────────────────────────


class BiologySingleAgent(CUAAgent):
    """Single LLM agent that selects genes for all targets simultaneously.

    Ablation of BiologyOrchestratorAgent: same environment, same information,
    but no specialist decomposition and no discussion protocol.

    Usage
    -----
    agent = BiologySingleAgent(
        target_names=["schmidt_2021_ifng", "schmidt_2021_il2"],
        provider=provider,
        batch_size=20,
    )
    obs = env.reset()
    while True:
        action = agent.get_python_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    """

    def __init__(
        self,
        target_names: list[str],
        provider: AbstractProvider,
        batch_size: int,
        keep_last_n_obs: int | None = 5,
    ):
        """
        Parameters
        ----------
        target_names  : list of screen names for which to enrich the system prompt
        provider      : LLM provider
        batch_size    : genes to select each cycle (used in observation prompt)
        keep_last_n_obs: conversation history window
        """
        system_prompt = _build_system_prompt(target_names)

        obs_flags = ObsFlags(
            use_html=False,
            use_ax_tree=False,
            use_screenshot=False,
            use_error_logs=True,
        )

        super().__init__(
            provider=provider,
            action_subsets=ActionSet.BIOLOGY,
            system_prompt=system_prompt,
            keep_last_n_obs=keep_last_n_obs,
            multiaction=False,
            tool_choice="any",
            obs_flags=obs_flags,
            action_flags=ActionFlags(
                action_set=ActionSet.BIOLOGY,
                multi_actions=False,
            ),
            be_cautious=False,
        )

        self.target_names = target_names
        self.batch_size = batch_size
        # Inject phenotype/cell_type context from SCREEN_BIOLOGY
        self._bio_contexts: dict[str, dict] = {
            name: SCREEN_BIOLOGY.get(name, {"phenotype": name, "cell_type": ""})
            for name in target_names
        }

    def _append_observation(self, obs: Observation) -> None:
        """Render a SingleAgentObservationPrompt covering all K targets.

        The orchestrator populates obs.specialist_views with one view per target.
        We enrich each view with phenotype/cell_type from SCREEN_BIOLOGY (the env
        leaves these blank to stay dataset-agnostic), then render a unified prompt.
        """
        # Enrich specialist views with biology context
        if obs.specialist_views:
            for view in obs.specialist_views:
                bio = self._bio_contexts.get(view.target_name, {})
                if not view.phenotype:
                    view.phenotype = bio.get("phenotype", view.target_name)
                if not view.cell_type:
                    view.cell_type = bio.get("cell_type", "")
        else:
            # Build minimal placeholder views if env didn't populate them
            from dataclasses import replace as dc_replace
            views = [
                SpecialistViewHints(
                    target_name=name,
                    phenotype=self._bio_contexts[name].get("phenotype", name),
                    cell_type=self._bio_contexts[name].get("cell_type", ""),
                )
                for name in self.target_names
            ]
            # Attach to obs (mutating is safe — this is a local copy from orchestrator)
            obs = Observation(
                experiment=obs.experiment,
                target_metrics=obs.target_metrics,
                error=obs.error,
                step_info=obs.step_info,
                specialist_views=views,
            )

        prompt = SingleAgentObservationPrompt(obs=obs, batch_size=self.batch_size)
        self.discussion.append(prompt.to_user_message())
