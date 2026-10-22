"""Specialist biology agent — one per phenotype/target.

Each BiologySpecialistAgent is an LLM agent scoped to a single CRISPR screen
phenotype.  It receives a SpecialistObservationPrompt showing only its target's
outcomes and a pool sample, then calls nominate_genes(genes=[...], reasoning="...")
to propose candidates for the next batch.

The specialist deliberately has NO cross-target visibility — it reasons purely
from its phenotype biology and its own hit/miss history.  The orchestrator
(BiologyOrchestratorAgent) aggregates nominations across all specialists and
decides the final gene selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from acua.agents.cua_agent import CUAAgent, StructuredDiscussion
from acua.envs.action_sets import ActionSet
from acua.envs.hints import SpecialistViewHints
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    ActionFlags,
    ObsFlags,
    SpecialistObservationPrompt,
)
from acua.providers.provider import AbstractProvider
from acua.providers.util import APIPayload, MessageBuilder

try:
    from mygene_prompts.integration import enriched_phenotype_context
    _MYGENE_AVAILABLE = True
except ImportError:
    _MYGENE_AVAILABLE = False

# Literature-canonical anchor genes per screen — used to seed
# mygene_prompts.phenotype_summary so it can list recurrent pathways and
# regulators without leaking the dataset's own hit labels.
SCREEN_ANCHOR_GENES: dict[str, list[str]] = {
    "schmidt_2021_ifng":  ["JAK1", "JAK2", "STAT1", "IRF1", "IFNGR1"],
    "schmidt_2021_il2":   ["IL2RA", "IL2RB", "JAK3", "STAT5A", "FOXP3"],
    "zhuang_2019_nk":     ["PRF1", "GZMB", "NKG7", "KLRK1", "NCR1"],
    "sanchez_2021_tau":   ["MAPT", "APP", "PSEN1", "GSK3B", "BIN1"],
    "zhu_2021_sarscov2":  ["ACE2", "TMPRSS2", "CTSL", "FURIN", "BSG"],
}

# Biology context for each GeneDisco screen — mirrors maehd/agents/biology_agent.py
# so the specialist prompt is grounded in the real experimental biology.
SCREEN_BIOLOGY: dict[str, dict[str, str]] = {
    "schmidt_2021_ifng": {
        "phenotype": "IFNγ (interferon-gamma) cytokine production",
        "cell_type": "primary human CD4+ T cells",
        "context": (
            "Genome-wide CRISPR knockout screen in primary human CD4+ T cells. "
            "Cells were activated with anti-CD3/CD28 beads and IFNγ secretion "
            "was measured by intracellular staining + FACS. Hit genes are those "
            "whose knockout significantly changes IFNγ production."
        ),
    },
    "schmidt_2021_il2": {
        "phenotype": "IL-2 (interleukin-2) cytokine production",
        "cell_type": "primary human CD4+ T cells",
        "context": (
            "Genome-wide CRISPR knockout screen in primary human CD4+ T cells. "
            "Cells were activated with anti-CD3/CD28 beads and IL-2 secretion "
            "was measured. Hit genes are those whose knockout significantly "
            "changes IL-2 production."
        ),
    },
    "zhuang_2019_nk": {
        "phenotype": "NK cell-mediated cytotoxicity against tumour cells",
        "cell_type": "human natural killer (NK) cells",
        "context": (
            "CRISPR screen in NK cells co-cultured with tumour cell lines. "
            "The readout is tumour cell killing efficiency. Hit genes affect "
            "NK cell cytotoxic function or tumour cell susceptibility to killing."
        ),
    },
    "sanchez_2021_tau": {
        "phenotype": "Tau protein aggregation and levels",
        "cell_type": "human iPSC-derived cortical neurons",
        "context": (
            "CRISPR screen in iPSC-derived neurons measuring Tau protein levels. "
            "Relevant to Alzheimer's disease and tauopathies. Hit genes change "
            "Tau protein expression, phosphorylation, or aggregation."
        ),
    },
    "zhu_2021_sarscov2": {
        "phenotype": "SARS-CoV-2 viral entry and replication",
        "cell_type": "human lung epithelial cells (A549-ACE2)",
        "context": (
            "CRISPR screen for host factors required for SARS-CoV-2 infection. "
            "A549 cells expressing ACE2 were infected and surviving cells were "
            "sequenced. Hit genes are host factors essential for viral entry, "
            "replication, or virus-induced cell death."
        ),
    },
}

# Generic fallback for synthetic datasets or unknown screens
_GENERIC_BIOLOGY = {
    "phenotype": "gene-perturbation phenotype",
    "cell_type": "human cell line",
    "context": "Genome-wide CRISPR perturbation screen measuring phenotypic effect.",
}

# ── System prompt template ────────────────────────────────────────────────────
# Filled in per-specialist with phenotype/cell-type/context from SCREEN_BIOLOGY.

_SPECIALIST_SYSTEM = """\
You are a computational biology expert specialising in the following CRISPR screen:

PHENOTYPE : {phenotype}
CELL TYPE : {cell_type}
CONTEXT   : {context}

Your role in a multi-agent experiment:
  - Each cycle you receive a summary of YOUR target's progress (hits found, misses).
  - You nominate exactly the requested number of genes for the next query batch.
  - An orchestrator collects nominations from all target-specialists and decides
    the final selection — you do NOT interact with other specialists directly.

Nomination strategy:
  1. Follow confirmed hits: nominate pathway neighbours, binding partners, upstream
     and downstream regulators of genes already confirmed as hits.
  2. Prioritise mechanistic logic: kinases, transcription factors, and signalling
     adaptors directly relevant to this phenotype are higher-value than distant genes.
  3. Diversify: do not cluster all nominations in one gene family — spread across
     related-but-distinct pathways to maximise information per batch.
  4. Learn from misses: if a gene family consistently produced misses, deprioritise
     further members of that family.
  5. Use HUGO gene symbols exactly (e.g. JAK1, STAT3, not jak1 or Jak1).

You have deep knowledge of the biology above — use it confidently.
"""


@dataclass
class SpecialistConfig:
    """Specification for one specialist agent.

    Attributes
    ----------
    target_name : screen name (must match a MULTI_TARGET_SETUPS key or a
                  target_name used in BiologyEnvConfig), e.g. "schmidt_2021_ifng"
    provider    : LLM provider for this specialist (may be shared across specialists
                  to reduce API costs, or each specialist can use a separate provider)
    nominations : how many genes this specialist nominates per cycle
    """
    target_name: str
    provider: AbstractProvider
    nominations: int = 5


class BiologySpecialistAgent(CUAAgent):
    """Single-phenotype specialist biology agent.

    Uses nominate_genes() instead of select_genes() — it proposes candidates
    with reasoning, and the BiologyOrchestratorAgent decides the final batch.

    The system prompt is constructed from SCREEN_BIOLOGY[target_name] so each
    specialist carries phenotype-specific domain knowledge.  For target names
    not found in SCREEN_BIOLOGY (e.g. synthetic dataset targets), a generic
    prompt is used.
    """

    def __init__(
        self,
        target_name: str,
        provider: AbstractProvider,
        nominations: int = 5,
        keep_last_n_obs: int | None = 5,   # retain last 5 cycles for context
    ):
        bio = SCREEN_BIOLOGY.get(target_name, _GENERIC_BIOLOGY)

        system_prompt = _SPECIALIST_SYSTEM.format(
            phenotype=bio["phenotype"],
            cell_type=bio["cell_type"],
            context=bio["context"],
        )

        if _MYGENE_AVAILABLE and target_name in SCREEN_ANCHOR_GENES:
            try:
                addendum = enriched_phenotype_context(
                    screen_names=[target_name],
                    base_context="",
                    cache=None,
                    exemplar_genes_per_screen={
                        target_name: SCREEN_ANCHOR_GENES[target_name]
                    },
                )
                if addendum.strip():
                    system_prompt = (
                        f"{system_prompt}\n\nKNOWN-BIOLOGY ANCHOR (MyGene.info "
                        f"summary of canonical regulators, NOT this dataset's "
                        f"labels):\n{addendum}"
                    )
            except Exception as _e:
                # Network or schema failure — fall back to plain prompt silently.
                pass

        # Specialists use BIOLOGY_NOMINATE: only nominate_genes + finish.
        # They never call select_genes — that belongs to the orchestrator.
        obs_flags = ObsFlags(
            use_html=False,
            use_ax_tree=False,
            use_screenshot=False,
            use_error_logs=True,
            # Biology fields are not used here — the specialist receives a
            # SpecialistObservationPrompt built directly by the orchestrator.
        )

        super().__init__(
            provider=provider,
            action_subsets=ActionSet.BIOLOGY_NOMINATE,
            system_prompt=system_prompt,
            keep_last_n_obs=keep_last_n_obs,
            multiaction=False,
            tool_choice="any",
            obs_flags=obs_flags,
            action_flags=ActionFlags(
                action_set=ActionSet.BIOLOGY_NOMINATE,
                multi_actions=False,
            ),
            be_cautious=False,
        )

        self.target_name = target_name
        self.nominations = nominations
        # Enrich the specialist_view with phenotype/cell_type for prompt rendering
        self._bio_context = bio

    def _append_observation(self, obs: Observation) -> None:
        """Render a SpecialistObservationPrompt for the single target this agent owns.

        The orchestrator pre-selects the correct SpecialistViewHints and stores it
        in obs.specialist_views[0] before calling get_python_action().
        """
        # The orchestrator always populates specialist_views[0] for this agent.
        if obs.specialist_views and len(obs.specialist_views) > 0:
            view = obs.specialist_views[0]
            # Inject phenotype/cell_type from SCREEN_BIOLOGY (env leaves these blank)
            view.phenotype = self._bio_context.get("phenotype", view.target_name)
            view.cell_type = self._bio_context.get("cell_type", "")
        else:
            # Fallback: build a minimal view so the agent always has something to see
            view = SpecialistViewHints(
                target_name=self.target_name,
                phenotype=self._bio_context.get("phenotype", self.target_name),
                cell_type=self._bio_context.get("cell_type", ""),
            )

        prompt = SpecialistObservationPrompt(
            obs=obs,
            specialist_view=view,
            nominations_requested=self.nominations,
        )
        self.discussion.append(prompt.to_user_message())
