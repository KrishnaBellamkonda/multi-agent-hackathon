"""Multi-agent orchestrator for joint gene-perturbation discovery.

Architecture
------------
BiologyOrchestratorAgent coordinates K specialist agents (one per phenotype)
and a moderator LLM.  Per acquisition cycle:

  Phase 1 — Nomination
      Each BiologySpecialistAgent receives a single-target observation and calls
      nominate_genes(genes=[...], reasoning="...").  Specialists see ONLY their
      own target's hit/miss history — no cross-target information.

  Phase 2 — Moderation
      The orchestrator (moderator LLM) receives a discussion transcript with
      all nominations and reasons.  It considers cross-target overlap, recall
      imbalances, and biology logic to produce the final select_genes([...]).

  Phase 3 — Environment step
      The final action is returned to BiologyEnv.step(), which marks the
      selected genes as observed and returns the next shared observation.

Key design decisions
--------------------
- No surrogate model (GP).  All decisions are grounded in biology reasoning
  and the observed hit/miss record, not statistical acquisition scores.
- Specialists are independent: they do not see each other's nominations before
  submitting their own.  Only the moderator has the full picture.
- The moderator does NOT call an environment action directly — it always calls
  select_genes(), which is the single point of contact with BiologyEnv.
- Shared providers are allowed (the same gpt-4o-mini can serve all specialists
  and the moderator) to keep API costs manageable during development.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

from acua.agents.biology_specialist import BiologySpecialistAgent, SpecialistConfig
from acua.agents.cua_agent import CUAAgent, StructuredDiscussion
from acua.envs.action_sets import ActionSet
from acua.envs.hints import SpecialistViewHints
from acua.envs.observation import Observation
from acua.prompts.dynamic_prompting import (
    ActionFlags,
    ObsFlags,
    OrchestratorDiscussionPrompt,
)
from acua.providers.provider import AbstractProvider
from acua.providers.util import APIPayload, LLMOutput, MessageBuilder

logger = logging.getLogger(__name__)


# ── Moderator system prompt ───────────────────────────────────────────────────

_MODERATOR_SYSTEM = """\
You are the scientific moderator of a multi-target CRISPR perturbation experiment.

Your role:
  - K specialist agents have each proposed gene candidates for their phenotype.
  - You receive their nominations and reasoning in a discussion transcript.
  - You decide the final set of genes to query this cycle.

Moderation principles:
  1. Cross-target value: genes nominated by MULTIPLE specialists are high-priority
     because querying them yields information for all targets simultaneously.
  2. Recall-weighted allocation: targets with lower recall deserve proportionally
     more slots — help lagging targets catch up.
  3. Biology coherence: if a specialist's reasoning is mechanistically sound,
     trust it even if another specialist didn't mention those genes.
  4. Diversity: the batch should span different pathways and gene families to
     maximise the information gained per cycle.
  5. Strict budget: you must select EXACTLY the requested number of genes.
     Do not over-select or under-select.

You respond ONLY with: select_genes(genes=["GENE1", "GENE2", ...])
"""


# ── Helpers for parsing action strings ───────────────────────────────────────


def _coerce_genes_string(s: str) -> list[str]:
    """Convert a string genes value to a list, handling all common LLM formats.

    Handles '["JAK1","STAT3"]' (JSON array string), 'JAK1,STAT3' (CSV),
    and newline-separated variants.
    """
    import json as _json
    s = s.strip()
    if s.startswith("["):
        try:
            result = _json.loads(s)
            if isinstance(result, list):
                return [str(g).strip() for g in result if str(g).strip()]
        except (_json.JSONDecodeError, ValueError):
            pass
        s = s.strip("[]")
    return [g.strip().strip("\"'") for g in s.replace("\n", ",").split(",") if g.strip().strip("\"' ")]


def _normalise_action(action_str: str) -> str:
    """Strip markdown fences and collapse whitespace for robust AST parsing."""
    lines = action_str.strip().splitlines()
    stripped = [l for l in lines if not l.strip().startswith("```")]
    return " ".join(" ".join(stripped).split())


def _parse_nominate_genes(action_str: str) -> tuple[list[str], str]:
    """Extract (genes, reasoning) from a nominate_genes(...) call string.

    Returns ([], "") if parsing fails — the orchestrator will fall back to
    an empty nomination for this specialist and log a warning.
    """
    action_str = _normalise_action(action_str)
    try:
        tree = ast.parse(action_str, mode="eval")
    except SyntaxError:
        return [], ""

    if not isinstance(tree.body, ast.Call):
        return [], ""
    if not isinstance(tree.body.func, ast.Name):
        return [], ""
    if tree.body.func.id not in ("nominate_genes", "finish"):
        return [], ""

    # finish() → empty nomination (specialist signals it has nothing to add)
    if tree.body.func.id == "finish":
        return [], "Specialist signalled finish — no nominations this cycle."

    kwargs: dict = {}
    for kw in tree.body.keywords:
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            pass
    # positional arg support: nominate_genes(["A","B"], "reasoning...")
    for i, arg in enumerate(tree.body.args):
        try:
            val = ast.literal_eval(arg)
            if i == 0:
                kwargs.setdefault("genes", val)
            elif i == 1:
                kwargs.setdefault("reasoning", val)
        except (ValueError, TypeError):
            pass

    genes = kwargs.get("genes", [])
    reasoning = kwargs.get("reasoning", "")
    if isinstance(genes, str):
        genes = _coerce_genes_string(genes)
    if not isinstance(genes, list):
        genes = []
    return genes, str(reasoning)


def _parse_select_genes(action_str: str) -> list[str]:
    """Extract gene list from a select_genes(...) call string.

    Returns [] if parsing fails — the caller should handle the empty case.
    """
    action_str = _normalise_action(action_str)
    try:
        tree = ast.parse(action_str, mode="eval")
    except SyntaxError:
        return []

    if not isinstance(tree.body, ast.Call):
        return []
    if not isinstance(tree.body.func, ast.Name):
        return []
    if tree.body.func.id != "select_genes":
        return []

    kwargs: dict = {}
    for kw in tree.body.keywords:
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            pass
    # positional support
    if tree.body.args:
        try:
            kwargs.setdefault("genes", ast.literal_eval(tree.body.args[0]))
        except (ValueError, TypeError):
            pass

    genes = kwargs.get("genes", [])
    if isinstance(genes, str):
        genes = _coerce_genes_string(genes)
    return genes if isinstance(genes, list) else []


# ── Moderator wrapper ─────────────────────────────────────────────────────────

# Pattern to extract select_genes([...]) from free text (handles both list and
# comma-string formats that gpt-4o-mini tends to produce).
_SELECT_GENES_RE = re.compile(
    r'select_genes\s*\(\s*genes\s*=\s*(.+?)\s*\)',
    re.DOTALL,
)


def _extract_select_genes_from_text(text: str) -> str:
    """Parse a select_genes() call from free-form LLM text.

    Handles:
      select_genes(genes=["A","B"])
      select_genes(genes='["A","B"]')
      select_genes(genes='A,B,C')
      plain JSON array on its own line
    Returns a canonical select_genes(genes=[...]) string, or "" on failure.
    """
    text = _normalise_action(text)

    # Try regex match for select_genes(...)
    m = _SELECT_GENES_RE.search(text)
    if m:
        raw = m.group(1).strip()
        genes = _coerce_genes_string(raw) if isinstance(raw, str) else []
        if not isinstance(genes, list):
            try:
                genes = ast.literal_eval(raw)
            except Exception:
                genes = []
        genes = [g for g in genes if isinstance(g, str) and g.strip()]
        if genes:
            return f"select_genes(genes={genes!r})"

    # Fallback: look for a bare JSON array in the text
    import json as _json
    for chunk in re.findall(r'\[.*?\]', text, re.DOTALL):
        try:
            lst = _json.loads(chunk)
            if isinstance(lst, list) and all(isinstance(g, str) for g in lst) and lst:
                return f"select_genes(genes={lst!r})"
        except Exception:
            pass

    return ""


class _ModeratorAgent:
    """Simple text-based moderator: builds chat history, calls provider, parses output.

    Does NOT use tool calling — gpt-4o-mini via the Responses API returns
    malformed tool-call JSON (arguments truncated to ':[').  Asking the model
    to output plain text and then regex-parsing select_genes([...]) is more
    reliable.
    """

    def __init__(self, provider: AbstractProvider, keep_last_n_obs: int | None = 5):
        self._provider = provider
        self._keep = max(1, keep_last_n_obs or 5)
        # Rolling history: list of (user_text, assistant_text) pairs
        self._history: list[tuple[str, str]] = []

    def get_action(self, discussion_text: str) -> str:
        """Ask moderator to choose genes, return select_genes([...]) action string or ''."""
        # Build messages: system + rolling history + current prompt
        msgs: list[MessageBuilder] = [MessageBuilder.system().add_text(_MODERATOR_SYSTEM)]

        for user_txt, asst_txt in self._history[-self._keep:]:
            msgs.append(MessageBuilder.user().add_text(user_txt))
            msgs.append(MessageBuilder.assistant().add_text(asst_txt))

        msgs.append(MessageBuilder.user().add_text(discussion_text))

        # Call provider in plain-text mode (no tools, no tool_choice)
        response: LLMOutput = self._provider(
            APIPayload(messages=msgs, tools=None, tool_choice=None)
        )

        # Extract text from the response (handles Responses API message items)
        text = self._extract_text(response)

        # Store turn in history
        self._history.append((discussion_text, text or ""))

        return _extract_select_genes_from_text(text)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _extract_text(response: LLMOutput) -> str:
        """Pull plain text out of a provider response (tool-free path)."""
        # think is populated by _extract_thinking_content_from_response which
        # captures message-type output items too
        if response.think:
            return response.think.strip()
        # Fallback: action string might contain the text if pre-parsed
        if response.action:
            return response.action
        return ""


# ── Main orchestrator ─────────────────────────────────────────────────────────


class BiologyOrchestratorAgent:
    """Multi-specialist orchestrator for biology gene-perturbation experiments.

    Usage
    -----
    specialist_configs = [
        SpecialistConfig("schmidt_2021_ifng", provider_a, nominations=5),
        SpecialistConfig("schmidt_2021_il2",  provider_b, nominations=5),
    ]
    orchestrator = BiologyOrchestratorAgent(
        moderator_provider=provider_c,
        specialist_configs=specialist_configs,
    )

    obs = env.reset()
    while True:
        action = orchestrator.get_python_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    Parameters
    ----------
    moderator_provider : LLM provider for the moderator's decision step.
    specialist_configs : one SpecialistConfig per target/phenotype.
    batch_size         : genes selected per cycle (overrides env's batch_size
                         if set; derived from obs.experiment.batch_size if None).
    keep_last_n_obs    : conversation window for specialists and moderator.
    """

    def __init__(
        self,
        moderator_provider: AbstractProvider,
        specialist_configs: list[SpecialistConfig],
        batch_size: int | None = None,
        keep_last_n_obs: int | None = 5,
    ):
        if not specialist_configs:
            raise ValueError("At least one SpecialistConfig is required.")

        # Build one BiologySpecialistAgent per config
        self._specialists: dict[str, BiologySpecialistAgent] = {}
        self._target_order: list[str] = []  # preserves insertion order
        for cfg in specialist_configs:
            if cfg.target_name in self._specialists:
                raise ValueError(f"Duplicate target_name: {cfg.target_name!r}")
            self._specialists[cfg.target_name] = BiologySpecialistAgent(
                target_name=cfg.target_name,
                provider=cfg.provider,
                nominations=cfg.nominations,
                keep_last_n_obs=keep_last_n_obs,
            )
            self._target_order.append(cfg.target_name)

        # Moderator has its own discussion history (accumulates across cycles)
        self._moderator = _ModeratorAgent(
            provider=moderator_provider,
            keep_last_n_obs=keep_last_n_obs,
        )

        self._batch_size = batch_size
        self.last_nominations: dict[str, dict] = {}   # for logging / inspection

    # ── Public interface ──────────────────────────────────────────────────────

    def get_python_action(self, obs: Observation) -> str:
        """Run one full discussion cycle and return select_genes([...]).

        Steps:
          1. For each specialist: build a single-target observation view and
             get a nominate_genes() action string.
          2. Parse all nominations into {target_name: {genes, reasoning}}.
          3. Build OrchestratorDiscussionPrompt and ask the moderator to
             produce the final select_genes().
          4. Validate and return the final action string.
        """
        batch_size = self._batch_size
        if batch_size is None and obs.experiment is not None:
            batch_size = obs.experiment.batch_size
        if batch_size is None:
            batch_size = 10   # safe fallback

        # Phase 1: collect nominations from all specialists
        nominations: dict[str, dict] = {}
        for target_name in self._target_order:
            specialist = self._specialists[target_name]
            specialist_obs = self._build_specialist_obs(obs, target_name)

            nomination_action = specialist.get_python_action(specialist_obs)
            genes, reasoning = _parse_nominate_genes(nomination_action)

            nominations[target_name] = {"genes": genes, "reasoning": reasoning}
            logger.info(
                "[Specialist %s] nominated %d genes: %s",
                target_name, len(genes), genes[:5],
            )

        self.last_nominations = nominations

        # Phase 2: moderator resolves nominations into final selection
        action = self._moderate(obs, nominations, batch_size)

        logger.info("[Moderator] final action: %s", action[:120])
        return action

    def reset(self) -> None:
        """Reset all specialist and moderator conversation histories."""
        for specialist in self._specialists.values():
            specialist.reset()
        self._moderator.reset()
        self.last_nominations = {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_specialist_obs(self, obs: Observation, target_name: str) -> Observation:
        """Build a single-target Observation for one specialist.

        The specialist sees:
          - experiment hints (cycle, budget) from the shared observation
          - specialist_views[0]: only this target's view (NOT others)

        The env populates specialist_views[k] for all K targets; we extract
        the one matching target_name and wrap it in a fresh Observation.
        """
        # Find this specialist's view in the shared observation
        view: SpecialistViewHints | None = None
        if obs.specialist_views:
            for v in obs.specialist_views:
                if v.target_name == target_name:
                    view = v
                    break

        # If no specialist view found (shouldn't happen in normal flow), build
        # a minimal placeholder so the specialist still gets a sensible prompt.
        if view is None:
            logger.warning(
                "No SpecialistViewHints found for target %r in obs.specialist_views. "
                "Falling back to empty view.",
                target_name,
            )
            view = SpecialistViewHints(target_name=target_name)

        # Return a slimmed-down observation: only experiment hints + this view.
        # The specialist agent's _append_observation reads specialist_views[0].
        from dataclasses import replace as dc_replace
        return Observation(
            experiment=obs.experiment,
            error=obs.error,
            step_info=obs.step_info,
            specialist_views=[view],   # exactly one view for this specialist
        )

    def _moderate(
        self,
        obs: Observation,
        nominations: dict[str, dict],
        batch_size: int,
    ) -> str:
        """Ask the moderator LLM to produce the final select_genes() action.

        Uses _ModeratorAgent.get_action() which calls the LLM in plain-text
        mode (no tool calling) and regex-parses select_genes([...]) from the
        response.  Falls back to the union-of-nominations heuristic if parsing
        fails or the gene list is empty.
        """
        discussion_prompt = OrchestratorDiscussionPrompt(
            obs=obs,
            nominations=nominations,
            batch_size=batch_size,
        )
        discussion_text = discussion_prompt.prompt

        action = self._moderator.get_action(discussion_text)

        # Validate: parsed list must contain at least one plausible gene name,
        # then deduplicate (moderator may repeat genes trying to fill the batch)
        if action.startswith("select_genes"):
            genes = _parse_select_genes(action)
            real_genes = [g for g in genes if len(g) > 2]
            if not real_genes:
                action = ""
            else:
                seen: set[str] = set()
                deduped: list[str] = []
                for g in real_genes:
                    if g not in seen:
                        seen.add(g)
                        deduped.append(g)
                action = f"select_genes(genes={deduped!r})"

        if not action.startswith("select_genes"):
            logger.warning(
                "Moderator returned no valid action (got %r) — using fallback heuristic.",
                action[:80] if action else "<empty>",
            )
            action = self._fallback_selection(nominations, batch_size, obs)

        return action

    def _fallback_selection(
        self,
        nominations: dict[str, dict],
        batch_size: int,
        obs: Observation,
    ) -> str:
        """Heuristic fallback: union of nominations, deduplicated, trimmed to batch_size.

        Prioritises genes nominated by multiple specialists, then fills from
        remaining nominations in round-robin order across targets.
        """
        from collections import Counter

        all_genes: list[str] = []
        for nom in nominations.values():
            all_genes.extend(nom.get("genes", []))

        # Build observed gene set to avoid re-querying
        observed_set: set[str] = set()
        if obs.experiment and obs.experiment.observed_gene_names:
            observed_set = set(obs.experiment.observed_gene_names)

        # Deduplicate preserving priority: multi-target genes first
        counts = Counter(all_genes)
        ordered = sorted(set(all_genes), key=lambda g: (-counts[g], all_genes.index(g)))
        valid = [g for g in ordered if g not in observed_set]

        # Trim to batch_size
        selected = valid[:batch_size]

        # If not enough nominations, log a warning — the env will handle the shortfall
        if len(selected) < batch_size:
            logger.warning(
                "Fallback selection has only %d/%d genes — nominations were insufficient.",
                len(selected), batch_size,
            )

        return f"select_genes(genes={selected!r})"
