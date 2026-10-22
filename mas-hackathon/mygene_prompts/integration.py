"""Adapters that wire ``mygene_prompts`` into maehd-style call sites.

These functions are *examples* — they show how to compose the lower-
level primitives in :mod:`mygene_prompts.prompts` with the existing
``BiologyKnowledgeAgent`` surface area, without modifying ``maehd``.

Each adapter accepts an optional ``cache: GeneCache``.  If ``None``,
the adapter does a live ``batch_fetch`` once and warns; for the
hackathon pipeline you almost always want to pre-build the cache.
"""

from __future__ import annotations

import logging
from typing import Iterable

from mygene_prompts.cache import GeneCache
from mygene_prompts.client import batch_fetch
from mygene_prompts.prompts import (
    PHENOTYPE_SHORT,
    context_compact,
    context_targeted,
    feature_set_brief,
    phenotype_summary,
)

log = logging.getLogger(__name__)


def _resolve_records(
    symbols: Iterable[str],
    cache: GeneCache | None,
) -> dict[str, dict]:
    syms = list(dict.fromkeys(symbols))
    if cache is None:
        log.warning(
            "No GeneCache passed; falling back to a live MyGene batch "
            "fetch for %d symbols.  For repeated calls, build a cache "
            "with `python -m mygene_prompts build-cache`.",
            len(syms),
        )
        return batch_fetch(syms)

    have = {s: cache[s] for s in syms if s in cache}
    missing = [s for s in syms if s not in cache]
    if missing:
        log.warning(
            "%d/%d symbols missing from cache; fetching live: %s%s",
            len(missing), len(syms),
            ", ".join(missing[:5]),
            "..." if len(missing) > 5 else "",
        )
        fetched = batch_fetch(missing)
        have.update(fetched)
        cache.update(fetched)
    return have


def enriched_phenotype_context(
    screen_names: list[str],
    base_context: str,
    cache: GeneCache | None = None,
    exemplar_genes_per_screen: dict[str, list[str]] | None = None,
) -> str:
    """Augment ``BiologyKnowledgeAgent.get_phenotype_context`` output.

    Returns ``base_context`` followed by one ``phenotype_summary``
    paragraph per screen.  If ``exemplar_genes_per_screen`` is omitted,
    only the canonical short description is appended (no recurrent
    pathway / regulator list).

    The result is intended to be passed straight into
    ``llm_agent._build_system_prompt`` in place of the bare biology
    context.
    """
    parts = [base_context.rstrip()] if base_context else []
    exemplar_genes_per_screen = exemplar_genes_per_screen or {}

    for screen in screen_names:
        exemplars = exemplar_genes_per_screen.get(screen, [])
        recs_by_sym = _resolve_records(exemplars, cache) if exemplars else {}
        rec_list = [recs_by_sym[s] for s in exemplars if s in recs_by_sym]
        parts.append(phenotype_summary(screen, rec_list))

    return "\n\n".join(p for p in parts if p)


def enriched_gene_block(
    gene_names: list[str],
    phenotype: str,
    cache: GeneCache | None = None,
    pattern: str = "targeted",
) -> str:
    """A newline-joined block of per-gene context fragments.

    Splice into ``_BATCH_SCORE_PROMPT`` (in ``biology_agent.py``) before
    sending to the LLM.  ``pattern`` selects ``context_compact`` or
    ``context_targeted``; targeted is the default because the caller
    already knows which screen it is scoring for.
    """
    if pattern not in ("compact", "targeted"):
        raise ValueError(f"pattern must be 'compact' or 'targeted', got {pattern!r}")

    recs = _resolve_records(gene_names, cache)
    blocks: list[str] = []
    for sym in gene_names:
        rec = recs.get(sym)
        if pattern == "compact":
            blocks.append(context_compact(rec))
        else:
            blocks.append(context_targeted(rec, phenotype))
    return "\n\n".join(blocks)


def enriched_feature_recommendation_hint(
    screen_names: list[str],
    feature_sets: list[str],
    cache: GeneCache | None = None,
    exemplar_genes_per_screen: dict[str, list[str]] | None = None,
) -> str:
    """Prepend to the ``recommend_features`` prompt.

    Returns a paragraph naming each feature set and the recurrent
    pathway pattern across each screen's known regulators.  The LLM
    then has to reason about which feature set is most likely to
    *capture* that pattern.
    """
    parts: list[str] = []

    # Feature-set briefs.
    fs_lines = [feature_set_brief(name) for name in feature_sets]
    if fs_lines:
        parts.append("Feature-set descriptions:\n" + "\n\n".join(fs_lines))

    # Per-screen recurrent-pathway summary.
    exemplar_genes_per_screen = exemplar_genes_per_screen or {}
    for screen in screen_names:
        exemplars = exemplar_genes_per_screen.get(screen, [])
        recs_by_sym = _resolve_records(exemplars, cache) if exemplars else {}
        rec_list = [recs_by_sym[s] for s in exemplars if s in recs_by_sym]
        parts.append(phenotype_summary(screen, rec_list))

    parts.append(
        "Choose the feature set whose biology (interaction graph, "
        "essentiality fingerprint, or co-expression module) is most "
        "likely to capture the recurrent pathways listed above."
    )
    return "\n\n".join(parts)


__all__ = [
    "PHENOTYPE_SHORT",
    "enriched_phenotype_context",
    "enriched_gene_block",
    "enriched_feature_recommendation_hint",
]
