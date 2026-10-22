"""Prompt-fragment builders for MyGene.info gene records.

Three families:

* ``context_compact`` / ``context_targeted`` — per-gene blurbs.
* ``phenotype_summary`` — a paragraph for system-prompt augmentation.
* ``feature_set_brief`` — one-paragraph descriptions of the GeneDisco
  feature sets (achilles / string / ccle).

Plus ``lookup_gene_tool_schema`` for function-calling LLMs.

Design rules:

* Lead with ``**SYMBOL** (full name)`` so the LLM sees the identifier first.
* Default to HIGH+MED evidence — LOW (IEA) annotations are noisy.
* Never invent biology.  If a record is empty, say so plainly.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from mygene_prompts.normalize import (
    aliases,
    display_name,
    go_terms,
    interpro_domains,
    pathways,
    summary_text,
    type_of_gene,
)

# ---- Phenotype metadata ----------------------------------------------------

PHENOTYPE_KEYWORDS: dict[str, list[str]] = {
    "schmidt_2021_ifng": [
        "interferon", "cytokine", "immune", "T cell", "t-cell",
        "inflammat", "JAK", "STAT",
    ],
    "schmidt_2021_il2": [
        "interleukin", "IL-2", "T cell", "t-cell",
        "proliferation", "cytokine",
    ],
    "zhuang_2019_nk": [
        "natural killer", "NK cell", "cytotoxic",
        "cell killing", "tumor cell",
    ],
    "sanchez_2021_tau": [
        "tau", "neuron", "synap", "microtubule", "neurodegenerat",
    ],
    "zhu_2021_sarscov2": [
        "virus", "viral", "entry", "replication",
        "innate immune", "antiviral",
    ],
}

PHENOTYPE_SHORT: dict[str, str] = {
    "schmidt_2021_ifng": (
        "T-cell IFNγ secretion (Schmidt 2021, primary human T cells)"
    ),
    "schmidt_2021_il2": (
        "T-cell IL-2 secretion (Schmidt 2021, primary human T cells)"
    ),
    "zhuang_2019_nk": (
        "NK-cell tumour-killing capacity (Zhuang 2019)"
    ),
    "sanchez_2021_tau": (
        "Tau pathology in neurons (Sanchez 2021)"
    ),
    "zhu_2021_sarscov2": (
        "SARS-CoV-2 host-factor infectability (Zhu 2021)"
    ),
}


def _lower_contains_any(text: str, needles: Iterable[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(n.lower() in t for n in needles)


# ---- Per-gene prompt fragments --------------------------------------------


def _format_go_list(go_entries: list[dict], max_n: int) -> str:
    if not go_entries:
        return ""
    parts = []
    for e in go_entries[:max_n]:
        term = e.get("term") or e.get("id") or "?"
        ev = e.get("evidence") or ""
        parts.append(f"{term} [{ev}]" if ev else term)
    return "; ".join(parts)


def _format_pathway_list(pw_entries: list[dict], max_n: int) -> str:
    if not pw_entries:
        return ""
    parts = []
    for p in pw_entries[:max_n]:
        name = p.get("name") or "?"
        src = p.get("source") or ""
        parts.append(f"{name} ({src})" if src else name)
    return "; ".join(parts)


def context_compact(rec: dict | None, max_go: int = 5) -> str:
    """A ~80-token blurb suitable for batch gene blocks.

    Lists the gene name, RefSeq summary (truncated), and top GO BP terms.
    """
    if not rec:
        return "**UNKNOWN** — no MyGene record available."

    lines: list[str] = []
    lines.append(f"**{display_name(rec)}**")

    summary = summary_text(rec, max_chars=240)
    if summary:
        lines.append(f"Summary: {summary}")

    go_bp = go_terms(rec, "BP")
    if go_bp:
        lines.append(f"GO BP: {_format_go_list(go_bp, max_go)}")

    pw = pathways(rec)
    if pw:
        lines.append(f"Pathways: {_format_pathway_list(pw, 3)}")

    if len(lines) == 1:
        lines.append("No high-evidence annotations available for this gene.")
    return "\n".join(lines)


def context_targeted(
    rec: dict | None,
    phenotype: str,
    max_go: int = 5,
) -> str:
    """A ~50-token blurb biased toward annotations relevant to ``phenotype``.

    GO terms whose ``term`` text matches any phenotype keyword are
    surfaced first; pathways are scored the same way.  Falls back to
    ``context_compact`` if nothing matches.
    """
    if not rec:
        return "**UNKNOWN** — no MyGene record available."

    keywords = PHENOTYPE_KEYWORDS.get(phenotype, [])

    go_bp = go_terms(rec, "BP")
    if keywords:
        go_bp.sort(
            key=lambda e: (
                0 if _lower_contains_any(e.get("term") or "", keywords) else 1,
                e.get("evidence") != "IDA",
            )
        )

    pw = pathways(rec)
    if keywords:
        pw.sort(
            key=lambda p: (
                0 if _lower_contains_any(p.get("name") or "", keywords) else 1,
            )
        )

    lines: list[str] = [f"**{display_name(rec)}**"]

    summary = summary_text(rec, max_chars=200)
    if summary and (
        not keywords or _lower_contains_any(summary, keywords)
    ):
        lines.append(f"Summary: {summary}")

    if go_bp:
        # Keep targeted output tight: only show the top-matching GO terms.
        relevant = [
            e for e in go_bp
            if not keywords or _lower_contains_any(e.get("term") or "", keywords)
        ][:max_go]
        if relevant:
            lines.append(
                f"Relevant GO BP for {phenotype}: "
                f"{_format_go_list(relevant, max_go)}"
            )

    if pw:
        relevant_pw = [
            p for p in pw
            if not keywords or _lower_contains_any(p.get("name") or "", keywords)
        ][:3]
        if relevant_pw:
            lines.append(
                f"Relevant pathways: {_format_pathway_list(relevant_pw, 3)}"
            )

    if len(lines) == 1:
        # Nothing matched the phenotype keywords; fall back to compact.
        return context_compact(rec, max_go=max_go)
    return "\n".join(lines)


# ---- Per-phenotype block --------------------------------------------------


def phenotype_summary(
    phenotype: str,
    exemplar_genes: list[dict],
    max_genes: int = 5,
    max_pathways: int = 3,
) -> str:
    """A paragraph for ``BiologyKnowledgeAgent.get_phenotype_context``.

    Composes: canonical short description (``PHENOTYPE_SHORT``);
    the ``max_pathways`` pathways that recur most often across
    ``exemplar_genes`` (frequency-ranked); up to ``max_genes`` example
    regulators (by display name).
    """
    short = PHENOTYPE_SHORT.get(phenotype, phenotype)

    pathway_counter: Counter[str] = Counter()
    for rec in exemplar_genes:
        seen_in_rec: set[str] = set()
        for p in pathways(rec):
            name = p.get("name")
            if name and name not in seen_in_rec:
                pathway_counter[name] += 1
                seen_in_rec.add(name)

    top_pathways = [name for name, _ in pathway_counter.most_common(max_pathways)]

    gene_labels: list[str] = []
    for rec in exemplar_genes[:max_genes]:
        sym = rec.get("symbol") or rec.get("query")
        if sym:
            gene_labels.append(sym)

    parts = [f"Screen: {short}."]
    if top_pathways:
        parts.append(
            "Recurrent pathways among known regulators: "
            + "; ".join(top_pathways)
            + "."
        )
    if gene_labels:
        parts.append(
            "Example regulators in this screen: "
            + ", ".join(gene_labels)
            + "."
        )
    if not top_pathways and not gene_labels:
        parts.append(
            "No exemplar-gene annotations supplied; LLM should fall back "
            "to its prior knowledge for this phenotype."
        )
    return " ".join(parts)


# ---- Feature-set briefs ---------------------------------------------------

_FEATURE_SET_BRIEFS: dict[str, str] = {
    "achilles": (
        "**achilles** — gene-essentiality fingerprints from the DepMap "
        "Project Achilles CRISPR screen across ~1,000 human cancer cell "
        "lines.  Each gene is described by its loss-of-function fitness "
        "effect in every line, so two genes look similar when they are "
        "essential in the same cellular contexts.  Captures co-functional "
        "modules even when the genes are not physically interacting."
    ),
    "string": (
        "**string** — STRING-DB protein–protein interaction network "
        "embeddings.  Each gene is described by its position in the "
        "graph of physical and functional protein associations, so two "
        "genes look similar when they participate in the same protein "
        "complexes or signalling cascades.  Captures direct interaction "
        "neighbourhoods well; weaker for purely transcriptional regulators."
    ),
    "ccle": (
        "**ccle** — Cancer Cell Line Encyclopaedia protein-quantification "
        "profiles.  Each gene is described by the abundance of its "
        "protein product across hundreds of cancer cell lines, so two "
        "genes look similar when their expression is co-regulated.  "
        "Captures co-expression modules; orthogonal to genetic dependency."
    ),
}


def feature_set_brief(feature_set_name: str) -> str:
    """One-paragraph description of a GeneDisco feature set."""
    name = (feature_set_name or "").lower()
    brief = _FEATURE_SET_BRIEFS.get(name)
    if brief is None:
        return (
            f"**{feature_set_name}** — no canonical description on file; "
            "treat as an opaque numeric embedding of the gene."
        )
    return brief


# ---- Tool-use schema ------------------------------------------------------


def lookup_gene_tool_schema() -> dict:
    """JSON-schema for a ``lookup_gene`` function-calling tool.

    Suitable for OpenAI / Anthropic tool-use formats: the outer dict is
    a single tool definition with ``name``, ``description``, and an
    ``input_schema`` JSON-schema.
    """
    return {
        "name": "lookup_gene",
        "description": (
            "Look up a human gene by HGNC symbol and return a short "
            "biological context fragment.  Use this when reasoning about "
            "an unfamiliar gene symbol.  Optionally pass `phenotype` to "
            "get an annotation list filtered for that screen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "HGNC gene symbol, e.g. 'STAT1', 'JAK1', 'IFNG'."
                    ),
                },
                "phenotype": {
                    "type": "string",
                    "description": (
                        "Optional screen identifier, one of: "
                        + ", ".join(sorted(PHENOTYPE_SHORT))
                        + "."
                    ),
                    "enum": sorted(PHENOTYPE_SHORT),
                },
            },
            "required": ["symbol"],
        },
    }


__all__ = [
    "PHENOTYPE_KEYWORDS",
    "PHENOTYPE_SHORT",
    "context_compact",
    "context_targeted",
    "phenotype_summary",
    "feature_set_brief",
    "lookup_gene_tool_schema",
]
