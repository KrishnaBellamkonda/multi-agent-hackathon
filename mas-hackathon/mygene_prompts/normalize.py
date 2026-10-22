"""Tolerant accessors over MyGene's JSON quirks.

MyGene returns single-vs-list ambiguity for many fields:
``go.BP`` can be one dict or a list of dicts; ``pathway.reactome`` can
be a dict or a list.  All accessors here normalise to lists and accept
``rec is None``.
"""

from __future__ import annotations

from typing import Iterable

# Evidence-code → tier mapping used to filter noisy annotations.
# Tiers are derived from the Gene Ontology evidence-code documentation:
#   HIGH = direct experimental evidence
#   MED  = curator-reviewed / homology / phylogeny / traceable author
#   LOW  = machine-generated, no review
EVIDENCE_TIERS: dict[str, str] = {
    # High: experimental
    "IDA": "HIGH", "IMP": "HIGH", "IGI": "HIGH", "IPI": "HIGH", "EXP": "HIGH",
    # Medium: curator-reviewed / homology / phylogeny / traceable author
    "ISS": "MED",  "ISO": "MED",  "IBA": "MED",  "ISA": "MED",
    "ISM": "MED",  "RCA": "MED",  "TAS": "MED",  "IC":  "MED",
    # Low: machine-generated, no review
    "IEA": "LOW",  "NAS": "LOW",  "ND":  "LOW",
}

_HIGH_MED = frozenset({"HIGH", "MED"})


def as_list(x) -> list:
    """Normalise None / scalar / dict / list to a list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def display_name(rec: dict | None) -> str:
    if not rec:
        return "UNKNOWN"
    sym = rec.get("symbol") or rec.get("query") or "UNKNOWN"
    name = rec.get("name")
    return f"{sym} ({name})" if name else str(sym)


def summary_text(rec: dict | None, max_chars: int = 400) -> str:
    if not rec:
        return ""
    s = rec.get("summary") or ""
    s = s.strip()
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s


def aliases(rec: dict | None, limit: int = 5) -> list[str]:
    if not rec:
        return []
    return [str(a) for a in as_list(rec.get("alias"))][:limit]


def go_terms(
    rec: dict | None,
    category: str = "BP",
    evidence_tiers: Iterable[str] = _HIGH_MED,
) -> list[dict]:
    """Return GO annotations for one category, filtered by tier.

    ``category`` is one of ``"BP"`` (biological process),
    ``"MF"`` (molecular function), ``"CC"`` (cellular component).
    """
    if not rec:
        return []
    tiers = set(evidence_tiers)
    go = rec.get("go") or {}
    entries = as_list(go.get(category))
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        ev = e.get("evidence") or ""
        tier = EVIDENCE_TIERS.get(ev, "LOW")
        if tier not in tiers:
            continue
        out.append(
            {
                "id": e.get("id"),
                "term": e.get("term"),
                "evidence": ev,
                "tier": tier,
                "qualifier": e.get("qualifier"),
            }
        )
    return out


_PATHWAY_SOURCES = ("reactome", "wikipathways", "kegg", "biocarta", "pid")


def pathways(
    rec: dict | None,
    sources: Iterable[str] = _PATHWAY_SOURCES,
) -> list[dict]:
    """Return pathway memberships from the listed sources, deduplicated by name."""
    if not rec:
        return []
    pw_block = rec.get("pathway") or {}
    seen: set[str] = set()
    out: list[dict] = []
    for src in sources:
        for entry in as_list(pw_block.get(src)):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append({"source": src, "id": entry.get("id"), "name": name})
    return out


def interpro_domains(rec: dict | None) -> list[dict]:
    if not rec:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    for entry in as_list(rec.get("interpro")):
        if not isinstance(entry, dict):
            continue
        short = entry.get("short_desc") or entry.get("desc") or entry.get("id")
        if not short or short in seen:
            continue
        seen.add(short)
        out.append(
            {
                "id": entry.get("id"),
                "short_desc": entry.get("short_desc"),
                "desc": entry.get("desc"),
            }
        )
    return out


def type_of_gene(rec: dict | None) -> str:
    if not rec:
        return ""
    return rec.get("type_of_gene") or ""
