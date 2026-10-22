"""Offline tests for prompts.py — no network."""

from __future__ import annotations

from mygene_prompts.prompts import (
    PHENOTYPE_KEYWORDS,
    PHENOTYPE_SHORT,
    context_compact,
    context_targeted,
    feature_set_brief,
    lookup_gene_tool_schema,
    phenotype_summary,
)


def test_context_compact_includes_symbol_and_summary(sample_records):
    out = context_compact(sample_records["STAT1"])
    assert "**STAT1" in out
    assert "Summary:" in out or "GO BP:" in out


def test_context_compact_handles_missing_record():
    out = context_compact(None)
    assert "UNKNOWN" in out
    assert "no mygene record" in out.lower()


def test_context_targeted_surfaces_relevant_terms(sample_records):
    out = context_targeted(sample_records["STAT1"], "schmidt_2021_ifng")
    assert "**STAT1" in out
    # STAT1 must surface at least one cytokine / JAK / interferon-related line.
    lower = out.lower()
    assert any(
        kw in lower
        for kw in PHENOTYPE_KEYWORDS["schmidt_2021_ifng"]
        if kw == kw.lower()  # keep simple substring keywords
    ) or "Relevant" in out


def test_context_targeted_falls_back_when_no_match(sample_records):
    # TP53 is not strongly cytokine-related; should still produce *some*
    # text rather than an empty string.
    out = context_targeted(sample_records["TP53"], "schmidt_2021_ifng")
    assert "**TP53" in out
    assert len(out) > len("**TP53**")


def test_context_targeted_handles_missing_record():
    out = context_targeted(None, "schmidt_2021_ifng")
    assert "UNKNOWN" in out


def test_phenotype_summary_uses_canonical_short(sample_records):
    exemplars = [sample_records["STAT1"], sample_records["JAK1"], sample_records["IFNG"]]
    out = phenotype_summary("schmidt_2021_ifng", exemplars)
    assert PHENOTYPE_SHORT["schmidt_2021_ifng"] in out
    assert "Example regulators" in out
    assert "STAT1" in out and "JAK1" in out


def test_phenotype_summary_handles_no_exemplars():
    out = phenotype_summary("schmidt_2021_ifng", [])
    assert PHENOTYPE_SHORT["schmidt_2021_ifng"] in out
    assert "fall back" in out.lower()


def test_phenotype_summary_ranks_pathways_by_frequency():
    # Three exemplars all share pathway A but only one has B.
    a = {"symbol": "A", "pathway": {"reactome": [{"id": "1", "name": "shared pathway"}]}}
    b = {"symbol": "B", "pathway": {"reactome": [{"id": "1", "name": "shared pathway"}]}}
    c = {
        "symbol": "C",
        "pathway": {
            "reactome": [
                {"id": "1", "name": "shared pathway"},
                {"id": "2", "name": "unique pathway"},
            ]
        },
    }
    out = phenotype_summary("schmidt_2021_ifng", [a, b, c])
    assert "shared pathway" in out
    # "shared pathway" should appear before "unique pathway"
    if "unique pathway" in out:
        assert out.index("shared pathway") < out.index("unique pathway")


def test_feature_set_brief_known_and_unknown():
    assert "Achilles" in feature_set_brief("achilles")
    assert "STRING-DB" in feature_set_brief("string")
    assert "Cancer Cell Line" in feature_set_brief("ccle")
    # Unknown feature set must not crash.
    out = feature_set_brief("zzz_unknown")
    assert "zzz_unknown" in out


def test_lookup_gene_tool_schema_shape():
    sch = lookup_gene_tool_schema()
    assert sch["name"] == "lookup_gene"
    assert "symbol" in sch["input_schema"]["properties"]
    assert "phenotype" in sch["input_schema"]["properties"]
    assert sch["input_schema"]["required"] == ["symbol"]
    enum = sch["input_schema"]["properties"]["phenotype"]["enum"]
    assert set(enum) == set(PHENOTYPE_SHORT.keys())
