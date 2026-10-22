"""Offline tests for normalize.py — no network."""

from __future__ import annotations

from mygene_prompts.normalize import (
    EVIDENCE_TIERS,
    aliases,
    as_list,
    display_name,
    go_terms,
    interpro_domains,
    pathways,
    summary_text,
)


def test_as_list_handles_none_scalar_dict_list():
    assert as_list(None) == []
    assert as_list({"a": 1}) == [{"a": 1}]
    assert as_list([1, 2]) == [1, 2]
    assert as_list("x") == ["x"]


def test_display_name_falls_back_when_record_empty():
    assert display_name(None) == "UNKNOWN"
    assert display_name({}) == "UNKNOWN"
    assert display_name({"symbol": "STAT1"}) == "STAT1"
    assert (
        display_name({"symbol": "STAT1", "name": "signal transducer"})
        == "STAT1 (signal transducer)"
    )


def test_summary_text_truncates(sample_records):
    s = summary_text(sample_records["STAT1"], max_chars=40)
    assert len(s) <= 40
    assert s.endswith("…")
    assert summary_text(None) == ""
    assert summary_text({}) == ""


def test_go_terms_filters_by_evidence_tier(sample_records):
    high_med = go_terms(sample_records["IFNG"], "BP")
    all_tiers = go_terms(
        sample_records["IFNG"], "BP",
        evidence_tiers={"HIGH", "MED", "LOW"},
    )
    assert len(high_med) > 0
    assert len(high_med) <= len(all_tiers)
    for entry in high_med:
        assert entry["tier"] in {"HIGH", "MED"}
        assert EVIDENCE_TIERS[entry["evidence"]] in {"HIGH", "MED"}


def test_go_terms_tolerates_single_dict_vs_list():
    # Forge a record where go.BP is a single dict, not a list.
    forged = {
        "symbol": "FOO",
        "go": {"BP": {"id": "GO:1", "term": "fake process", "evidence": "IDA"}},
    }
    out = go_terms(forged, "BP")
    assert len(out) == 1
    assert out[0]["term"] == "fake process"
    assert out[0]["tier"] == "HIGH"


def test_go_terms_empty_inputs():
    assert go_terms(None) == []
    assert go_terms({}) == []
    assert go_terms({"go": None}) == []


def test_pathways_dedupes_by_name(sample_records):
    # STAT1 lives in many overlapping reactome / wikipathway entries;
    # dedup by name keeps the first source we encounter.
    pw = pathways(sample_records["STAT1"])
    names = [p["name"] for p in pw]
    assert len(names) == len(set(names))


def test_pathways_tolerates_single_dict_vs_list():
    forged = {
        "pathway": {
            "reactome": {"id": "R-HSA-1", "name": "fake reactome pathway"},
            "wikipathways": [
                {"id": "WP1", "name": "fake wiki pathway"},
                {"id": "WP1", "name": "fake wiki pathway"},  # dup
            ],
        }
    }
    pw = pathways(forged)
    names = [p["name"] for p in pw]
    assert names == ["fake reactome pathway", "fake wiki pathway"]


def test_interpro_domains_dedupes(sample_records):
    out = interpro_domains(sample_records["JAK1"])
    seen = {entry["short_desc"] or entry["id"] for entry in out}
    assert len(seen) == len(out)


def test_aliases_returns_list(sample_records):
    out = aliases(sample_records["IFNG"], limit=3)
    assert isinstance(out, list)
    assert len(out) <= 3
