"""Offline tests for cache.py — uses tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from mygene_prompts.cache import GeneCache, build_full_cache


def test_genecache_roundtrip(tmp_path: Path):
    p = tmp_path / "cache.pkl.gz"
    cache = GeneCache(path=p)
    cache.update({"FOO": {"symbol": "FOO", "name": "fake one"}})
    cache.update({"BAR": {"symbol": "BAR", "name": "fake two"}})
    cache.save()
    assert p.exists()

    cache2 = GeneCache.load(p)
    assert len(cache2) == 2
    assert "FOO" in cache2
    assert cache2["FOO"]["symbol"] == "FOO"
    assert cache2.get("MISSING") is None


def test_genecache_load_missing_returns_empty(tmp_path: Path):
    p = tmp_path / "does_not_exist.pkl.gz"
    cache = GeneCache.load(p)
    assert len(cache) == 0
    assert cache.path == p


def test_genecache_save_is_atomic(tmp_path: Path):
    """`.tmp` file must not survive a successful save."""
    p = tmp_path / "cache.pkl.gz"
    cache = GeneCache(path=p)
    cache.update({"X": {"symbol": "X"}})
    cache.save()
    siblings = list(tmp_path.iterdir())
    assert len(siblings) == 1
    assert siblings[0] == p


class _FakeClient:
    def __init__(self, mapping: dict[str, dict]):
        self._mapping = mapping
        self.calls: list[list[str]] = []

    def batch_fetch(self, symbols, fields=None, chunk=1000, pause=0.0):
        syms = list(symbols)
        self.calls.append(syms)
        return {s: self._mapping[s] for s in syms if s in self._mapping}


def test_build_full_cache_is_idempotent(tmp_path: Path):
    p = tmp_path / "cache.pkl.gz"
    universe = {
        "FOO": {"symbol": "FOO"},
        "BAR": {"symbol": "BAR"},
        "BAZ": {"symbol": "BAZ"},
    }
    client = _FakeClient(universe)

    cache = build_full_cache(
        ["FOO", "BAR", "BAZ"],
        path=p, client=client, chunk=2, pause=0.0,
    )
    assert len(cache) == 3
    assert sum(len(c) for c in client.calls) == 3

    # Second run with overlapping symbols should only fetch the new one.
    client2 = _FakeClient(universe)
    cache2 = build_full_cache(
        ["FOO", "BAR", "BAZ"],
        path=p, client=client2, chunk=2, pause=0.0,
    )
    assert len(cache2) == 3
    # No symbols missing → no batch calls at all.
    assert client2.calls == []


def test_build_full_cache_resume_false_refetches_everything(tmp_path: Path):
    p = tmp_path / "cache.pkl.gz"
    universe = {"FOO": {"symbol": "FOO"}, "BAR": {"symbol": "BAR"}}

    client = _FakeClient(universe)
    build_full_cache(["FOO"], path=p, client=client, chunk=10, pause=0.0)

    client2 = _FakeClient(universe)
    cache2 = build_full_cache(
        ["FOO", "BAR"],
        path=p, client=client2, chunk=10, pause=0.0, resume=False,
    )
    # resume=False → cache starts empty, both symbols are fetched.
    assert len(cache2) == 2
    assert sum(len(c) for c in client2.calls) == 2
