"""Disk-backed gene-record cache.

The full 17K-gene cache is ~8–12 MB gzipped.  We pickle the in-memory
dict and write atomically via ``os.replace`` so a crash mid-write does
not corrupt the file.
"""

from __future__ import annotations

import gzip
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Iterable

from mygene_prompts.client import DEFAULT_FIELDS, MyGeneClient

log = logging.getLogger(__name__)


def default_cache_path() -> Path:
    """Cache lives under ExperimentalDesign/data/ which is gitignored."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # mas-hackathon/
    return repo_root / "ExperimentalDesign" / "data" / "mygene_cache.pkl.gz"


class GeneCache:
    def __init__(self, path: Path | None = None, data: dict[str, dict] | None = None):
        self.path = Path(path) if path is not None else default_cache_path()
        self._data: dict[str, dict] = dict(data) if data else {}

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data

    def __getitem__(self, symbol: str) -> dict:
        return self._data[symbol]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get(self, symbol: str, default=None):
        return self._data.get(symbol, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def update(self, mapping: dict[str, dict]) -> None:
        self._data.update(mapping)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with gzip.open(tmp, "wb") as fh:
            pickle.dump(self._data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)

    @classmethod
    def load(cls, path: Path | None = None) -> "GeneCache":
        p = Path(path) if path is not None else default_cache_path()
        if not p.exists():
            return cls(path=p)
        with gzip.open(p, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Cache at {p} is not a dict")
        return cls(path=p, data=data)


def build_full_cache(
    symbols: Iterable[str],
    path: Path | None = None,
    client: MyGeneClient | None = None,
    fields: str = DEFAULT_FIELDS,
    chunk: int = 1000,
    pause: float = 0.5,
    resume: bool = True,
    progress: bool = True,
) -> GeneCache:
    """Idempotent: load existing cache if present, fetch only missing symbols."""
    cache = GeneCache.load(path) if resume else GeneCache(path=path)
    client = client or MyGeneClient()

    wanted = list(dict.fromkeys(s for s in symbols if s))
    missing = [s for s in wanted if s not in cache]
    if not missing:
        log.info("Cache already has all %d requested symbols", len(wanted))
        return cache

    log.info(
        "Building cache: %d wanted, %d already cached, %d to fetch",
        len(wanted), len(wanted) - len(missing), len(missing),
    )

    t0 = time.time()
    total_fetched = 0
    for i in range(0, len(missing), chunk):
        batch = missing[i : i + chunk]
        result = client.batch_fetch(batch, fields=fields, chunk=chunk, pause=0.0)
        cache.update(result)
        total_fetched += len(result)
        if progress:
            elapsed = time.time() - t0
            log.info(
                "  chunk %d/%d: +%d records, %d total cached (%.1fs elapsed)",
                i // chunk + 1,
                (len(missing) + chunk - 1) // chunk,
                len(result),
                len(cache),
                elapsed,
            )
        if pause and i + chunk < len(missing):
            time.sleep(pause)
    cache.save()
    log.info(
        "Cache build complete: fetched %d new records in %.1fs; saved to %s",
        total_fetched, time.time() - t0, cache.path,
    )
    return cache
