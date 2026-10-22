"""HTTP client for MyGene.info v3.

The batch endpoint that works is ``POST /v3/query`` with form body
``q=<csv>``, ``scopes=symbol``, ``species=human``, ``fields=<csv>``.
``POST /v3/gene`` was tried during prior exploration and returned an
empty result set, so do not switch back to it without a reason.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import requests

log = logging.getLogger(__name__)

MG_BASE = "https://mygene.info/v3"
DEFAULT_FIELDS = (
    "symbol,name,summary,go,pathway,interpro,alias,type_of_gene,entrezgene"
)


class MyGeneClient:
    def __init__(
        self,
        base_url: str = MG_BASE,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        species: str = "human",
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.species = species
        self.session = session or requests.Session()

    def _post_with_retry(self, url: str, data: dict) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.post(url, data=data, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt == self.max_retries:
                    raise
                sleep = self.backoff * (2 ** attempt)
                log.warning("MyGene POST failed (%s); retrying in %.1fs", e, sleep)
                time.sleep(sleep)
                continue

            if 500 <= r.status_code < 600 and attempt < self.max_retries:
                sleep = self.backoff * (2 ** attempt)
                log.warning(
                    "MyGene returned %d; retrying in %.1fs", r.status_code, sleep
                )
                time.sleep(sleep)
                continue
            r.raise_for_status()
            return r
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    def fetch_gene(self, symbol: str, fields: str = DEFAULT_FIELDS) -> dict:
        result = self.batch_fetch([symbol], fields=fields, chunk=1, pause=0.0)
        return result.get(symbol, {})

    def batch_fetch(
        self,
        symbols: Iterable[str],
        fields: str = DEFAULT_FIELDS,
        chunk: int = 1000,
        pause: float = 0.2,
    ) -> dict[str, dict]:
        url = f"{self.base_url}/query"
        out: dict[str, dict] = {}
        syms = list(dict.fromkeys(s for s in symbols if s))  # dedup, keep order
        for i in range(0, len(syms), chunk):
            batch = syms[i : i + chunk]
            data = {
                "q": ",".join(batch),
                "scopes": "symbol",
                "species": self.species,
                "fields": fields,
            }
            r = self._post_with_retry(url, data)
            for rec in r.json():
                if rec.get("notfound"):
                    continue
                query = rec.get("query")
                if query is None:
                    continue
                # MyGene can return multiple hits per symbol; keep the first one
                # to land for each query (their default ordering already ranks).
                out.setdefault(query, rec)
            if pause and i + chunk < len(syms):
                time.sleep(pause)
        return out


_default_client: MyGeneClient | None = None


def _get_default_client() -> MyGeneClient:
    global _default_client
    if _default_client is None:
        _default_client = MyGeneClient()
    return _default_client


def fetch_gene(symbol: str, **kw) -> dict:
    return _get_default_client().fetch_gene(symbol, **kw)


def batch_fetch(symbols: Iterable[str], **kw) -> dict[str, dict]:
    return _get_default_client().batch_fetch(symbols, **kw)
