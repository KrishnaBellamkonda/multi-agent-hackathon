"""Shared fixtures: load sample_records.json once per session."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_records() -> dict[str, dict]:
    path = Path(__file__).parent / "fixtures" / "sample_records.json"
    with path.open() as fh:
        return json.load(fh)
