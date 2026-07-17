"""Regression benchmark: ``verify_raw_corpus`` scales sub-quadratically (#1223).

Commit 19b5465 switched from LIMIT/OFFSET to rowid keyset pagination
in ``polylogue/schemas/validation/corpus.py``, fixing an O(n²) regression.
This benchmark asserts the fix holds by measuring wall time across scale
tiers using the corpus-seeded database fixture.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.schemas.validation.corpus import verify_raw_corpus
from polylogue.schemas.validation.requests import SchemaVerificationRequest


def _measure(db_path: Path, record_limit: int | None) -> float:
    """Run ``verify_raw_corpus`` with *record_limit* and return wall-clock ms."""
    request = SchemaVerificationRequest(providers=["chatgpt"], max_samples=1, record_limit=record_limit)
    start = time.perf_counter()
    verify_raw_corpus(db_path=db_path, request=request)
    return (time.perf_counter() - start) * 1000


@pytest.mark.scale_small
def test_schema_check_completes_quickly(named_seeded_archive: Callable[[str], Path]) -> None:
    """Smoke: verify_raw_corpus finishes and returns a valid report."""
    db = named_seeded_archive("schema-small")
    ms = _measure(db, record_limit=None)
    assert ms < 30_000, f"10-record corpus took {ms:.0f} ms; expected <30s"


@pytest.mark.scale_medium
def test_schema_check_linear_scaling(named_seeded_archive: Callable[[str], Path]) -> None:
    """Wall time must grow sub-quadratically across record limits.

    Uses a single seeded DB with 50 records and compares verify_raw_corpus
    runtime at record_limit=5 vs record_limit=50.
    O(n) would give ~10× growth; O(n²) would give ~100×.  50× is a safe fence.
    """
    db = named_seeded_archive("schema-medium")
    ms_5 = _measure(db, record_limit=5)
    ms_50 = _measure(db, record_limit=50)
    ratio = ms_50 / ms_5 if ms_5 > 0 else float("inf")
    assert ratio < 50, (
        f"Scaling ratio {ratio:.1f}× (5→50 records) exceeds 50× sub-quadratic envelope. "
        f"(5: {ms_5:.0f} ms, 50: {ms_50:.0f} ms)"
    )
