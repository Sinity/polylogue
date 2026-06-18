"""Tiered scale tests demonstrating the small/medium/large fixture contract.

Each test exercises a measured surface (FTS5 search, hybrid RRF search,
list_sessions) against the corresponding scale fixture and asserts
basic shape so a regression in the seeder or query path surfaces even
when ``--benchmark-disable`` is in effect.

Marker plumbing (issue #1183):

  ``@pytest.mark.scale_small``  — default ``devtools verify`` gate.
  ``@pytest.mark.scale_medium`` — ``devtools verify --lab`` only.
  ``@pytest.mark.scale_large``  — nightly CI / explicit campaigns only.

The default ``devtools verify`` pytest step passes ``-m "not scale_medium
and not scale_large"`` so only the small tier runs in the inner loop.
Tests that want measured timings should also carry ``@pytest.mark.benchmark``
and run under ``devtools bench campaign``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.helpers import open_bench_store


def _fts_query(db_path: Path, term: str, *, limit: int = 20) -> int:
    """Run an FTS search and return the number of hits."""
    with open_bench_store(db_path) as store:
        results = store.run(store.repository.search_summaries(term, limit=limit))
    return len(list(results))


def _list_query(db_path: Path, *, limit: int = 20) -> int:
    with open_bench_store(db_path) as store:
        results = store.run(store.repository.list_summaries(limit=limit))
    return len(list(results))


# ---------------------------------------------------------------------------
# Small tier — runs in default `devtools verify` baseline.
# ---------------------------------------------------------------------------


@pytest.mark.scale_small
def test_scale_small_fts_search_returns_results(tier_small_db: Path) -> None:
    """FTS5 search over the small tier returns at most ``limit`` rows."""
    hits = _fts_query(tier_small_db, "analysis", limit=10)
    assert 0 <= hits <= 10


@pytest.mark.scale_small
def test_scale_small_list_sessions(tier_small_db: Path) -> None:
    """``list_summaries`` returns up to ``limit`` entries from the small tier."""
    rows = _list_query(tier_small_db, limit=10)
    assert 0 < rows <= 10


# ---------------------------------------------------------------------------
# Medium tier — runs in `devtools verify --lab`.
# ---------------------------------------------------------------------------


@pytest.mark.scale_medium
def test_scale_medium_fts_search_returns_results(tier_medium_db: Path) -> None:
    hits = _fts_query(tier_medium_db, "analysis", limit=20)
    assert 0 <= hits <= 20


@pytest.mark.scale_medium
def test_scale_medium_list_sessions(tier_medium_db: Path) -> None:
    rows = _list_query(tier_medium_db, limit=20)
    assert 0 < rows <= 20


# ---------------------------------------------------------------------------
# Large tier — runs in nightly CI / explicit campaigns only.
#
# The fixture creation cost dominates wall-clock time for this tier
# (~tens of seconds to minutes depending on host). The tests below are
# intentionally cheap once the fixture is materialized so the gate stays
# focused on "did the fixture survive its lifecycle" rather than
# "is this host fast enough."
# ---------------------------------------------------------------------------


@pytest.mark.scale_large
def test_scale_large_fts_search_returns_results(tier_large_db: Path) -> None:
    hits = _fts_query(tier_large_db, "analysis", limit=50)
    assert 0 <= hits <= 50


@pytest.mark.scale_large
def test_scale_large_list_sessions(tier_large_db: Path) -> None:
    rows = _list_query(tier_large_db, limit=50)
    assert 0 < rows <= 50
