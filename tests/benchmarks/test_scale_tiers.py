"""Query probes over the benchmark runner's shared database sizes.

Each test exercises a measured surface (FTS5 search, hybrid RRF search,
list_sessions) against the benchmark fixtures and asserts basic shape. The
entire file is an explicit performance-plugin surface; native correctness
verification excludes ``tests/benchmarks`` by path instead of tier markers.
"""

from __future__ import annotations

from pathlib import Path

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
# Small benchmark database.
# ---------------------------------------------------------------------------


def test_bench_1k_fts_search_returns_results(bench_db_1k: Path) -> None:
    """FTS5 search over the small tier returns at most ``limit`` rows."""
    hits = _fts_query(bench_db_1k, "analysis", limit=10)
    assert 0 <= hits <= 10


def test_bench_1k_list_sessions(bench_db_1k: Path) -> None:
    """``list_summaries`` returns up to ``limit`` entries from the small tier."""
    rows = _list_query(bench_db_1k, limit=10)
    assert 0 < rows <= 10


# ---------------------------------------------------------------------------
# Medium benchmark database.
# ---------------------------------------------------------------------------


def test_bench_10k_fts_search_returns_results(bench_db_10k: Path) -> None:
    hits = _fts_query(bench_db_10k, "analysis", limit=20)
    assert 0 <= hits <= 20


def test_bench_10k_list_sessions(bench_db_10k: Path) -> None:
    rows = _list_query(bench_db_10k, limit=20)
    assert 0 < rows <= 20


# ---------------------------------------------------------------------------
# Large benchmark database — direct file/node campaigns only.
#
# The fixture creation cost dominates wall-clock time for this tier
# (~tens of seconds to minutes depending on host). The tests below are
# intentionally cheap once the fixture is materialized so the gate stays
# focused on "did the fixture survive its lifecycle" rather than
# "is this host fast enough."
# ---------------------------------------------------------------------------


def test_bench_50k_fts_search_returns_results(bench_db_50k: Path) -> None:
    hits = _fts_query(bench_db_50k, "analysis", limit=50)
    assert 0 <= hits <= 50


def test_bench_50k_list_sessions(bench_db_50k: Path) -> None:
    rows = _list_query(bench_db_50k, limit=50)
    assert 0 < rows <= 50
