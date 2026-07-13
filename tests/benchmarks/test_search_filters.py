"""Search and filter benchmark tests.

Covers: FTS5 search latency (common/rare/multi-word terms),
SessionFilter execution at scale.

Run with:
    pytest tests/benchmarks/test_search_filters.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.query_models import SessionRecordQuery
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_store_call


@pytest.mark.benchmark
def test_bench_fts_search_common_term(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """FTS5 search for common word — many results, measures BM25 scoring cost."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.search_summaries("analysis", limit=20),
    )


@pytest.mark.benchmark
def test_bench_fts_search_rare_term(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """FTS5 search for rare term — fast empty result path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.search_summaries("xyzzy_nonexistent_term_42", limit=20),
    )


@pytest.mark.benchmark
def test_bench_fts_search_multi_word(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """FTS5 multi-term AND search — intersection cost."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.search_summaries("python analysis", limit=20),
    )


# ``SessionFilter`` executes over ``index.db`` and is keyword-only
# (``SessionFilter(archive_root=...)``). The benchmark fixtures seed a
# SQLiteBackend database, so these benchmarks exercise the
# equivalent repository query surface (``list_summaries`` / ``count_by_query``)
# at scale.


@pytest.mark.benchmark
def test_bench_filter_provider(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """Provider-scoped summary listing on 5k DB."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.list_summaries(origin="chatgpt"),
    )


@pytest.mark.benchmark
def test_bench_filter_has_tool_use(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """has_tool_use filter — stats LEFT JOIN path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.list_summaries(has_tool_use=True),
    )


@pytest.mark.benchmark
def test_bench_filter_semantic_file_ops(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """Semantic file-operation action filter — EXISTS subquery path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.list_summaries(action_terms=["file_read", "file_write", "file_edit"]),
    )


@pytest.mark.benchmark
def test_bench_filter_count(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """count_by_query — single COUNT(*) query, no data fetch."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.count_by_query(SessionRecordQuery(origin="chatgpt")),
    )


@pytest.mark.benchmark
def test_bench_filter_combined(benchmark: BenchmarkFixture, bench_db_10k: Path) -> None:
    """Stacked: provider + has_tool_use + min_messages — worst-case filter stack."""
    benchmark_store_call(
        benchmark,
        bench_db_10k,
        lambda store: store.repository.list_summaries(
            origin="claude-ai-export",
            has_tool_use=True,
            min_messages=2,
        ),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("limit", [10, 50, 200])
def test_bench_filter_limit_scaling(benchmark: BenchmarkFixture, bench_db_10k: Path, limit: int) -> None:
    """How does result count affect list_sessions cost? Tests LIMIT effect."""
    benchmark_store_call(
        benchmark,
        bench_db_10k,
        lambda store: store.backend.queries.list_sessions(SessionRecordQuery(limit=limit)),
    )
