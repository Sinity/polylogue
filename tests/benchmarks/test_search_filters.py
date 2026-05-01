"""Search and filter benchmark tests.

Covers: FTS5 search latency (common/rare/multi-word terms),
ConversationFilter execution at scale.

Run with:
    pytest tests/benchmarks/test_search_filters.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.filter.filters import ConversationFilter
from polylogue.storage.query_models import ConversationRecordQuery
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


@pytest.mark.benchmark
def test_bench_filter_provider(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """ConversationFilter + provider=chatgpt on 5k DB."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: ConversationFilter(store.repository).provider("chatgpt").list_summaries(),
    )


@pytest.mark.benchmark
def test_bench_filter_has_tool_use(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """ConversationFilter + has_tool_use() — stats LEFT JOIN path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: ConversationFilter(store.repository).has_tool_use().list_summaries(),
    )


@pytest.mark.benchmark
def test_bench_filter_semantic_file_ops(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """ConversationFilter + has_file_operations() — EXISTS subquery path (schema v3)."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: ConversationFilter(store.repository).has_file_operations().list_summaries(),
    )


@pytest.mark.benchmark
def test_bench_filter_count(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """ConversationFilter.count() — single COUNT(*) query, no data fetch."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: ConversationFilter(store.repository).provider("chatgpt").count(),
    )


@pytest.mark.benchmark
def test_bench_filter_combined(benchmark: BenchmarkFixture, bench_db_10k: Path) -> None:
    """Stacked: provider + has_tool_use + min_messages — worst-case filter stack."""
    benchmark_store_call(
        benchmark,
        bench_db_10k,
        lambda store: (
            ConversationFilter(store.repository).provider("claude-ai").has_tool_use().min_messages(2).list_summaries()
        ),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("limit", [10, 50, 200])
def test_bench_filter_limit_scaling(benchmark: BenchmarkFixture, bench_db_10k: Path, limit: int) -> None:
    """How does result count affect list_conversations cost? Tests LIMIT effect."""
    benchmark_store_call(
        benchmark,
        bench_db_10k,
        lambda store: store.backend.queries.list_conversations(ConversationRecordQuery(limit=limit)),
    )
