"""Search and filter benchmark tests.

Covers: FTS5 search latency (common/rare/multi-word terms),
ConversationFilter execution at scale.

Run with:
    pytest tests/benchmarks/test_search_filters.py --benchmark-enable -p no:xdist -v
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polylogue.lib.filters import ConversationFilter
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository


def _ensure_fts(db_path: Path) -> None:
    """Ensure FTS5 index is populated for search benchmarks."""
    with open_connection(db_path) as conn:
        rebuild_index(conn)


@pytest.mark.benchmark
def test_bench_fts_search_common_term(benchmark, bench_db_5k: Path) -> None:
    """FTS5 search for common word — many results, measures BM25 scoring cost."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(repo.search_summaries("analysis", limit=20)))
    loop.close()


@pytest.mark.benchmark
def test_bench_fts_search_rare_term(benchmark, bench_db_5k: Path) -> None:
    """FTS5 search for rare term — fast empty result path."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(repo.search_summaries("xyzzy_nonexistent_term_42", limit=20)))
    loop.close()


@pytest.mark.benchmark
def test_bench_fts_search_multi_word(benchmark, bench_db_5k: Path) -> None:
    """FTS5 multi-term AND search — intersection cost."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(repo.search_summaries("python analysis", limit=20)))
    loop.close()


@pytest.mark.benchmark
def test_bench_filter_provider(benchmark, bench_db_5k: Path) -> None:
    """ConversationFilter + provider=chatgpt on 5k DB."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(
        ConversationFilter(repo).provider("chatgpt").list_summaries()
    ))
    loop.close()


@pytest.mark.benchmark
def test_bench_filter_has_tool_use(benchmark, bench_db_5k: Path) -> None:
    """ConversationFilter + has_tool_use() — stats LEFT JOIN path."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(
        ConversationFilter(repo).has_tool_use().list_summaries()
    ))
    loop.close()


@pytest.mark.benchmark
def test_bench_filter_semantic_file_ops(benchmark, bench_db_5k: Path) -> None:
    """ConversationFilter + has_file_operations() — EXISTS subquery path (schema v3)."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(
        ConversationFilter(repo).has_file_operations().list_summaries()
    ))
    loop.close()


@pytest.mark.benchmark
def test_bench_filter_count(benchmark, bench_db_5k: Path) -> None:
    """ConversationFilter.count() — single COUNT(*) query, no data fetch."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(
        ConversationFilter(repo).provider("chatgpt").count()
    ))
    loop.close()


@pytest.mark.benchmark
def test_bench_filter_combined(benchmark, bench_db_10k: Path) -> None:
    """Stacked: provider + has_tool_use + min_messages — worst-case filter stack."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_10k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)

    benchmark(lambda: loop.run_until_complete(
        ConversationFilter(repo).provider("claude").has_tool_use().min_messages(2).list_summaries()
    ))
    loop.close()


@pytest.mark.benchmark
@pytest.mark.parametrize("limit", [10, 50, 200])
def test_bench_filter_limit_scaling(benchmark, bench_db_10k: Path, limit: int) -> None:
    """How does result count affect list_conversations cost? Tests LIMIT effect."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_10k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(backend.list_conversations(limit=limit)))
    loop.close()
