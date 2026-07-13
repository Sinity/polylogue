"""Storage benchmark tests.

Covers: batch inserts, list queries (unfiltered, provider, stats-join,
semantic filter), batch get operations.

Run with:
    pytest tests/benchmarks/test_storage.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from polylogue.storage.query_models import SessionRecordQuery
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_store_call, open_bench_store
from tests.infra.storage_records import (
    make_content_block,
    make_message,
    make_session,
    save_current_archive_records,
)


def _make_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


@pytest.mark.benchmark
def test_bench_list_sessions_no_filter(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """list_sessions(limit=50) on 5k-message DB — baseline query cost."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.backend.queries.list_sessions(SessionRecordQuery(limit=50)),
    )


@pytest.mark.benchmark
def test_bench_list_sessions_provider_filter(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """list with provider=chatgpt — tests simple WHERE on indexed column."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.backend.queries.list_sessions(SessionRecordQuery(origin="chatgpt", limit=50)),
    )


@pytest.mark.benchmark
def test_bench_list_sessions_has_tool_use(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """list with has_tool_use=True — tests stats LEFT JOIN path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.backend.queries.list_sessions(SessionRecordQuery(has_tool_use=True, limit=50)),
    )


@pytest.mark.benchmark
def test_bench_list_sessions_semantic_filter(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """list with action_terms=file_read — tests semantic EXISTS subquery path."""
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.backend.queries.list_sessions(SessionRecordQuery(action_terms=("file_read",), limit=50)),
    )


@pytest.mark.benchmark
def test_bench_list_sessions_combined_filter(benchmark: BenchmarkFixture, bench_db_10k: Path) -> None:
    """provider + has_tool_use + min_messages — combined filter stack."""
    benchmark_store_call(
        benchmark,
        bench_db_10k,
        lambda store: store.backend.queries.list_sessions(
            SessionRecordQuery(
                origin="claude-ai-export",
                has_tool_use=True,
                min_messages=2,
                limit=50,
            )
        ),
    )


@pytest.mark.benchmark
def test_bench_get_many_100(benchmark: BenchmarkFixture, bench_db_5k: Path) -> None:
    """get_many() with 100 IDs — parallel batch fetch cost."""
    ids = [f"bench-conv-{i:05d}" for i in range(100)]
    benchmark_store_call(
        benchmark,
        bench_db_5k,
        lambda store: store.repository.get_many(ids),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [100, 500, 1000])
def test_bench_write_session_messages(benchmark: BenchmarkFixture, tmp_path: Path, n: int) -> None:
    """Write a session of N messages through the production archive writer."""
    db_path = tmp_path / f"save_bench_{n}.db"
    with open_bench_store(db_path) as store:
        conv = make_session(
            session_id="bench-save-conv",
            source_name="chatgpt",
            provider_session_id="bench-save-conv",
            title="Save Bench",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash=_make_hash("save-conv"),
        )
        msgs = [
            make_message(
                message_id=f"bench-save-conv-m{j}",
                session_id="bench-save-conv",
                role="user" if j % 2 == 0 else "assistant",
                text=f"Benchmark message {j} with some content words",
                timestamp=f"2025-01-01T00:00:{j % 60:02d}Z",
                content_hash=_make_hash(f"save-msg-{j}"),
                source_name="chatgpt",
                word_count=7,
            )
            for j in range(n)
        ]

        benchmark(
            lambda: store.run(
                save_current_archive_records(store.repository, session=conv, messages=msgs, attachments=[])
            )
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_msgs", [100, 500])
def test_bench_write_session_with_blocks(benchmark: BenchmarkFixture, tmp_path: Path, n_msgs: int) -> None:
    """Write a session of N messages (5 blocks each) through the production writer."""
    db_path = tmp_path / f"blocks_bench_{n_msgs}.db"
    _BLOCK_TYPES = ["tool_use", "tool_use", "tool_use", "tool_result", "thinking"]
    with open_bench_store(db_path) as store:
        conv = make_session(
            session_id="bench-blocks-conv",
            source_name="chatgpt",
            provider_session_id="bench-blocks-conv",
            title="Blocks Bench",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash=_make_hash("blocks-conv"),
        )
        msgs = [
            make_message(
                message_id=f"bench-blocks-conv-m{j}",
                session_id="bench-blocks-conv",
                role="user" if j % 2 == 0 else "assistant",
                text=f"Block bench message {j}",
                timestamp=f"2025-01-01T00:00:{j % 60:02d}Z",
                content_hash=_make_hash(f"blocks-msg-{j}"),
                source_name="chatgpt",
                word_count=4,
                blocks=[
                    make_content_block(
                        message_id=f"bench-blocks-conv-m{j}",
                        session_id="bench-blocks-conv",
                        block_index=k,
                        block_type=_BLOCK_TYPES[k],
                    )
                    for k in range(5)
                ],
            )
            for j in range(n_msgs)
        ]

        benchmark(
            lambda: store.run(
                save_current_archive_records(store.repository, session=conv, messages=msgs, attachments=[])
            )
        )
