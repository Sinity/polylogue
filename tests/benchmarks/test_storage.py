"""Storage benchmark tests.

Covers: batch inserts, list queries (unfiltered, provider, stats-join,
semantic filter), batch get operations.

Run with:
    pytest tests/benchmarks/test_storage.py --benchmark-enable -p no:xdist -v
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord


def _make_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


@pytest.mark.benchmark
def test_bench_list_conversations_no_filter(benchmark, bench_db_5k: Path) -> None:
    """list_conversations(limit=50) on 5k-message DB — baseline query cost."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(backend.list_conversations(limit=50)))
    loop.close()


@pytest.mark.benchmark
def test_bench_list_conversations_provider_filter(benchmark, bench_db_5k: Path) -> None:
    """list with provider=chatgpt — tests simple WHERE on indexed column."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(backend.list_conversations(provider="chatgpt", limit=50)))
    loop.close()


@pytest.mark.benchmark
def test_bench_list_conversations_has_tool_use(benchmark, bench_db_5k: Path) -> None:
    """list with has_tool_use=True — tests stats LEFT JOIN path."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(backend.list_conversations(has_tool_use=True, limit=50)))
    loop.close()


@pytest.mark.benchmark
def test_bench_list_conversations_semantic_filter(benchmark, bench_db_5k: Path) -> None:
    """list with has_file_ops=True — tests EXISTS subquery path (schema v3)."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(backend.list_conversations(has_file_ops=True, limit=50)))
    loop.close()


@pytest.mark.benchmark
def test_bench_list_conversations_combined_filter(benchmark, bench_db_10k: Path) -> None:
    """provider + has_tool_use + min_messages — combined filter stack."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_10k)
    loop.run_until_complete(backend._ensure_schema_once())

    benchmark(lambda: loop.run_until_complete(
        backend.list_conversations(provider="claude", has_tool_use=True, min_messages=2, limit=50)
    ))
    loop.close()


@pytest.mark.benchmark
def test_bench_get_many_100(benchmark, bench_db_5k: Path) -> None:
    """get_many() with 100 IDs — parallel batch fetch cost."""
    loop = asyncio.new_event_loop()
    backend = SQLiteBackend(db_path=bench_db_5k)
    loop.run_until_complete(backend._ensure_schema_once())
    repo = ConversationRepository(backend=backend)
    ids = [f"bench-conv-{i:05d}" for i in range(100)]

    benchmark(lambda: loop.run_until_complete(repo.get_many(ids)))
    loop.close()


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [100, 500, 1000])
def test_bench_save_messages_batch(benchmark, tmp_path: Path, n: int) -> None:
    """save_messages() batch insert — measures executemany throughput."""
    loop = asyncio.new_event_loop()
    db_path = tmp_path / f"save_bench_{n}.db"
    backend = SQLiteBackend(db_path=db_path)
    # Seed conversation first
    conv = ConversationRecord(
        conversation_id="bench-save-conv",
        provider_name="chatgpt",
        provider_conversation_id="prov-save",
        title="Save Bench",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        content_hash=_make_hash("save-conv"),
    )
    loop.run_until_complete(backend.save_conversation_record(conv))

    msgs = [
        MessageRecord(
            message_id=f"bench-save-conv-m{j}",
            conversation_id="bench-save-conv",
            role="user" if j % 2 == 0 else "assistant",
            text=f"Benchmark message {j} with some content words",
            timestamp=f"2025-01-01T00:00:{j % 60:02d}Z",
            content_hash=_make_hash(f"save-msg-{j}"),
            provider_name="chatgpt",
            word_count=7,
        )
        for j in range(n)
    ]

    # Re-use same messages each iteration (upsert = idempotent)
    benchmark(lambda: loop.run_until_complete(backend.save_messages(msgs)))
    loop.close()


@pytest.mark.benchmark
@pytest.mark.parametrize("n_msgs", [100, 500])
def test_bench_save_content_blocks(benchmark, tmp_path: Path, n_msgs: int) -> None:
    """save_content_blocks() — 5 blocks per message (tool_use + thinking mix)."""
    loop = asyncio.new_event_loop()
    db_path = tmp_path / f"blocks_bench_{n_msgs}.db"
    backend = SQLiteBackend(db_path=db_path)
    conv = ConversationRecord(
        conversation_id="bench-blocks-conv",
        provider_name="chatgpt",
        provider_conversation_id="prov-blocks",
        title="Blocks Bench",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        content_hash=_make_hash("blocks-conv"),
    )
    loop.run_until_complete(backend.save_conversation_record(conv))
    msgs = [
        MessageRecord(
            message_id=f"bench-blocks-conv-m{j}",
            conversation_id="bench-blocks-conv",
            role="user" if j % 2 == 0 else "assistant",
            text=f"Block bench message {j}",
            timestamp=f"2025-01-01T00:00:{j % 60:02d}Z",
            content_hash=_make_hash(f"blocks-msg-{j}"),
            provider_name="chatgpt",
            word_count=4,
            has_tool_use=1,
            has_thinking=1,
        )
        for j in range(n_msgs)
    ]
    loop.run_until_complete(backend.save_messages(msgs))

    # 3 tool_use + 1 tool_result + 1 thinking per message = 5 blocks each
    _BLOCK_TYPES = ["tool_use", "tool_use", "tool_use", "tool_result", "thinking"]
    blocks = [
        ContentBlockRecord(
            block_id=ContentBlockRecord.make_id(f"bench-blocks-conv-m{j}", k),
            message_id=f"bench-blocks-conv-m{j}",
            conversation_id="bench-blocks-conv",
            block_index=k,
            type=_BLOCK_TYPES[k],
        )
        for j in range(n_msgs)
        for k in range(5)
    ]

    # Upsert semantics — idempotent across benchmark iterations
    benchmark(lambda: loop.run_until_complete(backend.save_content_blocks(blocks)))
    loop.close()
