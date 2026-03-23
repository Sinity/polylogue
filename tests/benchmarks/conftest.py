"""Shared fixtures for benchmark tests.

Session-scoped databases are seeded once per session to avoid
fixture overhead dominating benchmark measurements.

Usage:
    nix develop -c python -m devtools.benchmark_campaign list
    nix develop -c python -m devtools.benchmark_campaign run search-filters
    nix develop -c python -m devtools.benchmark_campaign compare \\
        docs/benchmark-campaigns/<baseline>.json docs/benchmark-campaigns/<candidate>.json
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord

PROVIDERS = ["chatgpt", "claude-ai", "claude-code", "gemini"]
# Words used to make FTS5 searches meaningful
_WORD_POOL = [
    "python", "function", "error", "test", "data", "model", "training",
    "async", "await", "database", "query", "result", "analysis", "code",
    "git", "commit", "branch", "merge", "review", "performance",
]
# Semantic types cycling through values that semantic filters recognise.
# None entries simulate "plain tool_use with no semantic classification" (~50%).
SEMANTIC_CYCLE = ['file_read', 'file_write', 'git', 'shell', 'subagent', None, None, None]


def _make_content_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


async def _seed_bench_db(db_path: Path, conv_count: int, msgs_per_conv: int) -> None:
    """Seed benchmark DB with realistic mixed data.

    Populates:
    - conversations table
    - messages table (with has_tool_use / has_thinking flags)
    - content_blocks table (tool_use + thinking blocks, with semantic_type)
    - conversation_stats table (via upsert_conversation_stats)
    - FTS5 index (rebuild_index at end)
    """
    backend = SQLiteBackend(db_path=db_path)

    conv_records = [
        ConversationRecord(
            conversation_id=f"bench-conv-{i:05d}",
            provider_name=PROVIDERS[i % len(PROVIDERS)],
            provider_conversation_id=f"prov-bench-{i:05d}",
            title=f"Benchmark Conversation {i}: {_WORD_POOL[i % len(_WORD_POOL)]} {_WORD_POOL[(i + 3) % len(_WORD_POOL)]}",
            created_at=f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T00:00:00Z",
            updated_at=f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T00:00:00Z",
            content_hash=_make_content_hash(f"conv-{i}"),
        )
        for i in range(conv_count)
    ]
    all_msgs: list[MessageRecord] = []
    for i in range(conv_count):
        for j in range(msgs_per_conv):
            # ~30% messages have tool_use, ~10% have thinking
            has_tool = 1 if (i * msgs_per_conv + j) % 3 == 0 else 0
            has_think = 1 if (i * msgs_per_conv + j) % 10 == 0 else 0
            words = _WORD_POOL[(i + j) % len(_WORD_POOL)]
            all_msgs.append(MessageRecord(
                message_id=f"bench-conv-{i:05d}-m{j}",
                conversation_id=f"bench-conv-{i:05d}",
                role="user" if j % 2 == 0 else "assistant",
                text=f"{words} analysis result for conversation {i} message {j} with data processing",
                timestamp=f"2025-01-01T{j:02d}:00:00Z",
                content_hash=_make_content_hash(f"msg-{i}-{j}"),
                provider_name=PROVIDERS[i % len(PROVIDERS)],
                word_count=10,
                has_tool_use=has_tool,
                has_thinking=has_think,
            ))
    # --- Seed content_blocks ---
    # tool_use messages get a block with a cycling semantic_type;
    # thinking messages get a 'thinking' type block.
    all_blocks: list[ContentBlockRecord] = []
    for idx, msg in enumerate(all_msgs):
        if msg.has_tool_use:
            sem = SEMANTIC_CYCLE[idx % len(SEMANTIC_CYCLE)]
            block_id = ContentBlockRecord.make_id(msg.message_id, 0)
            all_blocks.append(ContentBlockRecord(
                block_id=block_id,
                message_id=msg.message_id,
                conversation_id=msg.conversation_id,
                block_index=0,
                type='tool_use',
                semantic_type=sem,
            ))
        if msg.has_thinking:
            block_id = ContentBlockRecord.make_id(msg.message_id, 1)
            all_blocks.append(ContentBlockRecord(
                block_id=block_id,
                message_id=msg.message_id,
                conversation_id=msg.conversation_id,
                block_index=1,
                type='thinking',
            ))
    # --- Populate conversation_stats ---
    msgs_by_conv: dict[str, list[MessageRecord]] = defaultdict(list)
    for msg in all_msgs:
        msgs_by_conv[msg.conversation_id].append(msg)

    provider_by_cid = {r.conversation_id: r.provider_name for r in conv_records}

    async with backend.bulk_connection():
        for index, record in enumerate(conv_records, start=1):
            await backend.save_conversation_record(record)
            if index % 500 == 0:
                await backend.bulk_flush()

        await backend.save_messages(all_msgs)
        await backend.save_content_blocks(all_blocks)

        for index, (cid, msgs) in enumerate(msgs_by_conv.items(), start=1):
            await backend.upsert_conversation_stats(cid, provider_by_cid[cid], msgs)
            if index % 500 == 0:
                await backend.bulk_flush()

    # --- Rebuild FTS5 index (session-level cost, not per-benchmark) ---
    with open_connection(db_path) as conn:
        rebuild_index(conn)


@pytest.fixture(scope="session")
def bench_db_1k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """1k-message DB: 100 convos × 10 msgs. Mixed providers, tool_use, thinking."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_1k.db"
    asyncio.run(_seed_bench_db(db_path, conv_count=100, msgs_per_conv=10))
    return db_path


@pytest.fixture(scope="session")
def bench_db_5k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """5k-message DB: 500 convos × 10 msgs."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_5k.db"
    asyncio.run(_seed_bench_db(db_path, conv_count=500, msgs_per_conv=10))
    return db_path


@pytest.fixture(scope="session")
def bench_db_10k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """10k-message DB: 2000 convos × 5 msgs. Wide distribution."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_10k.db"
    asyncio.run(_seed_bench_db(db_path, conv_count=2000, msgs_per_conv=5))
    return db_path
