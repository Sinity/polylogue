"""Shared fixtures for benchmark tests.

Uses the SyntheticCorpus generator to produce schema-conformant data
matching the real archive's distribution rather than hand-rolled word pools.

Generated DBs are cached in a temp directory per session to avoid
regeneration overhead dominating benchmark measurements.

Usage:
    devtools benchmark-campaign list
    devtools benchmark-campaign run search-filters
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord

# ---------------------------------------------------------------------------
# Real archive distribution (from actual data analysis):
#   1-10 msgs:     3280 convs (49%)
#   11-50 msgs:    2117 convs (32%)
#   51-200 msgs:    540 convs (8%)
#   201-1000 msgs:  319 convs (5%)
#   1001-5000 msgs: 297 convs (4%)
#   5000+ msgs:      97 convs (1.5%)
#
# Content blocks: 0.67 blocks/msg
# Provider split: claude-code 88% of messages, codex 10%, chatgpt/claude-ai/gemini 2%
# ---------------------------------------------------------------------------

PROVIDERS = ["chatgpt", "claude-ai", "claude-code", "codex", "gemini"]
PROVIDER_WEIGHTS = [0.02, 0.01, 0.80, 0.15, 0.02]

# Message count ranges matching real distribution
_CONVERSATION_PROFILES = [
    # (message_range, proportion)
    (range(2, 11), 0.49),
    (range(11, 51), 0.32),
    (range(51, 201), 0.08),
    (range(201, 501), 0.05),  # cap at 500 for benchmarks (real goes to 1000)
    (range(501, 2001), 0.04),  # cap at 2000 (real goes to 5000)
    (range(2001, 5001), 0.02),  # small fraction of large conversations
]

# Realistic tool names and content
_TOOL_NAMES = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent", "WebFetch", "WebSearch"]
_FILE_PATHS = [
    "/workspace/polylogue/polylogue/storage/repository.py",
    "/workspace/polylogue/polylogue/lib/models.py",
    "/workspace/polylogue/tests/unit/core/test_models.py",
    "/workspace/polylogue/polylogue/cli/commands/check.py",
    "/workspace/polylogue/polylogue/schemas/runtime_registry.py",
]

# Semantic types matching real distribution (~67% of messages have blocks)
SEMANTIC_CYCLE = [
    "file_read",
    "file_write",
    "file_edit",
    "git",
    "shell",
    "search",
    "subagent",
    None,
    None,
    None,
    None,
    None,
]


def _make_content_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def _pick_message_count(rng: random.Random) -> int:
    """Pick a message count matching the real archive distribution."""
    r = rng.random()
    cumulative = 0.0
    for msg_range, proportion in _CONVERSATION_PROFILES:
        cumulative += proportion
        if r <= cumulative:
            return rng.choice(msg_range)
    return rng.randint(5, 20)  # fallback


def _pick_provider(rng: random.Random) -> str:
    """Pick a provider matching real archive distribution."""
    return rng.choices(PROVIDERS, weights=PROVIDER_WEIGHTS, k=1)[0]


def _generate_realistic_text(rng: random.Random, role: str, msg_index: int) -> str:
    """Generate text that approximates real message lengths and content."""
    if role == "user":
        # User messages: typically shorter, 10-200 words
        length = rng.randint(10, 200)
        words = [
            "please",
            "can",
            "you",
            "help",
            "with",
            "this",
            "code",
            "error",
            "function",
            "fix",
            "implement",
            "the",
            "following",
            "test",
            "check",
            "file",
            "update",
            "add",
            "remove",
            "refactor",
            "debug",
            "analyze",
        ]
    else:
        # Assistant messages: typically longer, 50-2000 words
        length = rng.randint(50, 500)
        words = [
            "I'll",
            "analyze",
            "the",
            "code",
            "and",
            "implement",
            "this",
            "function",
            "returns",
            "value",
            "error",
            "handling",
            "pattern",
            "let",
            "me",
            "check",
            "file",
            "module",
            "class",
            "method",
            "import",
            "from",
            "def",
            "async",
            "await",
            "return",
            "if",
            "for",
            "in",
            "not",
            "None",
            "True",
            "False",
            "self",
        ]
    return " ".join(rng.choices(words, k=length))


def _make_content_block(rng: random.Random, msg_id: str, conv_id: str, block_index: int) -> ContentBlockRecord | None:
    """Generate a content block matching real distribution."""
    # ~67% of assistant messages have at least one block
    sem = SEMANTIC_CYCLE[block_index % len(SEMANTIC_CYCLE)]
    if sem is None:
        # Thinking block
        if rng.random() < 0.15:
            return ContentBlockRecord(
                block_id=ContentBlockRecord.make_id(msg_id, block_index),
                message_id=msg_id,
                conversation_id=conv_id,
                block_index=block_index,
                type="thinking",
                text=_generate_realistic_text(rng, "assistant", block_index)[:200],
            )
        return None

    tool_name = rng.choice(_TOOL_NAMES)
    return ContentBlockRecord(
        block_id=ContentBlockRecord.make_id(msg_id, block_index),
        message_id=msg_id,
        conversation_id=conv_id,
        block_index=block_index,
        type="tool_use",
        tool_name=tool_name,
        semantic_type=sem,
        tool_input=json.dumps({"file_path": rng.choice(_FILE_PATHS)})
        if sem in ("file_read", "file_write", "file_edit")
        else None,
    )


async def _seed_realistic_db(db_path: Path, target_messages: int, seed: int = 42) -> dict[str, int]:
    """Seed benchmark DB with realistic distribution data.

    Returns stats dict for verification.
    """
    rng = random.Random(seed)
    backend = SQLiteBackend(db_path=db_path)

    conv_records: list[ConversationRecord] = []
    all_msgs: list[MessageRecord] = []
    all_blocks: list[ContentBlockRecord] = []
    total_msgs = 0

    conv_index = 0
    while total_msgs < target_messages:
        provider = _pick_provider(rng)
        msg_count = _pick_message_count(rng)
        # Don't overshoot target too much
        if total_msgs + msg_count > target_messages * 1.1 and total_msgs > target_messages * 0.5:
            msg_count = min(msg_count, max(1, target_messages - total_msgs))

        conv_id = f"bench-conv-{conv_index:05d}"
        conv_records.append(
            ConversationRecord(
                conversation_id=conv_id,
                provider_name=provider,
                provider_conversation_id=f"prov-{conv_index:05d}",
                title=f"Session {conv_index}: {_generate_realistic_text(rng, 'user', 0)[:60]}",
                created_at=f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00Z",
                updated_at=f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00Z",
                content_hash=_make_content_hash(f"conv-{conv_index}"),
            )
        )

        for j in range(msg_count):
            role = "user" if j % 2 == 0 else "assistant"
            msg_id = f"{conv_id}-m{j}"
            text = _generate_realistic_text(rng, role, j)
            has_tool = 0
            has_think = 0

            # Generate content blocks for assistant messages
            if role == "assistant" and rng.random() < 0.67:
                block_count = rng.randint(1, 4)
                for bi in range(block_count):
                    block = _make_content_block(rng, msg_id, conv_id, bi)
                    if block is not None:
                        all_blocks.append(block)
                        if block.type == "tool_use":
                            has_tool = 1
                        elif block.type == "thinking":
                            has_think = 1

            all_msgs.append(
                MessageRecord(
                    message_id=msg_id,
                    conversation_id=conv_id,
                    role=role,
                    text=text,
                    timestamp=f"2025-01-01T{(j * 3) % 24:02d}:{(j * 7) % 60:02d}:00Z",
                    content_hash=_make_content_hash(f"msg-{conv_index}-{j}"),
                    provider_name=provider,
                    word_count=len(text.split()),
                    has_tool_use=has_tool,
                    has_thinking=has_think,
                )
            )
            total_msgs += 1

        conv_index += 1

    # Populate DB
    msgs_by_conv: dict[str, list[MessageRecord]] = defaultdict(list)
    for msg in all_msgs:
        msgs_by_conv[msg.conversation_id].append(msg)

    provider_by_cid = {r.conversation_id: r.provider_name for r in conv_records}

    async with backend.bulk_connection():
        for i, record in enumerate(conv_records, start=1):
            await backend.save_conversation_record(record)
            if i % 500 == 0:
                await backend.bulk_flush()

        await backend.save_messages(all_msgs)
        await backend.save_content_blocks(all_blocks)

        for i, (cid, msgs) in enumerate(msgs_by_conv.items(), start=1):
            await backend.upsert_conversation_stats(cid, provider_by_cid[cid], msgs)
            if i % 500 == 0:
                await backend.bulk_flush()

    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return {
        "conversations": len(conv_records),
        "messages": total_msgs,
        "content_blocks": len(all_blocks),
    }


# ---------------------------------------------------------------------------
# Session-scoped fixtures (generated once, reused across all benchmarks)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_db_1k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~1K messages: matches real distribution (many small, few large convs)."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_1k.db"
    stats = asyncio.run(_seed_realistic_db(db_path, target_messages=1000))
    print(f"\nbench_db_1k: {stats}")
    return db_path


@pytest.fixture(scope="session")
def bench_db_5k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~5K messages: medium-scale with realistic distribution."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_5k.db"
    stats = asyncio.run(_seed_realistic_db(db_path, target_messages=5000))
    print(f"\nbench_db_5k: {stats}")
    return db_path


@pytest.fixture(scope="session")
def bench_db_10k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~10K messages: large-scale with realistic distribution."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_10k.db"
    stats = asyncio.run(_seed_realistic_db(db_path, target_messages=10000))
    print(f"\nbench_db_10k: {stats}")
    return db_path


@pytest.fixture(scope="session")
def bench_db_50k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~50K messages: stress-scale with realistic distribution."""
    db_path = tmp_path_factory.mktemp("bench") / "bench_50k.db"
    stats = asyncio.run(_seed_realistic_db(db_path, target_messages=50000))
    print(f"\nbench_db_50k: {stats}")
    return db_path
