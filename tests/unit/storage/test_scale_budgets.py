"""Scale-aware performance budget tests.

Ensures that key operations stay within expected time bounds
at moderate scale. Uses synthetic data for reproducibility.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, MessageRecord


BUDGET_CONV_COUNT = 1000
BUDGET_MSGS_PER_CONV = 5


async def _seed_budget_conversations(
    backend: SQLiteBackend, count: int, msgs_per_conv: int = 5
) -> list[str]:
    """Seed a database with conversations and messages for budget tests.

    Uses bulk_connection for efficient inserts at scale.
    """
    ids = [f"budget-conv-{i:05d}" for i in range(count)]
    providers = ["chatgpt", "claude-ai", "claude-code", "gemini"]

    async with backend.bulk_connection():
        for i in range(count):
            conv = ConversationRecord(
                conversation_id=ids[i],
                provider_name=providers[i % len(providers)],
                provider_conversation_id=f"prov-{ids[i]}",
                title=f"Budget Test Conversation {i} with searchable content",
                created_at=f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T{i % 24:02d}:00:00Z",
                updated_at=f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T{i % 24:02d}:30:00Z",
                content_hash=f"hash-{ids[i]}",
            )
            await backend.save_conversation_record(conv)

            messages = [
                MessageRecord(
                    message_id=f"{ids[i]}-m{j}",
                    conversation_id=ids[i],
                    role="user" if j % 2 == 0 else "assistant",
                    text=f"Message {j} in conversation {i} about topic alpha beta gamma",
                    content_hash=f"hash-{ids[i]}-m{j}",
                    provider_name=providers[i % len(providers)],
                    word_count=10,
                )
                for j in range(msgs_per_conv)
            ]
            await backend.save_messages(messages)

            # Flush every 200 conversations to avoid large transactions
            if i > 0 and i % 200 == 0:
                await backend.bulk_flush()

    return ids


@pytest.fixture(scope="module")
def budget_db(tmp_path_factory):
    """Module-scoped fixture that seeds a 1000-conversation database.

    Reused across all budget tests in this module for efficiency.
    """
    tmp_dir = tmp_path_factory.mktemp("budget_scale")
    db_path = tmp_dir / "budget.db"
    backend = SQLiteBackend(db_path=db_path)

    async def _setup():
        ids = await _seed_budget_conversations(
            backend, BUDGET_CONV_COUNT, BUDGET_MSGS_PER_CONV
        )
        await backend.close()
        return ids

    ids = asyncio.run(_setup())

    # Build FTS index
    with open_connection(db_path) as conn:
        rebuild_index(conn)

    return db_path, ids


@pytest.mark.slow
@pytest.mark.scale("medium")
class TestScaleBudgets:
    """Performance budget tests at 1000-conversation scale.

    Each test asserts that a key operation completes within an acceptable
    time budget. Budgets are conservative (5-20x typical workstation times)
    to avoid flaky CI failures.
    """

    async def test_list_conversations_under_budget(self, budget_db):
        """1000 conversations, list should complete in < 2s."""
        db_path, ids = budget_db
        backend = SQLiteBackend(db_path=db_path)
        try:
            t0 = time.monotonic()
            results = await backend.list_conversations(limit=100)
            elapsed = time.monotonic() - t0

            assert len(results) == 100
            assert elapsed < 2.0, (
                f"list_conversations(limit=100) took {elapsed:.3f}s (budget: 2.0s)"
            )
        finally:
            await backend.close()

    async def test_fts_search_under_budget(self, budget_db):
        """Search across 1000 conversations, should complete in < 1s."""
        db_path, ids = budget_db
        backend = SQLiteBackend(db_path=db_path)
        try:
            repo = ConversationRepository(backend=backend)
            t0 = time.monotonic()
            results = await repo.search_summaries("Message", limit=50)
            elapsed = time.monotonic() - t0

            assert elapsed < 1.0, (
                f"FTS search took {elapsed:.3f}s (budget: 1.0s)"
            )
        finally:
            await backend.close()

    async def test_filter_pushdown_under_budget(self, budget_db):
        """Filter by provider/date across 1000 conversations, < 1s."""
        db_path, ids = budget_db
        backend = SQLiteBackend(db_path=db_path)
        try:
            # Provider filter
            t0 = time.monotonic()
            results = await backend.list_conversations(
                provider="chatgpt", limit=100
            )
            elapsed_provider = time.monotonic() - t0

            # Date range filter
            t0 = time.monotonic()
            results = await backend.list_conversations(
                since="2025-03-01", until="2025-06-30", limit=100
            )
            elapsed_date = time.monotonic() - t0

            # Combined filter
            t0 = time.monotonic()
            results = await backend.list_conversations(
                provider="claude-ai",
                since="2025-01-01",
                until="2025-12-31",
                limit=100,
            )
            elapsed_combined = time.monotonic() - t0

            max_elapsed = max(elapsed_provider, elapsed_date, elapsed_combined)
            assert max_elapsed < 1.0, (
                f"Filter pushdown took {max_elapsed:.3f}s "
                f"(provider={elapsed_provider:.3f}s, "
                f"date={elapsed_date:.3f}s, "
                f"combined={elapsed_combined:.3f}s, "
                f"budget: 1.0s)"
            )
        finally:
            await backend.close()

    async def test_batch_insert_under_budget(self, tmp_path):
        """Insert 100 conversations with messages, should complete in < 5s."""
        db_path = tmp_path / "insert_budget.db"
        backend = SQLiteBackend(db_path=db_path)

        t0 = time.monotonic()
        try:
            ids = await _seed_budget_conversations(backend, 100, msgs_per_conv=5)
            elapsed = time.monotonic() - t0

            assert len(ids) == 100
            assert elapsed < 5.0, (
                f"Batch insert of 100 conversations took {elapsed:.3f}s (budget: 5.0s)"
            )
        finally:
            await backend.close()
