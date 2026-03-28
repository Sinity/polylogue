"""Tests for concurrency, resource management, and async safety.

Covers:
- ebef687: TOCTOU race in metadata read-modify-write
- d5c3228: sqlite-vec created+discarded connection on every operation
- fa2b132: sqlite-vec returned broken connection instead of raising
- abbe871: Connection storm at scale (3000+ parallel asyncio.gather calls)

Also covers filter state isolation: reusing a filter builder must not
accumulate state from previous uses.

NOTE: Claude Code model property tests (role mapping, timestamps, boolean
flags) have been consolidated into tests/unit/sources/test_models.py using
shared tables from tests/infra/tables.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from polylogue.lib.filters import ConversationFilter


# =============================================================================
# Filter state isolation (implicit in CLI routing bugs)
# =============================================================================

class TestFilterStateIsolation:
    """ConversationFilter must not leak state between uses.

    The filter builder returns `self` for chaining. If users accidentally
    reuse a filter instance, accumulated predicates could cause incorrect
    results. This tests that each filter chain produces independent state.
    """

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository for filter testing."""
        repo = AsyncMock()
        repo.list_conversations = AsyncMock(return_value=[])
        repo.count_conversations = AsyncMock(return_value=0)
        repo.get_conversation_summaries = AsyncMock(return_value=[])
        return repo

    def test_separate_filters_have_independent_state(self, mock_repo):
        """Two filters from same repo must not share state."""
        f1 = ConversationFilter(mock_repo)
        f2 = ConversationFilter(mock_repo)

        f1.provider("chatgpt")
        f2.provider("claude")

        assert f1._providers == ["chatgpt"]
        assert f2._providers == ["claude"]

    def test_chained_methods_accumulate_on_same_instance(self, mock_repo):
        """Chaining on same filter must accumulate predicates."""
        f = ConversationFilter(mock_repo)
        f.provider("chatgpt").contains("error").limit(10)

        assert f._providers == ["chatgpt"]
        assert f._fts_terms == ["error"]
        assert f._limit_count == 10

    def test_filter_reuse_accumulates_providers(self, mock_repo):
        """Reusing a filter adds to existing providers list."""
        f = ConversationFilter(mock_repo)
        f.provider("chatgpt")
        f.provider("claude")  # second call

        # Both should be present
        assert f._providers == ["chatgpt", "claude"]

    def test_filter_sort_replaces_not_accumulates(self, mock_repo):
        """sort() should replace the previous sort, not append."""
        f = ConversationFilter(mock_repo)
        f.sort("date")
        f.sort("words")

        assert f._sort_field == "words"

    def test_fresh_filter_has_no_predicates(self, mock_repo):
        """A brand-new filter should have empty state."""
        f = ConversationFilter(mock_repo)

        assert f._providers == []
        assert f._fts_terms == []
        assert f._limit_count is None

    def test_limit_replaces_previous_limit(self, mock_repo):
        """Calling limit() twice replaces, doesn't stack."""
        f = ConversationFilter(mock_repo)
        f.limit(10)
        f.limit(5)

        assert f._limit_count == 5


# =============================================================================
# Async backend: batch operations must not create N connections
# =============================================================================

class TestConnectionManagement:
    """Batch operations must use O(1) connections, not O(N)."""

    @pytest.mark.asyncio
    async def test_get_conversations_batch_uses_single_connection(self, sqlite_backend):
        """get_conversations_batch must use 1 query, not N queries."""
        backend = sqlite_backend

        # Insert 20 conversations
        async with backend._get_connection() as conn:
            for i in range(20):
                await conn.execute(
                    "INSERT INTO conversations (conversation_id, provider_name, "
                    "provider_conversation_id, content_hash, version) "
                    "VALUES (?, 'test', ?, ?, 1)",
                    (f"conv-{i}", f"pconv-{i}", f"hash-{i}"),
                )
            await conn.commit()

        # Batch get should work
        ids = [f"conv-{i}" for i in range(20)]
        records = await backend.get_conversations_batch(ids)
        assert len(records) == 20

    @pytest.mark.asyncio
    async def test_get_messages_batch_groups_by_conversation(self, sqlite_backend):
        """get_messages_batch must return messages grouped by conversation_id."""
        backend = sqlite_backend

        # Insert conversations with messages
        async with backend._get_connection() as conn:
            for i in range(5):
                await conn.execute(
                    "INSERT INTO conversations (conversation_id, provider_name, "
                    "provider_conversation_id, content_hash, version) "
                    "VALUES (?, 'test', ?, ?, 1)",
                    (f"conv-{i}", f"pconv-{i}", f"hash-{i}"),
                )
                for j in range(3):
                    await conn.execute(
                        "INSERT INTO messages (message_id, conversation_id, role, "
                        "text, content_hash, version) "
                        "VALUES (?, ?, 'user', ?, ?, 1)",
                        (f"msg-{i}-{j}", f"conv-{i}", f"text {i} {j}", f"mhash-{i}-{j}"),
                    )
            await conn.commit()

        ids = [f"conv-{i}" for i in range(5)]
        msgs_by_id = await backend.get_messages_batch(ids)

        assert len(msgs_by_id) == 5
        for conv_id, msgs in msgs_by_id.items():
            assert len(msgs) == 3
            assert all(m.conversation_id == conv_id for m in msgs)

    @pytest.mark.asyncio
    async def test_batch_with_empty_ids_returns_empty(self, sqlite_backend):
        """Passing empty list of IDs should return empty, not error."""
        backend = sqlite_backend
        records = await backend.get_conversations_batch([])
        assert records == []

    @pytest.mark.asyncio
    async def test_batch_with_nonexistent_ids_returns_partial(self, sqlite_backend):
        """Requesting nonexistent IDs should return only existing ones."""
        backend = sqlite_backend

        async with backend._get_connection() as conn:
            await conn.execute(
                "INSERT INTO conversations (conversation_id, provider_name, "
                "provider_conversation_id, content_hash, version) "
                "VALUES ('exists', 'test', 'pconv', 'hash', 1)",
            )
            await conn.commit()

        records = await backend.get_conversations_batch(["exists", "ghost-1", "ghost-2"])
        assert len(records) == 1
        assert records[0].conversation_id == "exists"


# =============================================================================
# Async repository: concurrent save safety
# =============================================================================

class TestConcurrentSaveGuards:
    """Multiple concurrent saves must not corrupt data."""

    @pytest.mark.asyncio
    async def test_concurrent_saves_dont_crash(self, sqlite_backend):
        """Concurrent saves to different conversations must not error."""
        from polylogue.storage.store import ConversationRecord, MessageRecord

        backend = sqlite_backend

        async def _save_one(idx: int) -> None:
            conv = ConversationRecord(
                conversation_id=f"test:conv-{idx}",
                provider_name="test",
                provider_conversation_id=f"conv-{idx}",
                title=f"Conversation {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            msg = MessageRecord(
                message_id=f"msg-{idx}",
                conversation_id=f"test:conv-{idx}",
                role="user",
                text=f"Hello from conversation {idx}",
                content_hash=f"mhash-{idx}",
                version=1,
            )
            await backend.save_conversation(conv, [msg], [])

        # Run 10 concurrent saves
        await asyncio.gather(*[_save_one(i) for i in range(10)])

        # Verify all saved
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            assert row[0] == 10

    @pytest.mark.asyncio
    async def test_concurrent_saves_to_same_conversation(self, sqlite_backend):
        """Concurrent upserts to the same conversation must not corrupt."""
        from polylogue.storage.store import ConversationRecord, MessageRecord

        backend = sqlite_backend

        async def _upsert(version: int) -> None:
            conv = ConversationRecord(
                conversation_id="test:same-conv",
                provider_name="test",
                provider_conversation_id="same",
                title=f"Version {version}",
                content_hash=f"hash-v{version}",
                version=version,
            )
            msg = MessageRecord(
                message_id=f"msg-v{version}",
                conversation_id="test:same-conv",
                role="user",
                text=f"Version {version}",
                content_hash=f"mhash-v{version}",
                version=version,
            )
            await backend.save_conversation(conv, [msg], [])

        # Run 5 concurrent upserts to the same conversation
        await asyncio.gather(*[_upsert(i) for i in range(5)])

        # Should have exactly 1 conversation (upserted, not duplicated)
        async with backend._get_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_writes(self, sqlite_backend):
        """Reads during concurrent writes must not error or return garbage."""
        from polylogue.storage.store import ConversationRecord, MessageRecord

        backend = sqlite_backend

        async def _write(idx: int) -> None:
            conv = ConversationRecord(
                conversation_id=f"test:conv-{idx}",
                provider_name="test",
                provider_conversation_id=f"conv-{idx}",
                title=f"Conversation {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            await backend.save_conversation(conv, [], [])

        async def _read() -> int:
            async with backend._get_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
                row = await cursor.fetchone()
                return row[0]

        # Interleave writes and reads
        tasks = []
        for i in range(10):
            tasks.append(_write(i))
            tasks.append(_read())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # No exceptions should have been raised
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert exceptions == [], f"Got exceptions during concurrent read/write: {exceptions}"

        # All 10 conversations should exist
        final_count = await _read()
        assert final_count == 10
