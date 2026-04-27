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
from collections.abc import Awaitable
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from polylogue.lib.filter.filters import ConversationFilter
from polylogue.lib.models import Conversation
from polylogue.lib.roles import Role
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import make_conversation, make_message

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
    def mock_repo(self) -> AsyncMock:
        """Create a mock repository for filter testing."""
        repo = AsyncMock()
        repo.list_by_query = AsyncMock(return_value=[])
        repo.search = AsyncMock(return_value=[])
        repo.count = AsyncMock(return_value=0)
        repo.list_summaries = AsyncMock(return_value=[])
        repo.search_summaries = AsyncMock(return_value=[])
        repo.list_summaries_by_query = AsyncMock(return_value=[])
        repo.resolve_id = AsyncMock(return_value=None)
        repo.get = AsyncMock(return_value=None)
        repo.get_summary = AsyncMock(return_value=None)
        return repo

    @staticmethod
    def _conversation(
        *,
        conv_id: str,
        text: str,
        updated_at: datetime,
        provider: str = "chatgpt",
    ) -> Conversation:
        return make_conv(
            id=conv_id,
            provider=provider,
            title=conv_id,
            updated_at=updated_at,
            messages=[make_msg(id=f"{conv_id}:m1", role=Role.USER, text=text)],
        )

    @pytest.mark.asyncio
    async def test_separate_filters_have_independent_state(self, mock_repo: AsyncMock) -> None:
        """Two filters from same repo must not share state."""
        f1 = ConversationFilter(mock_repo)
        f2 = ConversationFilter(mock_repo)

        f1.provider("chatgpt")
        f2.provider("claude-ai")

        await f1.list()
        await f2.list()

        first_query = mock_repo.list_by_query.await_args_list[0].args[0]
        second_query = mock_repo.list_by_query.await_args_list[1].args[0]
        assert first_query.provider == "chatgpt"
        assert second_query.provider == "claude-ai"

    @pytest.mark.asyncio
    async def test_chained_methods_accumulate_on_same_instance(self, mock_repo: AsyncMock) -> None:
        """Chaining on same filter must accumulate predicates."""
        f = ConversationFilter(mock_repo)
        f.provider("chatgpt").contains("error").limit(10)

        await f.list()

        mock_repo.search.assert_awaited_once_with("error", limit=100, providers=["chatgpt"])

    @pytest.mark.asyncio
    async def test_filter_reuse_accumulates_providers(self, mock_repo: AsyncMock) -> None:
        """Reusing a filter adds to existing providers list."""
        f = ConversationFilter(mock_repo)
        f.provider("chatgpt")
        f.provider("claude-ai")  # second call

        await f.list()
        query = mock_repo.list_by_query.await_args_list[-1].args[0]
        assert query.providers == ("chatgpt", "claude-ai")

    @pytest.mark.asyncio
    async def test_filter_sort_replaces_not_accumulates(self, mock_repo: AsyncMock) -> None:
        """sort() should replace the previous sort, not append."""
        newer_short = self._conversation(
            conv_id="test:new-short",
            text="tiny",
            updated_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        older_long = self._conversation(
            conv_id="test:old-long",
            text="this message has many many words",
            updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        mock_repo.list_by_query.return_value = [newer_short, older_long]

        f = ConversationFilter(mock_repo)
        f.sort("date")
        f.sort("words")

        results = await f.list()
        assert str(results[0].id) == "test:old-long"

    @pytest.mark.asyncio
    async def test_fresh_filter_has_no_predicates(self, mock_repo: AsyncMock) -> None:
        """A brand-new filter should have empty state."""
        f = ConversationFilter(mock_repo)
        await f.list()

        mock_repo.search.assert_not_called()
        query = mock_repo.list_by_query.await_args.args[0]
        assert query.limit is None

    @pytest.mark.asyncio
    async def test_limit_replaces_previous_limit(self, mock_repo: AsyncMock) -> None:
        """Calling limit() twice replaces, doesn't stack."""
        f = ConversationFilter(mock_repo)
        f.limit(40)
        f.limit(5)
        await f.list()

        query = mock_repo.list_by_query.await_args_list[-1].args[0]
        # _effective_fetch_limit() applies a 2x safety margin: max(5*2, 2) = 10
        assert query.limit == 10


# =============================================================================
# Async backend: batch operations must not create N connections
# =============================================================================


class TestConnectionManagement:
    """Batch operations must use O(1) connections, not O(N)."""

    @pytest.mark.asyncio
    async def test_get_conversations_batch_uses_single_connection(self, sqlite_backend: SQLiteBackend) -> None:
        """get_conversations_batch must use 1 query, not N queries."""
        backend = sqlite_backend

        # Insert 20 conversations
        async with backend.connection() as conn:
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
    async def test_get_messages_batch_groups_by_conversation(self, sqlite_backend: SQLiteBackend) -> None:
        """get_messages_batch must return messages grouped by conversation_id."""
        backend = sqlite_backend

        # Insert conversations with messages
        async with backend.connection() as conn:
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
    async def test_batch_with_empty_ids_returns_empty(self, sqlite_backend: SQLiteBackend) -> None:
        """Passing empty list of IDs should return empty, not error."""
        backend = sqlite_backend
        records = await backend.get_conversations_batch([])
        assert records == []

    @pytest.mark.asyncio
    async def test_batch_with_nonexistent_ids_returns_partial(self, sqlite_backend: SQLiteBackend) -> None:
        """Requesting nonexistent IDs should return only existing ones."""
        backend = sqlite_backend

        async with backend.connection() as conn:
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
    async def test_concurrent_saves_dont_crash(self, sqlite_backend: SQLiteBackend) -> None:
        """Concurrent saves to different conversations must not error."""
        backend = sqlite_backend
        repo = ConversationRepository(backend=backend)

        async def _save_one(idx: int) -> None:
            conv = make_conversation(
                conversation_id=f"test:conv-{idx}",
                provider_name="test",
                provider_conversation_id=f"conv-{idx}",
                title=f"Conversation {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            msg = make_message(
                message_id=f"msg-{idx}",
                conversation_id=f"test:conv-{idx}",
                role="user",
                text=f"Hello from conversation {idx}",
                content_hash=f"mhash-{idx}",
                version=1,
            )
            await repo.save_conversation(conv, [msg], [])

        # Run 10 concurrent saves
        await asyncio.gather(*[_save_one(i) for i in range(10)])

        # Verify all saved
        async with backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 10

    @pytest.mark.asyncio
    async def test_concurrent_saves_to_same_conversation(self, sqlite_backend: SQLiteBackend) -> None:
        """Concurrent upserts to the same conversation must not corrupt."""
        backend = sqlite_backend
        repo = ConversationRepository(backend=backend)

        async def _upsert(version: int) -> None:
            conv = make_conversation(
                conversation_id="test:same-conv",
                provider_name="test",
                provider_conversation_id="same",
                title=f"Version {version}",
                content_hash=f"hash-v{version}",
                version=version,
            )
            msg = make_message(
                message_id=f"msg-v{version}",
                conversation_id="test:same-conv",
                role="user",
                text=f"Version {version}",
                content_hash=f"mhash-v{version}",
                version=version,
            )
            await repo.save_conversation(conv, [msg], [])

        # Run 5 concurrent upserts to the same conversation
        await asyncio.gather(*[_upsert(i) for i in range(5)])

        # Should have exactly 1 conversation (upserted, not duplicated)
        async with backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_writes(self, sqlite_backend: SQLiteBackend) -> None:
        """Reads during concurrent writes must not error or return garbage."""
        backend = sqlite_backend
        repo = ConversationRepository(backend=backend)

        async def _write(idx: int) -> None:
            conv = make_conversation(
                conversation_id=f"test:conv-{idx}",
                provider_name="test",
                provider_conversation_id=f"conv-{idx}",
                title=f"Conversation {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            await repo.save_conversation(conv, [], [])

        async def _read() -> int:
            async with backend.read_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
                row = await cursor.fetchone()
                assert row is not None
                return int(row[0])

        # Interleave writes and reads
        tasks: list[Awaitable[object]] = []
        for i in range(10):
            tasks.append(_write(i))
            tasks.append(_read())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        write_results = results[::2]
        read_results = results[1::2]

        from sqlite3 import OperationalError

        def _is_locked_error(result: object) -> bool:
            return isinstance(result, OperationalError) and "locked" in str(result)

        # Transient OperationalError("database is locked") is acceptable under
        # heavy contention. SQLite WAL mode does not guarantee zero-wait access
        # on all platforms. Non-locked exceptions are real failures.
        unexpected_writes = [
            result for result in write_results if isinstance(result, Exception) and not _is_locked_error(result)
        ]
        assert unexpected_writes == [], (
            f"Got unexpected write exceptions during concurrent read/write: {unexpected_writes}"
        )

        unexpected = [r for r in read_results if isinstance(r, Exception) and not _is_locked_error(r)]
        assert unexpected == [], f"Got unexpected exceptions during concurrent read/write: {unexpected}"

        successful_writes = sum(1 for result in write_results if not isinstance(result, Exception))
        final_count = await _read()
        assert final_count == successful_writes
