"""Tests for AsyncPolylogue facade.

Covers:
- Context manager lifecycle
- get_conversation / get_conversations
- list_conversations
- search (via asyncio.to_thread)
- parse_file / parse_sources
- stats (returns ArchiveStats)
- filter (returns ConversationFilter)
- rebuild_index
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.async_facade import AsyncPolylogue
from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend
from polylogue.storage.store import ConversationRecord, MessageRecord


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestAsyncPolylogueLifecycle:
    """Tests for context manager and lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager_enters_and_exits(self):
        """async with enters and exits cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                assert archive is not None
                assert isinstance(archive, AsyncPolylogue)

    @pytest.mark.asyncio
    async def test_manual_close(self):
        """Manual close works without context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = AsyncPolylogue(db_path=Path(tmpdir) / "test.db")
            await archive.close()
            # Double close should be safe
            await archive.close()

    @pytest.mark.asyncio
    async def test_repr(self):
        """__repr__ returns useful string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = AsyncPolylogue(db_path=Path(tmpdir) / "test.db")
            r = repr(archive)
            assert "AsyncPolylogue" in r
            await archive.close()

    @pytest.mark.asyncio
    async def test_config_property(self):
        """config property returns Config object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                config = archive.config
                assert config is not None
                assert hasattr(config, "archive_root")

    @pytest.mark.asyncio
    async def test_archive_root_property(self):
        """archive_root property returns Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(
                archive_root=tmpdir,
                db_path=Path(tmpdir) / "test.db",
            ) as archive:
                assert isinstance(archive.archive_root, Path)
                assert str(archive.archive_root) == str(Path(tmpdir).resolve())


# =============================================================================
# Get Conversation Tests
# =============================================================================


class TestGetConversation:
    """Tests for get_conversation and get_conversations."""

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self):
        """Returns None for missing conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                result = await archive.get_conversation("nonexistent:id")
                assert result is None

    @pytest.mark.asyncio
    async def test_get_conversation_found(self):
        """Returns conversation when it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Pre-populate DB
            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()
            conv = ConversationRecord(
                conversation_id="test:found",
                provider_name="test",
                provider_conversation_id="ext-1",
                title="Found Me",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                MessageRecord(
                    message_id="m1",
                    conversation_id="test:found",
                    role="user",
                    text="Hello",
                    timestamp=now,
                    content_hash="hash1",
                ),
            ]
            await backend.save_conversation(conv, messages, [])
            await backend.close()

            # Now test facade
            async with AsyncPolylogue(db_path=db_path) as archive:
                result = await archive.get_conversation("test:found")
                assert result is not None
                assert result.id == "test:found"
                assert result.title == "Found Me"

    @pytest.mark.asyncio
    async def test_get_conversations_batch(self):
        """Fetches multiple conversations in parallel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()

            for i in range(3):
                conv = ConversationRecord(
                    conversation_id=f"test:batch-{i}",
                    provider_name="test",
                    provider_conversation_id=f"ext-{i}",
                    title=f"Batch {i}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                await backend.save_conversation(conv, [], [])
            await backend.close()

            async with AsyncPolylogue(db_path=db_path) as archive:
                results = await archive.get_conversations(
                    ["test:batch-0", "test:batch-1", "test:batch-2"]
                )
                assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_conversations_partial_failure(self):
        """Skips failed IDs gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()
            conv = ConversationRecord(
                conversation_id="test:exists",
                provider_name="test",
                provider_conversation_id="ext-1",
                title="Exists",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            await backend.save_conversation(conv, [], [])
            await backend.close()

            async with AsyncPolylogue(db_path=db_path) as archive:
                results = await archive.get_conversations(
                    ["test:exists", "test:missing1", "test:missing2"]
                )
                # Only the existing one should be returned
                assert len(results) == 1
                assert results[0].id == "test:exists"


# =============================================================================
# List Conversations Tests
# =============================================================================


class TestListConversations:
    """Tests for list_conversations."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Empty DB returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                convs = await archive.list_conversations()
                assert convs == []

    @pytest.mark.asyncio
    async def test_list_with_provider_filter(self):
        """Filters by provider name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()

            for provider, cid in [("chatgpt", "cg:1"), ("claude", "cl:1"), ("chatgpt", "cg:2")]:
                conv = ConversationRecord(
                    conversation_id=cid,
                    provider_name=provider,
                    provider_conversation_id=cid,
                    title=f"From {provider}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                await backend.save_conversation(conv, [], [])
            await backend.close()

            async with AsyncPolylogue(db_path=db_path) as archive:
                chatgpt = await archive.list_conversations(provider="chatgpt")
                assert len(chatgpt) == 2

                claude = await archive.list_conversations(provider="claude")
                assert len(claude) == 1

    @pytest.mark.asyncio
    async def test_list_with_limit(self):
        """Respects limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()

            for i in range(5):
                conv = ConversationRecord(
                    conversation_id=f"test:lim-{i}",
                    provider_name="test",
                    provider_conversation_id=f"ext-{i}",
                    title=f"Lim {i}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                await backend.save_conversation(conv, [], [])
            await backend.close()

            async with AsyncPolylogue(db_path=db_path) as archive:
                limited = await archive.list_conversations(limit=3)
                assert len(limited) == 3


# =============================================================================
# Stats Tests
# =============================================================================


class TestStats:
    """Tests for stats method."""

    @pytest.mark.asyncio
    async def test_empty_stats(self):
        """Stats on empty DB returns zeros."""
        from polylogue.async_facade import ArchiveStats

        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                stats = await archive.stats()
                assert isinstance(stats, ArchiveStats)
                assert stats.conversation_count == 0
                assert stats.message_count == 0
                assert stats.word_count == 0
                assert stats.providers == {}

    @pytest.mark.asyncio
    async def test_stats_with_data(self):
        """Stats counts conversations, messages, words."""
        from polylogue.async_facade import ArchiveStats

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            backend = AsyncSQLiteBackend(db_path=db_path)
            now = datetime.now(timezone.utc).isoformat()

            conv = ConversationRecord(
                conversation_id="test:stats",
                provider_name="chatgpt",
                provider_conversation_id="ext-1",
                title="Stats Test",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                MessageRecord(
                    message_id=f"m{i}",
                    conversation_id="test:stats",
                    role="user",
                    text=f"Word1 word2 word3 message {i}",
                    timestamp=now,
                    content_hash=uuid4().hex[:16],
                )
                for i in range(3)
            ]
            await backend.save_conversation(conv, messages, [])
            await backend.close()

            async with AsyncPolylogue(db_path=db_path) as archive:
                stats = await archive.stats()
                assert isinstance(stats, ArchiveStats)
                assert stats.conversation_count == 1
                assert stats.message_count == 3
                assert stats.word_count > 0
                assert "chatgpt" in stats.providers
                assert stats.providers["chatgpt"] == 1


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search method."""

    @pytest.mark.asyncio
    async def test_search_empty_db(self, workspace_env):
        """Search on empty DB returns empty results."""
        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)

        async with AsyncPolylogue(
            archive_root=archive_root,
            db_path=workspace_env["data_root"] / "polylogue" / "polylogue.db",
        ) as archive:
            results = await archive.search("anything")
            assert hasattr(results, "hits")
            assert len(results.hits) == 0


# =============================================================================
# Filter Tests
# =============================================================================


class TestFilter:
    """Tests for filter method."""

    @pytest.mark.asyncio
    async def test_filter_returns_conversation_filter(self):
        """filter() returns ConversationFilter instance."""
        from polylogue.lib.filters import ConversationFilter

        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                f = archive.filter()
                assert isinstance(f, ConversationFilter)

    @pytest.mark.asyncio
    async def test_filter_list_empty(self):
        """filter().list() returns empty list on empty DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                convs = await archive.filter().list()
                assert convs == []


# =============================================================================
# Rebuild Index Tests
# =============================================================================


class TestRebuildIndex:
    """Tests for rebuild_index method."""

    @pytest.mark.asyncio
    async def test_rebuild_index_empty_db(self):
        """Rebuild on empty DB succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncPolylogue(db_path=Path(tmpdir) / "test.db") as archive:
                result = await archive.rebuild_index()
                assert result is True
