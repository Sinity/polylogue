"""Tests for async pipeline helper modules.

Covers:
- async_prepare.py: prepare_records, async_save_bundle
- async_index.py: async_ensure_index, async_rebuild_index, async_update_index_for_conversations, async_index_status
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import ConversationRecord, MessageRecord


# =============================================================================
# async_index.py Tests
# =============================================================================


class TestAsyncEnsureIndex:
    """Tests for async_ensure_index."""

    @pytest.mark.asyncio
    async def test_creates_fts_table(self):
        """FTS table is created when it doesn't exist."""
        from polylogue.storage.async_index import async_ensure_index, async_index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            # Initialize schema
            await backend.list_conversations()

            await async_ensure_index(backend)

            status = await async_index_status(backend)
            assert status["exists"] is True
            await backend.close()

    @pytest.mark.asyncio
    async def test_idempotent(self):
        """Calling ensure_index multiple times is safe."""
        from polylogue.storage.async_index import async_ensure_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()

            # Call twice - should not raise
            await async_ensure_index(backend)
            await async_ensure_index(backend)
            await backend.close()


class TestAsyncRebuildIndex:
    """Tests for async_rebuild_index."""

    @pytest.mark.asyncio
    async def test_populates_from_messages(self):
        """Rebuild populates FTS from messages table."""
        from polylogue.storage.async_index import async_index_status, async_rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)

            now = datetime.now(timezone.utc).isoformat()

            # Save a conversation with messages
            conv = ConversationRecord(
                conversation_id="test:rebuild",
                provider_name="test",
                provider_conversation_id="ext-1",
                title="Rebuild Test",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                MessageRecord(
                    message_id=f"m{i}",
                    conversation_id="test:rebuild",
                    role="user",
                    text=f"Message {i} about testing",
                    timestamp=now,
                    content_hash=uuid4().hex[:16],
                )
                for i in range(5)
            ]
            await backend.save_conversation(conv, messages, [])

            # Rebuild index
            await async_rebuild_index(backend)

            # Verify index has entries
            status = await async_index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 5
            await backend.close()

    @pytest.mark.asyncio
    async def test_rebuild_clears_stale_entries(self):
        """Rebuild removes entries for deleted messages."""
        from polylogue.storage.async_index import async_index_status, async_rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)

            now = datetime.now(timezone.utc).isoformat()

            # Save conversation with messages
            conv = ConversationRecord(
                conversation_id="test:stale",
                provider_name="test",
                provider_conversation_id="ext-stale",
                title="Stale Test",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                MessageRecord(
                    message_id=f"stale-m{i}",
                    conversation_id="test:stale",
                    role="user",
                    text=f"Stale message {i}",
                    timestamp=now,
                    content_hash=uuid4().hex[:16],
                )
                for i in range(3)
            ]
            await backend.save_conversation(conv, messages, [])

            # Build index
            await async_rebuild_index(backend)
            status1 = await async_index_status(backend)
            assert status1["count"] == 3

            # Delete conversation
            await backend.delete_conversation("test:stale")

            # Rebuild should clear stale entries
            await async_rebuild_index(backend)
            status2 = await async_index_status(backend)
            assert status2["count"] == 0
            await backend.close()


class TestAsyncUpdateIndex:
    """Tests for async_update_index_for_conversations."""

    @pytest.mark.asyncio
    async def test_incremental_update(self):
        """Updates index for specific conversations correctly."""
        from polylogue.storage.async_index import (
            async_index_status,
            async_update_index_for_conversations,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)

            now = datetime.now(timezone.utc).isoformat()

            # Save two conversations
            for conv_id in ["test:a", "test:b"]:
                conv = ConversationRecord(
                    conversation_id=conv_id,
                    provider_name="test",
                    provider_conversation_id=conv_id.split(":")[1],
                    title=f"Conv {conv_id}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                messages = [
                    MessageRecord(
                        message_id=f"{conv_id}-m1",
                        conversation_id=conv_id,
                        role="user",
                        text=f"Message for {conv_id}",
                        timestamp=now,
                        content_hash=uuid4().hex[:16],
                    )
                ]
                await backend.save_conversation(conv, messages, [])

            # Schema init auto-indexes; verify both are indexed
            status = await async_index_status(backend)
            assert status["count"] == 2

            # Delete conv a messages from FTS, then re-index just conv a
            # This should preserve count (idempotent re-index)
            await async_update_index_for_conversations(["test:a"], backend)

            status = await async_index_status(backend)
            assert status["count"] == 2  # Both still indexed
            await backend.close()

    @pytest.mark.asyncio
    async def test_update_empty_list(self):
        """Empty conversation list is a no-op."""
        from polylogue.storage.async_index import async_update_index_for_conversations

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()

            # Should not raise
            await async_update_index_for_conversations([], backend)
            await backend.close()


class TestAsyncIndexStatus:
    """Tests for async_index_status."""

    @pytest.mark.asyncio
    async def test_reports_exists_and_count(self):
        """Returns correct exists flag and count."""
        from polylogue.storage.async_index import async_index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            # Schema init auto-creates FTS table
            await backend.list_conversations()

            status = await async_index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 0  # No messages yet
            await backend.close()


# =============================================================================
# async_prepare.py Tests
# =============================================================================


class TestAsyncPrepareRecords:
    """Tests for prepare_records."""

    @pytest.mark.asyncio
    async def test_new_conversation(self):
        """Creates new records for unseen conversation."""
        from polylogue.pipeline.prepare import prepare_records
        from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
        from polylogue.storage.repository import ConversationRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            convo = ParsedConversation(
                provider_name="chatgpt",
                provider_conversation_id="ext-123",
                title="New Conversation",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                messages=[
                    ParsedMessage(
                        provider_message_id="pm1",
                        role="user",
                        text="Hello!",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    ParsedMessage(
                        provider_message_id="pm2",
                        role="assistant",
                        text="Hi there!",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                ],
                attachments=[],
            )

            cid, counts, changed = await prepare_records(
                convo,
                "test-source",
                archive_root=Path(tmpdir),
                backend=backend,
                repository=repo,
            )

            assert cid is not None
            assert counts["conversations"] == 1
            assert counts["messages"] == 2
            assert changed is False  # New conversation, not changed
            await backend.close()

    @pytest.mark.asyncio
    async def test_existing_unchanged(self):
        """Skips when content hash matches existing."""
        from polylogue.pipeline.prepare import prepare_records
        from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
        from polylogue.storage.repository import ConversationRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            convo = ParsedConversation(
                provider_name="chatgpt",
                provider_conversation_id="ext-dup",
                title="Duplicate",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                messages=[
                    ParsedMessage(
                        provider_message_id="pm1",
                        role="user",
                        text="Same content",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                ],
                attachments=[],
            )

            # First save
            cid1, counts1, _ = await prepare_records(
                convo, "src", archive_root=Path(tmpdir), backend=backend, repository=repo,
            )

            # Second save — same content
            cid2, counts2, changed = await prepare_records(
                convo, "src", archive_root=Path(tmpdir), backend=backend, repository=repo,
            )

            assert cid1 == cid2
            assert changed is False  # Hash unchanged
            # Messages should be skipped (already exist)
            assert counts2["skipped_messages"] >= 1
            await backend.close()

    @pytest.mark.asyncio
    async def test_existing_changed(self):
        """Detects content change via hash mismatch."""
        from polylogue.pipeline.prepare import prepare_records
        from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
        from polylogue.storage.repository import ConversationRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path=db_path)
            repo = ConversationRepository(backend=backend)

            convo_v1 = ParsedConversation(
                provider_name="chatgpt",
                provider_conversation_id="ext-change",
                title="Original",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                messages=[
                    ParsedMessage(
                        provider_message_id="pm1",
                        role="user",
                        text="Version 1",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                ],
                attachments=[],
            )

            # First save
            await prepare_records(
                convo_v1, "src", archive_root=Path(tmpdir), backend=backend, repository=repo,
            )

            # Modified version (different text = different hash)
            convo_v2 = ParsedConversation(
                provider_name="chatgpt",
                provider_conversation_id="ext-change",
                title="Modified",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                messages=[
                    ParsedMessage(
                        provider_message_id="pm1",
                        role="user",
                        text="Version 2 with changes",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    ParsedMessage(
                        provider_message_id="pm2",
                        role="assistant",
                        text="New reply",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                ],
                attachments=[],
            )

            cid, counts, changed = await prepare_records(
                convo_v2, "src", archive_root=Path(tmpdir), backend=backend, repository=repo,
            )

            assert changed is True  # Hash changed
            assert counts["messages"] >= 1  # New message added
            await backend.close()
