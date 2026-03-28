"""Focused tests for low-level async index helpers."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import ConversationRecord, MessageRecord


class TestAsyncEnsureIndex:
    """Tests for ensure_index."""

    @pytest.mark.asyncio
    async def test_creates_fts_table(self):
        from polylogue.storage.async_index import ensure_index, index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()
            await ensure_index(backend)
            status = await index_status(backend)
            assert status["exists"] is True
            await backend.close()

    @pytest.mark.asyncio
    async def test_idempotent(self):
        from polylogue.storage.async_index import ensure_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()
            await ensure_index(backend)
            await ensure_index(backend)
            await backend.close()


class TestAsyncRebuildIndex:
    """Tests for rebuild_index."""

    @pytest.mark.asyncio
    async def test_populates_from_messages(self):
        from polylogue.storage.async_index import index_status, rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            now = datetime.now(timezone.utc).isoformat()
            conversation = ConversationRecord(
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
            await backend.save_conversation(conversation, messages, [])
            await rebuild_index(backend)
            status = await index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 5
            await backend.close()

    @pytest.mark.asyncio
    async def test_rebuild_clears_stale_entries(self):
        from polylogue.storage.async_index import index_status, rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            now = datetime.now(timezone.utc).isoformat()
            conversation = ConversationRecord(
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
            await backend.save_conversation(conversation, messages, [])
            await rebuild_index(backend)
            status_before = await index_status(backend)
            assert status_before["count"] == 3
            await backend.delete_conversation("test:stale")
            await rebuild_index(backend)
            status_after = await index_status(backend)
            assert status_after["count"] == 0
            await backend.close()


class TestAsyncUpdateIndex:
    """Tests for update_index_for_conversations."""

    @pytest.mark.asyncio
    async def test_incremental_update(self):
        from polylogue.storage.async_index import index_status, update_index_for_conversations

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            now = datetime.now(timezone.utc).isoformat()
            for conversation_id in ["test:a", "test:b"]:
                conversation = ConversationRecord(
                    conversation_id=conversation_id,
                    provider_name="test",
                    provider_conversation_id=conversation_id.split(":")[1],
                    title=f"Conv {conversation_id}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                messages = [
                    MessageRecord(
                        message_id=f"{conversation_id}-m1",
                        conversation_id=conversation_id,
                        role="user",
                        text=f"Message for {conversation_id}",
                        timestamp=now,
                        content_hash=uuid4().hex[:16],
                    )
                ]
                await backend.save_conversation(conversation, messages, [])

            status = await index_status(backend)
            assert status["count"] == 2
            await update_index_for_conversations(["test:a"], backend)
            status = await index_status(backend)
            assert status["count"] == 2
            await backend.close()

    @pytest.mark.asyncio
    async def test_update_empty_list(self):
        from polylogue.storage.async_index import update_index_for_conversations

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()
            await update_index_for_conversations([], backend)
            await backend.close()


class TestAsyncIndexStatus:
    """Tests for index_status."""

    @pytest.mark.asyncio
    async def test_reports_exists_and_count(self):
        from polylogue.storage.async_index import index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.list_conversations()
            status = await index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 0
            await backend.close()
