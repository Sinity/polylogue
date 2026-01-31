"""Tests for StorageBackend protocol conformance and behavior.

These tests verify that SQLiteBackend (and future backends) correctly implement
the StorageBackend protocol by testing actual behavior, not just method existence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.protocols import StorageBackend
from polylogue.storage.backends import SQLiteBackend
from tests.helpers import make_attachment, make_conversation, make_message


class TestStorageBackendProtocol:
    """Verify SQLiteBackend implements StorageBackend protocol."""

    def test_sqlite_backend_is_storage_backend(self, tmp_path: Path) -> None:
        """SQLiteBackend is a runtime-checkable StorageBackend."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        assert isinstance(backend, StorageBackend)
        backend.close()


class TestConversationOperations:
    """Test conversation save/retrieve operations."""

    def test_save_and_get_conversation(self, tmp_path: Path) -> None:
        """save_conversation persists data retrievable by get_conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1", title="Test Conversation", provider_name="claude")
        backend.save_conversation(conv)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_name == "claude"
        backend.close()

    def test_get_nonexistent_conversation_returns_none(self, tmp_path: Path) -> None:
        """get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.get_conversation("nonexistent")
        assert result is None
        backend.close()

    def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """save_conversation updates existing conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv1 = make_conversation("conv-1", title="Original Title")
        backend.save_conversation(conv1)

        conv2 = make_conversation("conv-1", title="Updated Title")
        backend.save_conversation(conv2)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        backend.close()

    def test_list_conversations_returns_all(self, tmp_path: Path) -> None:
        """list_conversations returns all stored conversations."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(3):
            conv = make_conversation(f"conv-{i}", title=f"Conversation {i}")
            backend.save_conversation(conv)

        all_convs = backend.list_conversations()
        assert len(all_convs) == 3
        assert {c.conversation_id for c in all_convs} == {"conv-0", "conv-1", "conv-2"}
        backend.close()

    def test_list_conversations_filters_by_provider(self, tmp_path: Path) -> None:
        """list_conversations filters by provider_name."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("c1", provider_name="claude"))
        backend.save_conversation(make_conversation("c2", provider_name="chatgpt"))
        backend.save_conversation(make_conversation("c3", provider_name="claude"))

        claude_convs = backend.list_conversations(provider="claude")
        assert len(claude_convs) == 2
        assert all(c.provider_name == "claude" for c in claude_convs)
        backend.close()

    def test_list_conversations_with_limit_and_offset(self, tmp_path: Path) -> None:
        """list_conversations supports pagination."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(10):
            conv = make_conversation(f"conv-{i:02d}")
            backend.save_conversation(conv)

        page1 = backend.list_conversations(limit=3, offset=0)
        page2 = backend.list_conversations(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        # Pages should not overlap
        page1_ids = {c.conversation_id for c in page1}
        page2_ids = {c.conversation_id for c in page2}
        assert page1_ids.isdisjoint(page2_ids)
        backend.close()


class TestMessageOperations:
    """Test message save/retrieve operations."""

    def test_save_and_get_messages(self, tmp_path: Path) -> None:
        """save_messages persists data retrievable by get_messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        messages = [
            make_message("m1", "conv-1", role="user", text="Hello"),
            make_message("m2", "conv-1", role="assistant", text="Hi there"),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert {m.message_id for m in retrieved} == {"m1", "m2"}
        assert {m.role for m in retrieved} == {"user", "assistant"}
        backend.close()

    def test_get_messages_for_empty_conversation(self, tmp_path: Path) -> None:
        """get_messages returns empty list for conversation with no messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        retrieved = backend.get_messages("conv-1")
        assert retrieved == []
        backend.close()


class TestAttachmentOperations:
    """Test attachment save/retrieve operations."""

    def test_save_and_get_attachments(self, tmp_path: Path) -> None:
        """save_attachments persists data retrievable by get_attachments."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1", mime_type="image/png", size_bytes=1024),
            make_attachment("att2", "conv-1", mime_type="text/plain", size_bytes=256),
        ]
        backend.save_attachments(attachments)

        retrieved = backend.get_attachments("conv-1")
        assert len(retrieved) == 2
        assert {a.attachment_id for a in retrieved} == {"att1", "att2"}
        backend.close()

    def test_prune_attachments_removes_unlisted(self, tmp_path: Path) -> None:
        """prune_attachments removes attachments not in keep set."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        attachments = [
            make_attachment("att1", "conv-1"),
            make_attachment("att2", "conv-1"),
            make_attachment("att3", "conv-1"),
        ]
        backend.save_attachments(attachments)

        # Keep only att1 and att3
        backend.prune_attachments("conv-1", {"att1", "att3"})

        retrieved = backend.get_attachments("conv-1")
        assert {a.attachment_id for a in retrieved} == {"att1", "att3"}
        backend.close()


class TestTransactionOperations:
    """Test transaction management."""

    def test_begin_commit_persists_data(self, tmp_path: Path) -> None:
        """Data saved within begin/commit is persisted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.begin()
        conv = make_conversation("tx-conv")
        backend.save_conversation(conv)
        backend.commit()

        retrieved = backend.get_conversation("tx-conv")
        assert retrieved is not None
        backend.close()

    def test_rollback_discards_changes(self, tmp_path: Path) -> None:
        """Data saved within begin/rollback is discarded."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # Start a transaction explicitly
        backend.begin()
        # Save something that we'll rollback
        backend.save_conversation(make_conversation("rollback-conv"))
        # Rollback should discard the save
        backend.rollback()

        # The rollback-conv should not exist
        assert backend.get_conversation("rollback-conv") is None
        backend.close()


class TestMetadataOperations:
    """Test metadata CRUD operations."""

    def test_update_and_get_metadata(self, tmp_path: Path) -> None:
        """update_metadata sets key, get_metadata retrieves it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "rating", 5)
        backend.update_metadata("conv-1", "reviewed", True)

        metadata = backend.get_metadata("conv-1")
        assert metadata.get("rating") == 5
        assert metadata.get("reviewed") is True
        backend.close()

    def test_delete_metadata(self, tmp_path: Path) -> None:
        """delete_metadata removes a key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.update_metadata("conv-1", "temp", "value")
        backend.delete_metadata("conv-1", "temp")

        metadata = backend.get_metadata("conv-1")
        assert "temp" not in metadata
        backend.close()

    def test_add_and_remove_tag(self, tmp_path: Path) -> None:
        """add_tag adds to tags list, remove_tag removes it."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conv-1")
        backend.save_conversation(conv)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "work")

        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" in tags

        backend.remove_tag("conv-1", "work")
        metadata = backend.get_metadata("conv-1")
        tags = metadata.get("tags", [])
        assert "important" in tags
        assert "work" not in tags
        backend.close()


class TestSearchOperations:
    """Test search and resolve operations."""

    def test_resolve_id_exact_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for exact match."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("conversation-12345")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("conversation-12345")
        assert resolved == "conversation-12345"
        backend.close()

    def test_resolve_id_prefix_match(self, tmp_path: Path) -> None:
        """resolve_id returns full ID for unique prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("unique-prefix-abc123")
        backend.save_conversation(conv)

        resolved = backend.resolve_id("unique-prefix")
        assert resolved == "unique-prefix-abc123"
        backend.close()

    def test_resolve_id_ambiguous_returns_none(self, tmp_path: Path) -> None:
        """resolve_id returns None for ambiguous prefix."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        backend.save_conversation(make_conversation("prefix-abc"))
        backend.save_conversation(make_conversation("prefix-def"))

        resolved = backend.resolve_id("prefix")
        assert resolved is None
        backend.close()


class TestDeleteOperations:
    """Test deletion operations."""

    def test_delete_conversation(self, tmp_path: Path) -> None:
        """delete_conversation removes conversation and related data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        conv = make_conversation("to-delete")
        backend.save_conversation(conv)
        backend.save_messages([make_message("m1", "to-delete")])
        backend.save_attachments([make_attachment("a1", "to-delete")])

        result = backend.delete_conversation("to-delete")
        assert result is True

        assert backend.get_conversation("to-delete") is None
        assert backend.get_messages("to-delete") == []
        assert backend.get_attachments("to-delete") == []
        backend.close()

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """delete_conversation returns False for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        result = backend.delete_conversation("nonexistent")
        assert result is False
        backend.close()
