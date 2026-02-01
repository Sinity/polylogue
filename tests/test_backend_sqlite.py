"""Tests for SQLite storage backend implementation."""

from __future__ import annotations

import pytest

from polylogue.storage.backends import SQLiteBackend
from tests.helpers import make_attachment, make_conversation, make_message


# sqlite_backend fixture is in conftest.py


def test_backend_save_and_get_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving a conversation."""
    conv = make_conversation("conv1", title="Test Conversation", content_hash="hash123")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.commit()

    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"
    assert retrieved.title == "Test Conversation"
    assert retrieved.content_hash == "hash123"


def test_backend_save_and_get_messages(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving messages."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.commit()

    messages = sqlite_backend.get_messages("conv1")
    assert len(messages) == 2
    assert messages[0].message_id == "msg1"
    assert messages[0].text == "Hello"
    assert messages[1].message_id == "msg2"
    assert messages[1].text == "Hi there"


def test_backend_list_conversations(sqlite_backend: SQLiteBackend) -> None:
    """Test listing conversations."""
    conv1 = make_conversation("conv1", title="First", created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z", provider_meta={"source": "claude"})
    conv2 = make_conversation("conv2", title="Second", created_at="2024-01-02T00:00:00Z", updated_at="2024-01-02T00:00:00Z", provider_meta={"source": "chatgpt"})

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv1)
    sqlite_backend.save_conversation(conv2)
    sqlite_backend.commit()

    # List all
    all_convs = sqlite_backend.list_conversations()
    assert len(all_convs) == 2

    # List by source
    claude_convs = sqlite_backend.list_conversations(source="claude")
    assert len(claude_convs) == 1
    assert claude_convs[0].conversation_id == "conv1"


def test_backend_save_attachments(sqlite_backend: SQLiteBackend) -> None:
    """Test saving and retrieving attachments."""
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="Hello")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    attachments = sqlite_backend.get_attachments("conv1")
    assert len(attachments) == 1
    assert attachments[0].attachment_id == "att1"
    assert attachments[0].mime_type == "image/png"
    assert attachments[0].size_bytes == 1024


def test_backend_transaction_rollback(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction rollback."""
    conv = make_conversation("conv1", title="Test")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.rollback()

    # Should not exist after rollback
    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_transaction_context_manager(sqlite_backend: SQLiteBackend) -> None:
    """Test using the transaction context manager."""
    conv = make_conversation("conv1", title="Test")

    with sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)

    # Should exist after successful transaction
    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is not None
    assert retrieved.conversation_id == "conv1"


def test_backend_transaction_context_manager_exception(sqlite_backend: SQLiteBackend) -> None:
    """Test transaction context manager rolls back on exception."""
    conv = make_conversation("conv1", title="Test")

    with pytest.raises(ValueError), sqlite_backend.transaction():
        sqlite_backend.save_conversation(conv)
        raise ValueError("Test error")

    # Should not exist after rollback
    retrieved = sqlite_backend.get_conversation("conv1")
    assert retrieved is None


def test_backend_delete_conversation(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a conversation and all related records."""
    conv = make_conversation("conv1", title="Test")
    msg1 = make_message("msg1", "conv1", text="Hello")
    msg2 = make_message("msg2", "conv1", role="assistant", text="Hi there")
    att = make_attachment("att1", "conv1", "msg1", mime_type="image/png", size_bytes=1024)

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg1, msg2])
    sqlite_backend.save_attachments([att])
    sqlite_backend.commit()

    # Verify exists
    assert sqlite_backend.get_conversation("conv1") is not None
    assert len(sqlite_backend.get_messages("conv1")) == 2
    assert len(sqlite_backend.get_attachments("conv1")) == 1

    # Delete
    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    # Verify deleted
    assert sqlite_backend.get_conversation("conv1") is None
    assert len(sqlite_backend.get_messages("conv1")) == 0
    assert len(sqlite_backend.get_attachments("conv1")) == 0


def test_backend_delete_conversation_not_found(sqlite_backend: SQLiteBackend) -> None:
    """Test deleting a non-existent conversation returns False."""
    result = sqlite_backend.delete_conversation("nonexistent")
    assert result is False


def test_backend_delete_conversation_cleans_fts(sqlite_backend: SQLiteBackend) -> None:
    """Test that deleting a conversation also cleans up FTS entries."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations

    # Create and index conversation
    conv = make_conversation("conv1", title="Test")
    msg = make_message("msg1", "conv1", text="searchable content here")

    sqlite_backend.begin()
    sqlite_backend.save_conversation(conv)
    sqlite_backend.save_messages([msg])
    sqlite_backend.commit()

    # Build FTS index
    conn = sqlite_backend._get_connection()
    ensure_index(conn)
    update_index_for_conversations(["conv1"], conn)
    conn.commit()

    # Verify FTS entry exists
    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count > 0

    # Delete conversation
    result = sqlite_backend.delete_conversation("conv1")
    assert result is True

    # Verify FTS entries cleaned up
    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert fts_count == 0
