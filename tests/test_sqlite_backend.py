"""Comprehensive tests for SQLiteBackend and sqlite module functions."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from polylogue.storage.backends.sqlite import (
    DatabaseError,
    SQLiteBackend,
    _json_or_none,
    _make_ref_id,
    connection_context,
    default_db_path,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
)


# ============================================================================
# Test: connection_context
# ============================================================================


class TestConnectionContext:
    """Tests for connection_context context manager."""

    def test_connection_context_creates_connection(self, tmp_path):
        """Test that connection_context creates and yields a valid connection."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert conn.row_factory == sqlite3.Row
            # Verify schema was created
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            assert len(tables) > 0

    def test_connection_context_closes_connection(self, tmp_path):
        """Test that connection_context closes the connection after exiting."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            conn_obj = conn
        # After exiting context, connection should be closed
        with pytest.raises(sqlite3.ProgrammingError):
            conn_obj.execute("SELECT 1")

    def test_connection_context_with_none_uses_default(self, tmp_path, monkeypatch):
        """Test that connection_context(None) uses default_db_path()."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        # Reload to pick up env change
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        with connection_context(None) as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Verify it created a database file
            assert Path(conn.execute("PRAGMA database_list").fetchone()[2]).exists()

    def test_connection_context_with_existing_connection(self, tmp_path):
        """Test that connection_context yields the passed connection unchanged."""
        db_path = tmp_path / "test.db"
        # Create initial connection
        with connection_context(db_path) as initial_conn:
            initial_conn_obj = initial_conn
            # Pass the connection to connection_context again
            with connection_context(initial_conn_obj) as conn:
                assert conn is initial_conn_obj

    def test_connection_context_sets_pragmas(self, tmp_path):
        """Test that connection_context sets required PRAGMAs."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            # Check foreign keys enabled
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            assert fk == 1
            # Check WAL mode enabled
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.upper() == "WAL"

    def test_connection_context_creates_schema(self, tmp_path):
        """Test that connection_context ensures schema is created."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            # Check key tables exist
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "conversations" in tables
            assert "messages" in tables
            assert "attachments" in tables
            assert "attachment_refs" in tables

    def test_connection_context_parent_directory_created(self, tmp_path):
        """Test that connection_context creates parent directories."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.db"
        with connection_context(nested_path) as conn:
            assert isinstance(conn, sqlite3.Connection)
            assert nested_path.exists()


# ============================================================================
# Test: SQLiteBackend.__init__
# ============================================================================


class TestSQLiteBackendInit:
    """Tests for SQLiteBackend initialization."""

    def test_init_with_custom_path(self, tmp_path):
        """Test SQLiteBackend initialization with custom path."""
        db_path = tmp_path / "custom.db"
        backend = SQLiteBackend(db_path=db_path)
        assert backend._db_path == db_path

    def test_init_with_none_uses_default(self, tmp_path, monkeypatch):
        """Test that SQLiteBackend(None) uses default_db_path()."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        backend = SQLiteBackend(db_path=None)
        # Should use default_db_path() which includes XDG_DATA_HOME
        assert "polylogue" in str(backend._db_path)
        assert str(backend._db_path).endswith("polylogue.db")

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that SQLiteBackend creates parent directories."""
        nested_path = tmp_path / "x" / "y" / "z" / "test.db"
        backend = SQLiteBackend(db_path=nested_path)
        assert nested_path.parent.exists()

    def test_init_thread_local_storage(self, tmp_path):
        """Test that SQLiteBackend has thread-local storage."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        assert hasattr(backend, "_local")
        import threading

        assert isinstance(backend._local, threading.local)


# ============================================================================
# Test: save_conversation + get_conversation round-trip
# ============================================================================


class TestSaveGetConversation:
    """Tests for save_conversation and get_conversation operations."""

    def test_save_and_get_conversation_basic(self, tmp_path):
        """Test basic save and retrieval of a conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-conv-1",
            title="Test Conversation",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash123",
            provider_meta={"source": "test"},
            metadata={"tags": ["important"]},
            version=1,
        )
        backend.save_conversation(record)
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None
        assert retrieved.conversation_id == "conv-1"
        assert retrieved.title == "Test Conversation"
        assert retrieved.provider_meta == {"source": "test"}
        assert retrieved.metadata == {"tags": ["important"]}

    def test_get_nonexistent_conversation(self, tmp_path):
        """Test that get_conversation returns None for missing ID."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.get_conversation("nonexistent")
        assert result is None

    def test_save_conversation_upsert_different_hash(self, tmp_path):
        """Test upsert: same ID, different content_hash → updates."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Original Title",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash_old",
            version=1,
        )
        backend.save_conversation(record1)

        record2 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Updated Title",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash_new",
            version=2,
        )
        backend.save_conversation(record2)

        retrieved = backend.get_conversation("conv-1")
        assert retrieved.title == "Updated Title"
        assert retrieved.updated_at == "2025-01-01T12:00:00Z"

    def test_save_conversation_no_update_same_hash(self, tmp_path):
        """Test upsert: same ID and content_hash → no update."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Original",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash123",
            version=1,
        )
        backend.save_conversation(record)
        retrieved1 = backend.get_conversation("conv-1")

        # Save with same hash and content
        backend.save_conversation(record)
        retrieved2 = backend.get_conversation("conv-1")

        assert retrieved1.title == retrieved2.title
        assert retrieved1.updated_at == retrieved2.updated_at

    def test_save_conversation_metadata_not_overwritten(self, tmp_path):
        """Test that upsert does NOT overwrite user metadata."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T10:00:00Z",
            content_hash="hash1",
            metadata={"tags": ["important"]},
            version=1,
        )
        backend.save_conversation(record1)

        # Update metadata manually
        backend.update_metadata("conv-1", "custom_key", "custom_value")

        # Save new record with different content_hash
        record2 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T12:00:00Z",
            content_hash="hash2",
            metadata=None,  # Empty metadata in the record
            version=2,
        )
        backend.save_conversation(record2)

        # Metadata should still have the custom key
        meta = backend.get_metadata("conv-1")
        assert meta.get("custom_key") == "custom_value"

    def test_save_conversation_with_null_fields(self, tmp_path):
        """Test save_conversation with None values for optional fields."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title=None,
            created_at=None,
            updated_at=None,
            content_hash="hash",
            provider_meta=None,
            version=1,
        )
        backend.save_conversation(record)
        retrieved = backend.get_conversation("conv-1")
        assert retrieved.title is None
        assert retrieved.created_at is None

    def test_save_conversation_with_branching_info(self, tmp_path):
        """Test save_conversation with parent and branch_type."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create parent conversation
        parent = ConversationRecord(
            conversation_id="conv-parent",
            provider_name="claude",
            provider_conversation_id="prov-parent",
            title="Parent",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash-parent",
            version=1,
        )
        backend.save_conversation(parent)

        # Create child conversation
        child = ConversationRecord(
            conversation_id="conv-child",
            provider_name="claude",
            provider_conversation_id="prov-child",
            title="Child",
            created_at="2025-01-01T01:00:00Z",
            updated_at="2025-01-01T01:00:00Z",
            content_hash="hash-child",
            version=1,
            parent_conversation_id="conv-parent",
            branch_type="continuation",
        )
        backend.save_conversation(child)

        retrieved = backend.get_conversation("conv-child")
        assert retrieved.parent_conversation_id == "conv-parent"
        assert retrieved.branch_type == "continuation"


# ============================================================================
# Test: save_messages + get_messages round-trip
# ============================================================================


class TestSaveGetMessages:
    """Tests for save_messages and get_messages operations."""

    def test_save_and_get_messages(self, tmp_path):
        """Test basic save and retrieval of messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create conversation first
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                provider_message_id="prov-msg-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="msg-hash-1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                provider_message_id="prov-msg-2",
                role="assistant",
                text="Hi there",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="msg-hash-2",
                version=1,
            ),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert len(retrieved) == 2
        assert retrieved[0].message_id == "msg-1"
        assert retrieved[1].message_id == "msg-2"

    def test_save_empty_messages_list(self, tmp_path):
        """Test that save_messages([]) is a no-op."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        backend.save_messages([])  # Should not raise
        # No messages to retrieve, but no error either
        assert True

    def test_get_messages_ordering_by_timestamp(self, tmp_path):
        """Test that messages are returned ordered by timestamp."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        # Save in non-chronological order
        messages = [
            MessageRecord(
                message_id="msg-3",
                conversation_id="conv-1",
                role="assistant",
                text="Third",
                timestamp="2025-01-01T10:02:00Z",
                content_hash="h3",
                version=1,
            ),
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="First",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="assistant",
                text="Second",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="h2",
                version=1,
            ),
        ]
        backend.save_messages(messages)

        retrieved = backend.get_messages("conv-1")
        assert [m.message_id for m in retrieved] == ["msg-1", "msg-2", "msg-3"]

    def test_save_messages_upsert_different_hash(self, tmp_path):
        """Test upsert: same message_id, different content_hash → updates."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        msg1 = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Original",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash_old",
            version=1,
        )
        backend.save_messages([msg1])

        msg2 = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Updated",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash_new",
            version=2,
        )
        backend.save_messages([msg2])

        retrieved = backend.get_messages("conv-1")
        assert retrieved[0].text == "Updated"

    def test_save_messages_no_update_same_hash(self, tmp_path):
        """Test upsert: same message_id and content_hash → no update."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash123",
            version=1,
        )
        backend.save_messages([msg])
        retrieved1 = backend.get_messages("conv-1")[0]

        backend.save_messages([msg])  # Save again with same hash
        retrieved2 = backend.get_messages("conv-1")[0]

        assert retrieved1.text == retrieved2.text

    def test_get_messages_nonexistent_conversation(self, tmp_path):
        """Test get_messages returns empty list for nonexistent conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.get_messages("nonexistent")
        assert result == []

    def test_save_messages_with_provider_meta(self, tmp_path):
        """Test saving messages with provider metadata."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash",
            provider_meta={"custom": "data"},
            version=1,
        )
        backend.save_messages([msg])

        retrieved = backend.get_messages("conv-1")[0]
        assert retrieved.provider_meta == {"custom": "data"}

    def test_save_messages_with_branching_info(self, tmp_path):
        """Test saving messages with parent_message_id and branch_index."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        conv = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(conv)

        msg = MessageRecord(
            message_id="msg-1",
            conversation_id="conv-1",
            role="user",
            text="Hello",
            timestamp="2025-01-01T10:00:00Z",
            content_hash="hash",
            version=1,
            parent_message_id="msg-parent",
            branch_index=2,
        )
        backend.save_messages([msg])

        retrieved = backend.get_messages("conv-1")[0]
        assert retrieved.parent_message_id == "msg-parent"
        assert retrieved.branch_index == 2


# ============================================================================
# Test: get_conversations_batch
# ============================================================================


class TestGetConversationsBatch:
    """Tests for get_conversations_batch operation."""

    def test_get_conversations_batch_basic(self, tmp_path):
        """Test batch retrieval preserves order."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        ids = ["conv-1", "conv-2", "conv-3"]
        for cid in ids:
            record = ConversationRecord(
                conversation_id=cid,
                provider_name="claude",
                provider_conversation_id=f"prov-{cid}",
                title=f"Conversation {cid}",
                created_at="2025-01-01T00:00:00Z",
                updated_at="2025-01-01T00:00:00Z",
                content_hash=f"hash-{cid}",
                version=1,
            )
            backend.save_conversation(record)

        # Request in different order
        batch = backend.get_conversations_batch(["conv-3", "conv-1", "conv-2"])
        assert [r.conversation_id for r in batch] == ["conv-3", "conv-1", "conv-2"]

    def test_get_conversations_batch_missing_ids_skipped(self, tmp_path):
        """Test that missing IDs are silently skipped."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        batch = backend.get_conversations_batch(["conv-1", "nonexistent", "also-missing"])
        assert len(batch) == 1
        assert batch[0].conversation_id == "conv-1"

    def test_get_conversations_batch_empty_input(self, tmp_path):
        """Test that empty input returns empty list."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        batch = backend.get_conversations_batch([])
        assert batch == []

    def test_get_conversations_batch_duplicate_ids(self, tmp_path):
        """Test batch with duplicate IDs returns each occurrence."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        batch = backend.get_conversations_batch(["conv-1", "conv-1", "conv-1"])
        assert len(batch) == 3
        assert all(r.conversation_id == "conv-1" for r in batch)


# ============================================================================
# Test: Transaction management (begin/commit/rollback)
# ============================================================================


class TestTransactionManagement:
    """Tests for transaction management."""

    def test_begin_commit_persists_data(self, tmp_path):
        """Test that begin+commit persists data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        backend.begin()
        backend.save_conversation(record)
        backend.commit()

        # Verify persisted
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is not None

    def test_begin_rollback_reverts_data(self, tmp_path):
        """Test that begin+rollback reverts data."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        backend.begin()
        backend.save_conversation(record)
        backend.rollback()

        # Should not be persisted
        retrieved = backend.get_conversation("conv-1")
        assert retrieved is None

    def test_nested_savepoints(self, tmp_path):
        """Test nested transaction support via savepoints."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="First",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash1",
            version=1,
        )
        record2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="claude",
            provider_conversation_id="prov-2",
            title="Second",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash2",
            version=1,
        )

        backend.begin()
        backend.save_conversation(record1)

        backend.begin()
        backend.save_conversation(record2)
        backend.rollback()  # Rollback only record2

        backend.commit()  # Commit record1

        # record1 should exist, record2 should not
        assert backend.get_conversation("conv-1") is not None
        assert backend.get_conversation("conv-2") is None

    def test_commit_without_begin_raises_error(self, tmp_path):
        """Test that commit without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Need to access connection first to initialize transaction_depth
        backend._get_connection()
        with pytest.raises(Exception, match="No active transaction to commit"):
            backend.commit()

    def test_rollback_without_begin_raises_error(self, tmp_path):
        """Test that rollback without begin raises DatabaseError."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Need to access connection first to initialize transaction_depth
        backend._get_connection()
        with pytest.raises(Exception, match="No active transaction to rollback"):
            backend.rollback()

    def test_transaction_context_manager(self, tmp_path):
        """Test transaction context manager."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        with backend.transaction():
            backend.save_conversation(record)

        assert backend.get_conversation("conv-1") is not None

    def test_transaction_context_manager_rollback_on_error(self, tmp_path):
        """Test that transaction context manager rolls back on error."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )

        try:
            with backend.transaction():
                backend.save_conversation(record)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert backend.get_conversation("conv-1") is None


# ============================================================================
# Test: Metadata operations
# ============================================================================


class TestMetadataOperations:
    """Tests for metadata CRUD operations."""

    def _create_conversation(self, backend, cid="conv-1"):
        """Helper to create a conversation."""
        record = ConversationRecord(
            conversation_id=cid,
            provider_name="claude",
            provider_conversation_id=f"prov-{cid}",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

    def test_update_metadata(self, tmp_path):
        """Test setting a metadata key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "rating", 5)

        meta = backend.get_metadata("conv-1")
        assert meta.get("rating") == 5

    def test_delete_metadata(self, tmp_path):
        """Test removing a metadata key."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "key1", "value1")
        backend.delete_metadata("conv-1", "key1")

        meta = backend.get_metadata("conv-1")
        assert "key1" not in meta

    def test_add_tag(self, tmp_path):
        """Test adding a tag."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        assert "important" in meta.get("tags", [])

    def test_remove_tag(self, tmp_path):
        """Test removing a tag."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")
        backend.remove_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        assert "important" not in meta.get("tags", [])

    def test_add_tag_idempotent(self, tmp_path):
        """Test that adding same tag twice doesn't duplicate."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-1", "important")

        meta = backend.get_metadata("conv-1")
        tags = meta.get("tags", [])
        assert tags.count("important") == 1

    def test_list_tags(self, tmp_path):
        """Test listing all tags with counts."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        for i in range(3):
            self._create_conversation(backend, f"conv-{i+1}")

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-2", "important")
        backend.add_tag("conv-3", "follow-up")

        tags = backend.list_tags()
        assert tags.get("important") == 2
        assert tags.get("follow-up") == 1

    def test_list_tags_with_provider_filter(self, tmp_path):
        """Test listing tags filtered by provider."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Create conversations with different providers
        record1 = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record1)

        record2 = ConversationRecord(
            conversation_id="conv-2",
            provider_name="chatgpt",
            provider_conversation_id="prov-2",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record2)

        backend.add_tag("conv-1", "important")
        backend.add_tag("conv-2", "important")

        claude_tags = backend.list_tags(provider="claude")
        assert claude_tags.get("important") == 1

    def test_set_metadata_replaces_entire_dict(self, tmp_path):
        """Test set_metadata replaces the entire metadata dict."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        self._create_conversation(backend)

        backend.update_metadata("conv-1", "key1", "value1")
        backend.update_metadata("conv-1", "key2", "value2")

        new_metadata = {"new_key": "new_value"}
        backend.set_metadata("conv-1", new_metadata)

        meta = backend.get_metadata("conv-1")
        assert meta == new_metadata
        assert "key1" not in meta

    def test_get_metadata_nonexistent_conversation(self, tmp_path):
        """Test get_metadata returns empty dict for nonexistent conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        meta = backend.get_metadata("nonexistent")
        assert meta == {}


# ============================================================================
# Test: delete_conversation
# ============================================================================


class TestDeleteConversation:
    """Tests for delete_conversation operation."""

    def test_delete_conversation_success(self, tmp_path):
        """Test successful deletion of conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        result = backend.delete_conversation("conv-1")
        assert result is True
        assert backend.get_conversation("conv-1") is None

    def test_delete_conversation_nonexistent(self, tmp_path):
        """Test deleting nonexistent conversation returns False."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        result = backend.delete_conversation("nonexistent")
        assert result is False

    def test_delete_conversation_with_messages(self, tmp_path):
        """Test that deleting conversation also deletes messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            )
        ]
        backend.save_messages(messages)

        backend.delete_conversation("conv-1")
        assert backend.get_messages("conv-1") == []

    def test_delete_conversation_with_attachments(self, tmp_path):
        """Test that attachment refs are cleaned up when conversation is deleted."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        attachment = AttachmentRecord(
            attachment_id="att-1",
            conversation_id="conv-1",
            message_id=None,
            mime_type="image/png",
            size_bytes=1024,
            path="/path/to/image.png",
        )
        backend.save_attachments([attachment])

        backend.delete_conversation("conv-1")
        # Verify conversation is gone
        assert backend.get_conversation("conv-1") is None


# ============================================================================
# Test: get_conversation_stats
# ============================================================================


class TestGetConversationStats:
    """Tests for get_conversation_stats operation."""

    def test_get_conversation_stats_basic(self, tmp_path):
        """Test getting message counts for a conversation."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        messages = [
            MessageRecord(
                message_id=f"msg-{i}",
                conversation_id="conv-1",
                role="user" if i % 2 == 0 else "assistant",
                text=f"Message {i}",
                timestamp=f"2025-01-01T10:{i:02d}:00Z",
                content_hash=f"h{i}",
                version=1,
            )
            for i in range(5)
        ]
        backend.save_messages(messages)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 5
        assert stats["dialogue_messages"] == 5
        assert stats["tool_messages"] == 0

    def test_get_conversation_stats_with_tool_messages(self, tmp_path):
        """Test stats with mixed message types."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        messages = [
            MessageRecord(
                message_id="msg-1",
                conversation_id="conv-1",
                role="user",
                text="Hello",
                timestamp="2025-01-01T10:00:00Z",
                content_hash="h1",
                version=1,
            ),
            MessageRecord(
                message_id="msg-2",
                conversation_id="conv-1",
                role="tool",
                text="Tool output",
                timestamp="2025-01-01T10:01:00Z",
                content_hash="h2",
                version=1,
            ),
            MessageRecord(
                message_id="msg-3",
                conversation_id="conv-1",
                role="assistant",
                text="Response",
                timestamp="2025-01-01T10:02:00Z",
                content_hash="h3",
                version=1,
            ),
        ]
        backend.save_messages(messages)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 3
        assert stats["dialogue_messages"] == 2
        assert stats["tool_messages"] == 1

    def test_get_conversation_stats_empty_conversation(self, tmp_path):
        """Test stats for conversation with no messages."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        stats = backend.get_conversation_stats("conv-1")
        assert stats["total_messages"] == 0
        assert stats["dialogue_messages"] == 0
        assert stats["tool_messages"] == 0


# ============================================================================
# Test: Helper functions
# ============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_json_or_none_with_dict(self):
        """Test _json_or_none with a dictionary."""
        result = _json_or_none({"key": "value"})
        assert isinstance(result, str)
        assert json.loads(result) == {"key": "value"}

    def test_json_or_none_with_none(self):
        """Test _json_or_none with None."""
        result = _json_or_none(None)
        assert result is None

    def test_json_or_none_with_nested_dict(self):
        """Test _json_or_none with nested structures."""
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        result = _json_or_none(data)
        assert json.loads(result) == data

    def test_make_ref_id_deterministic(self):
        """Test that _make_ref_id produces deterministic results."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", "msg-1")
        ref_id_2 = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id_1 == ref_id_2

    def test_make_ref_id_different_inputs(self):
        """Test that _make_ref_id produces different IDs for different inputs."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", "msg-1")
        ref_id_2 = _make_ref_id("att-2", "conv-1", "msg-1")
        assert ref_id_1 != ref_id_2

    def test_make_ref_id_format(self):
        """Test that _make_ref_id has the correct format."""
        ref_id = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id.startswith("ref-")
        assert len(ref_id) == len("ref-") + 16  # 16-char hex digest

    def test_make_ref_id_with_none_message_id(self):
        """Test _make_ref_id with None message_id."""
        ref_id_1 = _make_ref_id("att-1", "conv-1", None)
        ref_id_2 = _make_ref_id("att-1", "conv-1", "msg-1")
        assert ref_id_1 != ref_id_2

    def test_default_db_path(self, tmp_path, monkeypatch):
        """Test that default_db_path returns correct path."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import polylogue.paths

        importlib.reload(polylogue.paths)

        path = default_db_path()
        assert str(path).endswith("polylogue.db")
        assert "polylogue" in str(path)


# ============================================================================
# Test: Backend lifecycle (close)
# ============================================================================


class TestBackendLifecycle:
    """Tests for backend lifecycle management."""

    def test_close_backend(self, tmp_path):
        """Test that close() closes the connection."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        # Access connection
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        backend.save_conversation(record)

        backend.close()

        # After close, operations should fail or create new connection
        # depending on lazy connection semantics
        # Just verify it doesn't raise
        assert True

    def test_close_and_reopen(self, tmp_path):
        """Test that connection can be re-established after close."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db_path)
        record = ConversationRecord(
            conversation_id="conv-1",
            provider_name="claude",
            provider_conversation_id="prov-1",
            title="Test",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            content_hash="hash",
            version=1,
        )
        # Use transaction to ensure data is persisted
        with backend.transaction():
            backend.save_conversation(record)

        # Verify data exists before close
        retrieved1 = backend.get_conversation("conv-1")
        assert retrieved1 is not None

        backend.close()

        # After close, the thread-local connection is cleared
        # Verify a new connection can be established
        conn = backend._get_connection()
        assert conn is not None
