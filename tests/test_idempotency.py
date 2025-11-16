"""Comprehensive idempotency tests for Polylogue operations.

Tests that operations can be safely re-run after interruption without
data loss or duplication.
"""
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.conversation import process_conversation
from polylogue.db import replace_messages
from polylogue.document_store import persist_document
from polylogue.render import MarkdownDocument
from polylogue.services.conversation_registrar import ConversationRegistrar


class TestDatabaseIdempotency:
    """Test database operations are atomic and idempotent."""

    def test_replace_messages_atomic_on_interruption(self, tmp_path):
        """Verify replace_messages uses SAVEPOINT to prevent data loss."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create schema
        conn.execute("CREATE TABLE messages (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, parent_id TEXT, position INT, timestamp TEXT, role TEXT, content_hash TEXT, rendered_text TEXT, raw_json TEXT, token_count INT, word_count INT, attachment_count INT, metadata_json TEXT)")
        conn.execute("CREATE TABLE messages_fts (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, content TEXT)")

        # Insert initial messages
        initial_messages = [
            {"message_id": "msg1", "rendered_text": "Hello"},
            {"message_id": "msg2", "rendered_text": "World"},
        ]

        for msg in initial_messages:
            conn.execute(
                "INSERT INTO messages (provider, conversation_id, branch_id, message_id, parent_id, position, timestamp, role, content_hash, rendered_text, raw_json, token_count, word_count, attachment_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test", "conv1", "main", msg["message_id"], None, 0, None, "user", None, msg["rendered_text"], None, 0, 0, 0, None)
            )
        conn.commit()

        # Verify initial state
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 2

        # Create a trigger that will cause INSERT to fail
        # This simulates a crash or error during the INSERT phase
        conn.execute("""
            CREATE TRIGGER prevent_insert
            BEFORE INSERT ON messages
            BEGIN
                SELECT RAISE(ABORT, 'Simulated failure');
            END;
        """)
        conn.commit()

        new_messages = [
            {"message_id": "msg3", "rendered_text": "New1", "parent_id": None,
             "position": 0, "timestamp": None, "role": "user", "content_hash": None,
             "raw_json": None, "token_count": 0, "word_count": 0, "attachment_count": 0,
             "metadata": None},
        ]

        # This should fail due to trigger
        try:
            replace_messages(
                conn,
                provider="test",
                conversation_id="conv1",
                branch_id="main",
                messages=new_messages
            )
        except sqlite3.IntegrityError:
            pass  # Expected due to trigger

        # Verify: OLD messages still present (SAVEPOINT rolled back)
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id='conv1'").fetchone()[0]
        assert count == 2, "Old messages should still exist after rollback"

        messages = conn.execute("SELECT message_id FROM messages ORDER BY message_id").fetchall()
        assert [m[0] for m in messages] == ["msg1", "msg2"]

    def test_replace_messages_commits_on_success(self, tmp_path):
        """Verify replace_messages commits when successful."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        
        conn.execute("CREATE TABLE messages (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, parent_id TEXT, position INT, timestamp TEXT, role TEXT, content_hash TEXT, rendered_text TEXT, raw_json TEXT, token_count INT, word_count INT, attachment_count INT, metadata_json TEXT)")
        conn.execute("CREATE TABLE messages_fts (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, content TEXT)")
        
        initial_messages = [{"message_id": "msg1", "rendered_text": "Hello"}]
        for msg in initial_messages:
            conn.execute(
                "INSERT INTO messages (provider, conversation_id, branch_id, message_id, parent_id, position, timestamp, role, content_hash, rendered_text, raw_json, token_count, word_count, attachment_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test", "conv1", "main", msg["message_id"], None, 0, None, "user", None, msg["rendered_text"], None, 0, 0, 0, None)
            )
        conn.commit()
        
        # Replace messages successfully
        new_messages = [
            {
                "message_id": "msg2",
                "parent_id": None,
                "position": 0,
                "timestamp": None,
                "role": "assistant",
                "content_hash": None,
                "rendered_text": "Replaced",
                "raw_json": None,
                "token_count": 1,
                "word_count": 1,
                "attachment_count": 0,
                "metadata": None,
            }
        ]
        
        replace_messages(
            conn,
            provider="test",
            conversation_id="conv1",
            branch_id="main",
            messages=new_messages
        )
        
        # Verify new messages present
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id='conv1'").fetchone()[0]
        assert count == 1
        
        messages = conn.execute("SELECT message_id, rendered_text FROM messages").fetchall()
        assert messages[0] == ("msg2", "Replaced")


class TestForceAndAllowDirty:
    """Test --force and --allow-dirty flag interaction."""

    def test_force_alone_rejects_dirty_files(self, tmp_path):
        """Verify --force without --allow-dirty raises error on dirty files."""
        # Create conversation directory and file
        conv_dir = tmp_path / "test-conv"
        conv_dir.mkdir()
        md_path = conv_dir / "conversation.md"
        md_path.write_text("---\npolylogue:\n  localHash: abc123\n---\nOriginal content")

        # Create state with different hash (simulating user edit)
        registrar = Mock()
        registrar.get_state.return_value = {
            "localHash": "different",  # Different hash = dirty
            "contentHash": "xyz",  # Different from actual content hash
            "lastUpdated": "2024-01-01T00:00:00Z",
        }

        doc = MarkdownDocument(
            body="New content from remote",
            metadata={"polylogue": {}},
            attachments=[],
            stats={},
        )

        # Attempt persist with force=True but allow_dirty=False
        with pytest.raises(ValueError, match="local edits"):
            persist_document(
                provider="test",
                conversation_id="conv1",
                title="Test",
                document=doc,
                output_dir=tmp_path,
                collapse_threshold=24,
                attachments=[],
                updated_at="2024-01-01T00:00:00Z",
                created_at="2024-01-01T00:00:00Z",
                html=False,
                html_theme="light",
                force=True,
                allow_dirty=False,  # Should raise error
                registrar=registrar,
                slug_hint="test-conv",  # Control the slug
            )

    def test_force_with_allow_dirty_overwrites(self, tmp_path):
        """Verify --force --allow-dirty overwrites dirty files."""
        # Create conversation directory and file
        conv_dir = tmp_path / "test-conv"
        conv_dir.mkdir()
        md_path = conv_dir / "conversation.md"
        md_path.write_text("---\npolylogue:\n  localHash: abc123\n---\nUser edited content")

        registrar = Mock()
        registrar.get_state.return_value = {
            "localHash": "different",
            "contentHash": "xyz",  # Different from actual content hash
            "lastUpdated": "2024-01-01T00:00:00Z",
        }

        doc = MarkdownDocument(
            body="New content from remote",
            metadata={"polylogue": {}},
            attachments=[],
            stats={},
        )

        # Should succeed with both flags
        result = persist_document(
            provider="test",
            conversation_id="conv1",
            title="Test",
            document=doc,
            output_dir=tmp_path,
            collapse_threshold=24,
            attachments=[],
            updated_at="2024-01-01T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            html=False,
            html_theme="light",
            force=True,
            allow_dirty=True,  # Both flags = allow overwrite
            registrar=registrar,
            slug_hint="test-conv",  # Control the slug
        )

        assert not result.skipped
        assert "New content from remote" in md_path.read_text()


class TestSyncIdempotency:
    """Test sync operations are idempotent."""

    def test_sync_twice_produces_same_result(self, tmp_path):
        """Running sync twice on same data produces identical output."""
        # This is a higher-level test that would require more setup
        # For now, documenting the test structure
        pytest.skip("Requires full sync infrastructure setup")

    def test_interrupted_sync_resumable(self, tmp_path):
        """Interrupted sync can resume without duplicating work."""
        pytest.skip("Requires sync pipeline mocking")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
