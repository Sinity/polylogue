"""Integration tests for health repair workflows against a real database."""

from __future__ import annotations

import hashlib

RUN_ALL_REPAIRS_EXPECTED = {
    "orphaned_messages",
    "empty_conversations",
    "dangling_fts",
    "orphaned_attachments",
    "wal_checkpoint",
}


def _insert_conversation(conn, conversation_id: str, provider_name: str = "test", title: str = "Test"):
    """Helper to insert a conversation with all required fields."""
    content_hash = hashlib.sha256(f"{provider_name}:{conversation_id}".encode()).hexdigest()
    conn.execute(
        "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, title, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
        (conversation_id, provider_name, f"{provider_name}-id", title, content_hash, 1),
    )


def _insert_message(
    conn,
    message_id: str,
    conversation_id: str,
    role: str = "user",
    text: str = "Text",
    allow_orphaned: bool = False,
):
    """Helper to insert a message with all required fields."""
    content_hash = hashlib.sha256(f"{message_id}:{text}".encode()).hexdigest()
    if allow_orphaned:
        # Temporarily disable foreign keys to insert orphaned message
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, text, content_hash, 1),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, text, content_hash, 1),
        )


class TestRepairOrphanedMessages:
    """Tests for repair_orphaned_messages function."""

    def test_clean_state_no_orphaned_messages(self, cli_workspace):
        """repair_orphaned_messages should return 0 when no orphans exist."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_messages
        from polylogue.storage.backends.connection import connection_context

        # Setup: create a valid conversation and message
        config = get_config()
        with connection_context(None) as conn:
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1")
            conn.commit()

        # Act
        result = repair_orphaned_messages(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_messages"
        assert result.repaired_count == 0
        assert result.success is True
        assert "No orphaned" in result.detail

    def test_orphaned_messages_found_and_deleted(self, cli_workspace):
        """repair_orphaned_messages should find and delete orphaned messages."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_messages
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert orphaned message (references non-existent conversation)
            _insert_message(conn, "orphan-1", "nonexistent-conv", "user", "I am orphaned", allow_orphaned=True)
            _insert_message(conn, "orphan-2", "nonexistent-conv", "assistant", "Also orphaned", allow_orphaned=True)
            conn.commit()

        # Act
        result = repair_orphaned_messages(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_messages"
        assert result.repaired_count == 2
        assert result.success is True

        # Verify they were actually deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert remaining == 0

    def test_orphaned_messages_dry_run_counts_but_doesnt_delete(self, cli_workspace):
        """repair_orphaned_messages with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_messages
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_message(conn, "orphan-1", "nonexistent-conv", "user", "I am orphaned", allow_orphaned=True)
            conn.commit()

        # Act
        result = repair_orphaned_messages(config, dry_run=True)

        # Assert
        assert result.repaired_count == 1
        assert result.success is True
        assert "Would:" in result.detail

        # Verify data was NOT deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert remaining == 1

    def test_orphaned_messages_idempotency(self, cli_workspace):
        """Running repair twice should find 0 on second run."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_messages
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_message(conn, "orphan-1", "nonexistent-conv", "user", "I am orphaned", allow_orphaned=True)
            conn.commit()

        # First repair
        result1 = repair_orphaned_messages(config, dry_run=False)
        assert result1.repaired_count == 1

        # Second repair should find nothing
        result2 = repair_orphaned_messages(config, dry_run=False)
        assert result2.repaired_count == 0
        assert result2.success is True


class TestRepairEmptyConversations:
    """Tests for repair_empty_conversations function."""

    def test_clean_state_no_empty_conversations(self, cli_workspace):
        """repair_empty_conversations should return 0 when no empty convos exist."""
        from polylogue.config import get_config
        from polylogue.health import repair_empty_conversations
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create conversation with message
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Hello")
            conn.commit()

        # Act
        result = repair_empty_conversations(config, dry_run=False)

        # Assert
        assert result.name == "empty_conversations"
        assert result.repaired_count == 0
        assert result.success is True

    def test_empty_conversations_found_and_deleted(self, cli_workspace):
        """repair_empty_conversations should find and delete empty conversations."""
        from polylogue.config import get_config
        from polylogue.health import repair_empty_conversations
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create empty conversations (no messages)
            _insert_conversation(conn, "empty-1")
            _insert_conversation(conn, "empty-2")
            conn.commit()

        # Act
        result = repair_empty_conversations(config, dry_run=False)

        # Assert
        assert result.name == "empty_conversations"
        assert result.repaired_count == 2
        assert result.success is True

        # Verify they were deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert remaining == 0

    def test_empty_conversations_dry_run_counts_but_doesnt_delete(self, cli_workspace):
        """repair_empty_conversations with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.health import repair_empty_conversations
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_conversation(conn, "empty-1")
            conn.commit()

        # Act
        result = repair_empty_conversations(config, dry_run=True)

        # Assert
        assert result.repaired_count == 1
        assert "Would:" in result.detail

        # Verify not deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert remaining == 1

    def test_empty_conversations_idempotency(self, cli_workspace):
        """Running repair twice should find 0 on second run."""
        from polylogue.config import get_config
        from polylogue.health import repair_empty_conversations
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_conversation(conn, "empty-1")
            conn.commit()

        # First repair
        result1 = repair_empty_conversations(config, dry_run=False)
        assert result1.repaired_count == 1

        # Second repair should find nothing
        result2 = repair_empty_conversations(config, dry_run=False)
        assert result2.repaired_count == 0


class TestRepairDanglingFts:
    """Tests for repair_dangling_fts function."""

    def test_clean_fts_state(self, cli_workspace):
        """repair_dangling_fts should return 0 when FTS is in sync."""
        from polylogue.config import get_config
        from polylogue.health import repair_dangling_fts
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create conversation and message (FTS auto-syncs)
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Searchable text")
            conn.commit()

        # Act
        result = repair_dangling_fts(config, dry_run=False)

        # Assert
        assert result.name == "dangling_fts"
        assert result.repaired_count == 0
        assert result.success is True

    def test_dangling_fts_orphaned_entry(self, cli_workspace):
        """repair_dangling_fts should find and delete FTS entries without messages."""
        from polylogue.config import get_config
        from polylogue.health import repair_dangling_fts
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert a message first (creates FTS entry via trigger)
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            conn.commit()

            # Get the rowid that was assigned
            rowid = conn.execute("SELECT rowid FROM messages WHERE message_id = ?", ("msg-1",)).fetchone()[0]

            # Now manually delete the message but manually insert FTS entry to simulate orphaned state
            conn.execute("DELETE FROM messages WHERE message_id = ?", ("msg-1",))

            # Manually insert a dangling FTS entry (rowid won't match any message)
            conn.execute(
                "INSERT INTO messages_fts (rowid, message_id, conversation_id, text) VALUES (?, ?, ?, ?)",
                (rowid + 999, "orphan-fts", "conv-1", "dangling entry"),
            )
            conn.commit()

        # Act
        result = repair_dangling_fts(config, dry_run=False)

        # Assert
        assert result.name == "dangling_fts"
        assert result.repaired_count == 1
        assert result.success is True

    def test_dangling_fts_missing_entry(self, cli_workspace):
        """repair_dangling_fts should insert missing FTS entries for messages."""
        from polylogue.config import get_config
        from polylogue.health import repair_dangling_fts
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert conversation and message
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text1")
            conn.commit()

            # Disable triggers temporarily and delete FTS entry
            conn.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
            conn.execute("DELETE FROM messages_fts WHERE message_id = ?", ("msg-1",))
            conn.commit()

            # Re-create the trigger
            conn.execute(
                """CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                   DELETE FROM messages_fts WHERE rowid = OLD.rowid;
                END"""
            )
            conn.commit()

        # Act
        result = repair_dangling_fts(config, dry_run=False)

        # Assert
        assert result.name == "dangling_fts"
        assert result.repaired_count == 1
        assert result.success is True

    def test_dangling_fts_dry_run_counts_but_doesnt_modify(self, cli_workspace):
        """repair_dangling_fts with dry_run=True should count but not modify."""
        from polylogue.config import get_config
        from polylogue.health import repair_dangling_fts
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            _insert_message(conn, "msg-2", "conv-1", "user", "Text2")
            conn.commit()

            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            initial_diff = abs(msg_count - fts_count)

        # Act
        result = repair_dangling_fts(config, dry_run=True)

        # Assert - dry_run returns count estimate even if FTS is in sync
        assert result.success is True

        # Verify no actual change
        with connection_context(None) as conn:
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            assert abs(msg_count - fts_count) == initial_diff


class TestRepairOrphanedAttachments:
    """Tests for repair_orphaned_attachments function."""

    def test_clean_attachments_state(self, cli_workspace):
        """repair_orphaned_attachments should return 0 when all attachments are referenced."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_attachments
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create conversation, message, attachment, and reference
            _insert_conversation(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                ("ref-1", "att-1", "conv-1", "msg-1"),
            )
            conn.commit()

        # Act
        result = repair_orphaned_attachments(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_attachments"
        assert result.repaired_count == 0
        assert result.success is True

    def test_orphaned_attachments_orphaned_refs(self, cli_workspace):
        """repair_orphaned_attachments should detect orphaned attachment refs in dry-run."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_attachments
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create attachment with no references
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            _insert_conversation(conn, "conv-1")
            # Insert a proper ref
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                ("ref-1", "att-1", "conv-1", None),
            )
            conn.commit()

        # Act - create another attachment that's unreferenced
        with connection_context(None) as conn:
            conn.execute("INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)", ("att-2", "image/png"))
            conn.commit()

        # Now run repair
        result = repair_orphaned_attachments(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_attachments"
        # Should delete at least the unreferenced attachment
        assert result.repaired_count >= 1

    def test_orphaned_attachments_unreferenced_attachment(self, cli_workspace):
        """repair_orphaned_attachments should delete unreferenced attachments."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_attachments
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create attachment with no references
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            conn.commit()

        # Act
        result = repair_orphaned_attachments(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_attachments"
        assert result.repaired_count >= 1  # Should delete the unreferenced attachment

    def test_orphaned_attachments_dry_run(self, cli_workspace):
        """repair_orphaned_attachments with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.health import repair_orphaned_attachments
        from polylogue.storage.backends.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            conn.commit()

        # Act
        result = repair_orphaned_attachments(config, dry_run=True)

        # Assert
        assert "Would:" in result.detail

        # Verify not deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert remaining == 1


class TestRepairWalCheckpoint:
    """Tests for repair_wal_checkpoint function."""

    def test_wal_checkpoint_succeeds(self, cli_workspace):
        """repair_wal_checkpoint should execute without crashing."""
        from polylogue.config import get_config
        from polylogue.health import repair_wal_checkpoint

        config = get_config()

        # Act
        result = repair_wal_checkpoint(config, dry_run=False)

        # Assert
        assert result.name == "wal_checkpoint"
        assert result.success is True
        # repaired_count and detail depend on WAL file state; just verify no crash

    def test_wal_checkpoint_dry_run(self, cli_workspace):
        """repair_wal_checkpoint with dry_run=True should inspect WAL without modifying."""
        from polylogue.config import get_config
        from polylogue.health import repair_wal_checkpoint

        config = get_config()

        # Act
        result = repair_wal_checkpoint(config, dry_run=True)

        # Assert
        assert result.name == "wal_checkpoint"
        assert result.success is True
        assert "Would:" in result.detail or "nothing to checkpoint" in result.detail


class TestRunAllRepairs:
    """Tests for run_all_repairs orchestration."""

    def test_run_all_repairs_returns_all_results(self, cli_workspace):
        """run_all_repairs should return results for all repair functions."""
        from polylogue.config import get_config
        from polylogue.health import run_all_repairs

        config = get_config()

        # Act
        results = run_all_repairs(config, dry_run=False)

        # Assert
        assert isinstance(results, list)
        assert len(results) == 5  # All current repair functions

        # Verify we got each repair type
        repair_names = {r.name for r in results}
        assert repair_names == RUN_ALL_REPAIRS_EXPECTED

    def test_run_all_repairs_all_success(self, cli_workspace):
        """run_all_repairs should have all successful on clean database."""
        from polylogue.config import get_config
        from polylogue.health import run_all_repairs

        config = get_config()

        # Act
        results = run_all_repairs(config, dry_run=False)

        # Assert
        assert all(r.success for r in results), f"Some repairs failed: {results}"

    def test_run_all_repairs_dry_run_all_have_would(self, cli_workspace):
        """run_all_repairs with dry_run=True should have 'Would:' in details."""
        from polylogue.config import get_config
        from polylogue.health import run_all_repairs
        from polylogue.storage.backends.connection import connection_context

        config = get_config()

        # Setup: add some issues
        with connection_context(None) as conn:
            _insert_message(conn, "orphan-1", "nonexistent-conv", "user", "Orphaned", allow_orphaned=True)
            conn.commit()

        # Act
        results = run_all_repairs(config, dry_run=True)

        # Assert
        # Some should have "Would:" when issues are found
        assert any("Would:" in r.detail for r in results), "Expected at least one 'Would:' in dry-run results"

    def test_run_all_repairs_idempotency_after_full_run(self, cli_workspace):
        """run_all_repairs twice should find 0 issues on second run."""
        from polylogue.config import get_config
        from polylogue.health import run_all_repairs
        from polylogue.storage.backends.connection import connection_context

        config = get_config()

        # Setup: add issues
        with connection_context(None) as conn:
            _insert_conversation(conn, "empty-1")
            conn.commit()

        # First repair run
        results1 = run_all_repairs(config, dry_run=False)
        total_repaired_1 = sum(r.repaired_count for r in results1)
        assert total_repaired_1 > 0

        # Second repair run should find nothing
        results2 = run_all_repairs(config, dry_run=False)
        total_repaired_2 = sum(r.repaired_count for r in results2)
        assert total_repaired_2 == 0
