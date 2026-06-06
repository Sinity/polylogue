"""Integration tests for health repair workflows against a real database."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TypeAlias

CliWorkspace: TypeAlias = dict[str, Path]

RUN_ALL_REPAIRS_EXPECTED = {
    "session_insights",
    "orphaned_messages",
    "orphaned_content_blocks",
    "empty_sessions",
    "action_event_read_model",
    "dangling_fts",
    "orphaned_attachments",
    "wal_checkpoint",
}


def _insert_session(conn: sqlite3.Connection, session_id: str, source_name: str = "test", title: str = "Test") -> None:
    """Helper to insert a session with all required fields."""
    content_hash = hashlib.sha256(f"{source_name}:{session_id}".encode()).hexdigest()
    conn.execute(
        "INSERT INTO sessions (session_id, source_name, provider_session_id, title, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, source_name, f"{source_name}-id", title, content_hash, 1),
    )


def _insert_message(
    conn: sqlite3.Connection,
    message_id: str,
    session_id: str,
    role: str = "user",
    text: str = "Text",
    allow_orphaned: bool = False,
) -> None:
    """Helper to insert a message with all required fields."""
    content_hash = hashlib.sha256(f"{message_id}:{text}".encode()).hexdigest()
    if allow_orphaned:
        # Temporarily disable foreign keys to insert orphaned message
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                "INSERT INTO messages (message_id, session_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, session_id, role, text, content_hash, 1),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        conn.execute(
            "INSERT INTO messages (message_id, session_id, role, text, content_hash, version) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, text, content_hash, 1),
        )


def _insert_content_block(
    conn: sqlite3.Connection,
    block_id: str,
    message_id: str,
    session_id: str,
    *,
    allow_orphaned: bool = False,
    type: str = "tool_use",
    tool_name: str | None = "Read",
    text: str | None = None,
) -> None:
    """Helper to insert a content block with optional broken parent references."""
    if allow_orphaned:
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                """
                INSERT INTO content_blocks (
                    block_id, message_id, session_id, block_index, type, text, tool_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (block_id, message_id, session_id, 0, type, text, tool_name),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        conn.execute(
            """
            INSERT INTO content_blocks (
                block_id, message_id, session_id, block_index, type, text, tool_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (block_id, message_id, session_id, 0, type, text, tool_name),
        )


class TestRepairOrphanedMessages:
    """Tests for repair_orphaned_messages function."""

    def test_clean_state_no_orphaned_messages(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_messages should return 0 when no orphans exist."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_messages
        from polylogue.storage.sqlite.connection import connection_context

        # Setup: create a valid session and message
        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1")
            conn.commit()

        # Act
        result = repair_orphaned_messages(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_messages"
        assert result.repaired_count == 0
        assert result.success is True
        assert "No orphaned" in result.detail

    def test_orphaned_messages_found_and_deleted(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_messages should find and delete orphaned messages."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_messages
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert orphaned message (references non-existent session)
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

    def test_orphaned_messages_dry_run_counts_but_doesnt_delete(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_messages with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_messages
        from polylogue.storage.sqlite.connection import connection_context

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

    def test_orphaned_messages_idempotency(self, cli_workspace: CliWorkspace) -> None:
        """Running repair twice should find 0 on second run."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_messages
        from polylogue.storage.sqlite.connection import connection_context

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


class TestRepairOrphanedContentBlocks:
    """Tests for repair_orphaned_content_blocks function."""

    def test_clean_state_no_orphaned_content_blocks(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_content_blocks
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1")
            _insert_content_block(conn, "blk-1", "msg-1", "conv-1")
            conn.commit()

        result = repair_orphaned_content_blocks(config, dry_run=False)

        assert result.name == "orphaned_content_blocks"
        assert result.repaired_count == 0
        assert result.success is True

    def test_orphaned_content_blocks_found_and_deleted(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_content_blocks
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_content_block(
                conn,
                "blk-orphan-conv",
                "missing-msg",
                "missing-conv",
                allow_orphaned=True,
            )
            _insert_session(conn, "conv-1")
            _insert_content_block(
                conn,
                "blk-orphan-msg",
                "missing-msg-2",
                "conv-1",
                allow_orphaned=True,
            )
            conn.commit()

        result = repair_orphaned_content_blocks(config, dry_run=False)

        assert result.name == "orphaned_content_blocks"
        assert result.repaired_count == 2
        assert result.success is True

        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
            assert remaining == 0

    def test_orphaned_content_blocks_dry_run_counts_but_doesnt_delete(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_content_blocks
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_content_block(
                conn,
                "blk-orphan-conv",
                "missing-msg",
                "missing-conv",
                allow_orphaned=True,
            )
            conn.commit()

        result = repair_orphaned_content_blocks(config, dry_run=True)

        assert result.repaired_count == 1
        assert result.success is True
        assert "Would:" in result.detail

        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
            assert remaining == 1

    def test_orphaned_content_blocks_idempotency(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_content_blocks
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_content_block(
                conn,
                "blk-orphan-conv",
                "missing-msg",
                "missing-conv",
                allow_orphaned=True,
            )
            conn.commit()

        result1 = repair_orphaned_content_blocks(config, dry_run=False)
        assert result1.repaired_count == 1

        result2 = repair_orphaned_content_blocks(config, dry_run=False)
        assert result2.repaired_count == 0
        assert result2.success is True


class TestRepairEmptySessions:
    """Tests for repair_empty_sessions function."""

    def test_clean_state_no_empty_sessions(self, cli_workspace: CliWorkspace) -> None:
        """repair_empty_sessions should return 0 when no empty convos exist."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_empty_sessions
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create session with message
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Hello")
            conn.commit()

        # Act
        result = repair_empty_sessions(config, dry_run=False)

        # Assert
        assert result.name == "empty_sessions"
        assert result.repaired_count == 0
        assert result.success is True

    def test_empty_sessions_found_and_deleted(self, cli_workspace: CliWorkspace) -> None:
        """repair_empty_sessions should find and delete empty sessions."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_empty_sessions
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create empty sessions (no messages)
            _insert_session(conn, "empty-1")
            _insert_session(conn, "empty-2")
            conn.commit()

        # Act
        result = repair_empty_sessions(config, dry_run=False)

        # Assert
        assert result.name == "empty_sessions"
        assert result.repaired_count == 2
        assert result.success is True

        # Verify they were deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            assert remaining == 0

    def test_empty_sessions_dry_run_counts_but_doesnt_delete(self, cli_workspace: CliWorkspace) -> None:
        """repair_empty_sessions with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_empty_sessions
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "empty-1")
            conn.commit()

        # Act
        result = repair_empty_sessions(config, dry_run=True)

        # Assert
        assert result.repaired_count == 1
        assert "Would:" in result.detail

        # Verify not deleted
        with connection_context(None) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            assert remaining == 1

    def test_empty_sessions_idempotency(self, cli_workspace: CliWorkspace) -> None:
        """Running repair twice should find 0 on second run."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_empty_sessions
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "empty-1")
            conn.commit()

        # First repair
        result1 = repair_empty_sessions(config, dry_run=False)
        assert result1.repaired_count == 1

        # Second repair should find nothing
        result2 = repair_empty_sessions(config, dry_run=False)
        assert result2.repaired_count == 0


class TestRepairDanglingFts:
    """Tests for repair_dangling_fts function."""

    def test_clean_fts_state(self, cli_workspace: CliWorkspace) -> None:
        """repair_dangling_fts should return 0 when FTS is in sync."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_dangling_fts
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create session and message (FTS auto-syncs)
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Searchable text")
            conn.commit()

        # Act
        result = repair_dangling_fts(config, dry_run=False)

        # Assert
        assert result.name == "dangling_fts"
        assert result.repaired_count == 0
        assert result.success is True

    def test_dangling_fts_orphaned_entry(self, cli_workspace: CliWorkspace) -> None:
        """repair_dangling_fts should find and delete FTS entries without messages."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_dangling_fts
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert a message first (creates FTS entry via trigger)
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            conn.commit()

            # Get the rowid that was assigned
            rowid = conn.execute("SELECT rowid FROM messages WHERE message_id = ?", ("msg-1",)).fetchone()[0]

            # Now manually delete the message but manually insert FTS entry to simulate orphaned state
            conn.execute("DELETE FROM messages WHERE message_id = ?", ("msg-1",))

            # Manually insert a dangling FTS entry (rowid won't match any message)
            conn.execute(
                "INSERT INTO messages_fts (rowid, message_id, session_id, text) VALUES (?, ?, ?, ?)",
                (rowid + 999, "orphan-fts", "conv-1", "dangling entry"),
            )
            conn.commit()

        # Act
        result = repair_dangling_fts(config, dry_run=False)

        # Assert
        assert result.name == "dangling_fts"
        assert result.repaired_count == 1
        assert result.success is True

    def test_dangling_fts_missing_entry(self, cli_workspace: CliWorkspace) -> None:
        """repair_dangling_fts should insert missing FTS entries for messages."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_dangling_fts
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Insert session and message
            _insert_session(conn, "conv-1")
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

    def test_dangling_fts_dry_run_counts_but_doesnt_modify(self, cli_workspace: CliWorkspace) -> None:
        """repair_dangling_fts with dry_run=True should count but not modify."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_dangling_fts
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
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

    def test_dangling_fts_ignores_null_text_messages(self, cli_workspace: CliWorkspace) -> None:
        """Null-text messages should not count as missing FTS rows."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_dangling_fts
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            _insert_message(conn, "msg-2", "conv-1", "assistant", "Placeholder")
            conn.execute("UPDATE messages SET text = NULL WHERE message_id = ?", ("msg-2",))
            conn.commit()

            indexed_rows = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            assert indexed_rows == 1

        result = repair_dangling_fts(config, dry_run=True)

        assert result.success is True
        assert result.repaired_count == 0
        assert result.detail == "FTS index in sync"


class TestRepairActionEventReadModel:
    """Tests for repair_action_event_read_model function."""

    def test_clean_action_event_state(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_action_event_read_model
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", text="Read file")
            _insert_content_block(conn, "blk-1", "msg-1", "conv-1", tool_name="Read")
            conn.commit()

        result = repair_action_event_read_model(config, dry_run=False)

        assert result.name == "action_event_read_model"
        assert result.success is True

        with connection_context(None) as conn:
            assert conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0] > 0
            assert conn.execute("SELECT COUNT(*) FROM action_events_fts").fetchone()[0] > 0

    def test_action_event_read_model_dry_run(self, cli_workspace: CliWorkspace) -> None:
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_action_event_read_model
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", text="Read file")
            _insert_content_block(conn, "blk-1", "msg-1", "conv-1", tool_name="Read")
            conn.commit()

        result = repair_action_event_read_model(config, dry_run=True)

        assert result.name == "action_event_read_model"
        assert result.success is True
        assert "Would:" in result.detail


class TestRepairOrphanedAttachments:
    """Tests for repair_orphaned_attachments function."""

    def test_clean_attachments_state(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_attachments should return 0 when all attachments are referenced."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_attachments
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create session, message, attachment, and reference
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1", "user", "Text")
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, session_id, message_id) VALUES (?, ?, ?, ?)",
                ("ref-1", "att-1", "conv-1", "msg-1"),
            )
            conn.commit()

        # Act
        result = repair_orphaned_attachments(config, dry_run=False)

        # Assert
        assert result.name == "orphaned_attachments"
        assert result.repaired_count == 0
        assert result.success is True

    def test_orphaned_attachments_orphaned_refs(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_attachments should detect orphaned attachment refs in dry-run."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_attachments
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()
        with connection_context(None) as conn:
            # Create attachment with no references
            conn.execute(
                "INSERT INTO attachments (attachment_id, mime_type) VALUES (?, ?)",
                ("att-1", "text/plain"),
            )
            _insert_session(conn, "conv-1")
            # Insert a proper ref
            conn.execute(
                "INSERT INTO attachment_refs (ref_id, attachment_id, session_id, message_id) VALUES (?, ?, ?, ?)",
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

    def test_orphaned_attachments_unreferenced_attachment(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_attachments should delete unreferenced attachments."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_attachments
        from polylogue.storage.sqlite.connection import connection_context

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

    def test_orphaned_attachments_dry_run(self, cli_workspace: CliWorkspace) -> None:
        """repair_orphaned_attachments with dry_run=True should count but not delete."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_orphaned_attachments
        from polylogue.storage.sqlite.connection import connection_context

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

    def test_wal_checkpoint_succeeds(self, cli_workspace: CliWorkspace) -> None:
        """repair_wal_checkpoint should execute without crashing."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_wal_checkpoint

        config = get_config()

        # Act
        result = repair_wal_checkpoint(config, dry_run=False)

        # Assert
        assert result.name == "wal_checkpoint"
        assert result.success is True
        # repaired_count and detail depend on WAL file state; just verify no crash

    def test_wal_checkpoint_dry_run(self, cli_workspace: CliWorkspace) -> None:
        """repair_wal_checkpoint with dry_run=True should inspect WAL without modifying."""
        from polylogue.config import get_config
        from polylogue.storage.repair import repair_wal_checkpoint

        config = get_config()

        # Act
        result = repair_wal_checkpoint(config, dry_run=True)

        # Assert
        assert result.name == "wal_checkpoint"
        assert result.success is True
        assert "Would:" in result.detail or "nothing to checkpoint" in result.detail


class TestMaintenanceSelection:
    """Tests for safe repair and archive cleanup orchestration."""

    def test_run_safe_repairs_returns_safe_results(self, cli_workspace: CliWorkspace) -> None:
        """run_safe_repairs should return only non-destructive maintenance functions."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_safe_repairs

        config = get_config()

        results = run_safe_repairs(config, dry_run=False)

        assert isinstance(results, list)
        assert {r.name for r in results} == {
            "session_insights",
            "action_event_read_model",
            "dangling_fts",
            "wal_checkpoint",
        }
        assert all(r.destructive is False for r in results)
        assert all(r.category.value in {"derived_repair", "database_maintenance"} for r in results)

    def test_run_archive_cleanup_returns_destructive_results(self, cli_workspace: CliWorkspace) -> None:
        """run_archive_cleanup should return only destructive cleanup functions."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_archive_cleanup

        config = get_config()
        results = run_archive_cleanup(config, dry_run=False)
        assert {r.name for r in results} == {
            "orphaned_messages",
            "orphaned_content_blocks",
            "empty_sessions",
            "orphaned_attachments",
        }
        assert all(r.destructive is True for r in results)
        assert all(r.category.value == "archive_cleanup" for r in results)

    def test_run_selected_maintenance_dry_run_all_have_would(self, cli_workspace: CliWorkspace) -> None:
        """Selected maintenance dry run should describe pending work explicitly."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_selected_maintenance
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()

        # Setup: add some issues
        with connection_context(None) as conn:
            _insert_message(conn, "orphan-1", "nonexistent-conv", "user", "Orphaned", allow_orphaned=True)
            conn.commit()

        results = run_selected_maintenance(config, repair=True, cleanup=True, dry_run=True)

        assert any("Would:" in r.detail for r in results), "Expected at least one 'Would:' in dry-run results"

    def test_run_selected_maintenance_uses_preview_counts(self, cli_workspace: CliWorkspace) -> None:
        """Dry-run maintenance can reuse known counts instead of rescanning the archive."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_selected_maintenance

        config = get_config()
        results = run_selected_maintenance(
            config,
            repair=True,
            cleanup=True,
            dry_run=True,
            preview_counts={
                "session_insights": 13,
                "action_event_read_model": 7,
                "dangling_fts": 3,
                "orphaned_messages": 2,
                "orphaned_content_blocks": 11,
                "empty_sessions": 5,
            },
        )

        by_name = {result.name: result for result in results}
        assert by_name["session_insights"].repaired_count == 13
        assert by_name["action_event_read_model"].repaired_count == 7
        assert by_name["dangling_fts"].repaired_count == 3
        assert by_name["orphaned_messages"].repaired_count == 2
        assert by_name["orphaned_content_blocks"].repaired_count == 11
        assert by_name["empty_sessions"].repaired_count == 5

    def test_run_selected_maintenance_can_scope_to_session_insights(self, cli_workspace: CliWorkspace) -> None:
        """Scoped maintenance should repair only the durable session-insight layer."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_selected_maintenance
        from polylogue.storage.sqlite.connection import connection_context
        from tests.infra.storage_records import SessionBuilder

        config = get_config()
        db_path = cli_workspace["db_path"]
        (
            SessionBuilder(db_path, "conv-insights")
            .provider("claude-code")
            .title("Scoped Insight Repair")
            .add_message("u1", role="user", text="Plan the change")
            .save()
        )

        results = run_selected_maintenance(
            config,
            repair=True,
            cleanup=False,
            dry_run=False,
            targets=("session_insights",),
        )

        assert [result.name for result in results] == ["session_insights"]
        with connection_context(None) as conn:
            profile_count = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]
            thread_count = conn.execute("SELECT COUNT(*) FROM work_threads").fetchone()[0]
        assert profile_count == 1
        assert thread_count == 1

    def test_session_insight_repair_cleans_duplicate_fts_rows(self, cli_workspace: CliWorkspace) -> None:
        """Session-insight repair must finish when only insight FTS debt remains."""
        from polylogue.config import get_config
        from polylogue.storage.insights.session.status import session_insight_status_sync
        from polylogue.storage.repair import repair_session_insights
        from polylogue.storage.sqlite.connection import connection_context
        from tests.infra.storage_records import SessionBuilder

        config = get_config()
        db_path = cli_workspace["db_path"]
        (
            SessionBuilder(db_path, "conv-insight-fts")
            .provider("claude-code")
            .title("Duplicate Insight FTS")
            .add_message("u1", role="user", text="Implement the data repair")
            .add_message("a1", role="assistant", text="Updated the session insight materializer")
            .save()
        )
        assert repair_session_insights(config).success is True

        with connection_context(None) as conn:
            event = conn.execute(
                """
                SELECT event_id, session_id, source_name, heuristic_label, search_text
                FROM session_work_events
                LIMIT 1
                """
            ).fetchone()
            thread = conn.execute(
                """
                SELECT thread_id, root_id, search_text
                FROM work_threads
                LIMIT 1
                """
            ).fetchone()
            assert event is not None
            assert thread is not None
            conn.execute(
                """
                INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["session_id"],
                    event["source_name"],
                    event["heuristic_label"],
                    event["search_text"],
                ),
            )
            conn.execute(
                "INSERT INTO work_threads_fts (thread_id, root_id, text) VALUES (?, ?, ?)",
                (thread["thread_id"], thread["root_id"], thread["search_text"]),
            )
            conn.commit()
            status = session_insight_status_sync(conn)
            assert status.ready_flag("work_event_inference_fts_ready") is False
            assert status.ready_flag("threads_fts_ready") is False

        result = repair_session_insights(config)

        assert result.success is True
        assert result.detail == "Session insights ready"
        with connection_context(None) as conn:
            status = session_insight_status_sync(conn)
        assert status.ready_flag("work_event_inference_fts_ready") is True
        assert status.ready_flag("threads_fts_ready") is True

    def test_run_selected_maintenance_idempotency_after_full_run(self, cli_workspace: CliWorkspace) -> None:
        """Selected maintenance twice should find 0 issues on second run."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_selected_maintenance
        from polylogue.storage.sqlite.connection import connection_context

        config = get_config()

        # Setup: add issues
        with connection_context(None) as conn:
            _insert_session(conn, "empty-1")
            conn.commit()

        results1 = run_selected_maintenance(config, repair=True, cleanup=True, dry_run=False)
        total_repaired_1 = sum(r.repaired_count for r in results1)
        assert total_repaired_1 > 0

        results2 = run_selected_maintenance(config, repair=True, cleanup=True, dry_run=False)
        total_repaired_2 = sum(r.repaired_count for r in results2)
        assert total_repaired_2 == 0
