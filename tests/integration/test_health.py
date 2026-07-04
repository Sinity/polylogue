"""Integration tests for health repair workflows against a real database."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TypeAlias

CliWorkspace: TypeAlias = dict[str, Path]

TEST_ORIGIN = "codex-session"

RUN_ALL_REPAIRS_EXPECTED = {
    "session_insights",
    "orphaned_messages",
    "empty_sessions",
    "orphaned_attachments",
}


def _session_id(native_id: str) -> str:
    return native_id if ":" in native_id else f"{TEST_ORIGIN}:{native_id}"


def _message_id(message_native_id: str, session_native_id: str) -> str:
    return message_native_id if ":" in message_native_id else f"{_session_id(session_native_id)}:{message_native_id}"


def _position(native_id: str) -> int:
    return int.from_bytes(hashlib.sha256(native_id.encode()).digest()[:4], "big")


def _insert_session(conn: sqlite3.Connection, session_id: str, source_name: str = "test", title: str = "Test") -> None:
    """Helper to insert a session with all required fields."""
    content_hash = hashlib.sha256(f"{source_name}:{session_id}".encode()).digest()
    conn.execute(
        """
        INSERT INTO sessions (native_id, origin, title, content_hash)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, TEST_ORIGIN, title, content_hash),
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
    content_hash = hashlib.sha256(f"{message_id}:{text}".encode()).digest()
    stored_session_id = _session_id(session_id)
    position = _position(message_id)
    if allow_orphaned:
        # Temporarily disable foreign keys to insert orphaned message
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                """
                INSERT INTO messages (session_id, native_id, position, role, content_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (stored_session_id, message_id, position, role, content_hash),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        conn.execute(
            """
            INSERT INTO messages (session_id, native_id, position, role, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (stored_session_id, message_id, position, role, content_hash),
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
    stored_session_id = _session_id(session_id)
    stored_message_id = _message_id(message_id, session_id)
    position = _position(block_id)
    if allow_orphaned:
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                """
                INSERT INTO blocks (
                    message_id, session_id, position, block_type, text, tool_name
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (stored_message_id, stored_session_id, position, type, text, tool_name),
            )
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    else:
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_name
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (stored_message_id, stored_session_id, position, type, text, tool_name),
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
                "INSERT INTO attachments (attachment_id, display_name, media_type, blob_hash) VALUES (?, ?, ?, ?)",
                ("att-1", "note.txt", "text/plain", b"1" * 32),
            )
            conn.execute(
                "INSERT INTO attachment_refs (attachment_id, session_id, message_id, position) VALUES (?, ?, ?, ?)",
                ("att-1", _session_id("conv-1"), _message_id("msg-1", "conv-1"), 0),
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
                "INSERT INTO attachments (attachment_id, display_name, media_type, blob_hash) VALUES (?, ?, ?, ?)",
                ("att-1", "note.txt", "text/plain", b"1" * 32),
            )
            _insert_session(conn, "conv-1")
            _insert_message(conn, "msg-1", "conv-1")
            # Insert a proper ref
            conn.execute(
                "INSERT INTO attachment_refs (attachment_id, session_id, message_id, position) VALUES (?, ?, ?, ?)",
                ("att-1", _session_id("conv-1"), _message_id("msg-1", "conv-1"), 0),
            )
            conn.commit()

        # Act - create another attachment that's unreferenced
        with connection_context(None) as conn:
            conn.execute(
                "INSERT INTO attachments (attachment_id, display_name, media_type, blob_hash) VALUES (?, ?, ?, ?)",
                ("att-2", "image.png", "image/png", b"2" * 32),
            )
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
                "INSERT INTO attachments (attachment_id, display_name, media_type, blob_hash) VALUES (?, ?, ?, ?)",
                ("att-1", "note.txt", "text/plain", b"1" * 32),
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
                "INSERT INTO attachments (attachment_id, display_name, media_type, blob_hash) VALUES (?, ?, ?, ?)",
                ("att-1", "note.txt", "text/plain", b"1" * 32),
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
            "message_type_backfill",
        }
        assert all(r.destructive is False for r in results)
        assert all(r.category.value == "derived_repair" for r in results)

    def test_run_archive_cleanup_returns_destructive_results(self, cli_workspace: CliWorkspace) -> None:
        """run_archive_cleanup should return only destructive cleanup functions."""
        from polylogue.config import get_config
        from polylogue.storage.repair import run_archive_cleanup

        config = get_config()
        results = run_archive_cleanup(config, dry_run=False)
        assert {r.name for r in results} == {
            "orphaned_messages",
            "empty_sessions",
            "orphaned_attachments",
            "orphaned_blobs",
            "superseded_raw_snapshots",
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
                "orphaned_messages": 2,
                "empty_sessions": 5,
            },
        )

        by_name = {result.name: result for result in results}
        assert by_name["session_insights"].repaired_count == 13
        assert by_name["orphaned_messages"].repaired_count == 2
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
        assert profile_count == 1

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
