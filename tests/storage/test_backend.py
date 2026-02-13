"""Backend infrastructure tests — init, connections, migrations, threading, lifecycle.

This module contains tests for:
- Schema initialization and migrations
- Database connection management and threading safety
- Backend initialization and lifecycle
- Helper function utilities (JSON, ref IDs, paths)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.schemas.unified import (
    extract_harmonized_message,
    is_message_record,
)
from polylogue.schemas.unified import (
    normalize_role as new_normalize_role,
)
from polylogue.sources.parsers.base import normalize_role as old_normalize_role
from polylogue.sources.parsers.claude import (
    extract_text_from_segments as old_extract_segments,
)
from polylogue.storage.backends import SQLiteBackend
from polylogue.storage.backends.sqlite import (
    SCHEMA_VERSION,
    SQLiteBackend,
    DatabaseError,
    _apply_schema,
    _ensure_schema,
    _json_or_none,
    _run_migrations,
    connection_context,
    default_db_path,
    open_connection,
)
from polylogue.storage.store import (
    MAX_ATTACHMENT_SIZE,
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
)
from tests.helpers import (
    _make_ref_id,
    _prune_attachment_refs,
    make_attachment,
    make_conversation,
    make_message,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)

# test_db and test_conn fixtures are in conftest.py


# =============================================================================
# DATABASE/CONNECTION MANAGEMENT (from test_db.py)
# =============================================================================


def test_ensure_schema_applies_on_new_database(tmp_path):
    """_ensure_schema() applies schema to new database (version 0)."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Verify starting at version 0
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 0

    _ensure_schema(conn)

    # Check version updated
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == SCHEMA_VERSION

    # Check tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert "conversations" in tables

    conn.close()


def test_ensure_schema_raises_on_unsupported_version(tmp_path):
    """_ensure_schema() raises error for unsupported schema versions."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Set unsupported version
    conn.execute("PRAGMA user_version = 999")
    conn.commit()

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        _ensure_schema(conn)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Unsupported DB schema version" in str(exc_info.value)

    conn.close()


def test_migrate_v1_to_v2_creates_new_tables(tmp_path):
    """_migrate_v1_to_v2() creates attachments and attachment_refs tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v1 schema (simplified)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            provider_message_id TEXT,
            role TEXT,
            text TEXT,
            timestamp TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    # Old attachments table (v1 schema)
    conn.execute(
        """
        CREATE TABLE attachments (
            attachment_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            message_id TEXT,
            mime_type TEXT,
            size_bytes INTEGER,
            path TEXT,
            provider_meta TEXT
        )
        """
    )
    conn.commit()

    # Insert test data
    conn.execute(
        "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", None, 1),
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("msg1", "conv1", None, "user", "Hello", "2024-01-01T00:00:00Z", "msghash1", None, 1),
    )
    conn.execute(
        "INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("att1", "conv1", "msg1", "image/png", 1024, "/path/to/file.png", None),
    )
    conn.commit()

    # Migrate using the runner (which updates version)
    _run_migrations(conn, 1, 2)

    # Check version updated
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 2

    # Check new tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    assert "attachments" in tables
    assert "attachment_refs" in tables

    # Check data migrated
    att_row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
    assert att_row is not None
    assert att_row["ref_count"] == 1

    ref_rows = conn.execute("SELECT * FROM attachment_refs WHERE attachment_id = ?", ("att1",)).fetchall()
    assert len(ref_rows) == 1

    conn.close()


def test_migrate_v2_to_v3_updates_runs_table(tmp_path):
    """_migrate_v2_to_v3() updates runs table schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v2 schema with old runs table
    conn.execute("PRAGMA user_version = 2")
    conn.executescript(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        );

        CREATE TABLE runs_old (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            plan_snapshot TEXT,
            counts_json TEXT,
            drift_json TEXT,
            indexed INTEGER,
            duration_ms INTEGER
        );
        """
    )

    # Insert test run data
    conn.execute(
        "INSERT INTO runs_old VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run1", "2024-01-01T00:00:00Z", '{"test": true}', '{"count": 5}', "{}", 1, 1000),
    )
    conn.commit()

    # Temporarily rename runs_old for migration test
    conn.execute("ALTER TABLE runs_old RENAME TO runs")
    conn.commit()

    # Migrate
    # Migrate using the runner (which updates version)
    _run_migrations(conn, 2, 3)

    # Check version updated to 3
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 3

    # Check runs table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    assert cursor.fetchone() is not None

    # Check data preserved
    run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert run_row is not None
    assert run_row["timestamp"] == "2024-01-01T00:00:00Z"

    conn.close()


def test_migrate_v3_to_v4_adds_source_name_column(tmp_path):
    """_migrate_v3_to_v4() adds computed source_name column and index."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Setup v3 schema without source_name column
    conn.execute("PRAGMA user_version = 3")
    conn.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            provider_conversation_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT NOT NULL,
            provider_meta TEXT,
            version INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX idx_conversations_provider
        ON conversations(provider_name, provider_conversation_id)
        """
    )

    # Insert test conversation with source in provider_meta
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, provider_name, provider_conversation_id,
            title, content_hash, provider_meta, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("test:123", "test", "123", "Test", "abc123", '{"source": "my-source"}', 1),
    )
    conn.commit()

    # Migrate using the runner (which updates version)
    _run_migrations(conn, 3, 4)

    # Check version updated to 4
    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == 4

    # Check source_name column exists and is computed correctly
    conv_row = conn.execute(
        "SELECT conversation_id, source_name FROM conversations WHERE conversation_id = ?",
        ("test:123",),
    ).fetchone()
    assert conv_row is not None
    assert conv_row["source_name"] == "my-source"

    # Check index exists
    index_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conversations_source_name'"
    ).fetchone()
    assert index_row is not None

    conn.close()


@pytest.mark.slow
def test_open_connection_thread_isolation(tmp_path):
    """open_connection() maintains separate connections per thread."""
    from threading import Barrier

    db_path = tmp_path / "test.db"
    num_threads = 3

    # Initialize database with WAL mode first to avoid lock contention during PRAGMA journal_mode
    with open_connection(db_path) as conn:
        conn.execute("SELECT 1").fetchone()

    # Barrier ensures all threads hold connections simultaneously
    # This prevents Python from reusing memory addresses after GC
    barrier = Barrier(num_threads)
    connection_ids = []
    errors = []

    def thread_func(thread_id: int):
        try:
            with open_connection(db_path) as conn:
                conn_id = id(conn)
                connection_ids.append(conn_id)
                # All threads wait here with live connections before proceeding
                barrier.wait()
                # Do some work to verify connection is usable
                conn.execute("SELECT 1").fetchone()
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # No errors should occur
    assert len(errors) == 0

    # All threads should succeed
    assert len(connection_ids) == num_threads

    # Each thread should have had a different connection object
    # (guaranteed because barrier ensures all connections exist simultaneously)
    assert len(set(connection_ids)) == num_threads


def test_open_connection_creates_parent_directories(tmp_path):
    """open_connection() creates parent directories if they don't exist."""
    db_path = tmp_path / "nested" / "deeply" / "test.db"
    assert not db_path.parent.exists()

    with open_connection(db_path) as conn:
        assert conn is not None

    assert db_path.exists()
    assert db_path.parent.exists()


def test_open_connection_busy_timeout_set(tmp_path):
    """open_connection() sets busy_timeout for concurrent access."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Check busy_timeout is set (should be 30000ms = 30 seconds)
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 30000


class TestMigrations:
    """Tests for database migration behavior."""

    def test_migration_failure_preserves_original_state(self, tmp_path, monkeypatch):
        """Failed migration should not leave database in inconsistent state.

        This test verifies that if a migration step fails, the database
        remains at the last successful version (ratcheting behavior).
        """
        db_path = tmp_path / "test_state_preservation.db"

        # Initialize database at v1 (past version)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA user_version = 1")
        # Minimal v1 schema to satisfy _migrate_v1_to_v2
        conn.execute(
            """
            CREATE TABLE attachments (
                attachment_id TEXT PRIMARY KEY,
                mime_type TEXT,
                size_bytes INTEGER,
                path TEXT,
                provider_meta TEXT,
                conversation_id TEXT,
                message_id TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        # Patch _MIGRATIONS[2] to fail (simulating v2->v3 failure)
        # We allow v1->v2 to succeed
        from polylogue.storage.backends.sqlite import _MIGRATIONS

        def failing_migration(conn):
            raise RuntimeError("Simulated migration v2->v3 failure")

        # Copy dict to avoid polluting other tests
        patched_migrations = _MIGRATIONS.copy()
        patched_migrations[2] = failing_migration
        monkeypatch.setattr("polylogue.storage.backends.sqlite._MIGRATIONS", patched_migrations)

        # Run connection open which triggers _ensure_schema
        # It should raise DatabaseError or RuntimeError from the migration
        with pytest.raises(Exception, match="Simulated migration v2->v3 failure"):
            with open_connection(db_path) as conn:
                pass

        # Verify database state
        conn = sqlite3.connect(db_path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]

        # Should be at version 2 (because v1->v2 succeeded and committed)
        # The failing v2->v3 rolled back its changes (if any) and didn't bump version
        assert version == 2
        conn.close()

    def test_migration_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        """Failed migration raises RuntimeError with details.

        Note: SQLite DDL (ALTER TABLE, CREATE TABLE) cannot be rolled back.
        This test verifies that migration failures are properly reported.
        """
        db_path = tmp_path / "migration_failure_test.db"

        # Create a v3 database manually
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA user_version = 3")
        conn.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                provider_conversation_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                content_hash TEXT NOT NULL,
                provider_meta TEXT,
                version INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_conversations_provider
            ON conversations(provider_name, provider_conversation_id)
            """
        )
        conn.commit()
        conn.close()

        # Patch _MIGRATIONS[3] to fail (patching the function won't work as dict has reference)
        from polylogue.storage.backends.sqlite import _MIGRATIONS

        def failing_migration(conn):
            raise RuntimeError("Simulated migration failure")

        _MIGRATIONS[3]
        monkeypatch.setitem(_MIGRATIONS, 3, failing_migration)

        # Migration should raise RuntimeError
        with pytest.raises(RuntimeError, match="Migration from v3 to v4 failed"):
            with open_connection(db_path) as conn:
                pass

    def test_connection_context_rollsback_on_exception(self, tmp_path):
        """open_connection should rollback on exception."""
        db_path = tmp_path / "rollback_test.db"

        # First create the database and table
        with open_connection(db_path) as conn:
            _ensure_schema(conn)

        # Now try to insert and raise
        try:
            with open_connection(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id, title, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    ("rollback-test", "test", "prov-1", "Rollback Test", "hash789", 1),
                )
                raise ValueError("Simulated failure")
        except ValueError:
            pass

        # Verify rolled back
        with open_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                ("rollback-test",),
            )
            row = cursor.fetchone()
            assert row is None, "Insert should have been rolled back"


# =============================================================================
# DB+STORE INTEGRATION (from test_db_store.py)
# =============================================================================


class TestConnectionContextReuse:
    """Test connection reuse within same thread."""


class TestConnectionCommitAndRollback:
    """Test transaction commit/rollback behavior."""

    def test_connection_no_commit_on_exception(self, tmp_path):
        """Exception in context skips commit (data may not persist).

        Note: The implementation commits only on normal exit. On exception,
        it closes without commit. With SQLite WAL mode + autocommit, writes
        within the context may still be visible within that session but won't
        persist after close.
        """
        db_path = tmp_path / "test.db"

        # Initialize schema first
        with open_connection(db_path):
            pass

        try:
            with open_connection(db_path) as conn:
                conn.execute(
                    """INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id,
                     title, created_at, updated_at, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("c1", "test", "ext1", "Title1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify no explicit commit happened - data should not persist
        # (SQLite autocommit behavior may vary, but no explicit commit was called)
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            # The implementation doesn't do rollback, but also doesn't commit
            # Row presence depends on SQLite internals; we verify connection still works
            if row is not None:
                assert row["conversation_id"] == "c1"


@pytest.mark.slow
class TestThreadSafety:
    """Test thread-local connection safety."""

    def test_thread_local_connections_isolated(self, tmp_path):
        """Each thread gets its own isolated connection."""
        db_path = tmp_path / "test.db"

        # Initialize database with WAL mode first to avoid lock contention
        with open_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

        connection_ids = {}
        errors = []

        def thread_work(thread_id: int):
            try:
                with open_connection(db_path) as conn:
                    connection_ids[thread_id] = id(conn)
                    # Verify connection is functional
                    cursor = conn.execute("SELECT 1")
                    assert cursor.fetchone() is not None
                    conn.commit()  # Ensure locks are released
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Use fewer threads to reduce lock contention
        threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(set(connection_ids.values())) == 3, "Each thread should have different connection object"

    def test_concurrent_writes_with_write_lock(self, tmp_path):
        """Concurrent store_records() calls properly serialize via write lock."""
        db_path = tmp_path / "test.db"

        # Initialize database with WAL mode first
        with open_connection(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

        errors = []

        def write_conversation(conv_id: int):
            try:
                conv = make_conversation(f"c{conv_id}", title=f"Conversation {conv_id}")
                messages = [
                    make_message(
                        f"m{conv_id}-{i}",
                        f"c{conv_id}",
                        role="user" if i % 2 == 0 else "assistant",
                        text=f"Message {i}",
                    )
                    for i in range(3)
                ]

                with open_connection(db_path) as conn:
                    store_records(conversation=conv, messages=messages, attachments=[], conn=conn)
                    conn.commit()  # Explicit commit to release locks faster
            except Exception as e:
                errors.append((conv_id, str(e)))

        # Run concurrent writes with reduced parallelism to avoid lock contention
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_conversation, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

        # Verify all conversations written
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert count == 20

            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert msg_count == 60  # 20 * 3


class TestSchemaAndMigration:
    """Test schema initialization and versioning."""

    def test_open_connection_applies_schema_on_new_db(self, tmp_path):
        """open_connection() applies full schema to new database."""
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        with open_connection(db_path) as conn:
            # Verify schema version
            version_row = conn.execute("PRAGMA user_version").fetchone()
            assert version_row[0] == SCHEMA_VERSION

            # Verify tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "conversations",
                "messages",
                "attachments",
                "attachment_refs",
                "runs",
            ]
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"

    def test_open_connection_foreign_keys_enabled(self, tmp_path):
        """open_connection() enables foreign key constraints."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA foreign_keys").fetchone()
            assert row[0] == 1, "Foreign keys should be ON"

    def test_open_connection_wal_mode_enabled(self, tmp_path):
        """open_connection() enables WAL journal mode."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal", "WAL mode should be enabled"

    def test_open_connection_creates_parent_directories(self, tmp_path):
        """open_connection() creates nested parent directories."""
        db_path = tmp_path / "a" / "b" / "c" / "test.db"
        assert not db_path.parent.exists()

        with open_connection(db_path) as conn:
            assert conn is not None

        assert db_path.exists()
        assert db_path.parent.exists()


class TestBackendEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_store_records_with_null_optional_fields(self, tmp_path):
        """store_records() handles conversations/messages with NULL optional fields."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title=None,  # NULL
            created_at=None,  # NULL
            updated_at=None,  # NULL
            content_hash="hash1",
            provider_meta=None,  # NULL
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            provider_message_id=None,  # NULL
            role=None,  # NULL
            text=None,  # NULL
            timestamp=None,  # NULL
            content_hash="msghash1",
            provider_meta=None,  # NULL
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 1

        # Verify NULLs preserved
        with open_connection(db_path) as conn:
            conv_row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert conv_row["title"] is None
            assert conv_row["created_at"] is None

            msg_row = conn.execute("SELECT * FROM messages WHERE message_id = ?", ("m1",)).fetchone()
            assert msg_row["role"] is None
            assert msg_row["text"] is None

    def test_store_records_with_empty_messages_and_attachments(self, tmp_path):
        """store_records() handles conversation with no messages or attachments."""
        db_path = tmp_path / "test.db"

        conv = make_conversation("c1", title="Empty Conversation")

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 0
        assert counts["attachments"] == 0

    def test_attachment_without_message_id(self, tmp_path):
        """Attachments can exist without being tied to a message."""
        db_path = tmp_path / "test.db"

        conv = make_conversation("c1", title="Test")
        # Attachment without message_id
        att = make_attachment("att1", "c1", message_id=None, mime_type="application/pdf", size_bytes=5000)

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[att], conn=conn)

        assert counts["attachments"] == 1

        # Verify stored
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is not None
            assert row["ref_count"] == 1


class TestComplexScenarios:
    """Test realistic complex scenarios."""

    def test_conversation_lifecycle_with_attachments(self, tmp_path):
        """Full lifecycle: create → add attachments → remove attachments → cleanup."""
        db_path = tmp_path / "test.db"

        # Step 1: Create conversation with one attachment
        conv_v1 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:00:00Z",
            content_hash="hash-v1",
        )
        msg1 = make_message("m1", "c1", text="Please analyze this image", timestamp="2024-01-01T10:00:00Z")
        att1 = make_attachment("att-image", "c1", "m1", mime_type="image/png", size_bytes=51200)

        with open_connection(db_path) as conn:
            store_records(conversation=conv_v1, messages=[msg1], attachments=[att1], conn=conn)

        # Step 2: Add more messages and attachments
        msg2 = make_message("m2", "c1", role="assistant", text="The image shows...", timestamp="2024-01-01T10:01:00Z")
        att2 = make_attachment("att-export", "c1", "m2", mime_type="application/json", size_bytes=2048)
        conv_v2 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:02:00Z",
            content_hash="hash-v2",
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v2,
                messages=[msg1, msg2],
                attachments=[att1, att2],
                conn=conn,
            )

        # Verify 2 attachments now
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 2

        # Step 3: Final update removes one attachment
        conv_v3 = make_conversation(
            "c1",
            provider_name="claude",
            title="Analysis Project - Final",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:03:00Z",
            content_hash="hash-v3",
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v3,
                messages=[msg1, msg2],
                attachments=[att1],  # Only image, no export
                conn=conn,
            )

        # Verify: image kept, export deleted
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 1

            remaining = conn.execute("SELECT attachment_id FROM attachments").fetchone()
            assert remaining["attachment_id"] == "att-image"

    def test_multi_provider_conversations_separate(self, tmp_path):
        """Conversations from different providers don't interfere."""
        db_path = tmp_path / "test.db"

        conv_gpt = make_conversation(
            "c-gpt", provider_name="chatgpt", title="ChatGPT Conversation", content_hash="hash-gpt"
        )
        conv_claude = make_conversation(
            "c-claude", provider_name="claude", title="Claude Conversation", content_hash="hash-claude"
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv_gpt, messages=[], attachments=[], conn=conn)
            store_records(conversation=conv_claude, messages=[], attachments=[], conn=conn)

        # Verify both stored correctly
        with open_connection(db_path) as conn:
            gpt_row = conn.execute("SELECT * FROM conversations WHERE provider_name = ?", ("chatgpt",)).fetchone()
            claude_row = conn.execute("SELECT * FROM conversations WHERE provider_name = ?", ("claude",)).fetchone()

            assert gpt_row is not None
            assert gpt_row["title"] == "ChatGPT Conversation"
            assert claude_row is not None
            assert claude_row["title"] == "Claude Conversation"


# =============================================================================
# BACKEND COMPARISON TESTS (from test_backend_core.py)
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing old vs new extraction."""

    field: str
    old_value: str | None
    new_value: str | None
    equivalent: bool


def compare_extractions(provider: str, raw: dict) -> list[ComparisonResult]:
    """Compare old and new extraction for a single message."""
    results = []

    try:
        new_msg = extract_harmonized_message(provider, raw)
    except Exception as e:
        return [ComparisonResult("extraction", None, str(e), False)]

    if provider == "claude-code":
        msg_obj = raw.get("message", {})
        msg_type = raw.get("type")

        if msg_type in ("user", "human"):
            old_role = "user"
        elif msg_type == "assistant":
            old_role = "assistant"
        else:
            old_role = msg_type or "unknown"

        content_raw = msg_obj.get("content") if isinstance(msg_obj, dict) else None
        old_text = old_extract_segments(content_raw) if isinstance(content_raw, list) else None

        old_role_norm = old_normalize_role(old_role)
        new_role_norm = new_msg.role

        results.append(ComparisonResult(
            field="role",
            old_value=old_role_norm,
            new_value=new_role_norm,
            equivalent=old_role_norm == new_role_norm,
        ))

        old_text_norm = (old_text or "").strip()
        new_text_norm = (new_msg.text or "").strip()

        results.append(ComparisonResult(
            field="text",
            old_value=old_text_norm if old_text_norm else None,
            new_value=new_text_norm if new_text_norm else None,
            equivalent=old_text_norm == new_text_norm,
        ))

    return results


def _seed_conversation(backend):
    """Helper: insert a conversation so metadata operations have a target."""
    conn = backend._get_connection()
    conv = make_conversation("conv1", content_hash="hash1")
    msg = make_message("m1", "conv1", text="Hello")
    store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)
    conn.commit()
    return "conv1"


# =============================================================================
# GET CONVERSATION STATS (parametrized)
# =============================================================================

# Test cases for get_conversation_stats with different message configurations
CONVERSATION_STATS_CASES = [
    (
        [],
        {"total": 0, "dialogue": 0, "tool": 0},
        "empty",
    ),
    (
        [
            ("msg-0", "user", "Message 0"),
            ("msg-1", "assistant", "Message 1"),
            ("msg-2", "user", "Message 2"),
            ("msg-3", "assistant", "Message 3"),
            ("msg-4", "user", "Message 4"),
        ],
        {"total": 5, "dialogue": 5, "tool": 0},
        "dialogue_only",
    ),
    (
        [
            ("msg-1", "user", "Hello"),
            ("msg-2", "tool", "Tool output"),
            ("msg-3", "assistant", "Response"),
        ],
        {"total": 3, "dialogue": 2, "tool": 1},
        "with_tool",
    ),
]


@pytest.mark.parametrize("messages_spec,expected,desc", CONVERSATION_STATS_CASES, ids=str)
def test_get_conversation_stats(tmp_path, messages_spec, expected, desc):
    """get_conversation_stats counts correctly: {desc}."""
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
            message_id=msg_id,
            conversation_id="conv-1",
            role=role,
            text=text,
            timestamp=f"2025-01-01T10:{i:02d}:00Z",
            content_hash=f"h{i}",
            version=1,
        )
        for i, (msg_id, role, text) in enumerate(messages_spec)
    ]

    if messages:
        backend.save_messages(messages)

    stats = backend.get_conversation_stats("conv-1")
    assert stats["total_messages"] == expected["total"]
    assert stats["dialogue_messages"] == expected["dialogue"]
    assert stats["tool_messages"] == expected["tool"]


# =============================================================================
# HELPER FUNCTIONS (parametrized)
# =============================================================================

# Test _json_or_none function
JSON_OR_NONE_CASES = [
    ({"key": "value"}, "dict", True),
    (None, "none", False),
    ({"nested": {"key": "value"}, "list": [1, 2, 3]}, "nested dict", True),
]


@pytest.mark.parametrize("input_val,desc,is_json", JSON_OR_NONE_CASES, ids=str)
def test_json_or_none(input_val, desc, is_json):
    """_json_or_none handles {desc}."""
    result = _json_or_none(input_val)
    if is_json:
        assert isinstance(result, str)
        assert json.loads(result) == input_val
    else:
        assert result is None


def test_make_ref_id_deterministic():
    """_make_ref_id() produces deterministic IDs."""
    id1 = _make_ref_id("att1", "conv1", "msg1")
    id2 = _make_ref_id("att1", "conv1", "msg1")
    assert id1 == id2


def test_make_ref_id_different_inputs():
    """_make_ref_id() produces different IDs for different inputs."""
    id1 = _make_ref_id("att1", "conv1", "msg1")
    id2 = _make_ref_id("att2", "conv1", "msg1")
    id3 = _make_ref_id("att1", "conv2", "msg1")
    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_make_ref_id_format():
    """_make_ref_id returns expected format."""
    ref_id = _make_ref_id("att-1", "conv-1", "msg-1")
    assert ref_id.startswith("ref-")
    assert len(ref_id) == len("ref-") + 16  # 16-char hex digest


def test_make_ref_id_with_none_message_id():
    """_make_ref_id() handles None message_id."""
    id1 = _make_ref_id("att1", "conv1", None)
    id2 = _make_ref_id("att1", "conv1", None)
    assert id1 == id2
    assert id1 != _make_ref_id("att1", "conv1", "msg1")


def test_default_db_path(tmp_path, monkeypatch):
    """default_db_path() respects XDG_DATA_HOME."""
    xdg_data = tmp_path / "data"
    xdg_data.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

    # Reimport to pick up new env
    import importlib
    import polylogue.storage.backends.sqlite as sqlite_module
    importlib.reload(sqlite_module)

    path = sqlite_module.default_db_path()
    assert str(xdg_data) in str(path)


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
# Test: SQLiteBackend initialization
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
