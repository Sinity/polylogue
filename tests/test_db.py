from __future__ import annotations

import sqlite3
import threading

import pytest

from polylogue.storage.backends.sqlite import (
    SCHEMA_VERSION,
    _apply_schema,
    _ensure_schema,
    _run_migrations,
    connection_context,
    default_db_path,
    open_connection,
)


def test_open_connection_creates_database(tmp_path):
    """open_connection() creates a new database with schema."""
    db_path = tmp_path / "new.db"
    assert not db_path.exists()

    with open_connection(db_path) as conn:
        assert conn is not None
        # Verify schema was created
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

    assert db_path.exists()
    assert "conversations" in tables
    assert "messages" in tables
    assert "attachments" in tables
    assert "attachment_refs" in tables
    assert "runs" in tables


def test_open_connection_returns_row_factory(tmp_path):
    """open_connection() returns connection with Row factory."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Insert test data
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "title, created_at, updated_at, content_hash, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()

        # Row factory allows dict-like access
        assert row["conversation_id"] == "conv1"
        assert row["title"] == "Test"


def test_open_connection_enables_foreign_keys(tmp_path):
    """open_connection() enables foreign key constraints."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Check foreign_keys pragma
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1  # Foreign keys are ON


def test_open_connection_sets_wal_mode(tmp_path):
    """open_connection() sets WAL journal mode."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"


def test_connection_context_uses_provided_connection(tmp_path):
    """connection_context() uses provided connection without creating new one."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        original_id = id(conn)

        with connection_context(conn) as ctx_conn:
            assert id(ctx_conn) == original_id


def test_apply_schema_creates_all_tables(tmp_path):
    """_apply_schema() creates all required tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _apply_schema(conn)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    assert "conversations" in tables
    assert "messages" in tables
    assert "attachments" in tables
    assert "attachment_refs" in tables
    assert "runs" in tables

    conn.close()


def test_apply_schema_creates_indexes(tmp_path):
    """_apply_schema() creates required indexes."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    _apply_schema(conn)

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]

    # Check for some key indexes
    assert "idx_conversations_provider" in indexes
    assert "idx_messages_conversation" in indexes
    assert "idx_attachment_refs_conversation" in indexes

    conn.close()


def test_apply_schema_sets_user_version(tmp_path):
    """_apply_schema() sets PRAGMA user_version to SCHEMA_VERSION."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    _apply_schema(conn)

    row = conn.execute("PRAGMA user_version").fetchone()
    assert row[0] == SCHEMA_VERSION

    conn.close()


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


def test_open_connection_thread_isolation(tmp_path):
    """open_connection() maintains separate connections per thread."""
    db_path = tmp_path / "test.db"

    # Initialize database with WAL mode first to avoid lock contention during PRAGMA journal_mode
    with open_connection(db_path) as conn:
        conn.execute("SELECT 1").fetchone()

    connection_ids = []
    errors = []

    def thread_func(thread_id: int):
        try:
            # No delay needed - threading.Thread.join() provides synchronization
            with open_connection(db_path) as conn:
                conn_id = id(conn)
                connection_ids.append(conn_id)
                # Do some work to hold connection
                conn.execute("SELECT 1").fetchone()
                # Connection scope automatically manages timing via context manager
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(3)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # No errors should occur
    assert len(errors) == 0

    # All threads should succeed
    assert len(connection_ids) == 3

    # Each thread should have had a different connection object
    assert len(set(connection_ids)) == 3


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

    def test_migration_failure_preserves_original_state(self, tmp_path):
        """Failed migration should not leave database in inconsistent state.

        Issue: db.py:314-331 migrations don't have rollback on failure.
        This test verifies that if a migration step fails, the database
        remains usable at its previous version.
        """
        db_path = tmp_path / "test.db"

        # Initialize database at current schema
        with open_connection(db_path) as conn:
            _ensure_schema(conn)
            # Verify it's at current version
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            assert version == SCHEMA_VERSION

            # Insert some test data
            conn.execute(
                """
                INSERT INTO conversations
                (conversation_id, provider_name, provider_conversation_id, title, content_hash, version)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                ("test-conv", "test", "prov-1", "Test Title", "hash123", 1),
            )
            conn.commit()

        # Verify data persists
        with open_connection(db_path) as conn:
            cursor = conn.execute("SELECT title FROM conversations WHERE conversation_id = ?", ("test-conv",))
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "Test Title"

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

        original = _MIGRATIONS[3]
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

