from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.db import (
    SCHEMA_VERSION,
    DatabaseError,
    _apply_schema,
    _ensure_schema,
    _migrate_v1_to_v2,
    _migrate_v2_to_v3,
    _migrate_v3_to_v4,
    connection_context,
    default_db_path,
    open_connection,
)


def test_default_db_path_uses_xdg_state_home(monkeypatch, tmp_path):
    """default_db_path() respects XDG_STATE_HOME."""
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    path = default_db_path()
    assert path == state_home / "polylogue" / "polylogue.db"


def test_default_db_path_fallback_to_home(monkeypatch, tmp_path):
    """default_db_path() falls back to ~/.local/state when XDG_STATE_HOME not set."""
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    path = default_db_path()
    assert path == tmp_path / ".local/state" / "polylogue" / "polylogue.db"


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


def test_open_connection_reuses_thread_local_connection(tmp_path):
    """open_connection() reuses connection for nested calls in same thread."""
    db_path = tmp_path / "test.db"
    connections = []

    with open_connection(db_path) as conn1:
        connections.append(id(conn1))
        with open_connection(db_path) as conn2:
            connections.append(id(conn2))
            with open_connection(db_path) as conn3:
                connections.append(id(conn3))

    # All should be the same connection object
    assert len(set(connections)) == 1


def test_open_connection_prevents_different_paths_same_thread(tmp_path):
    """open_connection() raises error if different path requested in same thread."""
    db_path1 = tmp_path / "db1.db"
    db_path2 = tmp_path / "db2.db"

    with open_connection(db_path1):
        with pytest.raises(DatabaseError, match="Existing connection opened"):
            with open_connection(db_path2):
                pass


def test_open_connection_depth_tracking(tmp_path):
    """open_connection() properly tracks connection depth."""
    db_path = tmp_path / "test.db"

    from polylogue.db import _get_state

    with open_connection(db_path):
        state = _get_state()
        assert state["depth"] == 1

        with open_connection(db_path):
            assert state["depth"] == 2

            with open_connection(db_path):
                assert state["depth"] == 3

            assert state["depth"] == 2

        assert state["depth"] == 1

    # After all contexts exit, depth should be 0 and connection None
    state = _get_state()
    assert state["depth"] == 0
    assert state["conn"] is None


def test_open_connection_commits_on_exit(tmp_path):
    """open_connection() commits changes on successful exit."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "title, created_at, updated_at, content_hash, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
        )

    # Reopen and verify commit
    with open_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
        assert row is not None
        assert row["conversation_id"] == "conv1"


def test_open_connection_closes_on_exception(tmp_path):
    """open_connection() closes connection after exception."""
    db_path = tmp_path / "test.db"

    from polylogue.db import _get_state

    try:
        with open_connection(db_path) as conn:
            state = _get_state()
            assert state["conn"] is not None
            raise ValueError("Test error")
    except ValueError:
        pass

    # Connection should be closed
    state = _get_state()
    assert state["conn"] is None
    assert state["depth"] == 0


def test_connection_context_uses_provided_connection(tmp_path):
    """connection_context() uses provided connection without creating new one."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        original_id = id(conn)

        with connection_context(conn) as ctx_conn:
            assert id(ctx_conn) == original_id


def test_connection_context_creates_new_if_none(tmp_path):
    """connection_context() creates new connection if none provided."""
    db_path = tmp_path / "test.db"

    with connection_context(None, db_path) as conn:
        assert conn is not None

        # Verify it's functional
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "title, created_at, updated_at, content_hash, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
        )

    # Verify commit happened
    with open_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
        assert row is not None


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

    with pytest.raises(DatabaseError, match="Unsupported DB schema version"):
        _ensure_schema(conn)

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

    # Migrate
    _migrate_v1_to_v2(conn)

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
        ("run1", "2024-01-01T00:00:00Z", '{"test": true}', '{"count": 5}', '{}', 1, 1000),
    )
    conn.commit()

    # Temporarily rename runs_old for migration test
    conn.execute("ALTER TABLE runs_old RENAME TO runs")
    conn.commit()

    # Migrate
    _migrate_v2_to_v3(conn)

    # Check version updated to 3 (not SCHEMA_VERSION which might be higher)
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

    # Migrate
    _migrate_v3_to_v4(conn)

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
    import time

    db_path = tmp_path / "test.db"
    connection_ids = []
    errors = []

    def thread_func(thread_id: int):
        try:
            # Add small delay to reduce contention
            time.sleep(0.01 * thread_id)
            with open_connection(db_path) as conn:
                conn_id = id(conn)
                connection_ids.append(conn_id)
                # Do some work to hold connection
                conn.execute("SELECT 1").fetchone()
                time.sleep(0.05)  # Hold connection longer
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


def test_connection_context_transaction_semantics(tmp_path):
    """connection_context() maintains transaction boundaries."""
    db_path = tmp_path / "test.db"

    # First, insert data successfully
    with connection_context(None, db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
            "title, created_at, updated_at, content_hash, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("conv1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
        )

    # Verify commit
    with open_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv1",)).fetchone()
        assert row is not None

    # Now test rollback on exception (if using provided connection)
    with open_connection(db_path) as main_conn:
        try:
            with connection_context(main_conn) as conn:
                conn.execute(
                    "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, "
                    "title, created_at, updated_at, content_hash, version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("conv2", "test", "ext2", "Test2", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash2", 1),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        # Explicit rollback needed when using connection_context with provided conn
        main_conn.rollback()

    # Verify conv2 was NOT committed
    with open_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("conv2",)).fetchone()
        assert row is None


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
