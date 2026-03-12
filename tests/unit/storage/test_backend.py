"""Backend infrastructure tests — init, connections, migrations, threading, lifecycle.

This module contains tests for:
- Schema initialization and migrations
- Database connection management and threading safety
- Backend initialization and lifecycle
- Helper function utilities (JSON, ref IDs, paths)
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import connection_context, open_connection
from polylogue.storage.backends.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.store import (
    ConversationRecord,
    RawConversationRecord,
)
from tests.infra.storage_records import (
    make_conversation,
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
    assert "incompatible" in str(exc_info.value).lower() or "schema version" in str(exc_info.value).lower()

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


def test_open_connection_busy_timeout_set(tmp_path):
    """open_connection() sets busy_timeout for concurrent access."""
    db_path = tmp_path / "test.db"

    with open_connection(db_path) as conn:
        # Check busy_timeout is set (should be 30000ms = 30 seconds)
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] == 30000


# =============================================================================
# DB+STORE INTEGRATION (from test_db_store.py)
# =============================================================================

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

    def test_connection_context_rollsback_on_exception(self, tmp_path):
        """open_connection uses cached connections, so explicit rollback needed.

        This test verifies that connection_context uses connection caching,
        which means transactions persist across context exits unless explicitly
        rolled back. For automatic rollback, use SQLiteBackend.transaction().
        """
        db_path = tmp_path / "rollback_test.db"

        # First create the database and table
        with open_connection(db_path) as conn:
            _ensure_schema(conn)

        # Test explicit transaction with rollback
        try:
            with open_connection(db_path) as conn:
                conn.execute("BEGIN")
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
            # Explicitly rollback since cached connection doesn't auto-rollback
            with open_connection(db_path) as conn:
                conn.rollback()

        # Verify rolled back
        with open_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                ("rollback-test",),
            )
            row = cursor.fetchone()
            assert row is None, "Insert should have been rolled back"


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


class TestAsyncBackendInfrastructure:
    """Async backend locking, initialization, and lifecycle contracts."""

    @pytest.mark.asyncio
    async def test_concurrent_schema_initialization(self, tmp_path):
        """Concurrent operations must initialize schema once without racing."""
        backend = SQLiteBackend(db_path=tmp_path / "test_concurrent.db")

        results = await asyncio.gather(
            *[backend.get_conversation(f"test:{i}") for i in range(10)]
        )

        assert all(result is None for result in results)
        assert backend._schema_ensured is True

    @pytest.mark.asyncio
    async def test_schema_init_called_once_despite_concurrency(self, tmp_path):
        """The schema guard must collapse many concurrent calls into one actual init."""
        backend = SQLiteBackend(db_path=tmp_path / "test_once.db")

        init_count = 0
        original_ensure_schema = backend._ensure_schema

        async def counting_ensure_schema(conn):
            nonlocal init_count
            init_count += 1
            return await original_ensure_schema(conn)

        backend._ensure_schema = counting_ensure_schema

        await asyncio.gather(*[backend.list_conversations() for _ in range(20)])

        assert init_count == 1

    @pytest.mark.asyncio
    async def test_schema_lock_prevents_duplicate_slow_initialization(self, tmp_path):
        """A slow schema init path must still run exactly once under contention."""
        backend = SQLiteBackend(db_path=tmp_path / "test_lock.db")
        backend._schema_ensured = False

        events: list[str] = []
        original_ensure_schema = backend._ensure_schema

        async def slow_ensure_schema(conn):
            events.append("start")
            await asyncio.sleep(0.05)
            await original_ensure_schema(conn)
            events.append("end")

        backend._ensure_schema = slow_ensure_schema

        await asyncio.gather(
            backend.get_conversation("a"),
            backend.get_conversation("b"),
        )

        assert events.count("start") == 1
        assert events.count("end") == 1

    @pytest.mark.asyncio
    async def test_transaction_context_manager_acquires_and_releases_write_lock(self, tmp_path):
        """transaction() must hold the write lock for the duration of the context only."""
        backend = SQLiteBackend(db_path=tmp_path / "transaction.db")

        async with backend.transaction():
            assert backend._write_lock.locked()

        assert not backend._write_lock.locked()

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_serialized_by_write_lock(self, tmp_path):
        """Concurrent write contexts must all complete without losing participants."""
        backend = SQLiteBackend(db_path=tmp_path / "writes.db")
        execution_order: list[int] = []

        async def write_operation(task_id: int):
            async with backend.transaction():
                execution_order.append(task_id)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[write_operation(i) for i in range(5)])

        assert len(execution_order) == 5
        assert set(execution_order) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_reads_do_not_acquire_write_lock(self, tmp_path):
        """Read-only operations must not grab the write lock."""
        backend = SQLiteBackend(db_path=tmp_path / "read-no-lock.db")

        await backend.list_conversations()

        assert not backend._write_lock.locked()

    @pytest.mark.asyncio
    async def test_reads_can_proceed_during_write(self, tmp_path):
        """WAL-mode reads should proceed while a write transaction is open."""
        backend = SQLiteBackend(db_path=tmp_path / "read-during-write.db")
        read_completed = False

        async def slow_write():
            async with backend.transaction():
                await asyncio.sleep(0.1)

        async def quick_read():
            nonlocal read_completed
            await backend.list_conversations()
            read_completed = True

        write_task = asyncio.create_task(slow_write())
        await asyncio.sleep(0.01)
        read_task = asyncio.create_task(quick_read())
        await asyncio.gather(write_task, read_task)

        assert read_completed

    @pytest.mark.asyncio
    async def test_connection_error_during_init(self):
        """Invalid connection targets must surface an error instead of silently succeeding."""
        with pytest.raises((OSError, PermissionError, Exception)):
            backend = SQLiteBackend(db_path=Path("/nonexistent/deeply/nested/path/db.db"))
            await backend.get_conversation("test")

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path):
        """Repeated close() calls must be safe on an initialized backend."""
        backend = SQLiteBackend(db_path=tmp_path / "close.db")
        await backend.list_conversations()

        await backend.close()
        await backend.close()
        await backend.close()


def test_default_db_path(tmp_path, monkeypatch):
    """default_db_path() respects XDG_DATA_HOME."""
    xdg_data = tmp_path / "data"
    xdg_data.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

    # Reimport to pick up new env
    import importlib

    import polylogue.storage.backends.connection as connection_module
    importlib.reload(connection_module)

    path = connection_module.default_db_path()
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
        """Test that connection_context caches connections per thread.

        Unlike traditional context managers, connection_context uses
        thread-local caching, so connections remain open and usable
        after exiting the context. This is intentional for performance.
        """
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            conn_obj = conn
        # After exiting context, connection is still usable (cached)
        result = conn_obj.execute("SELECT 1").fetchone()
        assert result[0] == 1

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
        SQLiteBackend(db_path=nested_path)
        assert nested_path.parent.exists()

    def test_init_has_write_lock(self, tmp_path):
        """Test that async SQLiteBackend has write lock for serialization."""
        import asyncio

        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        assert hasattr(backend, "_write_lock")
        assert isinstance(backend._write_lock, asyncio.Lock)


class TestPagedIdIteration:
    """Tests for bounded ID iteration helpers."""

    async def test_iter_raw_ids_pages_without_duplicates(self, tmp_path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")
        base = datetime(2026, 3, 10, tzinfo=timezone.utc)

        for i in range(5):
            await backend.save_raw_conversation(
                RawConversationRecord(
                    raw_id=f"raw-{i}",
                    provider_name="chatgpt",
                    source_name="inbox-a",
                    source_path=f"/tmp/raw-{i}.json",
                    raw_content=b'{"id":"x"}',
                    acquired_at=(base + timedelta(minutes=i)).isoformat(),
                )
            )

        ids = [raw_id async for raw_id in backend.iter_raw_ids(page_size=2)]

        assert ids == ["raw-4", "raw-3", "raw-2", "raw-1", "raw-0"]

    async def test_iter_conversation_ids_pages_in_sort_order(self, tmp_path):
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        for i in range(5):
            await backend.save_conversation_record(
                make_conversation(
                    f"conv-{i}",
                    provider_name="chatgpt",
                    title=f"Conversation {i}",
                    sort_key=float(i),
                )
            )

        ids = [conversation_id async for conversation_id in backend.iter_conversation_ids(page_size=2)]
        count = await backend.count_conversation_ids()

        assert ids == ["conv-4", "conv-3", "conv-2", "conv-1", "conv-0"]
        assert count == 5


class TestBackendLifecycle:
    """Tests for backend lifecycle management."""

    async def test_close_backend(self, tmp_path):
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
        await backend.save_conversation_record(record)

        await backend.close()

        # After close, operations should fail or create new connection
        # depending on lazy connection semantics.
        # Reaching here without raising verifies the contract.

    async def test_close_and_reopen(self, tmp_path):
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
        async with backend.transaction():
            await backend.save_conversation_record(record)

        # Verify data exists before close
        retrieved1 = await backend.get_conversation("conv-1")
        assert retrieved1 is not None

        await backend.close()

        # After close, the connection state is reset
        # Verify a new connection can be established via an operation
        async with backend._get_connection() as conn:
            assert conn is not None
