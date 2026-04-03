"""Focused infrastructure contracts for the SQLite backend."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import connection_context, open_connection
from polylogue.storage.backends.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import ConversationRecord, RawConversationRecord
from tests.infra.storage_records import make_attachment, make_conversation, make_message


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def _build_record(conversation_id: str = "conv-1") -> ConversationRecord:
    return ConversationRecord(
        conversation_id=conversation_id,
        provider_name="claude-ai",
        provider_conversation_id=conversation_id,
        title="Test",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        content_hash=f"hash-{conversation_id}",
        version=1,
    )


def test_ensure_schema_contract(tmp_path: Path) -> None:
    """Schema application must upgrade a fresh database and create the core tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert {"artifact_observations", "conversations", "messages", "attachments", "attachment_refs", "runs", "publications"}.issubset(
        _table_names(conn)
    )
    assert "message_meta" not in _table_names(conn)
    conn.close()


def test_ensure_schema_rejects_unsupported_version(tmp_path: Path) -> None:
    """Unsupported schema versions must surface a database error."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 999")
    conn.commit()

    with pytest.raises(Exception) as exc_info:
        _ensure_schema(conn)

    assert exc_info.type.__name__ == "DatabaseError"
    assert "schema version" in str(exc_info.value).lower() or "incompatible" in str(exc_info.value).lower()
    conn.close()


def test_open_connection_contract(tmp_path: Path) -> None:
    """open_connection() must create directories, apply schema, and set sqlite pragmas."""
    db_path = tmp_path / "nested" / "path" / "test.db"

    with open_connection(db_path) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert {"artifact_observations", "conversations", "messages", "attachments", "attachment_refs", "publications"}.issubset(
            _table_names(conn)
        )

    assert db_path.exists()
    assert db_path.parent.exists()


@pytest.mark.slow
def test_open_connection_thread_isolation(tmp_path: Path) -> None:
    """Each thread must receive an isolated, usable connection object."""
    db_path = tmp_path / "threaded.db"
    barrier = threading.Barrier(3)
    connection_ids: list[int] = []
    errors: list[tuple[int, str]] = []

    with open_connection(db_path) as conn:
        conn.execute("SELECT 1").fetchone()

    def thread_func(thread_id: int) -> None:
        try:
            with open_connection(db_path) as conn:
                connection_ids.append(id(conn))
                barrier.wait()
                assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
                conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append((thread_id, str(exc)))

    threads = [threading.Thread(target=thread_func, args=(idx,)) for idx in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(connection_ids)) == 3


def test_connection_context_contract(tmp_path: Path, monkeypatch) -> None:
    """connection_context() must support explicit, default, and already-open connections."""
    explicit_path = tmp_path / "explicit.db"
    with connection_context(explicit_path) as explicit_conn:
        assert isinstance(explicit_conn, sqlite3.Connection)
        assert {"artifact_observations", "conversations", "messages", "attachments", "attachment_refs", "publications"}.issubset(
            _table_names(explicit_conn)
        )

    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    import polylogue.paths
    import polylogue.storage.backends.connection as connection_module

    importlib.reload(polylogue.paths)
    importlib.reload(connection_module)

    with connection_module.connection_context(None) as default_conn:
        db_file = Path(default_conn.execute("PRAGMA database_list").fetchone()[2])
        assert db_file.exists()
        assert str(data_home) in str(db_file)

    with connection_module.connection_context(explicit_path) as first_conn:
        with connection_module.connection_context(first_conn) as reused_conn:
            assert reused_conn is first_conn
        assert first_conn.execute("SELECT 1").fetchone()[0] == 1


def test_default_db_path_respects_xdg_data_home(tmp_path: Path, monkeypatch) -> None:
    """default_db_path() must honor XDG_DATA_HOME."""
    xdg_data = tmp_path / "data"
    xdg_data.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

    import polylogue.storage.backends.connection as connection_module

    importlib.reload(connection_module)
    assert str(xdg_data) in str(connection_module.default_db_path())


async def test_sqlite_backend_init_contract(tmp_path: Path, monkeypatch) -> None:
    """SQLiteBackend init must preserve paths, create parents, and allocate the write lock."""
    custom_path = tmp_path / "custom" / "db.sqlite"
    custom_backend = SQLiteBackend(db_path=custom_path)
    assert custom_backend._db_path == custom_path
    assert custom_path.parent.exists()
    assert isinstance(custom_backend._write_lock, asyncio.Lock)
    await custom_backend.close()

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import polylogue.paths

    importlib.reload(polylogue.paths)
    default_backend = SQLiteBackend(db_path=None)
    assert "polylogue" in str(default_backend._db_path)
    assert str(default_backend._db_path).endswith("polylogue.db")
    await default_backend.close()


async def test_async_backend_schema_and_lock_contracts(tmp_path: Path) -> None:
    """Async schema guards and write locks must serialize correctly without blocking readers."""
    backend = SQLiteBackend(db_path=tmp_path / "async.db")

    results = await asyncio.gather(*[backend.get_conversation(f"conv:{idx}") for idx in range(10)])
    assert all(result is None for result in results)
    assert backend._schema_ensured is True

    init_count = 0
    original_ensure_schema = backend._ensure_schema

    async def counting_ensure_schema(conn):
        nonlocal init_count
        init_count += 1
        return await original_ensure_schema(conn)

    backend._schema_ensured = False
    backend._ensure_schema = counting_ensure_schema
    await asyncio.gather(*[backend.queries.list_conversations(ConversationRecordQuery()) for _ in range(20)])
    assert init_count == 1

    slow_backend = SQLiteBackend(db_path=tmp_path / "slow.db")
    events: list[str] = []
    original_slow = slow_backend._ensure_schema

    async def slow_ensure_schema(conn):
        events.append("start")
        await asyncio.sleep(0.05)
        await original_slow(conn)
        events.append("end")

    slow_backend._schema_ensured = False
    slow_backend._ensure_schema = slow_ensure_schema
    await asyncio.gather(slow_backend.get_conversation("a"), slow_backend.get_conversation("b"))
    assert events.count("start") == 1
    assert events.count("end") == 1

    async with backend.transaction():
        assert backend._write_lock.locked()
    assert not backend._write_lock.locked()

    execution_order: list[int] = []

    async def write_operation(task_id: int) -> None:
        async with backend.transaction():
            execution_order.append(task_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(*[write_operation(idx) for idx in range(5)])
    assert set(execution_order) == {0, 1, 2, 3, 4}

    await backend.queries.list_conversations(ConversationRecordQuery())
    assert not backend._write_lock.locked()

    read_completed = False

    async def slow_write() -> None:
        async with backend.transaction():
            await asyncio.sleep(0.1)

    async def quick_read() -> None:
        nonlocal read_completed
        await backend.queries.list_conversations(ConversationRecordQuery())
        read_completed = True

    write_task = asyncio.create_task(slow_write())
    await asyncio.sleep(0.01)
    await asyncio.gather(write_task, asyncio.create_task(quick_read()))
    assert read_completed is True

    await backend.close()
    await backend.close()
    await slow_backend.close()


async def test_async_backend_connection_error_surfaces() -> None:
    """Invalid targets must surface an error instead of silently succeeding."""
    with pytest.raises((OSError, PermissionError, Exception)):
        backend = SQLiteBackend(db_path=Path("/nonexistent/deeply/nested/path/db.db"))
        await backend.get_conversation("test")


async def test_paged_id_iteration_contract(tmp_path: Path) -> None:
    """Bounded ID iterators must preserve canonical descending sort order."""
    backend = SQLiteBackend(db_path=tmp_path / "paged.db")
    base = datetime(2026, 3, 10, tzinfo=timezone.utc)

    for idx in range(5):
        await backend.save_raw_conversation(
            RawConversationRecord(
                raw_id=f"raw-{idx}",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path=f"/tmp/raw-{idx}.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=(base + timedelta(minutes=idx)).isoformat(),
            )
        )
        await backend.save_conversation_record(
            make_conversation(
                f"conv-{idx}",
                provider_name="chatgpt",
                title=f"Conversation {idx}",
                sort_key=float(idx),
            )
        )

    assert [raw_id async for raw_id in backend.iter_raw_ids(page_size=2)] == [
        "raw-4",
        "raw-3",
        "raw-2",
        "raw-1",
        "raw-0",
    ]
    assert [cid async for cid in backend.iter_conversation_ids(page_size=2)] == [
        "conv-4",
        "conv-3",
        "conv-2",
        "conv-1",
        "conv-0",
    ]
    assert await backend.count_conversation_ids() == 5
    await backend.close()


async def test_backend_transaction_contracts(tmp_path: Path) -> None:
    """Explicit and context-managed transactions must commit on success and roll back on failure."""
    backend = SQLiteBackend(db_path=tmp_path / "transaction.db")
    conv_rollback = make_conversation("conv-rollback", title="Rollback")
    await backend.begin()
    await backend.save_conversation_record(conv_rollback)
    await backend.rollback()
    assert await backend.get_conversation("conv-rollback") is None

    conv_commit = make_conversation("conv-commit", title="Commit")
    async with backend.transaction():
        await backend.save_conversation_record(conv_commit)
    committed = await backend.get_conversation("conv-commit")
    assert committed is not None
    assert committed.conversation_id == "conv-commit"

    with pytest.raises(ValueError):
        async with backend.transaction():
            await backend.save_conversation_record(make_conversation("conv-error", title="Error"))
            raise ValueError("Test error")
    assert await backend.get_conversation("conv-error") is None
    await backend.close()


async def test_backend_delete_contracts(tmp_path: Path) -> None:
    """Deleting a conversation must remove rows, attachments, and FTS entries exactly once."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations
    from polylogue.storage.session_product_rebuild import rebuild_session_products_sync

    backend = SQLiteBackend(db_path=tmp_path / "delete.db")
    repo = ConversationRepository(backend=backend)
    conv = make_conversation("conv-delete", title="Delete")
    msg1 = make_message("msg-1", "conv-delete", text="Hello")
    msg2 = make_message("msg-2", "conv-delete", role="assistant", text="Hi there")
    att = make_attachment("att-1", "conv-delete", "msg-1", mime_type="image/png", size_bytes=1024)

    async with backend.transaction():
        await backend.save_conversation_record(conv)
        await backend.save_messages([msg1, msg2])
        await backend.save_attachments([att])

    with open_connection(backend.db_path) as conn:
        ensure_index(conn)
        update_index_for_conversations(["conv-delete"], conn)
        rebuild_session_products_sync(conn)
        conn.commit()
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] > 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] > 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_phases WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM day_session_summaries").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0] >= 1

    assert await repo.delete_conversation("conv-delete") is True
    assert await backend.get_conversation("conv-delete") is None
    assert len(await backend.get_messages("conv-delete")) == 0
    assert len(await backend.get_attachments("conv-delete")) == 0
    assert await repo.delete_conversation("conv-delete") is False

    with open_connection(backend.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_phases WHERE conversation_id = ?",
            ("conv-delete",),
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM day_session_summaries").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0] == 0
    await backend.close()


async def test_backend_lifecycle_reopen_contract(tmp_path: Path) -> None:
    """Closing a backend must be idempotent and future operations may lazily reopen it."""
    backend = SQLiteBackend(db_path=tmp_path / "lifecycle.db")
    await backend.save_conversation_record(_build_record("conv-life"))
    await backend.close()
    await backend.close()

    async with backend._get_connection() as conn:
        assert conn is not None

    assert await backend.get_conversation("conv-life") is not None
    await backend.close()
