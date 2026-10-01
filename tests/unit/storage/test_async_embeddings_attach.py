"""Async index connections that attach ``embeddings.db`` can read its vec0 table."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest

from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_async
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.async_sqlite import configure_connection, configure_read_connection


@pytest.mark.parametrize("read_only", [True, False], ids=["read", "write"])
def test_attached_embeddings_are_measurable_on_async_index_connections(tmp_path: Path, read_only: bool) -> None:
    """Embedding coverage on a fresh archive is a measured zero, not unmeasurable.

    ``message_embeddings`` is a vec0 virtual table. Anti-vacuity: drop the
    sqlite-vec load in ``_attach_sibling_tiers`` and the count raises
    "no such module: vec0", so ``embedded_messages`` comes back ``None``.
    """
    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    index_db = root / "index.db"

    async def measure() -> tuple[int, int | None]:
        target, uri = (f"file:{index_db}?mode=ro", True) if read_only else (str(index_db), False)
        async with aiosqlite.connect(target, uri=uri) as conn:
            await (configure_read_connection if read_only else configure_connection)(conn, archive_root=root)
            cursor = await conn.execute("SELECT COUNT(*) FROM message_embeddings")
            row = await cursor.fetchone()
            assert row is not None
            snapshot = await read_embedding_stats_async(conn)
            return int(row[0]), snapshot.embedded_messages

    count, embedded = asyncio.run(measure())
    assert count == 0
    assert embedded == 0


def test_pending_population_refuses_async_constructor_and_cached_read_pool(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)

    async def exercise() -> None:
        backend = SQLiteBackend(root / "index.db")
        async with backend.read_pool(size=1):
            marker = root / POPULATION_PENDING
            marker.write_text('{"fixture":"unfinished-population"}')
            try:
                with pytest.raises(ArchivePopulationPendingError):
                    SQLiteBackend(root / "index.db")
                with pytest.raises(ArchivePopulationPendingError):
                    async with backend.connection():
                        pytest.fail("a cached async connection crossed the pending fence")
                with pytest.raises(ArchivePopulationPendingError):
                    async with backend._get_read_connection():
                        pytest.fail("a cached async query connection crossed the pending fence")
            finally:
                marker.unlink()

    asyncio.run(exercise())


@pytest.mark.parametrize("read_only", [True, False], ids=["read", "write"])
def test_pending_population_refuses_async_configuration_before_pragmas_or_attachments(
    tmp_path: Path, read_only: bool
) -> None:
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)

    async def exercise() -> None:
        async with aiosqlite.connect(root / "index.db") as conn:
            async with conn.execute("PRAGMA user_version") as cursor:
                version = await cursor.fetchone()
            marker = root / POPULATION_PENDING
            marker.write_text('{"fixture":"unfinished-population"}')
            try:
                with pytest.raises(ArchivePopulationPendingError):
                    await (configure_read_connection if read_only else configure_connection)(conn, archive_root=root)
                async with conn.execute("PRAGMA database_list") as cursor:
                    assert [row[1] for row in await cursor.fetchall()] == ["main"]
                async with conn.execute("PRAGMA user_version") as cursor:
                    assert await cursor.fetchone() == version
            finally:
                marker.unlink()

    asyncio.run(exercise())


def test_async_pool_refusal_closes_prior_and_new_configuration_handles(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import async_sqlite
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = workspace_env["archive_root"]
    handles: list[aiosqlite.Connection] = []
    configure = async_sqlite.configure_read_connection

    async def configure_with_pending(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
        handles.append(conn)
        if len(handles) == 2:
            (root / POPULATION_PENDING).write_text('{"fixture":"unfinished-population"}')
        await configure(conn, archive_root=root)

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(root / "index.db")
        await backend._ensure_schema_once()
        monkeypatch.setattr(async_sqlite, "configure_read_connection", configure_with_pending)
        try:
            with pytest.raises(ArchivePopulationPendingError):
                async with backend.read_pool(size=2):
                    pytest.fail("a refused pool was published")
            assert len(handles) == 2
            assert backend._read_pool is None
            for conn in handles:
                with pytest.raises(ValueError):
                    await conn.execute("SELECT 1")
        finally:
            (root / POPULATION_PENDING).unlink()

    asyncio.run(exercise())


@pytest.mark.parametrize("route", ["bulk", "transaction", "begin"])
def test_async_writer_refusal_closes_unconfigured_handle_without_publishing_it(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    from polylogue.storage.sqlite import async_sqlite
    from polylogue.storage.sqlite.population_admission import POPULATION_PENDING, ArchivePopulationPendingError

    root = workspace_env["archive_root"]
    handles: list[aiosqlite.Connection] = []
    configure = async_sqlite.configure_connection

    async def configure_with_pending(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
        handles.append(conn)
        (root / POPULATION_PENDING).write_text('{"fixture":"unfinished-population"}')
        await configure(conn, archive_root=root)

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(root / "index.db")
        await backend._ensure_schema_once()
        monkeypatch.setattr(async_sqlite, "configure_connection", configure_with_pending)
        try:
            with pytest.raises(ArchivePopulationPendingError):
                if route == "begin":
                    await backend.begin()
                else:
                    async with backend.bulk_connection() if route == "bulk" else backend.transaction():
                        pytest.fail("a refused writer was published")
            assert len(handles) == 1
            assert backend._txn_conn is None and backend._bulk_conn is None
            assert backend.transaction_depth == 0
            with pytest.raises(ValueError):
                await handles[0].execute("SELECT 1")
        finally:
            (root / POPULATION_PENDING).unlink()

    asyncio.run(exercise())


def test_pool_refusal_retains_failed_raw_handles_and_attempts_all_closes(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(workspace_env["archive_root"] / "index.db")
        await backend._ensure_schema_once()
        handles = []
        close_attempts = []
        configure = async_sqlite.configure_read_connection
        execute = cast(Callable[..., Awaitable[Any]], aiosqlite.Connection._execute)
        refuse_close = True
        primary = ValueError("synthetic configuration refusal")

        async def configure_last(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
            handles.append(conn)
            await configure(conn, archive_root=archive_root)
            if len(handles) == 3:
                raise primary

        async def execute_with_close_fault(conn: aiosqlite.Connection, function: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(function, "__name__", None) == "close_raw":
                close_attempts.append(conn)
                if refuse_close and conn in (handles[0], handles[-1]):
                    raise OSError("synthetic native close refusal")
            return await execute(conn, function, *args, **kwargs)

        monkeypatch.setattr(async_sqlite, "configure_read_connection", configure_last)
        monkeypatch.setattr(aiosqlite.Connection, "_execute", execute_with_close_fault)
        try:
            with pytest.raises(ValueError) as caught:
                async with backend.read_pool(size=3):
                    pytest.fail("a refused pool was published")
            assert caught.value is primary
            assert len(close_attempts) == 3 and set(close_attempts) == set(handles)
            assert backend._read_pool is None
            for conn in (handles[0], handles[2]):
                assert conn._connection is not None and conn._running
                assert async_sqlite._BACKEND_CONNECTIONS[id(conn)].backend is backend
                async with conn.execute("SELECT 1") as cursor:
                    row = await cursor.fetchone()
                    assert row is not None and row[0] == 1
            assert handles[1]._connection is None
        finally:
            refuse_close = False
            await backend.close()
        assert all(conn._connection is None and not conn._thread.is_alive() for conn in handles)
        assert not any(entry.backend is backend for entry in async_sqlite._BACKEND_CONNECTIONS.values())

    asyncio.run(exercise())


def test_failed_writer_configuration_keeps_actual_handle_until_backend_retirement(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(workspace_env["archive_root"] / "index.db")
        await backend._ensure_schema_once()
        handles = []
        primary = ValueError("synthetic writer configuration refusal")
        execute = cast(Callable[..., Awaitable[Any]], aiosqlite.Connection._execute)
        refuse_close = True

        async def configure(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
            handles.append(conn)
            raise primary

        async def execute_with_close_fault(conn: aiosqlite.Connection, function: Any, *args: Any, **kwargs: Any) -> Any:
            if refuse_close and getattr(function, "__name__", None) == "close_raw":
                raise OSError("synthetic native close refusal")
            return await execute(conn, function, *args, **kwargs)

        monkeypatch.setattr(async_sqlite, "configure_connection", configure)
        monkeypatch.setattr(aiosqlite.Connection, "_execute", execute_with_close_fault)
        try:
            with pytest.raises(ValueError) as caught:
                await backend.begin()
            assert caught.value is primary
            conn = handles[0]
            assert backend._txn_conn is None
            assert async_sqlite._BACKEND_CONNECTIONS[id(conn)].backend is backend
            assert conn._connection is not None and conn._running
        finally:
            refuse_close = False
            await backend.close()
        assert conn._connection is None
        assert id(conn) not in async_sqlite._BACKEND_CONNECTIONS

    asyncio.run(exercise())


def test_close_settlement_still_closes_raw_handle_after_rollback_failure(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(workspace_env["archive_root"] / "index.db")
        conn = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
        failure = OSError("synthetic rollback failure")

        async def rollback() -> None:
            raise failure

        monkeypatch.setattr(conn, "rollback", rollback)
        with pytest.raises(OSError) as caught:
            await async_sqlite._close_backend_connection(conn, rollback=True)
        assert caught.value is failure
        assert conn._connection is None and not conn._running
        assert not conn._thread.is_alive()
        assert id(conn) not in async_sqlite._BACKEND_CONNECTIONS
        await backend.close()

    asyncio.run(exercise())


def test_cancelled_close_waiter_drains_actual_worker_before_retiring_handle(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from polylogue.storage.sqlite import async_sqlite

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(workspace_env["archive_root"] / "index.db")
        conn = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
        entered, release = threading.Event(), threading.Event()
        execute = cast(Callable[..., Awaitable[Any]], conn._execute)

        async def delay_close(function: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(function, "__name__", None) == "close_raw":

                def queued_close() -> None:
                    entered.set()
                    release.wait()
                    function()

                return await execute(queued_close)
            return await execute(function, *args, **kwargs)

        monkeypatch.setattr(conn, "_execute", delay_close)
        closing = asyncio.create_task(async_sqlite._close_backend_connection(conn))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
            assert conn._connection is not None and id(conn) in async_sqlite._BACKEND_CONNECTIONS
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
        assert conn._connection is None and not conn._running
        assert not conn._thread.is_alive()
        assert id(conn) not in async_sqlite._BACKEND_CONNECTIONS
        await backend.close()

    asyncio.run(exercise())


def test_failed_worker_stop_retains_owner_until_actual_thread_exit(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import async_sqlite

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(workspace_env["archive_root"] / "index.db")
        conn = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
        stop = conn.stop
        failure = OSError("synthetic worker stop refusal")

        def refuse_stop() -> None:
            raise failure

        monkeypatch.setattr(conn, "stop", refuse_stop)
        try:
            with pytest.raises(OSError) as caught:
                await async_sqlite._close_backend_connection(conn)
            assert caught.value is failure
            assert conn._connection is None
            assert conn._thread.is_alive()
            assert id(conn) in async_sqlite._BACKEND_CONNECTIONS
        finally:
            monkeypatch.setattr(conn, "stop", stop)
            await backend.close()
        assert not conn._thread.is_alive()
        assert id(conn) not in async_sqlite._BACKEND_CONNECTIONS

    asyncio.run(exercise())


def test_failed_connection_construction_drains_its_already_stopping_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queuing a second stop sentinel leaves the production opener suspended."""
    from polylogue.storage.sqlite import async_sqlite

    connections: list[aiosqlite.Connection] = []
    connect = aiosqlite.connect

    def capture_connection(*args: Any, **kwargs: Any) -> aiosqlite.Connection:
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(aiosqlite, "connect", capture_connection)

    async def exercise() -> None:
        backend = async_sqlite.SQLiteBackend(tmp_path / "index.db")
        backend.db_path.unlink()
        with pytest.raises(sqlite3.OperationalError):
            _ = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
        assert len(connections) == 1
        conn = connections[0]
        assert conn._connection is None
        assert not conn._thread.is_alive()
        assert id(conn) not in async_sqlite._BACKEND_CONNECTIONS
        await backend.close()

    asyncio.run(exercise())
