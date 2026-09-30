"""Async index connections that attach ``embeddings.db`` can read its vec0 table."""

from __future__ import annotations

import asyncio
from pathlib import Path

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
            await (configure_read_connection if read_only else configure_connection)(conn)
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
                    await (configure_read_connection if read_only else configure_connection)(conn)
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

    async def configure_with_pending(conn: aiosqlite.Connection) -> None:
        handles.append(conn)
        if len(handles) == 2:
            (root / POPULATION_PENDING).write_text('{"fixture":"unfinished-population"}')
        await configure(conn)

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

    async def configure_with_pending(conn: aiosqlite.Connection) -> None:
        handles.append(conn)
        (root / POPULATION_PENDING).write_text('{"fixture":"unfinished-population"}')
        await configure(conn)

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
