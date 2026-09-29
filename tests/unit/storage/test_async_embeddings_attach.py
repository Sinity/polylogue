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
