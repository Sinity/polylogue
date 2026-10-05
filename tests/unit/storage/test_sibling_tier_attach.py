"""Async index reads attach every sibling tier of a freshly bootstrapped root."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop


@pytest.mark.asyncio
async def test_fresh_root_read_connection_attaches_embeddings_without_identity_row(tmp_path: Path) -> None:
    """Requiring a derived schema identity on ``embeddings.db`` makes this raise.

    The embeddings tier is not a ``DerivedTier`` and bootstrap stamps no
    ``schema_identity`` table in it, so every async read on a fresh archive
    failed with ``no such table: embeddings.schema_identity``.
    """
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    try:
        async with backend.read_connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM embeddings.sqlite_master")
            row = await cursor.fetchone()
            assert row is not None and row[0] > 0
    finally:
        await backend.close()
