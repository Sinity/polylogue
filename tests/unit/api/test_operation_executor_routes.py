"""Production facade routes for derived maintenance mutations.

These tests deliberately enter the public async facade. They fail if a route
stops constructing the real actuator or bypasses ``OperationExecutor`` and
calls the storage rebuild primitive directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.operations.mutation_transaction import OperationExecutor
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_archive(archive_root: Path, *, native_id: str) -> str:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    raw_id = f"raw-{native_id}"
    session_id = f"codex-session:{native_id}"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)
            """,
            (raw_id, native_id, str(archive_root / f"{native_id}.jsonl")),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
            """,
            (native_id, raw_id, f"Maintenance route {native_id}"),
        )
    return session_id


@pytest.mark.asyncio
async def test_facade_rebuild_and_update_index_use_executor_and_real_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_id = _seed_archive(archive_root, native_id="route-index")
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    calls: list[str] = []
    original_execute = OperationExecutor.execute

    def record_execute(self: OperationExecutor, actuator, plan, authorization, args):  # type: ignore[no-untyped-def]
        calls.append(actuator.operation)
        return original_execute(self, actuator, plan, authorization, args)

    monkeypatch.setattr(OperationExecutor, "execute", record_execute)
    try:
        assert await archive.rebuild_index() is True
        assert await archive.update_index([session_id]) is True
    finally:
        await archive.close()

    assert calls == ["mutate-rebuild-index", "mutate-update-index"]


@pytest.mark.asyncio
async def test_facade_rebuild_insights_uses_executor_and_real_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    session_id = _seed_archive(archive_root, native_id="route-insights")
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    calls: list[str] = []
    original_execute = OperationExecutor.execute

    def record_execute(self: OperationExecutor, actuator, plan, authorization, args):  # type: ignore[no-untyped-def]
        calls.append(actuator.operation)
        return original_execute(self, actuator, plan, authorization, args)

    monkeypatch.setattr(OperationExecutor, "execute", record_execute)
    try:
        counts = await archive.rebuild_insights(session_ids=[session_id])
    finally:
        await archive.close()

    assert calls == ["mutate-rebuild-insights"]
    assert counts.total() >= 0
