"""Production ingest acceptance/outbox transaction wiring."""

from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.pipeline.services.ingest_batch._core import _persist_batch_raw_state_updates
from polylogue.sinex.material_adapter import PublicationEncodingError
from polylogue.sinex.models import PublicationMode
from polylogue.sinex.obligations import PublicationPayloadInvalidError
from tests.unit.sinex._fixtures import publication_payload


class _AsyncCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> object | None:
        return cast(object | None, self._cursor.fetchone())

    async def fetchall(self) -> list[object]:
        return cast(list[object], self._cursor.fetchall())


class _AsyncConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _AsyncCursor:
        return _AsyncCursor(self._conn.execute(sql, parameters))


class _SourceBackend:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.active: sqlite3.Connection | None = None

    @asynccontextmanager
    async def bulk_connection(self) -> AsyncIterator[None]:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        self.active = conn
        try:
            yield None
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            self.active = None
            conn.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[_AsyncConnection]:
        assert self.active is not None
        yield _AsyncConnection(self.active)


class _Repository:
    def __init__(self, backend: _SourceBackend) -> None:
        self._source_backend = backend

    @property
    def source_backend(self) -> _SourceBackend:
        return self._source_backend

    async def update_raw_state(self, raw_id: str, *, state: object) -> None:
        assert self._source_backend.active is not None
        self._source_backend.active.execute(
            "INSERT OR REPLACE INTO test_raw_acceptance(raw_id, accepted) VALUES (?, 1)",
            (raw_id,),
        )


def _counts(path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(path)
    try:
        accepted_row = conn.execute("SELECT COUNT(*) FROM test_raw_acceptance").fetchone()
        obligation_row = conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()
        assert accepted_row is not None
        assert obligation_row is not None
        return (int(accepted_row[0]), int(obligation_row[0]))
    finally:
        conn.close()


def test_real_raw_state_helper_commits_or_rolls_back_acceptance_with_payload(
    workspace_env: dict[str, Path],
) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    conn = sqlite3.connect(source_db)
    conn.execute("CREATE TABLE IF NOT EXISTS test_raw_acceptance(raw_id TEXT PRIMARY KEY, accepted INTEGER NOT NULL)")
    conn.commit()
    conn.close()
    backend = _SourceBackend(source_db)
    service = SimpleNamespace(repository=_Repository(backend))

    invalid = dataclasses.replace(publication_payload(), manifest_digest="0" * 64)
    with pytest.raises(PublicationPayloadInvalidError):
        asyncio.run(
            _persist_batch_raw_state_updates(
                service,
                backend,
                outcomes={},
                succeeded_raw_ids={"raw-1"},
                skipped_raw_ids=set(),
                failed_raw_ids={},
                validation_mode="advisory",
                publication_mode=PublicationMode.PRIMARY,
                publication_payloads_by_raw_id={"raw-1": [invalid]},
            )
        )
    assert _counts(source_db) == (0, 0)

    valid = publication_payload()
    asyncio.run(
        _persist_batch_raw_state_updates(
            service,
            backend,
            outcomes={},
            succeeded_raw_ids={"raw-1"},
            skipped_raw_ids=set(),
            failed_raw_ids={},
            validation_mode="advisory",
            publication_mode=PublicationMode.PRIMARY,
            publication_payloads_by_raw_id={"raw-1": [valid]},
        )
    )
    assert _counts(source_db) == (1, 1)


def test_backed_mode_refuses_acceptance_without_source_tier_backend() -> None:
    service = SimpleNamespace(repository=SimpleNamespace(source_backend=None))
    backend = SimpleNamespace()
    with pytest.raises(PublicationEncodingError, match="durable source-tier"):
        asyncio.run(
            _persist_batch_raw_state_updates(
                service,
                backend,
                outcomes={},
                succeeded_raw_ids={"raw-1"},
                skipped_raw_ids=set(),
                failed_raw_ids={},
                validation_mode="advisory",
                publication_mode=PublicationMode.MIRROR,
                publication_payloads_by_raw_id={"raw-1": [publication_payload()]},
            )
        )
