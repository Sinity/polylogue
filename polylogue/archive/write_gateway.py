"""Archive write gateway for committing archive write side effects.

The sync ingest writer owns row materialization today. This gateway owns the
commit/effects boundary after those rows are written: restoring FTS triggers,
repairing affected indexes, committing, and invalidating read caches.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from polylogue.storage.sqlite.connection_profile import open_connection as _open_conn

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    INGEST = "ingest"
    RESET = "reset"
    DELETE = "delete"
    TAG_UPDATE = "tag_update"
    METADATA_UPDATE = "metadata_update"


WriteResultStatus = Literal["committed", "rejected", "deferred"]


@dataclass(frozen=True, slots=True)
class WriteResult:
    operation_id: str
    operation: WriteOperation
    rows_affected: int
    status: WriteResultStatus


class ArchiveWriteGateway:
    """Commit archive write side effects through one production-wired path.

    Parameters
    ----------
    db_path:
        Path to the archive SQLite database.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._write_lock = asyncio.Lock()
        self._sync_write_lock = RLock()

    async def commit_write(self, op: WriteOperation, payload: dict[str, Any]) -> WriteResult:
        """Commit write side effects asynchronously.

        Parameters
        ----------
        op:
            The write operation type.
        payload:
            Operation payload. If a ``_connection`` key is present with an
            open ``sqlite3.Connection`` value, the gateway will use that
            connection (the caller owns its lifecycle). Otherwise the gateway
            opens and closes its own connection.
        """
        async with self._write_lock:
            if "_connection" in payload:
                return self.commit_write_sync(op, payload)
            return await asyncio.to_thread(self.commit_write_sync, op, payload)

    def commit_write_sync(self, op: WriteOperation, payload: dict[str, Any]) -> WriteResult:
        """Commit write side effects synchronously."""
        with self._sync_write_lock:
            return self._execute_local_sync(op, payload)

    def _execute_local_sync(self, op: WriteOperation, payload: dict[str, Any]) -> WriteResult:
        """Run commit/effects locally, optionally on a caller-owned connection."""
        from polylogue.archive.write_effects import commit_archive_write_effects

        payload = dict(payload)
        conn: sqlite3.Connection | None = payload.pop("_connection", None)
        owns_conn = conn is None

        if owns_conn:
            conn = _open_conn(self._db_path)

        assert conn is not None
        try:
            return commit_archive_write_effects(conn, op, payload)
        finally:
            if owns_conn:
                conn.close()


__all__ = [
    "ArchiveWriteGateway",
    "WriteOperation",
    "WriteResult",
    "WriteResultStatus",
]
