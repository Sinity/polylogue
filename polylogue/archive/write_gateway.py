"""Archive write gateway for committing declared archive writer transactions.

Each writer owns row materialization in its tier. This gateway owns the
commit/effects boundary: index writers repair affected derived products, while
user-overlay writers commit through the same boundary without scheduling
index-only FTS or cache work.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal

from polylogue.storage.sqlite.connection_profile import open_connection as _open_conn

if TYPE_CHECKING:
    from polylogue.archive.write_effects import WriteEffectReceipt


class WriteOperation(Enum):
    INGEST = "ingest"
    RESET = "reset"
    DELETE = "delete"
    TAG_UPDATE = "tag_update"
    METADATA_UPDATE = "metadata_update"


WriteEffectScope = Literal["archive-index", "user-overlay"]


@dataclass(frozen=True, slots=True)
class WriteOperationPolicy:
    """One production writer family at the shared commit boundary.

    ``archive-index`` writes can stale index-derived products and therefore
    run the registered archive effects. ``user-overlay`` writes commit through
    the same gateway but deliberately skip those effects: user.db assertions
    do not modify indexed session content, FTS, or the search cache.
    """

    scope: WriteEffectScope
    run_archive_effects: bool
    actuator: str


WRITE_OPERATION_POLICIES: dict[WriteOperation, tuple[WriteOperationPolicy, ...]] = {
    WriteOperation.INGEST: (WriteOperationPolicy("archive-index", True, "ingest batch session writer"),),
    WriteOperation.RESET: (
        WriteOperationPolicy("archive-index", True, "identity-reset index session delete"),
        WriteOperationPolicy("user-overlay", False, "identity-reset suppression assertion writer"),
    ),
    WriteOperation.DELETE: (WriteOperationPolicy("archive-index", True, "ArchiveStore.delete_sessions"),),
    WriteOperation.TAG_UPDATE: (
        WriteOperationPolicy("user-overlay", False, "ArchiveStore user tag assertion writers"),
    ),
    WriteOperation.METADATA_UPDATE: (
        WriteOperationPolicy("user-overlay", False, "ArchiveStore user metadata assertion writers"),
    ),
}
"""Exhaustive production admission inventory for declared write operations."""


def write_operation_policy_for(op: WriteOperation, scope: WriteEffectScope) -> WriteOperationPolicy:
    """Return the declared effect policy for one real writer transaction."""
    for policy in WRITE_OPERATION_POLICIES[op]:
        if policy.scope == scope:
            return policy
    raise ValueError(f"write operation {op.value!r} has no policy for {scope!r}")


WriteResultStatus = Literal["committed", "rejected", "deferred"]


@dataclass(frozen=True, slots=True)
class WriteResult:
    operation_id: str
    operation: WriteOperation
    rows_affected: int
    status: WriteResultStatus
    effect_receipts: tuple[WriteEffectReceipt, ...] = ()


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
        payload.setdefault("_db_path", self._db_path)
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
    "WRITE_OPERATION_POLICIES",
    "WriteEffectScope",
    "WriteOperationPolicy",
    "WriteOperation",
    "WriteResult",
    "WriteResultStatus",
    "write_operation_policy_for",
]
