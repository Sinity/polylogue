"""Archive write gateway -- single canonical path for all archive writes.

Reads always use direct SQLite. Writes route through this gateway,
which either delegates to the daemon (if available) or acquires a
process-level advisory lock and runs the write contract locally.

This reconciles "daemon owns all writes" with "library-first, daemon optional."
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    INGEST = "ingest"
    RESET = "reset"
    DELETE = "delete"
    TAG_UPDATE = "tag_update"
    METADATA_UPDATE = "metadata_update"
    BLOB_STORE = "blob_store"


@dataclass(frozen=True, slots=True)
class WriteResult:
    operation_id: str
    operation: WriteOperation
    rows_affected: int
    status: str  # "committed", "rejected", "deferred"


class DaemonUnavailableError(Exception):
    """Raised when the daemon RPC target is unreachable or not running."""


class ArchiveWriteGateway:
    """Gateway for all archive write operations.

    Protocol
    --------
    1. Attempt daemon RPC if daemon is running.
    2. If daemon not available, acquire process-level advisory lock,
       run the write contract locally, then release.
    3. Daemon detects external writes via PRAGMA data_version before
       continuing its own work.

    Parameters
    ----------
    db_path:
        Path to the archive SQLite database.
    daemon_port:
        Optional daemon RPC port. If ``None``, all writes run locally.
    """

    def __init__(self, db_path: str, *, daemon_port: int | None = None) -> None:
        self._db_path = db_path
        self._daemon_port = daemon_port
        self._write_lock = asyncio.Lock()

    async def commit_write(self, op: WriteOperation, payload: dict[str, Any]) -> WriteResult:
        """Execute a write through the canonical path.

        If the daemon is available, delegates to it. Otherwise acquires
        the process-level write lock and runs locally.

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
        if self._daemon_port is not None:
            try:
                return await self._delegate_to_daemon(op, payload)
            except DaemonUnavailableError:
                logger.debug("daemon unavailable, falling back to direct write")

        async with self._write_lock:
            return await self._execute_local(op, payload)

    async def _delegate_to_daemon(
        self,
        op: WriteOperation,
        payload: dict[str, Any],
    ) -> WriteResult:
        """Delegate a write to the daemon process via RPC.

        This is a stub -- daemon RPC is deferred to slice G (#717).
        """
        raise DaemonUnavailableError("Daemon RPC not yet implemented. Set daemon_port=None or wait for slice G (#717).")

    async def _execute_local(
        self,
        op: WriteOperation,
        payload: dict[str, Any],
    ) -> WriteResult:
        """Execute a write operation locally with side effects.

        If the payload contains a ``_connection`` key, uses that connection
        (caller-owned lifecycle). Otherwise opens a fresh connection to
        *db_path*.
        """
        from polylogue.archive.write_effects import commit_archive_write_effects

        conn: sqlite3.Connection | None = payload.pop("_connection", None)
        owns_conn = conn is None

        if owns_conn:
            conn = sqlite3.connect(self._db_path)

        assert conn is not None
        try:
            return await asyncio.to_thread(commit_archive_write_effects, conn, op, payload)
        finally:
            if owns_conn:
                conn.close()


__all__ = [
    "ArchiveWriteGateway",
    "DaemonUnavailableError",
    "WriteOperation",
    "WriteResult",
]
