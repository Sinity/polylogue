"""Live archive-write helpers for tests.

Tests seed the archive through the same ``write_parsed_session_to_archive``
path the daemon uses — there is no separate test-only ingest stack. This
helper is the single seam: hand it a :class:`ParsedSession` and a backend (or
a database path) and it writes the canonical session/message/block/attachment/
event rows and returns the archive ``session_id``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


def write_session_sync(
    db_path: Path,
    session: ParsedSession,
    *,
    raw_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Write one parsed session through the live archive writer (sync).

    Opens a dedicated sync connection on ``db_path`` so writers and async
    readers do not share a connection. Returns the archive ``session_id``.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        return write_parsed_session_to_archive(
            conn,
            session,
            content_hash=content_hash if content_hash is not None else session_content_hash(session),
            raw_id=raw_id,
        )
    finally:
        conn.close()


async def ingest_session(
    session: ParsedSession,
    backend: SQLiteBackend,
    *,
    raw_id: str | None = None,
    content_hash: str | None = None,
) -> str:
    """Write one parsed session via the live writer; return the archive ``session_id``.

    Runs the sync writer in a worker thread so it composes with async tests
    that read back through ``backend.connection()``.
    """
    return await asyncio.to_thread(
        write_session_sync,
        backend.db_path,
        session,
        raw_id=raw_id,
        content_hash=content_hash,
    )


__all__ = ["ingest_session", "write_session_sync"]
