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
from polylogue.storage.sqlite.archive_tiers.write import ArchiveWriteOutcome, write_parsed_session_to_archive
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


def write_session_counts_sync(
    db_path: Path,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
) -> dict[str, int]:
    """Write an index-only fixture and return the writer's count contract."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    outcomes: list[ArchiveWriteOutcome] = []
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        write_parsed_session_to_archive(
            conn,
            session,
            content_hash=content_hash if content_hash is not None else session_content_hash(session),
            write_outcome=outcomes,
        )
        skipped = bool(outcomes and getattr(outcomes[0], "stale_skipped", False))
        if skipped:
            counts = {
                "sessions": 0,
                "messages": 0,
                "attachments": 0,
                "session_events": 0,
                "skipped_sessions": 1,
                "skipped_messages": len(session.messages),
                "skipped_attachments": len(session.attachments),
                "skipped_session_events": len(session.session_events),
                "raw_links": 0,
            }
        else:
            counts = {
                "sessions": 1,
                "messages": len(session.messages),
                "attachments": len(session.attachments),
                "session_events": len(session.session_events),
                "skipped_sessions": 0,
                "skipped_messages": 0,
                "skipped_attachments": 0,
                "skipped_session_events": 0,
                "raw_links": 0,
            }
        counts["stale_skipped"] = int(skipped)
        conn.commit()
        return counts
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


def write_index_session(
    archive: object,
    session: ParsedSession,
    *,
    content_hash: str | None = None,
) -> str:
    """Seed an index-only fixture through the canonical row writer.

    These fixtures intentionally have no raw acquisition to admit. Production
    ingest must use ``write_raw_and_parsed_result`` instead.
    """
    db_path = getattr(archive, "index_db_path", None)
    if not isinstance(db_path, Path):
        raise TypeError("index-only fixture requires an ArchiveStore index_db_path")
    if any(
        attachment.inline_bytes is not None or attachment.precomputed_blob is not None
        for attachment in session.attachments
    ):
        preacquire = getattr(archive, "_preacquire_attachment_blobs", None)
        connection = getattr(archive, "_conn", None)
        publisher = getattr(archive, "_blob_publisher", None)
        if not callable(preacquire) or not isinstance(connection, sqlite3.Connection) or publisher is None:
            raise TypeError("index-only attachment fixture requires an archive-owned blob publisher")
        acquired, refs = preacquire(
            session,
            source_path=f"session:{session.provider_session_id}",
            acquired_at_ms=0,
        )
        publisher.flush()
        try:
            session_id = write_parsed_session_to_archive(
                connection,
                session,
                content_hash=content_hash if content_hash is not None else session_content_hash(session),
                preacquired_attachment_blobs=acquired,
            )
            pending = getattr(archive, "_pending_index_blob_receipts", None)
            consume = getattr(archive, "_consume_index_blob_receipts", None)
            if not isinstance(pending, list) or not callable(consume):
                raise TypeError("index-only attachment fixture requires archive receipt handling")
            pending.extend(
                (ref.publication_receipt_id, ref.blob_hash) for ref in refs if ref.publication_receipt_id is not None
            )
            consume()
            return session_id
        except Exception:
            publisher.discard_pending()
            raise
    return write_session_sync(db_path, session, content_hash=content_hash)


__all__ = ["ingest_session", "write_index_session", "write_session_counts_sync", "write_session_sync"]
