"""Daemon FTS startup readiness — extracted from ``polylogue.daemon.cli``.

This module owns the SQLite + FTS-index health probe + recovery flow that
runs once at daemon startup. See ``docs/internals.md`` "FTS5 Model" and
"WAL Management" for the full lifecycle context.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)
_ARCHIVE_MESSAGE_FTS_TRIGGERS = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
_SESSION_WORK_EVENT_FTS_TRIGGERS = (
    "session_work_events_fts_ai",
    "session_work_events_fts_ad",
    "session_work_events_fts_au",
)
_THREAD_FTS_TRIGGERS = ("threads_fts_ai", "threads_fts_ad", "threads_fts_au")
_FTS_STARTUP_BUSY_TIMEOUT_MS = 120_000


def _open_fts_startup_write_connection(db_path: Path) -> sqlite3.Connection:
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(db_path, timeout=_FTS_STARTUP_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_FTS_STARTUP_BUSY_TIMEOUT_MS}")
    except BaseException:
        conn.close()
        raise
    return conn


def missing_fts_triggers_sync(conn: sqlite3.Connection) -> list[str]:
    """Return the names of expected FTS triggers that don't exist on ``conn``."""
    expected = active_fts_triggers_sync(conn)
    if not expected:
        return []
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        expected,
    ).fetchall()
    present = {row[0] for row in rows}
    return [name for name in expected if name not in present]


def table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether ``table_name`` is present in ``sqlite_master``."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def record_fts_freshness_snapshot_sync(conn: sqlite3.Connection) -> None:
    """Write per-surface freshness rows after a successful startup readiness pass.

    Without this, the bounded-repair and healthy startup paths leave
    ``fts_freshness_state`` empty, so ``/healthz/ready`` returns 503
    indefinitely and ``message_fts_search_readiness_sync`` keeps refusing
    queries even though triggers + index are intact (#1628).
    """
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync
    from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

    try:
        snapshot = fts_invariant_snapshot_sync(conn)
    except sqlite3.Error:
        logger.warning("daemon: FTS startup freshness snapshot failed", exc_info=True)
        return
    record_fts_invariant_snapshot_sync(conn, snapshot)


def active_fts_triggers_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return the FTS triggers that should exist given the schema present."""
    expected: list[str] = []
    if table_exists_sync(conn, "blocks") and table_exists_sync(conn, "messages_fts"):
        expected.extend(_ARCHIVE_MESSAGE_FTS_TRIGGERS)
    for table_names, trigger_names in (
        (("session_work_events", "session_work_events_fts"), _SESSION_WORK_EVENT_FTS_TRIGGERS),
        (("threads", "threads_fts"), _THREAD_FTS_TRIGGERS),
    ):
        if all(table_exists_sync(conn, table_name) for table_name in table_names):
            expected.extend(trigger_names)
    return tuple(expected)


def _count_or_zero(conn: sqlite3.Connection, sql: str) -> int:
    try:
        return int(conn.execute(sql).fetchone()[0] or 0)
    except sqlite3.Error:
        return 0


def _missing_named_triggers_sync(conn: sqlite3.Connection, trigger_names: tuple[str, ...]) -> list[str]:
    placeholders = ",".join("?" for _ in trigger_names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        trigger_names,
    ).fetchall()
    present = {row[0] for row in rows}
    return [name for name in trigger_names if name not in present]


def _active_fts_startup_db_path() -> Path:
    from polylogue import paths

    return paths.active_index_db_path()


def _ensure_archive_messages_fts_startup_readiness_sync(conn: sqlite3.Connection) -> bool:
    """Run message FTS startup maintenance when applicable."""
    try:
        has_blocks_table = table_exists_sync(conn, "blocks")
    except Exception:
        return False
    if not has_blocks_table:
        return False

    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_tier(conn, ArchiveTier.INDEX)
    fts_exists = table_exists_sync(conn, "messages_fts")
    docsize_exists = table_exists_sync(conn, "messages_fts_docsize")
    triggers_present = not _missing_named_triggers_sync(conn, _ARCHIVE_MESSAGE_FTS_TRIGGERS)
    if fts_exists and docsize_exists and triggers_present and _message_fts_freshness_ready_sync(conn):
        logger.info("daemon: archive message FTS startup readiness trusted freshness ledger")
        return True
    source_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
    indexed_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM messages_fts_docsize") if docsize_exists else 0
    if fts_exists and (not triggers_present or indexed_rows != source_rows):
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync

        logger.warning("daemon: archive message FTS drift detected on startup. Rebuilding once.")
        rebuild_fts_index_sync(conn)
        indexed_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM messages_fts_docsize")
        triggers_present = not _missing_named_triggers_sync(conn, _ARCHIVE_MESSAGE_FTS_TRIGGERS)

    ready = fts_exists and triggers_present and indexed_rows == source_rows
    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if ready else STALE,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=max(source_rows - indexed_rows, 0),
        excess_rows=max(indexed_rows - source_rows, 0),
        detail=None if ready else "archive startup FTS readiness failed",
    )
    return True


def _message_fts_freshness_ready_sync(conn: sqlite3.Connection) -> bool:
    from polylogue.storage.fts.freshness import (
        MESSAGE_SURFACE,
        ensure_fts_freshness_table_sync,
        freshness_ready_record_trusted,
    )

    try:
        ensure_fts_freshness_table_sync(conn)
        row = conn.execute(
            """
            SELECT state, source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows
            FROM fts_freshness_state
            WHERE surface = ?
            """,
            (MESSAGE_SURFACE,),
        ).fetchone()
    except sqlite3.Error:
        return False
    if row is None:
        return False
    state = str(row[0]) if row[0] is not None else None
    source_rows = _int_or_zero(row[1])
    indexed_rows = _int_or_zero(row[2])
    missing_rows = _int_or_zero(row[3])
    excess_rows = _int_or_zero(row[4])
    duplicate_rows = _int_or_zero(row[5])
    source_has_rows: bool | None = None
    if source_rows == 0 and indexed_rows == 0:
        source_has_rows = _blocks_search_text_has_rows_sync(conn)
    return freshness_ready_record_trusted(
        state=state,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=missing_rows,
        excess_rows=excess_rows,
        duplicate_rows=duplicate_rows,
        source_has_rows=source_has_rows,
    )


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        try:
            return int(value)
        except ValueError:
            return 0
    try:
        return int(str(value))
    except ValueError:
        return 0


def _blocks_search_text_has_rows_sync(conn: sqlite3.Connection) -> bool | None:
    try:
        return conn.execute("SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return None


async def ensure_fts_startup_readiness() -> None:
    """Run daemon startup FTS maintenance without blocking the event loop."""
    await asyncio.to_thread(ensure_fts_startup_readiness_sync)


def ensure_fts_startup_readiness_sync() -> None:
    """Ensure FTS triggers and index are healthy on daemon startup.

    Three failure modes are recovered here:

    1. FTS table missing entirely (historical rows pre-date FTS) →
       rebuild from messages.
    2. FTS table exists but is empty while messages exist → rebuild.
    3. FTS triggers missing (the SIGKILL-during-bulk-suspend signature
       called out in ``docs/internals.md`` "FTS5 Model → Risk") →
       restore triggers before serving. The bulk-ingest path drops
       these triggers as DDL (auto-committed); a SIGKILL between the
       drop and the matching restore leaves them gone across process
       death, and subsequent writes silently bypass the FTS index. See
       #1242.

    This path restores obvious broken structure first and records durable
    freshness state for search request handlers. It deliberately avoids exact
    full-archive invariant scans during ordinary startup; bounded missing-row
    repair is enough for the normal SIGKILL-after-trigger-suspend failure mode.
    """
    db = _active_fts_startup_db_path()
    if not db.exists():
        return

    conn: sqlite3.Connection | None = None
    try:
        conn = _open_fts_startup_write_connection(db)
        if _ensure_archive_messages_fts_startup_readiness_sync(conn):
            conn.commit()
            return
        logger.info("daemon: FTS startup check skipped — current archive blocks table absent.")
        return
    except Exception:
        logger.warning("daemon: FTS startup check failed", exc_info=True)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


__all__ = [
    "active_fts_triggers_sync",
    "ensure_fts_startup_readiness",
    "ensure_fts_startup_readiness_sync",
    "missing_fts_triggers_sync",
    "record_fts_freshness_snapshot_sync",
    "table_exists_sync",
]
