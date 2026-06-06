"""Daemon FTS startup readiness — extracted from ``polylogue.daemon.cli``.

This module owns the SQLite + FTS-index health probe + recovery flow that
runs once at daemon startup. See ``docs/internals.md`` "FTS5 Model" and
"WAL Management" for the full lifecycle context.

The pre-existing public re-exports (``polylogue.daemon.cli._missing_fts_triggers_sync``,
``polylogue.daemon.cli._table_exists_sync``, ``polylogue.daemon.cli._record_fts_freshness_snapshot_sync``,
``polylogue.daemon.cli._active_fts_triggers_sync``, ``polylogue.daemon.cli._ensure_fts_startup_readiness``,
``polylogue.daemon.cli._ensure_fts_startup_readiness_sync``) are kept as
thin proxies that forward to this module, so existing tests and any
external code that imported through ``daemon.cli`` continue to work.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.storage.fts.fts_lifecycle import FTS_TRIGGER_NAMES as _EXPECTED_FTS_TRIGGERS

logger = get_logger(__name__)
_ARCHIVE_BLOCKS_FTS_TRIGGERS = ("blocks_fts_ai", "blocks_fts_ad", "blocks_fts_au")


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
    if table_exists_sync(conn, "blocks") and table_exists_sync(conn, "blocks_fts"):
        expected.extend(_ARCHIVE_BLOCKS_FTS_TRIGGERS)
    if table_exists_sync(conn, "messages") and table_exists_sync(conn, "messages_fts"):
        expected.extend(_EXPECTED_FTS_TRIGGERS[: 6 if table_exists_sync(conn, "content_blocks") else 3])
    for table_names, trigger_names in (
        (("action_events", "action_events_fts"), _EXPECTED_FTS_TRIGGERS[6:9]),
        (("session_work_events", "session_work_events_fts"), _EXPECTED_FTS_TRIGGERS[9:12]),
        (("work_threads", "work_threads_fts"), _EXPECTED_FTS_TRIGGERS[12:15]),
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

    return paths.resolve_active_index_db_path(db_anchor=paths.db_path(), index_db=paths.index_db_path())


def _ensure_archive_blocks_fts_startup_readiness_sync(conn: sqlite3.Connection) -> bool:
    """Run blocks FTS startup maintenance when applicable."""
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
    source_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM blocks WHERE text IS NOT NULL")
    fts_exists = table_exists_sync(conn, "blocks_fts")
    docsize_exists = table_exists_sync(conn, "blocks_fts_docsize")
    triggers_present = not _missing_named_triggers_sync(conn, _ARCHIVE_BLOCKS_FTS_TRIGGERS)
    indexed_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM blocks_fts_docsize") if docsize_exists else 0
    if fts_exists and (not triggers_present or indexed_rows != source_rows):
        logger.warning("daemon: archive blocks FTS drift detected on startup. Rebuilding once.")
        conn.execute("INSERT INTO blocks_fts(blocks_fts) VALUES('rebuild')")
        indexed_rows = _count_or_zero(conn, "SELECT COUNT(*) FROM blocks_fts_docsize")
        triggers_present = not _missing_named_triggers_sync(conn, _ARCHIVE_BLOCKS_FTS_TRIGGERS)

    ready = fts_exists and triggers_present and indexed_rows == source_rows
    record_fts_surface_state_sync(
        conn,
        surface="blocks_fts",
        state=READY if ready else STALE,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=max(source_rows - indexed_rows, 0),
        excess_rows=max(indexed_rows - source_rows, 0),
        detail=None if ready else "archive startup FTS readiness failed",
    )
    return True


async def ensure_fts_startup_readiness() -> None:
    """Run daemon startup FTS maintenance without blocking the event loop."""
    await asyncio.to_thread(ensure_fts_startup_readiness_sync)


def ensure_fts_startup_readiness_sync() -> None:
    """Ensure FTS triggers and index are healthy on daemon startup.

    Three failure modes are recovered here:

    1. FTS table missing entirely (historical rows pre-date FTS) →
       rebuild from messages/action_events.
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
    from polylogue.storage.sqlite.connection_profile import open_connection

    db = _active_fts_startup_db_path()
    if not db.exists():
        return

    conn: sqlite3.Connection | None = None
    try:
        conn = open_connection(db, timeout=10.0)
        if _ensure_archive_blocks_fts_startup_readiness_sync(conn):
            conn.commit()
            return
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        has_fts_table = row is not None
        # Fresh-init race guard (#1603): bootstrap may have committed
        # ``messages_fts`` but not yet ``messages``; bailing keeps the
        # operator log clean instead of panicking in repair_stale_fts_rows.
        messages_row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'").fetchone()
        if messages_row is None:
            logger.info("daemon: FTS startup check skipped — fresh-init in flight.")
            return

        from polylogue.storage.fts.dangling_repair import (
            configure_bounded_repair_connection,
            repair_missing_fts_rows,
            repair_stale_fts_rows,
        )
        from polylogue.storage.fts.freshness import (
            ensure_fts_freshness_table_sync,
        )
        from polylogue.storage.fts.fts_lifecycle import (
            ensure_fts_index_sync,
            rebuild_fts_index_sync,
            restore_fts_triggers_sync,
        )

        configure_bounded_repair_connection(conn)
        ensure_fts_freshness_table_sync(conn)

        if not has_fts_table:
            logger.warning("daemon: message FTS table missing. Rebuilding once.")
            rebuild_fts_index_sync(conn)
            conn.commit()
            logger.info("daemon: FTS rebuild complete.")
            return

        missing_triggers = missing_fts_triggers_sync(conn)
        if missing_triggers:
            logger.warning(
                "daemon: FTS triggers missing on startup (%s). "
                "SIGKILL-during-bulk-suspend signature; rebuilding before serving search.",
                ", ".join(missing_triggers),
            )
            restore_fts_triggers_sync(conn)
            outcome = repair_missing_fts_rows(conn)
            if not outcome.success:
                logger.warning("daemon: bounded FTS repair failed after trigger restore: %s", outcome.detail)
                rebuild_fts_index_sync(conn)
            record_fts_freshness_snapshot_sync(conn)
            conn.commit()
            logger.info("daemon: FTS trigger recovery complete.")
            return

        ensure_fts_index_sync(conn)
        outcome = repair_stale_fts_rows(conn)
        if not outcome.success:
            logger.warning("daemon: bounded FTS startup repair failed: %s. Rebuilding.", outcome.detail)
            restore_fts_triggers_sync(conn)
            rebuild_fts_index_sync(conn)
            conn.commit()
            logger.info("daemon: FTS rebuild complete.")
            return

        record_fts_freshness_snapshot_sync(conn)
        conn.commit()
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
