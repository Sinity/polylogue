"""Periodic archive-wide FTS orphan audit (polylogue-5vbs).

``messages_fts`` is a contentless FTS5 index over ``blocks.search_text``, kept
in sync by per-row triggers plus an in-transaction repair
(``repair_message_fts_index_sync``) that session/message writes run as a
``WriteEffect`` with ``failure_policy=abort`` -- a repair failure on that path
poisons the whole write rather than landing silently. That guarantee only
covers the in-transaction write path itself. It says nothing about FTS drift
introduced OUTSIDE it: external DB surgery, a partial migration, or a future
writer that skips the ``write_effects.py`` registry.

Nothing else ever notices that class of drift today. ``make_fts_stage``'s
path-scoped ``check``/``check_many`` (``daemon/convergence_stages.py``) are
deliberate no-ops in archive/split-file mode for exactly the reason above --
but the session-scoped ``check_sessions``/``execute_sessions`` retry
primitives on the same stage are real, working implementations that simply
never receive a ``stage="fts", subject_type="session_id"`` subject for drift
introduced outside the write path (two live-ingest call sites in
``sources/live/batch.py`` do feed that subject for their own deferred-FTS
writes, but only for those two writes, not for the archive at large).

This module is that missing feeder: a bounded, quiet-cadence sweep that finds
sessions with blocks missing their ``messages_fts`` row and records
convergence debt for them, so the existing (already correct) session-scoped
retry path picks the repair up on its next drain.

Deliberately NOT a ``DaemonConverger``/``ConvergenceStage`` (see
``docs/retro/2026-05-24-1498-cascade.md``: ``convergence_stages.py`` is
already large and its own verdict is "refactor before adding a fourth
stage"). Instead this follows the ``periodic_fts_identity_drift_recompute``
(polylogue-miwv) shape: a plain ``asyncio`` loop scheduled directly in
``daemon/cli.py`` alongside the other maintenance loops, self-contained and
independently testable.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

logger = get_logger(__name__)

#: Distinct sessions inspected/fed per sweep. A large archive is covered
#: incrementally across repeated sweeps rather than one unbounded scan.
FTS_ORPHAN_AUDIT_SESSION_LIMIT = 200

#: Quiet-cadence interval, matching ``periodic_fts_identity_drift_recompute``'s
#: reasoning: this is a defense-in-depth sweep for drift the write path
#: itself should already prevent, not a hot-path check.
FTS_ORPHAN_AUDIT_INTERVAL_SECONDS = 3_600  # 1 hour


@dataclass(frozen=True, slots=True)
class FtsOrphanAuditResult:
    """Bounded, secret-safe summary of one sweep for daemon logging."""

    ran: bool = False
    orphaned_sessions_found: int = 0


def find_orphaned_fts_sessions_sync(
    db_path: Path,
    *,
    limit: int = FTS_ORPHAN_AUDIT_SESSION_LIMIT,
) -> list[str]:
    """Return up to ``limit`` distinct session ids with an FTS-orphaned block.

    A block is "FTS-orphaned" when ``blocks.search_text`` is non-empty (it is
    supposed to be indexed) but no corresponding row exists in
    ``messages_fts_docsize`` (the contentless FTS5 shadow table proving a row
    was actually written). This is the same anti-join
    ``_archive_fts_needs_repair``/``repair_message_fts_index_sync`` already
    use for session-scoped work, applied archive-wide via the existing
    ``idx_blocks_search_text_populated`` partial index instead of only for a
    caller-supplied session set.

    ``db_path`` names ``index.db`` directly (single-file legacy archives) or
    an archive root's ``index.db`` (split-file archives resolve through
    ``ArchiveLocation``, mirroring every other convergence probe). A missing
    or unreadable archive is a bounded empty result, matching every other
    periodic daemon maintenance probe's tolerance for an archive that is not
    ready yet.
    """
    from polylogue.storage.archive_identity import ArchiveLocation
    from polylogue.storage.introspection import table_exists

    resolved = ArchiveLocation.resolve(db_path.parent).active_index_path
    target = resolved if resolved.exists() else (db_path if db_path.exists() else None)
    if target is None:
        return []
    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5.0)
        try:
            if not table_exists(conn, "blocks") or not table_exists(conn, "messages_fts_docsize"):
                return []
            rows = conn.execute(
                """
                SELECT DISTINCT b.session_id
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                WHERE b.search_text != '' AND d.id IS NULL
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [str(row[0]) for row in rows]
        finally:
            conn.close()
    except sqlite3.Error:
        logger.warning("fts_orphan_audit: probe failed", exc_info=True)
        return []


def run_fts_orphan_audit_once_sync(db_path: Path) -> FtsOrphanAuditResult:
    """Find FTS-orphaned sessions and record retryable convergence debt for them.

    Feeds ``stage="fts", subject_type="session_id"`` convergence debt through
    :class:`~polylogue.sources.live.cursor.CursorStore` -- the exact subject
    shape ``daemon/cli.py``'s convergence-debt drain loop already dispatches
    to ``make_fts_stage``'s ``check_sessions``/``execute_sessions``, which
    call the real, already-tested ``repair_message_fts_index_sync`` repair.
    This function only ever *records* debt; it never repairs directly, so a
    single sweep can never race the sole-writer archive connection held by
    whatever else is mid-write when this runs.
    """
    if not db_path.exists():
        return FtsOrphanAuditResult()

    from polylogue.sources.live.cursor import CursorStore

    session_ids = find_orphaned_fts_sessions_sync(db_path)
    cursor = CursorStore(db_path)
    for session_id in session_ids:
        cursor.record_convergence_debt(
            stage="fts",
            subject_type="session_id",
            subject_id=session_id,
            error="periodic archive-wide FTS orphan audit found drift outside the write path",
        )
    return FtsOrphanAuditResult(ran=True, orphaned_sessions_found=len(session_ids))


async def periodic_fts_orphan_audit(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically sweep the archive for FTS orphans and feed convergence debt.

    Gated on ``catch_up_complete`` (when given) so the first pass never races
    initial source catch-up's own FTS churn -- same gating shape as
    ``periodic_fts_identity_drift_recompute``/``periodic_judgment_automation_sweep``.
    Runs on ``daemon_write_coordinator`` like every other periodic write, so
    it serializes with live ingest instead of racing it for the SQLite writer
    lock; the actual repair happens later, off this loop, via the ordinary
    convergence-debt drain.
    """
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import archive_root

    await _await_catch_up_gate(catch_up_complete, loop_name="fts orphan audit")
    while True:
        await asyncio.sleep(FTS_ORPHAN_AUDIT_INTERVAL_SECONDS)
        db_path = archive_root() / "index.db"
        try:
            result = await daemon_write_coordinator().run_sync(
                "maintenance.fts_orphan_audit",
                run_fts_orphan_audit_once_sync,
                db_path,
            )
            if result.ran and result.orphaned_sessions_found:
                logger.info(
                    "fts_orphan_audit: found orphaned_sessions=%d; recorded as convergence debt",
                    result.orphaned_sessions_found,
                )
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("fts_orphan_audit: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("fts_orphan_audit: sweep failed", exc_info=True)
        except Exception:
            logger.warning("fts_orphan_audit: sweep failed", exc_info=True)


__all__ = [
    "FTS_ORPHAN_AUDIT_INTERVAL_SECONDS",
    "FTS_ORPHAN_AUDIT_SESSION_LIMIT",
    "FtsOrphanAuditResult",
    "find_orphaned_fts_sessions_sync",
    "periodic_fts_orphan_audit",
    "run_fts_orphan_audit_once_sync",
]
