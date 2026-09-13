"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.logging import get_logger
from polylogue.operations.fts_derivation import archive_fts_surface
from polylogue.operations.fts_derivation import fts_triggers_present as _triggers_present
from polylogue.storage.fts.fts_lifecycle import FtsInvariantSnapshot, FtsSurfaceInvariant, fts_invariant_snapshot_sync
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)


class FTSReadiness(BaseModel):
    indexed_surface: str = "messages_fts"
    messages_ready: bool = False
    session_work_events_ready: bool = False
    invariant_ready: bool = False
    message_indexed_count: int | None = 0
    message_indexable_count: int | None = 0
    coverage_pct: float | None = 0.0
    coverage_exact: bool = True
    surfaces: dict[str, dict[str, int | bool | str | None]] = Field(default_factory=dict)


def _surface_payload(surface: FtsSurfaceInvariant) -> dict[str, int | bool | str | None]:
    return {
        "source_exists": surface.source_exists,
        "exists": surface.exists,
        "source_rows": surface.source_rows,
        "indexed_rows": surface.indexed_rows,
        "triggers_present": surface.triggers_present,
        "missing_rows": surface.missing_rows,
        "excess_rows": surface.excess_rows,
        "duplicate_rows": surface.duplicate_rows,
        "identity_mismatch_rows": surface.identity_mismatch_rows,
        "ready": surface.ready,
        "exact": True,
    }


def _payload_int(surface: dict[str, int | bool | str | None], key: str) -> int:
    return _row_int(surface.get(key))


def _archive_index_path_for(dbf: Path) -> Path | None:
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(dbf.parent).active_index_path
    return index_db if index_db.exists() else None


def _archive_blocks_surface(conn: sqlite3.Connection) -> dict[str, int | bool | str | None]:
    """Read authoritative FTS membership; freshness rows never certify it."""
    return archive_fts_surface(conn)


def _archive_readiness_payload(conn: sqlite3.Connection, *, exact: bool) -> dict[str, object] | None:
    if not _table_exists(conn, "blocks") and not _table_exists(conn, "messages_fts"):
        return None
    del exact
    blocks = _archive_blocks_surface(conn)
    effective_exact = True
    block_source_rows = _payload_int(blocks, "source_rows")
    block_indexed_rows = _payload_int(blocks, "indexed_rows")
    invariant_ready = bool(blocks["ready"])
    event_source_exists = _table_exists(conn, "session_work_events")
    event_exists = _table_exists(conn, "session_work_events_fts")
    event_docsize_exists = _table_exists(conn, "session_work_events_fts_docsize")
    event_triggers_present = event_exists and _triggers_present(
        conn, ("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au")
    )
    if not event_source_exists:
        event_ready = not event_exists
    elif not event_exists or not event_docsize_exists:
        event_ready = False
    else:
        event_source_rows = int(conn.execute("SELECT COUNT(*) FROM session_work_events").fetchone()[0] or 0)
        event_indexed_rows = int(
            conn.execute("SELECT COUNT(*) FROM session_work_events_fts_docsize").fetchone()[0] or 0
        )
        event_ready = event_triggers_present and event_source_rows == event_indexed_rows
    return {
        "indexed_surface": "messages_fts",
        "messages_ready": invariant_ready,
        "session_work_events_ready": event_ready,
        "invariant_ready": invariant_ready and event_ready,
        "message_indexed_count": block_indexed_rows,
        "message_indexable_count": block_source_rows,
        # source_rows == 0 means there is nothing to index -- a genuine,
        # vacuously-true "fully covered" state. invariant_ready alone is
        # NOT evidence of coverage: it only proves triggers/tables exist,
        # so reporting 100.0 whenever it happens to be true (regardless of
        # source_rows) fabricates a measured percentage for an unmeasured
        # branch (polylogue-oitx). Report None ("not applicable / unknown")
        # instead; consumers already render that explicitly (polylogue-roax,
        # cli/commands/status.py's null-coverage handling).
        "coverage_pct": (round((block_indexed_rows / block_source_rows) * 100, 1) if block_source_rows > 0 else None),
        "coverage_exact": effective_exact,
        "surfaces": {"messages_fts": blocks},
    }


def _archive_readiness_info(index_db: Path, *, exact: bool) -> dict[str, object] | None:
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            return _archive_readiness_payload(conn, exact=exact)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("fts archive readiness query failed for %s: %s", index_db, exc, exc_info=True)
        return {
            "indexed_surface": "messages_fts",
            "messages_ready": False,
            "session_work_events_ready": True,
            "invariant_ready": False,
            "coverage_pct": 0.0,
            "surfaces": {},
        }


def _exact_readiness_payload(snapshot: FtsInvariantSnapshot) -> dict[str, object]:
    surfaces = {surface.name: _surface_payload(surface) for surface in snapshot.surfaces}
    messages = snapshot.messages
    # source_rows == 0 is a zero-denominator case -- report unmeasured
    # (None) rather than a fabricated 100.0 (polylogue-oitx).
    coverage_pct = round((messages.indexed_rows / messages.source_rows) * 100, 1) if messages.source_rows > 0 else None
    return {
        "messages_ready": messages.ready,
        "session_work_events_ready": snapshot.session_work_events.ready,
        "invariant_ready": snapshot.ready,
        "message_indexed_count": messages.indexed_rows,
        "message_indexable_count": messages.source_rows,
        "coverage_pct": coverage_pct,
        "coverage_exact": True,
        "surfaces": surfaces,
    }


def fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Return FTS readiness for health/status probes.

    Readiness is always computed from the current output relation and its
    canonical inputs.  ``exact`` is retained as a call-compatible parameter;
    no freshness/debt observation can make the result cheaper or certify it.
    """
    if not dbf.exists():
        archive_index = _archive_index_path_for(dbf)
        if archive_index is not None:
            archive_info = _archive_readiness_info(archive_index, exact=exact)
            if archive_info is not None:
                return archive_info
        return {
            "messages_ready": False,
            "coverage_pct": 0.0,
        }
    try:
        # Readiness reports a skewed or unstamped tier as not-ready data; it must
        # not raise the status surface out of service.
        conn = open_readonly_connection(dbf, validate_schema=False)
        try:
            conn.execute("BEGIN")
            archive_info = _archive_readiness_payload(conn, exact=True)
            if archive_info is not None:
                return archive_info
            return _exact_readiness_payload(fts_invariant_snapshot_sync(conn))
        finally:
            conn.rollback()
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("fts readiness query failed for %s: %s", dbf, exc, exc_info=True)
        return {
            "messages_ready": False,
            "session_work_events_ready": False,
            "invariant_ready": False,
            "coverage_pct": 0.0,
            "surfaces": {},
        }
