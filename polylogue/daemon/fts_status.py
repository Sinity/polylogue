"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.storage.fts.fts_lifecycle import FtsInvariantSnapshot, FtsSurfaceInvariant, fts_invariant_snapshot_sync
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_FTS_SURFACES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "messages_fts",
        "messages",
        "messages_fts",
        ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"),
    ),
    (
        "action_events_fts",
        "action_events",
        "action_events_fts",
        ("action_events_fts_ai", "action_events_fts_ad", "action_events_fts_au"),
    ),
    (
        "session_work_events_fts",
        "session_work_events",
        "session_work_events_fts",
        ("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au"),
    ),
    (
        "work_threads_fts",
        "work_threads",
        "work_threads_fts",
        ("work_threads_fts_ai", "work_threads_fts_ad", "work_threads_fts_au"),
    ),
)


class FTSReadiness(BaseModel):
    messages_ready: bool = False
    action_events_ready: bool = False
    session_work_events_ready: bool = False
    work_threads_ready: bool = False
    invariant_ready: bool = False
    message_indexed_count: int = 0
    message_indexable_count: int = 0
    action_event_indexed_count: int = 0
    action_event_count: int = 0
    coverage_pct: float = 0.0
    surfaces: dict[str, dict[str, int | bool]] = Field(default_factory=dict)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _triggers_present(conn: sqlite3.Connection, trigger_names: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in trigger_names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        trigger_names,
    ).fetchall()
    present = {row[0] for row in rows}
    return all(name in present for name in trigger_names)


def _surface_payload(surface: FtsSurfaceInvariant) -> dict[str, int | bool]:
    return {
        "source_exists": surface.source_exists,
        "exists": surface.exists,
        "source_rows": surface.source_rows,
        "indexed_rows": surface.indexed_rows,
        "triggers_present": surface.triggers_present,
        "missing_rows": surface.missing_rows,
        "excess_rows": surface.excess_rows,
        "duplicate_rows": surface.duplicate_rows,
        "ready": surface.ready,
        "exact": True,
    }


def _exact_readiness_payload(snapshot: FtsInvariantSnapshot) -> dict[str, object]:
    surfaces = {surface.name: _surface_payload(surface) for surface in snapshot.surfaces}
    messages = snapshot.messages
    action_events = snapshot.action_events
    coverage_pct = round((messages.indexed_rows / messages.source_rows) * 100, 1) if messages.source_rows > 0 else 100.0
    return {
        "messages_ready": messages.ready,
        "action_events_ready": action_events.ready,
        "session_work_events_ready": snapshot.session_work_events.ready,
        "work_threads_ready": snapshot.work_threads.ready,
        "invariant_ready": snapshot.ready,
        "message_indexed_count": messages.indexed_rows,
        "message_indexable_count": messages.source_rows,
        "action_event_indexed_count": action_events.indexed_rows,
        "action_event_count": action_events.source_rows,
        "coverage_pct": coverage_pct,
        "coverage_exact": True,
        "surfaces": surfaces,
    }


def fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Return FTS readiness for health/status probes.

    The default is request-safe and structural: it proves tables/triggers
    exist without scanning FTS shadow tables. Use ``exact=True`` for hard
    readiness/search decisions where stale search must never be served.
    """
    if not dbf.exists():
        return {"messages_ready": False, "action_events_ready": False, "coverage_pct": 0.0}
    try:
        conn = open_readonly_connection(dbf)
        try:
            if exact:
                return _exact_readiness_payload(fts_invariant_snapshot_sync(conn))
            surfaces: dict[str, dict[str, int | bool]] = {}
            for name, source_table, fts_table, triggers in _FTS_SURFACES:
                source_exists = _table_exists(conn, source_table)
                exists = _table_exists(conn, fts_table)
                triggers_present = exists and _triggers_present(conn, triggers)
                ready = (exists and triggers_present) if source_exists else not exists
                surfaces[name] = {
                    "source_exists": source_exists,
                    "exists": exists,
                    "source_rows": 0,
                    "indexed_rows": 0,
                    "triggers_present": triggers_present,
                    "missing_rows": 0,
                    "excess_rows": 0,
                    "duplicate_rows": 0,
                    "ready": ready,
                    "exact": False,
                }
        finally:
            conn.close()
    except sqlite3.Error:
        return {
            "messages_ready": False,
            "action_events_ready": False,
            "session_work_events_ready": False,
            "work_threads_ready": False,
            "invariant_ready": False,
            "coverage_pct": 0.0,
            "surfaces": {},
        }

    messages = surfaces["messages_fts"]
    action_events = surfaces["action_events_fts"]
    session_work_events = surfaces["session_work_events_fts"]
    work_threads = surfaces["work_threads_fts"]
    invariant_ready = all(bool(surface["ready"]) for surface in surfaces.values())
    return {
        "messages_ready": messages["ready"],
        "action_events_ready": action_events["ready"],
        "session_work_events_ready": session_work_events["ready"],
        "work_threads_ready": work_threads["ready"],
        "invariant_ready": invariant_ready,
        "message_indexed_count": 0,
        "message_indexable_count": 0,
        "action_event_indexed_count": 0,
        "action_event_count": 0,
        "coverage_pct": 100.0 if invariant_ready else 0.0,
        "coverage_exact": False,
        "surfaces": surfaces,
    }
