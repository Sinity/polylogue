"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


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


def fts_readiness_info(dbf: Path) -> dict[str, object]:
    """Check FTS table presence and source-count parity through bounded probes."""
    if not dbf.exists():
        return {"messages_ready": False, "action_events_ready": False, "coverage_pct": 0.0}
    try:
        conn = open_readonly_connection(dbf)
        try:
            from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

            snapshot = fts_invariant_snapshot_sync(conn)
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

    message_indexable_count = snapshot.messages.source_rows
    message_indexed_count = snapshot.messages.indexed_rows
    action_event_count = snapshot.action_events.source_rows
    action_event_indexed_count = snapshot.action_events.indexed_rows
    coverage_pct = 100.0 if message_indexable_count == 0 else 100 * message_indexed_count / message_indexable_count
    return {
        "messages_ready": snapshot.messages.ready,
        "action_events_ready": snapshot.action_events.ready,
        "session_work_events_ready": snapshot.session_work_events.ready,
        "work_threads_ready": snapshot.work_threads.ready,
        "invariant_ready": snapshot.ready,
        "message_indexed_count": message_indexed_count,
        "message_indexable_count": message_indexable_count,
        "action_event_indexed_count": action_event_indexed_count,
        "action_event_count": action_event_count,
        "coverage_pct": round(max(0.0, coverage_pct), 1),
        "surfaces": {
            surface.name: {
                "source_exists": surface.source_exists,
                "exists": surface.exists,
                "source_rows": surface.source_rows,
                "indexed_rows": surface.indexed_rows,
                "triggers_present": surface.triggers_present,
                "missing_rows": surface.missing_rows,
                "excess_rows": surface.excess_rows,
                "duplicate_rows": surface.duplicate_rows,
                "ready": surface.ready,
            }
            for surface in snapshot.surfaces
        },
    }
