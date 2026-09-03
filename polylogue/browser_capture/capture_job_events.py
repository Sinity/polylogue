"""Read projection for receiver-authoritative capture-job events."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict


def read_capture_job_retention(connection: sqlite3.Connection, job_id: str) -> dict[str, object] | None:
    """Read lifecycle fields for receiver read-surface consumers."""
    row = connection.execute("SELECT retention_json FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    return {"retention": json.loads(row[0])}


def read_capture_job_events(connection: sqlite3.Connection, job_id: str, limit: int) -> list[dict[str, object]]:
    """Read a bounded, receiver-ordered event page from the registry."""
    rows = connection.execute(
        "SELECT event_id, job_id, event_revision, job_revision, kind, refs_json, payload_json, request_id, occurred_at "
        "FROM capture_job_events WHERE job_id=? ORDER BY event_revision LIMIT ?",
        (job_id, limit + 1),
    ).fetchall()
    events: list[dict[str, object]] = []
    for row in rows[:limit]:
        payload = json.loads(row["payload_json"])
        events.append(
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "event_revision": row["event_revision"],
                "job_revision": row["job_revision"],
                "kind": row["kind"],
                "refs": json.loads(row["refs_json"]),
                "payload": payload.get("value", payload),
                "request_id": row["request_id"],
                "occurred_at": row["occurred_at"],
            }
        )
    return events


def project_capture_job_timelines(events: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Project receiver events into reverse-chronological conversation timelines."""
    timelines: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        refs = event.get("refs")
        if isinstance(refs, dict) and isinstance(refs.get("conversation_ref"), str) and refs["conversation_ref"]:
            timelines[refs["conversation_ref"]].append(event)
    return {
        key: sorted(
            value,
            key=lambda item: revision if isinstance((revision := item.get("event_revision")), int) else -1,
            reverse=True,
        )
        for key, value in timelines.items()
    }
