"""Read projection for receiver-authoritative capture-job events."""

from __future__ import annotations

import json
import sqlite3


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
