"""Legacy-monolith read path for embedding catch-up runs.

The split-file archive's only ``embedding_catchup_runs`` writer is
``upsert_embedding_catchup_run`` in ``storage/sqlite/archive_tiers/ops_write.py``
against the ops-tier DDL in ``archive_tiers/ops.py`` (statuses ``running`` /
``completed`` / ``failed`` / ``cancelled``). This module used to define a
second, same-named table with its own shape and status vocabulary
(``stopped`` / ``interrupted``); that writer half is deleted. What remains is
the read-only accessor for pre-split single-file archives, whose legacy table
shape (``started_at`` TEXT, ``stop_reason``, ``rebuild``, planned/limit
columns) still surfaces through the embedding-status fallback in
``storage/embeddings/status_payload.py`` and ``daemon/metrics.py``.
"""

from __future__ import annotations

import sqlite3
from typing import TypedDict


class EmbeddingCatchupRunPayload(TypedDict):
    run_id: str
    started_at: str
    updated_at: str
    completed_at: str | None
    status: str
    stop_reason: str | None
    rebuild: bool
    max_sessions: int | None
    max_messages: int | None
    stop_after_seconds: int | None
    max_errors: int | None
    planned_sessions: int
    planned_messages: int
    processed_sessions: int
    embedded_sessions: int
    skipped_sessions: int
    error_count: int
    embedded_messages: int
    estimated_cost_usd: float
    last_session_id: str | None


def latest_embedding_catchup_run(conn: sqlite3.Connection) -> EmbeddingCatchupRunPayload | None:
    """Return the most recent catch-up run payload, if any."""

    row = conn.execute(
        """
        SELECT run_id, started_at, updated_at, completed_at, status, stop_reason,
               rebuild, max_sessions, max_messages, stop_after_seconds, max_errors,
               planned_sessions, planned_messages, processed_sessions,
               embedded_sessions, skipped_sessions, error_count,
               embedded_messages, estimated_cost_usd, last_session_id
        FROM embedding_catchup_runs
        ORDER BY started_at DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": str(row[0]),
        "started_at": str(row[1]),
        "updated_at": str(row[2]),
        "completed_at": None if row[3] is None else str(row[3]),
        "status": str(row[4]),
        "stop_reason": None if row[5] is None else str(row[5]),
        "rebuild": bool(row[6]),
        "max_sessions": row[7],
        "max_messages": row[8],
        "stop_after_seconds": row[9],
        "max_errors": row[10],
        "planned_sessions": int(row[11] or 0),
        "planned_messages": int(row[12] or 0),
        "processed_sessions": int(row[13] or 0),
        "embedded_sessions": int(row[14] or 0),
        "skipped_sessions": int(row[15] or 0),
        "error_count": int(row[16] or 0),
        "embedded_messages": int(row[17] or 0),
        "estimated_cost_usd": float(row[18] or 0.0),
        "last_session_id": None if row[19] is None else str(row[19]),
    }


__all__ = [
    "EmbeddingCatchupRunPayload",
    "latest_embedding_catchup_run",
]
