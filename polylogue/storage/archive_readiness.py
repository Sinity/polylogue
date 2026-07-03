"""Shared archive readiness helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def active_rebuild_index_attempts(ops_db: Path) -> list[dict[str, object]]:
    """Return active index-rebuild attempts recorded in the ops tier."""
    if not ops_db.exists():
        return []
    try:
        with sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT attempt_id, phase, started_at_ms, heartbeat_at_ms, parsed_raw_count, materialized_count
                FROM ingest_attempts
                WHERE status = 'running'
                  AND phase = 'rebuild-index'
                ORDER BY started_at_ms DESC
                LIMIT 8
                """
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "attempt_id": str(row["attempt_id"]),
            "phase": str(row["phase"]),
            "started_at_ms": int(row["started_at_ms"]),
            "heartbeat_at_ms": int(row["heartbeat_at_ms"]) if row["heartbeat_at_ms"] is not None else None,
            "parsed_raw_count": int(row["parsed_raw_count"] or 0),
            "materialized_count": int(row["materialized_count"] or 0),
        }
        for row in rows
    ]


def _read_int(readiness: Mapping[str, Any], key: str) -> int:
    try:
        return int(readiness.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def raw_materialization_ready(readiness: Mapping[str, Any] | object | None) -> bool:
    """Return whether raw acquisition and index materialization are converged.

    Classified alias/non-session join gaps are acceptable: they mean the raw
    row has been explained. Actionable/open/blocking debt is not acceptable for
    product archive readiness.
    """
    if readiness is None:
        return False
    if not isinstance(readiness, Mapping):
        model_dump = getattr(readiness, "model_dump", None)
        if callable(model_dump):
            readiness = model_dump()
        else:
            return False
    if not bool(readiness.get("available", False)):
        return False
    blocking_keys = (
        "critical",
        "warning",
        "actionable",
        "blocked",
        "affected_actionable",
        "affected_blocked",
        "affected_open",
    )
    return all(_read_int(readiness, key) == 0 for key in blocking_keys)


__all__ = ["active_rebuild_index_attempts", "raw_materialization_ready"]
