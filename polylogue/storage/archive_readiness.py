"""Shared archive readiness helpers."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ACTIVE_REBUILD_STALE_AFTER_S = 180.0
"""Maximum heartbeat/start age for a rebuild-index row to count as active."""


def active_rebuild_index_attempts(ops_db: Path) -> list[dict[str, object]]:
    """Return active index-rebuild attempts recorded in the ops tier."""
    if not ops_db.exists():
        return []
    cutoff_ms = int((time.time() - ACTIVE_REBUILD_STALE_AFTER_S) * 1000)
    try:
        with sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT attempt_id, phase, started_at_ms, heartbeat_at_ms, parsed_raw_count, materialized_count
                FROM ingest_attempts
                WHERE status = 'running'
                  AND phase = 'rebuild-index'
                  AND COALESCE(heartbeat_at_ms, started_at_ms) >= ?
                ORDER BY started_at_ms DESC
                LIMIT 8
                """,
                (cutoff_ms,),
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
            dumped = model_dump()
            if not isinstance(dumped, Mapping):
                return False
            readiness = dumped
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


def raw_materialization_readiness_snapshot(active_archive: Path) -> dict[str, object]:
    """Return compact raw→index materialization readiness for an archive root."""
    try:
        from polylogue.operations.archive_debt import archive_debt_list

        payload = archive_debt_list(
            archive_root=active_archive,
            kinds=("raw-materialization",),
            limit=None,
            exact_fts=False,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    rows = [row for row in payload.rows if row.kind == "raw-materialization"]
    affected_actionable = 0
    affected_blocked = 0
    affected_open = 0
    affected_classified = 0
    for row in rows:
        affected = max(1, int(row.affected_count or 1))
        if row.status == "actionable":
            affected_actionable += affected
        elif row.status == "blocked":
            affected_blocked += affected
        elif row.status == "open":
            affected_open += affected
        elif row.status == "classified":
            affected_classified += affected

    return {
        "available": True,
        "total": len(rows),
        "critical": sum(1 for row in rows if row.severity == "critical"),
        "warning": sum(1 for row in rows if row.severity == "warning"),
        "actionable": sum(1 for row in rows if row.status == "actionable"),
        "blocked": sum(1 for row in rows if row.status == "blocked"),
        "classified": sum(1 for row in rows if row.status == "classified"),
        "affected_total": int(payload.totals.affected_total),
        "affected_actionable": affected_actionable,
        "affected_blocked": affected_blocked,
        "affected_open": affected_open,
        "affected_classified": affected_classified,
    }


__all__ = [
    "ACTIVE_REBUILD_STALE_AFTER_S",
    "active_rebuild_index_attempts",
    "raw_materialization_readiness_snapshot",
    "raw_materialization_ready",
]
