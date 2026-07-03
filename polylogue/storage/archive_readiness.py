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
        "unchecked",
        "affected_unchecked",
    )
    return all(_read_int(readiness, key) == 0 for key in blocking_keys)


def raw_materialization_readiness_snapshot(active_archive: Path) -> dict[str, object]:
    """Return compact raw→index materialization readiness for an archive root.

    This function is used by status/readiness surfaces and must stay cheap on
    large archives. It deliberately reports the indexed raw-id join gap without
    opening raw blob payloads to classify sidecars or aliases; the exact
    classifier lives in ``polylogue ops debt list --kind raw-materialization``.
    """
    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {"available": False, "error": "source.db or index.db missing"}
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
            row = conn.execute(
                """
                WITH raw_rows AS (
                    SELECT
                        r.raw_id,
                        r.origin,
                        r.validation_status,
                        r.parse_error,
                        r.parsed_at_ms,
                        EXISTS (
                            SELECT 1
                            FROM main.sessions s
                            WHERE s.raw_id = r.raw_id
                        ) AS is_materialized
                    FROM source.raw_sessions r
                    WHERE COALESCE(r.validation_status, '') != 'skipped'
                ),
                materialization AS (
                    SELECT
                        COUNT(*) AS raw_artifact_count,
                        COALESCE(SUM(CASE WHEN is_materialized THEN 1 ELSE 0 END), 0)
                            AS materialized_raw_artifact_count
                    FROM raw_rows
                ),
                session_count AS (
                    SELECT COUNT(*) AS archive_session_count FROM main.sessions
                ),
                gaps AS (
                    SELECT raw_id, origin, validation_status, parse_error, parsed_at_ms
                    FROM raw_rows
                    WHERE NOT is_materialized
                )
                SELECT
                    materialization.raw_artifact_count,
                    materialization.materialized_raw_artifact_count,
                    session_count.archive_session_count,
                    (materialization.raw_artifact_count - materialization.materialized_raw_artifact_count)
                        AS join_gap_count,
                    COUNT(gaps.raw_id) AS total,
                    COALESCE(SUM(CASE WHEN validation_status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped,
                    COALESCE(SUM(CASE WHEN parse_error IS NOT NULL THEN 1 ELSE 0 END), 0) AS parse_failed,
                    COALESCE(SUM(CASE WHEN parsed_at_ms IS NOT NULL AND parse_error IS NULL THEN 1 ELSE 0 END), 0)
                        AS parsed_without_index_session
                FROM gaps
                CROSS JOIN materialization
                CROSS JOIN session_count
                """
            ).fetchone()
            family_rows = conn.execute(
                """
                SELECT r.origin, COUNT(*) AS count
                FROM source.raw_sessions r
                LEFT JOIN main.sessions s ON s.raw_id = r.raw_id
                WHERE s.raw_id IS NULL
                  AND COALESCE(r.validation_status, '') != 'skipped'
                GROUP BY r.origin
                ORDER BY count DESC, r.origin
                LIMIT 16
                """
            ).fetchall()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    total = int(row["total"] or 0)
    raw_artifact_count = int(row["raw_artifact_count"] or 0)
    materialized_raw_artifact_count = int(row["materialized_raw_artifact_count"] or 0)
    archive_session_count = int(row["archive_session_count"] or 0)
    join_gap_count = int(row["join_gap_count"] or total)
    skipped = int(row["skipped"] or 0)
    parse_failed = int(row["parse_failed"] or 0)
    parsed_without_index_session = int(row["parsed_without_index_session"] or 0)
    return {
        "available": True,
        "classification": "not_run",
        "precision": "raw_id_join_gap",
        "raw_artifact_count": raw_artifact_count,
        "materialized_raw_artifact_count": materialized_raw_artifact_count,
        "archive_session_count": archive_session_count,
        "join_gap_count": join_gap_count,
        "total": total,
        "critical": 0,
        "warning": 0,
        "actionable": 0,
        "blocked": 0,
        "classified": 0,
        "unchecked": total,
        "affected_total": total,
        "affected_actionable": 0,
        "affected_blocked": 0,
        "affected_open": 0,
        "affected_classified": 0,
        "affected_unchecked": total,
        "category_counts": {
            "raw_id_join_gap": total,
            "skipped": skipped,
            "parse_failed": parse_failed,
            "parsed_without_index_session": parsed_without_index_session,
        },
        "source_family_counts": {str(item["origin"]): int(item["count"] or 0) for item in family_rows},
    }


__all__ = [
    "ACTIVE_REBUILD_STALE_AFTER_S",
    "active_rebuild_index_attempts",
    "raw_materialization_readiness_snapshot",
    "raw_materialization_ready",
]
