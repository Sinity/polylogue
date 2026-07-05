"""Shared archive readiness helpers."""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from polylogue.archive.raw_materialization import (
    parsed_non_session_artifact_reason,
    source_path_native_id_candidates,
)

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
        "lost_source_evidence_count",
        "unchecked",
        "affected_unchecked",
    )
    return all(_read_int(readiness, key) == 0 for key in blocking_keys)


def raw_materialization_readiness_snapshot(active_archive: Path) -> dict[str, object]:
    """Return compact raw→index materialization readiness for an archive root.

    This function is used by status/readiness surfaces and must stay cheap on
    large archives. It classifies only cheap structural explanations for
    raw-id join gaps: rows already materialized by provider/source aliases and
    parsed sidecar/metadata artifacts. Remaining gaps are daemon convergence
    input, not an operator maintenance workflow.
    """
    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {"available": False, "error": "source.db or index.db missing"}
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
            raw_columns = _table_columns(conn, "source", "raw_sessions")
            session_columns = _table_columns(conn, "main", "sessions")
            raw_select_columns = _raw_gap_select_columns(raw_columns)
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
            gap_rows = conn.execute(
                f"""
                WITH raw_rows AS (
                    SELECT
                        {raw_select_columns},
                        EXISTS (
                            SELECT 1
                            FROM main.sessions s
                            WHERE s.raw_id = r.raw_id
                        ) AS is_materialized
                    FROM source.raw_sessions r
                    WHERE COALESCE(r.validation_status, '') != 'skipped'
                )
                SELECT *
                FROM raw_rows
                WHERE NOT is_materialized
                """,
            ).fetchall()
            classified_counts = _classify_raw_gap_rows(
                conn,
                active_archive,
                gap_rows,
                raw_columns=raw_columns,
                session_columns=session_columns,
            )
            lost_source_evidence_count = _missing_source_raw_session_count(conn)
            lost_source_evidence_samples = _missing_source_raw_session_samples(conn)
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
    classified = sum(classified_counts.values())
    parse_failed_origins = {str(item["origin"] or "unknown") for item in gap_rows if item["parse_error"] is not None}
    actionable = len(parse_failed_origins)
    critical = actionable
    affected_actionable = parse_failed
    unchecked = max(total - classified - affected_actionable, 0)
    classification = "cheap_projection" if classified else "not_run"
    raw_id_join_gap_count = unchecked
    category_counts: dict[str, int] = {
        "raw_id_join_gap": raw_id_join_gap_count,
        "skipped": skipped,
        "parse_failed": parse_failed,
        "parsed_without_index_session": parsed_without_index_session,
    }
    category_counts.update(dict(classified_counts))
    return {
        "available": True,
        "classification": classification,
        "precision": "raw_id_join_gap",
        "raw_artifact_count": raw_artifact_count,
        "materialized_raw_artifact_count": materialized_raw_artifact_count,
        "archive_session_count": archive_session_count,
        "join_gap_count": join_gap_count,
        "total": total,
        "critical": critical,
        "warning": 0,
        "actionable": actionable,
        "blocked": 0,
        "classified": classified,
        "unchecked": unchecked,
        "affected_total": total,
        "affected_actionable": affected_actionable,
        "affected_blocked": 0,
        "affected_open": 0,
        "affected_classified": classified,
        "affected_unchecked": unchecked,
        "lost_source_evidence_count": lost_source_evidence_count,
        "lost_source_evidence_samples": lost_source_evidence_samples,
        "category_counts": category_counts,
        "source_family_counts": {str(item["origin"]): int(item["count"] or 0) for item in family_rows},
    }


def missing_source_raw_session_evidence(active_archive: Path, *, limit: int = 10) -> dict[str, object]:
    """Return indexed sessions whose source raw evidence is no longer present.

    This is the reverse of raw materialization debt. Raw materialization asks
    whether source rows have reached the index. This helper asks whether an
    indexed session still has the source row named by ``sessions.raw_id``. A
    missing row is lost source evidence until the exact raw artifact is
    recovered; it must not be repaired by relinking to a same-native but
    different source row.
    """

    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {
            "available": False,
            "reason": "source.db or index.db missing",
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
            if not _table_columns(conn, "main", "sessions") or not _table_columns(conn, "source", "raw_sessions"):
                return {
                    "available": False,
                    "reason": "sessions or raw_sessions table missing",
                    "missing_raw_session_count": 0,
                    "missing_raw_session_samples": [],
                    "lost_source_evidence_count": 0,
                    "lost_source_evidence_samples": [],
                }
            count = _missing_source_raw_session_count(conn)
            samples = _missing_source_raw_session_samples(conn, limit=limit)
    except sqlite3.Error as exc:
        return {
            "available": False,
            "reason": str(exc),
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    return {
        "available": True,
        "reason": None,
        "missing_raw_session_count": count,
        "missing_raw_session_samples": samples,
        "lost_source_evidence_count": count,
        "lost_source_evidence_samples": samples,
    }


def _missing_source_raw_session_count(conn: sqlite3.Connection) -> int:
    session_columns = _table_columns(conn, "main", "sessions")
    if "raw_id" not in session_columns:
        return 0
    return _readiness_scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM source.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        """,
    )


def _missing_source_raw_session_samples(conn: sqlite3.Connection, *, limit: int = 10) -> list[dict[str, object]]:
    session_columns = _table_columns(conn, "main", "sessions")
    if not {"session_id", "raw_id"} <= session_columns:
        return []
    origin_expr = "s.origin" if "origin" in session_columns else "NULL"
    native_id_expr = "s.native_id" if "native_id" in session_columns else "NULL"
    message_count_expr = "s.message_count" if "message_count" in session_columns else "NULL"
    updated_at_expr = "s.updated_at_ms" if "updated_at_ms" in session_columns else "NULL"
    order_expr = "s.updated_at_ms DESC, s.session_id" if "updated_at_ms" in session_columns else "s.session_id"
    rows = conn.execute(
        f"""
        SELECT s.session_id,
               {origin_expr} AS origin,
               {native_id_expr} AS native_id,
               s.raw_id,
               {message_count_expr} AS message_count,
               {updated_at_expr} AS updated_at_ms
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM source.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        ORDER BY {order_expr}
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "session_id": str(row["session_id"]),
            "origin": str(row["origin"]),
            "native_id": str(row["native_id"]),
            "missing_raw_id": str(row["raw_id"]),
            "message_count": int(row["message_count"] or 0),
            "updated_at_ms": None if row["updated_at_ms"] is None else int(row["updated_at_ms"]),
            "evidence_status": "lost_source_evidence",
            "loss_reason": "index_raw_id_missing_from_source_tier",
            "recovery_requirement": "restore_exact_raw_artifact_or_keep_blocked",
        }
        for row in rows
    ]


def _readiness_scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row is not None else 0


def _table_columns(conn: sqlite3.Connection, schema: str, table: str) -> frozenset[str]:
    try:
        rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row["name"] if isinstance(row, sqlite3.Row) else row[1]) for row in rows)


def _raw_gap_select_columns(raw_columns: frozenset[str]) -> str:
    def column(name: str) -> str:
        return f"r.{name}" if name in raw_columns else f"NULL AS {name}"

    names = (
        "raw_id",
        "origin",
        "native_id",
        "source_path",
        "blob_hash",
        "validation_status",
        "parse_error",
        "parsed_at_ms",
    )
    return ",\n                        ".join(column(name) for name in names)


def _classify_raw_gap_rows(
    conn: sqlite3.Connection,
    archive_root: Path,
    rows: list[sqlite3.Row],
    *,
    raw_columns: frozenset[str],
    session_columns: frozenset[str],
) -> Counter[str]:
    if not rows:
        return Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        if _raw_gap_materialized_by_alias(conn, row, session_columns=session_columns):
            counts["materialized-alias"] += 1
            continue
        if _raw_gap_matches_missing_index_raw_link(conn, row, session_columns=session_columns):
            counts["lost-source-evidence-alias"] += 1
            continue
        if _raw_gap_parsed_non_session_artifact(archive_root, row, raw_columns=raw_columns):
            counts["parsed-non-session-artifact"] += 1
    return counts


def _raw_gap_materialized_by_alias(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    session_columns: frozenset[str],
) -> bool:
    if not {"origin", "native_id"} <= session_columns:
        return False
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            """
            SELECT 1
            FROM main.sessions AS s
            JOIN source.raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
            WHERE s.origin = ?
              AND s.native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_gap_matches_missing_index_raw_link(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    session_columns: frozenset[str],
) -> bool:
    if not {"origin", "native_id", "raw_id"} <= session_columns:
        return False
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            """
            SELECT 1
            FROM main.sessions AS s
            WHERE s.origin = ?
              AND s.native_id = ?
              AND s.raw_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM source.raw_sessions AS existing_raw WHERE existing_raw.raw_id = s.raw_id
              )
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_gap_parsed_non_session_artifact(
    archive_root: Path,
    row: sqlite3.Row,
    *,
    raw_columns: frozenset[str],
) -> bool:
    if "blob_hash" not in raw_columns:
        return False
    if row["parse_error"] or row["parsed_at_ms"] is None:
        return False
    return (
        parsed_non_session_artifact_reason(
            archive_root=archive_root,
            origin=str(row["origin"] or ""),
            source_path=str(row["source_path"] or ""),
            blob_hash=row["blob_hash"],
        )
        is not None
    )


__all__ = [
    "ACTIVE_REBUILD_STALE_AFTER_S",
    "active_rebuild_index_attempts",
    "missing_source_raw_session_evidence",
    "raw_materialization_readiness_snapshot",
    "raw_materialization_ready",
]
