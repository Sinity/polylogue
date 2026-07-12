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
from polylogue.archive.revision_authority import BYTE_AUTHORITY_CENSUS_DETAIL
from polylogue.logging import get_logger

logger = get_logger(__name__)

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
    except sqlite3.Error as exc:
        logger.warning("active rebuild-index attempts query failed for %s: %s", ops_db, exc, exc_info=True)
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
            classified_counts, parse_failed_origins = _classify_raw_gap_rows(
                conn,
                active_archive,
                gap_rows,
                raw_columns=raw_columns,
                session_columns=session_columns,
                has_revision_applications=bool(_table_columns(conn, "main", "raw_revision_applications")),
                has_membership_census=bool(_table_columns(conn, "source", "raw_membership_census")),
                has_session_memberships=bool(_table_columns(conn, "source", "raw_session_memberships")),
            )
            adoption_deferred_count = 0
            if _table_columns(conn, "main", "raw_revision_applications"):
                adoption_deferred_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(DISTINCT r.raw_id)
                        FROM source.raw_sessions AS r
                        JOIN main.raw_revision_applications AS a ON a.raw_id = r.raw_id
                        WHERE a.decision = 'deferred'
                          AND a.detail = 'ordinary_replay:incomparable_existing_index_state'
                          AND NOT EXISTS (
                              SELECT 1 FROM main.sessions AS s WHERE s.raw_id = r.raw_id
                          )
                        """
                    ).fetchone()[0]
                    or 0
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
    raw_parse_failed = int(row["parse_failed"] or 0)
    parsed_without_index_session = int(row["parsed_without_index_session"] or 0)
    parse_failed = classified_counts.get("parse-failed", 0)
    classified = sum(count for category, count in classified_counts.items() if category != "parse-failed")
    actionable = len(parse_failed_origins)
    critical = actionable
    affected_actionable = parse_failed
    unchecked = max(total - classified - affected_actionable - adoption_deferred_count, 0)
    classification = "cheap_projection" if classified or adoption_deferred_count else "not_run"
    raw_id_join_gap_count = unchecked
    category_counts: dict[str, int] = {
        "raw_id_join_gap": raw_id_join_gap_count,
        "skipped": skipped,
        "parse_failed": parse_failed,
        "raw_parse_failed": raw_parse_failed,
        "parsed_without_index_session": parsed_without_index_session,
    }
    if adoption_deferred_count:
        category_counts["adoption_deferred"] = adoption_deferred_count
    category_counts.update(
        {category: count for category, count in classified_counts.items() if category != "parse-failed"}
    )
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
        "blocked": adoption_deferred_count,
        "classified": classified,
        "unchecked": unchecked,
        "affected_total": total,
        "affected_actionable": affected_actionable,
        "affected_blocked": adoption_deferred_count,
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
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_columns(conn: sqlite3.Connection, schema: str, table: str) -> frozenset[str]:
    try:
        # ``sessions.session_id`` and other identity columns are generated.
        # table_info omits generated/hidden columns, which made exact lost-raw
        # counts pair with empty samples on the canonical archive schema.
        rows = conn.execute(f"PRAGMA {schema}.table_xinfo({table})").fetchall()
    except sqlite3.Error as exc:
        logger.warning("archive readiness table-columns probe failed for %s.%s: %s", schema, table, exc, exc_info=True)
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
        "source_index",
        "revision_authority",
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
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> tuple[Counter[str], set[str]]:
    if not rows:
        return Counter(), set()
    counts: Counter[str] = Counter()
    parse_failed_origins: set[str] = set()
    for row in rows:
        category = _raw_gap_category(
            conn,
            archive_root,
            row,
            raw_columns=raw_columns,
            session_columns=session_columns,
            has_revision_applications=has_revision_applications,
            has_membership_census=has_membership_census,
            has_session_memberships=has_session_memberships,
        )
        if category is not None:
            counts[category] += 1
            if category == "parse-failed":
                parse_failed_origins.add(str(row["origin"] or "unknown"))
    return counts, parse_failed_origins


def _raw_gap_category(
    conn: sqlite3.Connection,
    archive_root: Path,
    row: sqlite3.Row,
    *,
    raw_columns: frozenset[str],
    session_columns: frozenset[str],
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> str | None:
    can_reconcile_alias = not row["parse_error"] or _retryable_decode_missing_blob_error(row["parse_error"])
    if can_reconcile_alias and _raw_gap_materialized_by_alias(conn, row, session_columns=session_columns):
        return "materialized-alias"
    if can_reconcile_alias and _raw_gap_matches_missing_index_raw_link(conn, row, session_columns=session_columns):
        return "lost-source-evidence-alias"
    if _raw_gap_parsed_non_session_artifact(archive_root, row, raw_columns=raw_columns):
        return "parsed-non-session-artifact"
    if row["parse_error"]:
        return "parse-failed"
    authority_category = _raw_gap_authority_category(
        conn,
        row,
        has_revision_applications=has_revision_applications,
        has_membership_census=has_membership_census,
        has_session_memberships=has_session_memberships,
    )
    if authority_category is not None:
        return authority_category
    return None


def _raw_gap_authority_category(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> str | None:
    raw_id = str(row["raw_id"])
    if has_revision_applications:
        terminal = conn.execute(
            """
            SELECT 1 FROM main.raw_revision_applications
            WHERE raw_id = ?
              AND decision IN ('selected_baseline', 'applied_append', 'superseded', 'ambiguous')
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if terminal is not None:
            return "revision-application-terminal"

    if has_membership_census and has_session_memberships:
        membership = conn.execute(
            """
            SELECT 1
            FROM source.raw_membership_census AS c
            WHERE c.raw_id = ?
              AND c.status = 'complete'
              AND c.member_count > 0
              AND c.member_count = (
                SELECT COUNT(*) FROM source.raw_session_memberships AS counted
                WHERE counted.raw_id = c.raw_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM source.raw_session_memberships AS m
                WHERE m.raw_id = c.raw_id
                  AND (m.decision IS NULL OR m.decision = 'deferred')
              )
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if membership is not None:
            return "membership-authority-classified"

    if row["source_index"] == -1:
        if row["revision_authority"] == "byte_proven":
            return "append-authority-proven"
        if has_membership_census:
            quarantined = conn.execute(
                """
                SELECT 1 FROM source.raw_membership_census
                WHERE raw_id = ? AND status = 'failed' AND detail = ?
                LIMIT 1
                """,
                (raw_id, BYTE_AUTHORITY_CENSUS_DETAIL),
            ).fetchone()
            if quarantined is not None:
                return "append-authority-quarantined"
    return None


def _retryable_decode_missing_blob_error(parse_error: object) -> bool:
    if not isinstance(parse_error, str):
        return False
    return parse_error.startswith("decode:") and "No such file or directory" in parse_error


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
