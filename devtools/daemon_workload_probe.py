"""Read-only daemon workload probe for live ingest and convergence hot paths.

The probe captures a stable, JSON-serializable snapshot of daemon-relevant
state that can be diffed across convergence cycles.  Operators run::

    polylogue ops diagnostics workload --json > before.json
    # ...run convergence work...
    polylogue ops diagnostics workload --json > after.json
    polylogue ops diagnostics workload --compare before.json after.json

to produce structured before/after convergence evidence.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.paths import active_index_db_path
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

# Bumped when the JSON shape gains new top-level keys or changes a field type.
# The compare path uses this to refuse incompatible inputs loudly.
REPORT_VERSION = 12  # v12 exposes physical DB/table locations for logical observability surfaces.
UNKNOWN_TABLE_COUNT = -2

_EXPECTED_FTS_TRIGGERS: tuple[str, ...] = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")

_KNOWN_STORAGE_ROUTES: frozenset[str] = frozenset(
    {
        "archive_full",
        "archive_append",
        "archive_file_set",
        "unsupported_polylogue_batch",
        "unknown",
    }
)

# Tables whose row counts form the convergence "boundary" — daemon work moves
# rows into and out of these tables, so the deltas describe convergence shape.
_BOUNDARY_TABLES: tuple[str, ...] = (
    "raw_sessions",
    "sessions",
    "messages",
    "artifact_observations",
    "messages_fts_docsize",
    "message_embeddings",
    "session_profiles",
    "live_ingest_attempt",
    "convergence_debt",
    "pending_blob_refs",
    "repos",
    "session_repos",
    "session_commits",
    "blocks",
    "session_events",
    "session_links",
)

_OPS_BOUNDARY_TABLES: tuple[str, ...] = (
    "ingest_attempts",
    "convergence_debt",
    "cursor_lag_samples",
    "daemon_stage_events",
    "daemon_events",
    "otlp_telemetry",
)

_ARCHIVE_OBSERVABILITY_TABLES: dict[ArchiveTier, tuple[str, ...]] = {
    ArchiveTier.SOURCE: (
        "raw_sessions",
        "blob_refs",
        "raw_artifacts",
        "raw_hook_events",
        "history_sidecars",
    ),
    ArchiveTier.INDEX: (
        "sessions",
        "messages",
        "blocks",
        "messages_fts_docsize",
        "session_events",
        "session_links",
        "threads",
        "thread_sessions",
        "session_working_dirs",
        "repos",
        "session_repos",
        "session_commits",
        "attachments",
        "attachment_refs",
        "paste_spans",
        "session_tags",
        "insight_materialization",
        "session_work_events",
        "session_phases",
        "session_profiles",
    ),
    ArchiveTier.EMBEDDINGS: (
        "message_embeddings",
        "embeddings_meta",
        "embedding_status",
    ),
    ArchiveTier.USER: (
        "assertions",
        "session_tags",
        "session_metadata",
    ),
    ArchiveTier.OPS: (
        "ingest_cursor",
        "ingest_attempts",
        "convergence_debt",
        "cursor_lag_samples",
        "daemon_stage_events",
        "daemon_events",
        "embedding_catchup_runs",
        "otlp_spans",
        "otlp_telemetry",
    ),
}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _attached_table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0) if row is not None else 0


def _table_count(conn: sqlite3.Connection, table: str, *, exact: bool) -> int:
    """Return an exact or cheap estimated table count."""

    if not _table_exists(conn, table):
        return -1
    if exact:
        return _scalar_int(conn, f"SELECT COUNT(*) FROM {table}")
    with suppress(sqlite3.Error, ValueError, IndexError):
        row = conn.execute(
            """
            SELECT stat
            FROM sqlite_stat1
            WHERE tbl = ?
            ORDER BY idx IS NOT NULL, idx
            LIMIT 1
            """,
            (table,),
        ).fetchone()
        if row is not None and row[0] is not None:
            return max(0, int(str(row[0]).split()[0]))
    return UNKNOWN_TABLE_COUNT


def _presence_count(conn: sqlite3.Connection, table: str) -> int:
    """Return 1 when a table has at least one row, without scanning it."""

    try:
        row = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
    except sqlite3.Error:
        return 0
    return 1 if row is not None else 0


def _recent_attempts(conn: sqlite3.Connection, *, limit: int, ops_db: Path | None = None) -> list[dict[str, Any]]:
    ops_attempts = _ops_recent_attempts(ops_db, limit=limit)
    if ops_attempts:
        return ops_attempts
    if not _table_exists(conn, "live_ingest_attempt"):
        return ops_attempts
    columns = _columns(conn, "live_ingest_attempt")
    stale_expr = "stale_cursor_write_count" if "stale_cursor_write_count" in columns else "0"
    source_paths_expr = "source_paths_json" if "source_paths_json" in columns else "'[]'"
    rows = conn.execute(
        f"""
        SELECT
            attempt_id,
            started_at,
            updated_at,
            completed_at,
            status,
            phase,
            queued_file_count,
            needed_file_count,
            succeeded_file_count,
            failed_file_count,
            input_bytes,
            source_payload_read_bytes,
            cursor_fingerprint_read_bytes,
            parse_time_s,
            convergence_time_s,
            {stale_expr},
            {source_paths_expr}
        FROM live_ingest_attempt
        ORDER BY updated_at DESC, started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    attempts: list[dict[str, Any]] = []
    for row in rows:
        input_bytes = int(row[10] or 0)
        read_bytes = int(row[11] or 0) + int(row[12] or 0)
        attempts.append(
            {
                "attempt_id": row[0],
                "started_at": row[1],
                "updated_at": row[2],
                "completed_at": row[3],
                "status": row[4],
                "phase": row[5],
                "queued_file_count": int(row[6] or 0),
                "needed_file_count": int(row[7] or 0),
                "succeeded_file_count": int(row[8] or 0),
                "failed_file_count": int(row[9] or 0),
                "input_bytes": input_bytes,
                "source_payload_read_bytes": int(row[11] or 0),
                "cursor_fingerprint_read_bytes": int(row[12] or 0),
                "total_read_bytes": read_bytes,
                "read_amplification": round(read_bytes / input_bytes, 3) if input_bytes > 0 else 0.0,
                "parse_time_s": float(row[13] or 0.0),
                "convergence_time_s": float(row[14] or 0.0),
                "stale_cursor_write_count": int(row[15] or 0),
                "source_paths": _json_list(row[16]),
            }
        )
    return attempts


def _ops_recent_attempts(ops_db: Path | None, *, limit: int) -> list[dict[str, Any]]:
    if ops_db is None or not ops_db.exists():
        return []
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return []
            rows = conn.execute(
                """
                SELECT attempt_id, started_at_ms, heartbeat_at_ms, finished_at_ms,
                       status, phase, parsed_raw_count, materialized_count,
                       error_message, source_paths_json
                FROM ingest_attempts
                ORDER BY COALESCE(heartbeat_at_ms, finished_at_ms, started_at_ms) DESC,
                         started_at_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            payloads = _ops_stage_payloads(conn, [str(row[0]) for row in rows])
        finally:
            conn.close()
    except sqlite3.Error:
        return []

    attempts: list[dict[str, Any]] = []
    for row in rows:
        attempt_id = str(row[0])
        payload = payloads.get(attempt_id, {})
        input_bytes = _payload_int(payload, "input_bytes")
        source_payload_read_bytes = _payload_int(payload, "source_payload_read_bytes")
        cursor_fingerprint_read_bytes = _payload_int(payload, "cursor_fingerprint_read_bytes")
        total_read_bytes = source_payload_read_bytes + cursor_fingerprint_read_bytes
        source_paths = _json_list(row[9])
        attempts.append(
            {
                "attempt_id": attempt_id,
                "started_at": _iso_from_epoch_ms(row[1]),
                "updated_at": _iso_from_epoch_ms(row[2] or row[3] or row[1]),
                "completed_at": _iso_from_epoch_ms(row[3]),
                "status": row[4],
                "phase": row[5],
                "queued_file_count": _payload_int(payload, "queued_file_count", default=len(source_paths)),
                "needed_file_count": _payload_int(payload, "needed_file_count", default=len(source_paths)),
                "succeeded_file_count": int(row[7] or 0),
                "failed_file_count": 1 if row[4] == "failed" else 0,
                "input_bytes": input_bytes,
                "source_payload_read_bytes": source_payload_read_bytes,
                "cursor_fingerprint_read_bytes": cursor_fingerprint_read_bytes,
                "total_read_bytes": total_read_bytes,
                "read_amplification": round(total_read_bytes / input_bytes, 3) if input_bytes > 0 else 0.0,
                "parse_time_s": _payload_float(payload, "parse_time_s"),
                "convergence_time_s": _payload_float(payload, "convergence_time_s"),
                "stale_cursor_write_count": _payload_int(payload, "stale_cursor_write_count"),
                "storage_route": _normalise_storage_route(payload.get("storage_route")),
                "source_paths": source_paths,
            }
        )
    return attempts


def _ops_stage_payloads(conn: sqlite3.Connection, attempt_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not attempt_ids or not _table_exists(conn, "daemon_stage_events"):
        return {}
    placeholders = ",".join("?" for _ in attempt_ids)
    rows = conn.execute(
        f"""
        SELECT attempt_id, payload_json
        FROM daemon_stage_events
        WHERE attempt_id IN ({placeholders})
        ORDER BY observed_at_ms DESC, rowid DESC
        """,
        tuple(attempt_ids),
    ).fetchall()
    payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        attempt_id = str(row[0])
        try:
            decoded = json.loads(str(row[1] or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            merged = payloads.setdefault(attempt_id, {})
            for key, value in decoded.items():
                merged.setdefault(key, value)
    return payloads


def _payload_int(payload: dict[str, Any], key: str, *, default: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return int(value)


def _payload_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _payload_str(payload: dict[str, Any], key: str, *, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _normalise_storage_route(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    return value if value in _KNOWN_STORAGE_ROUTES else "other"


def _storage_route_counts(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> dict[str, int]:
    ops_counts = _ops_storage_route_counts(ops_db)
    if ops_counts is not None:
        return ops_counts

    counts = dict.fromkeys(sorted(_KNOWN_STORAGE_ROUTES), 0)
    counts["other"] = 0
    if not _table_exists(conn, "live_ingest_attempt"):
        return counts
    columns = _columns(conn, "live_ingest_attempt")
    if "storage_route" not in columns:
        counts["unknown"] = _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt")
        return counts
    rows = conn.execute("SELECT storage_route, COUNT(*) FROM live_ingest_attempt GROUP BY storage_route").fetchall()
    for row in rows:
        route = _normalise_storage_route(row[0])
        counts[route] = counts.get(route, 0) + int(row[1] or 0)
    return counts


def _ops_storage_route_counts(ops_db: Path | None) -> dict[str, int] | None:
    if ops_db is None or not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return None
            rows = conn.execute("SELECT attempt_id FROM ingest_attempts").fetchall()
            attempt_ids = [str(row[0]) for row in rows]
            if not attempt_ids:
                return None
            payloads = _ops_stage_payloads(conn, attempt_ids)
            counts = dict.fromkeys(sorted(_KNOWN_STORAGE_ROUTES), 0)
            counts["other"] = 0
            for attempt_id in attempt_ids:
                route = _normalise_storage_route(payloads.get(attempt_id, {}).get("storage_route"))
                counts[route] = counts.get(route, 0) + 1
            return counts
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _attempt_counts(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> dict[str, Any]:
    ops_counts = _ops_attempt_counts(ops_db)
    if ops_counts is not None and (ops_counts["total"] > 0 or not _table_exists(conn, "live_ingest_attempt")):
        return ops_counts
    if not _table_exists(conn, "live_ingest_attempt"):
        return {
            "total": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stale_cursor_writes": 0,
            "overlapping_source_paths": [],
        }
    running_rows = conn.execute(
        """
        SELECT source_paths_json
        FROM live_ingest_attempt
        WHERE status = 'running'
        """
    ).fetchall()
    path_counts: dict[str, int] = {}
    for row in running_rows:
        for path in _json_list(row[0]):
            path_counts[path] = path_counts.get(path, 0) + 1
    overlapping = [
        {"source_path": path, "running_attempt_count": count}
        for path, count in sorted(path_counts.items())
        if count > 1
    ]
    has_stale_col = "stale_cursor_write_count" in _columns(conn, "live_ingest_attempt")
    return {
        "total": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt"),
        "running": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'running'"),
        "completed": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'completed'"),
        "failed": _scalar_int(conn, "SELECT COUNT(*) FROM live_ingest_attempt WHERE status = 'failed'"),
        "stale_cursor_writes": _scalar_int(
            conn,
            "SELECT COALESCE(SUM(stale_cursor_write_count), 0) FROM live_ingest_attempt"
            if has_stale_col
            else "SELECT 0",
        ),
        "overlapping_source_paths": overlapping[:20],
    }


def _ops_attempt_counts(ops_db: Path | None) -> dict[str, Any] | None:
    if ops_db is None or not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return None
            rows = conn.execute(
                """
                SELECT source_paths_json
                FROM ingest_attempts
                WHERE status = 'running'
                """
            ).fetchall()
            path_counts: dict[str, int] = {}
            for row in rows:
                for path in _json_list(row[0]):
                    path_counts[path] = path_counts.get(path, 0) + 1
            overlapping = [
                {"source_path": path, "running_attempt_count": count}
                for path, count in sorted(path_counts.items())
                if count > 1
            ]
            return {
                "total": _scalar_int(conn, "SELECT COUNT(*) FROM ingest_attempts"),
                "running": _scalar_int(conn, "SELECT COUNT(*) FROM ingest_attempts WHERE status = 'running'"),
                "completed": _scalar_int(conn, "SELECT COUNT(*) FROM ingest_attempts WHERE status = 'completed'"),
                "failed": _scalar_int(conn, "SELECT COUNT(*) FROM ingest_attempts WHERE status = 'failed'"),
                "stale_cursor_writes": 0,
                "overlapping_source_paths": overlapping[:20],
            }
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _source_path_churn(
    conn: sqlite3.Connection,
    *,
    attempts: list[dict[str, Any]],
    limit: int,
    db: Path,
) -> list[dict[str, Any]]:
    archive_churn = _archive_source_path_churn(db.parent, attempts=attempts, limit=limit)
    if archive_churn:
        return archive_churn
    if not _table_exists(conn, "raw_sessions"):
        return []
    source_paths = sorted({path for attempt in attempts for path in attempt.get("source_paths", []) if path})
    if not source_paths:
        return []
    has_sessions = _table_exists(conn, "sessions")
    join = "LEFT JOIN sessions AS c ON c.raw_id = r.raw_id" if has_sessions else ""
    session_count = "COUNT(DISTINCT c.session_id)" if has_sessions else "0"
    placeholders = ",".join("?" for _ in source_paths)
    rows = conn.execute(
        f"""
        SELECT
            r.source_path,
            COUNT(*) AS raw_count,
            SUM(CASE WHEN COALESCE(r.source_index, 0) >= 0 THEN 1 ELSE 0 END) AS full_raw_count,
            SUM(CASE WHEN r.source_index = -1 THEN 1 ELSE 0 END) AS append_raw_count,
            {session_count} AS session_count,
            SUM(COALESCE(r.blob_size, 0)) AS total_blob_bytes,
            MAX(r.acquired_at) AS latest_acquired_at
        FROM raw_sessions AS r
        {join}
        WHERE r.source_path IN ({placeholders})
        GROUP BY r.source_path
        HAVING raw_count > 1
        ORDER BY raw_count DESC, total_blob_bytes DESC, r.source_path
        LIMIT ?
        """,
        (*source_paths, limit),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        raw_count = int(row[1] or 0)
        session_count_value = int(row[4] or 0)
        items.append(
            {
                "source_path": row[0],
                "raw_count": raw_count,
                "full_raw_count": int(row[2] or 0),
                "append_raw_count": int(row[3] or 0),
                "session_count": session_count_value,
                "orphan_raw_count": max(0, raw_count - session_count_value),
                "total_blob_bytes": int(row[5] or 0),
                "latest_acquired_at": row[6],
            }
        )
    return items


def _archive_source_path_churn(
    root: Path,
    *,
    attempts: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    source_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    index_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].filename
    if not source_db.exists() or not index_db.exists():
        return []
    source_paths = sorted({path for attempt in attempts for path in attempt.get("source_paths", []) if path})
    if not source_paths:
        return []

    placeholders = ",".join("?" for _ in source_paths)
    try:
        conn = open_readonly_connection(index_db)
        try:
            conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
            if not _table_exists(conn, "sessions") or not _attached_table_exists(conn, "source_tier", "raw_sessions"):
                return []
            rows = conn.execute(
                f"""
                SELECT
                    r.source_path,
                    COUNT(*) AS raw_count,
                    SUM(CASE WHEN COALESCE(r.source_index, 0) >= 0 THEN 1 ELSE 0 END) AS full_raw_count,
                    SUM(CASE WHEN r.source_index = -1 THEN 1 ELSE 0 END) AS append_raw_count,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(DISTINCT CASE WHEN s.session_id IS NOT NULL THEN r.raw_id END) AS materialized_raw_count,
                    SUM(COALESCE(r.blob_size, 0)) AS total_blob_bytes,
                    MAX(r.acquired_at_ms) AS latest_acquired_at_ms
                FROM source_tier.raw_sessions AS r
                LEFT JOIN sessions AS s ON s.raw_id = r.raw_id
                WHERE r.source_path IN ({placeholders})
                GROUP BY r.source_path
                HAVING raw_count > 1
                ORDER BY raw_count DESC, total_blob_bytes DESC, r.source_path
                LIMIT ?
                """,
                (*source_paths, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []

    items: list[dict[str, Any]] = []
    for row in rows:
        raw_count = int(row[1] or 0)
        materialized_raw_count = int(row[5] or 0)
        items.append(
            {
                "source_path": row[0],
                "storage_route": "archive_file_set",
                "raw_count": raw_count,
                "full_raw_count": int(row[2] or 0),
                "append_raw_count": int(row[3] or 0),
                "session_count": int(row[4] or 0),
                "materialized_raw_count": materialized_raw_count,
                "orphan_raw_count": max(0, raw_count - materialized_raw_count),
                "total_blob_bytes": int(row[6] or 0),
                "latest_acquired_at": _iso_from_epoch_ms(row[7]),
            }
        )
    return items


def _cursor_lag_baselines(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> dict[str, Any]:
    """Rolling-window cursor-lag baseline state per source family.

    Prefer ``ops.db`` ``cursor_lag_samples`` and fall back to single-file
    ``live_cursor_lag_sample``. Returns the same per-family snapshot the
    anomaly check uses: sample-count, time range, p50, p95.
    Stable JSON shape so before/after probes can detect baseline drift in
    convergence evidence snapshots (e.g. *"after this run, claude-code-session's
    rolling p95 dropped from 600s to 12s — convergence is healthy"*).
    """
    ops_baselines = _ops_cursor_lag_baselines(ops_db)
    if ops_baselines is not None:
        return ops_baselines
    if not _table_exists(conn, "live_cursor_lag_sample"):
        return {"table_present": False, "family_count": 0, "total_sample_count": 0, "families": []}
    rows = conn.execute(
        """
        SELECT family,
               COUNT(*),
               MIN(observed_at),
               MAX(observed_at),
               MAX(max_lag_s),
               AVG(max_lag_s),
               SUM(stuck_file_count)
        FROM live_cursor_lag_sample
        GROUP BY family
        ORDER BY COUNT(*) DESC, family
        """
    ).fetchall()
    families: list[dict[str, Any]] = []
    for row in rows:
        family = str(row[0])
        sample_count = int(row[1] or 0)
        # Compute p50/p95 per family with a small follow-up read; bounded
        # at ~200K rows total across families this stays under a second.
        per_family = [
            float(r[0])
            for r in conn.execute(
                "SELECT max_lag_s FROM live_cursor_lag_sample WHERE family = ? ORDER BY max_lag_s",
                (family,),
            ).fetchall()
        ]
        p50 = _percentile_from_sorted(per_family, 0.5)
        p95 = _percentile_from_sorted(per_family, 0.95)
        families.append(
            {
                "family": family,
                "sample_count": sample_count,
                "first_observed_at": row[2],
                "last_observed_at": row[3],
                "max_lag_s_seen": round(float(row[4] or 0.0), 3),
                "mean_lag_s": round(float(row[5] or 0.0), 3),
                "stuck_file_total": int(row[6] or 0),
                "p50_lag_s": round(p50, 3),
                "p95_lag_s": round(p95, 3),
            }
        )
    return {
        "table_present": True,
        "family_count": len(families),
        "total_sample_count": sum(f["sample_count"] for f in families),
        "families": families,
    }


def _ops_cursor_lag_baselines(ops_db: Path | None) -> dict[str, Any] | None:
    if ops_db is None or not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "cursor_lag_samples"):
                return None
            columns = _columns(conn, "cursor_lag_samples")
            if "family" not in columns:
                return None
            rows = conn.execute(
                """
                SELECT family,
                       COUNT(*),
                       MIN(sampled_at_ms),
                       MAX(sampled_at_ms),
                       MAX(lag_ms),
                       AVG(lag_ms),
                       SUM(stuck_file_count)
                FROM cursor_lag_samples
                GROUP BY family
                ORDER BY COUNT(*) DESC, family
                """
            ).fetchall()
            families: list[dict[str, Any]] = []
            for row in rows:
                family = str(row[0])
                sample_count = int(row[1] or 0)
                per_family = [
                    float(r[0]) / 1000.0
                    for r in conn.execute(
                        "SELECT lag_ms FROM cursor_lag_samples WHERE family = ? ORDER BY lag_ms",
                        (family,),
                    ).fetchall()
                ]
                families.append(
                    {
                        "family": family,
                        "sample_count": sample_count,
                        "first_observed_at": _iso_from_epoch_ms(row[2]),
                        "last_observed_at": _iso_from_epoch_ms(row[3]),
                        "max_lag_s_seen": round(float(row[4] or 0.0) / 1000.0, 3),
                        "mean_lag_s": round(float(row[5] or 0.0) / 1000.0, 3),
                        "stuck_file_total": int(row[6] or 0),
                        "p50_lag_s": round(_percentile_from_sorted(per_family, 0.5), 3),
                        "p95_lag_s": round(_percentile_from_sorted(per_family, 0.95), 3),
                    }
                )
            return {
                "table_present": True,
                "family_count": len(families),
                "total_sample_count": sum(f["sample_count"] for f in families),
                "families": families,
            }
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _iso_from_epoch_ms(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _percentile_from_sorted(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _convergence_debt(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> dict[str, Any]:
    ops_rows = _ops_convergence_debt_rows(ops_db)
    if ops_rows:
        return {
            "failed_count": sum(count for _stage, count in ops_rows),
            "by_stage": [{"stage": stage, "failed_count": count} for stage, count in ops_rows],
        }
    if not _table_exists(conn, "live_convergence_debt"):
        return {"failed_count": 0, "by_stage": []}
    rows = conn.execute(
        """
        SELECT stage, COUNT(*) AS failed_count
        FROM live_convergence_debt
        WHERE status != 'resolved'
        GROUP BY stage
        ORDER BY failed_count DESC, stage
        """
    ).fetchall()
    return {
        "failed_count": sum(int(row[1] or 0) for row in rows),
        "by_stage": [{"stage": row[0], "failed_count": int(row[1] or 0)} for row in rows],
    }


def _ops_convergence_debt_rows(ops_db: Path | None) -> list[tuple[str, int]]:
    if ops_db is None or not ops_db.exists():
        return []
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "convergence_debt"):
                return []
            rows = conn.execute(
                """
                SELECT stage, COUNT(*) AS failed_count
                FROM convergence_debt
                GROUP BY stage
                ORDER BY failed_count DESC, stage
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [(str(row[0] or "unknown"), int(row[1] or 0)) for row in rows]


def _boundary_table_counts(
    conn: sqlite3.Connection,
    *,
    ops_db: Path | None = None,
    source_db: Path | None = None,
    exact: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _BOUNDARY_TABLES:
        counts[table] = _table_count(conn, table, exact=exact)
    if source_db is not None and source_db.exists():
        try:
            source_conn = open_readonly_connection(source_db)
            try:
                for table in ("raw_sessions",):
                    counts[table] = _table_count(source_conn, table, exact=exact)
            finally:
                source_conn.close()
        except sqlite3.Error:
            counts.setdefault("raw_sessions", -1)
    if ops_db is not None and ops_db.exists():
        try:
            ops_conn = open_readonly_connection(ops_db)
            try:
                for table in _OPS_BOUNDARY_TABLES:
                    counts[table] = _table_count(ops_conn, table, exact=exact)
            finally:
                ops_conn.close()
        except sqlite3.Error:
            for table in _OPS_BOUNDARY_TABLES:
                counts.setdefault(table, -1)
    return counts


def _location_entry(
    db_path: Path,
    *,
    table: str,
    tier: str,
    logical_table: str | None = None,
) -> dict[str, Any]:
    exists = False
    if db_path.exists():
        try:
            conn = open_readonly_connection(db_path)
            try:
                exists = _table_exists(conn, table)
            finally:
                conn.close()
        except sqlite3.Error:
            exists = False
    return {
        "logical_table": logical_table or table,
        "physical_table": table,
        "tier": tier,
        "db_path": str(db_path),
        "exists": exists,
    }


def _observability_locations(
    db: Path,
    *,
    observed_db: Path | None,
    ops_db: Path | None,
) -> dict[str, Any]:
    index_db = observed_db or db
    source_db = db.with_name("source.db")
    resolved_ops_db = ops_db or db.with_name("ops.db")
    logical_tables = {
        "live_ingest_attempt": _location_entry(
            resolved_ops_db,
            table="ingest_attempts",
            tier="ops",
            logical_table="live_ingest_attempt",
        ),
        "live_ingest_attempt_stage_events": _location_entry(
            resolved_ops_db,
            table="daemon_stage_events",
            tier="ops",
            logical_table="live_ingest_attempt_stage_events",
        ),
        "convergence_debt": _location_entry(resolved_ops_db, table="convergence_debt", tier="ops"),
        "raw_sessions": _location_entry(source_db, table="raw_sessions", tier="source"),
        "sessions": _location_entry(index_db, table="sessions", tier="index"),
        "messages": _location_entry(index_db, table="messages", tier="index"),
        "blocks": _location_entry(index_db, table="blocks", tier="index"),
        "messages_fts": _location_entry(index_db, table="messages_fts", tier="index"),
        "pending_blob_refs": _location_entry(source_db, table="pending_blob_refs", tier="source"),
    }
    return {
        "archive_root": str(db.parent),
        "tiers": {
            "index": str(index_db),
            "source": str(source_db),
            "ops": str(resolved_ops_db),
        },
        "logical_tables": logical_tables,
    }


def _archive_tier_state(
    anchor_db: Path,
    *,
    observed_db: Path | None = None,
    integrity_check: bool = False,
    exact_derived_counts: bool = False,
    exact_table_counts: bool = False,
) -> dict[str, Any]:
    """Report the archive database files around an archive root."""

    root = anchor_db.parent
    tiers: dict[str, Any] = {}
    present_count = 0
    complete = True
    schema_mismatches: list[str] = []
    missing_backup_required: list[str] = []
    observed_tier: str | None = None
    if observed_db is not None:
        for tier, spec in ARCHIVE_TIER_SPECS.items():
            if observed_db == root / spec.filename:
                observed_tier = tier.value
                break

    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = root / spec.filename
        tier_payload = _archive_single_tier_state(
            path,
            tier,
            integrity_check=integrity_check,
            exact_table_counts=exact_table_counts,
        )
        tier_payload.update(
            {
                "tier": tier.value,
                "filename": spec.filename,
                "durability": spec.durability,
                "backup_required": spec.backup_required,
                "expected_user_version": spec.version,
                "observed_by_probe": observed_db == path if observed_db is not None else False,
            }
        )
        if tier_payload["exists"]:
            present_count += 1
            if tier_payload.get("user_version") != spec.version:
                schema_mismatches.append(tier.value)
        else:
            complete = False
            if spec.backup_required:
                missing_backup_required.append(tier.value)
        tiers[tier.value] = tier_payload

    derived_readiness = _archive_derived_readiness(root, exact_counts=exact_derived_counts)
    user_overlay_orphans = _archive_user_overlay_orphans(root)
    layout_readiness = _archive_layout_readiness(
        present_count=present_count,
        complete=complete,
        schema_mismatches=schema_mismatches,
        missing_backup_required=missing_backup_required,
        derived_readiness=derived_readiness,
        user_overlay_orphans=user_overlay_orphans,
    )

    return {
        "root": str(root),
        "present": present_count > 0,
        "complete": complete,
        "present_count": present_count,
        "expected_count": len(ARCHIVE_TIER_SPECS),
        "observed_tier": observed_tier,
        "table_count_mode": "exact" if exact_table_counts else "estimated",
        "schema_mismatches": schema_mismatches,
        "missing_backup_required": missing_backup_required,
        "layout_readiness": layout_readiness,
        "derived_readiness": derived_readiness,
        "user_overlay_orphans": user_overlay_orphans,
        "tiers": tiers,
    }


def _archive_layout_readiness(
    *,
    present_count: int,
    complete: bool,
    schema_mismatches: list[str],
    missing_backup_required: list[str],
    derived_readiness: dict[str, Any],
    user_overlay_orphans: dict[str, Any],
) -> dict[str, Any]:
    """Project tier/detail state into one archive layout operator verdict."""

    blockers: list[str] = []
    if present_count == 0:
        blockers.append("no_archive_tiers_present")
    if not complete:
        blockers.append("missing_archive_tiers")
    blockers.extend(f"schema_mismatch:{tier}" for tier in schema_mismatches)
    blockers.extend(f"missing_backup_required_tier:{tier}" for tier in missing_backup_required)
    derived_checked = bool(derived_readiness.get("checked"))
    surface_readiness = derived_readiness.get("surface_readiness") or {}
    if not derived_checked:
        reason = derived_readiness.get("reason") or "unknown"
        blockers.append(f"derived_readiness_unchecked:{reason}")
    else:
        if not derived_readiness.get("source_check_available"):
            blockers.append("source_tier_not_attached_to_index")
        for surface_name, info in sorted(surface_readiness.items()):
            if info.get("ready") is True:
                continue
            surface_blockers = info.get("blockers") or ["not_ready"]
            for blocker in surface_blockers:
                blockers.append(f"surface:{surface_name}:{blocker}")

    overlay_checked = bool(user_overlay_orphans.get("checked"))
    overlay_orphans = int(user_overlay_orphans.get("total_orphan_session_references") or 0)
    if not overlay_checked:
        reason = user_overlay_orphans.get("reason") or "unknown"
        blockers.append(f"user_overlay_unchecked:{reason}")
    elif overlay_orphans:
        blockers.append("user_overlay_orphan_session_references")

    return {
        "state": "archive_ready" if not blockers else "not_archive_ready",
        "archive_ready": not blockers,
        "blockers": blockers,
        "evidence": {
            "present_count": present_count,
            "expected_count": len(ARCHIVE_TIER_SPECS),
            "complete": complete,
            "schema_mismatch_count": len(schema_mismatches),
            "missing_backup_required_count": len(missing_backup_required),
            "derived_readiness_checked": derived_checked,
            "derived_surface_count": len(surface_readiness),
            "blocked_surface_count": sum(1 for info in surface_readiness.values() if info.get("ready") is not True),
            "user_overlay_checked": overlay_checked,
            "user_overlay_orphan_session_references": overlay_orphans,
        },
    }


def _archive_single_tier_state(
    path: Path,
    tier: ArchiveTier,
    *,
    integrity_check: bool = False,
    exact_table_counts: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "user_version": None,
            "integrity": "missing",
            "table_counts": {},
            "error": None,
        }
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "user_version": None,
        "integrity": "not_checked",
        "table_counts": {},
        "error": None,
    }
    try:
        conn = open_readonly_connection(path)
        try:
            version_row = conn.execute("PRAGMA user_version").fetchone()
            payload["user_version"] = int(version_row[0] or 0) if version_row is not None else 0
            if integrity_check:
                integrity_row = conn.execute("PRAGMA quick_check").fetchone()
                payload["integrity"] = str(integrity_row[0]) if integrity_row is not None else "unknown"
            payload["table_counts"] = {
                table: _table_count(conn, table, exact=exact_table_counts)
                for table in _ARCHIVE_OBSERVABILITY_TABLES[tier]
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        payload["integrity"] = "error"
        payload["error"] = str(exc)
    return payload


def _archive_derived_readiness(root: Path, *, exact_counts: bool = False) -> dict[str, Any]:
    """Report cross-table readiness for derived surfaces."""

    index_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].filename
    source_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    if not index_db.exists():
        return {
            "checked": False,
            "reason": "missing_index_tier",
            "counts": {},
            "materialization_counts": {},
            "missing_materialization_counts": {},
            "ready": {},
        }

    conn = open_readonly_connection(index_db)
    source_attached = False
    source_check_available = False
    try:
        if source_db.exists():
            conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
            source_attached = True
            source_check_available = _attached_table_exists(conn, "source_tier", "raw_sessions")
        counts = _archive_derived_counts(conn, source_check_available=source_check_available, exact_counts=exact_counts)
        materialization_counts = _archive_materialization_counts(conn)
        missing_materialization_counts = _archive_missing_materialization_counts(conn)
        messages_fts_ready = (
            counts["text_block_count"] == counts["messages_fts_count"]
            if exact_counts
            else _fts_trigger_state(conn)["all_present"]
            and (counts["block_count"] == 0 or counts["messages_fts_count"] > 0)
        )
        ready = {
            "raw_links_ready": counts["missing_raw_session_count"] == 0 if source_check_available else None,
            "messages_fts_ready": messages_fts_ready,
            "profile_rows_ready": counts["missing_profile_row_count"] == 0 and counts["orphan_profile_row_count"] == 0,
            "profile_counts_ready": (
                counts["profile_work_event_count_mismatch"] == 0 and counts["profile_phase_count_mismatch"] == 0
            ),
            "profile_materialization_ready": missing_materialization_counts["session_profile"] == 0,
            "work_event_materialization_ready": missing_materialization_counts["work_events"] == 0,
            "phase_materialization_ready": missing_materialization_counts["phases"] == 0,
            "thread_materialization_ready": missing_materialization_counts["thread"] == 0,
            "latency_materialization_ready": missing_materialization_counts["latency"] == 0,
        }
        surface_readiness = _archive_surface_readiness(
            counts,
            missing_materialization_counts=missing_materialization_counts,
            source_check_available=source_check_available,
        )
        return {
            "checked": True,
            "reason": None,
            "source_check_available": source_check_available,
            "counts": counts,
            "materialization_counts": materialization_counts,
            "missing_materialization_counts": missing_materialization_counts,
            "ready": ready,
            "surface_readiness": surface_readiness,
        }
    except sqlite3.Error as exc:
        return {
            "checked": False,
            "reason": str(exc),
            "counts": {},
            "materialization_counts": {},
            "missing_materialization_counts": {},
            "ready": {},
            "surface_readiness": {},
        }
    finally:
        if source_attached:
            with suppress(sqlite3.Error):
                conn.execute("DETACH DATABASE source_tier")
        conn.close()


def _archive_derived_counts(
    conn: sqlite3.Connection, *, source_check_available: bool, exact_counts: bool = False
) -> dict[str, Any]:
    missing_raw = 0
    if source_check_available:
        missing_raw = _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE s.raw_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM source_tier.raw_sessions AS r WHERE r.raw_id = s.raw_id
              )
            """,
        )
    block_count = _scalar_int(conn, "SELECT COUNT(*) FROM blocks")
    if exact_counts:
        text_block_count = _scalar_int(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
        messages_fts_count = _scalar_int(conn, "SELECT COUNT(*) FROM messages_fts")
    else:
        text_block_count = block_count
        messages_fts_count = _scalar_int(conn, "SELECT COUNT(*) FROM messages_fts_docsize")
    return {
        "session_count": _scalar_int(conn, "SELECT COUNT(*) FROM sessions"),
        "raw_link_count": _scalar_int(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL"),
        "missing_raw_session_count": missing_raw,
        "message_count": _scalar_int(conn, "SELECT COUNT(*) FROM messages"),
        "block_count": block_count,
        "text_block_count": text_block_count,
        "messages_fts_count": messages_fts_count,
        "messages_fts_exact_counts": exact_counts,
        "profile_row_count": _scalar_int(conn, "SELECT COUNT(*) FROM session_profiles"),
        "missing_profile_row_count": _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
            )
            """,
        ),
        "orphan_profile_row_count": _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE NOT EXISTS (
                SELECT 1 FROM sessions AS s WHERE s.session_id = p.session_id
            )
            """,
        ),
        "work_event_row_count": _scalar_int(conn, "SELECT COUNT(*) FROM session_work_events"),
        "phase_row_count": _scalar_int(conn, "SELECT COUNT(*) FROM session_phases"),
        "thread_count": _scalar_int(conn, "SELECT COUNT(*) FROM threads"),
        "thread_session_count": _scalar_int(conn, "SELECT COUNT(*) FROM thread_sessions"),
        "session_tag_count": _scalar_int(conn, "SELECT COUNT(*) FROM session_tags"),
        "action_count": _scalar_int(conn, "SELECT COUNT(*) FROM actions")
        if exact_counts
        else _presence_count(conn, "actions"),
        "action_count_exact": exact_counts,
        "cost_profile_count": _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM session_profiles
            WHERE cost_usd IS NOT NULL OR cost_credits IS NOT NULL
            """,
        ),
        "profile_work_event_count_mismatch": _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE p.work_event_count != (
                SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
            )
            """,
        ),
        "profile_phase_count_mismatch": _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM session_profiles AS p
            WHERE p.phase_count != (
                SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
            )
            """,
        ),
    }


def _archive_surface_readiness(
    counts: dict[str, Any],
    *,
    missing_materialization_counts: dict[str, int],
    source_check_available: bool,
) -> dict[str, Any]:
    """Project low-level archive counters into operator-facing surface verdicts."""

    def surface(
        *,
        ready: bool | None,
        blockers: list[str],
        evidence: dict[str, int | bool | None],
    ) -> dict[str, Any]:
        return {
            "ready": ready,
            "blockers": blockers,
            "evidence": evidence,
        }

    def materialization_surface(insight_type: str) -> tuple[bool, list[str]]:
        missing = missing_materialization_counts.get(insight_type, 0)
        if missing == 0:
            return True, []
        return False, [f"missing_{insight_type}_materialization"]

    raw_ready: bool | None
    raw_blockers: list[str] = []
    if not source_check_available:
        raw_ready = None
        raw_blockers.append("source_tier_unavailable")
    elif counts["missing_raw_session_count"] == 0:
        raw_ready = True
    else:
        raw_ready = False
        raw_blockers.append("missing_source_raw_sessions")

    search_blockers: list[str] = []
    if counts["messages_fts_exact_counts"] and counts["text_block_count"] != counts["messages_fts_count"]:
        search_blockers.append("messages_fts_row_mismatch")

    profile_ready = (
        counts["missing_profile_row_count"] == 0
        and counts["orphan_profile_row_count"] == 0
        and counts["profile_work_event_count_mismatch"] == 0
        and counts["profile_phase_count_mismatch"] == 0
        and missing_materialization_counts["session_profile"] == 0
    )
    profile_blockers: list[str] = []
    if counts["missing_profile_row_count"]:
        profile_blockers.append("missing_profile_rows")
    if counts["orphan_profile_row_count"]:
        profile_blockers.append("orphan_profile_rows")
    if counts["profile_work_event_count_mismatch"]:
        profile_blockers.append("profile_work_event_count_mismatch")
    if counts["profile_phase_count_mismatch"]:
        profile_blockers.append("profile_phase_count_mismatch")
    if missing_materialization_counts["session_profile"]:
        profile_blockers.append("missing_session_profile_materialization")

    work_events_ready, work_events_blockers = materialization_surface("work_events")
    phases_ready, phases_blockers = materialization_surface("phases")
    thread_ready, thread_blockers = materialization_surface("thread")
    latency_ready, latency_blockers = materialization_surface("latency")

    return {
        "archive_sessions": surface(
            ready=True,
            blockers=[],
            evidence={
                "session_count": counts["session_count"],
                "message_count": counts["message_count"],
                "text_block_count": counts["text_block_count"],
            },
        ),
        "raw_artifacts": surface(
            ready=raw_ready,
            blockers=raw_blockers,
            evidence={
                "source_check_available": source_check_available,
                "raw_link_count": counts["raw_link_count"],
                "missing_raw_session_count": counts["missing_raw_session_count"],
            },
        ),
        "search": surface(
            ready=not search_blockers,
            blockers=search_blockers,
            evidence={
                "text_block_count": counts["text_block_count"],
                "messages_fts_count": counts["messages_fts_count"],
                "messages_fts_exact_counts": counts["messages_fts_exact_counts"],
            },
        ),
        "session_profiles": surface(
            ready=profile_ready,
            blockers=profile_blockers,
            evidence={
                "profile_row_count": counts["profile_row_count"],
                "missing_profile_row_count": counts["missing_profile_row_count"],
                "orphan_profile_row_count": counts["orphan_profile_row_count"],
                "missing_materialization_count": missing_materialization_counts["session_profile"],
            },
        ),
        "timeline_work_events": surface(
            ready=work_events_ready,
            blockers=work_events_blockers,
            evidence={
                "work_event_row_count": counts["work_event_row_count"],
                "missing_materialization_count": missing_materialization_counts["work_events"],
            },
        ),
        "timeline_phases": surface(
            ready=phases_ready,
            blockers=phases_blockers,
            evidence={
                "phase_row_count": counts["phase_row_count"],
                "missing_materialization_count": missing_materialization_counts["phases"],
            },
        ),
        "threads": surface(
            ready=thread_ready,
            blockers=thread_blockers,
            evidence={
                "thread_count": counts["thread_count"],
                "thread_session_count": counts["thread_session_count"],
                "missing_materialization_count": missing_materialization_counts["thread"],
            },
        ),
        "tag_rollups": surface(
            ready=True,
            blockers=[],
            evidence={"session_tag_count": counts["session_tag_count"]},
        ),
        "tool_usage": surface(
            ready=True,
            blockers=[],
            evidence={"action_count": counts["action_count"], "action_count_exact": counts["action_count_exact"]},
        ),
        "session_costs": surface(
            ready=profile_ready,
            blockers=list(profile_blockers),
            evidence={
                "cost_profile_count": counts["cost_profile_count"],
                "missing_profile_row_count": counts["missing_profile_row_count"],
                "missing_materialization_count": missing_materialization_counts["session_profile"],
            },
        ),
        "latency_profiles": surface(
            ready=latency_ready,
            blockers=latency_blockers,
            evidence={"missing_materialization_count": missing_materialization_counts["latency"]},
        ),
    }


def _archive_materialization_counts(conn: sqlite3.Connection) -> dict[str, int]:
    keys = ("session_profile", "work_events", "phases", "latency", "thread")
    rows = conn.execute(
        """
        SELECT insight_type, COUNT(*)
        FROM insight_materialization
        GROUP BY insight_type
        """
    ).fetchall()
    counts = dict.fromkeys(keys, 0)
    counts.update({str(row[0]): int(row[1] or 0) for row in rows})
    return counts


def _archive_missing_materialization_counts(conn: sqlite3.Connection) -> dict[str, int]:
    keys = ("session_profile", "work_events", "phases", "latency", "thread")
    return {
        key: _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM insight_materialization AS m
                WHERE m.insight_type = ? AND m.session_id = s.session_id
            )
            """,
            (key,),
        )
        for key in keys
    }


def _archive_user_overlay_orphans(root: Path) -> dict[str, Any]:
    """Count user-tier references that point at sessions absent from index.db."""

    user_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.USER].filename
    index_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].filename
    if not user_db.exists() or not index_db.exists():
        return {
            "checked": False,
            "reason": "missing_user_or_index_tier",
            "orphan_session_reference_counts": {},
            "total_orphan_session_references": 0,
        }
    conn = open_readonly_connection(user_db)
    try:
        conn.execute("ATTACH DATABASE ? AS index_tier", (f"file:{index_db}?mode=ro",))
        checks = {
            "assertion_marks": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'mark' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_annotations": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'annotation' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_corrections": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'correction' AND u.target_ref LIKE 'insight:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_suppressions": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'suppression' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_tags": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'tag' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_metadata": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'metadata' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
            "assertion_notes": (
                "SELECT COUNT(*) FROM assertions AS u "
                "WHERE u.kind = 'note' AND u.target_ref LIKE 'session:%' "
                "AND COALESCE(u.status, '') != 'deleted' "
                "AND NOT EXISTS (SELECT 1 FROM index_tier.sessions AS s WHERE s.session_id = substr(u.target_ref, 9))"
            ),
        }
        counts = {
            name: _scalar_int(conn, sql)
            if (name.startswith("assertion_") and _table_exists(conn, "assertions"))
            or (not name.startswith("assertion_") and _table_exists(conn, name))
            else -1
            for name, sql in checks.items()
        }
        total = sum(count for count in counts.values() if count > 0)
        return {
            "checked": True,
            "reason": None,
            "orphan_session_reference_counts": counts,
            "total_orphan_session_references": total,
        }
    except sqlite3.Error as exc:
        return {
            "checked": False,
            "reason": str(exc),
            "orphan_session_reference_counts": {},
            "total_orphan_session_references": 0,
        }
    finally:
        conn.close()


def _topology_quarantine_state(conn: sqlite3.Connection) -> dict[str, Any]:
    """Report the state of the session-link cycle quarantine.

    Surfaces:
    - ``table_present`` — is ``session_links`` materialized in the database
    - ``unresolved_count`` — links still awaiting their parent
    - ``resolved_count`` — links whose parent has been ingested
    - ``quarantined_count`` — links rejected because they would create a
      cycle in ``sessions.parent_session_id``
    - ``oldest_quarantined_at`` — oldest ``resolved_at_ms`` timestamp on a
      quarantined link (the field is repurposed as the
      "decision-recorded-at" timestamp for non-resolved terminal states)

    Operators monitor ``quarantined_count`` as a health indicator — a
    non-zero value means at least one source emitted a cycle and the
    fast-path graph was prevented from entering it.
    """

    if not _table_exists(conn, "session_links"):
        return {
            "table_present": False,
            "unresolved_count": 0,
            "resolved_count": 0,
            "quarantined_count": 0,
            "oldest_quarantined_at": None,
        }
    rows = conn.execute(
        """
        SELECT CASE
                   WHEN status IS NOT NULL THEN status
                   WHEN resolved_dst_session_id IS NOT NULL THEN 'resolved'
                   ELSE 'unresolved'
               END AS link_state,
               COUNT(*) AS n
          FROM session_links
         GROUP BY link_state
        """
    ).fetchall()
    status_counts = {str(row[0]): int(row[1]) for row in rows}
    oldest_row = conn.execute("SELECT MIN(resolved_at_ms) FROM session_links WHERE status = 'quarantined'").fetchone()
    return {
        "table_present": True,
        "unresolved_count": int(status_counts.get("unresolved", 0)),
        "resolved_count": int(status_counts.get("resolved", 0)),
        "quarantined_count": int(status_counts.get("quarantined", 0)),
        "oldest_quarantined_at": oldest_row[0] if oldest_row else None,
    }


def _blob_lease_state(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "pending_blob_refs"):
        return {
            "table_present": False,
            "pending_lease_count": 0,
            "distinct_operations": 0,
            "oldest_acquired_at": None,
        }
    pending = _scalar_int(conn, "SELECT COUNT(*) FROM pending_blob_refs")
    operations = _scalar_int(conn, "SELECT COUNT(DISTINCT operation_id) FROM pending_blob_refs")
    oldest_row = conn.execute("SELECT MIN(acquired_at_ms) FROM pending_blob_refs").fetchone()
    oldest = int(oldest_row[0]) if oldest_row is not None and oldest_row[0] is not None else None
    return {
        "table_present": True,
        "pending_lease_count": pending,
        "distinct_operations": operations,
        "oldest_acquired_at": oldest,
    }


def _gc_state(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "gc_generations"):
        return {
            "table_present": False,
            "high_water_generation": 0,
            "last_completed_at": None,
            "generation_count": 0,
        }
    row = conn.execute(
        """
        SELECT
            COALESCE(MAX(CAST(generation_id AS INTEGER)), 0) AS high_water,
            COUNT(*) AS generation_count
        FROM gc_generations
        """
    ).fetchone()
    high_water = int(row[0] or 0) if row is not None else 0
    generation_count = int(row[1] or 0) if row is not None else 0
    completed_row = conn.execute(
        "SELECT completed_at_ms FROM gc_generations WHERE generation_id = ? LIMIT 1",
        (str(high_water),),
    ).fetchone()
    completed_at = int(completed_row[0]) if completed_row is not None and completed_row[0] is not None else None
    return {
        "table_present": True,
        "high_water_generation": high_water,
        "last_completed_at": completed_at,
        "generation_count": generation_count,
    }


def _tier_state_or_current(conn: sqlite3.Connection, tier_db: Path, reader: Any) -> dict[str, Any]:
    if tier_db.exists():
        try:
            tier_conn = open_readonly_connection(tier_db)
            try:
                return dict(reader(tier_conn))
            finally:
                tier_conn.close()
        except sqlite3.Error:
            pass
    return dict(reader(conn))


def _fts_trigger_state(conn: sqlite3.Connection) -> dict[str, Any]:
    present = sorted(
        str(row[0])
        for row in conn.execute(
            f"""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN ({",".join("?" for _ in _EXPECTED_FTS_TRIGGERS)})
            """,
            _EXPECTED_FTS_TRIGGERS,
        ).fetchall()
    )
    expected = set(_EXPECTED_FTS_TRIGGERS)
    missing = sorted(expected - set(present))
    return {
        "expected": list(_EXPECTED_FTS_TRIGGERS),
        "present": present,
        "missing": missing,
        "all_present": not missing,
    }


def _convergence_stage_timings(
    attempts: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    completed = [a for a in attempts if a.get("status") == "completed"]
    if not completed:
        return {
            "sample_size": 0,
            "parse_time_s": _summary_stat([]),
            "convergence_time_s": _summary_stat([]),
            "read_amplification": _summary_stat([]),
            "per_stage_s": {},
        }
    parse_times = [float(a.get("parse_time_s") or 0.0) for a in completed]
    convergence_times = [float(a.get("convergence_time_s") or 0.0) for a in completed]
    amps = [float(a.get("read_amplification") or 0.0) for a in completed]
    return {
        "sample_size": len(completed),
        "parse_time_s": _summary_stat(parse_times),
        "convergence_time_s": _summary_stat(convergence_times),
        "read_amplification": _summary_stat(amps),
        "per_stage_s": _per_stage_timings(
            [str(a.get("attempt_id")) for a in completed if a.get("attempt_id")],
            conn,
        ),
    }


def _per_stage_timings(
    attempt_ids: list[str],
    conn: sqlite3.Connection | None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-stage timings from ``live_ingest_stage_event``.

    Returns ``{stage_name: summary_stat}`` over the latest ``stage_timings_json``
    payload for each attempt.  Returns ``{}`` when the table does not exist or
    no completed attempts carry timings.
    """
    if conn is None or not attempt_ids:
        return {}
    if not _table_exists(conn, "live_ingest_stage_event"):
        return {}
    placeholders = ",".join("?" for _ in attempt_ids)
    rows = conn.execute(
        f"""
        SELECT attempt_id, stage_timings_json
        FROM live_ingest_stage_event
        WHERE attempt_id IN ({placeholders})
          AND stage_timings_json IS NOT NULL
        ORDER BY attempt_id, sequence DESC
        """,
        tuple(attempt_ids),
    ).fetchall()
    latest_by_attempt: dict[str, str] = {}
    for row in rows:
        attempt_id = str(row[0])
        if attempt_id in latest_by_attempt:
            continue
        if isinstance(row[1], str) and row[1]:
            latest_by_attempt[attempt_id] = row[1]
    per_stage: dict[str, list[float]] = {}
    for payload in latest_by_attempt.values():
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        for stage_name, value in decoded.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            per_stage.setdefault(str(stage_name), []).append(float(value))
    return {stage: _summary_stat(values) for stage, values in sorted(per_stage.items())}


def _summary_stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "sum": 0.0, "mean": 0.0}
    total = sum(values)
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "sum": round(total, 6),
        "mean": round(total / len(values), 6),
    }


def _daemon_resource_signal(
    attempts: list[dict[str, Any]],
    conn: sqlite3.Connection,
    *,
    ops_db: Path | None = None,
) -> dict[str, Any]:
    """Read the most recent RSS / cgroup snapshot from `live_ingest_attempt`.

    These are the only daemon-RSS signals the probe can read without IPC.
    The probe is intentionally read-only.
    """

    ops_signal = _ops_daemon_resource_signal(ops_db)
    if ops_signal is not None:
        return ops_signal
    if not _table_exists(conn, "live_ingest_attempt"):
        return {"available": False}
    columns = _columns(conn, "live_ingest_attempt")
    optional = (
        "rss_current_mb",
        "rss_peak_self_mb",
        "rss_peak_children_mb",
        "cgroup_memory_current_mb",
        "cgroup_memory_peak_mb",
        "cgroup_memory_swap_current_mb",
        "cgroup_memory_anon_mb",
        "cgroup_memory_file_mb",
        "cgroup_memory_inactive_file_mb",
        "worker_in_flight_count",
        "worker_completed_count",
        "worker_total_count",
    )
    available_columns = [name for name in optional if name in columns]
    if not available_columns:
        return {"available": False}
    select_list = ", ".join(available_columns)
    row = conn.execute(
        f"""
        SELECT {select_list}
        FROM live_ingest_attempt
        ORDER BY updated_at DESC, started_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return {"available": False}
    return {"available": True, **{col: row[idx] for idx, col in enumerate(available_columns)}}


def _ops_daemon_resource_signal(ops_db: Path | None) -> dict[str, Any] | None:
    if ops_db is None or not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "daemon_stage_events"):
                return None
            row = conn.execute(
                """
                SELECT payload_json
                FROM daemon_stage_events
                ORDER BY observed_at_ms DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        payload = json.loads(str(row[0] or "{}"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    fields = {
        key: payload[key]
        for key in (
            "rss_current_mb",
            "rss_peak_self_mb",
            "rss_peak_children_mb",
            "cgroup_memory_current_mb",
            "cgroup_memory_peak_mb",
            "cgroup_memory_swap_current_mb",
            "cgroup_memory_anon_mb",
            "cgroup_memory_file_mb",
            "cgroup_memory_inactive_file_mb",
            "worker_in_flight_count",
            "worker_completed_count",
            "worker_total_count",
        )
        if key in payload
    }
    return {"available": bool(fields), **fields}


def _explain(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> list[str]:
    try:
        return [str(row[3]) for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()]
    except sqlite3.Error as exc:
        return [f"unavailable: {exc}"]


def _query_plans(conn: sqlite3.Connection, *, db: Path) -> dict[str, Any]:
    plans: dict[str, Any] = _archive_query_plans(db.parent)
    if _table_exists(conn, "raw_sessions") and _table_exists(conn, "sessions"):
        source_row = conn.execute(
            "SELECT source_path FROM raw_sessions WHERE source_path IS NOT NULL LIMIT 1"
        ).fetchone()
        if source_row is not None:
            plan = _explain(
                conn,
                """
                SELECT DISTINCT r.source_path, c.session_id
                FROM raw_sessions AS r
                JOIN sessions AS c ON c.raw_id = r.raw_id
                WHERE r.source_path IN (?)
                ORDER BY r.source_path, c.session_id
                """,
                (source_row[0],),
            )
            plans["source_path_lookup"] = {
                "plan": plan,
                "hazards": [item for item in plan if "SCAN c" in item],
                "storage_route": "archive_file_set",
            }
    if _table_exists(conn, "blocks") and _table_exists(conn, "messages_fts_docsize"):
        conv_row = conn.execute("SELECT session_id FROM blocks LIMIT 1").fetchone()
        if conv_row is not None:
            plan = _explain(
                conn,
                """
                SELECT COUNT(*)
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                WHERE b.search_text != ''
                  AND b.session_id IN (?)
                  AND d.id IS NULL
                """,
                (conv_row[0],),
            )
            plans["message_fts_gap_probe"] = {
                "plan": plan,
                "hazards": [item for item in plan if "messages_fts " in item],
            }
    return plans


def _archive_query_plans(root: Path) -> dict[str, Any]:
    source_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    index_db = root / ARCHIVE_TIER_SPECS[ArchiveTier.INDEX].filename
    if not source_db.exists() or not index_db.exists():
        return {}
    plans: dict[str, Any] = {}
    try:
        conn = open_readonly_connection(index_db)
        try:
            conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
            if _table_exists(conn, "sessions") and _attached_table_exists(conn, "source_tier", "raw_sessions"):
                source_row = conn.execute(
                    "SELECT source_path FROM source_tier.raw_sessions WHERE source_path IS NOT NULL LIMIT 1"
                ).fetchone()
                if source_row is not None:
                    plan = _explain(
                        conn,
                        """
                        SELECT DISTINCT r.source_path, s.session_id
                        FROM source_tier.raw_sessions AS r
                        JOIN sessions AS s ON s.raw_id = r.raw_id
                        WHERE r.source_path IN (?)
                        ORDER BY r.source_path, s.session_id
                        """,
                        (source_row[0],),
                    )
                    plans["source_path_lookup"] = {
                        "plan": plan,
                        "hazards": [item for item in plan if "SCAN s" in item],
                        "storage_route": "archive_file_set",
                    }
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return plans


def probe(
    db: Path,
    *,
    limit: int = 5,
    integrity_check: bool = False,
    exact_derived_counts: bool = False,
    exact_table_counts: bool = False,
) -> dict[str, Any]:
    ops_db = db.with_name("ops.db")
    index_db = db.with_name("index.db")
    if not db.exists() and not index_db.exists() and not ops_db.exists():
        return {
            "ok": False,
            "report_version": REPORT_VERSION,
            "captured_at": _now_iso(),
            "db_path": str(db),
            "error": "database does not exist",
        }
    observed_db: Path | None
    if db.exists():
        observed_db = db
        conn = open_readonly_connection(db)
    elif index_db.exists():
        observed_db = index_db
        conn = open_readonly_connection(index_db)
    else:
        observed_db = None
        conn = sqlite3.connect(":memory:")
    try:
        recent_attempts = _recent_attempts(conn, limit=limit, ops_db=ops_db)
        return {
            "ok": True,
            "report_version": REPORT_VERSION,
            "captured_at": _now_iso(),
            "db_path": str(db),
            "observability_locations": _observability_locations(db, observed_db=observed_db, ops_db=ops_db),
            "attempt_counts": _attempt_counts(conn, ops_db=ops_db),
            "recent_attempts": recent_attempts,
            "storage_route_counts": _storage_route_counts(conn, ops_db=ops_db),
            "convergence_stage_timings": _convergence_stage_timings(recent_attempts, conn),
            "boundary_table_count_mode": "exact" if exact_table_counts else "estimated",
            "boundary_table_counts": _boundary_table_counts(
                conn,
                ops_db=ops_db,
                source_db=db.with_name("source.db"),
                exact=exact_table_counts,
            ),
            "archive_tiers": _archive_tier_state(
                db,
                observed_db=observed_db,
                integrity_check=integrity_check,
                exact_derived_counts=exact_derived_counts,
                exact_table_counts=exact_table_counts,
            ),
            "topology_quarantine_state": _topology_quarantine_state(conn),
            "blob_lease_state": _tier_state_or_current(conn, db.with_name("source.db"), _blob_lease_state),
            "gc_state": _tier_state_or_current(conn, db.with_name("source.db"), _gc_state),
            "fts_trigger_state": _fts_trigger_state(conn),
            "daemon_resource_signal": _daemon_resource_signal(recent_attempts, conn, ops_db=ops_db),
            "source_path_churn": _source_path_churn(conn, attempts=recent_attempts, limit=limit, db=db),
            "convergence_debt": _convergence_debt(conn, ops_db=ops_db),
            "cursor_lag_baselines": _cursor_lag_baselines(conn, ops_db=ops_db),
            "query_plans": _query_plans(conn, db=db),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Produce a structured before/after diff of two probe reports.

    The diff is keyed by canonical sections.  Each section's value contains
    enough state to be read on its own (no implicit ordering across sections).
    """

    before_version = int(before.get("report_version", 0))
    after_version = int(after.get("report_version", 0))
    if before_version != after_version:
        return {
            "ok": False,
            "error": (
                f"report_version mismatch: before={before_version} after={after_version}; "
                "regenerate both reports with the same devtools build"
            ),
            "before_version": before_version,
            "after_version": after_version,
        }
    if not before.get("ok") or not after.get("ok"):
        return {
            "ok": False,
            "error": "compare requires two successful probe reports",
            "before_ok": bool(before.get("ok")),
            "after_ok": bool(after.get("ok")),
        }

    before_counts = _coerce_int_map(before.get("attempt_counts") or {})
    after_counts = _coerce_int_map(after.get("attempt_counts") or {})
    attempt_delta = {
        key: {
            "before": before_counts.get(key, 0),
            "after": after_counts.get(key, 0),
            "delta": after_counts.get(key, 0) - before_counts.get(key, 0),
        }
        for key in sorted(set(before_counts) | set(after_counts))
    }

    before_tables = _coerce_int_map(before.get("boundary_table_counts") or {})
    after_tables = _coerce_int_map(after.get("boundary_table_counts") or {})
    table_delta = {
        key: {
            "before": before_tables.get(key, 0),
            "after": after_tables.get(key, 0),
            "delta": after_tables.get(key, 0) - before_tables.get(key, 0),
        }
        for key in sorted(set(before_tables) | set(after_tables))
    }

    before_routes = _coerce_int_map(before.get("storage_route_counts") or {})
    after_routes = _coerce_int_map(after.get("storage_route_counts") or {})
    route_delta = _int_map_delta(before_routes, after_routes)

    before_leases = before.get("blob_lease_state") or {}
    after_leases = after.get("blob_lease_state") or {}
    lease_delta = {
        "pending_lease_count": {
            "before": int(before_leases.get("pending_lease_count") or 0),
            "after": int(after_leases.get("pending_lease_count") or 0),
            "delta": int(after_leases.get("pending_lease_count") or 0)
            - int(before_leases.get("pending_lease_count") or 0),
        },
        "distinct_operations": {
            "before": int(before_leases.get("distinct_operations") or 0),
            "after": int(after_leases.get("distinct_operations") or 0),
            "delta": int(after_leases.get("distinct_operations") or 0)
            - int(before_leases.get("distinct_operations") or 0),
        },
    }

    before_gc = before.get("gc_state") or {}
    after_gc = after.get("gc_state") or {}
    gc_delta = {
        "high_water_generation": {
            "before": int(before_gc.get("high_water_generation") or 0),
            "after": int(after_gc.get("high_water_generation") or 0),
            "delta": int(after_gc.get("high_water_generation") or 0) - int(before_gc.get("high_water_generation") or 0),
        },
        "generation_count": {
            "before": int(before_gc.get("generation_count") or 0),
            "after": int(after_gc.get("generation_count") or 0),
            "delta": int(after_gc.get("generation_count") or 0) - int(before_gc.get("generation_count") or 0),
        },
    }

    before_fts = before.get("fts_trigger_state") or {}
    after_fts = after.get("fts_trigger_state") or {}
    fts_delta = {
        "all_present_before": bool(before_fts.get("all_present")),
        "all_present_after": bool(after_fts.get("all_present")),
        "missing_before": list(before_fts.get("missing") or []),
        "missing_after": list(after_fts.get("missing") or []),
        "regressed": sorted(set(after_fts.get("missing") or []) - set(before_fts.get("missing") or [])),
        "restored": sorted(set(before_fts.get("missing") or []) - set(after_fts.get("missing") or [])),
    }

    before_debt = before.get("convergence_debt") or {}
    after_debt = after.get("convergence_debt") or {}
    debt_delta = {
        "failed_count": {
            "before": int(before_debt.get("failed_count") or 0),
            "after": int(after_debt.get("failed_count") or 0),
            "delta": int(after_debt.get("failed_count") or 0) - int(before_debt.get("failed_count") or 0),
        }
    }

    before_timings = before.get("convergence_stage_timings") or {}
    after_timings = after.get("convergence_stage_timings") or {}
    timing_delta: dict[str, Any] = {
        "sample_size_before": int(before_timings.get("sample_size") or 0),
        "sample_size_after": int(after_timings.get("sample_size") or 0),
    }
    for field in ("parse_time_s", "convergence_time_s", "read_amplification"):
        b = before_timings.get(field) or {}
        a = after_timings.get(field) or {}
        timing_delta[field] = {
            "mean": {
                "before": float(b.get("mean") or 0.0),
                "after": float(a.get("mean") or 0.0),
                "delta": float(a.get("mean") or 0.0) - float(b.get("mean") or 0.0),
            },
            "max": {
                "before": float(b.get("max") or 0.0),
                "after": float(a.get("max") or 0.0),
                "delta": float(a.get("max") or 0.0) - float(b.get("max") or 0.0),
            },
        }

    return {
        "ok": True,
        "report_version": REPORT_VERSION,
        "before_captured_at": before.get("captured_at"),
        "after_captured_at": after.get("captured_at"),
        "before_db_path": before.get("db_path"),
        "after_db_path": after.get("db_path"),
        "attempt_counts": attempt_delta,
        "storage_route_counts": route_delta,
        "boundary_table_counts": table_delta,
        "archive_tiers": _archive_tier_delta(before, after),
        "source_path_churn": _source_path_churn_delta(before, after),
        "blob_lease_state": lease_delta,
        "gc_state": gc_delta,
        "fts_trigger_state": fts_delta,
        "convergence_debt": debt_delta,
        "convergence_stage_timings": timing_delta,
    }


def _coerce_int_map(source: dict[str, Any]) -> dict[str, int]:
    """Pull out ``int``-valued entries from a mixed-shape dict.

    The probe stores integer counts alongside list/dict context (for example
    ``attempt_counts.overlapping_source_paths``).  Compare only diffs the
    scalar integer fields; non-int values are skipped so the diff stays
    arithmetic.
    """

    out: dict[str, int] = {}
    for key, value in source.items():
        if isinstance(value, bool):  # bool is int-subclass; reject explicitly
            continue
        if isinstance(value, int):
            out[key] = value
    return out


def _archive_tier_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_state = before.get("archive_tiers") or {}
    after_state = after.get("archive_tiers") or {}
    before_tiers = before_state.get("tiers") or {}
    after_tiers = after_state.get("tiers") or {}
    tier_names = sorted(set(before_tiers) | set(after_tiers))
    tiers: dict[str, Any] = {}
    for tier in tier_names:
        before_tier = before_tiers.get(tier) or {}
        after_tier = after_tiers.get(tier) or {}
        before_counts = _coerce_int_map(before_tier.get("table_counts") or {})
        after_counts = _coerce_int_map(after_tier.get("table_counts") or {})
        table_deltas = {
            table: {
                "before": before_counts.get(table, 0),
                "after": after_counts.get(table, 0),
                "delta": after_counts.get(table, 0) - before_counts.get(table, 0),
            }
            for table in sorted(set(before_counts) | set(after_counts))
        }
        tiers[tier] = {
            "exists_before": bool(before_tier.get("exists")),
            "exists_after": bool(after_tier.get("exists")),
            "user_version_before": before_tier.get("user_version"),
            "user_version_after": after_tier.get("user_version"),
            "integrity_before": before_tier.get("integrity"),
            "integrity_after": after_tier.get("integrity"),
            "size_bytes_before": before_tier.get("size_bytes"),
            "size_bytes_after": after_tier.get("size_bytes"),
            "table_counts": table_deltas,
        }
    return {
        "present_before": bool(before_state.get("present")),
        "present_after": bool(after_state.get("present")),
        "complete_before": bool(before_state.get("complete")),
        "complete_after": bool(after_state.get("complete")),
        "present_count_before": int(before_state.get("present_count") or 0),
        "present_count_after": int(after_state.get("present_count") or 0),
        "observed_tier_before": before_state.get("observed_tier"),
        "observed_tier_after": after_state.get("observed_tier"),
        "schema_mismatches_before": list(before_state.get("schema_mismatches") or []),
        "schema_mismatches_after": list(after_state.get("schema_mismatches") or []),
        "missing_backup_required_before": list(before_state.get("missing_backup_required") or []),
        "missing_backup_required_after": list(after_state.get("missing_backup_required") or []),
        "layout_readiness": _archive_layout_readiness_delta(before_state, after_state),
        "derived_readiness": _archive_derived_readiness_delta(before_state, after_state),
        "user_overlay_orphans": _archive_user_overlay_orphan_delta(before_state, after_state),
        "tiers": tiers,
    }


def _archive_layout_readiness_delta(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    before = before_state.get("layout_readiness") or {}
    after = after_state.get("layout_readiness") or {}
    before_blockers = list(before.get("blockers") or [])
    after_blockers = list(after.get("blockers") or [])
    return {
        "state_before": before.get("state"),
        "state_after": after.get("state"),
        "archive_ready_before": bool(before.get("archive_ready")),
        "archive_ready_after": bool(after.get("archive_ready")),
        "blockers_before": before_blockers,
        "blockers_after": after_blockers,
        "introduced_blockers": sorted(set(after_blockers) - set(before_blockers)),
        "resolved_blockers": sorted(set(before_blockers) - set(after_blockers)),
        "evidence": _int_map_delta(
            _coerce_int_map(before.get("evidence") or {}),
            _coerce_int_map(after.get("evidence") or {}),
        ),
    }


def _archive_derived_readiness_delta(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    before_readiness = before_state.get("derived_readiness") or {}
    after_readiness = after_state.get("derived_readiness") or {}
    before_counts = _coerce_int_map(before_readiness.get("counts") or {})
    after_counts = _coerce_int_map(after_readiness.get("counts") or {})
    before_materialized = _coerce_int_map(before_readiness.get("materialization_counts") or {})
    after_materialized = _coerce_int_map(after_readiness.get("materialization_counts") or {})
    before_missing = _coerce_int_map(before_readiness.get("missing_materialization_counts") or {})
    after_missing = _coerce_int_map(after_readiness.get("missing_materialization_counts") or {})
    before_surfaces = before_readiness.get("surface_readiness") or {}
    after_surfaces = after_readiness.get("surface_readiness") or {}
    return {
        "checked_before": bool(before_readiness.get("checked")),
        "checked_after": bool(after_readiness.get("checked")),
        "source_check_available_before": bool(before_readiness.get("source_check_available")),
        "source_check_available_after": bool(after_readiness.get("source_check_available")),
        "counts": _int_map_delta(before_counts, after_counts),
        "materialization_counts": _int_map_delta(before_materialized, after_materialized),
        "missing_materialization_counts": _int_map_delta(before_missing, after_missing),
        "ready_before": dict(before_readiness.get("ready") or {}),
        "ready_after": dict(after_readiness.get("ready") or {}),
        "surface_readiness": _archive_surface_readiness_delta(before_surfaces, after_surfaces),
    }


def _archive_surface_readiness_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    surfaces: dict[str, Any] = {}
    for surface_name in sorted(set(before) | set(after)):
        before_surface = before.get(surface_name) or {}
        after_surface = after.get(surface_name) or {}
        surfaces[surface_name] = {
            "ready_before": before_surface.get("ready"),
            "ready_after": after_surface.get("ready"),
            "blockers_before": list(before_surface.get("blockers") or []),
            "blockers_after": list(after_surface.get("blockers") or []),
            "evidence": _int_map_delta(
                _coerce_int_map(before_surface.get("evidence") or {}),
                _coerce_int_map(after_surface.get("evidence") or {}),
            ),
        }
    return surfaces


def _int_map_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        key: {
            "before": before.get(key, 0),
            "after": after.get(key, 0),
            "delta": after.get(key, 0) - before.get(key, 0),
        }
        for key in sorted(set(before) | set(after))
    }


def _source_path_churn_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_items = _source_path_churn_by_path(before.get("source_path_churn") or [])
    after_items = _source_path_churn_by_path(after.get("source_path_churn") or [])
    fields = (
        "raw_count",
        "full_raw_count",
        "append_raw_count",
        "session_count",
        "materialized_raw_count",
        "orphan_raw_count",
        "total_blob_bytes",
    )
    paths: dict[str, Any] = {}
    for source_path in sorted(set(before_items) | set(after_items)):
        before_item = before_items.get(source_path) or {}
        after_item = after_items.get(source_path) or {}
        paths[source_path] = {
            "storage_route_before": before_item.get("storage_route"),
            "storage_route_after": after_item.get("storage_route"),
            "latest_acquired_at_before": before_item.get("latest_acquired_at"),
            "latest_acquired_at_after": after_item.get("latest_acquired_at"),
            "counts": {
                field: {
                    "before": int(before_item.get(field) or 0),
                    "after": int(after_item.get(field) or 0),
                    "delta": int(after_item.get(field) or 0) - int(before_item.get(field) or 0),
                }
                for field in fields
            },
        }
    return {
        "path_count_before": len(before_items),
        "path_count_after": len(after_items),
        "paths": paths,
    }


def _source_path_churn_by_path(items: object) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    by_path: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        source_path = item.get("source_path")
        if isinstance(source_path, str) and source_path:
            by_path[source_path] = item
    return by_path


def _archive_user_overlay_orphan_delta(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, Any]:
    before_orphans = before_state.get("user_overlay_orphans") or {}
    after_orphans = after_state.get("user_overlay_orphans") or {}
    before_counts = _coerce_int_map(before_orphans.get("orphan_session_reference_counts") or {})
    after_counts = _coerce_int_map(after_orphans.get("orphan_session_reference_counts") or {})
    table_deltas = {
        table: {
            "before": before_counts.get(table, 0),
            "after": after_counts.get(table, 0),
            "delta": after_counts.get(table, 0) - before_counts.get(table, 0),
        }
        for table in sorted(set(before_counts) | set(after_counts))
    }
    before_total = int(before_orphans.get("total_orphan_session_references") or 0)
    after_total = int(after_orphans.get("total_orphan_session_references") or 0)
    return {
        "checked_before": bool(before_orphans.get("checked")),
        "checked_after": bool(after_orphans.get("checked")),
        "reason_before": before_orphans.get("reason"),
        "reason_after": after_orphans.get("reason"),
        "total": {"before": before_total, "after": after_total, "delta": after_total - before_total},
        "tables": table_deltas,
    }


def _format_compare_human(diff: dict[str, Any]) -> str:
    if not diff.get("ok"):
        return f"compare failed: {diff.get('error')}"
    lines: list[str] = []
    lines.append("Daemon workload probe — before/after diff")
    lines.append(f"  before: {diff.get('before_captured_at')} ({diff.get('before_db_path')})")
    lines.append(f"  after:  {diff.get('after_captured_at')} ({diff.get('after_db_path')})")
    lines.append("")
    lines.append("Attempt counts:")
    for key, entry in diff["attempt_counts"].items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    lines.append("")
    lines.append("Boundary table counts:")
    for key, entry in diff["boundary_table_counts"].items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    archive_tiers = diff.get("archive_tiers") or {}
    lines.append("")
    lines.append(
        "Archive tiers: "
        f"{archive_tiers.get('present_count_before', 0)} -> {archive_tiers.get('present_count_after', 0)} present, "
        f"complete {archive_tiers.get('complete_before')} -> {archive_tiers.get('complete_after')}, "
        f"observed {archive_tiers.get('observed_tier_before')} -> {archive_tiers.get('observed_tier_after')}"
    )
    mismatches = archive_tiers.get("schema_mismatches_after") or []
    missing_required = archive_tiers.get("missing_backup_required_after") or []
    if mismatches:
        lines.append(f"  schema mismatches after: {', '.join(mismatches)}")
    if missing_required:
        lines.append(f"  missing backup-required tiers after: {', '.join(missing_required)}")
    layout = archive_tiers.get("layout_readiness") or {}
    if layout:
        blockers_after = layout.get("blockers_after") or []
        lines.append(
            "  archive layout: "
            f"{layout.get('state_before')} -> {layout.get('state_after')}; "
            f"ready {layout.get('archive_ready_before')} -> {layout.get('archive_ready_after')}"
        )
        if blockers_after:
            lines.append(f"  archive layout blockers after: {', '.join(blockers_after[:8])}")
    readiness = archive_tiers.get("derived_readiness") or {}
    surface_deltas = readiness.get("surface_readiness") or {}
    changed_surfaces = [
        name
        for name, info in surface_deltas.items()
        if info.get("ready_before") != info.get("ready_after")
        or info.get("blockers_before") != info.get("blockers_after")
    ]
    if changed_surfaces:
        lines.append("  surface readiness changes:")
        for name in changed_surfaces:
            info = surface_deltas[name]
            blockers = info.get("blockers_after") or []
            blocker_text = ", ".join(blockers) if blockers else "none"
            lines.append(
                f"    {name}: ready {info.get('ready_before')} -> {info.get('ready_after')}; "
                f"blockers after: {blocker_text}"
            )
    for tier, info in (archive_tiers.get("tiers") or {}).items():
        if info.get("exists_before") != info.get("exists_after") or info.get("user_version_before") != info.get(
            "user_version_after"
        ):
            lines.append(
                f"  {tier}: exists {info.get('exists_before')} -> {info.get('exists_after')}, "
                f"user_version {info.get('user_version_before')} -> {info.get('user_version_after')}"
            )
    leases = diff["blob_lease_state"]
    lines.append("")
    lines.append("Blob lease state:")
    for key, entry in leases.items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    gc = diff["gc_state"]
    lines.append("")
    lines.append("GC state:")
    for key, entry in gc.items():
        lines.append(f"  {key}: {entry['before']} -> {entry['after']} (Δ {entry['delta']:+d})")
    fts = diff["fts_trigger_state"]
    lines.append("")
    lines.append("FTS trigger state:")
    lines.append(f"  all_present: {fts['all_present_before']} -> {fts['all_present_after']}")
    if fts["regressed"]:
        lines.append(f"  regressed (newly missing): {', '.join(fts['regressed'])}")
    if fts["restored"]:
        lines.append(f"  restored: {', '.join(fts['restored'])}")
    if not fts["regressed"] and not fts["restored"]:
        lines.append("  no trigger drift")
    debt = diff["convergence_debt"]["failed_count"]
    lines.append("")
    lines.append(f"Convergence debt failed_count: {debt['before']} -> {debt['after']} (Δ {debt['delta']:+d})")
    timings = diff["convergence_stage_timings"]
    lines.append("")
    lines.append(
        f"Convergence stage timings (completed-attempt sample {timings['sample_size_before']} -> "
        f"{timings['sample_size_after']}):"
    )
    for field in ("parse_time_s", "convergence_time_s", "read_amplification"):
        entry = timings[field]
        mean = entry["mean"]
        max_ = entry["max"]
        lines.append(f"  {field}.mean: {mean['before']:.3f} -> {mean['after']:.3f} (Δ {mean['delta']:+.3f})")
        lines.append(f"  {field}.max:  {max_['before']:.3f} -> {max_['after']:.3f} (Δ {max_['delta']:+.3f})")
    churn = diff.get("source_path_churn") or {}
    churn_paths = churn.get("paths") or {}
    lines.append("")
    lines.append(
        f"Source path churn: {churn.get('path_count_before', 0)} -> {churn.get('path_count_after', 0)} hot paths"
    )
    changed_churn = [
        (path, info)
        for path, info in churn_paths.items()
        if any((entry.get("delta") or 0) != 0 for entry in (info.get("counts") or {}).values())
    ]
    for path, info in sorted(
        changed_churn,
        key=lambda item: abs(int(((item[1].get("counts") or {}).get("orphan_raw_count") or {}).get("delta") or 0)),
        reverse=True,
    )[:5]:
        counts = info.get("counts") or {}
        raw = counts.get("raw_count") or {}
        orphan = counts.get("orphan_raw_count") or {}
        materialized = counts.get("materialized_raw_count") or {}
        lines.append(
            f"  {path}: raw {raw.get('before', 0)} -> {raw.get('after', 0)} (Δ {raw.get('delta', 0):+d}), "
            f"materialized {materialized.get('before', 0)} -> {materialized.get('after', 0)} "
            f"(Δ {materialized.get('delta', 0):+d}), "
            f"orphan {orphan.get('before', 0)} -> {orphan.get('after', 0)} (Δ {orphan.get('delta', 0):+d})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Archive SQLite database path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--limit", type=int, default=5, help="Recent attempt limit")
    parser.add_argument(
        "--integrity-check",
        action="store_true",
        help="Run PRAGMA quick_check for each archive tier (can be expensive on large archives)",
    )
    parser.add_argument(
        "--exact-derived-counts",
        action="store_true",
        help="Run exact derived-readiness reconciliation counts (can scan large archive tables)",
    )
    parser.add_argument(
        "--exact-table-counts",
        action="store_true",
        help="Run exact boundary/archive table counts instead of cheap planner estimates",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        type=Path,
        default=None,
        help="Compare two saved probe reports and report structured deltas",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compare is not None:
        before_path, after_path = args.compare
        try:
            before_payload = json.loads(Path(before_path).read_text())
            after_payload = json.loads(Path(after_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"failed to read compare inputs: {exc}")
            return 2
        diff = compare(before_payload, after_payload)
        if args.json:
            print(json.dumps(diff, indent=2, sort_keys=True))
        else:
            print(_format_compare_human(diff))
        return 0 if diff.get("ok") else 1

    payload = probe(
        args.db or active_index_db_path(),
        limit=max(1, args.limit),
        integrity_check=args.integrity_check,
        exact_derived_counts=args.exact_derived_counts,
        exact_table_counts=args.exact_table_counts,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("ok") else 1

    print(f"Daemon workload probe: {payload.get('db_path')}")
    print(f"  captured_at: {payload.get('captured_at')}")
    if not payload.get("ok"):
        print(f"  error: {payload.get('error')}")
        return 1
    counts = payload["attempt_counts"]
    print(f"  attempts: {counts['total']} total, {counts['running']} running, {counts['failed']} failed")
    print(f"  stale cursor writes: {counts.get('stale_cursor_writes', 0)}")
    overlaps = counts.get("overlapping_source_paths") or []
    print(f"  overlapping source paths: {len(overlaps)}")
    table_counts = payload.get("boundary_table_counts") or {}
    if table_counts:
        print("  boundary table counts:")
        for table, count in table_counts.items():
            if count == -1:
                shown = "missing"
            elif count == UNKNOWN_TABLE_COUNT:
                shown = "unknown"
            else:
                shown = str(count)
            print(f"    {table}: {shown}")
    archive_tiers = payload.get("archive_tiers") or {}
    if archive_tiers.get("present"):
        observed = archive_tiers.get("observed_tier") or "none"
        print(
            "  archive tiers: "
            f"{archive_tiers.get('present_count', 0)}/{archive_tiers.get('expected_count', 0)} present, "
            f"complete={archive_tiers.get('complete')}, observed={observed}"
        )
        layout = archive_tiers.get("layout_readiness") or {}
        if layout:
            blockers = layout.get("blockers") or []
            print(f"    archive layout: {layout.get('state')} ({len(blockers)} blocker(s))")
            for blocker in blockers[:8]:
                print(f"      - {blocker}")
        orphans = archive_tiers.get("user_overlay_orphans") or {}
        if orphans.get("checked"):
            print(f"    user overlay orphans: {orphans.get('total_orphan_session_references', 0)} session references")
        readiness = archive_tiers.get("derived_readiness") or {}
        if readiness.get("checked"):
            counts = readiness.get("counts") or {}
            ready = readiness.get("ready") or {}
            print(
                "    derived readiness: "
                f"sessions={counts.get('session_count', 0)} "
                f"missing_profiles={counts.get('missing_profile_row_count', 0)} "
                f"missing_raw={counts.get('missing_raw_session_count', 0)} "
                f"messages_fts_ready={ready.get('messages_fts_ready')}"
            )
            blocked_surfaces = {
                name: info
                for name, info in (readiness.get("surface_readiness") or {}).items()
                if info.get("ready") is not True
            }
            if blocked_surfaces:
                print("    blocked archive surfaces:")
                for name, info in blocked_surfaces.items():
                    blockers = ", ".join(info.get("blockers") or ["unknown"])
                    print(f"      {name}: ready={info.get('ready')} blockers={blockers}")
        for tier, info in (archive_tiers.get("tiers") or {}).items():
            if not info.get("exists"):
                continue
            print(
                f"    {tier}: user_version={info.get('user_version')} "
                f"integrity={info.get('integrity')} size={info.get('size_bytes')} bytes"
            )
    leases = payload.get("blob_lease_state") or {}
    print(
        "  blob leases: "
        f"{leases.get('pending_lease_count', 0)} pending across "
        f"{leases.get('distinct_operations', 0)} operations"
    )
    gc = payload.get("gc_state") or {}
    print(f"  gc: high-water generation {gc.get('high_water_generation', 0)} (count {gc.get('generation_count', 0)})")
    fts = payload.get("fts_trigger_state") or {}
    if fts.get("all_present"):
        print(f"  fts triggers: all {len(fts.get('expected') or [])} present")
    else:
        missing = ", ".join(fts.get("missing") or [])
        print(f"  fts triggers: missing {missing or '(unknown)'}")
    recent = payload["recent_attempts"]
    if recent:
        latest = recent[0]
        print(
            "  latest: "
            f"{latest['status']} {latest['phase']} "
            f"{latest['succeeded_file_count']}/{latest['needed_file_count']} files, "
            f"read amp {latest['read_amplification']:.2f}x"
        )
    timings = payload.get("convergence_stage_timings") or {}
    print(
        "  completed-attempt timings (n="
        f"{timings.get('sample_size', 0)}): "
        f"parse mean {timings.get('parse_time_s', {}).get('mean', 0.0):.3f}s, "
        f"convergence mean {timings.get('convergence_time_s', {}).get('mean', 0.0):.3f}s"
    )
    debt = payload["convergence_debt"]
    print(f"  convergence debt: {debt['failed_count']} unresolved")
    churn = payload.get("source_path_churn") or []
    if churn:
        worst = churn[0]
        print(
            "  hottest source path: "
            f"{worst['raw_count']} raw rows, {worst['full_raw_count']} full, "
            f"{worst['append_raw_count']} append, {worst['orphan_raw_count']} orphan"
        )
    plan_hazards = [
        f"{name}: {hazard}" for name, info in payload["query_plans"].items() for hazard in info.get("hazards", [])
    ]
    print(f"  query plan hazards: {len(plan_hazards)}")
    for hazard in plan_hazards:
        print(f"    {hazard}")
    return 0


def _json_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
