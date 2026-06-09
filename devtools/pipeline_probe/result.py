"""Result collection, budget capture, and DB statistics for pipeline probes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from devtools.pipeline_probe.request import BudgetReport, ProbeSummary, RawFanoutEntry
from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.scenarios import PipelineProbeRequest
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _count_table(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _archive_file_set_paths(db_path: Path) -> tuple[Path, Path] | None:
    if db_path.name == "source.db":
        source_db = db_path
        index_db = db_path.with_name("index.db")
    elif db_path.name == "index.db":
        source_db = db_path.with_name("source.db")
        index_db = db_path
    else:
        source_db = db_path.with_name("source.db")
        index_db = db_path.with_name("index.db")
    if source_db.exists():
        conn = open_readonly_connection(source_db)
        try:
            if _table_exists(conn, "raw_sessions"):
                return source_db, index_db
        finally:
            conn.close()
    if index_db.exists():
        conn = open_readonly_connection(index_db)
        try:
            # A combined single-file db carries raw_sessions in the same file,
            # so it is not archive storage. Only a true index tier
            # (sessions but no raw_sessions) qualifies here.
            if _table_exists(conn, "sessions") and not _table_exists(conn, "raw_sessions"):
                return source_db, index_db
        finally:
            conn.close()
    return None


def _db_row_counts(db_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    archive_file_set_paths = _archive_file_set_paths(db_path)
    if archive_file_set_paths is not None:
        source_db, index_db = archive_file_set_paths
        stats["archive_file_set_source_db_size_bytes"] = source_db.stat().st_size if source_db.exists() else 0
        stats["archive_file_set_index_db_size_bytes"] = index_db.stat().st_size if index_db.exists() else 0
        if source_db.exists():
            conn = open_readonly_connection(source_db)
            try:
                for table in ("raw_sessions", "blob_refs", "raw_artifacts", "raw_hook_events", "history_sidecars"):
                    stats[f"{table}_count"] = _count_table(conn, table)
            finally:
                conn.close()
        if index_db.exists():
            conn = open_readonly_connection(index_db)
            try:
                for table in ("sessions", "messages", "blocks", "paste_spans", "session_profiles"):
                    stats[f"{table}_count"] = _count_table(conn, table)
            finally:
                conn.close()
        return stats
    with open_connection(db_path) as conn:
        for table in ("raw_sessions", "sessions", "messages", "blocks"):
            stats[f"{table}_count"] = _count_table(conn, table)
    return stats


def _db_raw_fanout(db_path: Path) -> list[RawFanoutEntry]:
    if not db_path.exists():
        return []
    archive_file_set_paths = _archive_file_set_paths(db_path)
    if archive_file_set_paths is not None:
        return _archive_file_set_raw_fanout(*archive_file_set_paths)
    with open_connection(db_path) as conn:
        if not _table_exists(conn, "raw_sessions"):
            return []
        rows = conn.execute(
            """
            SELECT
                r.raw_id,
                COALESCE(r.payload_provider, r.source_name) AS payload_provider,
                r.source_name,
                r.blob_size,
                r.parse_error,
                COUNT(DISTINCT c.session_id) AS session_count,
                COUNT(m.message_id) AS message_count
            FROM raw_sessions r
            LEFT JOIN sessions c ON c.raw_id = r.raw_id
            LEFT JOIN messages m ON m.session_id = c.session_id
            GROUP BY
                r.raw_id,
                COALESCE(r.payload_provider, r.source_name),
                r.source_name,
                r.blob_size,
                r.parse_error
            ORDER BY r.blob_size DESC, r.raw_id ASC
            """
        ).fetchall()
    return [
        {
            "raw_id": str(row["raw_id"]),
            "payload_provider": row["payload_provider"],
            "source_name": row["source_name"],
            "blob_size_bytes": int(row["blob_size"]),
            "session_count": int(row["session_count"]),
            "message_count": int(row["message_count"]),
            "parse_error": row["parse_error"],
        }
        for row in rows
    ]


def _archive_file_set_raw_fanout(source_db: Path, index_db: Path) -> list[RawFanoutEntry]:
    if not source_db.exists():
        return []
    conn = open_readonly_connection(source_db)
    try:
        if not _table_exists(conn, "raw_sessions"):
            return []
        if index_db.exists():
            conn.execute("ATTACH DATABASE ? AS index_tier", (f"file:{index_db}?mode=ro",))
            has_sessions = True
        else:
            has_sessions = False
        session_join = (
            "LEFT JOIN index_tier.sessions s ON s.raw_id = r.raw_id"
            if has_sessions
            else "LEFT JOIN (SELECT NULL AS raw_id, NULL AS session_id) s ON 0"
        )
        message_join = (
            "LEFT JOIN index_tier.messages m ON m.session_id = s.session_id"
            if has_sessions
            else "LEFT JOIN (SELECT NULL AS session_id, NULL AS message_id) m ON 0"
        )
        rows = conn.execute(
            f"""
            SELECT
                r.raw_id,
                r.origin,
                r.source_path,
                r.blob_size,
                r.parse_error,
                COUNT(DISTINCT s.session_id) AS session_count,
                COUNT(m.message_id) AS message_count
            FROM raw_sessions r
            {session_join}
            {message_join}
            GROUP BY
                r.raw_id,
                r.origin,
                r.source_path,
                r.blob_size,
                r.parse_error
            ORDER BY r.blob_size DESC, r.raw_id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "raw_id": str(row[0]),
            "payload_provider": row[1],
            "source_name": row[1],
            "blob_size_bytes": int(row[3]),
            "session_count": int(row[5]),
            "message_count": int(row[6]),
            "parse_error": row[4],
            "storage_route": "archive_file_set",
        }
        for row in rows
    ]


def _json_object_or_empty(value: object | None) -> JSONDocument:
    return json_document(value)


def _json_float_or_none(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _json_string_sequence(value: list[str] | None) -> list[JSONValue] | None:
    if value is None:
        return None
    result: list[JSONValue] = []
    result.extend(value)
    return result


def _observed_peak_rss_mb(metrics: JSONDocument) -> tuple[JSONValue, JSONValue, JSONValue]:
    peak_self = metrics.get("peak_rss_self_mb")
    peak_children = metrics.get("peak_rss_children_mb")
    peak_self_value = _json_float_or_none(peak_self)
    peak_children_value = _json_float_or_none(peak_children)
    if peak_self_value is None:
        return peak_self, peak_self, peak_children
    if peak_children_value is None:
        return peak_self, peak_self, peak_children
    return round(peak_self_value + peak_children_value, 1), peak_self, peak_children


def _build_budget_report(summary: ProbeSummary, request: PipelineProbeRequest) -> BudgetReport | None:
    if request.max_total_ms is None and request.max_peak_rss_mb is None:
        return None

    run_payload = _json_object_or_empty(summary.get("run_payload"))
    metrics = _json_object_or_empty(run_payload.get("metrics"))
    result_payload = _json_object_or_empty(summary.get("result"))
    observed_total_ms = metrics.get("total_duration_ms", result_payload.get("duration_ms"))
    observed_peak_rss_mb, observed_peak_rss_self_mb, observed_peak_rss_children_mb = _observed_peak_rss_mb(metrics)
    violations: list[str] = []

    if request.max_total_ms is not None:
        if observed_total_ms is None:
            violations.append("missing total runtime metric")
        elif (observed_total_ms_value := _json_float_or_none(observed_total_ms)) is None:
            violations.append("non-numeric total runtime metric")
        elif observed_total_ms_value > request.max_total_ms:
            violations.append(
                f"total runtime {observed_total_ms_value:.1f} ms exceeded budget {request.max_total_ms:.1f} ms"
            )

    if request.max_peak_rss_mb is not None:
        if observed_peak_rss_mb is None:
            violations.append("missing peak RSS metric")
        elif (observed_peak_rss_mb_value := _json_float_or_none(observed_peak_rss_mb)) is None:
            violations.append("non-numeric peak RSS metric")
        elif observed_peak_rss_mb_value > request.max_peak_rss_mb:
            violations.append(
                f"peak RSS {observed_peak_rss_mb_value:.1f} MiB exceeded budget {request.max_peak_rss_mb:.1f} MiB"
            )

    return {
        "ok": not violations,
        "max_total_ms": request.max_total_ms,
        "observed_total_ms": observed_total_ms,
        "max_peak_rss_mb": request.max_peak_rss_mb,
        "observed_peak_rss_mb": observed_peak_rss_mb,
        "observed_peak_rss_self_mb": observed_peak_rss_self_mb,
        "observed_peak_rss_children_mb": observed_peak_rss_children_mb,
        "violations": violations,
    }


__all__ = [
    "_build_budget_report",
    "_db_raw_fanout",
    "_db_row_counts",
    "_json_float_or_none",
    "_json_object_or_empty",
    "_json_string_sequence",
    "_observed_peak_rss_mb",
]
