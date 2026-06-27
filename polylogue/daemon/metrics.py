"""Prometheus text-format metrics endpoint for the daemon HTTP API (#1321).

``GET /metrics`` returns the Prometheus exposition format
(``text/plain; version=0.0.4``) so operators can scrape the daemon
from Prometheus / Grafana / Victoria Metrics without an additional
sidecar.

Design notes:

- **No external dependency.** Polylogue does not depend on
  ``prometheus_client``. The exposition format is small and stable
  (see https://prometheus.io/docs/instrumenting/exposition_formats/),
  so we hand-roll it. This keeps the daemon's import surface flat
  and avoids dragging in a transitive C dependency for a metrics
  endpoint that emits at most a few dozen series.
- **No authentication.** Same posture as ``/healthz/*`` (see
  ``polylogue/daemon/healthz.py``): metrics endpoints are scraped by
  Prometheus, which does not carry credentials, and the daemon binds
  to loopback by default. The endpoint exposes only counts and gauges
  derived from existing daemon-state tables — no session
  content, no environment.
- **Read-only.** Every series is sourced from the archive SQLite
  database via ``open_readonly_connection``. The endpoint never
  writes and never blocks on a held writer.
- **Resilient to missing tables.** Each section gracefully degrades
  to zero / absent series when a backing table does not yet exist
  (fresh archives, schema bumps mid-rollout).
- **OTLP receiver deferred.** The OpenTelemetry HTTP receiver from
  the original "ambitious move" section of #1224 is intentionally
  out of scope here; it remains tracked under #1321 as a follow-up.

Exposed series (label policy: use labels only when a series naturally
varies on a known-bounded dimension):

- ``polylogue_daemon_uptime_seconds`` (gauge) — process uptime
- ``polylogue_daemon_build_info`` (gauge, value 1) — labels: version
- ``polylogue_status_snapshot_age_seconds`` (gauge) — cached status age
- ``polylogue_status_snapshot_state`` (gauge) — labels: state
- ``polylogue_live_ingest_attempts_total`` (counter) — labels: status
- ``polylogue_live_ingest_attempts_in_flight`` (gauge)
- ``polylogue_live_ingest_storage_route_total`` (counter) — labels: route
- ``polylogue_live_ingest_attempt_duration_seconds`` (gauge buckets:
  min, mean, max derived from recent completed attempts)
- ``polylogue_convergence_debt_count`` (gauge) — labels: stage
- ``polylogue_blob_lease_pending_count`` (gauge)
- ``polylogue_blob_lease_distinct_operations`` (gauge)
- ``polylogue_fts_trigger_present`` (gauge) — labels: trigger
- ``polylogue_fts_triggers_all_present`` (gauge, 0/1)
- ``polylogue_fts_freshness_ready`` (gauge, 0/1) — labels: surface
- ``polylogue_live_ingest_memory_mebibytes`` (gauge) — labels: kind
- ``polylogue_stale_cursor_writes_total`` (counter)
- ``polylogue_embedding_sessions`` (gauge) — labels: state
- ``polylogue_embedding_messages`` (gauge) — labels: state
- ``polylogue_embedding_coverage_percent`` (gauge)
- ``polylogue_embedding_status_state`` (gauge) — labels: status
- ``polylogue_embedding_retrieval_ready`` (gauge, 0/1)
- ``polylogue_embedding_latest_catchup_run_info`` (gauge) — labels: status, rebuild
- ``polylogue_embedding_latest_catchup_sessions`` (gauge) — labels: state
- ``polylogue_embedding_latest_catchup_messages`` (gauge) — labels: state
- ``polylogue_embedding_latest_catchup_estimated_cost_usd`` (gauge)
- ``polylogue_archive_tier_present`` (gauge) — labels: tier
- ``polylogue_archive_tier_count`` (gauge) — labels: state
- ``polylogue_archive_tier_file_size_bytes`` (gauge) — labels: tier, kind
- ``polylogue_archive_tier_user_version`` (gauge) — labels: tier
- ``polylogue_archive_storage_layout`` (gauge) — labels: layout
- ``polylogue_archive_storage_ready`` (gauge) — labels: state
- ``polylogue_archive_active_store`` (gauge) — labels: store
- ``polylogue_archive_active_tier_role`` (gauge) — labels: role
- ``polylogue_archive_ready`` (gauge, 0/1)
- ``polylogue_archive_blocker_count`` (gauge)
- ``polylogue_archive_blocker`` (gauge) — labels: blocker
- ``polylogue_archive_source_index_links_total`` (gauge) — labels: source, state

The handler signature mirrors ``healthz.py``'s ``ProbeResponder``
protocol so it is testable without the full ``BaseHTTPRequestHandler``
stack.
"""

from __future__ import annotations

import json
import sqlite3
from http import HTTPStatus
from pathlib import Path
from typing import Protocol, TypedDict

from polylogue.daemon.process_start import uptime_seconds
from polylogue.logging import get_logger
from polylogue.storage import archive_layout
from polylogue.storage.archive_layout import (
    ARCHIVE_ACTIVE_TIER_ROLES,
    ARCHIVE_LAYOUT_BLOCKER_LABELS,
    ARCHIVE_STORAGE_LAYOUTS,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

logger = get_logger(__name__)

# Derived from the canonical tier specs so the expected schema version per tier
# can never drift from ARCHIVE_VERSION_BY_TIER. (tier, filename, expected_version,
# backup_required), preserving the source→index→embeddings→user→ops order.
_ARCHIVE_TIER_FILES: tuple[tuple[str, str, int, bool], ...] = tuple(
    (spec.tier.value, spec.filename, spec.version, spec.backup_required) for spec in ARCHIVE_TIER_SPECS.values()
)

_KNOWN_STORAGE_ROUTES: frozenset[str] = frozenset(
    {
        "archive_full",
        "archive_append",
        "archive_file_set",
        "unsupported_polylogue_batch",
        "unknown",
    }
)


class EmbeddingMetricState(TypedDict):
    total_sessions: int
    embedded_sessions: int
    pending_sessions: int
    failed_sessions: int
    embedded_messages: int
    coverage_percent: float
    status: str
    retrieval_ready: int
    latest_status: str | None
    latest_rebuild: str
    latest_planned_sessions: int
    latest_processed_sessions: int
    latest_embedded_sessions: int
    latest_skipped_sessions: int
    latest_error_count: int
    latest_planned_messages: int
    latest_embedded_messages: int
    latest_estimated_cost_usd: float


class ArchiveEmbeddingRunState(TypedDict):
    status: str
    scanned_sessions: int
    embedded_messages: int
    error_count: int
    estimated_cost_usd: float


PROMETHEUS_CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"


class MetricsResponder(Protocol):
    """Subset of ``DaemonAPIHandler`` the metrics handler depends on."""

    def _send_text(
        self,
        status: HTTPStatus,
        body: str,
        *,
        content_type: str,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Exposition format helpers
# ---------------------------------------------------------------------------


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{name}="{_escape_label_value(val)}"' for name, val in sorted(labels.items()))
    return "{" + inner + "}"


def _format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    # Prometheus accepts integers and floats; emit floats with enough
    # precision to round-trip durations without scientific notation for
    # ordinary values.
    if value != value:  # NaN
        return "NaN"
    return repr(float(value))


def _emit_metric(
    lines: list[str],
    *,
    name: str,
    help_text: str,
    metric_type: str,
    samples: list[tuple[dict[str, str] | None, float | int]],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    if not samples:
        # Emit a zero sample so the series is discoverable even when no
        # backing rows exist yet.
        lines.append(f"{name} 0")
        return
    for labels, value in samples:
        lines.append(f"{name}{_render_labels(labels)} {_format_value(value)}")


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _attempt_counts(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> dict[str, int]:
    """Return totals of ``live_ingest_attempt`` rows by status plus stale writes."""
    if ops_db is not None:
        ops_counts = _ops_attempt_counts(ops_db)
        if ops_counts is not None:
            return ops_counts
    counts = {"running": 0, "completed": 0, "failed": 0, "stale_cursor_writes": 0}
    if not _table_exists(conn, "live_ingest_attempt"):
        return counts
    for status in ("running", "completed", "failed"):
        counts[status] = _scalar_int(
            conn,
            f"SELECT COUNT(*) FROM live_ingest_attempt WHERE status = '{status}'",
        )
    if "stale_cursor_write_count" in _columns(conn, "live_ingest_attempt"):
        counts["stale_cursor_writes"] = _scalar_int(
            conn,
            "SELECT COALESCE(SUM(stale_cursor_write_count), 0) FROM live_ingest_attempt",
        )
    return counts


def _ops_attempt_counts(ops_db: Path) -> dict[str, int] | None:
    if not ops_db.exists():
        return None
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return None
            rows = conn.execute("SELECT status, COUNT(*) FROM ingest_attempts GROUP BY status").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not rows:
        return None
    counts = {"running": 0, "completed": 0, "failed": 0, "stale_cursor_writes": 0}
    for row in rows:
        status = str(row[0])
        if status in counts:
            counts[status] = int(row[1] or 0)
    return counts


def _recent_attempt_durations(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    ops_db: Path | None = None,
) -> list[float]:
    """Return durations of recent completed attempts (seconds)."""
    if ops_db is not None:
        ops_durations = _ops_recent_attempt_durations(ops_db, limit=limit)
        if ops_durations:
            return ops_durations
    if not _table_exists(conn, "live_ingest_attempt"):
        return []
    cols = _columns(conn, "live_ingest_attempt")
    # convergence_time_s is the canonical end-to-end timing.
    if "convergence_time_s" not in cols:
        return []
    rows = conn.execute(
        """
        SELECT convergence_time_s
        FROM live_ingest_attempt
        WHERE status = 'completed' AND convergence_time_s IS NOT NULL
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [float(row[0]) for row in rows if row[0] is not None]


def _ops_recent_attempt_durations(ops_db: Path, *, limit: int = 50) -> list[float]:
    if not ops_db.exists():
        return []
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return []
            rows = conn.execute(
                """
                SELECT started_at_ms, finished_at_ms
                FROM ingest_attempts
                WHERE status = 'completed'
                  AND started_at_ms IS NOT NULL
                  AND finished_at_ms IS NOT NULL
                ORDER BY finished_at_ms DESC, started_at_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [max(0.0, (int(row[1]) - int(row[0])) / 1000.0) for row in rows if row[0] is not None and row[1] is not None]


def _convergence_debt_by_stage(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> list[tuple[str, int]]:
    ops_rows = _ops_convergence_debt_by_stage(ops_db) if ops_db is not None else []
    if ops_rows:
        return ops_rows
    if not _table_exists(conn, "live_convergence_debt"):
        return []
    rows = conn.execute(
        """
        SELECT stage, COUNT(*)
        FROM live_convergence_debt
        WHERE status != 'resolved'
        GROUP BY stage
        ORDER BY stage
        """
    ).fetchall()
    return [(str(row[0] or "unknown"), int(row[1] or 0)) for row in rows]


def _ops_convergence_debt_by_stage(ops_db: Path | None) -> list[tuple[str, int]]:
    if ops_db is None or not ops_db.exists():
        return []
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "convergence_debt"):
                return []
            rows = conn.execute(
                """
                SELECT stage, COUNT(*)
                FROM convergence_debt
                GROUP BY stage
                ORDER BY stage
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [(str(row[0] or "unknown"), int(row[1] or 0)) for row in rows]


def _blob_lease_state(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (pending_count, distinct_operations)."""
    if not _table_exists(conn, "pending_blob_refs"):
        return (0, 0)
    pending = _scalar_int(conn, "SELECT COUNT(*) FROM pending_blob_refs")
    operations = _scalar_int(conn, "SELECT COUNT(DISTINCT operation_id) FROM pending_blob_refs")
    return (pending, operations)


def _fts_trigger_presence(conn: sqlite3.Connection) -> dict[str, bool]:
    from polylogue.daemon.fts_startup import active_fts_triggers_sync

    expected = active_fts_triggers_sync(conn)
    if not expected:
        return {}
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        expected,
    ).fetchall()
    present = {str(row[0]) for row in rows}
    return {name: (name in present) for name in expected}


def _fts_freshness_ready(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    if not _table_exists(conn, "fts_freshness_state"):
        return []
    rows = conn.execute(
        """
        SELECT surface, state
        FROM fts_freshness_state
        ORDER BY surface
        """
    ).fetchall()
    return [(str(row[0]), 1 if str(row[1]) == "ready" else 0) for row in rows]


def _latest_ingest_memory(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> list[tuple[str, float]]:
    if ops_db is not None:
        ops_memory = _ops_latest_ingest_memory(ops_db)
        if ops_memory:
            return ops_memory
    if not _table_exists(conn, "live_ingest_attempt"):
        return []
    cols = _columns(conn, "live_ingest_attempt")
    metric_columns = {
        "rss_current": "rss_current_mb",
        "rss_peak_self": "rss_peak_self_mb",
        "rss_peak_children": "rss_peak_children_mb",
        "cgroup_current": "cgroup_memory_current_mb",
        "cgroup_peak": "cgroup_memory_peak_mb",
        "cgroup_swap_current": "cgroup_memory_swap_current_mb",
        "cgroup_anon": "cgroup_memory_anon_mb",
        "cgroup_file": "cgroup_memory_file_mb",
        "cgroup_inactive_file": "cgroup_memory_inactive_file_mb",
    }
    available = [(kind, column) for kind, column in metric_columns.items() if column in cols]
    if not available:
        return []
    select_list = ", ".join(column for _, column in available)
    row = conn.execute(
        f"""
        SELECT {select_list}
        FROM live_ingest_attempt
        ORDER BY updated_at DESC, started_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return []
    samples: list[tuple[str, float]] = []
    for idx, (kind, _column) in enumerate(available):
        value = row[idx]
        if value is not None:
            samples.append((kind, float(value)))
    return samples


def _ops_latest_ingest_memory(ops_db: Path) -> list[tuple[str, float]]:
    if not ops_db.exists():
        return []
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "daemon_stage_events"):
                return []
            row = conn.execute(
                """
                SELECT payload_json
                FROM daemon_stage_events
                ORDER BY observed_at_ms DESC, event_id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    if row is None:
        return []
    payload = _json_payload(row[0])
    metric_keys = {
        "rss_current": "rss_current_mb",
        "rss_peak_self": "rss_peak_self_mb",
        "rss_peak_children": "rss_peak_children_mb",
        "cgroup_current": "cgroup_memory_current_mb",
        "cgroup_peak": "cgroup_memory_peak_mb",
        "cgroup_swap_current": "cgroup_memory_swap_current_mb",
        "cgroup_anon": "cgroup_memory_anon_mb",
        "cgroup_file": "cgroup_memory_file_mb",
        "cgroup_inactive_file": "cgroup_memory_inactive_file_mb",
    }
    samples: list[tuple[str, float]] = []
    for kind, key in metric_keys.items():
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            samples.append((kind, float(value)))
    return samples


def _json_payload(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _normalise_storage_route(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "unknown"
    return value if value in _KNOWN_STORAGE_ROUTES else "other"


def _empty_storage_route_counts() -> dict[str, int]:
    counts = dict.fromkeys(sorted(_KNOWN_STORAGE_ROUTES), 0)
    counts["other"] = 0
    return counts


def _storage_route_counts(
    conn: sqlite3.Connection,
    *,
    ops_db: Path | None = None,
) -> dict[str, int]:
    """Return live-ingest attempt counts grouped by bounded storage route."""
    if ops_db is not None:
        ops_counts = _ops_storage_route_counts(ops_db)
        if ops_counts is not None:
            return ops_counts

    counts = _empty_storage_route_counts()
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


def _ops_storage_route_counts(ops_db: Path) -> dict[str, int] | None:
    if not ops_db.exists():
        return None
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return None
            counts = _empty_storage_route_counts()
            attempt_columns = _columns(conn, "ingest_attempts")
            if "storage_route" in attempt_columns:
                rows = conn.execute(
                    "SELECT storage_route, COUNT(*) FROM ingest_attempts GROUP BY storage_route"
                ).fetchall()
                for row in rows:
                    route = _normalise_storage_route(row[0])
                    counts[route] = counts.get(route, 0) + int(row[1] or 0)
                return counts

            total_attempts = _scalar_int(conn, "SELECT COUNT(*) FROM ingest_attempts")
            if not _table_exists(conn, "daemon_stage_events"):
                counts["unknown"] = total_attempts
                return counts
            index_names = {str(row[1]) for row in conn.execute("PRAGMA index_list('daemon_stage_events')")}
            if "idx_daemon_stage_events_attempt_observed" not in index_names:
                counts["unknown"] = total_attempts
                return counts

            rows = conn.execute(
                """
                SELECT (
                    SELECT payload_json
                    FROM daemon_stage_events AS e
                    WHERE e.attempt_id = a.attempt_id
                      AND e.payload_json LIKE '%"storage_route"%'
                    ORDER BY e.observed_at_ms DESC, e.rowid DESC
                    LIMIT 1
                ) AS payload_json,
                COUNT(*)
                FROM ingest_attempts AS a
                GROUP BY payload_json
                """
            ).fetchall()
            for row in rows:
                route = _normalise_storage_route(_json_payload(row[0]).get("storage_route"))
                counts[route] = counts.get(route, 0) + int(row[1] or 0)
            return counts
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _emit_storage_route_metrics(lines: list[str], counts: dict[str, int]) -> None:
    _emit_metric(
        lines,
        name="polylogue_live_ingest_storage_route_total",
        help_text="Live ingest attempts grouped by bounded storage route.",
        metric_type="counter",
        samples=[({"route": route}, count) for route, count in sorted(counts.items())],
    )


def _embedding_message_count(conn: sqlite3.Connection) -> int:
    if _table_exists(conn, "message_embeddings_rowids"):
        return _scalar_int(conn, "SELECT COUNT(*) FROM message_embeddings_rowids")
    if _table_exists(conn, "message_embeddings"):
        return _scalar_int(conn, "SELECT COUNT(*) FROM message_embeddings")
    return 0


def _embedding_state(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> EmbeddingMetricState:
    """Return bounded embedding backlog and latest catch-up state."""

    if _table_exists(conn, "sessions"):
        return _archive_embedding_state(conn, ops_db=ops_db)

    total_sessions = 0
    embedded_sessions = 0
    pending_sessions = total_sessions
    failed_sessions = 0
    if _table_exists(conn, "embedding_status"):
        columns = _columns(conn, "embedding_status")
        embedded_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0")
        pending_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1")
        if "error_message" in columns:
            failed_sessions = _scalar_int(
                conn,
                "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL",
            )
    embedded_messages = _embedding_message_count(conn)
    coverage_percent = max(0, total_sessions - pending_sessions) / total_sessions * 100 if total_sessions > 0 else 0.0
    if total_sessions <= 0 and pending_sessions <= 0 and embedded_messages <= 0:
        status = "empty"
    elif embedded_messages <= 0:
        status = "none"
    elif pending_sessions > 0:
        status = "partial"
    else:
        status = "complete"

    latest_run = None
    if _table_exists(conn, "embedding_catchup_runs"):
        from polylogue.storage.embeddings.progress import latest_embedding_catchup_run

        latest_run = latest_embedding_catchup_run(conn)

    return {
        "total_sessions": total_sessions,
        "embedded_sessions": embedded_sessions,
        "pending_sessions": pending_sessions,
        "failed_sessions": failed_sessions,
        "embedded_messages": embedded_messages,
        "coverage_percent": coverage_percent,
        "status": status,
        "retrieval_ready": 1 if embedded_messages > 0 else 0,
        "latest_status": latest_run["status"] if latest_run is not None else None,
        "latest_rebuild": str(bool(latest_run["rebuild"])).lower() if latest_run is not None else "false",
        "latest_planned_sessions": int(latest_run["planned_sessions"]) if latest_run is not None else 0,
        "latest_processed_sessions": int(latest_run["processed_sessions"]) if latest_run is not None else 0,
        "latest_embedded_sessions": int(latest_run["embedded_sessions"]) if latest_run is not None else 0,
        "latest_skipped_sessions": int(latest_run["skipped_sessions"]) if latest_run is not None else 0,
        "latest_error_count": int(latest_run["error_count"]) if latest_run is not None else 0,
        "latest_planned_messages": int(latest_run["planned_messages"]) if latest_run is not None else 0,
        "latest_embedded_messages": int(latest_run["embedded_messages"]) if latest_run is not None else 0,
        "latest_estimated_cost_usd": float(latest_run["estimated_cost_usd"]) if latest_run is not None else 0.0,
    }


def _archive_embedding_state(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> EmbeddingMetricState:
    total_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM sessions")
    embedded_sessions = 0
    pending_sessions = total_sessions
    failed_sessions = 0
    if _table_exists(conn, "embedding_status"):
        embedded_sessions = _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions s
            JOIN embedding_status e ON e.session_id = s.session_id
            WHERE e.needs_reindex = 0
              AND e.message_count_embedded >= s.message_count
            """,
        )
        pending_sessions = _scalar_int(
            conn,
            """
            SELECT COUNT(*)
            FROM sessions s
            LEFT JOIN embedding_status e ON e.session_id = s.session_id
            WHERE e.session_id IS NULL OR e.needs_reindex = 1 OR e.message_count_embedded < s.message_count
            """,
        )
        failed_sessions = _scalar_int(
            conn,
            "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL",
        )

    embedded_messages = _embedding_message_count(conn)
    coverage_percent = max(0, total_sessions - pending_sessions) / total_sessions * 100 if total_sessions > 0 else 0.0
    if total_sessions <= 0 and pending_sessions <= 0 and embedded_messages <= 0:
        status = "empty"
    elif embedded_messages <= 0:
        status = "none"
    elif pending_sessions > 0:
        status = "partial"
    else:
        status = "complete"

    latest = _archive_latest_embedding_run_state(ops_db)

    return {
        "total_sessions": total_sessions,
        "embedded_sessions": embedded_sessions,
        "pending_sessions": pending_sessions,
        "failed_sessions": failed_sessions,
        "embedded_messages": embedded_messages,
        "coverage_percent": coverage_percent,
        "status": status,
        "retrieval_ready": 1 if embedded_messages > 0 else 0,
        "latest_status": latest["status"] if latest is not None else None,
        "latest_rebuild": "false",
        "latest_planned_sessions": latest["scanned_sessions"] if latest is not None else 0,
        "latest_processed_sessions": latest["scanned_sessions"] if latest is not None else 0,
        "latest_embedded_sessions": 0,
        "latest_skipped_sessions": 0,
        "latest_error_count": latest["error_count"] if latest is not None else 0,
        "latest_planned_messages": 0,
        "latest_embedded_messages": latest["embedded_messages"] if latest is not None else 0,
        "latest_estimated_cost_usd": latest["estimated_cost_usd"] if latest is not None else 0.0,
    }


def _archive_latest_embedding_run_state(ops_db: Path | None) -> ArchiveEmbeddingRunState | None:
    if ops_db is None or not ops_db.exists():
        return None
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "embedding_catchup_runs"):
                return None
            runs = list_embedding_catchup_runs(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "status": run.status,
        "scanned_sessions": run.scanned_sessions,
        "embedded_messages": run.embedded_messages,
        "error_count": 1 if run.error_message else 0,
        "estimated_cost_usd": float(run.estimated_cost_usd or 0.0),
    }


def _emit_embedding_metrics(lines: list[str], state: EmbeddingMetricState) -> None:
    _emit_metric(
        lines,
        name="polylogue_embedding_sessions",
        help_text="Embedding session counts by state.",
        metric_type="gauge",
        samples=[
            ({"state": "total"}, int(state["total_sessions"])),
            ({"state": "embedded"}, int(state["embedded_sessions"])),
            ({"state": "pending"}, int(state["pending_sessions"])),
            ({"state": "failed"}, int(state["failed_sessions"])),
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_messages",
        help_text="Embedding message counts by state. Pending messages are intentionally not counted on scrape.",
        metric_type="gauge",
        samples=[({"state": "embedded"}, int(state["embedded_messages"]))],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_coverage_percent",
        help_text="Percent of sessions with current embeddings.",
        metric_type="gauge",
        samples=[(None, float(state["coverage_percent"]))],
    )
    current_status = str(state["status"])
    _emit_metric(
        lines,
        name="polylogue_embedding_status_state",
        help_text="One-hot embedding materialization state for bounded scrapes.",
        metric_type="gauge",
        samples=[
            ({"status": status}, 1 if current_status == status else 0)
            for status in ("empty", "none", "partial", "complete")
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_retrieval_ready",
        help_text="Whether semantic retrieval has at least one materialized vector available.",
        metric_type="gauge",
        samples=[(None, int(state["retrieval_ready"]))],
    )

    latest_status = str(state["latest_status"] or "none")
    _emit_metric(
        lines,
        name="polylogue_embedding_latest_catchup_run_info",
        help_text="1 for the latest embedding catch-up run status and rebuild mode, or 0 when none exists.",
        metric_type="gauge",
        samples=[
            ({"status": latest_status, "rebuild": str(state["latest_rebuild"])}, 0 if latest_status == "none" else 1)
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_latest_catchup_sessions",
        help_text="Latest embedding catch-up run session counts by state.",
        metric_type="gauge",
        samples=[
            ({"state": "planned"}, int(state["latest_planned_sessions"])),
            ({"state": "processed"}, int(state["latest_processed_sessions"])),
            ({"state": "embedded"}, int(state["latest_embedded_sessions"])),
            ({"state": "skipped"}, int(state["latest_skipped_sessions"])),
            ({"state": "failed"}, int(state["latest_error_count"])),
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_latest_catchup_messages",
        help_text="Latest embedding catch-up run message counts by state.",
        metric_type="gauge",
        samples=[
            ({"state": "planned"}, int(state["latest_planned_messages"])),
            ({"state": "embedded"}, int(state["latest_embedded_messages"])),
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_embedding_latest_catchup_estimated_cost_usd",
        help_text="Latest embedding catch-up run estimated provider cost in USD.",
        metric_type="gauge",
        samples=[(None, float(state["latest_estimated_cost_usd"]))],
    )


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def format_metrics(
    db: Path,
    *,
    now_monotonic: float | None = None,
) -> str:
    """Render the Prometheus text exposition for the daemon archive at ``db``.

    Exposes a stable set of series derived from existing daemon state
    tables. Missing tables degrade to zero samples rather than raising.
    Caller injects ``now_monotonic`` in tests to keep uptime stable.
    """
    uptime_s = uptime_seconds(now_monotonic=now_monotonic)
    lines: list[str] = []
    index_db = db.with_name("index.db")
    if db.name != "index.db" and index_db.exists():
        db = index_db

    _emit_metric(
        lines,
        name="polylogue_daemon_uptime_seconds",
        help_text="Daemon process uptime in seconds.",
        metric_type="gauge",
        samples=[(None, uptime_s)],
    )

    # Build info — version label so dashboards can break out per-release.
    try:
        from polylogue import __version__ as polylogue_version
    except Exception:
        polylogue_version = "unknown"
    _emit_metric(
        lines,
        name="polylogue_daemon_build_info",
        help_text="Constant 1 gauge labelled with daemon build identity.",
        metric_type="gauge",
        samples=[({"version": str(polylogue_version)}, 1)],
    )
    _emit_archive_storage_metrics(lines, db)

    from polylogue.daemon.status_snapshot import snapshot_state_for_metrics

    snapshot = snapshot_state_for_metrics()
    snapshot_state = str(snapshot.get("state", "missing"))
    snapshot_age = float(snapshot.get("age_s", -1.0))
    _emit_metric(
        lines,
        name="polylogue_status_snapshot_age_seconds",
        help_text="Age of the cached daemon status snapshot in seconds, or -1 when absent.",
        metric_type="gauge",
        samples=[(None, snapshot_age)],
    )
    _emit_metric(
        lines,
        name="polylogue_status_snapshot_state",
        help_text="1 for the current daemon status snapshot freshness state.",
        metric_type="gauge",
        samples=[({"state": state}, 1 if state == snapshot_state else 0) for state in ("fresh", "stale", "missing")],
    )

    if not db.exists():
        ops_body = _format_ops_only_metrics(lines, db.with_name("ops.db"))
        if ops_body is not None:
            return ops_body
        # Fresh install — emit the discovery skeleton with zeros so the
        # scraper sees a stable series set on day zero.
        for name, help_text in (
            ("polylogue_live_ingest_attempts_total", "Total live ingest attempts by status."),
            ("polylogue_live_ingest_attempts_in_flight", "Live ingest attempts currently running."),
            ("polylogue_live_ingest_storage_route_total", "Live ingest attempts grouped by storage route."),
            (
                "polylogue_live_ingest_attempt_duration_seconds",
                "Convergence time (seconds) of recent completed ingest attempts.",
            ),
            ("polylogue_convergence_debt_count", "Unresolved convergence-debt rows by stage."),
            ("polylogue_blob_lease_pending_count", "Pending blob leases in flight."),
            ("polylogue_blob_lease_distinct_operations", "Distinct lease-holding operations."),
            (
                "polylogue_fts_trigger_present",
                "1 when the named FTS sync trigger is installed in index.db.",
            ),
            ("polylogue_fts_triggers_all_present", "All expected FTS sync triggers are installed."),
            ("polylogue_fts_freshness_ready", "1 when the daemon freshness ledger marks an FTS surface ready."),
            ("polylogue_live_ingest_memory_mebibytes", "Latest live ingest memory sample in MiB by kind."),
            ("polylogue_stale_cursor_writes_total", "Total stale-cursor writes observed across ingest attempts."),
            ("polylogue_embedding_sessions", "Embedding session counts by state."),
            ("polylogue_embedding_messages", "Embedding message counts by state."),
            ("polylogue_embedding_coverage_percent", "Percent of sessions with current embeddings."),
            ("polylogue_embedding_status_state", "One-hot embedding materialization state for bounded scrapes."),
            ("polylogue_embedding_retrieval_ready", "Whether semantic retrieval has a materialized vector available."),
            ("polylogue_embedding_latest_catchup_run_info", "Latest embedding catch-up run status."),
            ("polylogue_embedding_latest_catchup_sessions", "Latest embedding catch-up run session counts."),
            ("polylogue_embedding_latest_catchup_messages", "Latest embedding catch-up run message counts."),
            (
                "polylogue_embedding_latest_catchup_estimated_cost_usd",
                "Latest embedding catch-up run estimated provider cost in USD.",
            ),
            (
                "polylogue_archive_source_index_links_total",
                "Raw source rows by index materialization state.",
            ),
        ):
            metric_type = (
                "gauge"
                if name == "polylogue_archive_source_index_links_total"
                else "counter"
                if name.endswith("_total")
                else "gauge"
            )
            _emit_metric(lines, name=name, help_text=help_text, metric_type=metric_type, samples=[])
        return "\n".join(lines) + "\n"

    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(db)
    try:
        ops_db = db.with_name("ops.db")
        attempts = _attempt_counts(conn, ops_db=ops_db)
        _emit_metric(
            lines,
            name="polylogue_live_ingest_attempts_total",
            help_text="Total live ingest attempts by status.",
            metric_type="counter",
            samples=[
                ({"status": "completed"}, attempts["completed"]),
                ({"status": "failed"}, attempts["failed"]),
                # Running counted separately so total = completed + failed + running.
                ({"status": "running"}, attempts["running"]),
            ],
        )
        _emit_metric(
            lines,
            name="polylogue_live_ingest_attempts_in_flight",
            help_text="Live ingest attempts currently running.",
            metric_type="gauge",
            samples=[(None, attempts["running"])],
        )
        _emit_metric(
            lines,
            name="polylogue_stale_cursor_writes_total",
            help_text="Total stale-cursor writes observed across ingest attempts.",
            metric_type="counter",
            samples=[(None, attempts["stale_cursor_writes"])],
        )
        _emit_storage_route_metrics(lines, _storage_route_counts(conn, ops_db=ops_db))

        durations = _recent_attempt_durations(conn, ops_db=ops_db)
        if durations:
            _emit_metric(
                lines,
                name="polylogue_live_ingest_attempt_duration_seconds",
                help_text=(
                    "Convergence time (seconds) of recent completed ingest attempts: "
                    "min/mean/max derived from the most recent 50 attempts."
                ),
                metric_type="gauge",
                samples=[
                    ({"quantile": "min"}, min(durations)),
                    ({"quantile": "mean"}, sum(durations) / len(durations)),
                    ({"quantile": "max"}, max(durations)),
                ],
            )
        else:
            _emit_metric(
                lines,
                name="polylogue_live_ingest_attempt_duration_seconds",
                help_text="Convergence time (seconds) of recent completed ingest attempts.",
                metric_type="gauge",
                samples=[],
            )

        debt = _convergence_debt_by_stage(conn, ops_db=ops_db)
        if debt:
            _emit_metric(
                lines,
                name="polylogue_convergence_debt_count",
                help_text="Unresolved convergence-debt rows by stage.",
                metric_type="gauge",
                samples=[({"stage": stage}, count) for stage, count in debt],
            )
        else:
            _emit_metric(
                lines,
                name="polylogue_convergence_debt_count",
                help_text="Unresolved convergence-debt rows by stage.",
                metric_type="gauge",
                samples=[(None, 0)],
            )

        pending, ops = _blob_lease_state(conn)
        _emit_metric(
            lines,
            name="polylogue_blob_lease_pending_count",
            help_text="Pending blob leases awaiting commit.",
            metric_type="gauge",
            samples=[(None, pending)],
        )
        _emit_metric(
            lines,
            name="polylogue_blob_lease_distinct_operations",
            help_text="Distinct operations currently holding at least one blob lease.",
            metric_type="gauge",
            samples=[(None, ops)],
        )

        triggers = _fts_trigger_presence(conn)
        _emit_metric(
            lines,
            name="polylogue_fts_trigger_present",
            help_text="1 when the named FTS sync trigger is installed in index.db.",
            metric_type="gauge",
            samples=[({"trigger": name}, 1 if present else 0) for name, present in triggers.items()],
        )
        _emit_metric(
            lines,
            name="polylogue_fts_triggers_all_present",
            help_text="1 when every expected FTS sync trigger is installed.",
            metric_type="gauge",
            samples=[(None, 1 if all(triggers.values()) else 0)],
        )

        freshness = _fts_freshness_ready(conn)
        _emit_metric(
            lines,
            name="polylogue_fts_freshness_ready",
            help_text="1 when the daemon freshness ledger marks an FTS surface ready.",
            metric_type="gauge",
            samples=[({"surface": surface}, ready) for surface, ready in freshness],
        )

        memory = _latest_ingest_memory(conn, ops_db=ops_db)
        _emit_metric(
            lines,
            name="polylogue_live_ingest_memory_mebibytes",
            help_text="Latest live ingest memory sample in MiB by kind.",
            metric_type="gauge",
            samples=[({"kind": kind}, value) for kind, value in memory],
        )

        _emit_embedding_metrics(lines, _embedding_state(conn, ops_db=ops_db))

        # ── Rich instrumentation (#1321 ambitious scope) ──────────
        _emit_archive_metrics(lines, conn)
        _emit_throughput_metrics(lines, conn, ops_db=ops_db)
        _emit_db_space_metrics(lines, db)
        _emit_raw_record_metrics(lines, conn, db_path=db)
        _emit_archive_source_index_link_metrics(lines, conn, db_path=db)

    finally:
        conn.close()

    return "\n".join(lines) + "\n"


def _format_ops_only_metrics(lines: list[str], ops_db: Path) -> str | None:
    attempts = _ops_attempt_counts(ops_db)
    durations = _ops_recent_attempt_durations(ops_db)
    debt = _ops_convergence_debt_by_stage(ops_db)
    memory = _ops_latest_ingest_memory(ops_db)
    if attempts is None and not durations and not debt and not memory:
        return None

    resolved_attempts = attempts or {"running": 0, "completed": 0, "failed": 0, "stale_cursor_writes": 0}
    _emit_metric(
        lines,
        name="polylogue_live_ingest_attempts_total",
        help_text="Total live ingest attempts by status.",
        metric_type="counter",
        samples=[
            ({"status": "completed"}, resolved_attempts["completed"]),
            ({"status": "failed"}, resolved_attempts["failed"]),
            ({"status": "running"}, resolved_attempts["running"]),
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_live_ingest_attempts_in_flight",
        help_text="Live ingest attempts currently running.",
        metric_type="gauge",
        samples=[(None, resolved_attempts["running"])],
    )
    _emit_metric(
        lines,
        name="polylogue_stale_cursor_writes_total",
        help_text="Total stale-cursor writes observed across ingest attempts.",
        metric_type="counter",
        samples=[(None, resolved_attempts["stale_cursor_writes"])],
    )
    _emit_storage_route_metrics(lines, _ops_storage_route_counts(ops_db) or {})
    _emit_metric(
        lines,
        name="polylogue_live_ingest_attempt_duration_seconds",
        help_text="Convergence time (seconds) of recent completed ingest attempts.",
        metric_type="gauge",
        samples=[
            ({"quantile": "min"}, min(durations)),
            ({"quantile": "mean"}, sum(durations) / len(durations)),
            ({"quantile": "max"}, max(durations)),
        ]
        if durations
        else [],
    )
    _emit_metric(
        lines,
        name="polylogue_convergence_debt_count",
        help_text="Unresolved convergence-debt rows by stage.",
        metric_type="gauge",
        samples=[({"stage": stage}, count) for stage, count in debt] if debt else [(None, 0)],
    )
    _emit_metric(
        lines,
        name="polylogue_live_ingest_memory_mebibytes",
        help_text="Latest live ingest memory sample in MiB by kind.",
        metric_type="gauge",
        samples=[({"kind": kind}, value) for kind, value in memory],
    )
    _emit_ops_throughput_metrics(lines, ops_db)
    for name, help_text in (
        ("polylogue_blob_lease_pending_count", "Pending blob leases in flight."),
        ("polylogue_blob_lease_distinct_operations", "Distinct lease-holding operations."),
        ("polylogue_fts_trigger_present", "1 when the named FTS sync trigger is installed in index.db."),
        ("polylogue_fts_triggers_all_present", "All expected FTS sync triggers are installed."),
        ("polylogue_fts_freshness_ready", "1 when the daemon freshness ledger marks an FTS surface ready."),
        ("polylogue_embedding_sessions", "Embedding session counts by state."),
        ("polylogue_embedding_messages", "Embedding message counts by state."),
        ("polylogue_embedding_coverage_percent", "Percent of sessions with current embeddings."),
        ("polylogue_embedding_status_state", "One-hot embedding materialization state for bounded scrapes."),
        ("polylogue_embedding_retrieval_ready", "Whether semantic retrieval has a materialized vector available."),
        ("polylogue_embedding_latest_catchup_run_info", "Latest embedding catch-up run status."),
        ("polylogue_embedding_latest_catchup_sessions", "Latest embedding catch-up run session counts."),
        ("polylogue_embedding_latest_catchup_messages", "Latest embedding catch-up run message counts."),
        (
            "polylogue_embedding_latest_catchup_estimated_cost_usd",
            "Latest embedding catch-up run estimated provider cost in USD.",
        ),
    ):
        _emit_metric(lines, name=name, help_text=help_text, metric_type="gauge", samples=[])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rich instrumentation (added for #1321 ambitious scope)
# ---------------------------------------------------------------------------


def _emit_archive_metrics(lines: list[str], conn: sqlite3.Connection) -> None:
    """Archive-level session and message counts by origin."""
    if _table_exists(conn, "sessions"):
        _emit_archive_index_metrics(lines, conn)
        return
    for name in ("polylogue_archive_sessions_total", "polylogue_archive_messages_total"):
        _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])


def _emit_archive_index_metrics(lines: list[str], conn: sqlite3.Connection) -> None:
    session_cols = _columns(conn, "sessions")
    if "origin" in session_cols:
        session_rows = conn.execute("SELECT origin, COUNT(*) FROM sessions GROUP BY origin ORDER BY origin").fetchall()
        session_samples: list[tuple[dict[str, str] | None, int | float]] = [
            ({"source": str(row[0])}, int(row[1])) for row in session_rows
        ]
    else:
        session_samples = [(None, _scalar_int(conn, "SELECT COUNT(*) FROM sessions"))]
    _emit_metric(
        lines,
        name="polylogue_archive_sessions_total",
        help_text="Total sessions in the archive by source family.",
        metric_type="gauge",
        samples=session_samples,
    )

    if "origin" in session_cols and "message_count" in session_cols:
        message_rows = conn.execute(
            """
            SELECT origin, COALESCE(SUM(message_count), 0)
            FROM sessions
            GROUP BY origin
            ORDER BY origin
            """
        ).fetchall()
        message_samples: list[tuple[dict[str, str] | None, int | float]] = [
            ({"source": str(row[0])}, int(row[1])) for row in message_rows
        ]
    elif _table_exists(conn, "messages"):
        message_cols = _columns(conn, "messages")
        if "origin" in message_cols:
            message_rows = conn.execute(
                "SELECT origin, COUNT(*) FROM messages GROUP BY origin ORDER BY origin"
            ).fetchall()
            message_samples = [({"source": str(row[0])}, int(row[1])) for row in message_rows]
        else:
            message_samples = [(None, _scalar_int(conn, "SELECT COUNT(*) FROM messages"))]
    else:
        message_samples = [(None, 0)]
    _emit_metric(
        lines,
        name="polylogue_archive_messages_total",
        help_text="Total messages in the archive by source family.",
        metric_type="gauge",
        samples=message_samples,
    )


def _emit_throughput_metrics(lines: list[str], conn: sqlite3.Connection, *, ops_db: Path | None = None) -> None:
    """Recent ingest throughput derived from live_ingest_attempt rows."""
    if ops_db is not None and _emit_ops_throughput_metrics(lines, ops_db):
        return
    if not _table_exists(conn, "live_ingest_attempt"):
        for name in (
            "polylogue_ingest_throughput_sessions_per_second",
            "polylogue_ingest_throughput_messages_per_second",
        ):
            _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])
        return

    cols = _columns(conn, "live_ingest_attempt")
    if "session_count" not in cols or "convergence_time_s" not in cols:
        return

    row = conn.execute(
        """
        SELECT session_count, message_count, convergence_time_s
        FROM live_ingest_attempt
        WHERE status = 'completed' AND convergence_time_s > 0
          AND session_count > 0
        ORDER BY started_at DESC
        LIMIT 1
        """
    ).fetchone()

    if row is not None:
        conv_count = int(row[0] or 0)
        msg_count = int(row[1] or 0)
        duration = max(float(row[2] or 0), 0.001)
        _emit_metric(
            lines,
            name="polylogue_ingest_throughput_sessions_per_second",
            help_text="Session throughput rate from the most recent completed ingest attempt.",
            metric_type="gauge",
            samples=[(None, conv_count / duration)],
        )
        _emit_metric(
            lines,
            name="polylogue_ingest_throughput_messages_per_second",
            help_text="Message throughput rate from the most recent completed ingest attempt.",
            metric_type="gauge",
            samples=[(None, msg_count / duration)],
        )


def _emit_ops_throughput_metrics(lines: list[str], ops_db: Path) -> bool:
    if not ops_db.exists():
        return False
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return False
            row = conn.execute(
                """
                SELECT parsed_raw_count, materialized_count, started_at_ms, finished_at_ms
                FROM ingest_attempts
                WHERE status = 'completed'
                  AND finished_at_ms IS NOT NULL
                  AND finished_at_ms > started_at_ms
                  AND (parsed_raw_count > 0 OR materialized_count > 0)
                ORDER BY finished_at_ms DESC, started_at_ms DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False

    if row is None:
        for name in (
            "polylogue_ingest_throughput_sessions_per_second",
            "polylogue_ingest_throughput_messages_per_second",
        ):
            _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])
        return True

    parsed_raw_count = int(row[0] or 0)
    materialized_count = int(row[1] or 0)
    duration = max((int(row[3]) - int(row[2])) / 1000.0, 0.001)
    _emit_metric(
        lines,
        name="polylogue_ingest_throughput_sessions_per_second",
        help_text="Source raw-row throughput rate from the most recent completed archive ingest attempt.",
        metric_type="gauge",
        samples=[(None, parsed_raw_count / duration)],
    )
    _emit_metric(
        lines,
        name="polylogue_ingest_throughput_messages_per_second",
        help_text="Materialized session throughput rate from the most recent completed archive ingest attempt.",
        metric_type="gauge",
        samples=[(None, materialized_count / duration)],
    )
    return True


def _emit_db_space_metrics(lines: list[str], db: Path) -> None:
    """Database file and page-level space metrics."""
    if not db.exists():
        for name in (
            "polylogue_db_file_size_bytes",
            "polylogue_db_allocated_bytes",
            "polylogue_db_freelist_bytes",
            "polylogue_db_page_size",
            "polylogue_db_page_count",
        ):
            _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])
        return

    try:
        file_size = db.stat().st_size
        _emit_metric(
            lines,
            name="polylogue_db_file_size_bytes",
            help_text="On-disk size of the archive SQLite database.",
            metric_type="gauge",
            samples=[(None, file_size)],
        )

        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        space_conn = open_readonly_connection(db)
        try:
            page_size = int(space_conn.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(space_conn.execute("PRAGMA page_count").fetchone()[0])
            freelist = int(space_conn.execute("PRAGMA freelist_count").fetchone()[0])
            allocated = page_size * page_count
            freelist_bytes = page_size * freelist

            _emit_metric(
                lines,
                name="polylogue_db_page_size",
                help_text="SQLite page size in bytes.",
                metric_type="gauge",
                samples=[(None, page_size)],
            )
            _emit_metric(
                lines,
                name="polylogue_db_page_count",
                help_text="Total SQLite pages in the database file.",
                metric_type="gauge",
                samples=[(None, page_count)],
            )
            _emit_metric(
                lines,
                name="polylogue_db_allocated_bytes",
                help_text="Allocated database space (page_size * page_count).",
                metric_type="gauge",
                samples=[(None, allocated)],
            )
            _emit_metric(
                lines,
                name="polylogue_db_freelist_bytes",
                help_text="Freelist (reusable) space in bytes.",
                metric_type="gauge",
                samples=[(None, freelist_bytes)],
            )
        finally:
            space_conn.close()
    except Exception:
        pass


def _emit_archive_storage_metrics(lines: list[str], db: Path) -> None:
    root = db.parent
    tier_paths = [
        (tier, root / filename, expected_version, backup_required)
        for tier, filename, expected_version, backup_required in _ARCHIVE_TIER_FILES
    ]
    present = {tier: path.exists() for tier, path, _expected_version, _backup_required in tier_paths}
    present_count = sum(1 for exists in present.values() if exists)
    missing_count = len(present) - present_count
    _emit_metric(
        lines,
        name="polylogue_archive_tier_present",
        help_text="1 when a archive database file exists.",
        metric_type="gauge",
        samples=[({"tier": tier}, 1 if exists else 0) for tier, exists in present.items()],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_tier_count",
        help_text="Archive file-set database inventory counts.",
        metric_type="gauge",
        samples=[
            ({"state": "present"}, present_count),
            ({"state": "missing"}, missing_count),
            ({"state": "expected"}, len(present)),
        ],
    )
    size_samples: list[tuple[dict[str, str] | None, float | int]] = []
    version_samples: list[tuple[dict[str, str] | None, float | int]] = []
    schema_mismatches: list[str] = []
    missing_backup_required: list[str] = []
    for tier, path, expected_version, backup_required in tier_paths:
        size_samples.append(({"tier": tier, "kind": "main"}, path.stat().st_size if path.exists() else 0))
        wal_path = Path(f"{path}-wal")
        size_samples.append(({"tier": tier, "kind": "wal"}, wal_path.stat().st_size if wal_path.exists() else 0))
        user_version = _archive_user_version(path) if path.exists() else 0
        version_samples.append(({"tier": tier}, user_version))
        if path.exists() and user_version != expected_version:
            schema_mismatches.append(tier)
        if not path.exists() and backup_required:
            missing_backup_required.append(tier)
    _emit_metric(
        lines,
        name="polylogue_archive_tier_file_size_bytes",
        help_text="On-disk size of each archive database file.",
        metric_type="gauge",
        samples=size_samples,
    )
    _emit_metric(
        lines,
        name="polylogue_archive_tier_user_version",
        help_text="SQLite user_version for each archive database file.",
        metric_type="gauge",
        samples=version_samples,
    )
    archive_ready = present["source"] and present["index"]
    final_shape_ready = all(present.values())
    storage_layout = archive_layout.classify_storage_layout(
        present_count=present_count, final_shape_ready=final_shape_ready
    )
    active_tier_role = archive_layout.active_tier_role(db, [(tier, path) for tier, path, _, _ in tier_paths])
    active_store = "archive_file_set" if archive_ready else "empty"
    blockers = archive_layout.archive_layout_blockers(
        present_count=present_count,
        final_shape_ready=final_shape_ready,
        schema_mismatches=schema_mismatches,
        missing_backup_required=missing_backup_required,
    )
    _emit_metric(
        lines,
        name="polylogue_archive_storage_layout",
        help_text="Current archive file-set layout shape.",
        metric_type="gauge",
        samples=[({"layout": layout}, 1 if layout == storage_layout else 0) for layout in ARCHIVE_STORAGE_LAYOUTS],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_storage_ready",
        help_text="Archive storage readiness states for daemon observability.",
        metric_type="gauge",
        samples=[
            ({"state": "archive_runtime"}, 1 if archive_ready else 0),
            ({"state": "final_shape"}, 1 if final_shape_ready else 0),
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_active_store",
        help_text="Active archive storage family for daemon reads and writes.",
        metric_type="gauge",
        samples=[({"store": store}, 1 if store == active_store else 0) for store in ("archive_file_set", "empty")],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_active_tier_role",
        help_text="Role of the database path used as the active metrics anchor.",
        metric_type="gauge",
        samples=[({"role": role}, 1 if role == active_tier_role else 0) for role in ARCHIVE_ACTIVE_TIER_ROLES],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_ready",
        help_text="1 when archive storage has no layout blockers.",
        metric_type="gauge",
        samples=[(None, 0 if blockers else 1)],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_blocker_count",
        help_text="Number of active archive layout blockers.",
        metric_type="gauge",
        samples=[(None, len(blockers))],
    )
    _emit_metric(
        lines,
        name="polylogue_archive_blocker",
        help_text="Archive layout blockers by bounded blocker label.",
        metric_type="gauge",
        samples=[({"blocker": blocker}, 1 if blocker in blockers else 0) for blocker in ARCHIVE_LAYOUT_BLOCKER_LABELS],
    )


def _emit_archive_source_index_link_metrics(
    lines: list[str],
    conn: sqlite3.Connection,
    *,
    db_path: Path,
) -> None:
    if not _table_exists(conn, "sessions"):
        _emit_metric(
            lines,
            name="polylogue_archive_source_index_links_total",
            help_text="Raw source rows by index materialization state.",
            metric_type="gauge",
            samples=[],
        )
        return

    session_cols = _columns(conn, "sessions")
    if "raw_id" not in session_cols:
        _emit_metric(
            lines,
            name="polylogue_archive_source_index_links_total",
            help_text="Raw source rows by index materialization state.",
            metric_type="gauge",
            samples=[],
        )
        return

    source_db = db_path.with_name("source.db")
    if not source_db.exists():
        raw_links = _scalar_int(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
        _emit_metric(
            lines,
            name="polylogue_archive_source_index_links_total",
            help_text="Raw source rows by index materialization state.",
            metric_type="gauge",
            samples=[({"source": "unknown", "state": "source_db_missing"}, raw_links)],
        )
        return

    source_uri = f"file:{source_db}?mode=ro"
    conn.execute("ATTACH DATABASE ? AS source_metrics", (source_uri,))
    try:
        source_table = conn.execute(
            "SELECT 1 FROM source_metrics.sqlite_master WHERE type='table' AND name='raw_sessions'"
        ).fetchone()
        if source_table is None:
            samples: list[tuple[dict[str, str] | None, float | int]] = []
        else:
            rows = conn.execute(
                """
                SELECT
                    r.origin AS source,
                    COUNT(*) AS acquired_raw,
                    SUM(CASE WHEN EXISTS (
                        SELECT 1 FROM sessions AS s WHERE s.raw_id = r.raw_id
                    ) THEN 1 ELSE 0 END) AS indexed_raw,
                    SUM(CASE WHEN NOT EXISTS (
                        SELECT 1 FROM sessions AS s WHERE s.raw_id = r.raw_id
                    ) THEN 1 ELSE 0 END) AS pending_index
                FROM source_metrics.raw_sessions AS r
                GROUP BY r.origin
                ORDER BY r.origin
                """
            ).fetchall()
            samples = []
            for row in rows:
                source = str(row[0])
                samples.extend(
                    [
                        ({"source": source, "state": "acquired_raw"}, int(row[1] or 0)),
                        ({"source": source, "state": "indexed_raw"}, int(row[2] or 0)),
                        ({"source": source, "state": "pending_index"}, int(row[3] or 0)),
                    ]
                )
            orphan_rows = conn.execute(
                """
                SELECT s.origin AS source, COUNT(*) AS orphan_index_link
                FROM sessions AS s
                WHERE s.raw_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM source_metrics.raw_sessions AS r WHERE r.raw_id = s.raw_id
                  )
                GROUP BY s.origin
                ORDER BY s.origin
                """
            ).fetchall()
            samples.extend(
                ({"source": str(row[0]), "state": "orphan_index_link"}, int(row[1] or 0)) for row in orphan_rows
            )
    finally:
        conn.execute("DETACH DATABASE source_metrics")

    _emit_metric(
        lines,
        name="polylogue_archive_source_index_links_total",
        help_text="Raw source rows by index materialization state.",
        metric_type="gauge",
        samples=samples,
    )


def _archive_user_version(path: Path) -> int:
    try:
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        conn = open_readonly_connection(path)
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _emit_raw_record_metrics(lines: list[str], conn: sqlite3.Connection, *, db_path: Path | None = None) -> None:
    """Raw session record counts and parse health."""
    if _table_exists(conn, "raw_sessions"):
        _emit_archive_raw_record_metrics(lines, conn)
        return
    if db_path is not None:
        source_db = db_path.with_name("source.db")
        if source_db.exists():
            try:
                from polylogue.storage.sqlite.connection_profile import open_readonly_connection

                source_conn = open_readonly_connection(source_db)
                try:
                    if _table_exists(source_conn, "raw_sessions"):
                        _emit_archive_raw_record_metrics(lines, source_conn)
                        return
                finally:
                    source_conn.close()
            except sqlite3.Error:
                pass
    if not _table_exists(conn, "raw_sessions"):
        _emit_metric(
            lines, name="polylogue_raw_records_total", help_text="Total raw records.", metric_type="gauge", samples=[]
        )
        return


def _emit_archive_raw_record_metrics(lines: list[str], conn: sqlite3.Connection) -> None:
    counts = conn.execute(
        """SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN parsed_at_ms IS NOT NULL THEN 1 ELSE 0 END) AS parsed,
            SUM(CASE WHEN validated_at_ms IS NOT NULL THEN 1 ELSE 0 END) AS validated,
            SUM(CASE WHEN parse_error IS NOT NULL OR validation_status = 'failed'
                THEN 1 ELSE 0 END) AS errors
        FROM raw_sessions"""
    ).fetchone()
    total = counts[0]
    parsed = counts[1] or 0
    validated = counts[2] or 0
    with_errors = counts[3] or 0
    _emit_metric(
        lines,
        name="polylogue_raw_records_total",
        help_text="Total raw session records in the archive.",
        metric_type="gauge",
        samples=[
            ({"state": "total"}, total),
            ({"state": "parsed"}, parsed),
            ({"state": "validated"}, validated),
            ({"state": "errors"}, with_errors),
        ],
    )

    raw_cols = _columns(conn, "raw_sessions")
    if "origin" in raw_cols:
        rows = conn.execute("SELECT origin, COUNT(*) FROM raw_sessions GROUP BY origin ORDER BY origin").fetchall()
        _emit_metric(
            lines,
            name="polylogue_raw_records_by_source",
            help_text="Raw session records by source family.",
            metric_type="gauge",
            samples=[({"source": str(row[0])}, int(row[1])) for row in rows],
        )


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


def handle_metrics(responder: MetricsResponder, db: Path) -> None:
    """Serve ``GET /metrics`` — Prometheus text exposition.

    Errors degrade to a single ``polylogue_daemon_metrics_collection_error``
    sample (gauge value 1) so the scrape still succeeds and the operator
    sees the failure as a series rather than a 5xx page in Prometheus.
    """
    try:
        body = format_metrics(db)
    except Exception as exc:
        logger.exception("metrics collection failed")
        body = (
            "# HELP polylogue_daemon_metrics_collection_error Metrics collection failed.\n"
            "# TYPE polylogue_daemon_metrics_collection_error gauge\n"
            f'polylogue_daemon_metrics_collection_error{{error="{_escape_label_value(str(exc))}"}} 1\n'
        )
    responder._send_text(HTTPStatus.OK, body, content_type=PROMETHEUS_CONTENT_TYPE)


__all__ = [
    "PROMETHEUS_CONTENT_TYPE",
    "MetricsResponder",
    "format_metrics",
    "handle_metrics",
]
