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
- **Read-only.** Archive series are sourced from SQLite via
  ``open_readonly_connection``; process-local I/O phase counters are read
  from their bounded in-memory owner. The endpoint never
  writes and never blocks on a held writer.
- **Resilient to missing tables.** Each section gracefully degrades
  to zero / absent series when a backing table does not yet exist
  (fresh archives, schema bumps mid-rollout).
Exposed series (label policy: use labels only when a series naturally
varies on a known-bounded dimension):

- ``polylogue_daemon_uptime_seconds`` (gauge) — process uptime
- ``polylogue_daemon_build_info`` (gauge, value 1) — labels: version,
  revision, dirty. ``revision`` is the full (not truncated) git commit
  the running build was compiled from — join it against a consuming
  flake's ``flake.lock`` ``rev``/``narHash`` entry for the ``polylogue``
  input to attest that the deployed runtime matches the locked source
  (polylogue-6rvt).
- ``polylogue_status_snapshot_age_seconds`` (gauge) — cached status age
- ``polylogue_status_snapshot_state`` (gauge) — labels: state
- ``polylogue_detached_writer_failures_total`` (counter) — process-lifetime
  count of detached background daemon-writer tasks that raised (polylogue-es7b)
- ``polylogue_diagnostic_delivery_total`` (counter) — labels: outcome
- ``polylogue_diagnostic_queue_depth`` (gauge)
- ``polylogue_storage_io_phase_total`` (counter) — labels: tier, phase, inside_writer_lease, succeeded
- ``polylogue_storage_io_phase_seconds_total`` (counter) — same labels
- ``polylogue_storage_io_phase_observable`` (gauge) — labels: phase
- ``polylogue_live_ingest_attempts_total`` (counter) — labels: status
- ``polylogue_live_ingest_attempts_in_flight`` (gauge)
- ``polylogue_live_ingest_storage_route_total`` (counter) — labels: route
- ``polylogue_live_ingest_attempt_duration_seconds`` (gauge buckets:
  min, mean, max derived from recent completed attempts)
- ``polylogue_convergence_debt_count`` (gauge) — labels: stage, status
- ``polylogue_fts_trigger_present`` (gauge) — labels: trigger
- ``polylogue_fts_triggers_all_present`` (gauge, 0/1)
- ``polylogue_fts_freshness_ready`` (gauge, 0/1) — labels: surface
- ``polylogue_fts_drift_rows`` (gauge) — labels: surface, kind
  (missing/excess/duplicate/identity_mismatch), read from the current
  authoritative FTS membership inspection.
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
- ``polylogue_db_wal_file_size_bytes`` (gauge)
- ``polylogue_db_sqlite_stat1_rows`` (gauge)
- ``polylogue_archive_tier_user_version`` (gauge) — labels: tier
- ``polylogue_archive_storage_layout`` (gauge) — labels: layout
- ``polylogue_archive_storage_ready`` (gauge) — labels: state
- ``polylogue_archive_active_store`` (gauge) — labels: store
- ``polylogue_archive_active_tier_role`` (gauge) — labels: role
- ``polylogue_archive_ready`` (gauge, 0/1)
- ``polylogue_archive_blocker_count`` (gauge)
- ``polylogue_archive_blocker`` (gauge) — labels: blocker
- ``polylogue_archive_source_index_links_total`` (gauge) — labels: source, state
- ``polylogue_hook_flow_healthy`` (gauge, 0/1) — labels: harness
- ``polylogue_hook_flow_state`` (gauge, one-hot) — labels: harness, state
- ``polylogue_hook_sessions`` (gauge) — labels: harness, state
- ``polylogue_hook_event_observed_ratio`` (gauge) — labels: harness, event

The handler signature mirrors ``healthz.py``'s ``ProbeResponder``
protocol so it is testable without the full ``BaseHTTPRequestHandler``
stack.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Protocol, TypedDict

from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.daemon.process_start import uptime_seconds
from polylogue.logging import WARNING, diagnostic_snapshot, emit
from polylogue.operations.storage_io_observation import IoPhaseObservation, storage_io_observation
from polylogue.storage import archive_layout
from polylogue.storage.archive_layout import (
    ARCHIVE_ACTIVE_TIER_ROLES,
    ARCHIVE_LAYOUT_BLOCKER_LABELS,
    ARCHIVE_STORAGE_LAYOUTS,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

# Derived from the canonical tier specs so the expected schema version per tier
# can never drift from ARCHIVE_VERSION_BY_TIER. (tier, filename, expected_version,
# backup_required), preserving the canonical archive-tier order.
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


def _collection_reason(exc: Exception) -> str:
    if isinstance(exc, sqlite3.DatabaseError):
        return "archive_unreadable"
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return "schema_unavailable"
    return "collector_failed"


def _collect_group(
    lines: list[str],
    states: list[tuple[dict[str, str], int]],
    group: str,
    collect: Callable[[list[str]], object],
    *,
    path: Path | None = None,
) -> None:
    """Append a complete group, or report its absence without exposing exception text."""
    pending: list[str] = []
    try:
        collect(pending)
    except Exception as exc:
        reason = "collector_failed" if group.startswith("process_") else _collection_reason(exc)
        emit(
            "daemon.metrics.collection_failed",
            level=WARNING,
            outcome="degraded",
            reason=reason,
            path=path,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        available = 0
    else:
        lines.extend(pending)
        reason = "none"
        available = 1
    states.append(({"group": group, "reason": reason}, available))


class EmbeddingMetricState(TypedDict):
    total_sessions: int
    embedded_sessions: int
    pending_sessions: int
    failed_sessions: int
    embedded_messages: int
    # None when no sessions are eligible for embedding -- a genuine
    # measurement gap, not a measured percentage (polylogue-oitx).
    coverage_percent: float | None
    status: str
    retrieval_ready: int
    latest_status: str | None
    latest_rebuild: str
    latest_planned_sessions: int | None
    latest_processed_sessions: int
    latest_embedded_sessions: int
    latest_skipped_sessions: int
    latest_error_count: int
    latest_planned_messages: int | None
    latest_embedded_messages: int
    latest_estimated_cost_usd: float


class ArchiveEmbeddingRunState(TypedDict):
    status: str
    scanned_sessions: int
    embedded_sessions: int
    skipped_sessions: int
    embedded_messages: int
    error_count: int
    estimated_cost_usd: float

    # ``processed`` is the successful work denominator, not the number of
    # rows scanned/planned.  Keep this explicit so a capped or failed run
    # cannot publish the scan count under two different labels.


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


def _emit_periodic_loop_metrics(lines: list[str]) -> None:
    """Expose every daemon cadence loop's last run, due time and failures.

    Before the shared runner there was nothing here at all: a loop that had
    silently stopped ticking was indistinguishable from one whose archive had
    no work (polylogue-74wvj).
    """
    from polylogue.daemon.periodic import daemon_periodic_runner

    states = daemon_periodic_runner().snapshot()
    now = time.time()
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_interval_seconds",
        help_text="Declared cadence of each daemon periodic loop.",
        metric_type="gauge",
        samples=[({"loop": state.name}, state.interval_s) for state in states],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_age_seconds",
        help_text="Seconds since each daemon periodic loop last completed a pass.",
        metric_type="gauge",
        samples=[
            ({"loop": state.name}, now - state.last_run_completed_at)
            for state in states
            if state.last_run_completed_at is not None
        ],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_due_in_seconds",
        help_text="Seconds until each daemon periodic loop's next scheduled pass.",
        metric_type="gauge",
        samples=[({"loop": state.name}, state.next_run_at - now) for state in states if state.next_run_at is not None],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_runs_total",
        help_text="Completed passes per daemon periodic loop since process start.",
        metric_type="counter",
        samples=[({"loop": state.name}, state.runs) for state in states],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_failures_total",
        help_text="Recorded failures per daemon periodic loop since process start.",
        metric_type="counter",
        samples=[({"loop": state.name}, state.failures) for state in states],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_periodic_loop_blocked",
        help_text="1 while a daemon periodic loop is waiting on a named startup gate.",
        metric_type="gauge",
        samples=[({"loop": state.name}, 1 if state.blocked_on else 0) for state in states],
    )


_UNMEASURED_PROBE_METRIC = "polylogue_probe_unmeasured"


def _emit_unmeasured_probe(lines: list[str], probe: str) -> None:
    """Record a probe that could not be measured at all.

    polylogue-xvwpi: ``_emit_metric`` renders an empty sample list as
    ``<name> 0``, so every failed probe published a healthy-looking zero
    series indistinguishable from a true zero. An unmeasurable probe emits no
    value series at all -- absence is the only honest Prometheus reading --
    and raises this companion gauge so the gap is visible rather than silent.
    """

    header = f"# HELP {_UNMEASURED_PROBE_METRIC} "
    if not any(line.startswith(header) for line in lines):
        lines.append(f"{header}1 when a named status probe could not be measured on this scrape.")
        lines.append(f"# TYPE {_UNMEASURED_PROBE_METRIC} gauge")
    lines.append(f'{_UNMEASURED_PROBE_METRIC}{{probe="{probe}"}} 1')


def _convergence_debt_measurable(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> bool:
    """Whether a zero debt count would be a measurement rather than a guess."""

    if ops_db is not None and ops_db.exists():
        from polylogue.daemon.convergence_debt_status import convergence_debt_stage_counts_info

        if convergence_debt_stage_counts_info(ops_db, ops_db=ops_db).available:
            return True
    return _table_exists(conn, "live_convergence_debt")


def _emit_metric(
    lines: list[str],
    *,
    name: str,
    help_text: str,
    metric_type: str,
    samples: Sequence[tuple[dict[str, str] | None, float | int]],
    omit_when_empty: bool = False,
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    if not samples:
        if omit_when_empty:
            # The zero below is a *count* reading. For a gauge whose samples are
            # filtered by "has this been measured yet", that zero asserts the
            # measurement (an age of 0s reads as "just ran"), which is the
            # fabrication polylogue-xvwpi removes elsewhere. Absence is the only
            # honest Prometheus reading for an unmeasured gauge.
            return
        # Emit a zero sample so the series is discoverable even when no
        # backing rows exist yet.
        lines.append(f"{name} 0")
        return
    for labels, value in samples:
        lines.append(f"{name}{_render_labels(labels)} {_format_value(value)}")


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------


def _attached_table_exists(conn: sqlite3.Connection, schema_name: str, table: str) -> bool:
    if not schema_name.replace("_", "").isalnum() or not table.replace("_", "").isalnum():
        return False
    try:
        return _table_exists(conn, table, schema=schema_name)
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.probe_failed",
            level=WARNING,
            outcome="degraded",
            reason="attached_table_unreadable",
            schema_name=schema_name,
            table_name=table,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return False


def _attached_table_name(conn: sqlite3.Connection, schema_name: str, table: str) -> str:
    if _attached_table_exists(conn, schema_name, table):
        return f"{schema_name}.{table}"
    return ""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _qualified_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if "." not in table:
        return _columns(conn, table)
    schema, _, name = table.rpartition(".")
    if not schema.replace("_", "").isalnum() or not name.replace("_", "").isalnum():
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA {schema}.table_info({name})")}


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
        conn = open_readonly_connection(ops_db, validate_schema=False)
        try:
            if not _table_exists(conn, "ingest_attempts"):
                return None
            rows = conn.execute("SELECT status, COUNT(*) FROM ingest_attempts GROUP BY status").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="attempt_counts_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
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
        conn = open_readonly_connection(ops_db, validate_schema=False)
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
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="attempt_durations_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return []
    return [max(0.0, (int(row[1]) - int(row[0])) / 1000.0) for row in rows if row[0] is not None and row[1] is not None]


def _convergence_debt_by_stage(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> list[tuple[str, str, int]]:
    ops_rows = _ops_convergence_debt_by_stage(ops_db) if ops_db is not None else []
    if ops_rows:
        return ops_rows
    if not _table_exists(conn, "live_convergence_debt"):
        return []
    rows = conn.execute(
        """
        SELECT stage, status, COUNT(*)
        FROM live_convergence_debt
        WHERE status IN ('failed', 'deferred')
        GROUP BY stage, status
        ORDER BY stage, status
        """
    ).fetchall()
    return [(str(row[0] or "unknown"), str(row[1] or "unknown"), int(row[2] or 0)) for row in rows]


def _ops_convergence_debt_by_stage(ops_db: Path | None) -> list[tuple[str, str, int]]:
    """Return (stage, status, count) triples from the durable ops-tier ledger.

    Delegates to :func:`convergence_debt_status.convergence_debt_stage_counts_info`,
    which answers this question with one ``GROUP BY stage, status`` aggregate.
    It deliberately does NOT use ``convergence_debt_summary_info``: that
    projection selects every convergence-debt row, ``target_id`` and
    ``last_error`` strings included, so routing a ``/metrics`` scrape through it
    materialized the entire ledger to produce a handful of counters.

    The validation that made the summary projection worth delegating to is
    retained: both enforce the closed ``{failed, deferred}`` status vocabulary
    and a non-NULL ``stage``, and both surface a violation as ``available=False``
    (caught, logged, empty metrics) rather than passing an anomalous value
    through as an ``"unknown"`` bucket.
    """
    if ops_db is None or not ops_db.exists():
        return []
    from polylogue.daemon.convergence_debt_status import convergence_debt_stage_counts_info

    stage_counts = convergence_debt_stage_counts_info(ops_db, ops_db=ops_db)
    if not stage_counts.available:
        return []
    rows = [(stage, status, count) for stage, status, count in stage_counts.counts if count]
    rows.sort()
    return rows


def _fts_trigger_presence(conn: sqlite3.Connection) -> dict[str, bool]:
    from polylogue.storage.fts.derivation import active_fts_triggers_sync

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


def _fts_surface_metrics(
    conn: sqlite3.Connection,
) -> tuple[list[tuple[str, int]], list[tuple[str, str, int]]]:
    """Project readiness and drift from one authoritative domain inspection."""
    from polylogue.daemon.fts_status import fts_readiness_info

    database_row = conn.execute("PRAGMA database_list").fetchone()
    if database_row is None:
        return [], []
    readiness = fts_readiness_info(Path(str(database_row[2])))
    surfaces = readiness.get("surfaces")
    if not isinstance(surfaces, dict):
        return [], []
    ready_samples: list[tuple[str, int]] = []
    drift_samples: list[tuple[str, str, int]] = []
    for surface, payload in sorted(surfaces.items()):
        if not isinstance(payload, dict):
            continue
        ready_samples.append((str(surface), int(bool(payload.get("ready")))))
        for kind, column in (
            ("missing", "missing_rows"),
            ("excess", "excess_rows"),
            ("duplicate", "duplicate_rows"),
            ("identity_mismatch", "identity_mismatch_rows"),
        ):
            value = payload.get(column)
            if isinstance(value, int) and not isinstance(value, bool):
                drift_samples.append((str(surface), kind, value))
    return ready_samples, drift_samples


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
        conn = open_readonly_connection(ops_db, validate_schema=False)
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
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="ingest_memory_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
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
        conn = open_readonly_connection(ops_db, validate_schema=False)
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
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="storage_route_counts_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None


def _emit_storage_route_metrics(lines: list[str], counts: dict[str, int]) -> None:
    _emit_metric(
        lines,
        name="polylogue_live_ingest_storage_route_total",
        help_text="Live ingest attempts grouped by bounded storage route.",
        metric_type="counter",
        samples=[({"route": route}, count) for route, count in sorted(counts.items())],
    )


def _embedding_message_count(conn: sqlite3.Connection, *, status_table: str = "", meta_table: str = "") -> int:
    # message_embedding_refs is the per-message count (polylogue-q88p);
    # message_embeddings/message_embeddings_meta are content-addressed and
    # deduped, so their row count undercounts messages whenever identical
    # text is shared across sessions -- only fall back to them for legacy
    # (pre-v4) archives that predate the refs table.
    if _table_exists(conn, "message_embedding_refs"):
        return _scalar_int(conn, "SELECT COUNT(*) FROM message_embedding_refs")
    if _table_exists(conn, "message_embeddings_rowids"):
        return _scalar_int(conn, "SELECT COUNT(*) FROM message_embeddings_rowids")
    if _table_exists(conn, "message_embeddings"):
        return _scalar_int(conn, "SELECT COUNT(*) FROM message_embeddings")
    if status_table and "message_count_embedded" in _qualified_columns(conn, status_table):
        return _scalar_int(conn, f"SELECT COALESCE(SUM(message_count_embedded), 0) FROM {status_table}")
    if meta_table:
        return _scalar_int(conn, f"SELECT COUNT(*) FROM {meta_table}")
    return 0


def _archive_embedding_state(conn: sqlite3.Connection, *, ops_db: Path | None = None) -> EmbeddingMetricState:
    sessions_present = _table_exists(conn, "sessions")
    total_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM sessions") if sessions_present else 0
    embedded_sessions = 0
    pending_sessions = 0
    failed_sessions = 0
    status_table = "embedding_status" if _table_exists(conn, "embedding_status") else ""
    meta_table = "message_embeddings_meta" if _table_exists(conn, "message_embeddings_meta") else ""
    if ops_db is not None:
        embeddings_db = ops_db.parent / "embeddings.db"
    else:
        embeddings_db = Path(conn.execute("PRAGMA database_list").fetchone()[2]).with_name("embeddings.db")
    if not status_table and embeddings_db.exists():
        from polylogue.storage.sqlite.connection_profile import attach_readonly_database

        attach_readonly_database(conn, embeddings_db, alias="embeddings")
        status_table = _attached_table_name(conn, "embeddings", "embedding_status")
        meta_table = _attached_table_name(conn, "embeddings", "message_embeddings_meta")

    if status_table:
        embedded_sessions = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM {status_table}
            WHERE COALESCE(needs_reindex, 0) = 0
              AND error_message IS NULL
            """,
        )
        if sessions_present:
            pending_sessions = max(total_sessions - embedded_sessions, 0)
        else:
            # A partial index without a sessions table can still name what the
            # status table itself says is owed.
            pending_sessions = _scalar_int(
                conn,
                f"SELECT COUNT(*) FROM {status_table} WHERE COALESCE(needs_reindex, 0) <> 0 AND error_message IS NULL",
            )
        failed_sessions = _scalar_int(
            conn,
            f"SELECT COUNT(*) FROM {status_table} WHERE error_message IS NOT NULL",
        )
    elif _table_exists(conn, "messages"):
        from polylogue.storage.embeddings.materialization import count_archive_embedding_session_state

        session_state = count_archive_embedding_session_state(conn, status_table="", rebuild=False)
        pending_sessions = session_state.pending_sessions

    embedded_messages = _embedding_message_count(conn, status_table=status_table, meta_table=meta_table)
    eligible_sessions = embedded_sessions + pending_sessions
    coverage_percent: float | None
    if eligible_sessions > 0:
        coverage_percent = embedded_sessions / eligible_sessions * 100
    elif total_sessions > 0:
        # Sessions exist, but none are eligible for embedding (e.g. the
        # embedding schema branch was never queried) -- this is a genuine
        # measurement gap, not "fully embedded". Report unmeasured (None,
        # emitted as NaN on the Prometheus gauge) rather than a fabricated
        # 100.0 that an alerting pipeline would read as healthy
        # (polylogue-oitx).
        coverage_percent = None
    else:
        # No sessions at all: there is nothing to embed, and ``status``
        # below reports "empty" for this case.
        coverage_percent = 0.0
    if total_sessions <= 0 and pending_sessions <= 0 and embedded_messages <= 0:
        status = "empty"
    elif pending_sessions <= 0:
        status = "complete"
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
        # The archive-tier receipt records how many sessions were actually
        # visited (``scanned_sessions``), not the size of the pending window
        # before the run started.  It therefore cannot support a planned
        # denominator: omitting the series is more honest than publishing the
        # same quantity under both labels.
        "latest_planned_sessions": None if latest is not None else 0,
        "latest_processed_sessions": (
            latest["embedded_sessions"] + latest["skipped_sessions"] if latest is not None else 0
        ),
        "latest_embedded_sessions": latest["embedded_sessions"] if latest is not None else 0,
        "latest_skipped_sessions": latest["skipped_sessions"] if latest is not None else 0,
        "latest_error_count": latest["error_count"] if latest is not None else 0,
        "latest_planned_messages": None,
        "latest_embedded_messages": latest["embedded_messages"] if latest is not None else 0,
        "latest_estimated_cost_usd": latest["estimated_cost_usd"] if latest is not None else 0.0,
    }


def _archive_latest_embedding_run_state(ops_db: Path | None) -> ArchiveEmbeddingRunState | None:
    if ops_db is None or not ops_db.exists():
        return None
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db, validate_schema=False)
        try:
            if not _table_exists(conn, "embedding_catchup_runs"):
                return None
            runs = list_embedding_catchup_runs(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="latest_embedding_run_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "status": run.status,
        "scanned_sessions": run.scanned_sessions,
        "embedded_sessions": run.embedded_sessions,
        "skipped_sessions": run.skipped_sessions,
        "embedded_messages": run.embedded_messages,
        "error_count": run.error_count,
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
    coverage_percent = state["coverage_percent"]
    _emit_metric(
        lines,
        name="polylogue_embedding_coverage_percent",
        help_text=(
            "Percent of sessions with current embeddings. NaN when no sessions "
            "are eligible for embedding (a measurement gap, not 0% or 100%)."
        ),
        metric_type="gauge",
        samples=[(None, float("nan") if coverage_percent is None else float(coverage_percent))],
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
            *(
                [({"state": "planned"}, int(state["latest_planned_sessions"]))]
                if state["latest_planned_sessions"] is not None
                else []
            ),
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
            *(
                [({"state": "planned"}, int(state["latest_planned_messages"]))]
                if state["latest_planned_messages"] is not None
                else []
            ),
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


def _emit_cache_metrics(lines: list[str]) -> None:
    from polylogue.core.result_cache_metrics import result_cache_metrics

    cache_stats = result_cache_metrics()
    _emit_metric(
        lines,
        name="polylogue_daemon_result_cache_entries",
        help_text="Resident daemon read-result cache entries.",
        metric_type="gauge",
        samples=[(None, cache_stats["entries"])],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_result_cache_bytes",
        help_text="Resident daemon read-result cache bytes.",
        metric_type="gauge",
        samples=[(None, cache_stats["bytes"])],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_result_cache_hits_total",
        help_text="Daemon read-result cache hits since process start.",
        metric_type="counter",
        samples=[(None, cache_stats["hits"])],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_result_cache_misses_total",
        help_text="Daemon read-result cache misses since process start.",
        metric_type="counter",
        samples=[(None, cache_stats["misses"])],
    )
    _emit_metric(
        lines,
        name="polylogue_daemon_result_cache_evictions_total",
        help_text="Daemon read-result cache evictions since process start.",
        metric_type="counter",
        samples=[(None, cache_stats["evictions"])],
    )


def _emit_build_metrics(lines: list[str]) -> None:
    # Build info — version/revision/dirty labels so dashboards can break out
    # per-release and operators can attest a running daemon against a
    # consuming flake's locked `rev`/`narHash` for this input (polylogue-6rvt).
    try:
        from polylogue.version import VERSION_INFO

        build_version = VERSION_INFO.version
        build_revision = VERSION_INFO.commit or "unknown"
        build_dirty = VERSION_INFO.dirty
    except Exception:
        # polylogue-xvwpi: "dirty=false" is an attestation, and this branch
        # never read the working tree. An unread build identity says so on
        # every label rather than asserting a clean build.
        build_version = "unknown"
        build_revision = "unknown"
        build_dirty = None
    _emit_metric(
        lines,
        name="polylogue_daemon_build_info",
        help_text="Constant 1 gauge labelled with daemon build identity.",
        metric_type="gauge",
        samples=[
            (
                {
                    "version": str(build_version),
                    "revision": str(build_revision),
                    "dirty": "unknown" if build_dirty is None else ("true" if build_dirty else "false"),
                },
                1,
            )
        ],
    )


def _emit_status_metrics(lines: list[str]) -> None:
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


def _emit_writer_metrics(lines: list[str]) -> None:
    from polylogue.daemon.write_coordinator import daemon_write_telemetry_payload

    write_telemetry = daemon_write_telemetry_payload()
    detached_writer_failures = write_telemetry.get("detached_writer_failures", 0)
    _emit_metric(
        lines,
        name="polylogue_detached_writer_failures_total",
        help_text=(
            "Total detached background daemon-writer tasks that raised an exception "
            "since process start (polylogue-es7b)."
        ),
        metric_type="counter",
        samples=[(None, int(detached_writer_failures) if isinstance(detached_writer_failures, (int, float)) else 0)],
    )


def _emit_diagnostic_metrics(lines: list[str]) -> None:
    diagnostic = diagnostic_snapshot()
    _emit_metric(
        lines,
        name="polylogue_diagnostic_delivery_total",
        help_text="Process-local diagnostic records delivered, dropped, failed, or left undrained at shutdown.",
        metric_type="counter",
        samples=[
            ({"outcome": outcome}, diagnostic[outcome]) for outcome in ("delivered", "dropped", "failures", "undrained")
        ],
    )
    _emit_metric(
        lines,
        name="polylogue_diagnostic_queue_depth",
        help_text="Diagnostic records waiting for the configured sink.",
        metric_type="gauge",
        samples=[(None, diagnostic["queued"])],
    )


def _emit_io_metrics(lines: list[str]) -> None:
    io_observation = storage_io_observation()

    def io_labels(sample: IoPhaseObservation) -> dict[str, str]:
        return {
            "tier": sample.tier,
            "phase": sample.phase,
            "inside_writer_lease": "true" if sample.inside_writer_lease else "false",
            "succeeded": "true" if sample.succeeded else "false",
        }

    _emit_metric(
        lines,
        name="polylogue_storage_io_phase_total",
        help_text="Observed storage I/O phase calls by tier, phase, lease ownership, and outcome.",
        metric_type="counter",
        samples=[(io_labels(sample), sample.count) for sample in io_observation.samples],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_storage_io_phase_seconds_total",
        help_text="Observed wall time in storage I/O phases, in seconds.",
        metric_type="counter",
        samples=[(io_labels(sample), sample.elapsed_ns / 1_000_000_000) for sample in io_observation.samples],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_storage_io_phase_observable",
        help_text="1 for measured outer I/O phases; 0 for SQLite-internal phases unavailable to this process.",
        metric_type="gauge",
        samples=[
            ({"phase": phase}, 1)
            for phase in (
                "connection_create",
                "begin",
                "commit",
                "rollback",
                "checkpoint",
                "blob_file_fsync",
                "blob_directory_fsync",
            )
        ]
        + [({"phase": phase}, 0) for phase in io_observation.unavailable_phases],
    )


def _process_metric_lines(*, now_monotonic: float | None = None) -> tuple[list[str], list[tuple[dict[str, str], int]]]:
    lines: list[str] = []
    states: list[tuple[dict[str, str], int]] = []
    _collect_group(lines, states, "process_cache", _emit_cache_metrics)

    def emit_uptime(group: list[str]) -> None:
        _emit_metric(
            group,
            name="polylogue_daemon_uptime_seconds",
            help_text="Daemon process uptime in seconds.",
            metric_type="gauge",
            samples=[(None, uptime_seconds(now_monotonic=now_monotonic))],
        )

    _collect_group(
        lines,
        states,
        "process_uptime",
        emit_uptime,
    )

    _collect_group(lines, states, "process_build", _emit_build_metrics)
    _collect_group(lines, states, "process_status", _emit_status_metrics)
    _collect_group(lines, states, "process_writer", _emit_writer_metrics)
    _collect_group(lines, states, "process_diagnostic", _emit_diagnostic_metrics)
    _collect_group(lines, states, "process_io", _emit_io_metrics)
    _collect_group(lines, states, "process_periodic", _emit_periodic_loop_metrics)
    return lines, states


def format_metrics(
    db: Path,
    *,
    now_monotonic: float | None = None,
) -> str:
    """Render the Prometheus text exposition for the daemon archive at ``db``.

    Process counters remain available when an archive metric group fails.
    Missing archive measurements carry collection availability evidence.
    Caller injects ``now_monotonic`` in tests to keep uptime stable.
    """
    from polylogue.storage.archive_identity import ArchiveLocation

    lines, states = _process_metric_lines(now_monotonic=now_monotonic)
    # ``configured_root`` stays fixed at db's original parent even though
    # ``db`` itself gets reassigned below to the active index path -- an
    # index-only external generation is explicitly allowed to have no
    # sibling tiers of its own, so ops.db/source.db/embeddings.db/user.db
    # must always resolve against the configured root, never against
    # wherever the active index generation happens to physically live.
    configured_root = db.parent
    resolution_reason: str | None = None
    try:
        index_db = ArchiveLocation.resolve(configured_root).active_index_path
        if index_db.exists():
            db = index_db
    except Exception as exc:
        resolution_reason = _collection_reason(exc)
        emit(
            "daemon.metrics.collection_failed",
            level=WARNING,
            outcome="degraded",
            reason=resolution_reason,
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        states.append(({"group": "archive_resolution", "reason": resolution_reason}, 0))

    _collect_group(
        lines,
        states,
        "archive_storage",
        lambda group: _emit_archive_storage_metrics(group, db, configured_root=configured_root),
        path=db,
    )
    _collect_group(lines, states, "hook_flow", lambda group: _emit_hook_flow_metrics(group, configured_root), path=db)
    if resolution_reason is not None:
        states.append(({"group": "archive_index", "reason": resolution_reason}, 0))
    elif db.exists():
        _collect_group(
            lines,
            states,
            "archive_index",
            lambda group: _format_archive_metrics(group, db, configured_root),
            path=db,
        )
    else:
        ops_attempts_available = False

        def collect_ops_or_discovery(group: list[str]) -> None:
            nonlocal ops_attempts_available
            ops_attempts_available = _format_archive_metrics(group, db, configured_root) is True

        _collect_group(
            lines,
            states,
            "ops_or_discovery",
            collect_ops_or_discovery,
            path=configured_root / "ops.db",
        )
        states.append(
            (
                {"group": "ops_attempts", "reason": "none" if ops_attempts_available else "schema_unavailable"},
                1 if ops_attempts_available else 0,
            )
        )
        states.append(({"group": "archive_index", "reason": "schema_unavailable"}, 0))
    _emit_metric(
        lines,
        name="polylogue_daemon_metrics_collection_available",
        help_text="1 when a metrics collection group completed for this scrape.",
        metric_type="gauge",
        samples=states,
    )
    return "\n".join(lines) + "\n"


def _format_archive_metrics(lines: list[str], db: Path, configured_root: Path) -> bool | None:
    if not db.exists():
        ops_attempts_available = _format_ops_only_metrics(lines, configured_root / "ops.db")
        if ops_attempts_available is not None:
            return ops_attempts_available
        # Fresh install: expose family metadata without pretending an absent
        # index was measured as zero.
        for name, help_text in (
            ("polylogue_live_ingest_attempts_total", "Total live ingest attempts by status."),
            ("polylogue_live_ingest_attempts_in_flight", "Live ingest attempts currently running."),
            ("polylogue_live_ingest_storage_route_total", "Live ingest attempts grouped by storage route."),
            (
                "polylogue_live_ingest_attempt_duration_seconds",
                "Convergence time (seconds) of recent completed ingest attempts.",
            ),
            ("polylogue_convergence_debt_count", "Unresolved convergence-debt rows by stage."),
            (
                "polylogue_fts_trigger_present",
                "1 when the named FTS sync trigger is installed in index.db.",
            ),
            ("polylogue_fts_triggers_all_present", "All expected FTS sync triggers are installed."),
            ("polylogue_fts_freshness_ready", "1 when the daemon freshness ledger marks an FTS surface ready."),
            (
                "polylogue_fts_drift_rows",
                "FTS drift magnitude by surface and kind, read O(1) from fts_freshness_state.",
            ),
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
            _emit_metric(
                lines, name=name, help_text=help_text, metric_type=metric_type, samples=[], omit_when_empty=True
            )
        return False

    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    # Metrics report the archive's readiness, including a tier whose schema
    # does not match the runtime -- that is a blocker series, not a reason to
    # fail the exposition.
    conn = open_readonly_connection(db, validate_schema=False)
    try:
        ops_db = configured_root / "ops.db"
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
                help_text="Unresolved convergence-debt rows by stage and status.",
                metric_type="gauge",
                samples=[({"stage": stage, "status": status}, count) for stage, status, count in debt],
            )
        elif _convergence_debt_measurable(conn, ops_db=ops_db):
            # A measured zero: the ledger is readable and holds no debt rows.
            _emit_metric(
                lines,
                name="polylogue_convergence_debt_count",
                help_text="Unresolved convergence-debt rows by stage and status.",
                metric_type="gauge",
                samples=[(None, 0)],
            )
        else:
            _emit_unmeasured_probe(lines, "convergence_debt_count")

        triggers = _fts_trigger_presence(conn)
        _emit_metric(
            lines,
            name="polylogue_fts_trigger_present",
            help_text="1 when the named FTS sync trigger is installed in index.db.",
            metric_type="gauge",
            samples=[({"trigger": name}, 1 if present else 0) for name, present in triggers.items()],
        )
        if triggers:
            _emit_metric(
                lines,
                name="polylogue_fts_triggers_all_present",
                help_text="1 when every expected FTS sync trigger is installed.",
                metric_type="gauge",
                samples=[(None, 1 if all(triggers.values()) else 0)],
            )
        else:
            # ``all({})`` is True: an empty presence map published "every
            # expected trigger is installed" for a database whose triggers
            # were never inspected (polylogue-xvwpi).
            _emit_unmeasured_probe(lines, "fts_triggers_all_present")

        freshness, drift = _fts_surface_metrics(conn)
        _emit_metric(
            lines,
            name="polylogue_fts_freshness_ready",
            help_text="1 when authoritative FTS membership inspection marks a surface ready.",
            metric_type="gauge",
            samples=[({"surface": surface}, ready) for surface, ready in freshness],
        )

        _emit_metric(
            lines,
            name="polylogue_fts_drift_rows",
            help_text=(
                "FTS drift magnitude by surface and kind (missing/excess/duplicate/"
                "identity_mismatch), inspected from canonical output membership."
            ),
            metric_type="gauge",
            samples=[({"surface": surface, "kind": kind}, value) for surface, kind, value in drift],
        )

        memory = _latest_ingest_memory(conn, ops_db=ops_db)
        _emit_metric(
            lines,
            name="polylogue_live_ingest_memory_mebibytes",
            help_text="Latest live ingest memory sample in MiB by kind.",
            metric_type="gauge",
            samples=[({"kind": kind}, value) for kind, value in memory],
        )

        _emit_embedding_metrics(lines, _archive_embedding_state(conn, ops_db=ops_db))

        # ── Rich instrumentation (#1321 ambitious scope) ──────────
        _emit_archive_metrics(lines, conn)
        _emit_throughput_metrics(lines, ops_db=ops_db)
        _emit_db_space_metrics(lines, db)
        _emit_raw_record_metrics(lines, conn, db_path=configured_root / "index.db")
        _emit_archive_source_index_link_metrics(lines, conn, db_path=configured_root / "index.db")

    finally:
        conn.close()

    return None


def _format_ops_only_metrics(lines: list[str], ops_db: Path) -> bool | None:
    attempts = _ops_attempt_counts(ops_db)
    durations = _ops_recent_attempt_durations(ops_db)
    debt = _ops_convergence_debt_by_stage(ops_db)
    memory = _ops_latest_ingest_memory(ops_db)
    if attempts is None and not durations and not debt and not memory:
        return None

    _emit_metric(
        lines,
        name="polylogue_live_ingest_attempts_total",
        help_text="Total live ingest attempts by status.",
        metric_type="counter",
        samples=(
            [
                ({"status": "completed"}, attempts["completed"]),
                ({"status": "failed"}, attempts["failed"]),
                ({"status": "running"}, attempts["running"]),
            ]
            if attempts is not None
            else []
        ),
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_live_ingest_attempts_in_flight",
        help_text="Live ingest attempts currently running.",
        metric_type="gauge",
        samples=[(None, attempts["running"])] if attempts is not None else [],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_stale_cursor_writes_total",
        help_text="Total stale-cursor writes observed across ingest attempts.",
        metric_type="counter",
        samples=[(None, attempts["stale_cursor_writes"])] if attempts is not None else [],
        omit_when_empty=True,
    )
    route_counts = _ops_storage_route_counts(ops_db)
    if route_counts is not None:
        _emit_storage_route_metrics(lines, route_counts)
    else:
        _emit_metric(
            lines,
            name="polylogue_live_ingest_storage_route_total",
            help_text="Live ingest attempts grouped by storage route.",
            metric_type="counter",
            samples=[],
            omit_when_empty=True,
        )
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
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_convergence_debt_count",
        help_text="Unresolved convergence-debt rows by stage and status.",
        metric_type="gauge",
        samples=[({"stage": stage, "status": status}, count) for stage, status, count in debt],
        omit_when_empty=True,
    )
    _emit_metric(
        lines,
        name="polylogue_live_ingest_memory_mebibytes",
        help_text="Latest live ingest memory sample in MiB by kind.",
        metric_type="gauge",
        samples=[({"kind": kind}, value) for kind, value in memory],
        omit_when_empty=True,
    )
    _emit_ops_throughput_metrics(lines, ops_db)
    for name, help_text in (
        ("polylogue_fts_trigger_present", "1 when the named FTS sync trigger is installed in index.db."),
        ("polylogue_fts_triggers_all_present", "All expected FTS sync triggers are installed."),
        ("polylogue_fts_freshness_ready", "1 when the daemon freshness ledger marks an FTS surface ready."),
        (
            "polylogue_fts_drift_rows",
            "FTS drift magnitude by surface and kind, read O(1) from fts_freshness_state.",
        ),
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
        _emit_metric(lines, name=name, help_text=help_text, metric_type="gauge", samples=[], omit_when_empty=True)
    return attempts is not None


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


def _emit_hook_flow_metrics(lines: list[str], configured_root: Path) -> None:
    """Emit bounded hook wiring/liveness gauges from the shared projection."""

    from polylogue.hooks import flow_states, hook_statuses

    try:
        statuses = hook_statuses(coverage=True, archive_root_path=configured_root)
    except Exception as exc:
        # statuses=() reads identically to "no hooks configured" on the
        # emitted gauges. Log so a hook_statuses() bug doesn't masquerade as
        # a clean hook-flow dashboard.
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="hook_flow_statuses_unreadable",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        statuses = ()
        # polylogue-xvwpi: an empty status tuple reads identically to "no
        # hooks are configured" on the emitted gauges, and _emit_metric would
        # then publish a 0 for each. Name the gap instead.
        _emit_unmeasured_probe(lines, "hook_flow_healthy")
    healthy_samples: list[tuple[dict[str, str] | None, float | int]] = []
    state_samples: list[tuple[dict[str, str] | None, float | int]] = []
    session_samples: list[tuple[dict[str, str] | None, float | int]] = []
    coverage_samples: list[tuple[dict[str, str] | None, float | int]] = []
    for status in statuses:
        if status.flow_healthy is not None:
            healthy_samples.append(({"harness": status.harness}, 1 if status.flow_healthy else 0))
        state_samples.extend(
            (
                {"harness": status.harness, "state": state},
                1 if state == status.flow_state else 0,
            )
            for state in flow_states()
        )
        session_samples.extend(
            (
                {"harness": status.harness, "state": state},
                value,
            )
            for state, value in (
                ("eligible", status.eligible_session_count),
                ("with_events", status.sessions_with_hook_events),
                ("without_events", status.sessions_without_hook_events),
            )
        )
        coverage_samples.extend(
            ({"harness": status.harness, "event": row.event}, row.observed_rate)
            for row in status.coverage
            if row.observed_rate is not None
        )
    _emit_metric(
        lines,
        name="polylogue_hook_flow_healthy",
        help_text="1 when a configured harness has no hook-flow liveness gap.",
        metric_type="gauge",
        samples=healthy_samples,
    )
    _emit_metric(
        lines,
        name="polylogue_hook_flow_state",
        help_text="One-hot hook-flow state per harness.",
        metric_type="gauge",
        samples=state_samples,
    )
    _emit_metric(
        lines,
        name="polylogue_hook_sessions",
        help_text="Trailing-seven-day harness sessions by hook-evidence state.",
        metric_type="gauge",
        samples=session_samples,
    )
    _emit_metric(
        lines,
        name="polylogue_hook_event_observed_ratio",
        help_text="Share of eligible trailing-seven-day sessions observing each hook event.",
        metric_type="gauge",
        samples=coverage_samples,
    )


def _emit_archive_index_metrics(lines: list[str], conn: sqlite3.Connection) -> None:
    session_cols = _columns(conn, "sessions") if _table_exists(conn, "sessions") else set()
    if "origin" in session_cols:
        session_rows = conn.execute("SELECT origin, COUNT(*) FROM sessions GROUP BY origin ORDER BY origin").fetchall()
        session_samples: list[tuple[dict[str, str] | None, int | float]] = [
            ({"source": str(row[0])}, int(row[1])) for row in session_rows
        ]
    else:
        session_samples = [
            (None, _scalar_int(conn, "SELECT COUNT(*) FROM sessions") if _table_exists(conn, "sessions") else 0)
        ]
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


# ``ingest_attempts`` records raw rows and materialized sessions, never a
# message count, so there is no honest message-throughput series to publish.
# The name is not emitted at all: ``_emit_metric`` renders a sample-less
# metric as a literal ``0``, which would state a measured zero for something
# nothing measured -- the same class of error as publishing the session
# numerator under a messages name (polylogue-7z8do).
_THROUGHPUT_METRIC_NAMES = (
    "polylogue_ingest_throughput_raw_rows_per_second",
    "polylogue_ingest_throughput_sessions_per_second",
)


def _emit_throughput_metrics(lines: list[str], *, ops_db: Path | None = None) -> None:
    """Recent ingest throughput derived from the ops-tier ``ingest_attempts`` ledger.

    ``ops.db`` is the sole producer. The former ``index.db``
    ``live_ingest_attempt`` producer emitted the same two metric names from a
    different numerator (``message_count``) over a different denominator (the
    convergence phase alone rather than attempt wall time), so the reported
    value jumped by orders of magnitude the moment ``ops.db`` first appeared --
    a switch by file existence, not by configuration (polylogue-7z8do).
    """
    if ops_db is not None and _emit_ops_throughput_metrics(lines, ops_db):
        return
    for name in _THROUGHPUT_METRIC_NAMES:
        _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])


def _emit_ops_throughput_metrics(lines: list[str], ops_db: Path) -> bool:
    if not ops_db.exists():
        return False
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(ops_db, validate_schema=False)
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
    except sqlite3.Error as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="throughput_unreadable",
            path=ops_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return False

    if row is None:
        for name in _THROUGHPUT_METRIC_NAMES:
            _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])
        return True

    parsed_raw_count = int(row[0] or 0)
    materialized_count = int(row[1] or 0)
    duration = max((int(row[3]) - int(row[2])) / 1000.0, 0.001)
    _emit_metric(
        lines,
        name="polylogue_ingest_throughput_raw_rows_per_second",
        help_text="Source raw-row throughput rate from the most recent completed archive ingest attempt.",
        metric_type="gauge",
        samples=[(None, parsed_raw_count / duration)],
    )
    _emit_metric(
        lines,
        name="polylogue_ingest_throughput_sessions_per_second",
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
            "polylogue_db_wal_file_size_bytes",
            "polylogue_db_allocated_bytes",
            "polylogue_db_freelist_bytes",
            "polylogue_db_page_size",
            "polylogue_db_page_count",
            "polylogue_db_sqlite_stat1_rows",
        ):
            _emit_metric(lines, name=name, help_text=name, metric_type="gauge", samples=[])
        return

    try:
        file_size = db.stat().st_size
        wal_file = db.with_suffix(".db-wal")
        wal_size = wal_file.stat().st_size if wal_file.exists() else 0
        _emit_metric(
            lines,
            name="polylogue_db_file_size_bytes",
            help_text="On-disk size of the archive SQLite database.",
            metric_type="gauge",
            samples=[(None, file_size)],
        )
        _emit_metric(
            lines,
            name="polylogue_db_wal_file_size_bytes",
            help_text="On-disk size of the archive SQLite WAL sidecar.",
            metric_type="gauge",
            samples=[(None, wal_size)],
        )

        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        space_conn = open_readonly_connection(db)
        try:
            page_size = int(space_conn.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(space_conn.execute("PRAGMA page_count").fetchone()[0])
            freelist = int(space_conn.execute("PRAGMA freelist_count").fetchone()[0])
            allocated = page_size * page_count
            freelist_bytes = page_size * freelist
            stat1_rows = (
                int(space_conn.execute("SELECT COUNT(*) FROM sqlite_stat1").fetchone()[0])
                if _table_exists(space_conn, "sqlite_stat1")
                else 0
            )

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
            _emit_metric(
                lines,
                name="polylogue_db_sqlite_stat1_rows",
                help_text="Number of planner-stat rows present in sqlite_stat1.",
                metric_type="gauge",
                samples=[(None, stat1_rows)],
            )
        finally:
            space_conn.close()
    except Exception as exc:
        emit(
            "daemon.metrics.query_failed",
            level=WARNING,
            outcome="degraded",
            reason="db_space_unreadable",
            path=db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


def _emit_archive_storage_metrics(lines: list[str], db: Path, *, configured_root: Path) -> None:
    """Emit tier-presence/schema metrics.

    ``db`` is the currently active db connection path (used only to classify
    which tier role, if any, it plays via ``active_tier_role``); the durable
    tier siblings (ops/source/user/embeddings) always live under
    ``configured_root``, which an index-only external generation is
    explicitly allowed to diverge from.
    """
    tier_paths = [
        (tier, configured_root / filename, expected_version, backup_required)
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
    final_shape_ready = all(present.values())
    storage_layout = archive_layout.classify_storage_layout(
        present_count=present_count, final_shape_ready=final_shape_ready
    )
    active_tier_role = archive_layout.active_tier_role(db, [(tier, path) for tier, path, _, _ in tier_paths])
    physical_archive_store = present["source"] and present["index"]
    active_store = "archive_file_set" if physical_archive_store else "empty"
    blockers = archive_layout.archive_layout_blockers(
        present_count=present_count,
        final_shape_ready=final_shape_ready,
        schema_mismatches=schema_mismatches,
        missing_backup_required=missing_backup_required,
    )
    archive_ready = physical_archive_store and not blockers
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
            ({"state": "materialized"}, 1),
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
        help_text="1 when archive storage has no layout or materialization blockers.",
        metric_type="gauge",
        samples=[(None, 1 if archive_ready else 0)],
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
        raw_links = (
            _scalar_int(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
            if _table_exists(conn, "sessions")
            else 0
        )
        _emit_metric(
            lines,
            name="polylogue_archive_source_index_links_total",
            help_text="Raw source rows by index materialization state.",
            metric_type="gauge",
            samples=[({"source": "unknown", "state": "source_db_missing"}, raw_links)],
        )
        return

    from polylogue.storage.sqlite.connection_profile import attach_readonly_database

    attach_readonly_database(conn, source_db, alias="source_metrics")
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
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    # This probe reports the on-disk version, including a skewed one. A read
    # failure is not version zero; the containing storage group is unavailable.
    conn = open_readonly_connection(path, validate_schema=False)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    finally:
        conn.close()


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
            except sqlite3.Error as exc:
                emit(
                    "daemon.metrics.probe_failed",
                    level=WARNING,
                    outcome="degraded",
                    reason="raw_record_probe_unreadable",
                    path=source_db,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
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

    Unexpected failures retain the HTTP 200 scrape contract with a bounded
    error reason. Normal collector failures are isolated by ``format_metrics``.
    """
    try:
        body = format_metrics(db)
    except Exception as exc:
        emit(
            "daemon.metrics.collection_failed",
            level=WARNING,
            outcome="degraded",
            reason="metrics_collection_failed",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        lines, states = _process_metric_lines()
        states.append(({"group": "archive_index", "reason": "collector_failed"}, 0))
        _emit_metric(
            lines,
            name="polylogue_daemon_metrics_collection_available",
            help_text="1 when a metrics collection group completed for this scrape.",
            metric_type="gauge",
            samples=states,
        )
        _emit_metric(
            lines,
            name="polylogue_daemon_metrics_collection_error",
            help_text="Metrics collection failed.",
            metric_type="gauge",
            samples=[({"reason": "collector_failed"}, 1)],
        )
        body = "\n".join(lines) + "\n"
    responder._send_text(HTTPStatus.OK, body, content_type=PROMETHEUS_CONTENT_TYPE)


__all__ = [
    "PROMETHEUS_CONTENT_TYPE",
    "MetricsResponder",
    "format_metrics",
    "handle_metrics",
]
