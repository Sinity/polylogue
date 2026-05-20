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
  derived from existing daemon-state tables — no conversation
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
- ``polylogue_live_ingest_attempts_total`` (counter) — labels: status
- ``polylogue_live_ingest_attempts_in_flight`` (gauge)
- ``polylogue_live_ingest_attempt_duration_seconds`` (gauge buckets:
  min, mean, max derived from recent completed attempts)
- ``polylogue_convergence_debt_count`` (gauge) — labels: stage
- ``polylogue_blob_lease_pending_count`` (gauge)
- ``polylogue_blob_lease_distinct_operations`` (gauge)
- ``polylogue_fts_trigger_present`` (gauge) — labels: trigger
- ``polylogue_fts_triggers_all_present`` (gauge, 0/1)
- ``polylogue_stale_cursor_writes_total`` (counter)

The handler signature mirrors ``healthz.py``'s ``ProbeResponder``
protocol so it is testable without the full ``BaseHTTPRequestHandler``
stack.
"""

from __future__ import annotations

import sqlite3
import time
from http import HTTPStatus
from pathlib import Path
from typing import Protocol

from polylogue.logging import get_logger

logger = get_logger(__name__)


PROMETHEUS_CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"

# Process start captured at import — same posture as healthz.py.
_PROCESS_STARTED_AT_MONOTONIC: float = time.monotonic()


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


# Same expected FTS triggers as devtools/daemon_workload_probe.py — kept
# inline so the metrics module does not depend on the probe.
_EXPECTED_FTS_TRIGGERS: tuple[str, ...] = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
)


def _attempt_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return totals of ``live_ingest_attempt`` rows by status plus stale writes."""
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


def _recent_attempt_durations(conn: sqlite3.Connection, *, limit: int = 50) -> list[float]:
    """Return durations of recent completed attempts (seconds)."""
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


def _convergence_debt_by_stage(conn: sqlite3.Connection) -> list[tuple[str, int]]:
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


def _blob_lease_state(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (pending_count, distinct_operations)."""
    if not _table_exists(conn, "pending_blob_refs"):
        return (0, 0)
    pending = _scalar_int(conn, "SELECT COUNT(*) FROM pending_blob_refs")
    operations = _scalar_int(conn, "SELECT COUNT(DISTINCT operation_id) FROM pending_blob_refs")
    return (pending, operations)


def _fts_trigger_presence(conn: sqlite3.Connection) -> dict[str, bool]:
    placeholders = ",".join("?" for _ in _EXPECTED_FTS_TRIGGERS)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        _EXPECTED_FTS_TRIGGERS,
    ).fetchall()
    present = {str(row[0]) for row in rows}
    return {name: (name in present) for name in _EXPECTED_FTS_TRIGGERS}


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
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    uptime_s = max(0.0, now_monotonic - _PROCESS_STARTED_AT_MONOTONIC)
    lines: list[str] = []

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
    except Exception:  # pragma: no cover - import surface should always succeed
        polylogue_version = "unknown"
    _emit_metric(
        lines,
        name="polylogue_daemon_build_info",
        help_text="Constant 1 gauge labelled with daemon build identity.",
        metric_type="gauge",
        samples=[({"version": str(polylogue_version)}, 1)],
    )

    if not db.exists():
        # Fresh install — emit the discovery skeleton with zeros so the
        # scraper sees a stable series set on day zero.
        for name, help_text in (
            ("polylogue_live_ingest_attempts_total", "Total live ingest attempts by status."),
            ("polylogue_live_ingest_attempts_in_flight", "Live ingest attempts currently running."),
            (
                "polylogue_live_ingest_attempt_duration_seconds",
                "Convergence time (seconds) of recent completed ingest attempts.",
            ),
            ("polylogue_convergence_debt_count", "Unresolved convergence-debt rows by stage."),
            ("polylogue_blob_lease_pending_count", "Pending blob leases in flight."),
            ("polylogue_blob_lease_distinct_operations", "Distinct lease-holding operations."),
            (
                "polylogue_fts_trigger_present",
                "1 when the named FTS sync trigger is installed in the archive DB.",
            ),
            ("polylogue_fts_triggers_all_present", "All expected FTS sync triggers are installed."),
            ("polylogue_stale_cursor_writes_total", "Total stale-cursor writes observed across ingest attempts."),
        ):
            metric_type = "counter" if name.endswith("_total") else "gauge"
            _emit_metric(lines, name=name, help_text=help_text, metric_type=metric_type, samples=[])
        return "\n".join(lines) + "\n"

    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(db)
    try:
        attempts = _attempt_counts(conn)
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

        durations = _recent_attempt_durations(conn)
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

        debt = _convergence_debt_by_stage(conn)
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
            help_text="1 when the named FTS sync trigger is installed in the archive DB.",
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
    finally:
        conn.close()

    return "\n".join(lines) + "\n"


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
    except Exception as exc:  # pragma: no cover - defence in depth
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
