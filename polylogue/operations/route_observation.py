"""Bounded route-latency observation (polylogue-jtwu / polylogue-20d.17 AC #4).

Covers the routes ``query_runs`` (query-DSL executions) and ``mcp_call_log``
(whole MCP tool calls, durably delivered via an outbox) do not: CLI command
invocations and MCP sub-route detail a caller wants to time without routing
through either of those. Best-effort telemetry, not audit evidence -- a
caller that cannot reach ``ops.db`` (no archive configured, disposable tier
missing, locked) drops the observation rather than blocking or retrying the
operation being observed. :func:`compute_latency_percentiles` federates
``route_observations`` with ``mcp_call_log`` into one grouped p50/p95 view;
``query_runs`` is intentionally not included here (see the module's own
exactness/degraded-membership semantics, a different concern from raw
latency) -- a documented, not silent, scope decision.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)

_CONNECT_TIMEOUT_S = 2.0
_GIT_HEAD_TIMEOUT_S = 1.0
LOW_CONFIDENCE_SAMPLE_FLOOR = 5


@dataclass
class RouteObservationContext:
    """Mutable handle yielded by :func:`observe_route`.

    ``status`` defaults to ``"ok"`` and is set to ``"error"`` automatically
    if the observed block raises; callers may set it explicitly (e.g.
    ``"degraded"``) before the block exits. ``attributes`` is freeform
    caller-supplied detail (e.g. per-component states) recorded alongside
    the timing.
    """

    attributes: dict[str, object] = field(default_factory=dict)
    status: str = "ok"
    daemon_path: str | None = None
    """Set explicitly by the caller once known ('daemon' or 'direct'); the
    ``observe_route`` argument of the same name only seeds the initial
    value for callers that already know it when the block starts."""


def _current_git_head(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_HEAD_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


@contextmanager
def observe_route(
    *,
    archive_root: Path | None,
    surface: str,
    route: str,
    verb: str | None = None,
    daemon_path: str | None = None,
    trace_id: str | None = None,
    git_head_cwd: Path | None = None,
) -> Iterator[RouteObservationContext]:
    """Time one route invocation and record a best-effort receipt.

    Telemetry failures (no archive configured, locked ops.db, disposable
    tier missing) are logged at debug level and swallowed -- they must
    never surface as an error in, or block, the operation being observed.
    Re-raises whatever the observed block raises, unchanged.
    """
    ctx = RouteObservationContext(daemon_path=daemon_path)
    started_at_ms = int(time.time() * 1000)
    started_monotonic = time.monotonic()
    try:
        yield ctx
    except Exception:
        ctx.status = "error"
        raise
    finally:
        duration_ms = max(0, round((time.monotonic() - started_monotonic) * 1000))
        _emit_best_effort(
            archive_root=archive_root,
            trace_id=trace_id or str(uuid.uuid4()),
            surface=surface,
            route=route,
            verb=verb,
            daemon_path=ctx.daemon_path,
            started_at_ms=started_at_ms,
            duration_ms=duration_ms,
            status=ctx.status,
            git_head=_current_git_head(git_head_cwd) if git_head_cwd is not None else None,
            attributes=ctx.attributes,
        )


def _emit_best_effort(
    *,
    archive_root: Path | None,
    trace_id: str,
    surface: str,
    route: str,
    verb: str | None,
    daemon_path: str | None,
    started_at_ms: int,
    duration_ms: int,
    status: str,
    git_head: str | None,
    attributes: dict[str, object],
) -> None:
    if archive_root is None:
        return
    ops_db = Path(archive_root) / "ops.db"
    if not ops_db.exists():
        return
    try:
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_route_observation

        conn = sqlite3.connect(ops_db, timeout=_CONNECT_TIMEOUT_S)
        try:
            record_route_observation(
                conn,
                trace_id=trace_id,
                surface=surface,
                route=route,
                verb=verb,
                daemon_path=daemon_path,
                started_at_ms=started_at_ms,
                duration_ms=duration_ms,
                status=status,
                git_head=git_head,
                attributes=attributes,
            )
        finally:
            conn.close()
    except Exception:
        logger.debug(
            "route observation emit failed (best-effort, dropped): surface=%s route=%s",
            surface,
            route,
            exc_info=True,
        )


@dataclass(frozen=True, slots=True)
class RouteLatencyBucket:
    """One (surface, route) group's latency distribution."""

    surface: str
    route: str
    sample_count: int
    p50_ms: float | None
    p95_ms: float | None
    error_count: int
    error_rate: float
    oldest_at_ms: int | None
    newest_at_ms: int | None
    low_confidence: bool
    """True when ``sample_count`` is below :data:`LOW_CONFIDENCE_SAMPLE_FLOOR`
    -- a p95 over a handful of samples is not a reliable percentile; render
    it visibly caveated rather than as a confident budget number."""


def _percentile(sorted_values: Sequence[int], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def compute_latency_percentiles(
    route_observations: Sequence[object],
    mcp_calls: Sequence[object] = (),
) -> tuple[RouteLatencyBucket, ...]:
    """Group ``route_observations`` and ``mcp_calls`` by (surface, route) and compute p50/p95.

    Accepts ``ArchiveRouteObservation``/``ArchiveMcpCallLogEntry`` rows
    (typed as ``object`` here to avoid importing the ops-tier module at
    call sites that only need the pure aggregation); duck-types on the
    attributes each row type actually has.
    """
    grouped: dict[tuple[str, str], list[tuple[int, bool, int]]] = defaultdict(list)
    for obs in route_observations:
        grouped[(obs.surface, obs.route)].append(  # type: ignore[attr-defined]
            (obs.duration_ms, obs.status in ("error", "timed_out", "unavailable"), obs.started_at_ms)  # type: ignore[attr-defined]
        )
    for call in mcp_calls:
        grouped[("mcp", f"mcp.{call.tool_name}")].append(  # type: ignore[attr-defined]
            (call.duration_ms, not call.success, call.started_at_ms)  # type: ignore[attr-defined]
        )

    buckets: list[RouteLatencyBucket] = []
    for (surface, route), samples in sorted(grouped.items()):
        durations = sorted(duration for duration, _is_error, _started in samples)
        error_count = sum(1 for _duration, is_error, _started in samples if is_error)
        started_values = [started for _duration, _is_error, started in samples]
        buckets.append(
            RouteLatencyBucket(
                surface=surface,
                route=route,
                sample_count=len(samples),
                p50_ms=_percentile(durations, 0.50),
                p95_ms=_percentile(durations, 0.95),
                error_count=error_count,
                error_rate=error_count / len(samples) if samples else 0.0,
                oldest_at_ms=min(started_values) if started_values else None,
                newest_at_ms=max(started_values) if started_values else None,
                low_confidence=len(samples) < LOW_CONFIDENCE_SAMPLE_FLOOR,
            )
        )
    return tuple(buckets)


__all__ = [
    "LOW_CONFIDENCE_SAMPLE_FLOOR",
    "RouteLatencyBucket",
    "RouteObservationContext",
    "compute_latency_percentiles",
    "observe_route",
]
