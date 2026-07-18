"""Tests for the bounded route-latency observation module (polylogue-jtwu)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.operations.route_observation import (
    LOW_CONFIDENCE_SAMPLE_FLOOR,
    RouteLatencyBucket,
    compute_latency_percentiles,
    observe_route,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ArchiveMcpCallLogEntry,
    ArchiveRouteObservation,
    list_route_observations,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _init_ops(tmp_path: Path) -> Path:
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    return ops_db


def _observation(*, surface: str, route: str, duration_ms: int, status: str = "ok") -> ArchiveRouteObservation:
    return ArchiveRouteObservation(
        observation_id="o",
        trace_id="t",
        surface=surface,
        route=route,
        verb=None,
        daemon_path=None,
        phase="total",
        started_at_ms=1_700_000_000_000,
        duration_ms=duration_ms,
        status=status,
        git_head=None,
        archive_epoch=None,
        attributes={},
        sampled=True,
    )


def _mcp_call(*, tool_name: str, duration_ms: int, success: bool = True) -> ArchiveMcpCallLogEntry:
    return ArchiveMcpCallLogEntry(
        call_id="c",
        tool_name=tool_name,
        session_id=None,
        started_at_ms=1_700_000_000_000,
        finished_at_ms=1_700_000_000_000 + duration_ms,
        duration_ms=duration_ms,
        success=success,
        error_detail=None,
    )


def test_observe_route_records_a_receipt_with_measured_duration(tmp_path: Path) -> None:
    """Anti-vacuity: the receipt's recorded duration reflects real elapsed time.

    A route body that sleeps a controlled amount must produce a receipt
    whose duration_ms is at least that long -- this fails if the timing
    wrapper is a no-op or measures the wrong span.
    """
    _init_ops(tmp_path)

    with observe_route(archive_root=tmp_path, surface="cli", route="cli.test-route", verb="v1") as obs:
        time.sleep(0.05)
        obs.attributes["marker"] = "seen"

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="cli", route="cli.test-route")
    assert len(rows) == 1
    row = rows[0]
    assert row.duration_ms >= 45  # measured, not a stub -- real sleep was ~50ms
    assert row.status == "ok"
    assert row.verb == "v1"
    assert row.attributes == {"marker": "seen"}


def test_observe_route_records_error_status_on_exception_and_reraises(tmp_path: Path) -> None:
    _init_ops(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        with observe_route(archive_root=tmp_path, surface="cli", route="cli.failing-route"):
            raise RuntimeError("boom")

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="cli", route="cli.failing-route")
    assert len(rows) == 1
    assert rows[0].status == "error"


def test_observe_route_caller_can_override_status_to_degraded(tmp_path: Path) -> None:
    _init_ops(tmp_path)

    with observe_route(archive_root=tmp_path, surface="mcp", route="mcp.status.coordination") as obs:
        obs.status = "degraded"
        obs.daemon_path = "direct"
        obs.attributes["archive_evidence_degraded"] = True

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="mcp")
    assert rows[0].status == "degraded"
    assert rows[0].daemon_path == "direct"
    assert rows[0].attributes == {"archive_evidence_degraded": True}


def test_observe_route_is_a_silent_no_op_without_an_archive(tmp_path: Path) -> None:
    """Telemetry emission must never depend on, or fail loudly over, an absent archive."""
    missing_root = tmp_path / "does-not-exist"

    with observe_route(archive_root=missing_root, surface="cli", route="cli.test-route"):
        pass  # no exception -- best-effort drop, not a hard requirement


def test_observe_route_is_a_silent_no_op_with_no_archive_root(tmp_path: Path) -> None:
    with observe_route(archive_root=None, surface="cli", route="cli.test-route"):
        pass


def test_compute_latency_percentiles_matches_known_distribution() -> None:
    """A known, hand-computed distribution proves the percentile math, not just plumbing."""
    durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    observations = [_observation(surface="cli", route="cli.status", duration_ms=d) for d in durations]

    buckets = compute_latency_percentiles(observations)

    assert len(buckets) == 1
    bucket = buckets[0]
    assert bucket.surface == "cli"
    assert bucket.route == "cli.status"
    assert bucket.sample_count == 10
    assert bucket.p50_ms == pytest.approx(55.0)
    assert bucket.p95_ms == pytest.approx(95.5)
    assert bucket.error_count == 0
    assert bucket.low_confidence is False


def test_compute_latency_percentiles_flags_low_confidence_below_floor() -> None:
    observations = [
        _observation(surface="cli", route="cli.rare", duration_ms=d) for d in range(LOW_CONFIDENCE_SAMPLE_FLOOR - 1)
    ]
    buckets = compute_latency_percentiles(observations)
    assert buckets[0].low_confidence is True

    plenty = [
        _observation(surface="cli", route="cli.common", duration_ms=d) for d in range(LOW_CONFIDENCE_SAMPLE_FLOOR)
    ]
    buckets = compute_latency_percentiles(plenty)
    assert buckets[0].low_confidence is False


def test_compute_latency_percentiles_counts_error_and_timed_out_as_errors() -> None:
    observations = [
        _observation(surface="cli", route="cli.status", duration_ms=100, status="ok"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="error"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="timed_out"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="degraded"),
    ]
    bucket = compute_latency_percentiles(observations)[0]
    assert bucket.error_count == 2  # error + timed_out; degraded is not counted as an error
    assert bucket.error_rate == pytest.approx(0.5)


def test_compute_latency_percentiles_federates_mcp_call_log() -> None:
    observations = [_observation(surface="cli", route="cli.status", duration_ms=100)]
    calls = [
        _mcp_call(tool_name="status", duration_ms=200, success=True),
        _mcp_call(tool_name="status", duration_ms=400, success=False),
    ]

    buckets = compute_latency_percentiles(observations, calls)

    by_route = {(b.surface, b.route): b for b in buckets}
    assert ("cli", "cli.status") in by_route
    assert ("mcp", "mcp.status") in by_route
    mcp_bucket = by_route[("mcp", "mcp.status")]
    assert mcp_bucket.sample_count == 2
    assert mcp_bucket.error_count == 1


def test_compute_latency_percentiles_empty_input_returns_no_buckets() -> None:
    assert compute_latency_percentiles([]) == ()


def test_route_latency_bucket_is_frozen() -> None:
    bucket = RouteLatencyBucket(
        surface="cli",
        route="cli.status",
        sample_count=1,
        p50_ms=1.0,
        p95_ms=1.0,
        error_count=0,
        error_rate=0.0,
        oldest_at_ms=1,
        newest_at_ms=1,
        low_confidence=True,
    )
    with pytest.raises(AttributeError):
        bucket.sample_count = 2  # type: ignore[misc]
