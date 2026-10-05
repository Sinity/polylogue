"""Tests for the bounded route-latency observation module (polylogue-jtwu)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.operations.route_observation import (
    DEFAULT_ROUTE_PHASE,
    LOW_CONFIDENCE_SAMPLE_FLOOR,
    RouteLatencyBucket,
    RouteObservationDrops,
    RouteObservationSpec,
    compute_latency_percentiles,
    measure_route,
    reset_route_observation_drops,
    route_observation_drops,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    ArchiveMcpCallLogEntry,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _init_ops(tmp_path: Path) -> Path:
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    return ops_db


def _observation(*, surface: str, route: str, duration_ms: int, status: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
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


def _buckets(
    observations: Sequence[object],
    mcp_calls: Sequence[object] = (),
) -> tuple[RouteLatencyBucket, ...]:
    """Percentiles over a sample this test built itself, so drops are known-zero."""
    return compute_latency_percentiles(observations, mcp_calls, drops=RouteObservationDrops.none_observed()).buckets


def test_compute_latency_percentiles_matches_known_distribution() -> None:
    """A known, hand-computed distribution proves the percentile math, not just plumbing."""
    durations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    observations = [_observation(surface="cli", route="cli.status", duration_ms=d) for d in durations]

    buckets = _buckets(observations)

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
    buckets = _buckets(observations)
    assert buckets[0].low_confidence is True

    plenty = [
        _observation(surface="cli", route="cli.common", duration_ms=d) for d in range(LOW_CONFIDENCE_SAMPLE_FLOOR)
    ]
    buckets = _buckets(plenty)
    assert buckets[0].low_confidence is False


def test_compute_latency_percentiles_counts_error_and_timed_out_as_errors() -> None:
    observations = [
        _observation(surface="cli", route="cli.status", duration_ms=100, status="ok"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="error"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="timed_out"),
        _observation(surface="cli", route="cli.status", duration_ms=100, status="degraded"),
    ]
    bucket = _buckets(observations)[0]
    assert bucket.error_count == 2  # error + timed_out; degraded is not counted as an error
    assert bucket.error_rate == pytest.approx(0.5)


def test_compute_latency_percentiles_federates_mcp_call_log() -> None:
    observations = [_observation(surface="cli", route="cli.status", duration_ms=100)]
    calls = [
        _mcp_call(tool_name="status", duration_ms=200, success=True),
        _mcp_call(tool_name="status", duration_ms=400, success=False),
    ]

    buckets = _buckets(observations, calls)

    by_route = {(b.surface, b.route): b for b in buckets}
    assert ("cli", "cli.status") in by_route
    assert ("mcp", "mcp.status") in by_route
    mcp_bucket = by_route[("mcp", "mcp.status")]
    assert mcp_bucket.sample_count == 2
    assert mcp_bucket.error_count == 1


def test_compute_latency_percentiles_empty_input_returns_no_buckets() -> None:
    assert _buckets([]) == ()


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
        dropped_count=0,
        drop_accounting_complete=True,
    )
    with pytest.raises(AttributeError):
        bucket.sample_count = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The route-observation contract (polylogue-jtwu.1)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_drop_ledger() -> Iterator[None]:
    """The drop ledger is process-local; never let one test's drops leak."""
    reset_route_observation_drops()
    yield
    reset_route_observation_drops()


def test_receipt_carries_the_declared_scope_and_identity(tmp_path: Path) -> None:
    """One receipt carries build/archive/workload scope, request and run id, and phases."""
    _init_ops(tmp_path)
    spec = RouteObservationSpec(surface="cli", route="cli.status", verb="read", phases=(DEFAULT_ROUTE_PHASE, "render"))

    with measure_route(
        surface="cli",
        route="cli.status",
        spec=spec,
        trace_id="req-1",
        run_id="run-1",
        parent_run_id="run-0",
        archive_id="archive-3",
        archive_epoch="epoch-7",
        build_id="artifact-build-1",
    ) as obs:
        obs.daemon_path = "daemon"
        obs.response_bytes = 512
        with obs.phase("render"):
            time.sleep(0.01)

    receipt = obs.receipt
    assert receipt is not None
    assert receipt.spec is spec
    assert receipt.trace_id == "req-1"
    assert receipt.run_id == "run-1"
    assert receipt.parent_run_id == "run-0"
    assert receipt.build_id == "artifact-build-1"
    assert receipt.archive_id == "archive-3"
    assert receipt.archive_epoch == "epoch-7"
    assert receipt.daemon_path == "daemon"
    assert receipt.response_bytes == 512
    assert receipt.spec.workload_id == "route:cli:cli.status:read"
    assert {phase.name for phase in receipt.phases} == {DEFAULT_ROUTE_PHASE, "render"}
    assert all(phase.cpu_ms is not None for phase in receipt.phases)
    assert receipt.unavailable_measures == ()


def test_receipt_reports_an_unobtained_measure_as_unavailable(tmp_path: Path) -> None:
    """Unmeasured is not zero: a response size nobody supplied is declared missing."""
    _init_ops(tmp_path)
    with measure_route(surface="cli", route="cli.status") as obs:
        pass
    receipt = obs.receipt
    assert receipt is not None
    assert receipt.response_bytes is None
    assert "response_bytes" in receipt.unavailable_measures


def test_receipt_adapts_onto_a_workload_receipt(tmp_path: Path) -> None:
    """The latency adapter polylogue-jtwu's DESIGN names, exercised end to end."""
    from polylogue.scenarios.workload import WorkloadReceipt, WorkloadRunStatus

    _init_ops(tmp_path)
    with measure_route(surface="mcp", route="mcp.query", trace_id="req-9", run_id="run-9") as obs:
        pass

    receipt = obs.receipt
    assert receipt is not None
    workload = receipt.to_workload_receipt()

    assert isinstance(workload, WorkloadReceipt)
    assert workload.status is WorkloadRunStatus.SUCCEEDED
    assert workload.spec.family_id == "route-observation"
    assert workload.spec.workload_id == "route:mcp:mcp.query"
    assert tuple(phase.name for phase in workload.phases) == (DEFAULT_ROUTE_PHASE,)
    assert workload.receipt_id  # content-addressed, derived not asserted


def test_route_observation_is_a_declared_workload_adapter() -> None:
    """The measurement-path inventory names this adapter, so it is not a private path."""
    from polylogue.scenarios.workload import workload_adapter_declarations

    by_name = {declaration.name: declaration for declaration in workload_adapter_declarations()}
    assert "route-observation" in by_name
    assert by_name["route-observation"].owner == "polylogue.operations.route_observation"


def test_workload_receipt_reports_an_unentered_declared_phase_as_interrupted(tmp_path: Path) -> None:
    from polylogue.scenarios.workload import WorkloadRunStatus

    _init_ops(tmp_path)
    spec = RouteObservationSpec(surface="cli", route="cli.partial", phases=(DEFAULT_ROUTE_PHASE, "render"))
    with measure_route(surface="cli", route="cli.partial", spec=spec) as obs:
        pass  # "render" never entered

    receipt = obs.receipt
    assert receipt is not None
    assert receipt.to_workload_receipt().status is WorkloadRunStatus.INTERRUPTED


def test_degraded_route_is_not_projected_as_successful_work(tmp_path: Path) -> None:
    from polylogue.scenarios.workload import WorkloadRunStatus

    _init_ops(tmp_path)
    with measure_route(surface="mcp", route="mcp.gap") as obs:
        obs.status = "degraded"
    assert obs.receipt is not None
    assert obs.receipt.to_workload_receipt().status is WorkloadRunStatus.INTERRUPTED
    assert "response_bytes" in obs.receipt.to_workload_receipt().phases[0].unavailable


def test_spec_requires_the_total_phase_first() -> None:
    with pytest.raises(ValueError, match="first declared route phase"):
        RouteObservationSpec(surface="cli", route="cli.x", phases=("render",))


def test_context_refuses_an_undeclared_phase(tmp_path: Path) -> None:
    _init_ops(tmp_path)
    with measure_route(surface="cli", route="cli.x") as obs:
        with pytest.raises(ValueError, match="not declared"):
            with obs.phase("render"):
                pass


def test_repeated_declared_phases_do_not_fail_the_observed_route(tmp_path: Path) -> None:
    """Anti-vacuity: a duplicate phase cannot replace a successful route result."""
    _init_ops(tmp_path)
    spec = RouteObservationSpec(surface="cli", route="cli.repeat", phases=("total", "parse"))
    with measure_route(surface="cli", route="cli.repeat", spec=spec) as obs:
        for _ in range(2):
            with obs.phase("parse"):
                pass
    assert obs.receipt is not None
    assert [phase.name for phase in obs.receipt.phases] == ["total", "parse"]


def test_route_spec_cannot_relabel_a_different_call_site() -> None:
    spec = RouteObservationSpec(surface="mcp", route="mcp.query")
    with pytest.raises(ValueError, match="identity"):
        with measure_route(surface="cli", route="cli.status", spec=spec):
            pass


def test_invalid_daemon_path_drops_telemetry_without_replacing_route_error(tmp_path: Path) -> None:
    """Anti-vacuity: malformed optional telemetry cannot escape the finally block."""
    _init_ops(tmp_path)
    with pytest.raises(RuntimeError, match="operation failed"):
        with measure_route(surface="cli", route="cli.invalid-path") as obs:
            obs.daemon_path = "socket"
            raise RuntimeError("operation failed")
    assert route_observation_drops().by_reason.get("emit_failed") == 1


# ---------------------------------------------------------------------------
# Drop accounting (polylogue-jtwu.2)
# ---------------------------------------------------------------------------


def test_percentiles_cannot_be_computed_without_a_drop_disposition() -> None:
    """``drops`` is keyword-only and has no default: an unqualified percentile
    is not expressible through this API."""
    with pytest.raises(TypeError):
        compute_latency_percentiles([])  # type: ignore[call-arg]


def test_drops_are_reported_beside_the_percentiles_they_qualify() -> None:
    observations = [_observation(surface="cli", route="cli.status", duration_ms=d) for d in (10, 20, 30)]
    drops = RouteObservationDrops(
        accounting_complete=True,
        by_reason={"emit_failed": 7},
        by_route={"cli\tcli.status": 7},
    )

    report = compute_latency_percentiles(observations, drops=drops)

    bucket = report.buckets[0]
    assert bucket.sample_count == 3
    assert bucket.dropped_count == 7
    assert bucket.sample_completeness == pytest.approx(0.3)
    assert bucket.is_qualified is True
    assert report.is_complete is False
    assert report.to_payload()["drops"] == {
        "total": 7,
        "accounting_complete": True,
        "by_reason": {"emit_failed": 7},
        "by_route": {"cli\tcli.status": 7},
        "unattributed": 0,
    }


def test_drops_for_a_route_without_a_surviving_bucket_remain_attributed() -> None:
    drops = RouteObservationDrops(
        accounting_complete=True,
        by_reason={"pruned": 4},
        by_route={"cli\tvanished": 4},
    )
    report = compute_latency_percentiles([_observation(surface="cli", route="status", duration_ms=12)], drops=drops)
    assert report.unattributed_drops == 4
    assert cast(dict[str, object], report.to_payload()["drops"])["by_route"] == {"cli\tvanished": 4}


def test_unknown_drop_accounting_is_not_reported_as_zero_drops() -> None:
    """Zero drops and unknown drops are different answers."""
    observations = [_observation(surface="cli", route="cli.status", duration_ms=d) for d in (10, 20, 30)]

    report = compute_latency_percentiles(observations, drops=RouteObservationDrops.unaccounted())

    bucket = report.buckets[0]
    assert bucket.drop_accounting_complete is False
    assert bucket.sample_completeness is None
    assert bucket.is_qualified is True
    assert report.is_complete is False


def test_drops_charged_to_no_rendered_bucket_survive_as_unattributed() -> None:
    """A drop for a route with zero surviving samples must not vanish."""
    observations = [_observation(surface="cli", route="cli.status", duration_ms=10)]
    drops = RouteObservationDrops(
        accounting_complete=True,
        by_reason={"emit_failed": 4},
        by_route={"cli\tcli.vanished": 4},
    )

    report = compute_latency_percentiles(observations, drops=drops)

    assert [b.route for b in report.buckets] == ["cli.status"]
    assert report.buckets[0].dropped_count == 0
    assert report.unattributed_drops == 4
    assert report.drops.total == 4
    assert report.is_complete is False


def test_a_fully_accounted_report_is_ok_and_an_empty_one_is_empty() -> None:
    """Anti-vacuity: an outcome that ignored the drop disposition would call
    the degraded reader answer above ``ok``; one that ignored rows would call
    this ``ok`` report ``empty``."""
    full = compute_latency_percentiles(
        [_observation(surface="cli", route="cli.status", duration_ms=5)], drops=RouteObservationDrops.none_observed()
    )
    empty = compute_latency_percentiles([], drops=RouteObservationDrops.none_observed())

    assert full.outcome.state == "ok"
    assert full.to_payload()["outcome"] == full.outcome.to_dict()
    assert empty.outcome.state == "empty"


def test_latency_report_is_frozen() -> None:
    report = compute_latency_percentiles([], drops=RouteObservationDrops.none_observed())
    with pytest.raises(AttributeError):
        report.buckets = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Durable drop accounting across processes (polylogue-jtwu.2 J2-1)
# ---------------------------------------------------------------------------


def _read_report(ops_db: Path, *, since_ms: int, now_ms: int | None = None) -> object:
    from polylogue.operations.route_observation import read_latency_report

    conn = sqlite3.connect(ops_db)
    try:
        return read_latency_report(conn, since_ms=since_ms, now_ms=now_ms)
    finally:
        conn.close()


def test_unobserved_client_route_has_a_typed_reason_and_never_opens_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring client observation persistence reaches the refused opener."""
    from polylogue.operations.route_observation import (
        record_unobserved_client_route,
        reset_route_observation_drops,
        route_observation_drops,
    )

    def refuse_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a client does not own ops.db")

    monkeypatch.setattr("sqlite3.connect", refuse_open)
    emitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr("polylogue.logging._threshold", 20)
    monkeypatch.setattr("polylogue.logging._emit_raw", lambda _level, event, fields: emitted.append((event, fields)))
    reset_route_observation_drops()
    try:
        reason = record_unobserved_client_route(surface="cli", route="cli.status")
        assert reason.value == "client_not_owner"
        assert emitted == [
            (
                "route_observation.unobserved",
                {"outcome": "skipped", "reason": "client_not_owner", "route": "cli.status"},
            )
        ]
        assert route_observation_drops().by_reason == {"client_not_owner": 1}
        assert route_observation_drops().accounting_complete is False
    finally:
        reset_route_observation_drops()


def test_measured_cancellation_never_claims_success() -> None:
    import asyncio

    from polylogue.scenarios.workload import WorkloadRunStatus

    with pytest.raises(asyncio.CancelledError):
        with measure_route(surface="cli", route="cli.cancelled") as observation:
            raise asyncio.CancelledError
    assert observation.receipt is not None
    assert observation.receipt.status == "error"
    assert observation.receipt.to_workload_receipt().status is WorkloadRunStatus.FAILED


def test_mcp_reader_has_no_retired_table_dependency(tmp_path: Path) -> None:
    from polylogue.operations.route_observation import read_latency_report
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_mcp_call

    ops_db = _init_ops(tmp_path)
    with sqlite3.connect(ops_db) as conn:
        assert not conn.execute(
            "SELECT name FROM sqlite_schema WHERE name IN ('route_observations','route_observation_drops')"
        ).fetchall()
        for i in range(1200):
            start = 1700000000000 + i
            record_mcp_call(
                conn,
                call_id=f"call-{i}",
                tool_name="search",
                started_at_ms=start,
                finished_at_ms=start + (5000 if i < 200 else 10),
                success=True,
            )
        report = read_latency_report(conn, since_ms=1700000000000, now_ms=1700000002000)
        absent = read_latency_report(conn, since_ms=1700000000000, surface="cli", now_ms=1700000002000)
    assert len(report.buckets) == 1
    assert report.buckets[0].sample_count == 1200
    assert report.buckets[0].p95_ms == 5000
    assert report.drops.accounting_complete is False
    assert report.outcome.state == "degraded"
    assert not absent.buckets
    assert absent.outcome.state == "degraded"
