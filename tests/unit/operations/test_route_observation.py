"""Tests for the bounded route-latency observation module (polylogue-jtwu)."""

from __future__ import annotations

import sqlite3
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from polylogue.operations.route_observation import (
    DEFAULT_ROUTE_PHASE,
    LOW_CONFIDENCE_SAMPLE_FLOOR,
    RouteLatencyBucket,
    RouteObservationDrops,
    RouteObservationSpec,
    compute_latency_percentiles,
    observe_route,
    reset_route_observation_drops,
    route_observation_drops,
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


def _buckets(
    observations: Sequence[object],
    mcp_calls: Sequence[object] = (),
) -> tuple[RouteLatencyBucket, ...]:
    """Percentiles over a sample this test built itself, so drops are known-zero."""
    return compute_latency_percentiles(observations, mcp_calls, drops=RouteObservationDrops.none_observed()).buckets


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
    assert row.attributes["marker"] == "seen"
    # The receipt projection rides in the same freeform document.
    assert isinstance(row.attributes["route_receipt"], dict)


def test_observe_route_records_git_head_when_requested(tmp_path: Path) -> None:
    _init_ops(tmp_path)
    expected = subprocess.check_output(
        ["git", "-C", str(Path.cwd()), "rev-parse", "--short=12", "HEAD"],
        text=True,
    ).strip()

    with observe_route(
        archive_root=tmp_path,
        surface="mcp",
        route="mcp.status.coordination",
        git_head_cwd=Path.cwd(),
    ):
        pass

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="mcp", route="mcp.status.coordination")
    assert len(rows) == 1
    assert rows[0].git_head == expected


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
    assert rows[0].attributes["archive_evidence_degraded"] is True


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

    with observe_route(
        archive_root=tmp_path,
        surface="cli",
        route="cli.status",
        spec=spec,
        trace_id="req-1",
        run_id="run-1",
        parent_run_id="run-0",
        archive_id="archive-3",
        archive_epoch="epoch-7",
        git_head_cwd=Path.cwd(),
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
    assert receipt.build_id is not None  # git head, the build scope
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
    with observe_route(archive_root=tmp_path, surface="cli", route="cli.status") as obs:
        pass
    receipt = obs.receipt
    assert receipt is not None
    assert receipt.response_bytes is None
    assert "response_bytes" in receipt.unavailable_measures


def test_receipt_adapts_onto_a_workload_receipt(tmp_path: Path) -> None:
    """The latency adapter polylogue-jtwu's DESIGN names, exercised end to end."""
    from polylogue.scenarios.workload import WorkloadReceipt, WorkloadRunStatus

    _init_ops(tmp_path)
    with observe_route(
        archive_root=tmp_path, surface="mcp", route="mcp.query", trace_id="req-9", run_id="run-9"
    ) as obs:
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
    with observe_route(archive_root=tmp_path, surface="cli", route="cli.partial", spec=spec) as obs:
        pass  # "render" never entered

    receipt = obs.receipt
    assert receipt is not None
    assert receipt.to_workload_receipt().status is WorkloadRunStatus.INTERRUPTED


def test_persisted_row_and_workload_receipt_join_on_the_correlation_id(tmp_path: Path) -> None:
    """Anti-vacuity: strip the correlation id from the emitted receipt and the
    two projections of one invocation can no longer be joined."""
    _init_ops(tmp_path)
    with observe_route(
        archive_root=tmp_path,
        surface="cli",
        route="cli.correlated",
        trace_id="req-corr",
        run_id="run-corr",
    ) as obs:
        pass

    receipt = obs.receipt
    assert receipt is not None
    workload = receipt.to_workload_receipt()

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="cli", route="cli.correlated")
    assert len(rows) == 1
    persisted = rows[0]

    # Projection 1: the ops-tier row. Projection 2: the workload receipt.
    persisted_receipt = persisted.attributes["route_receipt"]
    assert isinstance(persisted_receipt, dict)
    persisted_refs = set(persisted_receipt["correlation_refs"])
    assert persisted_refs, "the persisted row carries no correlation refs to join on"
    assert persisted_refs <= set(workload.evidence_refs)
    assert f"route-run:{persisted_receipt['run_id']}" in set(workload.evidence_refs)
    assert persisted.trace_id == receipt.trace_id


def test_observe_route_keeps_the_existing_percentile_reader_working(tmp_path: Path) -> None:
    """J1-2: the contract is additive; ``polylogue analyze latency`` still reads these rows."""
    _init_ops(tmp_path)
    for _ in range(3):
        with observe_route(archive_root=tmp_path, surface="cli", route="cli.compat"):
            pass

    conn = sqlite3.connect(tmp_path / "ops.db")
    rows = list_route_observations(conn, surface="cli", route="cli.compat")
    buckets = _buckets(rows)
    assert len(buckets) == 1
    assert buckets[0].sample_count == 3
    assert buckets[0].p50_ms is not None


def test_spec_requires_the_total_phase_first() -> None:
    with pytest.raises(ValueError, match="first declared route phase"):
        RouteObservationSpec(surface="cli", route="cli.x", phases=("render",))


def test_context_refuses_an_undeclared_phase(tmp_path: Path) -> None:
    _init_ops(tmp_path)
    with observe_route(archive_root=tmp_path, surface="cli", route="cli.x") as obs:
        with pytest.raises(ValueError, match="not declared"):
            with obs.phase("render"):
                pass


# ---------------------------------------------------------------------------
# Drop accounting (polylogue-jtwu.2)
# ---------------------------------------------------------------------------


def test_every_observation_dropped_by_an_unavailable_ops_tier_is_counted(tmp_path: Path) -> None:
    """Anti-vacuity: a drop path that returns without incrementing leaves this at 0."""
    missing_root = tmp_path / "no-archive"
    for _ in range(4):
        with observe_route(archive_root=missing_root, surface="cli", route="cli.dropped"):
            pass

    drops = route_observation_drops()
    assert drops.total == 4
    assert drops.by_reason == {"ops_db_missing": 4}
    assert drops.attributed_to("cli", "cli.dropped") == 4


def test_a_missing_archive_root_is_counted_under_its_own_reason(tmp_path: Path) -> None:
    for _ in range(2):
        with observe_route(archive_root=None, surface="cli", route="cli.rootless"):
            pass
    assert route_observation_drops().by_reason == {"no_archive_root": 2}


def test_an_emit_failure_is_counted_not_just_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The condition the module docstring names -- a locked ops.db -- is countable."""
    _init_ops(tmp_path)

    def _locked(_ops_db: Path) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("polylogue.operations.route_observation.open_observation_connection", _locked)
    for _ in range(3):
        with observe_route(archive_root=tmp_path, surface="mcp", route="mcp.locked"):
            pass

    drops = route_observation_drops()
    assert drops.by_reason == {"emit_failed": 3}
    assert drops.attributed_to("mcp", "mcp.locked") == 3


def test_rows_removed_by_the_writers_row_cap_are_counted_as_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent bead's own bullet: retention/row-cap pruning was never counted."""
    _init_ops(tmp_path)
    monkeypatch.setattr("polylogue.storage.sqlite.archive_tiers.ops_write.ROUTE_OBSERVATION_ROW_CAP", 2)

    for _ in range(5):
        with observe_route(archive_root=tmp_path, surface="cli", route="cli.capped"):
            pass

    drops = route_observation_drops()
    assert drops.by_reason.get("pruned", 0) == 3  # rows 1..3 evicted to hold the cap at 2
    conn = sqlite3.connect(tmp_path / "ops.db")
    assert conn.execute("SELECT COUNT(*) FROM route_observations").fetchone()[0] == 2


def test_an_unsampled_spec_counts_its_invocations_rather_than_losing_them(tmp_path: Path) -> None:
    _init_ops(tmp_path)
    spec = RouteObservationSpec(surface="cli", route="cli.unsampled", sampled=False)
    with observe_route(archive_root=tmp_path, surface="cli", route="cli.unsampled", spec=spec):
        pass

    assert route_observation_drops().by_reason == {"not_sampled": 1}
    conn = sqlite3.connect(tmp_path / "ops.db")
    assert conn.execute("SELECT COUNT(*) FROM route_observations").fetchone()[0] == 0


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
        "unattributed": 0,
    }


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
    assert report.unattributed_drops == 0
    assert report.drops.total == 4
    assert report.is_complete is False


def test_a_reader_that_hit_its_row_limit_declares_the_truncation() -> None:
    from polylogue.operations.route_observation import read_side_drops

    drops = read_side_drops(observation_count=1000, mcp_call_count=3, row_limit=1000)
    assert drops.accounting_complete is False
    assert drops.by_reason == {"read_limit_truncated": 1}

    untruncated = read_side_drops(observation_count=12, mcp_call_count=3, row_limit=1000)
    assert untruncated.by_reason == {}
    assert untruncated.accounting_complete is False


def test_latency_report_is_frozen() -> None:
    report = compute_latency_percentiles([], drops=RouteObservationDrops.none_observed())
    with pytest.raises(AttributeError):
        report.buckets = ()  # type: ignore[misc]
