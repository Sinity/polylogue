"""Idempotency tests for :mod:`polylogue.maintenance.replay` (#1147).

These pin the convergence contract from the issue acceptance criteria:

* running an operation twice on the same archive produces no further
  changes after the first pass converges (the underlying repair
  functions are convergent by construction; the replay loop adds the
  multi-target guarantee that *no target is rebuilt twice and none is
  skipped*);
* a per-target failure does not abort the rest of the operation and
  surfaces as a bounded :class:`FailureSample`;
* an unsupported target is reported as a typed failure instead of
  silently succeeding.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.config import Config
from polylogue.maintenance.failure_routing import (
    count_maintenance_failures,
    read_maintenance_failures,
    route_failure_sample,
)
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.planner import (
    MAX_FAILURE_SAMPLES,
    BackfillStatus,
    FailureSample,
)
from polylogue.maintenance.replay import (
    UnsupportedReplayTargetError,
    execute_replay,
    supported_replay_targets,
)
from polylogue.storage.repair import RepairResult


def _make_config(tmp_path: Path) -> Config:
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    return Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
        db_path=tmp_path / "archive.db",
    )


def _result(name: str, repaired: int) -> RepairResult:
    return RepairResult(
        name=name,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=repaired,
        success=True,
        detail="",
    )


@pytest.fixture
def converging_dispatch() -> Iterator[dict[str, int]]:
    """Stub dispatch that simulates convergence.

    First call to each target reports ``repaired_count=N`` (initial
    rebuild). Subsequent calls return ``repaired_count=0`` because the
    target is already converged. This matches the real repair-function
    contract: re-running ``repair_dangling_fts`` on an already-consistent
    archive returns 0.
    """

    converged: dict[str, int] = {}

    def stub(name: str, initial: int):  # type: ignore[no-untyped-def]
        def _run(_config: Config, _dry_run: bool) -> RepairResult:
            attempt = converged.get(name, 0)
            converged[name] = attempt + 1
            repaired = initial if attempt == 0 else 0
            return _result(name, repaired)

        return _run

    fake = {
        "session_insights": stub("session_insights", 7),
        "dangling_fts": stub("dangling_fts", 5),
    }
    with patch("polylogue.maintenance.replay._REPLAY_DISPATCH", fake):
        yield converged


def test_second_run_makes_no_further_changes(tmp_path: Path, converging_dispatch: dict[str, int]) -> None:
    config = _make_config(tmp_path)

    first = execute_replay(
        config,
        targets=("session_insights", "dangling_fts"),
        operation_id="op-first",
    )
    assert first.status is BackfillStatus.COMPLETED
    assert first.affected_rows == 7 + 5

    # A fresh operation id on the same converged archive must report
    # zero further row changes — convergence holds across operations.
    second = execute_replay(
        config,
        targets=("session_insights", "dangling_fts"),
        operation_id="op-second",
    )
    assert second.status is BackfillStatus.COMPLETED
    assert second.affected_rows == 0
    # Each target was called exactly twice (once per op), not more.
    assert converging_dispatch["session_insights"] == 2
    assert converging_dispatch["dangling_fts"] == 2


def test_single_target_failure_does_not_abort_others(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    calls: list[str] = []

    def good(name: str):  # type: ignore[no-untyped-def]
        def _run(_config: Config, _dry_run: bool) -> RepairResult:
            calls.append(name)
            return _result(name, repaired=3)

        return _run

    def bad(_config: Config, _dry_run: bool) -> RepairResult:
        calls.append("dangling_fts")
        raise RuntimeError("simulated row-level failure")

    fake = {
        "session_insights": good("session_insights"),
        "dangling_fts": bad,
        "message_type_backfill": good("message_type_backfill"),
    }
    with patch("polylogue.maintenance.replay._REPLAY_DISPATCH", fake):
        op = execute_replay(
            config,
            targets=("session_insights", "dangling_fts", "message_type_backfill"),
            operation_id="op-mixed",
        )

    # The bad target failed but the surrounding targets still ran.
    assert calls == ["session_insights", "dangling_fts", "message_type_backfill"]
    assert op.status is BackfillStatus.FAILED
    assert op.affected_rows == 6  # only the two good targets contributed
    assert len(op.failure_samples.samples) == 1
    failure = op.failure_samples.samples[0]
    assert failure.kind == "RuntimeError"
    assert failure.locator == "target:dangling_fts"


def test_repair_reported_failure_surfaces_as_failure_sample(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    def reports_failure(_config: Config, _dry_run: bool) -> RepairResult:
        return RepairResult(
            name="dangling_fts",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail="schema mismatch",
        )

    with patch(
        "polylogue.maintenance.replay._REPLAY_DISPATCH",
        {"dangling_fts": reports_failure},
    ):
        op = execute_replay(
            config,
            targets=("dangling_fts",),
            operation_id="op-soft-fail",
        )

    assert op.status is BackfillStatus.FAILED
    assert op.failure_samples.samples[0].kind == "RepairReportedFailure"
    assert "schema mismatch" in op.failure_samples.samples[0].message


def test_replay_refuses_offline_repair_while_daemon_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr("polylogue.maintenance.offline_guard.running_daemon_pid", lambda _config: 1234)

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-live-daemon",
    )

    assert op.status is BackfillStatus.FAILED
    assert op.affected_rows == 0
    assert op.results[0]["name"] == "session_insights"
    assert op.failure_samples.samples[0].kind == "OfflineMaintenanceBlocked"
    assert "polylogued PID 1234 is running" in op.failure_samples.samples[0].message


def test_unwired_target_is_typed_failure_not_silent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with patch("polylogue.maintenance.replay._REPLAY_DISPATCH", {}):
        op = execute_replay(
            config,
            targets=("session_insights",),
            operation_id="op-unsupported",
        )

    assert op.status is BackfillStatus.FAILED
    sample = op.failure_samples.samples[0]
    assert sample.kind == UnsupportedReplayTargetError.__name__
    assert sample.locator == "target:session_insights"


def test_message_embeddings_replay_reports_daemon_owned_dormancy(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    route_failure_sample(
        FailureSample(
            kind=UnsupportedReplayTargetError.__name__,
            locator="target:message_embeddings",
            message="Target 'message_embeddings' is not yet wired",
        ),
        operation_id="op-old-unsupported",
        archive_root=config.archive_root,
        target="message_embeddings",
    )

    op = execute_replay(
        config,
        targets=("message_embeddings",),
        operation_id="op-embeddings",
        persist_state=False,
    )

    assert op.status is BackfillStatus.COMPLETED
    assert op.affected_rows == 0
    assert op.failure_samples.samples == ()
    assert op.results[0]["name"] == "message_embeddings"
    assert op.results[0]["success"] is True
    assert "daemon-owned and dormant" in str(op.results[0]["detail"])
    assert count_maintenance_failures(config.archive_root) == 0
    assert read_maintenance_failures(config.archive_root) == []


def test_supported_targets_cover_ac_required_set() -> None:
    """Replay supports the durable maintenance targets advertised in the catalog."""
    supported = set(supported_replay_targets())
    required = {
        "session_insights",
        "raw_materialization",
        "message_embeddings",
        "message_type_backfill",
        "dangling_fts",
        "orphaned_blobs",
    }
    assert required.issubset(supported)


def test_failure_samples_remain_bounded(tmp_path: Path) -> None:
    """Even a pathologically failing run cannot exceed the planner's
    bounded sample envelope."""

    config = _make_config(tmp_path)

    def always_raises(_config: Config, _dry_run: bool) -> RepairResult:
        raise RuntimeError("boom")

    # Build a synthetic dispatch with N > MAX_FAILURE_SAMPLES targets,
    # all failing. Reuse the existing target names because the catalog
    # is the source of resolution; targets repeated in the input are
    # deduplicated upstream, so we instead drive the loop directly
    # through the public dispatch by faking the catalog resolution.
    target_names = tuple(f"t{i}" for i in range(MAX_FAILURE_SAMPLES + 5))

    class _FakeSpec:
        def __init__(self, name: str) -> None:
            self.name = name

    fake_specs = tuple(_FakeSpec(name) for name in target_names)

    class _FakeCatalog:
        def resolve(self, _names: tuple[str, ...]) -> tuple[_FakeSpec, ...]:
            return fake_specs

    fake_dispatch = dict.fromkeys(target_names, always_raises)
    with (
        patch(
            "polylogue.maintenance.replay.build_maintenance_target_catalog",
            return_value=_FakeCatalog(),
        ),
        patch(
            "polylogue.maintenance.replay._REPLAY_DISPATCH",
            fake_dispatch,
        ),
    ):
        op = execute_replay(
            config,
            targets=target_names,
            operation_id="op-flood",
            persist_state=False,
        )

    assert op.status is BackfillStatus.FAILED
    assert len(op.failure_samples.samples) == MAX_FAILURE_SAMPLES
    assert op.failure_samples.truncated is True
