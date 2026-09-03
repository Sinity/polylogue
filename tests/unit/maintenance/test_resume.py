"""Resume tests for :mod:`polylogue.maintenance.replay` (#1147).

These pin the resume contract from the issue acceptance criteria:

* an interrupted replay leaves an on-disk state file that records the
  last completed target via a typed cursor;
* re-invoking the same ``operation_id`` resumes from that cursor and
  does not re-run already-completed targets;
* a successful replay clears the state file so a later run with the
  same id starts fresh.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.config import Config
from polylogue.core.enums import OperationStatus
from polylogue.maintenance import replay as replay_module
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.replay import (
    CURSOR_DONE,
    ReplayProgress,
    clear_state,
    execute_replay,
    load_state,
    state_path_for,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.storage import repair as repair_module
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


def _ok_result(name: str, repaired: int = 1, metrics: dict[str, float] | None = None) -> RepairResult:
    return RepairResult(
        name=name,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=repaired,
        success=True,
        detail=f"{name} ok",
        metrics=dict(metrics or {}),
    )


@pytest.fixture
def patched_dispatch() -> Iterator[dict[str, list[str]]]:
    """Replace the replay dispatch table with stub repair functions.

    Yields a call log keyed by target name so tests can assert which
    targets executed under each invocation.
    """

    calls: dict[str, list[str]] = {
        "session_insights": [],
        "empty_sessions": [],
        "superseded_raw_snapshots": [],
    }

    def stub(name: str):  # type: ignore[no-untyped-def]
        def _run(config: Config, dry_run: bool) -> RepairResult:
            calls[name].append("dry" if dry_run else "live")
            return _ok_result(name)

        return _run

    fake_dispatch = {name: stub(name) for name in calls}
    with patch.object(repair_module, "REPAIR_HANDLERS", fake_dispatch):
        yield calls


def test_clean_run_persists_done_and_clears_state(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)
    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-clean",
    )

    assert op.status is OperationStatus.COMPLETED
    assert op.resume_cursor == CURSOR_DONE
    # State file is removed after successful completion.
    assert not state_path_for(config, "op-clean").exists()
    assert patched_dispatch["session_insights"] == ["live"]


def test_completed_prefix_cursor_uses_historical_target_identity(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    """A historical cursor remains valid after completed targets are filtered."""
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-historical-prefix")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-historical-prefix",'
        '"targets":["session_insights","message_type_backfill","empty_sessions"],'
        '"completed_targets":["session_insights","message_type_backfill"],"cursor":"target:2"}'
    )

    op = execute_replay(
        config,
        targets=("empty_sessions",),
        operation_id="op-historical-prefix",
    )

    assert op.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["empty_sessions"] == ["live"]


def test_positional_persisted_cursor_without_identity_fails_closed(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-unverifiable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"operation_id":"op-unverifiable","cursor":"target:2"}')

    op = execute_replay(
        config,
        targets=("session_insights", "superseded_raw_snapshots"),
        operation_id="op-unverifiable",
    )

    assert op.status is OperationStatus.FAILED
    assert op.error == "Persisted replay state has no valid target identity list"
    assert op.failure_samples.samples[0].kind == "IncompatibleReplayState"
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["superseded_raw_snapshots"] == []


def test_chained_resume_retains_completed_identity_history(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-chained")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-chained",'
        '"targets":["session_insights","empty_sessions","superseded_raw_snapshots"],'
        '"completed_targets":["session_insights"],'
        '"cursor":"target:0"}'
    )

    def fail_empty(_config: Config, _dry_run: bool) -> RepairResult:
        raise RuntimeError("interrupted")

    with patch.object(
        repair_module,
        "REPAIR_HANDLERS",
        {
            "session_insights": patched_dispatch_callable(patched_dispatch, "session_insights"),
            "empty_sessions": fail_empty,
            "superseded_raw_snapshots": patched_dispatch_callable(patched_dispatch, "superseded_raw_snapshots"),
        },
    ):
        first = execute_replay(
            config,
            targets=("session_insights", "empty_sessions", "superseded_raw_snapshots"),
            operation_id="op-chained",
        )
    assert first.status is OperationStatus.FAILED
    checkpoint = load_state(config, "op-chained")
    assert checkpoint is not None
    assert checkpoint["targets"] == ["session_insights", "empty_sessions", "superseded_raw_snapshots"]
    assert checkpoint["completed_targets"] == ["session_insights", "superseded_raw_snapshots"]

    second = execute_replay(
        config,
        targets=("session_insights", "empty_sessions", "superseded_raw_snapshots"),
        operation_id="op-chained",
    )

    assert second.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["empty_sessions"] == ["live"]
    assert patched_dispatch["superseded_raw_snapshots"] == ["live"]


@pytest.mark.parametrize("contents", ["[]", "null", "not-json"])
def test_invalid_persisted_state_fails_closed(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], contents: str
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)

    op = execute_replay(
        config,
        targets=("session_insights", "superseded_raw_snapshots"),
        operation_id="op-invalid",
    )

    assert op.status is OperationStatus.FAILED
    assert op.error == "Persisted replay state is not a JSON object"
    assert op.failure_samples.samples[0].kind == "InvalidReplayState"
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["superseded_raw_snapshots"] == []


def test_generated_checkpoint_cursor_is_legacy_only_and_failed_target_retries(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)

    def reported_failure(_config: Config, _dry_run: bool) -> RepairResult:
        return RepairResult(
            name="superseded_raw_snapshots",
            category=MaintenanceCategory.ARCHIVE_CLEANUP,
            destructive=True,
            repaired_count=3,
            success=False,
            detail="retry me",
            metrics={"attempted": 3.0},
        )

    with patch.object(
        repair_module,
        "REPAIR_HANDLERS",
        {
            "session_insights": patched_dispatch_callable(patched_dispatch, "session_insights"),
            "empty_sessions": patched_dispatch_callable(patched_dispatch, "empty_sessions"),
            "superseded_raw_snapshots": reported_failure,
        },
    ):
        first = execute_replay(
            config,
            targets=("session_insights", "empty_sessions", "superseded_raw_snapshots"),
            operation_id="op-retry",
        )

    assert first.status is OperationStatus.FAILED
    checkpoint = load_state(config, "op-retry")
    assert checkpoint is not None
    assert checkpoint["completed_targets"] == ["session_insights", "empty_sessions"]
    assert checkpoint["cursor"] == "target:0"

    second = execute_replay(
        config,
        targets=("session_insights", "empty_sessions", "superseded_raw_snapshots"),
        operation_id="op-retry",
    )

    assert second.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == ["live"]
    assert patched_dispatch["empty_sessions"] == ["live"]
    assert patched_dispatch["superseded_raw_snapshots"] == ["live"]


def test_resume_aggregates_receipt_data_and_current_progress_after_remap(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-receipt",'
        '"targets":["session_insights","empty_sessions","superseded_raw_snapshots"],'
        '"completed_targets":["session_insights"],"cursor":"target:0",'
        '"started_at":"2026-01-01T00:00:00+00:00",'
        '"results":[{"name":"session_insights","repaired_count":4,"success":true}],'
        '"repaired_count":4,"failure_count":1,'
        '"failure_samples":[{"kind":"old","locator":"target:empty_sessions","message":"old"}],'
        '"metrics":{"prior":2.0}}'
    )
    snapshots: list[ReplayProgress] = []

    op = execute_replay(
        config,
        targets=("superseded_raw_snapshots", "empty_sessions"),
        operation_id="op-receipt",
        progress_callback=snapshots.append,
    )

    assert op.status is OperationStatus.COMPLETED
    assert op.started_at == "2026-01-01T00:00:00+00:00"
    assert len(op.results) == 3
    assert op.affected_rows == 6
    assert op.metrics["prior"] == 2.0
    assert op.failure_samples.samples[0].kind == "old"
    assert snapshots and {snapshot.total for snapshot in snapshots} == {2}
    assert [snapshot.target for snapshot in snapshots] == ["superseded_raw_snapshots", "empty_sessions"]


def test_fresh_explicit_done_and_malformed_cursor_are_typed_noops_or_failures(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)

    done = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-fresh-done",
        resume_cursor=CURSOR_DONE,
    )
    malformed = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-fresh-malformed",
        resume_cursor="not-a-cursor",
    )

    assert done.status is OperationStatus.COMPLETED
    assert done.progress == 1.0
    assert malformed.status is OperationStatus.FAILED
    assert malformed.failure_samples.samples[0].kind == "InvalidReplayCursor"
    assert patched_dispatch["session_insights"] == []


@pytest.mark.parametrize("resume_cursor", ["", "not-a-cursor", "target:not-an-integer", "target:-1"])
def test_explicit_malformed_cursor_precedes_blocker_and_preserves_state(
    tmp_path: Path,
    patched_dispatch: dict[str, list[str]],
    resume_cursor: str,
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-invalid-before-blocker")
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '{"operation_id":"op-invalid-before-blocker","cursor":"target:0"}'
    path.write_text(original)

    blocker = _ok_result("session_insights", repaired=0)
    blocker = RepairResult(
        name=blocker.name,
        category=blocker.category,
        destructive=blocker.destructive,
        repaired_count=0,
        success=False,
        detail="daemon is running",
    )
    with patch(
        "polylogue.maintenance.replay.offline_maintenance_blockers",
        return_value=[blocker],
    ) as blocker_check:
        op = execute_replay(
            config,
            targets=("session_insights",),
            operation_id="op-invalid-before-blocker",
            resume_cursor=resume_cursor,
        )

    assert op.status is OperationStatus.FAILED
    assert op.failure_samples.samples[0].kind == "InvalidReplayCursor"
    blocker_check.assert_not_called()
    assert path.read_text() == original
    assert patched_dispatch["session_insights"] == []


def test_explicit_malformed_cursor_without_state_does_not_create_state(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    with patch("polylogue.maintenance.replay.offline_maintenance_blockers") as blocker_check:
        op = execute_replay(
            config,
            targets=("session_insights",),
            operation_id="op-invalid-no-state",
            resume_cursor="",
        )

    assert op.status is OperationStatus.FAILED
    assert op.failure_samples.samples[0].kind == "InvalidReplayCursor"
    blocker_check.assert_not_called()
    assert not state_path_for(config, "op-invalid-no-state").exists()
    assert patched_dispatch["session_insights"] == []


def test_explicit_resume_cursor_maps_reordered_subset_by_identity(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-reorder")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-reorder",'
        '"targets":["session_insights","empty_sessions","superseded_raw_snapshots"],'
        '"completed_targets":["session_insights"],"cursor":"target:0"}'
    )

    op = execute_replay(
        config,
        targets=("superseded_raw_snapshots", "empty_sessions"),
        operation_id="op-reorder",
        resume_cursor="target:0",
    )

    assert op.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["superseded_raw_snapshots"] == ["live"]
    assert patched_dispatch["empty_sessions"] == ["live"]


def test_legacy_done_cursor_uses_success_records_and_retries_failed_target(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-legacy-retry")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-legacy-retry",'
        '"targets":["session_insights","empty_sessions"],"cursor":"done",'
        '"results":[{"name":"session_insights","success":true,"repaired_count":1}]}'
    )

    op = execute_replay(
        config,
        targets=("session_insights", "empty_sessions"),
        operation_id="op-legacy-retry",
    )

    assert op.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["empty_sessions"] == ["live"]


def test_legacy_done_without_authoritative_success_does_not_clear_state(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-legacy-unknown")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"operation_id":"op-legacy-unknown","targets":["session_insights"],"cursor":"done"}')

    op = execute_replay(config, targets=("session_insights",), operation_id="op-legacy-unknown")

    assert op.status is OperationStatus.FAILED
    assert op.error == "Legacy replay state has no authoritative successful targets"
    assert state_path_for(config, "op-legacy-unknown").exists()
    assert patched_dispatch["session_insights"] == []


def test_resume_rejects_mode_and_scope_context_changes(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-context")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-context","targets":["session_insights"],'
        '"completed_targets":[],"cursor":"target:0","dry_run":true,'
        '"scope_filter":{"session_ids":["s-1"],"origin":null,"source_family":null,'
        '"source_root":null,"time_range":null,"failure_kind":null,"parser_version":null}}'
    )

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-context",
        dry_run=False,
        scope_filter=MaintenanceScopeFilter(session_ids=("s-2",)),
    )

    assert op.status is OperationStatus.FAILED
    assert op.failure_samples.samples[0].kind == "ReplayContextMismatch"
    assert patched_dispatch["session_insights"] == []


def test_nested_receipt_fields_do_not_double_count_metrics(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-nested-receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-nested-receipt","targets":["session_insights"],'
        '"completed_targets":[],"cursor":"target:0","results":[],"metrics":{},'
        '"operation":{"started_at":"2026-02-03T04:05:06+00:00",'
        '"metrics":{"same":7.0},"failure_samples":{"samples":[],"truncated":false}}}'
    )

    op = execute_replay(config, targets=("session_insights",), operation_id="op-nested-receipt")

    assert op.status is OperationStatus.COMPLETED
    assert op.started_at == "2026-02-03T04:05:06+00:00"
    assert op.metrics["same"] == 7.0


def test_blocker_receipt_retains_nested_cumulative_metrics(
    tmp_path: Path,
    patched_dispatch: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-blocked-receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-blocked-receipt","targets":["session_insights"],'
        '"completed_targets":[],"cursor":"target:0","metrics":{},'
        '"operation":{"started_at":"2026-02-03T04:05:06+00:00",'
        '"metrics":{"same":7.0},"failure_samples":{"samples":[],"truncated":false}}}'
    )
    monkeypatch.setattr(
        "polylogue.maintenance.replay.offline_maintenance_blockers",
        lambda *args, **kwargs: [_ok_result("session_insights", repaired=0)],
    )

    op = execute_replay(config, targets=("session_insights",), operation_id="op-blocked-receipt")

    assert op.status is OperationStatus.FAILED
    assert op.metrics["same"] == 7.0
    assert op.metrics["repaired_count"] == 0.0
    persisted = load_state(config, "op-blocked-receipt")
    assert persisted is not None
    operation = persisted.get("operation")
    assert isinstance(operation, dict)
    scope = operation.get("scope")
    assert isinstance(scope, dict)
    assert scope.get("filter") == MaintenanceScopeFilter().to_dict()
    assert patched_dispatch["session_insights"] == []


def test_legacy_cursor_with_failure_samples_fails_closed(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-legacy-failure-prefix")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-legacy-failure-prefix",'
        '"targets":["session_insights","empty_sessions"],"cursor":"target:1",'
        '"failure_samples":[{"kind":"RuntimeError","locator":"target:session_insights",'
        '"message":"failed"}]}'
    )

    op = execute_replay(config, targets=("session_insights", "empty_sessions"), operation_id="op-legacy-failure-prefix")

    assert op.status is OperationStatus.FAILED
    assert "failure samples" in (op.error or "")
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["empty_sessions"] == []


def test_missing_persisted_cursor_is_invalid_not_fresh_execution(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-missing-cursor")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"operation_id":"op-missing-cursor","targets":["session_insights"]}')

    op = execute_replay(config, targets=("session_insights",), operation_id="op-missing-cursor")

    assert op.status is OperationStatus.FAILED
    assert op.failure_samples.samples[0].kind == "InvalidReplayCursor"
    assert patched_dispatch["session_insights"] == []


def test_scoped_resume_requires_persisted_scope_identity(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-missing-scope")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-missing-scope","targets":["session_insights"],"completed_targets":[],"cursor":"target:0"}'
    )

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-missing-scope",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
    )

    assert op.status is OperationStatus.FAILED
    assert op.failure_samples.samples[0].kind == "ReplayContextMismatch"
    assert patched_dispatch["session_insights"] == []


def test_replay_passes_session_scope_to_empty_session_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config(tmp_path)
    seen: list[tuple[str, ...] | None] = []

    def empty_sessions(_config: Config, _dry_run: bool, *, session_ids: tuple[str, ...] | None = None) -> RepairResult:
        seen.append(session_ids)
        return _ok_result("empty_sessions")

    monkeypatch.setattr(replay_module, "repair_empty_sessions", empty_sessions)
    monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "empty_sessions", empty_sessions)

    operation = execute_replay(
        config,
        targets=("empty_sessions",),
        operation_id="op-empty-session-scope",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1", "s-2")),
    )

    assert operation.status is OperationStatus.COMPLETED
    assert seen == [("s-1", "s-2")]


def test_replay_refuses_target_that_cannot_honor_session_scope(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    operation = execute_replay(
        config,
        targets=("superseded_raw_snapshots",),
        operation_id="op-unsupported-session-scope",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
    )

    assert operation.status is OperationStatus.FAILED
    assert operation.resume_cursor == CURSOR_DONE
    assert operation.failure_samples.samples[0].kind == "UnsupportedScopeDimension"
    assert patched_dispatch["superseded_raw_snapshots"] == []
    assert not (Path(config.archive_root) / ".maintenance-state" / "failures.jsonl").exists()

    resumed = execute_replay(
        config,
        targets=("superseded_raw_snapshots",),
        operation_id="op-unsupported-session-scope",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
    )

    assert resumed.status is OperationStatus.FAILED
    assert resumed.resume_cursor == CURSOR_DONE
    assert len(resumed.failure_samples.samples) == 1
    assert patched_dispatch["superseded_raw_snapshots"] == []


def test_legacy_result_metrics_are_reconstructed_when_aggregate_is_absent(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-legacy-metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-legacy-metrics",'
        '"targets":["session_insights","empty_sessions"],"cursor":"done",'
        '"results":[{"name":"session_insights","success":true,"metrics":{"same":7.0}}]}'
    )
    monkeypatch.setitem(
        repair_module.REPAIR_HANDLERS,
        "empty_sessions",
        lambda _config, _dry_run: _ok_result("empty_sessions", metrics={"same": 2.0}),
    )

    op = execute_replay(config, targets=("session_insights", "empty_sessions"), operation_id="op-legacy-metrics")

    assert op.status is OperationStatus.COMPLETED
    assert op.metrics["same"] == 9.0


def test_empty_persisted_cursor_fails_closed_instead_of_replaying(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-empty-cursor")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"operation_id":"op-empty-cursor","targets":["session_insights"],"cursor":""}')

    original = path.read_text()
    op = execute_replay(config, targets=("session_insights",), operation_id="op-empty-cursor")

    assert op.status is OperationStatus.FAILED
    assert op.error == "Persisted replay state has an invalid target cursor"
    assert op.failure_samples.samples[0].kind == "InvalidReplayCursor"
    assert patched_dispatch["session_insights"] == []
    assert path.read_text() == original


def test_progress_processed_is_monotonic_through_failure_and_inner_progress(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    progress: list[ReplayProgress] = []

    def session_insights(
        _config: Config,
        _dry_run: bool,
        *,
        session_ids: tuple[str, ...] | None,
        progress_callback: Any,
    ) -> RepairResult:
        del session_ids
        progress_callback(4, "inner")
        return _ok_result("session_insights")

    def fail(_config: Config, _dry_run: bool) -> RepairResult:
        raise RuntimeError("failed target")

    monkeypatch.setattr(replay_module, "repair_session_insights", session_insights)
    with patch.object(
        repair_module,
        "REPAIR_HANDLERS",
        {
            "session_insights": session_insights,
            "empty_sessions": fail,
            "superseded_raw_snapshots": patched_dispatch_callable(patched_dispatch, "superseded_raw_snapshots"),
        },
    ):
        op = execute_replay(
            config,
            targets=("session_insights", "empty_sessions", "superseded_raw_snapshots"),
            operation_id="op-progress-failure",
            progress_callback=progress.append,
        )

    assert op.status is OperationStatus.FAILED
    processed = [snapshot.processed for snapshot in progress]
    assert processed == sorted(processed)
    assert processed[-3:] == [1, 2, 3]
    assert all(snapshot.total == 3 for snapshot in progress)


def test_nested_metrics_add_new_results_without_double_counting_prior_rows(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-metric-resume")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-metric-resume","targets":["session_insights"],'
        '"completed_targets":[],"cursor":"target:0","metrics":{},'
        '"results":[{"name":"old","success":true,"metrics":{"same":7.0}}],'
        '"operation":{"metrics":{"same":7.0},"failure_samples":{"samples":[],"truncated":false}}}'
    )

    monkeypatch.setitem(
        repair_module.REPAIR_HANDLERS,
        "session_insights",
        lambda _config, _dry_run: _ok_result("session_insights", metrics={"same": 2.0}),
    )
    op = execute_replay(config, targets=("session_insights",), operation_id="op-metric-resume")

    assert op.status is OperationStatus.COMPLETED
    assert op.metrics["same"] == 9.0


def test_nested_truncation_flag_survives_completed_resume(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-truncated-receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = ",".join('{"kind":"old","locator":"target:x","message":"old"}' for _ in range(50))
    path.write_text(
        '{"operation_id":"op-truncated-receipt","targets":["session_insights"],'
        '"completed_targets":["session_insights"],"cursor":"target:0",'
        '"failure_samples":[' + samples + "],"
        '"operation":{"failure_samples":{"samples":[' + samples + '],"truncated":true}}}'
    )

    op = execute_replay(config, targets=("session_insights",), operation_id="op-truncated-receipt")

    assert op.status is OperationStatus.COMPLETED
    assert op.failure_samples.truncated is True
    assert patched_dispatch["session_insights"] == []


def test_scope_identity_normalizes_session_order_and_timezone_instants(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-scope-equivalent")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"operation_id":"op-scope-equivalent","targets":["session_insights"],'
        '"completed_targets":["session_insights"],"cursor":"target:0",'
        '"scope_filter":{"session_ids":["s-2","s-1"],"origin":null,"source_family":null,'
        '"source_root":null,"time_range":["2026-01-01T01:00:00+01:00","2026-01-02T01:00:00+01:00"],'
        '"failure_kind":null,"parser_version":null}}'
    )
    scope = MaintenanceScopeFilter(
        session_ids=("s-1", "s-2"),
        time_range=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)),
    )

    op = execute_replay(config, targets=("session_insights",), operation_id="op-scope-equivalent", scope_filter=scope)

    assert op.status is OperationStatus.COMPLETED
    assert patched_dispatch["session_insights"] == []


def test_failure_samples_are_bounded_across_resume_retries(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    path = state_path_for(config, "op-many-failures")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = ",".join('{"kind":"old","locator":"target:session_insights","message":"old"}' for _ in range(75))
    path.write_text(
        '{"operation_id":"op-many-failures","targets":["session_insights"],'
        '"completed_targets":[],"cursor":"target:0","failure_samples":[' + samples + "]}"
    )

    op = execute_replay(config, targets=("session_insights",), operation_id="op-many-failures")

    assert len(op.failure_samples.samples) <= 50


def test_explicit_resume_cursor_overrides_persisted_state(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    # Seed a stale persisted state claiming "done".
    state_path_for(config, "op-explicit").parent.mkdir(parents=True, exist_ok=True)
    state_path_for(config, "op-explicit").write_text('{"operation_id": "op-explicit", "cursor": "done"}')

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-explicit",
        resume_cursor="target:1",
    )

    assert op.status is OperationStatus.COMPLETED
    # Only the second target was executed (skipped session_insights).
    assert patched_dispatch["session_insights"] == []


def test_progress_callback_fires_per_target(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)
    snapshots: list[ReplayProgress] = []

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-progress",
        progress_callback=snapshots.append,
    )

    assert op.status is OperationStatus.COMPLETED
    assert [s.target for s in snapshots] == ["session_insights"]
    assert snapshots[0].processed == 1 and snapshots[0].total == 1
    assert snapshots[-1].cursor == CURSOR_DONE
    assert snapshots[-1].in_flight_failures == 0


def test_session_insight_progress_is_forwarded_within_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    snapshots: list[ReplayProgress] = []

    def repair_with_progress(
        _config: Config,
        _dry_run: bool,
        *,
        progress_callback: Any = None,
        progress_total: int | None = None,
        session_ids: tuple[str, ...] | None = None,
    ) -> RepairResult:
        assert progress_total is None
        assert session_ids is None
        assert callable(progress_callback)
        progress_callback(17, desc="rebuild: materialized 17/42 session profiles")
        return _ok_result("session_insights", repaired=42)

    monkeypatch.setattr("polylogue.maintenance.replay.repair_session_insights", repair_with_progress)
    monkeypatch.setitem(
        repair_module.REPAIR_HANDLERS,
        "session_insights",
        repair_with_progress,
    )

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-progress-inner",
        progress_callback=snapshots.append,
    )

    assert op.status is OperationStatus.COMPLETED
    assert [snapshot.progress_desc for snapshot in snapshots] == [
        "rebuild: materialized 17/42 session profiles",
        None,
    ]
    assert snapshots[0].processed == 0
    assert snapshots[0].progress_amount == 17
    assert snapshots[-1].processed == 1


def test_replay_operation_metrics_include_result_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    def repair_with_metrics(
        _config: Config,
        _dry_run: bool,
    ) -> RepairResult:
        return _ok_result(
            "session_insights",
            repaired=2,
            metrics={
                "rebuilt_profiles": 100.0,
                "source_sessions": 1_609_582_167.0,
            },
        )

    monkeypatch.setitem(
        repair_module.REPAIR_HANDLERS,
        "session_insights",
        repair_with_metrics,
    )

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-metrics",
    )

    assert op.status is OperationStatus.COMPLETED
    assert op.metrics["repaired_count"] == 2.0
    assert op.metrics["rebuilt_profiles"] == 100.0
    assert op.metrics["source_sessions"] == 1_609_582_167.0
    assert op.results[0]["metrics"] == {
        "rebuilt_profiles": 100.0,
        "source_sessions": 1_609_582_167.0,
    }


def test_unresolved_targets_short_circuit(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    op = execute_replay(config, targets=("does-not-exist",))
    assert op.status is OperationStatus.FAILED
    assert op.targets == ()
    assert op.error == "No valid targets resolved from input"


@pytest.mark.parametrize(
    "operation_id",
    ("", False, 0, "/tmp/escape", "../escape", "nested/escape", "nested\\\\escape", "/", ".."),
)
def test_operation_ids_reject_falsey_and_path_traversal_before_state_path(tmp_path: Path, operation_id: Any) -> None:
    config = _make_config(tmp_path)

    with pytest.raises(ValueError, match="operation_id"):
        state_path_for(config, operation_id)
    with pytest.raises(ValueError, match="operation_id"):
        execute_replay(config, targets=("session_insights",), operation_id=operation_id)

    assert not (tmp_path / "escape.json").exists()


def test_operation_id_none_generates_uuid_and_existing_id_is_preserved(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    generated = execute_replay(config, targets=("session_insights",), operation_id=None, persist_state=False)
    assert generated.operation_id
    assert generated.operation_id != "None"

    path = state_path_for(config, "legacy-operation-42")
    assert path == config.archive_root / ".maintenance-state" / "legacy-operation-42.json"


def test_clear_state_is_idempotent(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    # Clearing a non-existent state file is a no-op.
    clear_state(config, "never-existed")
    # Create then clear.
    path = state_path_for(config, "later-cleared")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    clear_state(config, "later-cleared")
    assert not path.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def patched_dispatch_callable(calls: dict[str, list[str]], name: str):  # type: ignore[no-untyped-def]
    """Return a stub that records into ``calls[name]`` and reports success."""

    def _run(_config: Config, dry_run: bool) -> RepairResult:
        calls[name].append("dry" if dry_run else "live")
        return _ok_result(name)

    return _run


def patched_dispatch_table(calls: dict[str, list[str]]) -> dict[str, object]:
    return {name: patched_dispatch_callable(calls, name) for name in calls}


def test_unsupported_scope_is_refused_before_the_offline_gate(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    """A scope a target cannot apply is terminal, not a retryable blocker.

    Anti-vacuity: moving the refusal back inside ``_run_one_target`` (after
    ``offline_maintenance_blockers``) makes this red -- the receipt would
    carry ``OfflineMaintenanceBlocked`` and invite a retry of a request that
    can never succeed.
    """
    config = _make_config(tmp_path)
    blocker = RepairResult(
        name="superseded_raw_snapshots",
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=0,
        success=False,
        detail="daemon owns writes",
    )
    with patch(
        "polylogue.maintenance.replay.offline_maintenance_blockers",
        return_value=[blocker],
    ):
        operation = execute_replay(
            config,
            targets=("superseded_raw_snapshots",),
            operation_id="op-scope-before-offline",
            scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
        )

    assert operation.status is OperationStatus.FAILED
    assert {sample.kind for sample in operation.failure_samples.samples} == {"UnsupportedScopeDimension"}


def test_terminal_scope_refusal_checkpoints_as_failed(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mid-run checkpoint of a permanent refusal is never COMPLETED.

    The checkpoint written after the last target -- before the final receipt
    exists -- is what a crash leaves behind, so it is read back here through
    the progress callback that fires immediately after it.

    Anti-vacuity: restoring ``status = COMPLETED if all_completed else
    FAILED`` in ``_build_in_progress_snapshot`` makes this red -- the crash
    window would persist COMPLETED for a permanently refused request.
    """
    config = _make_config(tmp_path)

    def empty_sessions(cfg: Config, dry_run: bool = False, **kwargs: Any) -> RepairResult:
        return _ok_result("empty_sessions")

    monkeypatch.setattr(replay_module, "repair_empty_sessions", empty_sessions)
    monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "empty_sessions", empty_sessions)

    checkpointed: list[str] = []

    def observe(_progress: ReplayProgress) -> None:
        state = load_state(config, "op-refusal-checkpoint")
        assert state is not None
        snapshot = state["operation"]
        assert isinstance(snapshot, dict)
        checkpointed.append(str(snapshot["status"]))

    operation = execute_replay(
        config,
        targets=("empty_sessions", "superseded_raw_snapshots"),
        operation_id="op-refusal-checkpoint",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
        progress_callback=observe,
    )
    assert operation.status is OperationStatus.FAILED
    assert checkpointed
    assert OperationStatus.COMPLETED.value not in checkpointed


def test_resume_ignores_a_refusal_from_an_excluded_target(
    tmp_path: Path, patched_dispatch: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrowed resume is judged on its own targets only.

    Anti-vacuity: dropping the ``targets`` filter from
    ``_has_terminal_scope_refusal`` makes this red -- the hydrated refusal
    recorded for ``superseded_raw_snapshots`` would fail a resume that no
    longer requests it.
    """
    config = _make_config(tmp_path)

    def empty_sessions(cfg: Config, dry_run: bool = False, **kwargs: Any) -> RepairResult:
        return _ok_result("empty_sessions")

    monkeypatch.setattr(replay_module, "repair_empty_sessions", empty_sessions)
    monkeypatch.setitem(repair_module.REPAIR_HANDLERS, "empty_sessions", empty_sessions)

    refused = execute_replay(
        config,
        targets=("empty_sessions", "superseded_raw_snapshots"),
        operation_id="op-narrowed-resume",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
    )
    assert refused.status is OperationStatus.FAILED

    resumed = execute_replay(
        config,
        targets=("empty_sessions",),
        operation_id="op-narrowed-resume",
        scope_filter=MaintenanceScopeFilter(session_ids=("s-1",)),
    )
    assert resumed.status is OperationStatus.COMPLETED
