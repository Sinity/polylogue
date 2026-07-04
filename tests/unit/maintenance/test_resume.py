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
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import polylogue.maintenance.replay as replay_module
from polylogue.config import Config
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.planner import BackfillStatus
from polylogue.maintenance.replay import (
    CURSOR_DONE,
    ReplayProgress,
    clear_state,
    execute_replay,
    load_state,
    state_path_for,
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
        "message_type_backfill": [],
        "wal_checkpoint": [],
    }

    def stub(name: str):  # type: ignore[no-untyped-def]
        def _run(config: Config, dry_run: bool) -> RepairResult:
            calls[name].append("dry" if dry_run else "live")
            return _ok_result(name)

        return _run

    fake_dispatch = {name: stub(name) for name in calls}
    with patch(
        "polylogue.maintenance.replay._REPLAY_DISPATCH",
        fake_dispatch,
    ):
        yield calls


def test_clean_run_persists_done_and_clears_state(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)
    op = execute_replay(
        config,
        targets=("session_insights", "wal_checkpoint"),
        operation_id="op-clean",
    )

    assert op.status is BackfillStatus.COMPLETED
    assert op.resume_cursor == CURSOR_DONE
    # State file is removed after successful completion.
    assert not state_path_for(config, "op-clean").exists()
    assert patched_dispatch["session_insights"] == ["live"]
    assert patched_dispatch["wal_checkpoint"] == ["live"]


def test_kill_mid_run_resumes_from_persisted_cursor(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)

    # First invocation: simulate a crash after the first target by
    # patching the dispatch to raise on the second target. The failure
    # path still advances the cursor (the executor's documented
    # behaviour) so the resume run picks up at target index 2.
    boom_calls: list[str] = []

    def boom(_config: Config, _dry_run: bool) -> RepairResult:
        boom_calls.append("called")
        raise RuntimeError("simulated kill")

    fake_dispatch = {
        "session_insights": patched_dispatch_callable(patched_dispatch, "session_insights"),
        "message_type_backfill": boom,
        "wal_checkpoint": patched_dispatch_callable(patched_dispatch, "wal_checkpoint"),
    }

    with patch("polylogue.maintenance.replay._REPLAY_DISPATCH", fake_dispatch):
        first = execute_replay(
            config,
            targets=("session_insights", "message_type_backfill", "wal_checkpoint"),
            operation_id="op-resume",
        )

    assert first.status is BackfillStatus.FAILED
    assert patched_dispatch["session_insights"] == ["live"]
    assert patched_dispatch["wal_checkpoint"] == ["live"]
    assert len(boom_calls) == 1
    # State persists because run did not converge cleanly.
    assert state_path_for(config, "op-resume").exists()

    # Manually rewind the cursor to simulate "kill after target 1
    # completed, before target 2 was attempted". This proves the
    # executor obeys the on-disk cursor and does not re-run target 1.
    persisted = load_state(config, "op-resume")
    assert persisted is not None
    rewind = state_path_for(config, "op-resume")
    rewind.write_text(rewind.read_text().replace(persisted["cursor"], "target:1"))  # type: ignore[arg-type]

    # Second invocation with same id and a "fixed" dispatch must skip
    # the already-completed first target.
    with patch("polylogue.maintenance.replay._REPLAY_DISPATCH", patched_dispatch_table(patched_dispatch)):
        second = execute_replay(
            config,
            targets=("session_insights", "message_type_backfill", "wal_checkpoint"),
            operation_id="op-resume",
        )

    assert second.status is BackfillStatus.COMPLETED
    assert second.resume_cursor == CURSOR_DONE
    # session_insights was not invoked a second time.
    assert patched_dispatch["session_insights"] == ["live"]
    # The remaining two targets were executed exactly once on resume.
    assert patched_dispatch["message_type_backfill"] == ["live"]
    assert patched_dispatch["wal_checkpoint"] == ["live", "live"]
    # State cleared after successful resume.
    assert not state_path_for(config, "op-resume").exists()


def test_explicit_resume_cursor_overrides_persisted_state(
    tmp_path: Path, patched_dispatch: dict[str, list[str]]
) -> None:
    config = _make_config(tmp_path)
    # Seed a stale persisted state claiming "done".
    state_path_for(config, "op-explicit").parent.mkdir(parents=True, exist_ok=True)
    state_path_for(config, "op-explicit").write_text('{"operation_id": "op-explicit", "cursor": "done"}')

    op = execute_replay(
        config,
        targets=("session_insights", "wal_checkpoint"),
        operation_id="op-explicit",
        resume_cursor="target:1",
    )

    assert op.status is BackfillStatus.COMPLETED
    # Only the second target was executed (skipped session_insights).
    assert patched_dispatch["session_insights"] == []
    assert patched_dispatch["wal_checkpoint"] == ["live"]


def test_progress_callback_fires_per_target(tmp_path: Path, patched_dispatch: dict[str, list[str]]) -> None:
    config = _make_config(tmp_path)
    snapshots: list[ReplayProgress] = []

    op = execute_replay(
        config,
        targets=("session_insights", "wal_checkpoint"),
        operation_id="op-progress",
        progress_callback=snapshots.append,
    )

    assert op.status is BackfillStatus.COMPLETED
    assert [s.target for s in snapshots] == ["session_insights", "wal_checkpoint"]
    assert snapshots[0].processed == 1 and snapshots[0].total == 2
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
        replay_module._REPLAY_DISPATCH,
        "session_insights",
        repair_with_progress,
    )

    op = execute_replay(
        config,
        targets=("session_insights",),
        operation_id="op-progress-inner",
        progress_callback=snapshots.append,
    )

    assert op.status is BackfillStatus.COMPLETED
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
            "wal_checkpoint",
            repaired=2,
            metrics={
                "checkpoint_frames": 100.0,
                "checkpoint_max_wal_bytes": 1_609_582_167.0,
            },
        )

    monkeypatch.setitem(
        replay_module._REPLAY_DISPATCH,
        "wal_checkpoint",
        repair_with_metrics,
    )

    op = execute_replay(
        config,
        targets=("wal_checkpoint",),
        operation_id="op-metrics",
    )

    assert op.status is BackfillStatus.COMPLETED
    assert op.metrics["repaired_count"] == 2.0
    assert op.metrics["checkpoint_frames"] == 100.0
    assert op.metrics["checkpoint_max_wal_bytes"] == 1_609_582_167.0
    assert op.results[0]["metrics"] == {
        "checkpoint_frames": 100.0,
        "checkpoint_max_wal_bytes": 1_609_582_167.0,
    }


def test_unresolved_targets_short_circuit(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    op = execute_replay(config, targets=("does-not-exist",))
    assert op.status is BackfillStatus.FAILED
    assert op.targets == ()
    assert op.error == "No valid targets resolved from input"


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
