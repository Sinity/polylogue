from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger, StageState


def test_converger_reports_only_current_run_stage_timings(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    executed: list[Path] = []

    def execute(candidate: Path) -> bool:
        executed.append(candidate)
        return True

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="insights",
                description="test stage",
                check=lambda candidate: candidate not in executed,
                execute=execute,
            )
        ]
    )

    first = converger.converge_file(path)
    first_stage_times = dict(first.stage_times)
    first_last_stage_times = dict(first.last_stage_times)
    second = converger.converge_file(path)

    assert set(first_stage_times) == {"insights"}
    assert set(first_last_stage_times) == {"insights"}
    assert second.stage_times == first_stage_times
    assert second.last_stage_times == {}


def test_converger_batches_stage_execution(tmp_path: Path) -> None:
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    checked: list[tuple[Path, ...]] = []
    executed: list[tuple[Path, ...]] = []

    def check_many(candidates: Sequence[Path]) -> set[Path]:
        checked.append(tuple(candidates))
        return set(candidates)

    def execute_many(candidates: Sequence[Path]) -> bool:
        executed.append(tuple(candidates))
        return True

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="insights",
                description="test batch stage",
                check=lambda _candidate: True,
                execute=lambda _candidate: False,
                check_many=check_many,
                execute_many=execute_many,
            )
        ]
    )

    states, stage_times = converger.converge_batch(paths)

    assert checked == [tuple(paths)]
    assert [set(candidates) for candidates in executed] == [set(paths)]
    assert set(states) == set(paths)
    assert set(stage_times) == {"insights"}
    assert all(state.converged for state in states.values())
    assert converger._file_states == {}


def test_converger_retains_failed_batch_state_for_diagnostics(tmp_path: Path) -> None:
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")

    failed = paths[1]

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="insights",
                description="test stage",
                check=lambda _candidate: True,
                execute=lambda candidate: candidate != failed,
            )
        ]
    )

    states, _stage_times = converger.converge_batch(paths)

    assert states[paths[0]].converged
    assert not states[failed].converged
    assert set(converger._file_states) == {failed}


def test_converger_keeps_bounded_false_as_pending_debt(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="embed",
                description="bounded embedding catch-up",
                check=lambda _candidate: True,
                execute=lambda _candidate: False,
                false_means_pending=True,
            )
        ]
    )

    state = converger.converge_file(path)

    assert state.stages["embed"] is StageState.PENDING
    assert state.error_count == 0
    assert state.last_error == "stage embed returned False"
    assert set(converger._file_states) == {path}


def test_converger_batches_session_execution() -> None:
    checked: list[tuple[str, ...]] = []
    executed: list[tuple[str, ...]] = []

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        checked.append(tuple(session_ids))
        return set(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> bool:
        executed.append(tuple(session_ids))
        return True

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="insights",
                description="test session stage",
                check=lambda _candidate: False,
                execute=lambda _candidate: False,
                check_sessions=check_sessions,
                execute_sessions=execute_sessions,
            )
        ]
    )

    states, stage_times = converger.converge_sessions(["conv-a", "conv-b", "conv-a"])

    assert checked == [("conv-a", "conv-b")]
    assert [set(candidates) for candidates in executed] == [{"conv-a", "conv-b"}]
    assert set(states) == {"conv-a", "conv-b"}
    assert set(stage_times) == {"insights"}
    assert all(state.converged for state in states.values())
    assert converger._session_states == {}


def test_converger_keeps_bounded_session_false_as_pending_debt() -> None:
    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="embed",
                description="bounded embedding catch-up",
                check=lambda _candidate: False,
                execute=lambda _candidate: False,
                check_sessions=lambda session_ids: set(session_ids),
                execute_sessions=lambda _session_ids: False,
                false_means_pending=True,
            )
        ]
    )

    states, _stage_times = converger.converge_sessions(["conv-a"])

    assert states["conv-a"].stages["embed"] is StageState.PENDING
    assert states["conv-a"].error_count == 0
    assert states["conv-a"].last_error == "session stage embed returned False"
    assert set(converger._session_states) == {"conv-a"}


@pytest.mark.asyncio
async def test_converger_start_skips_process_pool_for_io_only_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock()
    monkeypatch.setattr("polylogue.daemon.convergence.process_pool_executor", factory)

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="insights",
                description="io stage",
                check=lambda _candidate: False,
                execute=lambda _candidate: True,
            )
        ]
    )

    await converger.start()

    factory.assert_not_called()
    assert converger._executor is None


@pytest.mark.asyncio
async def test_converger_start_creates_process_pool_for_cpu_bound_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.shutdown_called = False

        def submit(self, func, *args):  # type: ignore[no-untyped-def]
            future: Future[bool] = Future()
            future.set_result(func(*args))
            return future

        def shutdown(self, *, wait: bool = True) -> None:
            self.shutdown_called = wait

    fake_executor = FakeExecutor()
    factory = Mock(return_value=fake_executor)
    monkeypatch.setattr("polylogue.daemon.convergence.process_pool_executor", factory)

    converger = DaemonConverger(
        [
            ConvergenceStage(
                name="parse",
                description="cpu stage",
                check=lambda _candidate: False,
                execute=lambda _candidate: True,
                cpu_bound=True,
            )
        ],
        max_workers=3,
    )

    await converger.start()
    await converger.stop()

    factory.assert_called_once_with(max_workers=3)
    assert fake_executor.shutdown_called is True
