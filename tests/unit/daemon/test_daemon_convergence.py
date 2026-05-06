from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger


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
