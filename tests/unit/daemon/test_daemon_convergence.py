from __future__ import annotations

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
