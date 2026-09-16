"""A stage a narrowed pass never ran is owed work, not converged work.

polylogue-zbzxs / polylogue-tjtua: ``converge_batch(whole_archive=False)`` used
to record archive-wide stages as ``SKIPPED``, which counts as converged, which
left ``convergence_debt_from_state`` with nothing to report -- and an empty
failed-stage set makes ``clear_convergence_debt_except`` degenerate into
"delete every debt row for this subject".
"""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger, StageState
from polylogue.sources.live.convergence_debt import (
    convergence_debt_from_states,
    is_deferred_stage_state,
)


def _converger() -> DaemonConverger:
    return DaemonConverger(
        [
            ConvergenceStage(
                name="attachments",
                description="archive-wide",
                check=lambda _path: True,
                execute=lambda _path: True,
                whole_archive=True,
            ),
            ConvergenceStage(
                name="derived",
                description="batch-scoped",
                check=lambda _path: True,
                execute=lambda _path: True,
            ),
        ]
    )


def test_narrowed_batch_yields_deferred_debt_for_the_stage_it_never_ran(tmp_path: Path) -> None:
    """Anti-vacuity: recording the un-run stage as DONE or SKIPPED (either of
    which makes ``FileState.converged`` true) empties ``debts`` and turns this
    red -- which is exactly the state that let a bounded catch-up clear debt
    for a stage it never executed."""
    path = tmp_path / "a.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    states, _timings = _converger().converge_batch([path], whole_archive=False)

    state = states[path]
    assert state.stages["attachments"] is StageState.NOT_RUN
    assert state.stages["derived"] is StageState.DONE
    assert not state.converged

    debts = convergence_debt_from_states([path], states)
    assert [debt.stage for debt in debts] == ["attachments"]
    # Owed work, not breakage: daemon health must not alert on it as a failure.
    assert debts[0].deferred is True
    assert is_deferred_stage_state(StageState.NOT_RUN)


def test_whole_archive_pass_runs_the_stage_and_leaves_no_debt(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    converger = _converger()
    converger.converge_batch([path], whole_archive=False)
    states, _timings = converger.converge_batch([path])

    assert states[path].stages["attachments"] is StageState.DONE
    assert states[path].converged
    assert convergence_debt_from_states([path], states) == []
