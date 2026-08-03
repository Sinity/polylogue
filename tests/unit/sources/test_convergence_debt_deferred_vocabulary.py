"""Unit coverage for the deferred-vs-failed convergence-debt classification.

polylogue-6krh: ``convergence_debt.status`` declares a two-value vocabulary
(``'failed'``, ``'deferred'``, ``storage/sqlite/archive_tiers/ops.py``'s
CHECK constraint) but every live ``false_means_pending`` deferral -- a
convergence stage doing bounded successful work and deferring the rest,
per ``daemon.convergence.ConvergenceStage.false_means_pending`` -- landed as
``'failed'`` for every stage except ``insights``, which had its exact error
string hardcoded into a one-off check. ``StageState.PENDING`` is the actual,
general signal for "this stage did not fail, it deferred": it is set only
by a ``false_means_pending`` stage's non-exceptional ``False`` return or by
downstream-barrier queuing (``daemon/convergence.py``), never by a
check/execute exception. These tests exercise
``polylogue.sources.live.convergence_debt`` directly against that signal,
independent of any one stage's error text.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.convergence import FileState, StageState
from polylogue.sources.live.convergence_debt import (
    ConvergenceDebt,
    convergence_debt_from_state,
    convergence_debt_from_states,
    is_deferred_stage_state,
)


def test_is_deferred_stage_state_true_only_for_pending() -> None:
    assert is_deferred_stage_state(StageState.PENDING) is True
    assert is_deferred_stage_state("pending") is True
    assert is_deferred_stage_state(StageState.FAILED) is False
    assert is_deferred_stage_state(StageState.IN_PROGRESS) is False
    assert is_deferred_stage_state(StageState.DONE) is False
    assert is_deferred_stage_state(StageState.SKIPPED) is False


def test_convergence_debt_from_state_marks_false_means_pending_stage_deferred() -> None:
    """A stage other than 'insights' deferring via false_means_pending must
    still produce ``deferred=True`` -- this is the exact bug: only the
    'insights' stage's specific error string was special-cased before.
    """
    path = Path("/tmp/source.jsonl")
    state = FileState(
        path=path,
        stages={"fts": StageState.PENDING},
        last_error="stage fts returned False",
    )

    debts = convergence_debt_from_state(path, state)

    assert len(debts) == 1
    assert debts[0] == ConvergenceDebt(
        path=path,
        stage="fts",
        error="stage fts returned False",
        deferred=True,
    )


def test_convergence_debt_from_state_keeps_genuine_failure_as_failed() -> None:
    path = Path("/tmp/source.jsonl")
    state = FileState(
        path=path,
        stages={"fts": StageState.FAILED},
        last_error="boom",
    )

    debts = convergence_debt_from_state(path, state)

    assert len(debts) == 1
    assert debts[0].deferred is False
    assert debts[0].stage == "fts"


def test_convergence_debt_from_state_mixed_stages_classify_independently() -> None:
    path = Path("/tmp/source.jsonl")
    state = FileState(
        path=path,
        stages={
            "fts": StageState.PENDING,
            "embed": StageState.FAILED,
            "insights": StageState.DONE,
        },
        last_error="stage embed returned False",
    )

    debts = {debt.stage: debt for debt in convergence_debt_from_state(path, state)}

    assert set(debts) == {"fts", "embed"}
    assert debts["fts"].deferred is True
    assert debts["embed"].deferred is False


def test_convergence_debt_from_states_propagates_deferred_flag() -> None:
    path = Path("/tmp/source.jsonl")
    state = FileState(path=path, stages={"sinex_publication": StageState.PENDING})

    debts = convergence_debt_from_states((path,), {path: state})

    assert len(debts) == 1
    assert debts[0].deferred is True
