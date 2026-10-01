"""Record evaluated post-ingest convergence obligations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from polylogue.sources.live.convergence_debt import ConvergenceDebt
from polylogue.sources.live.cursor import (
    ConvergenceDebtBatchEntry,
    ConvergenceDebtSettlement,
    ConvergenceDebtWrite,
    CursorStore,
)


def settled_convergence_stages(state: object) -> tuple[ConvergenceDebtSettlement, ...]:
    """Carry the evaluated subject directly from the engine's state.

    A file verdict does not establish any session verdict, even when those
    sessions were acquired from that file. Skipped and unrun stages settle
    nothing; only the evaluated subject's completed stage is authoritative.
    """
    from polylogue.sources.live.convergence_debt import stage_state_value

    path = getattr(state, "path", None)
    session_id = getattr(state, "session_id", None)
    if isinstance(path, Path):
        subject_type, subject_id = "source_path", str(path)
    elif isinstance(session_id, str):
        subject_type, subject_id = "session_id", session_id
    else:
        return ()
    stages = getattr(state, "stages", None)
    if not isinstance(stages, dict):
        return ()
    return tuple(
        ConvergenceDebtSettlement(subject_type, subject_id, str(stage))
        for stage, status in stages.items()
        if stage_state_value(status) == "done"
    )


def record_convergence_outcomes(
    cursor: CursorStore,
    outcomes: Iterable[tuple[Path, Iterable[ConvergenceDebt]]],
    *,
    settlements: Iterable[ConvergenceDebtSettlement] = (),
) -> None:
    entries = [ConvergenceDebtBatchEntry(tuple(settlements))]
    for path, debts in outcomes:
        writes = tuple(
            ConvergenceDebtWrite(debt.stage, "source_path", str(path), debt.error, debt.deferred) for debt in debts
        )
        entries.append(ConvergenceDebtBatchEntry(writes=writes))
    cursor.apply_convergence_debt_batch(entries)
