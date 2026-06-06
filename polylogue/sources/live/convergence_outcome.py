"""Record post-ingest convergence debt outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from polylogue.sources.live.batch_observability import session_ids_for_source_path
from polylogue.sources.live.convergence_debt import ConvergenceDebt
from polylogue.sources.live.cursor import CursorStore


def record_convergence_outcome(
    cursor: CursorStore,
    path: Path,
    debts: Iterable[ConvergenceDebt],
    *,
    archive_root: Path | None = None,
) -> None:
    debt_items = tuple(debts)
    failed_stages = tuple(dict.fromkeys(debt.stage for debt in debt_items))
    session_ids = session_ids_for_source_path(path, archive_root=archive_root)
    cursor.clear_convergence_debt_except(subject_type="source_path", subject_id=str(path), stages=failed_stages)
    for session_id in session_ids:
        cursor.clear_convergence_debt_except(
            subject_type="session_id",
            subject_id=session_id,
            stages=failed_stages,
        )
    if not debt_items:
        return
    for debt in debt_items:
        if session_ids:
            for session_id in session_ids:
                cursor.record_convergence_debt(
                    stage=debt.stage,
                    subject_type="session_id",
                    subject_id=session_id,
                    error=debt.error,
                )
        else:
            cursor.record_convergence_debt(
                stage=debt.stage,
                subject_type="source_path",
                subject_id=str(path),
                error=debt.error,
            )
