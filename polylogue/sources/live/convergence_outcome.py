"""Record post-ingest convergence debt outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from polylogue.sources.live.batch_observability import session_ids_for_source_path
from polylogue.sources.live.convergence_debt import ConvergenceDebt
from polylogue.sources.live.cursor import (
    ConvergenceDebtBatchEntry,
    ConvergenceDebtClear,
    ConvergenceDebtWrite,
    CursorStore,
)


def record_convergence_outcome(
    cursor: CursorStore,
    path: Path,
    debts: Iterable[ConvergenceDebt],
    *,
    archive_root: Path | None = None,
) -> None:
    record_convergence_outcomes(cursor, ((path, debts),), archive_root=archive_root)


def record_convergence_outcomes(
    cursor: CursorStore,
    outcomes: Iterable[tuple[Path, Iterable[ConvergenceDebt]]],
    *,
    archive_root: Path | None = None,
) -> None:
    entries: list[ConvergenceDebtBatchEntry] = []
    for path, debts in outcomes:
        debt_items = tuple(debts)
        # Hook-paste failures are recorded by the post-convergence owner, after
        # the generic stage states were produced. That owner clears its row after
        # a successful enrichment; generic outcome cleanup must not erase a retry
        # it just recorded.
        failed_stages = tuple(dict.fromkeys((*[debt.stage for debt in debt_items], "hook_paste_enrichment")))
        session_ids = session_ids_for_source_path(path, archive_root=archive_root)
        clears = [ConvergenceDebtClear("source_path", str(path), failed_stages)]
        clears.extend(ConvergenceDebtClear("session_id", session_id, failed_stages) for session_id in session_ids)
        writes = tuple(
            ConvergenceDebtWrite(debt.stage, subject_type, subject_id, debt.error, debt.deferred)
            for debt in debt_items
            for subject_type, subject_id in (
                (("session_id", session_id) for session_id in session_ids)
                if session_ids
                else (("source_path", str(path)),)
            )
        )
        entries.append(ConvergenceDebtBatchEntry(tuple(clears), writes))
    cursor.apply_convergence_debt_batch(entries)
