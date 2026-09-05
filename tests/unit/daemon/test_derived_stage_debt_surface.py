"""A stalled derived stage is ordinary, typed, retryable convergence debt.

Derived session rows have no lifecycle of their own. When their stage cannot
complete, the only thing that must happen is what happens for every other
derived object: a typed convergence-debt row, retryable, visible on the debt
surface, and reflected in the one readiness signal.

Anti-vacuity: each assertion names a distinct link in that chain, so the test
goes red if the converger stops reporting the stalled stage, if the debt row
loses its retry schedule or its stage label, if ``archive_debt_list`` stops
projecting convergence rows, or if the insight coverage report stops reading
convergence debt and goes back to declaring readiness on its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger
from polylogue.daemon.convergence_stages import make_fts_stage
from polylogue.operations.archive_debt import archive_debt_list
from polylogue.sources.live.convergence_debt import convergence_debt_from_states
from polylogue.sources.live.convergence_outcome import record_convergence_outcome
from polylogue.sources.live.cursor import CursorStore
from tests.infra.convergence_harness import (
    build_converged_archive,
    debt_ledger_row,
    rich_convergence_pathology,
)

_STALL_ERROR = "derived stage stalled for the debt-surface fixture"


def _stalled_derived_stage() -> ConvergenceStage:
    """The real stage contract with an execute that cannot complete."""

    def _raise(_target: object) -> bool:
        raise RuntimeError(_STALL_ERROR)

    return ConvergenceStage(
        name="derived",
        description="Refresh session-derived tables for new sessions",
        check=lambda _path: True,
        execute=_raise,
        check_sessions=lambda session_ids: set(session_ids),
        execute_sessions=_raise,
    )


def test_stalled_derived_stage_surfaces_as_retryable_convergence_debt(tmp_path: Path) -> None:
    archive = build_converged_archive(tmp_path / "archive", rich_convergence_pathology())
    index_db = archive.root / "index.db"
    ops_db = archive.root / "ops.db"

    converger = DaemonConverger((make_fts_stage(index_db), _stalled_derived_stage()))
    states, _timings = converger.converge_batch(archive.source_paths)

    # The converger must report the stall rather than swallowing it.
    assert any(not state.converged for state in states.values())

    debts = convergence_debt_from_states(archive.source_paths, states)
    assert {debt.stage for debt in debts} == {"derived"}, "the stall must be attributed to the derived stage"

    cursor = CursorStore(index_db, ops_db_path=ops_db)
    for path in archive.source_paths:
        record_convergence_outcome(
            cursor,
            path,
            [debt for debt in debts if debt.path == path],
            archive_root=archive.root,
        )

    # Typed and retryable in the durable ledger.
    rows = [
        row
        for path in archive.source_paths
        if (row := debt_ledger_row(ops_db, stage="derived", subject_type="source_path", subject_id=str(path)))
        is not None
    ]
    session_rows = [
        row
        for session_id in archive.session_ids
        if (row := debt_ledger_row(ops_db, stage="derived", subject_type="session_id", subject_id=session_id))
        is not None
    ]
    ledger_rows = rows + session_rows
    assert ledger_rows, "the stalled derived stage recorded no convergence debt"
    for row in ledger_rows:
        assert row.status == "failed", "a stage that raised is a failure, not a deferral"
        assert row.next_retry_at is not None, "convergence debt must carry a retry schedule"
        assert row.last_error is not None and _STALL_ERROR in row.last_error

    # Projected on the operator debt surface, with the stage named.
    payload = archive_debt_list(archive_root=archive.root, kinds=["convergence"])
    convergence_rows = [row for row in payload.rows if row.kind == "convergence"]
    assert convergence_rows, "archive_debt_list reported no convergence debt for a stalled stage"
    assert any(row.stage == "derived" for row in convergence_rows)


@pytest.mark.asyncio
async def test_insight_readiness_reads_convergence_debt_as_its_only_signal(tmp_path: Path) -> None:
    archive = build_converged_archive(tmp_path / "archive", rich_convergence_pathology())
    index_db = archive.root / "index.db"
    ops_db = archive.root / "ops.db"

    polylogue = Polylogue(archive_root=archive.root, db_path=index_db)
    try:
        converged = await polylogue.insight_readiness_report()
        assert converged.converged is True, "a converged archive must report caught-up convergence"
        assert converged.debt_stages == ()

        CursorStore(index_db, ops_db_path=ops_db).record_convergence_debt(
            stage="derived",
            subject_type="session_id",
            subject_id=archive.session_ids[0],
            error=_STALL_ERROR,
        )

        stalled = await polylogue.insight_readiness_report()
    finally:
        await polylogue.close()

    assert stalled.converged is False, "readiness must follow convergence debt, not a private verdict"
    assert "derived" in stalled.debt_stages
