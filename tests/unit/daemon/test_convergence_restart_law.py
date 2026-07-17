"""Survivor law for convergence interruption, restart, retry, and quiescence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.daemon.convergence import DaemonConverger, StageState
from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
from polylogue.scenarios import WorkloadPhaseObservation, WorkloadReceipt, WorkloadRunStatus
from polylogue.sources.live.convergence_debt import convergence_debt_from_states
from polylogue.sources.live.convergence_outcome import record_convergence_outcome
from polylogue.sources.live.cursor import CursorStore
from tests.infra.convergence_harness import (
    debt_ledger_row,
    make_messages_fts_stale,
    messages_fts_match_count,
    raw_authority_facts,
    seed_partial_convergence_archive,
    session_materialization_facts,
    set_debt_retry_at,
)

_DUE_RETRY_AT = "1970-01-01T00:00:00+00:00"
_NOT_DUE_RETRY_AT = "9999-01-01T00:00:00+00:00"
_FUTURE_MTIME = 4_102_444_800


def _run_retry_in_fresh_process(index_db: Path) -> int:
    """Execute the production debt drain across a genuine interpreter restart."""
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    script = (
        "from pathlib import Path\n"
        "from polylogue.daemon.cli import _drain_convergence_debt_once\n"
        f"print('RETRIED=' + str(_drain_convergence_debt_once(Path({str(index_db)!r}))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"fresh-process convergence retry failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RETRIED="):
            return int(line.removeprefix("RETRIED="))
    raise AssertionError(f"fresh-process convergence retry emitted no result marker: {completed.stdout!r}")


@pytest.mark.contract
@pytest.mark.timeout(90)
def test_convergence_debt_survives_restart_and_reaches_one_terminal_fact_set(tmp_path: Path) -> None:
    """The real daemon route preserves exact debt until its own stage succeeds.

    Production dependencies exercised: the typed archive writer, source-to-session
    resolver, ``DaemonConverger`` false-means-pending contract, insights and FTS
    stage implementations, ``CursorStore`` ops.db ledger, public debt status,
    and ``_drain_convergence_debt_once`` in fresh Python processes.

    Anti-vacuity mutations: deleting debt before stage success loses the hot
    retry row; deleting/recreating it changes ``debt_id`` or resets attempts;
    ignoring ``debt.stage`` repairs the deliberately stale FTS rows during an
    insights retry; skipping retry leaves the profile absent; resolving the
    wrong source materializes the unrelated session.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "recovered", target_hot=True)
    os.utime(recovered.target_source, (_FUTURE_MTIME, _FUTURE_MTIME))
    raw_facts_before = raw_authority_facts(recovered.source_db)

    # Bounded work is allowed to defer, but the exact stage/session obligation
    # must be written durably before this process ends.
    insights = make_insights_stage(recovered.index_db)
    initial_states, _timings = DaemonConverger((insights,)).converge_batch((recovered.target_source,))
    initial_state = initial_states[recovered.target_source]
    assert initial_state.stages == {"insights": StageState.PENDING}
    assert initial_state.last_error == "insights deferred until source quiet"
    record_convergence_outcome(
        CursorStore(recovered.index_db),
        recovered.target_source,
        convergence_debt_from_states((recovered.target_source,), initial_states),
        archive_root=recovered.root,
    )

    initial_insights_debt = debt_ledger_row(
        recovered.ops_db,
        stage="insights",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
    )
    assert initial_insights_debt is not None
    assert initial_insights_debt.status == "deferred"
    assert initial_insights_debt.attempts == 1
    assert convergence_debt_summary_info(recovered.index_db).failed_count == 0
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.target_session_id,
        ).profile
        is None
    )

    # Add unrelated real FTS debt for the same session. Its future deadline is
    # a deterministic barrier: an insights retry must not execute, consume, or
    # rewrite this other stage's obligation.
    deleted_fts_rows = make_messages_fts_stale(
        recovered.index_db,
        session_id=recovered.target_session_id,
    )
    fts = make_fts_stage(recovered.index_db)
    assert fts.check_sessions is not None
    assert fts.check_sessions((recovered.target_session_id,)) == {recovered.target_session_id}
    cursor = CursorStore(recovered.index_db)
    cursor.record_convergence_debt(
        stage="fts",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
        error="deliberate unrelated FTS backlog",
    )
    set_debt_retry_at(
        recovered.ops_db,
        stage="fts",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
        retry_at=_NOT_DUE_RETRY_AT,
    )
    unrelated_fts_debt = debt_ledger_row(
        recovered.ops_db,
        stage="fts",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
    )
    assert unrelated_fts_debt is not None
    assert unrelated_fts_debt.status == "failed"
    assert messages_fts_match_count(recovered.index_db, "Message") == 1

    status_with_fts_debt = convergence_debt_summary_info(recovered.index_db)
    assert status_with_fts_debt.failed_count == 1
    assert status_with_fts_debt.retry_due_count == 0
    assert [(item.stage, item.failed_count) for item in status_with_fts_debt.stage_summaries] == [("fts", 1)]

    # First restart: the source is still hot. Retrying must update the same
    # insights row in place and leave both FTS materialization and FTS debt
    # untouched.
    set_debt_retry_at(
        recovered.ops_db,
        stage="insights",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
        retry_at=_DUE_RETRY_AT,
    )
    due_insights_debt = debt_ledger_row(
        recovered.ops_db,
        stage="insights",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
    )
    assert due_insights_debt is not None
    assert _run_retry_in_fresh_process(recovered.index_db) == 1

    deferred_again = debt_ledger_row(
        recovered.ops_db,
        stage="insights",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
    )
    assert deferred_again is not None
    assert deferred_again.debt_id == due_insights_debt.debt_id
    assert deferred_again.created_at_ms == due_insights_debt.created_at_ms
    assert deferred_again.attempts == 2
    assert deferred_again.status == "deferred"
    assert (
        debt_ledger_row(
            recovered.ops_db,
            stage="fts",
            subject_type="session_id",
            subject_id=recovered.target_session_id,
        )
        == unrelated_fts_debt
    )
    assert fts.check_sessions((recovered.target_session_id,)) == {recovered.target_session_id}
    assert messages_fts_match_count(recovered.index_db, "Message") == 1
    assert raw_authority_facts(recovered.source_db) == raw_facts_before
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.target_session_id,
        ).profile
        is None
    )
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.unrelated_session_id,
        ).profile
        is None
    )

    # Second restart: after the source becomes quiet, only the recorded
    # insights stage may complete and consume only its exact debt row.
    recovered.make_target_quiet()
    set_debt_retry_at(
        recovered.ops_db,
        stage="insights",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
        retry_at=_DUE_RETRY_AT,
    )
    assert _run_retry_in_fresh_process(recovered.index_db) == 1
    assert (
        debt_ledger_row(
            recovered.ops_db,
            stage="insights",
            subject_type="session_id",
            subject_id=recovered.target_session_id,
        )
        is None
    )
    assert (
        debt_ledger_row(
            recovered.ops_db,
            stage="fts",
            subject_type="session_id",
            subject_id=recovered.target_session_id,
        )
        == unrelated_fts_debt
    )
    assert fts.check_sessions((recovered.target_session_id,)) == {recovered.target_session_id}
    recovered_terminal_facts = session_materialization_facts(
        recovered.index_db,
        session_id=recovered.target_session_id,
    )
    assert recovered_terminal_facts.profile is not None
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.unrelated_session_id,
        ).profile
        is None
    )
    assert raw_authority_facts(recovered.source_db) == raw_facts_before

    # The independent FTS obligation is due only now. Its own retry repairs the
    # missing rows without replaying or duplicating insights materialization.
    set_debt_retry_at(
        recovered.ops_db,
        stage="fts",
        subject_type="session_id",
        subject_id=recovered.target_session_id,
        retry_at=_DUE_RETRY_AT,
    )
    assert _run_retry_in_fresh_process(recovered.index_db) == 1
    assert (
        debt_ledger_row(
            recovered.ops_db,
            stage="fts",
            subject_type="session_id",
            subject_id=recovered.target_session_id,
        )
        is None
    )
    assert fts.check_sessions((recovered.target_session_id,)) == set()
    assert messages_fts_match_count(recovered.index_db, "Message") == deleted_fts_rows + 1
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.target_session_id,
        )
        == recovered_terminal_facts
    )

    # Repeated convergence is a no-op at quiescence and public status agrees
    # with the durable ledger rather than hiding a residual row.
    assert _run_retry_in_fresh_process(recovered.index_db) == 0
    assert CursorStore(recovered.index_db).list_convergence_debt(limit=20) == []
    quiescent_status = convergence_debt_summary_info(recovered.index_db)
    assert quiescent_status.failed_count == 0
    assert quiescent_status.retry_due_count == 0
    assert (
        session_materialization_facts(
            recovered.index_db,
            session_id=recovered.target_session_id,
        )
        == recovered_terminal_facts
    )
    assert raw_authority_facts(recovered.source_db) == raw_facts_before

    # Independent terminal oracle: an uninterrupted quiet archive reaches the
    # same stable profile/receipt/event/thread facts and leaves the unrelated
    # session untouched.
    baseline = seed_partial_convergence_archive(tmp_path / "baseline", target_hot=False)
    baseline_states, _baseline_timings = DaemonConverger((make_insights_stage(baseline.index_db),)).converge_batch(
        (baseline.target_source,)
    )
    assert baseline_states[baseline.target_source].converged
    baseline_terminal_facts = session_materialization_facts(
        baseline.index_db,
        session_id=baseline.target_session_id,
    )
    assert baseline_terminal_facts == recovered_terminal_facts
    assert (
        session_materialization_facts(
            baseline.index_db,
            session_id=baseline.unrelated_session_id,
        ).profile
        is None
    )

    receipt = WorkloadReceipt.from_observations(
        spec=recovered.workload_spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:f654480cadb7cc4c194704e24dfd483199547b35",
        runtime_id="pytest:fresh-python-process-retries",
        archive_id="archive:testdiet-02-partial-convergence",
        generation_id="typed-archive-writer:testdiet-02",
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="generate"),
            WorkloadPhaseObservation(name="ingest"),
            WorkloadPhaseObservation(name="observe-debt", progress_completed=0, progress_total=1),
            WorkloadPhaseObservation(name="converge", progress_completed=1, progress_total=1),
            WorkloadPhaseObservation(name="query"),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=True, quiescent=True),
        ),
        evidence_refs=(
            f"ops-debt:{due_insights_debt.debt_id}",
            f"session:{recovered.target_session_id}",
        ),
        cleanup_complete=True,
    )
    assert receipt.spec.workload_id == "canary:partial-convergence-drain"
    assert receipt.status is WorkloadRunStatus.SUCCEEDED
    assert receipt.cleanup_complete is True
    assert receipt.phases[-1].quiescent is True
