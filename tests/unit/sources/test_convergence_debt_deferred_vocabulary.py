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

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.daemon.convergence import ConvergenceStage, DaemonConverger, FileState, StageState
from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info
from polylogue.daemon.status import daemon_status_payload, format_daemon_status_lines
from polylogue.sources.live.convergence_debt import (
    ConvergenceDebt,
    convergence_debt_from_state,
    convergence_debt_from_states,
    is_deferred_stage_state,
)
from polylogue.sources.live.convergence_debt_retry import convergence_debt_source_path
from polylogue.sources.live.convergence_outcome import record_convergence_outcome
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_convergence_debt_lookups_follow_the_active_index_generation(tmp_path: Path) -> None:
    """Outcome and retry lookups ignore a stale conventional index database."""

    source_db = tmp_path / "source.db"
    shadow_index = tmp_path / "index.db"
    active_index = tmp_path / "generations" / "active" / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(shadow_index, ArchiveTier.INDEX)
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    source_path = tmp_path / "active.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('raw-active', 'codex-session', 'active', ?, 0, ?, 1, 1)
            """,
            (str(source_path), bytes(32)),
        )
        conn.commit()
    with sqlite3.connect(active_index) as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('active', 'codex-session', 'raw-active', 'active', ?)
            """,
            (bytes(32),),
        )
        conn.commit()
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")
    cursor = CursorStore(tmp_path / "ops.db")
    debt = ConvergenceDebt(path=source_path, stage="fts", error="deferred", deferred=True)

    record_convergence_outcome(cursor, source_path, (debt,), archive_root=tmp_path)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        session_debts = conn.execute(
            "SELECT target_id FROM convergence_debt WHERE target_type = 'session_id'"
        ).fetchall()
        assert (
            convergence_debt_source_path(
                conn,
                subject_type="session_id",
                subject_id="codex-session:active",
                archive_root=tmp_path,
            )
            == source_path
        )

    assert session_debts == [("codex-session:active",)]


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


def test_record_convergence_outcome_persists_deferred_and_failed_statuses(tmp_path: Path) -> None:
    """The post-ingest route must preserve classification in the ops ledger.

    This exercises the production bridge from ``FileState`` classification
    through ``record_convergence_outcome`` and ``CursorStore``. A test that
    only checks the dataclass flag would pass even if the writer silently
    converted every row back to ``status = 'failed'``.
    """
    deferred_path = tmp_path / "deferred.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    deferred_debts = convergence_debt_from_state(
        deferred_path,
        FileState(
            path=deferred_path,
            stages={"fts": StageState.PENDING},
            last_error="stage fts returned False",
        ),
    )
    failed_debts = convergence_debt_from_state(
        failed_path,
        FileState(
            path=failed_path,
            stages={"fts": StageState.FAILED},
            last_error="fts writer failed",
        ),
    )
    cursor = CursorStore(tmp_path / "live.sqlite")

    record_convergence_outcome(cursor, deferred_path, deferred_debts)
    record_convergence_outcome(cursor, failed_path, failed_debts)

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = dict(
            conn.execute(
                """
                SELECT target_id, status
                FROM convergence_debt
                WHERE target_type = 'source_path'
                """
            ).fetchall()
        )
    assert rows == {
        str(deferred_path): "deferred",
        str(failed_path): "failed",
    }


def test_real_converger_outcomes_reach_status_and_read_surfaces(tmp_path: Path) -> None:
    """The real converger route preserves pending and failed stage outcomes.

    This is deliberately a red twin for the production handoff: a stage that
    returns False with ``false_means_pending`` must remain deferred all the way
    through the ops ledger and status projections, while an exception remains
    failed. A test of ``StageState`` or ``ConvergenceDebt.deferred`` alone
    would miss either writer or read-surface regressions.
    """
    deferred_path = tmp_path / "deferred.jsonl"
    failed_path = tmp_path / "failed.jsonl"

    def execute(path: Path) -> bool:
        if path == deferred_path:
            return False
        raise RuntimeError("fts writer failed")

    converger = DaemonConverger(
        (
            ConvergenceStage(
                name="fts",
                description="bounded FTS convergence",
                check=lambda _path: True,
                execute=execute,
                false_means_pending=True,
            ),
        )
    )
    states, _timings = converger.converge_batch((deferred_path, failed_path))
    assert states[deferred_path].stages == {"fts": StageState.PENDING}
    assert states[failed_path].stages == {"fts": StageState.FAILED}

    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    record_convergence_outcome(
        cursor,
        deferred_path,
        convergence_debt_from_states((deferred_path,), states),
    )
    record_convergence_outcome(
        cursor,
        failed_path,
        convergence_debt_from_states((failed_path,), states),
    )

    summary = convergence_debt_summary_info(index_db)
    assert summary.failed_count == 1
    assert summary.deferred_count == 1
    assert {item.status for item in summary.recent} == {"failed", "deferred"}

    with (
        patch("polylogue.daemon.status.archive_root", return_value=tmp_path),
        patch("polylogue.daemon.status._active_status_db_path", return_value=index_db),
        patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
        patch("polylogue.daemon.status._blob_size_info", return_value=0),
        patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
        patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
    ):
        payload = daemon_status_payload(sources=())

    convergence = payload["convergence"]
    assert isinstance(convergence, dict)
    assert convergence["failed_count"] == 1
    assert convergence["deferred_count"] == 1
    lines = format_daemon_status_lines(payload)
    assert "Convergence debt: 1 failed, 1 deferred, 0 retry due" in lines


@pytest.mark.parametrize("initial_deferred,next_deferred", [(False, True), (True, False)])
def test_convergence_debt_status_transition_preserves_attempts_and_deadline(
    tmp_path: Path, initial_deferred: bool, next_deferred: bool
) -> None:
    """A classification-only change preserves retry scheduling in both directions."""
    index_db = tmp_path / "index.db"
    cursor = CursorStore(index_db)
    subject_id = str(tmp_path / "source.jsonl")
    error = "stage fts returned False"
    cursor.record_convergence_debt(
        stage="fts",
        subject_type="source_path",
        subject_id=subject_id,
        error=error,
        deferred=initial_deferred,
    )

    exact_deadline = "9999-01-01T00:00:00+00:00"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            """
            UPDATE convergence_debt
            SET next_retry_at = ?
            WHERE stage = 'fts' AND target_type = 'source_path' AND target_id = ?
            """,
            (exact_deadline, subject_id),
        )
        conn.commit()

    cursor.record_convergence_debt(
        stage="fts",
        subject_type="source_path",
        subject_id=subject_id,
        error=error,
        deferred=next_deferred,
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT status, attempts, next_retry_at
            FROM convergence_debt
            WHERE stage = 'fts' AND target_type = 'source_path' AND target_id = ?
            """,
            (subject_id,),
        ).fetchone()
    assert row == ("deferred" if next_deferred else "failed", 1, exact_deadline)
