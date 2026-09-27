"""Contract checks for the bounded direct-intake benchmark receipts."""

from __future__ import annotations

import pytest

from tests.infra.convergence_probe_contract import intake_measurement


def _measurement(*, stored_messages: int = 2, stage_summary: dict[str, int] | None = None) -> dict[str, int]:
    return intake_measurement(
        expected_files=1,
        expected_sessions=1,
        expected_messages=2,
        succeeded_files=1,
        failed_files=0,
        skipped_files=0,
        excluded_files=0,
        deferred_files=0,
        refused_bytes=0,
        stored_sessions=1,
        stored_messages=stored_messages,
        stage_summary=stage_summary or {"total": 1, "converged": 1, "failed": 0, "in_progress": 0},
    )


def test_stage_failures_and_zero_convergence_are_never_substituted_for_intake() -> None:
    with pytest.raises(ValueError, match="stage_converged_files': 0.*stage_failed_files': 3"):
        intake_measurement(
            expected_files=3,
            expected_sessions=3,
            expected_messages=6,
            succeeded_files=3,
            failed_files=0,
            skipped_files=0,
            excluded_files=0,
            deferred_files=0,
            refused_bytes=0,
            stored_sessions=3,
            stored_messages=6,
            stage_summary={"total": 3, "converged": 0, "failed": 3, "in_progress": 0},
        )


def test_missing_stored_message_rejects_complete_work_rate() -> None:
    with pytest.raises(ValueError, match="stored_messages': 1"):
        _measurement(stored_messages=1)


def test_pending_stage_work_rejects_complete_intake() -> None:
    with pytest.raises(ValueError, match="stage_converged_files': 0.*stage_in_progress_files': 3"):
        intake_measurement(
            expected_files=3,
            expected_sessions=3,
            expected_messages=6,
            succeeded_files=3,
            failed_files=0,
            skipped_files=0,
            excluded_files=0,
            deferred_files=0,
            refused_bytes=0,
            stored_sessions=3,
            stored_messages=6,
            stage_summary={"total": 3, "converged": 0, "failed": 0, "in_progress": 3},
        )


def test_complete_tiny_intake_reports_separate_observed_counts() -> None:
    # Successful file states are evicted from the converger after the batch.
    result = _measurement(stage_summary={"total": 0, "converged": 0, "failed": 0, "in_progress": 0})

    assert result["intake_succeeded_files"] == 1
    assert result["stored_sessions"] == 1
    assert result["stored_messages"] == 2
    assert result["stage_converged_files"] == 0
    assert result["stage_failed_files"] == 0
    assert result["stage_in_progress_files"] == 0
