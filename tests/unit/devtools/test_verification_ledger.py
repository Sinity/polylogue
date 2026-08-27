"""Contract tests for evidence-backed verification-failure classification."""

from datetime import UTC, datetime, timedelta

import pytest

from devtools.verification_ledger import (
    classify_failure,
    ledger_records,
    policy_diagnostics,
    validate_disposition,
)


def _receipt(*, head: str = "head", diagnosis: str | None = None, exit_code: int = 1) -> dict[str, object]:
    step: dict[str, object] = {"name": "test_example", "exit": exit_code, "duration_s": 1.5}
    if diagnosis:
        step["diagnosis"] = diagnosis
    return {
        "run_id": f"run-{head}",
        "status": "failed",
        "tier": "focused-test",
        "git_head": head,
        "git_dirty": False,
        "finished_at": "2026-08-26T00:00:00+00:00",
        "steps": [step],
    }


def test_classification_preserves_distinct_evidence_axes() -> None:
    current = _receipt()
    assert classify_failure(current, "test_example", history=[_receipt(head="older")]) == "deterministic_regression"
    assert (
        classify_failure(
            current, "test_example", history=[{**_receipt(), "steps": [{"name": "test_example", "exit": 0}]}]
        )
        == "same_head_variance"
    )
    assert (
        classify_failure(_receipt(diagnosis="checkout_import_mismatch", exit_code=125), "test_example")
        == "environment_contamination"
    )
    assert classify_failure(_receipt(diagnosis="pytest_timeout", exit_code=124), "test_example") == "timeout_resource"
    assert classify_failure(_receipt(), "test_example") == "unknown"


def test_ledger_record_retains_identity_environment_dependencies_and_artifacts() -> None:
    receipt = _receipt()
    receipt["environment_fingerprint"] = {
        "checkout_root": "/worktree",
        "python_executable": "/worktree/.venv/bin/python",
    }
    receipt["testmon_selection"] = {"selection_mode": "affected", "environment_digest": "env-1"}
    receipt["artifact_dir"] = ".cache/verify/runs/run-a"
    record = ledger_records(receipt)[0]
    for key in (
        "failure_id",
        "git_head",
        "git_dirty",
        "environment_fingerprint",
        "dependency_fingerprint",
        "first_seen",
        "last_seen",
        "runtime_s",
        "artifact_refs",
        "classification_confidence",
        "disposition",
        "expiry",
    ):
        assert key in record


def test_exception_requires_authority_bead_scope_and_future_expiry() -> None:
    with pytest.raises(ValueError):
        validate_disposition({"disposition": "baseline"})
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    validate_disposition(
        {
            "disposition": "baseline",
            "authority": "release-owner",
            "owner_bead": "polylogue-x",
            "scope": "test_example",
            "expiry": future,
        }
    )
    diagnostics = policy_diagnostics(
        [
            {
                "failure_id": "expired",
                "disposition": "baseline",
                "authority": "owner",
                "owner_bead": "x",
                "scope": "all",
                "expiry": "2020-01-01T00:00:00+00:00",
            }
        ]
    )
    assert diagnostics["policy_red"] is True
