"""Projection contracts for declared verification results."""

from __future__ import annotations

from devtools.verification_result import declared_verification_result


def test_projection_uses_the_finalized_step_and_aggregate_counts() -> None:
    result = declared_verification_result(
        {
            "exit_code": 1,
            "status": "failed",
            "verification_scope": "affected",
            "pytest_aggregate": {
                "selection_mode": "focused",
                "selected_union_count": 3,
                "terminal_union_count": 2,
                "terminal_green": False,
                "outcomes": {"failed": 1, "passed": 1},
            },
            "steps": [
                {
                    "name": "pytest focused",
                    "exit": 1,
                    "process_exit": 0,
                    "diagnosis": "pytest_report_incomplete",
                }
            ],
        },
        operation="verify",
    )

    assert result["semantic_status"] == "failed"
    assert result["pytest_outcomes"] == {
        "present": True,
        "selection_mode": "focused",
        "selected_count": 3,
        "terminal_count": 2,
        "terminal_green": False,
        "complete_corpus_covered": None,
        "corpus_digest": None,
        "outcomes": {"failed": 1, "passed": 1},
        "outcomes_truncated": False,
    }


def test_projection_does_not_invent_missing_pytest_aggregate_fields() -> None:
    result = declared_verification_result({"exit_code": 0, "status": "success"}, operation="verify")

    assert "ordinary_eligible" not in result["pytest_outcomes"]
    assert "diagnosis" not in result["pytest_outcomes"]


def test_projection_preserves_affected_admission_refusal_details() -> None:
    result = declared_verification_result(
        {
            "exit_code": 2,
            "status": "failed",
            "verification_scope": "affected",
            "testmon_selection": {
                "selection_mode": "affected",
                "admission": {
                    "status": "refused",
                    "selected_count": 1001,
                    "estimated_seconds": 900.0,
                    "reason": "exceeds cap",
                    "next_boundary": "devtools verify --all at the explicit master/corpus boundary",
                    "budget": {"max_selected_tests": 1000},
                },
            },
        },
        operation="verify",
    )

    assert result["admission"]["status"] == "refused"
    assert result["admission"]["selected_count"] == 1001
    assert result["admission"]["reason"] == "exceeds cap"
    assert "verify --all" in result["admission"]["next_boundary"]
