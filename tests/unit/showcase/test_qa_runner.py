"""Integration-focused tests for QA result composition and artifact writing."""

from __future__ import annotations

import json

from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.schemas.audit import AuditReport
from polylogue.showcase.exercises import Exercise, Validation
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_runner import QAResult, _save_qa_reports
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def _make_showcase_result(output_dir) -> ShowcaseResult:
    exercise = Exercise(
        "test-help",
        "structural",
        "Help output",
        ["--help"],
        Validation(stdout_contains=("polylogue",)),
    )
    result = ShowcaseResult(
        results=[
            ExerciseResult(
                exercise=exercise,
                passed=True,
                exit_code=0,
                output="Usage: polylogue\n",
                duration_ms=12.0,
            ),
        ],
        total_duration_ms=12.0,
        output_dir=output_dir,
    )
    return result


def test_save_qa_reports_writes_composed_session_artifacts(tmp_path):
    report_dir = tmp_path / "qa"
    qa_result = QAResult(
        audit_report=AuditReport(checks=[
            OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
        ]),
        showcase_result=_make_showcase_result(report_dir),
        invariant_results=[
            InvariantResult("json_valid", "test-help", OutcomeStatus.OK),
        ],
        report_dir=report_dir,
    )

    _save_qa_reports(qa_result, report_dir)

    qa_session = json.loads((report_dir / "qa-session.json").read_text())
    invariant_checks = json.loads((report_dir / "invariant-checks.json").read_text())

    assert qa_session["audit"]["status"] == "ok"
    assert qa_session["showcase"]["summary"]["passed"] == 1
    assert qa_session["invariants"]["summary"] == {"failed": 0, "passed": 1, "skipped": 0}
    assert invariant_checks == [
        {"exercise": "test-help", "invariant": "json_valid", "status": "ok"},
    ]
    assert (report_dir / "schema-audit.json").exists()
    assert (report_dir / "showcase-report.json").exists()
    assert (report_dir / "qa-session.md").exists()
