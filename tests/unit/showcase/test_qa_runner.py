"""Integration-focused tests for QA result composition and artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.scenarios import AssertionSpec, polylogue_execution
from polylogue.schemas.audit_models import AuditReport
from polylogue.schemas.verification_models import ArtifactProofReport, ProviderArtifactProof
from polylogue.showcase.exercises import Exercise
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_runner import QAResult, _save_qa_reports
from polylogue.showcase.runner import ExerciseResult, ShowcaseResult


def _make_showcase_result(output_dir: Path) -> ShowcaseResult:
    exercise = Exercise(
        name="test-help",
        group="structural",
        description="Help output",
        execution=polylogue_execution("--help"),
        assertion=AssertionSpec(stdout_contains=("polylogue",)),
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


def test_save_qa_reports_writes_composed_session_artifacts(tmp_path: Path) -> None:
    report_dir = tmp_path / "qa"
    qa_result = QAResult(
        audit_report=AuditReport(
            checks=[
                OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
            ]
        ),
        proof_report=ArtifactProofReport(
            providers={
                "chatgpt": ProviderArtifactProof(
                    provider="chatgpt",
                    total_records=1,
                    contract_backed_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"conversation_document": 1},
                    resolution_reasons={"exact_structure": 1},
                )
            },
            total_records=1,
        ),
        showcase_result=_make_showcase_result(report_dir),
        invariant_results=[
            InvariantResult("json_valid", "test-help", OutcomeStatus.OK),
        ],
        report_dir=report_dir,
    )

    _save_qa_reports(qa_result, report_dir)

    qa_session = json.loads((report_dir / "qa-session.json").read_text())
    proof_payload = json.loads((report_dir / "artifact-proof.json").read_text())
    invariant_checks = json.loads((report_dir / "invariant-checks.json").read_text())

    assert qa_session["audit"]["status"] == "ok"
    assert qa_session["proof"]["status"] == "ok"
    assert qa_session["showcase"]["summary"]["passed"] == 1
    assert qa_session["invariants"]["summary"] == {"failed": 0, "passed": 1, "skipped": 0}
    assert proof_payload["summary"]["contract_backed_records"] == 1
    assert proof_payload["summary"]["package_versions"] == {"v1": 1}
    assert proof_payload["summary"]["element_kinds"] == {"conversation_document": 1}
    assert invariant_checks == [
        {"exercise": "test-help", "invariant": "json_valid", "status": "ok"},
    ]
    assert (report_dir / "artifact-proof.json").exists()
    assert (report_dir / "schema-audit.json").exists()
    assert (report_dir / "showcase-report.json").exists()
    assert (report_dir / "qa-session.md").exists()


def test_qa_result_marks_skipped_proof_as_non_failing(tmp_path: Path) -> None:
    report_dir = tmp_path / "qa"
    qa_result = QAResult(
        audit_report=AuditReport(
            checks=[
                OutcomeCheck(name="privacy", status=OutcomeStatus.OK, summary="ok"),
            ]
        ),
        proof_skipped=True,
        showcase_result=_make_showcase_result(report_dir),
        invariant_results=[
            InvariantResult("json_valid", "test-help", OutcomeStatus.OK),
        ],
        report_dir=report_dir,
    )

    _save_qa_reports(qa_result, report_dir)

    qa_session = json.loads((report_dir / "qa-session.json").read_text())
    assert qa_result.all_passed is True
    assert qa_session["proof"]["status"] == "skip"
    assert qa_session["proof"]["skipped"] is True
