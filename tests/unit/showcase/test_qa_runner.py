"""Integration-focused tests for QA result composition and artifact writing."""

from __future__ import annotations

import json

from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.rendering.semantic_proof import SemanticProofReport, SemanticProofSuiteReport
from polylogue.schemas.audit import AuditReport
from polylogue.schemas.roundtrip_proof import (
    ProviderRoundtripProofReport,
    RoundtripProofSuiteReport,
    RoundtripStageReport,
)
from polylogue.schemas.verification_models import ArtifactProofReport, ProviderArtifactProof
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
        semantic_proof_report=SemanticProofSuiteReport(
            surface_reports={
                "canonical_markdown_v1": SemanticProofReport(
                    surface="canonical_markdown_v1",
                    conversations=[],
                    provider_reports={},
                )
            },
        ),
        roundtrip_proof_report=RoundtripProofSuiteReport(
            provider_reports={
                "chatgpt": ProviderRoundtripProofReport(
                    provider="chatgpt",
                    package_version="v1",
                    element_kind="conversation_document",
                    wire_encoding="json",
                    stages={
                        "selection": RoundtripStageReport("selection", "ok", "selected"),
                        "synthetic": RoundtripStageReport("synthetic", "ok", "generated", {"generated_artifacts": 1}),
                        "acquisition": RoundtripStageReport("acquisition", "ok", "acquired"),
                        "validation": RoundtripStageReport("validation", "ok", "validated"),
                        "parse_dispatch": RoundtripStageReport("parse_dispatch", "ok", "parsed", {"parsed_conversations": 1}),
                        "prepare_persist": RoundtripStageReport("prepare_persist", "ok", "persisted", {"persisted_conversations": 1}),
                        "corpus_verification": RoundtripStageReport("corpus_verification", "ok", "verified"),
                        "artifact_proof": RoundtripStageReport("artifact_proof", "ok", "proof"),
                    },
                )
            },
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
    semantic_payload = json.loads((report_dir / "semantic-proof.json").read_text())
    roundtrip_payload = json.loads((report_dir / "roundtrip-proof.json").read_text())
    invariant_checks = json.loads((report_dir / "invariant-checks.json").read_text())

    assert qa_session["audit"]["status"] == "ok"
    assert qa_session["proof"]["status"] == "ok"
    assert qa_session["semantic_proof"]["status"] == "ok"
    assert qa_session["roundtrip_proof"]["status"] == "ok"
    assert qa_session["showcase"]["summary"]["passed"] == 1
    assert qa_session["invariants"]["summary"] == {"failed": 0, "passed": 1, "skipped": 0}
    assert proof_payload["summary"]["contract_backed_records"] == 1
    assert proof_payload["summary"]["package_versions"] == {"v1": 1}
    assert proof_payload["summary"]["element_kinds"] == {"conversation_document": 1}
    assert semantic_payload["summary"]["clean"] is True
    assert semantic_payload["summary"]["surface_count"] == 1
    assert roundtrip_payload["summary"]["clean"] is True
    assert roundtrip_payload["summary"]["provider_count"] == 1
    assert invariant_checks == [
        {"exercise": "test-help", "invariant": "json_valid", "status": "ok"},
    ]
    assert (report_dir / "artifact-proof.json").exists()
    assert (report_dir / "semantic-proof.json").exists()
    assert (report_dir / "roundtrip-proof.json").exists()
    assert (report_dir / "schema-audit.json").exists()
    assert (report_dir / "showcase-report.json").exists()
    assert (report_dir / "qa-session.md").exists()
