# mypy: disable-error-code="arg-type,attr-defined,operator"

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from polylogue.cli.qa_capture import run_vhs_capture
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.qa_runner_reporting import format_qa_summary, save_qa_reports
from polylogue.showcase.qa_summary import generate_qa_summary
from polylogue.showcase.report_common import (
    format_count_mapping,
    format_semantic_metric_summary,
    serialize_invariant_result,
    status_label,
    summarize_invariants,
)


def _make_env() -> SimpleNamespace:
    return SimpleNamespace(ui=SimpleNamespace(console=MagicMock()))


def test_run_vhs_capture_returns_on_import_error(tmp_path: object) -> None:
    env = _make_env()
    showcase_result = SimpleNamespace(output_dir=tmp_path, results=[])
    real_import = __import__

    def _raising_import(
        name: str, globals: object = None, locals: object = None, fromlist: tuple[str, ...] = (), level: int = 0
    ) -> object:
        if name == "polylogue.showcase.vhs":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_raising_import):
        run_vhs_capture(env, showcase_result, json_output=False)

    env.ui.console.print.assert_not_called()


def test_run_vhs_capture_handles_missing_output_dir() -> None:
    env = _make_env()
    module = ModuleType("polylogue.showcase.vhs")
    module.check_vhs_available = lambda: True
    module.generate_all_tapes = lambda exercises, *, output_dir: ["demo"]
    module.run_vhs_capture = lambda tape_path, gif_path: True

    with patch.dict(sys.modules, {"polylogue.showcase.vhs": module}):
        run_vhs_capture(env, SimpleNamespace(output_dir=None, results=[]), json_output=False)

    env.ui.console.print.assert_not_called()


def test_run_vhs_capture_reports_tape_results_and_missing_binary(tmp_path: object) -> None:
    env = _make_env()
    showcase_result = SimpleNamespace(
        output_dir=tmp_path,
        results=[SimpleNamespace(exercise="ex-1"), SimpleNamespace(exercise="ex-2")],
    )
    module = ModuleType("polylogue.showcase.vhs")
    module.generate_all_tapes = lambda exercises, *, output_dir: ["demo", "broken"]
    capture_results = {"demo": True, "broken": False}
    module.run_vhs_capture = lambda tape_path, gif_path: capture_results[tape_path.stem]

    module.check_vhs_available = lambda: True
    with patch.dict(sys.modules, {"polylogue.showcase.vhs": module}):
        run_vhs_capture(env, showcase_result, json_output=False)

    assert env.ui.console.print.call_args_list[0].args == ("  VHS demo: ok",)
    assert env.ui.console.print.call_args_list[1].args == ("  VHS broken: FAILED",)

    env = _make_env()
    module.check_vhs_available = lambda: False
    with patch.dict(sys.modules, {"polylogue.showcase.vhs": module}):
        run_vhs_capture(env, showcase_result, json_output=False)

    assert "VHS binary not found" in env.ui.console.print.call_args.args[0]


def test_report_common_helpers_serialize_and_format() -> None:
    results = [
        InvariantResult("json_valid", "stats", OutcomeStatus.OK),
        InvariantResult("exit_code", "stats", OutcomeStatus.ERROR, error="boom"),
        InvariantResult("nonempty_output", "stats", OutcomeStatus.SKIP),
    ]

    assert serialize_invariant_result(results[1]) == {
        "invariant": "exit_code",
        "exercise": "stats",
        "status": "error",
        "error": "boom",
    }
    assert summarize_invariants(results) == {"passed": 1, "failed": 1, "skipped": 1}
    assert status_label(OutcomeStatus.OK) == "PASS"
    assert status_label(OutcomeStatus.WARNING) == "WARN"
    assert status_label(OutcomeStatus.ERROR) == "FAIL"
    assert status_label(OutcomeStatus.SKIP) == "SKIPPED"
    assert format_count_mapping({"b": 2, "a": 1}) == "a=1, b=2"
    assert (
        format_semantic_metric_summary(
            {
                "tokens": {"preserved": 2, "declared_loss": 1, "critical_loss": 0},
                "tools": {"preserved": 3},
            }
        )
        == "tokens(preserved=2, declared_loss=1, critical_loss=0), tools(preserved=3, declared_loss=0, critical_loss=0)"
    )


def test_generate_qa_summary_uses_provided_session_and_all_branches(tmp_path: object) -> None:
    result = QAResult(
        audit_report=SimpleNamespace(all_passed=True),
        proof_report=SimpleNamespace(is_clean=True),
        showcase_result=SimpleNamespace(failed=1),
        report_dir=tmp_path,
    )
    session = SimpleNamespace(
        proof=SimpleNamespace(
            report={
                "summary": {
                    "contract_backed_records": 3,
                    "unsupported_parseable_records": 1,
                    "recognized_non_parseable_records": 0,
                    "unknown_records": 0,
                    "decode_errors": 0,
                    "package_versions": {"v1": 2},
                    "element_kinds": {"conversation_document": 3},
                    "resolution_reasons": {"supported": 3},
                }
            }
        ),
        showcase=SimpleNamespace(
            summary=SimpleNamespace(passed=3, total=4, failed=1, skipped=0, total_duration_ms=1200.0)
        ),
        invariants=SimpleNamespace(skipped=False, summary=SimpleNamespace(passed=5, failed=1, skipped=2)),
    )

    summary = generate_qa_summary(result, session=session)

    assert "Schema Audit: PASS" in summary
    assert "Artifact Proof: contract_backed=3, unsupported=1, non_parseable=0, unknown=0, decode_errors=0" in summary
    assert "Packages: v1=2" in summary
    assert "Elements: conversation_document=3" in summary
    assert "Reasons: supported=3" in summary
    assert "Exercises: 3/4 passed, 1 failed, 0 skipped (1.2s)" in summary
    assert "Invariants: 5 pass, 1 fail, 2 skip" in summary
    assert "Overall: FAIL" in summary
    assert f"Reports: {tmp_path}" in summary


def test_generate_qa_summary_builds_session_when_missing_and_handles_skips() -> None:
    result = QAResult(
        audit_error="audit failed",
        proof_error="proof failed",
        exercises_skipped=True,
        invariants_skipped=True,
    )
    session = SimpleNamespace(
        proof=SimpleNamespace(report=None),
        showcase=SimpleNamespace(summary=None),
        invariants=SimpleNamespace(skipped=True, summary=SimpleNamespace(passed=0, failed=0, skipped=0)),
    )

    with (
        patch("polylogue.showcase.qa_session_payload.build_qa_session_record", return_value=session) as mock_session,
        patch("polylogue.showcase.showcase_report_payloads.build_showcase_session_record") as mock_showcase,
    ):
        summary = generate_qa_summary(result)

    mock_session.assert_called_once()
    mock_showcase.assert_not_called()
    assert "Schema Audit: FAIL" in summary
    assert "Artifact Proof: FAIL (proof failed)" in summary
    assert "Exercises: SKIPPED" in summary
    assert "Invariants: SKIPPED" in summary
    assert "Overall: FAIL" in summary


def test_save_qa_reports_writes_success_and_error_payloads(tmp_path: object) -> None:
    qa_session = SimpleNamespace(
        to_payload=lambda: {"status": "ok"},
        invariants=SimpleNamespace(checks=[SimpleNamespace(to_payload=lambda: {"name": "inv"})]),
    )
    result = QAResult(
        audit_report=SimpleNamespace(to_json=lambda: {"audit": "ok"}),
        proof_report=SimpleNamespace(to_dict=lambda: {"proof": "ok"}),
        showcase_result=SimpleNamespace(),
    )

    with (
        patch("polylogue.showcase.report_files.save_reports") as mock_save_reports,
        patch(
            "polylogue.showcase.showcase_report_payloads.build_showcase_session_record", return_value="showcase-session"
        ),
        patch("polylogue.showcase.qa_session_payload.build_qa_session_record", return_value=qa_session),
        patch("polylogue.showcase.qa_report.generate_qa_markdown", return_value="# QA report"),
    ):
        save_qa_reports(result, tmp_path)

    assert json.loads((tmp_path / "schema-audit.json").read_text()) == {"audit": "ok"}
    assert json.loads((tmp_path / "artifact-proof.json").read_text()) == {"proof": "ok"}
    assert json.loads((tmp_path / "qa-session.json").read_text()) == {"status": "ok"}
    assert json.loads((tmp_path / "invariant-checks.json").read_text()) == [{"name": "inv"}]
    assert (tmp_path / "qa-session.md").read_text() == "# QA report"
    mock_save_reports.assert_called_once_with(result.showcase_result)

    error_dir = tmp_path / "errors"
    error_result = QAResult(audit_error="audit failed", proof_error="proof failed")
    qa_session = SimpleNamespace(
        to_payload=lambda: {"status": "error"},
        invariants=SimpleNamespace(checks=[]),
    )

    with (
        patch("polylogue.showcase.report_files.save_reports") as mock_save_reports,
        patch("polylogue.showcase.qa_session_payload.build_qa_session_record", return_value=qa_session),
        patch("polylogue.showcase.qa_report.generate_qa_markdown", return_value="# errors"),
    ):
        save_qa_reports(error_result, error_dir)

    assert json.loads((error_dir / "schema-audit.json").read_text()) == {"error": "audit failed"}
    assert json.loads((error_dir / "artifact-proof.json").read_text()) == {"error": "proof failed"}
    mock_save_reports.assert_not_called()


def test_format_qa_summary_delegates_to_renderer() -> None:
    result = QAResult()

    with patch("polylogue.showcase.qa_report.generate_qa_summary", return_value="summary") as mock_generate:
        assert format_qa_summary(result) == "summary"

    mock_generate.assert_called_once_with(result)
