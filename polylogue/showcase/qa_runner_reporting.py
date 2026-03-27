"""Report persistence and summaries for QA sessions."""

from __future__ import annotations

from pathlib import Path

from polylogue.showcase.qa_runner_models import QAResult


def save_qa_reports(result: QAResult, report_dir: Path) -> None:
    """Save all QA artifacts to the report directory."""
    import json

    report_dir.mkdir(parents=True, exist_ok=True)

    if result.audit_report:
        (report_dir / "schema-audit.json").write_text(json.dumps(result.audit_report.to_json(), indent=2))
    elif result.audit_error:
        (report_dir / "schema-audit.json").write_text(json.dumps({"error": result.audit_error}, indent=2))

    if result.proof_report is not None:
        (report_dir / "artifact-proof.json").write_text(
            json.dumps(result.proof_report.to_dict(), indent=2, sort_keys=True)
        )
    elif result.proof_error is not None:
        (report_dir / "artifact-proof.json").write_text(
            json.dumps({"error": result.proof_error}, indent=2, sort_keys=True)
        )

    if result.roundtrip_proof_report is not None:
        (report_dir / "roundtrip-proof.json").write_text(
            json.dumps(result.roundtrip_proof_report.to_dict(), indent=2, sort_keys=True)
        )
    elif result.roundtrip_proof_error is not None:
        (report_dir / "roundtrip-proof.json").write_text(
            json.dumps({"error": result.roundtrip_proof_error}, indent=2, sort_keys=True)
        )

    from polylogue.showcase.qa_report import generate_qa_markdown, generate_qa_session
    from polylogue.showcase.report_files import save_reports

    if result.showcase_result:
        save_reports(result.showcase_result)

    qa_session = generate_qa_session(result)
    (report_dir / "qa-session.json").write_text(json.dumps(qa_session, indent=2, sort_keys=True))
    (report_dir / "invariant-checks.json").write_text(
        json.dumps(qa_session["invariants"]["checks"], indent=2, sort_keys=True)
    )
    (report_dir / "qa-session.md").write_text(generate_qa_markdown(result))


def format_qa_summary(result: QAResult) -> str:
    """Format a human-readable QA session summary."""
    from polylogue.showcase.qa_report import generate_qa_summary

    return generate_qa_summary(result)


__all__ = ["format_qa_summary", "save_qa_reports"]
