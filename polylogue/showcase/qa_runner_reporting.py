"""Report persistence and summaries for QA sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polylogue.showcase.qa_runner_models import QAResult


def save_qa_reports(result: QAResult, report_dir: Path) -> None:
    """Save all QA artifacts to the report directory."""
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

    from polylogue.showcase.qa_report import generate_qa_markdown
    from polylogue.showcase.qa_session_payload import build_qa_session_record
    from polylogue.showcase.report_files import save_reports
    from polylogue.showcase.showcase_report_payloads import build_showcase_session_record

    if result.showcase_result:
        save_reports(result.showcase_result)

    showcase_session = (
        build_showcase_session_record(
            result.showcase_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if result.showcase_result is not None
        else None
    )
    qa_session = build_qa_session_record(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        showcase_session=showcase_session,
    )
    qa_payload = qa_session.to_payload()
    (report_dir / "qa-session.json").write_text(json.dumps(qa_payload, indent=2, sort_keys=True))
    (report_dir / "invariant-checks.json").write_text(
        json.dumps([check.to_payload() for check in qa_session.invariants.checks], indent=2, sort_keys=True)
    )
    (report_dir / "qa-session.md").write_text(generate_qa_markdown(result))


def format_qa_summary(result: QAResult) -> str:
    """Format a human-readable QA session summary."""
    from polylogue.showcase.qa_report import generate_qa_summary

    return generate_qa_summary(result)


__all__ = ["format_qa_summary", "save_qa_reports"]
