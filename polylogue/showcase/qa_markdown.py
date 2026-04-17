"""Markdown QA session rendering."""

from __future__ import annotations

from typing import Any

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.report_common import status_label


def generate_qa_markdown(
    result: QAResult,
    *,
    session: dict[str, Any] | None = None,
    git_sha: str | None = None,
) -> str:
    """Generate a stable, diffable Markdown report for a full QA run."""
    from polylogue.showcase.qa_session_payload import generate_qa_session

    if session is None:
        session = generate_qa_session(result)
    lines: list[str] = ["# QA Session", ""]
    if git_sha:
        lines.append(f"**Git SHA**: `{git_sha}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Status |")
    lines.append("| --- | --- |")
    lines.append(f"| Schema Audit | {status_label(result.audit_status)} |")
    lines.append(f"| Artifact Proof | {status_label(result.proof_status)} |")
    lines.append(f"| Exercises | {status_label(result.showcase_status)} |")
    lines.append(f"| Invariants | {status_label(result.invariant_status)} |")
    lines.append(f"| Overall | {status_label(result.overall_status)} |")
    lines.append("")

    if result.audit_report is not None:
        audit_summary = result.audit_report.to_json()["summary"]
        lines.append("## Schema Audit")
        lines.append("")
        lines.append(f"- Passed: {audit_summary['passed']}")
        lines.append(f"- Warned: {audit_summary['warned']}")
        lines.append(f"- Failed: {audit_summary['failed']}")
        lines.append("")
    elif result.audit_error:
        lines.append("## Schema Audit")
        lines.append("")
        lines.append(f"- Error: {result.audit_error}")
        lines.append("")

    proof_report = session["proof"].get("report")
    if proof_report is not None:
        proof_summary = proof_report["summary"]
        lines.extend(
            [
                "## Artifact Proof",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Total raw records | {proof_report['total_records']} |",
                f"| Contract-backed | {proof_summary['contract_backed_records']} |",
                f"| Unsupported parseable | {proof_summary['unsupported_parseable_records']} |",
                f"| Recognized non-parseable | {proof_summary['recognized_non_parseable_records']} |",
                f"| Unknown | {proof_summary['unknown_records']} |",
                f"| Decode errors | {proof_summary['decode_errors']} |",
                f"| Linked sidecars | {proof_summary['linked_sidecars']} |",
                f"| Orphan sidecars | {proof_summary['orphan_sidecars']} |",
                "",
            ]
        )
        if proof_summary["package_versions"]:
            lines.extend(["### Resolved Packages", "", "| Package | Count |", "| --- | ---: |"])
            for version, count in proof_summary["package_versions"].items():
                lines.append(f"| {version} | {count} |")
            lines.append("")
        if proof_summary["element_kinds"]:
            lines.extend(["### Resolved Elements", "", "| Element kind | Count |", "| --- | ---: |"])
            for element_kind, count in proof_summary["element_kinds"].items():
                lines.append(f"| {element_kind} | {count} |")
            lines.append("")
        if proof_summary["resolution_reasons"]:
            lines.extend(["### Resolution Reasons", "", "| Reason | Count |", "| --- | ---: |"])
            for reason, count in proof_summary["resolution_reasons"].items():
                lines.append(f"| {reason} | {count} |")
            lines.append("")
        if proof_report["providers"]:
            lines.extend(
                [
                    "### Providers",
                    "",
                    "| Provider | Total | Contract-backed | Unsupported | Non-parseable | Unknown | Decode errors |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for provider, stats in proof_report["providers"].items():
                lines.append(
                    f"| {provider} | {stats['total_records']} | {stats['contract_backed_records']} | "
                    f"{stats['unsupported_parseable_records']} | {stats['recognized_non_parseable_records']} | "
                    f"{stats['unknown_records']} | {stats['decode_errors']} |"
                )
            lines.append("")
    elif result.proof_error:
        lines.extend(["## Artifact Proof", "", f"- Error: {result.proof_error}", ""])

    showcase_summary = session["showcase"]["summary"]
    if showcase_summary is not None:
        lines.extend(
            [
                "## Exercises",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Total | {showcase_summary['total']} |",
                f"| Passed | {showcase_summary['passed']} |",
                f"| Failed | {showcase_summary['failed']} |",
                f"| Skipped | {showcase_summary['skipped']} |",
                "",
                "### Results by Group",
                "",
                "| Group | Pass | Fail | Skip |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for group in GROUPS:
            counts = session["showcase"]["group_counts"].get(group, {"pass": 0, "fail": 0, "skip": 0})
            lines.append(f"| {group} | {counts['pass']} | {counts['fail']} | {counts['skip']} |")
        lines.append("")

    invariant_summary = session["invariants"]["summary"]
    if not session["invariants"]["skipped"]:
        lines.extend(
            [
                "## Invariants",
                "",
                f"- Passed: {invariant_summary['passed']}",
                f"- Failed: {invariant_summary['failed']}",
                f"- Skipped: {invariant_summary['skipped']}",
            ]
        )
        failures = [check for check in session["invariants"]["checks"] if check["status"] == OutcomeStatus.ERROR.value]
        if failures:
            lines.extend(["", "### Failures", ""])
            for failure in failures:
                lines.append(f"- `{failure['invariant']}` @ `{failure['exercise']}`: {failure.get('error', '')}")
        lines.append("")

    return "\n".join(lines)


__all__ = ["generate_qa_markdown"]
