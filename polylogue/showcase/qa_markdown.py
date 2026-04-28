"""Markdown QA session rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.products.authored_payloads import require_payload_mapping
from polylogue.showcase.exercises import GROUPS
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.report_common import status_label
from polylogue.showcase.report_models import QASessionRecord


def generate_qa_markdown(
    result: QAResult,
    *,
    session: QASessionRecord | None = None,
    git_sha: str | None = None,
) -> str:
    """Generate a stable, diffable Markdown report for a full QA run."""
    from polylogue.showcase.qa_session_payload import build_qa_session_record
    from polylogue.showcase.showcase_report_payloads import build_showcase_session_record

    if session is None:
        showcase_session = (
            build_showcase_session_record(
                result.showcase_result,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            if result.showcase_result is not None
            else None
        )
        session = build_qa_session_record(
            result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            showcase_session=showcase_session,
        )
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
        audit_summary = require_payload_mapping(
            result.audit_report.to_json().get("summary", {}), context="audit.summary"
        )
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

    proof_report = session.proof.report
    if proof_report is not None:
        proof_summary = require_payload_mapping(proof_report["summary"], context="proof.summary")
        package_versions = require_payload_mapping(
            proof_summary.get("package_versions", {}),
            context="proof.summary.package_versions",
        )
        element_kinds = require_payload_mapping(
            proof_summary.get("element_kinds", {}),
            context="proof.summary.element_kinds",
        )
        resolution_reasons = require_payload_mapping(
            proof_summary.get("resolution_reasons", {}),
            context="proof.summary.resolution_reasons",
        )
        providers = {
            provider: require_payload_mapping(stats, context=f"proof.providers.{provider}")
            for provider, stats in require_payload_mapping(
                proof_report.get("providers", {}),
                context="proof.providers",
            ).items()
        }
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
        if package_versions:
            lines.extend(["### Resolved Packages", "", "| Package | Count |", "| --- | ---: |"])
            for version, count in package_versions.items():
                lines.append(f"| {version} | {count} |")
            lines.append("")
        if element_kinds:
            lines.extend(["### Resolved Elements", "", "| Element kind | Count |", "| --- | ---: |"])
            for element_kind, count in element_kinds.items():
                lines.append(f"| {element_kind} | {count} |")
            lines.append("")
        if resolution_reasons:
            lines.extend(["### Resolution Reasons", "", "| Reason | Count |", "| --- | ---: |"])
            for reason, count in resolution_reasons.items():
                lines.append(f"| {reason} | {count} |")
            lines.append("")
        if providers:
            lines.extend(
                [
                    "### Providers",
                    "",
                    "| Provider | Total | Contract-backed | Unsupported | Non-parseable | Unknown | Decode errors |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for provider, stats in providers.items():
                lines.append(
                    f"| {provider} | {stats['total_records']} | {stats['contract_backed_records']} | "
                    f"{stats['unsupported_parseable_records']} | {stats['recognized_non_parseable_records']} | "
                    f"{stats['unknown_records']} | {stats['decode_errors']} |"
                )
            lines.append("")
    elif result.proof_error:
        lines.extend(["## Artifact Proof", "", f"- Error: {result.proof_error}", ""])

    showcase_summary = session.showcase.summary
    if showcase_summary is not None:
        lines.extend(
            [
                "## Exercises",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Total | {showcase_summary.total} |",
                f"| Passed | {showcase_summary.passed} |",
                f"| Failed | {showcase_summary.failed} |",
                f"| Skipped | {showcase_summary.skipped} |",
                "",
                "### Results by Group",
                "",
                "| Group | Pass | Fail | Skip |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for group in GROUPS:
            counts = session.showcase.group_counts.get(group)
            pass_count = counts.passed if counts is not None else 0
            fail_count = counts.failed if counts is not None else 0
            skip_count = counts.skipped if counts is not None else 0
            lines.append(f"| {group} | {pass_count} | {fail_count} | {skip_count} |")
        lines.append("")

    invariant_summary = session.invariants.summary
    if not session.invariants.skipped:
        lines.extend(
            [
                "## Invariants",
                "",
                f"- Passed: {invariant_summary.passed}",
                f"- Failed: {invariant_summary.failed}",
                f"- Skipped: {invariant_summary.skipped}",
            ]
        )
        failures = [check for check in session.invariants.checks if check.status == OutcomeStatus.ERROR.value]
        if failures:
            lines.extend(["", "### Failures", ""])
            for failure in failures:
                lines.append(f"- `{failure.invariant}` @ `{failure.exercise}`: {failure.error or ''}")
        lines.append("")

    return "\n".join(lines)


__all__ = ["generate_qa_markdown"]
