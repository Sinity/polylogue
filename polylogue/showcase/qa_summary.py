"""Human-readable QA summary rendering."""

from __future__ import annotations

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.report_common import (
    format_count_mapping,
    status_label,
)


def generate_qa_summary(result, *, session: dict | None = None) -> str:
    """Generate a human-readable summary for a full QA run."""
    from polylogue.showcase.qa_session_payload import generate_qa_session

    if session is None:
        session = generate_qa_session(result)
    lines: list[str] = []
    proof_summary = session["proof"].get("report", {}).get("summary")

    lines.append(f"Schema Audit: {status_label(result.audit_status)}")
    if result.proof_error:
        lines.append(f"Artifact Proof: FAIL ({result.proof_error})")
    elif proof_summary is not None:
        lines.append(
            "Artifact Proof: "
            f"contract_backed={proof_summary['contract_backed_records']}, "
            f"unsupported={proof_summary['unsupported_parseable_records']}, "
            f"non_parseable={proof_summary['recognized_non_parseable_records']}, "
            f"unknown={proof_summary['unknown_records']}, "
            f"decode_errors={proof_summary['decode_errors']}"
        )
        if proof_summary["package_versions"]:
            lines.append(
                f"  Packages: {format_count_mapping(proof_summary['package_versions'])}"
            )
        if proof_summary["element_kinds"]:
            lines.append(
                f"  Elements: {format_count_mapping(proof_summary['element_kinds'])}"
            )
        if proof_summary["resolution_reasons"]:
            lines.append(
                f"  Reasons: {format_count_mapping(proof_summary['resolution_reasons'])}"
            )
    if result.audit_status is OutcomeStatus.ERROR:
        if result.audit_error:
            lines.append(f"  Error: {result.audit_error}")
        elif result.audit_report is not None:
            summary = result.audit_report.to_json()["summary"]
            lines.append(
                f"  Checks: {summary['passed']} pass, {summary['warned']} warn, {summary['failed']} fail"
            )
        lines.append("Overall: FAIL")
        if result.report_dir:
            lines.append(f"Reports: {result.report_dir}")
        return "\n".join(lines)

    showcase_summary = session["showcase"]["summary"]
    if result.exercises_skipped:
        lines.append("Exercises: SKIPPED")
    elif showcase_summary is not None:
        lines.append(
            "Exercises: "
            f"{showcase_summary['passed']}/{showcase_summary['total']} passed, "
            f"{showcase_summary['failed']} failed, {showcase_summary['skipped']} skipped "
            f"({showcase_summary['total_duration_ms'] / 1000:.1f}s)"
        )

    invariant_summary = session["invariants"]["summary"]
    if session["invariants"]["skipped"]:
        lines.append("Invariants: SKIPPED")
    else:
        lines.append(
            "Invariants: "
            f"{invariant_summary['passed']} pass, "
            f"{invariant_summary['failed']} fail, "
            f"{invariant_summary['skipped']} skip"
        )

    lines.append("")
    lines.append(f"Overall: {status_label(result.overall_status)}")
    if result.report_dir:
        lines.append(f"Reports: {result.report_dir}")
    return "\n".join(lines)


__all__ = ["generate_qa_summary"]
