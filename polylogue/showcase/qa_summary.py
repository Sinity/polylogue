"""Human-readable QA summary rendering."""

from __future__ import annotations

from polylogue.authored_payloads import payload_count_mapping, require_payload_mapping
from polylogue.showcase.qa_runner_models import QAResult
from polylogue.showcase.report_common import (
    format_count_mapping,
    status_label,
)
from polylogue.showcase.report_models import QASessionRecord


def generate_qa_summary(result: QAResult, *, session: QASessionRecord | None = None) -> str:
    """Generate a human-readable summary for a full QA run."""
    from datetime import datetime, timezone

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
    lines: list[str] = []
    proof_report = session.proof.report
    proof_summary = (
        require_payload_mapping(proof_report["summary"], context="proof.summary") if proof_report is not None else None
    )

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
        package_versions = payload_count_mapping(
            proof_summary.get("package_versions", {}),
            context="proof.summary.package_versions",
        )
        element_kinds = payload_count_mapping(
            proof_summary.get("element_kinds", {}),
            context="proof.summary.element_kinds",
        )
        resolution_reasons = payload_count_mapping(
            proof_summary.get("resolution_reasons", {}),
            context="proof.summary.resolution_reasons",
        )
        if package_versions:
            lines.append(f"  Packages: {format_count_mapping(package_versions)}")
        if element_kinds:
            lines.append(f"  Elements: {format_count_mapping(element_kinds)}")
        if resolution_reasons:
            lines.append(f"  Reasons: {format_count_mapping(resolution_reasons)}")

    if result.exercises_skipped:
        lines.append("Exercises: SKIPPED")
    elif session.showcase.summary is not None:
        showcase_summary = session.showcase.summary
        lines.append(
            "Exercises: "
            f"{showcase_summary.passed}/{showcase_summary.total} passed, "
            f"{showcase_summary.failed} failed, "
            f"{showcase_summary.skipped} skipped "
            f"({showcase_summary.total_duration_ms / 1000:.1f}s)"
        )

    if session.invariants.skipped:
        lines.append("Invariants: SKIPPED")
    else:
        invariant_summary = session.invariants.summary
        lines.append(
            "Invariants: "
            f"{invariant_summary.passed} pass, "
            f"{invariant_summary.failed} fail, "
            f"{invariant_summary.skipped} skip"
        )

    lines.append("")
    lines.append(f"Overall: {status_label(result.overall_status)}")
    if result.report_dir:
        lines.append(f"Reports: {result.report_dir}")
    return "\n".join(lines)


__all__ = ["generate_qa_summary"]
