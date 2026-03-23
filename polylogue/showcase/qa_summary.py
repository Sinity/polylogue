"""Human-readable QA summary rendering."""

from __future__ import annotations

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.report_common import (
    format_count_mapping,
    format_semantic_metric_summary,
    status_label,
)


def generate_qa_summary(result, *, session: dict | None = None) -> str:
    """Generate a human-readable summary for a full QA run."""
    from polylogue.showcase.qa_session_payload import generate_qa_session

    if session is None:
        session = generate_qa_session(result)
    lines: list[str] = []
    proof_summary = session["proof"].get("report", {}).get("summary")
    semantic_summary = session["semantic_proof"].get("report", {}).get("summary")

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
    if result.semantic_proof_error:
        lines.append(f"Semantic Proof: FAIL ({result.semantic_proof_error})")
    elif semantic_summary is not None:
        lines.append(
            "Semantic Proof: "
            f"surfaces={semantic_summary['surface_count']}, "
            f"clean_surfaces={semantic_summary['clean_surfaces']}, "
            f"critical_surfaces={semantic_summary['critical_surfaces']}, "
            f"total_conversations={semantic_summary['total_conversations']}, "
            f"preserved_checks={semantic_summary['preserved_checks']}, "
            f"declared_loss_checks={semantic_summary['declared_loss_checks']}, "
            f"critical_loss_checks={semantic_summary['critical_loss_checks']}"
        )
        if semantic_summary["metric_summary"]:
            lines.append(
                f"  Metrics: {format_semantic_metric_summary(semantic_summary['metric_summary'])}"
            )
        for surface, surface_report in sorted(
            session["semantic_proof"].get("report", {}).get("surfaces", {}).items()
        ):
            surface_summary = surface_report["summary"]
            lines.append(
                f"  {surface}: clean={surface_summary['clean_conversations']}, "
                f"critical={surface_summary['critical_conversations']}, "
                f"preserved_checks={surface_summary['preserved_checks']}, "
                f"declared_loss_checks={surface_summary['declared_loss_checks']}, "
                f"critical_loss_checks={surface_summary['critical_loss_checks']}"
            )
    elif result.semantic_proof_status is OutcomeStatus.SKIP:
        lines.append("Semantic Proof: SKIPPED")
    roundtrip_summary = session["roundtrip_proof"].get("report", {}).get("summary")
    if result.roundtrip_proof_error:
        lines.append(f"Roundtrip Proof: FAIL ({result.roundtrip_proof_error})")
    elif roundtrip_summary is not None:
        lines.append(
            "Roundtrip Proof: "
            f"providers={roundtrip_summary['provider_count']}, "
            f"clean={roundtrip_summary['clean_providers']}, "
            f"failed={roundtrip_summary['failed_providers']}, "
            f"artifacts={roundtrip_summary['artifact_count']}, "
            f"parsed_conversations={roundtrip_summary['parsed_conversations']}, "
            f"persisted_conversations={roundtrip_summary['persisted_conversations']}"
        )
        for provider, provider_report in sorted(
            session["roundtrip_proof"].get("report", {}).get("providers", {}).items()
        ):
            provider_summary = provider_report["summary"]
            failed_stages = ", ".join(provider_summary["failed_stages"]) or "-"
            lines.append(
                f"  {provider}: package={provider_report['package_version']}, "
                f"element={provider_report['element_kind'] or '-'}, "
                f"failed_stages={failed_stages}"
            )
    elif result.roundtrip_proof_status is OutcomeStatus.SKIP:
        lines.append("Roundtrip Proof: SKIPPED")
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
