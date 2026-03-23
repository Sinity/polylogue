"""QA-specific report rendering built on showcase session payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.exercises import GROUPS

from .report_common import (
    format_count_mapping,
    format_semantic_metric_summary,
    serialize_invariant_result,
    status_label,
    summarize_invariants,
)

if TYPE_CHECKING:
    from polylogue.showcase.qa_runner import QAResult


def generate_qa_session(result: QAResult) -> dict[str, Any]:
    """Generate a structured full QA session record."""
    from polylogue.showcase.showcase_report import generate_showcase_session

    showcase_session = (
        generate_showcase_session(result.showcase_result)
        if result.showcase_result is not None
        else None
    )
    return build_qa_session_payload(
        result,
        timestamp=datetime.now(timezone.utc).isoformat(),
        showcase_session=showcase_session,
    )


def build_qa_session_payload(
    result: QAResult,
    *,
    timestamp: str,
    showcase_session: dict[str, Any] | None,
) -> dict[str, Any]:
    audit_payload: dict[str, Any] = {
        "status": result.audit_status.value,
        "skipped": result.audit_skipped,
    }
    if result.audit_report is not None:
        audit_payload["report"] = result.audit_report.to_json()
    if result.audit_error is not None:
        audit_payload["error"] = result.audit_error

    invariant_checks = [
        serialize_invariant_result(invariant_result)
        for invariant_result in result.invariant_results
    ]
    invariant_summary = summarize_invariants(result.invariant_results)

    proof_payload: dict[str, Any] = {
        "status": result.proof_status.value,
    }
    if result.proof_report is not None:
        proof_payload["report"] = result.proof_report.to_dict()
    if result.proof_error is not None:
        proof_payload["error"] = result.proof_error

    semantic_proof_payload: dict[str, Any] = {
        "status": result.semantic_proof_status.value,
    }
    if result.semantic_proof_report is not None:
        semantic_proof_payload["report"] = result.semantic_proof_report.to_dict()
    if result.semantic_proof_error is not None:
        semantic_proof_payload["error"] = result.semantic_proof_error

    roundtrip_proof_payload: dict[str, Any] = {
        "status": result.roundtrip_proof_status.value,
    }
    if result.roundtrip_proof_report is not None:
        roundtrip_proof_payload["report"] = result.roundtrip_proof_report.to_dict()
    if result.roundtrip_proof_error is not None:
        roundtrip_proof_payload["error"] = result.roundtrip_proof_error

    showcase_payload: dict[str, Any] = {
        "status": result.showcase_status.value,
        "skipped": result.exercises_skipped,
        "summary": showcase_session["summary"] if showcase_session else None,
        "group_counts": showcase_session["group_counts"] if showcase_session else {},
        "exercises": showcase_session["exercises"] if showcase_session else [],
    }

    return {
        "schema_version": 1,
        "timestamp": timestamp,
        "audit": audit_payload,
        "proof": proof_payload,
        "semantic_proof": semantic_proof_payload,
        "roundtrip_proof": roundtrip_proof_payload,
        "showcase": showcase_payload,
        "invariants": {
            "status": result.invariant_status.value,
            "skipped": result.invariants_skipped or result.showcase_result is None,
            "summary": invariant_summary,
            "checks": invariant_checks,
        },
        "overall_status": result.overall_status.value,
        "overall_passed": result.all_passed,
        "report_dir": str(result.report_dir) if result.report_dir else None,
    }


def generate_qa_summary(result: QAResult, *, session: dict[str, Any] | None = None) -> str:
    """Generate a human-readable summary for a full QA run."""
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


def generate_qa_markdown(
    result: QAResult,
    *,
    session: dict[str, Any] | None = None,
    git_sha: str | None = None,
) -> str:
    """Generate a stable, diffable Markdown report for a full QA run."""
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
    lines.append(f"| Semantic Proof | {status_label(result.semantic_proof_status)} |")
    lines.append(f"| Roundtrip Proof | {status_label(result.roundtrip_proof_status)} |")
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
        lines.append("## Artifact Proof")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Total raw records | {proof_report['total_records']} |")
        lines.append(f"| Contract-backed | {proof_summary['contract_backed_records']} |")
        lines.append(
            f"| Unsupported parseable | {proof_summary['unsupported_parseable_records']} |"
        )
        lines.append(
            f"| Recognized non-parseable | {proof_summary['recognized_non_parseable_records']} |"
        )
        lines.append(f"| Unknown | {proof_summary['unknown_records']} |")
        lines.append(f"| Decode errors | {proof_summary['decode_errors']} |")
        lines.append(f"| Linked sidecars | {proof_summary['linked_sidecars']} |")
        lines.append(f"| Orphan sidecars | {proof_summary['orphan_sidecars']} |")
        lines.append("")
        if proof_summary["package_versions"]:
            lines.append("### Resolved Packages")
            lines.append("")
            lines.append("| Package | Count |")
            lines.append("| --- | ---: |")
            for version, count in proof_summary["package_versions"].items():
                lines.append(f"| {version} | {count} |")
            lines.append("")
        if proof_summary["element_kinds"]:
            lines.append("### Resolved Elements")
            lines.append("")
            lines.append("| Element kind | Count |")
            lines.append("| --- | ---: |")
            for element_kind, count in proof_summary["element_kinds"].items():
                lines.append(f"| {element_kind} | {count} |")
            lines.append("")
        if proof_summary["resolution_reasons"]:
            lines.append("### Resolution Reasons")
            lines.append("")
            lines.append("| Reason | Count |")
            lines.append("| --- | ---: |")
            for reason, count in proof_summary["resolution_reasons"].items():
                lines.append(f"| {reason} | {count} |")
            lines.append("")
        if proof_report["providers"]:
            lines.append("### Providers")
            lines.append("")
            lines.append(
                "| Provider | Total | Contract-backed | Unsupported | Non-parseable | Unknown | Decode errors |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for provider, stats in proof_report["providers"].items():
                lines.append(
                    f"| {provider} | {stats['total_records']} | {stats['contract_backed_records']} | "
                    f"{stats['unsupported_parseable_records']} | {stats['recognized_non_parseable_records']} | "
                    f"{stats['unknown_records']} | {stats['decode_errors']} |"
                )
            lines.append("")
    elif result.proof_error:
        lines.append("## Artifact Proof")
        lines.append("")
        lines.append(f"- Error: {result.proof_error}")
        lines.append("")

    semantic_proof_report = session["semantic_proof"].get("report")
    if semantic_proof_report is not None:
        semantic_summary = semantic_proof_report["summary"]
        lines.append("## Semantic Proof")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Surface count | {semantic_summary['surface_count']} |")
        lines.append(f"| Clean surfaces | {semantic_summary['clean_surfaces']} |")
        lines.append(f"| Critical surfaces | {semantic_summary['critical_surfaces']} |")
        lines.append(
            f"| Total conversations | {semantic_summary['total_conversations']} |"
        )
        lines.append(f"| Preserved checks | {semantic_summary['preserved_checks']} |")
        lines.append(
            f"| Declared loss checks | {semantic_summary['declared_loss_checks']} |"
        )
        lines.append(
            f"| Critical loss checks | {semantic_summary['critical_loss_checks']} |"
        )
        lines.append("")
        if semantic_summary["metric_summary"]:
            lines.append("### Semantic Metrics")
            lines.append("")
            lines.append("| Metric | Preserved | Declared loss | Critical loss |")
            lines.append("| --- | ---: | ---: | ---: |")
            for metric, counts in semantic_summary["metric_summary"].items():
                lines.append(
                    f"| {metric} | {counts['preserved']} | {counts['declared_loss']} | {counts['critical_loss']} |"
                )
            lines.append("")
        if semantic_proof_report["surfaces"]:
            lines.append("### Surfaces")
            lines.append("")
            lines.append(
                "| Surface | Conversations | Clean | Critical | Preserved checks | Declared loss | Critical loss |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for surface, surface_report in semantic_proof_report["surfaces"].items():
                surface_summary = surface_report["summary"]
                lines.append(
                    f"| {surface} | {surface_summary['total_conversations']} | "
                    f"{surface_summary['clean_conversations']} | {surface_summary['critical_conversations']} | "
                    f"{surface_summary['preserved_checks']} | {surface_summary['declared_loss_checks']} | "
                    f"{surface_summary['critical_loss_checks']} |"
                )
            lines.append("")
            lines.append("### Surface Providers")
            lines.append("")
            lines.append(
                "| Surface | Provider | Conversations | Clean | Critical | Preserved checks | Declared loss | Critical loss |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for surface, surface_report in semantic_proof_report["surfaces"].items():
                for provider, stats in surface_report["providers"].items():
                    lines.append(
                        f"| {surface} | {provider} | {stats['total_conversations']} | {stats['clean_conversations']} | "
                        f"{stats['critical_conversations']} | {stats['preserved_checks']} | "
                        f"{stats['declared_loss_checks']} | {stats['critical_loss_checks']} |"
                    )
            lines.append("")
    elif result.semantic_proof_error:
        lines.append("## Semantic Proof")
        lines.append("")
        lines.append(f"- Error: {result.semantic_proof_error}")
        lines.append("")

    roundtrip_report = session["roundtrip_proof"].get("report")
    if roundtrip_report is not None:
        roundtrip_summary = roundtrip_report["summary"]
        lines.append("## Roundtrip Proof")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Providers | {roundtrip_summary['provider_count']} |")
        lines.append(f"| Clean providers | {roundtrip_summary['clean_providers']} |")
        lines.append(f"| Failed providers | {roundtrip_summary['failed_providers']} |")
        lines.append(f"| Synthetic artifacts | {roundtrip_summary['artifact_count']} |")
        lines.append(f"| Parsed conversations | {roundtrip_summary['parsed_conversations']} |")
        lines.append(f"| Persisted conversations | {roundtrip_summary['persisted_conversations']} |")
        lines.append("")
        lines.append("### Providers")
        lines.append("")
        lines.append("| Provider | Package | Element | Failed stages |")
        lines.append("| --- | --- | --- | --- |")
        for provider, provider_report in sorted(roundtrip_report["providers"].items()):
            provider_summary = provider_report["summary"]
            failed_stages = ", ".join(provider_summary["failed_stages"]) or "-"
            lines.append(
                f"| {provider} | {provider_report['package_version']} | "
                f"{provider_report['element_kind'] or '-'} | {failed_stages} |"
            )
        lines.append("")
    elif result.roundtrip_proof_error:
        lines.append("## Roundtrip Proof")
        lines.append("")
        lines.append(f"- Error: {result.roundtrip_proof_error}")
        lines.append("")

    showcase_summary = session["showcase"]["summary"]
    if showcase_summary is not None:
        lines.append("## Exercises")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Total | {showcase_summary['total']} |")
        lines.append(f"| Passed | {showcase_summary['passed']} |")
        lines.append(f"| Failed | {showcase_summary['failed']} |")
        lines.append(f"| Skipped | {showcase_summary['skipped']} |")
        lines.append("")
        lines.append("### Results by Group")
        lines.append("")
        lines.append("| Group | Pass | Fail | Skip |")
        lines.append("| --- | ---: | ---: | ---: |")
        for group in GROUPS:
            counts = session["showcase"]["group_counts"].get(
                group, {"pass": 0, "fail": 0, "skip": 0}
            )
            lines.append(
                f"| {group} | {counts['pass']} | {counts['fail']} | {counts['skip']} |"
            )
        lines.append("")

    invariant_summary = session["invariants"]["summary"]
    if not session["invariants"]["skipped"]:
        lines.append("## Invariants")
        lines.append("")
        lines.append(f"- Passed: {invariant_summary['passed']}")
        lines.append(f"- Failed: {invariant_summary['failed']}")
        lines.append(f"- Skipped: {invariant_summary['skipped']}")
        failures = [
            check
            for check in session["invariants"]["checks"]
            if check["status"] == OutcomeStatus.ERROR.value
        ]
        if failures:
            lines.append("")
            lines.append("### Failures")
            lines.append("")
            for failure in failures:
                lines.append(
                    f"- `{failure['invariant']}` @ `{failure['exercise']}`: {failure.get('error', '')}"
                )
        lines.append("")

    return "\n".join(lines)
