"""Structured QA session payload builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from polylogue.showcase.report_common import (
    serialize_invariant_result,
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

    proof_payload: dict[str, Any] = {"status": result.proof_status.value}
    if result.proof_report is not None:
        proof_payload["report"] = result.proof_report.to_dict()
    if result.proof_error is not None:
        proof_payload["error"] = result.proof_error

    semantic_proof_payload: dict[str, Any] = {"status": result.semantic_proof_status.value}
    if result.semantic_proof_report is not None:
        semantic_proof_payload["report"] = result.semantic_proof_report.to_dict()
    if result.semantic_proof_error is not None:
        semantic_proof_payload["error"] = result.semantic_proof_error

    roundtrip_proof_payload: dict[str, Any] = {"status": result.roundtrip_proof_status.value}
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


__all__ = ["build_qa_session_payload", "generate_qa_session"]
