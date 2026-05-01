"""Shared helpers for showcase and QA report rendering."""

from __future__ import annotations

from polylogue.core.outcomes import OutcomeStatus
from polylogue.showcase.invariants import InvariantResult
from polylogue.showcase.report_models import InvariantCheckRecord, InvariantSummary


def serialize_invariant_result(result: InvariantResult) -> dict[str, object]:
    return InvariantCheckRecord.from_result(result).to_payload()


def summarize_invariants(results: list[InvariantResult]) -> dict[str, int]:
    return InvariantSummary.from_results(results).to_payload()


def status_label(status: OutcomeStatus) -> str:
    return {
        OutcomeStatus.OK: "PASS",
        OutcomeStatus.WARNING: "WARN",
        OutcomeStatus.ERROR: "FAIL",
        OutcomeStatus.SKIP: "SKIPPED",
    }[status]


def format_count_mapping(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def format_semantic_metric_summary(metric_summary: dict[str, dict[str, int]]) -> str:
    parts = []
    for metric, counts in sorted(metric_summary.items()):
        parts.append(
            f"{metric}(preserved={counts.get('preserved', 0)}, "
            f"declared_loss={counts.get('declared_loss', 0)}, "
            f"critical_loss={counts.get('critical_loss', 0)})"
        )
    return ", ".join(parts)
