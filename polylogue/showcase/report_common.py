"""Shared helpers for showcase and QA report rendering."""

from __future__ import annotations

from typing import Any

from polylogue.lib.outcomes import OutcomeStatus
from polylogue.showcase.invariants import InvariantResult


def serialize_invariant_result(result: InvariantResult) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "invariant": result.invariant_name,
        "exercise": result.exercise_name,
        "status": result.status.value,
    }
    if result.error:
        entry["error"] = result.error
    return entry


def summarize_invariants(results: list[InvariantResult]) -> dict[str, int]:
    return {
        "passed": sum(1 for result in results if result.status is OutcomeStatus.OK),
        "failed": sum(1 for result in results if result.status is OutcomeStatus.ERROR),
        "skipped": sum(1 for result in results if result.status is OutcomeStatus.SKIP),
    }


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
