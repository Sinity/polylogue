"""Evidence-bearing success semantics for managed pytest steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_TERMINAL_OUTCOMES = frozenset({"passed", "failed", "skipped", "xfailed", "xpassed", "error", "rerun"})
TERMINATION_REASON_ENV = "POLYLOGUE_PYTEST_TERMINATION_REASON"


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _tests(report: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...] | None:
    if report is None:
        return None
    raw = report.get("tests")
    if not isinstance(raw, list):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def evaluate_pytest_evidence(
    *,
    report: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    events: Iterable[Mapping[str, Any]],
    exit_code: int,
    termination_reason: str | None = None,
    collection_only: bool = False,
) -> dict[str, Any]:
    """Return the one ordinary pytest success predicate and its diagnosis.

    The predicate intentionally consumes independent artifacts. A zero process
    exit is only one input, never the success decision. ``ordinary_eligible``
    keeps collection-only evidence explicit without allowing it into a normal
    verification receipt.
    """
    selection = selection if isinstance(selection, Mapping) else {}
    summary = summary if isinstance(summary, Mapping) else {}
    event_rows = tuple(events)
    selected_count = _int(selection.get("selected_count"))
    report_tests = _tests(report)
    terminal_tests = {
        str(test.get("nodeid"))
        for test in report_tests or ()
        if isinstance(test.get("nodeid"), str) and test.get("nodeid") and test.get("outcome") in _TERMINAL_OUTCOMES
    }
    outcomes: dict[str, int] = {}
    for test in report_tests or ():
        outcome = test.get("outcome")
        if isinstance(outcome, str):
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

    collection_finished = any(event.get("event") == "collection_finished" for event in event_rows)
    terminal_count = len(terminal_tests)
    diagnosis = "pytest_passed"
    if termination_reason == "stall":
        diagnosis = "pytest_stall_terminated"
    elif termination_reason in {"sigterm", "operator_interrupt", "sigint"}:
        diagnosis = "pytest_interrupted"
    elif exit_code == 3:
        diagnosis = "pytest_worker_loss"
    elif report is None:
        diagnosis = "pytest_no_report"
    elif report_tests is None:
        diagnosis = "pytest_report_unreadable"
    elif selected_count is None or selected_count <= 0:
        diagnosis = "pytest_no_tests_selected"
    elif not collection_finished:
        diagnosis = "pytest_collection_incomplete"
    elif terminal_count != selected_count:
        diagnosis = "pytest_report_incomplete"
    elif summary.get("exitstatus") not in (None, exit_code):
        diagnosis = "pytest_summary_inconsistent"
    elif exit_code != 0:
        diagnosis = "pytest_failed"

    ordinary_eligible = diagnosis == "pytest_passed" and not collection_only
    if collection_only and diagnosis == "pytest_passed":
        diagnosis = "pytest_collection_only"
    return {
        "ok": diagnosis in {"pytest_passed", "pytest_collection_only"},
        "ordinary_eligible": ordinary_eligible,
        "diagnosis": diagnosis,
        "report_status": "present" if report is not None else "missing",
        "collection_status": "complete" if collection_finished else "incomplete",
        "selected_count": selected_count,
        "terminal_count": terminal_count,
        "outcomes": outcomes,
        "exit_code": exit_code,
        "termination_reason": termination_reason,
    }


__all__ = ["TERMINATION_REASON_ENV", "evaluate_pytest_evidence"]
