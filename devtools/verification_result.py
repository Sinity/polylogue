"""Bounded semantic results for AgentCTL-declared verification operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

RECEIPT_SCHEMA_VERSION: Final = 2
MAX_GATE_OUTCOMES: Final = 16
MAX_DIAGNOSTIC_PATHS: Final = 16
MAX_PUBLIC_STRING_LENGTH: Final = 160


def declared_verification_result(payload: Mapping[str, Any], *, operation: str) -> dict[str, Any]:
    """Project verifier semantics without duplicating AgentCTL job metadata."""
    selection = _mapping(payload.get("testmon_selection"))
    aggregate = _mapping(payload.get("pytest_aggregate"))
    missing_paths = _strings(selection.get("missing_executable_paths"))
    runtime_data_paths = _strings(selection.get("runtime_data_paths"))
    verification_scope = _string(payload.get("verification_scope"))
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": "polylogue.verification-result",
        "operation": operation,
        "scope": {
            "verification_scope": verification_scope,
            "selection_mode": _string(selection.get("selection_mode")),
        },
        "testmon_environment": {
            "environment_digest": _string(selection.get("environment_digest")),
            "state": _string(selection.get("state_status")),
            "reason": _string(selection.get("state_reason")),
        },
        "gate_outcomes": _gate_outcomes(payload.get("steps")),
        "pytest_outcomes": _pytest_outcomes(aggregate),
        "diagnostics": {
            "diagnosis": _string(payload.get("diagnosis")),
            "checkout_diagnosis": _string(payload.get("checkout_diagnosis")),
            "selection_widened": _selection_widened(selection),
            "missing_edges": _bounded_strings(missing_paths),
            "runtime_data_paths": _bounded_strings(runtime_data_paths),
            "failure_ledger": _mapping(payload.get("failure_ledger")),
        },
        "semantic_status": _semantic_status(payload, verification_scope),
        "exit_code": _integer(payload.get("exit_code")),
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: object) -> str | None:
    return value[:MAX_PUBLIC_STRING_LENGTH] if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list | tuple) else []


def _bounded_strings(values: list[str]) -> dict[str, Any]:
    return {
        "count": len(values),
        "items": [_string(value) for value in values[:MAX_DIAGNOSTIC_PATHS]],
        "truncated": len(values) > MAX_DIAGNOSTIC_PATHS,
    }


def _gate_outcomes(value: object) -> dict[str, Any]:
    steps = [step for step in value if isinstance(step, Mapping)] if isinstance(value, list) else []
    gates = [step for step in steps if not str(step.get("name", "")).startswith("pytest")]
    return {
        "count": len(gates),
        "items": [
            {
                "name": _string(step.get("name")),
                "status": "running"
                if step.get("status") == "running"
                else "passed"
                if step.get("exit") == 0
                else "failed",
                "exit_code": _integer(step.get("exit")),
                "diagnosis": _string(step.get("diagnosis")),
            }
            for step in gates[:MAX_GATE_OUTCOMES]
        ],
        "truncated": len(gates) > MAX_GATE_OUTCOMES,
    }


def _pytest_outcomes(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = _mapping(aggregate.get("outcomes"))
    bounded_outcomes = {
        key[:MAX_PUBLIC_STRING_LENGTH]: value
        for key, value in sorted(
            ((key, value) for key, value in outcomes.items() if isinstance(value, int) and not isinstance(value, bool)),
            key=lambda item: item[0],
        )[:MAX_GATE_OUTCOMES]
    }
    corpus = _mapping(aggregate.get("corpus"))
    outcomes: dict[str, Any] = {
        "present": bool(aggregate),
        "selection_mode": _string(aggregate.get("selection_mode")),
        "selected_count": _integer(aggregate.get("selected_union_count")),
        "terminal_count": _integer(aggregate.get("terminal_union_count")),
        "terminal_green": _boolean(aggregate.get("terminal_green")),
        "complete_corpus_covered": _boolean(aggregate.get("complete_corpus_covered")),
        "corpus_digest": _string(corpus.get("digest")),
        "outcomes": bounded_outcomes,
        "outcomes_truncated": len(outcomes) > MAX_GATE_OUTCOMES,
    }
    covered_by = aggregate.get("covered_by_run")
    if isinstance(covered_by, str) and covered_by:
        # A skipped complete run names the run whose coverage it inherits.
        outcomes["covered_by_run"] = covered_by[:MAX_PUBLIC_STRING_LENGTH]
    return outcomes


def _selection_widened(selection: Mapping[str, Any]) -> bool:
    return selection.get("selection_mode") in {"all", "bootstrap"}


def _semantic_status(payload: Mapping[str, Any], verification_scope: str | None) -> str:
    if payload.get("status") == "running":
        return "running"
    exit_code = _integer(payload.get("exit_code"))
    if exit_code == 0:
        return {
            "affected": "affected-passed",
            "non-test": "non-test-passed",
        }.get(verification_scope or "", "passed")
    return "interrupted" if exit_code == 130 else "failed"
