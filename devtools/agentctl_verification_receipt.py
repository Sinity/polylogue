"""Bounded public verification results for declared AgentCTL operations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

AGENTCTL_OPERATION_ENV: Final = "SINNIXD_OPERATION"
AGENTCTL_CHECKOUT_ID_ENV: Final = "SINNIXD_CHECKOUT_ID"
AGENTCTL_CHECKOUT_HEAD_ENV: Final = "SINNIXD_CHECKOUT_HEAD"
AGENTCTL_JOB_ID_ENV: Final = "SINNIXD_JOB_ID"
SUPPORTED_OPERATIONS: Final = frozenset({"verify_affected", "verify_quick", "verify_all"})
RECEIPT_SCHEMA_VERSION: Final = 1
MAX_GATE_OUTCOMES: Final = 16
MAX_DIAGNOSTIC_PATHS: Final = 16
MAX_PUBLIC_STRING_LENGTH: Final = 160


def agentctl_verification_operation(env: Mapping[str, str] | None = None) -> str | None:
    """Return the declared verification operation, never a caller-selected name."""
    operation = (os.environ if env is None else env).get(AGENTCTL_OPERATION_ENV)
    return operation if operation in SUPPORTED_OPERATIONS else None


def agentctl_verification_receipt(
    payload: Mapping[str, Any], *, env: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """Project one verifier ledger into the stable AgentCTL result contract.

    The full verifier ledger remains checkout-local forensic evidence. This
    result intentionally excludes commands, output, exception text, and paths
    outside the declared repository so AgentCTL can retain and expose it.
    """
    values = os.environ if env is None else env
    operation = agentctl_verification_operation(values)
    if operation is None:
        return None

    selection = _mapping(payload.get("testmon_selection"))
    aggregate = _mapping(payload.get("pytest_aggregate"))
    missing_paths = _strings(selection.get("missing_executable_paths"))
    runtime_data_paths = _strings(selection.get("runtime_data_paths"))
    verification_scope = _string(payload.get("verification_scope"))
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": "polylogue.verification-result",
        "operation": operation,
        "job_id": _string(values.get(AGENTCTL_JOB_ID_ENV)),
        "run_id": _string(payload.get("run_id")),
        "workspace": {
            "checkout_id": _string(values.get(AGENTCTL_CHECKOUT_ID_ENV)),
            "declared_head": _string(values.get(AGENTCTL_CHECKOUT_HEAD_ENV)),
            "observed_head": _string(payload.get("git_head")),
            "final_head": _string(payload.get("final_git_head")),
            "worktree_fingerprint": _string(payload.get("worktree_fingerprint")),
            "final_worktree_fingerprint": _string(payload.get("final_worktree_fingerprint")),
        },
        "scope": {
            "verification_scope": verification_scope,
            "selection_mode": _string(selection.get("selection_mode")),
            "release_baseline_allowed": _boolean(payload.get("release_baseline_allowed")),
        },
        "testmon_environment": {
            "environment_digest": _string(selection.get("environment_digest")),
            "graph_content_digest": None,
            "graph_content_digest_available": False,
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
        },
        "semantic_status": _semantic_status(payload, verification_scope),
        "exit_code": _integer(payload.get("exit_code")),
    }


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value[:MAX_PUBLIC_STRING_LENGTH]


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
    items = [
        {
            "name": _string(step.get("name")),
            "status": _step_status(step),
            "exit_code": _integer(step.get("exit")),
            "diagnosis": _string(step.get("diagnosis")),
        }
        for step in gates[:MAX_GATE_OUTCOMES]
    ]
    return {"count": len(gates), "items": items, "truncated": len(gates) > MAX_GATE_OUTCOMES}


def _step_status(step: Mapping[str, Any]) -> str:
    if step.get("status") == "running":
        return "running"
    return "passed" if step.get("exit") == 0 else "failed"


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
    return {
        "present": bool(aggregate),
        "selection_mode": _string(aggregate.get("selection_mode")),
        "selected_count": _integer(aggregate.get("selected_union_count")),
        "terminal_count": _integer(aggregate.get("terminal_union_count")),
        "non_green_count": _integer(aggregate.get("non_green_count")),
        "terminal_green": _boolean(aggregate.get("terminal_green")),
        "complete_corpus_covered": _boolean(aggregate.get("complete_corpus_covered")),
        "corpus_digest": _string(corpus.get("digest")),
        "outcomes": bounded_outcomes,
        "outcomes_truncated": len(outcomes) > MAX_GATE_OUTCOMES,
    }


def _selection_widened(selection: Mapping[str, Any]) -> bool:
    return selection.get("selection_mode") in {"all", "bootstrap"}


def _semantic_status(payload: Mapping[str, Any], verification_scope: str | None) -> str:
    if payload.get("status") == "running":
        return "running"
    exit_code = _integer(payload.get("exit_code"))
    if exit_code == 0:
        if verification_scope == "release-baseline":
            return "release-baseline-passed"
        if verification_scope == "affected":
            return "affected-passed"
        if verification_scope == "non-test":
            return "non-test-passed"
        return "passed"
    if exit_code == 130:
        return "interrupted"
    return "failed"


__all__ = [
    "AGENTCTL_CHECKOUT_HEAD_ENV",
    "AGENTCTL_CHECKOUT_ID_ENV",
    "AGENTCTL_JOB_ID_ENV",
    "AGENTCTL_OPERATION_ENV",
    "RECEIPT_SCHEMA_VERSION",
    "SUPPORTED_OPERATIONS",
    "agentctl_verification_operation",
    "agentctl_verification_receipt",
]
