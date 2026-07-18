"""Executable fault injections for continuity replay anti-vacuity tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from devtools.continuity_replay import ArgumentMutator, DiscoveryMutator, ResponseMutator
from polylogue.core.json import JSONDocument, require_json_document

_INCIDENT_EXPRESSION = 'messages where text:parallel-child AND text:"workflow_run:wf_synthetic_841"'
_PRIOR_ART_MARKER = "prior-art-zebra-417"
_CURRICULUM_MARKER = "incident-curriculum:wf_synthetic_841"


@dataclass(frozen=True, slots=True)
class ContinuityMutation:
    """Named mutation and the diagnostic class it must provoke."""

    name: str
    scenario: str
    expected_kind: str
    expected_failure_class: str
    argument_mutator: ArgumentMutator | None = None
    response_mutator: ResponseMutator | None = None
    discovery_mutator: DiscoveryMutator | None = None


def continuity_mutation(name: str) -> ContinuityMutation:
    """Build a fresh mutation; stateful response faults must not be reused."""

    factories = {
        "lost-request-state-continuation": _lost_request_state_continuation,
        "capped-pseudo-total": _capped_pseudo_total,
        "identical-call-topology-replay": _identical_call_topology_replay,
        "hidden-fact-or-grammar-discovery": _hidden_grammar_discovery,
        "missing-source-coverage": _missing_source_coverage,
        "unreasonable-query-classification": _unreasonable_query_classification,
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise KeyError(f"unknown continuity mutation {name!r}") from exc


def continuity_mutation_names() -> tuple[str, ...]:
    return (
        "lost-request-state-continuation",
        "capped-pseudo-total",
        "identical-call-topology-replay",
        "hidden-fact-or-grammar-discovery",
        "missing-source-coverage",
        "unreasonable-query-classification",
    )


def _lost_request_state_continuation() -> ContinuityMutation:
    def mutate(tool: str, arguments: dict[str, object], invocation: int) -> dict[str, object]:
        del invocation
        expression = arguments.get("expression")
        if tool == "query_units" and expression == _INCIDENT_EXPRESSION and "continuation" in arguments:
            return {"expression": expression, "offset": 0, "limit": 17}
        return arguments

    return ContinuityMutation(
        name="lost-request-state-continuation",
        scenario="parallel-claude-incident",
        expected_kind="pagination_offset_mismatch",
        expected_failure_class="execution",
        argument_mutator=mutate,
    )


def _capped_pseudo_total() -> ContinuityMutation:
    mutated = False

    def mutate(tool: str, arguments: dict[str, object], invocation: int, response_text: str) -> str:
        nonlocal mutated
        del invocation
        if mutated or tool != "query_units" or arguments.get("expression") != _INCIDENT_EXPRESSION:
            return response_text
        payload = require_json_document(json.loads(response_text), context="capped pseudo-total response")
        payload["continuation"] = None
        payload["next_offset"] = None
        mutated = True
        return json.dumps(payload, sort_keys=True)

    return ContinuityMutation(
        name="capped-pseudo-total",
        scenario="parallel-claude-incident",
        expected_kind="pagination_count_mismatch",
        expected_failure_class="execution",
        response_mutator=mutate,
    )


def _identical_call_topology_replay() -> ContinuityMutation:
    first_response: str | None = None
    target_calls = 0

    def mutate(tool: str, arguments: dict[str, object], invocation: int, response_text: str) -> str:
        nonlocal first_response, target_calls
        del invocation
        if tool != "query_units" or arguments.get("expression") != _INCIDENT_EXPRESSION:
            return response_text
        target_calls += 1
        if target_calls == 1:
            first_response = response_text
            return response_text
        if target_calls == 2 and first_response is not None:
            replayed = require_json_document(json.loads(first_response), context="replayed first topology page")
            actual = require_json_document(json.loads(response_text), context="actual second topology page")
            for field in ("offset", "next_offset", "continuation", "query_ref", "result_ref"):
                replayed[field] = actual.get(field)
            return json.dumps(replayed, sort_keys=True)
        return response_text

    return ContinuityMutation(
        name="identical-call-topology-replay",
        scenario="parallel-claude-incident",
        expected_kind="duplicate_pagination_identity",
        expected_failure_class="execution",
        response_mutator=mutate,
    )


def _hidden_grammar_discovery() -> ContinuityMutation:
    def mutate(discovery: JSONDocument) -> JSONDocument:
        changed = deepcopy(discovery)
        tools = changed.get("tools")
        if not isinstance(tools, dict):
            return changed
        query_units = tools.get("query_units")
        if not isinstance(query_units, dict):
            return changed
        schema = query_units.get("input_schema")
        if not isinstance(schema, dict):
            return changed
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("continuation", None)
        return changed

    return ContinuityMutation(
        name="hidden-fact-or-grammar-discovery",
        scenario="parallel-claude-incident",
        expected_kind="missing_discovered_arguments",
        expected_failure_class="discovery",
        discovery_mutator=mutate,
    )


def _missing_source_coverage() -> ContinuityMutation:
    def mutate(tool: str, arguments: dict[str, object], invocation: int, response_text: str) -> str:
        del invocation
        expression = arguments.get("expression")
        if tool != "query_units" or not isinstance(expression, str) or _PRIOR_ART_MARKER not in expression:
            return response_text
        payload = require_json_document(json.loads(response_text), context="missing source coverage response")
        if expression.endswith(" | count"):
            items = payload.get("items")
            if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                items[0]["count"] = 0
        else:
            payload["items"] = []
            payload["total"] = 0
            payload["continuation"] = None
            payload["next_offset"] = None
        return json.dumps(payload, sort_keys=True)

    return ContinuityMutation(
        name="missing-source-coverage",
        scenario="prior-art",
        expected_kind="non_single_projection",
        expected_failure_class="source_coverage",
        response_mutator=mutate,
    )


def _unreasonable_query_classification() -> ContinuityMutation:
    def mutate(tool: str, arguments: dict[str, object], invocation: int, response_text: str) -> str:
        del invocation
        expression = arguments.get("expression")
        if tool != "query_units" or not isinstance(expression, str) or _CURRICULUM_MARKER not in expression:
            return response_text
        return response_text.replace(
            "case:sessions-only-query query_shape:sessions-only-query physical_size:bounded "
            "corpus_match:true structure_discovery:absent shipped_instruction:advertised",
            "case:sessions-only-query query_shape:sessions-only-query physical_size:bounded "
            "corpus_match:true structure_discovery:absent shipped_instruction:absent",
        )

    return ContinuityMutation(
        name="unreasonable-query-classification",
        scenario="parallel-claude-incident",
        expected_kind="attempt_grade_mismatch",
        expected_failure_class="reasoning",
        response_mutator=mutate,
    )


__all__ = [
    "ContinuityMutation",
    "continuity_mutation",
    "continuity_mutation_names",
]
