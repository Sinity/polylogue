"""Public projections of private observed numeric evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from polylogue.core.json import JSONDocument, json_document


def _project_numeric_distribution(distribution: JSONDocument) -> None:
    numeric = distribution.get("numeric")
    if isinstance(numeric, dict):
        distribution["numeric"] = {
            key: value
            for key in ("count", "non_finite_count")
            if isinstance(value := numeric.get(key), int) and not isinstance(value, bool) and value >= 0
        }


def redact_observed_numeric_schema(schema: JSONDocument) -> JSONDocument:
    """Copy schema structure while removing observed numeric magnitudes."""
    result = copy.deepcopy(schema)

    def visit(node: JSONDocument) -> None:
        node.pop("x-polylogue-range", None)
        node.pop("x-polylogue-time-deltas", None)
        evidence = node.get("x-polylogue-evidence")
        if isinstance(evidence, dict):
            evidence.pop("range", None)
        distribution = node.get("x-polylogue-observed-distribution")
        if isinstance(distribution, dict):
            _project_numeric_distribution(distribution)
        for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            children = node.get(keyword)
            if isinstance(children, dict):
                for child in children.values():
                    if isinstance(child, dict):
                        visit(child)
        for keyword in (
            "items",
            "additionalItems",
            "additionalProperties",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
            "unevaluatedProperties",
            "unevaluatedItems",
            "anyOf",
            "allOf",
            "oneOf",
            "prefixItems",
        ):
            children = node.get(keyword)
            if isinstance(children, dict):
                visit(children)
            elif isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        visit(child)

    visit(result)
    return result


def redact_observed_numeric_workload(profile: Mapping[str, object]) -> JSONDocument:
    """Copy public field profiles without private numeric sketches."""
    from polylogue.schemas.generation.workload_profiles import workload_profile_identity

    result = copy.deepcopy(json_document(dict(profile)))
    elements = result.get("elements")
    if isinstance(elements, dict):
        for element in elements.values():
            if not isinstance(element, dict):
                continue
            fields = element.get("field_profiles")
            if isinstance(fields, dict):
                for distribution in fields.values():
                    if isinstance(distribution, dict):
                        _project_numeric_distribution(distribution)
    if "profile_id" in result:
        result["profile_id"] = workload_profile_identity(result)
    return result
