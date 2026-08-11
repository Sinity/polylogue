"""Unit tests for the z9gh gap #3 cold-model plan-formulation contract.

These exercise the deterministic, catalog-only planning/pagination-validation
logic without a real MCP subprocess (see
``tests/integration/test_cold_model_continuity_replay.py`` for the real
stdio end-to-end proof). Anti-vacuity: :func:`select_cold_model_plan` is
scored against the real shipped ``QUERY_DISCOVERY_EXAMPLES`` catalog, not a
stand-in, and the mutation cases remove real catalog/schema evidence and
assert the planner fails honestly rather than silently guessing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from devtools import cold_model_continuity_replay as cmr
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES

_QUERY_TOOL_PROPERTIES: dict[str, object] = {
    "expression": {"type": "string"},
    "limit": {"type": "integer"},
    "continuation": {"type": "string"},
    "projection": {"type": "string"},
}

_CATALOG_PAYLOADS: tuple[Mapping[str, object], ...] = tuple(
    example.to_payload() for example in QUERY_DISCOVERY_EXAMPLES
)


def _goal(
    *,
    goal_keywords: Sequence[str] | None = None,
    substitutions: Mapping[str, str] | None = None,
) -> cmr.ColdModelGoal:
    base = cmr.DEFAULT_COLD_MODEL_GOAL
    return cmr.ColdModelGoal(
        goal_text=base.goal_text,
        goal_keywords=tuple(goal_keywords) if goal_keywords is not None else base.goal_keywords,
        substitutions=dict(substitutions) if substitutions is not None else base.substitutions,
        forced_page_limit=base.forced_page_limit,
        corpus_row_count=base.corpus_row_count,
    )


# ── Cold-model plan selection ──────────────────────────────────────────


def test_select_cold_model_plan_formulates_from_the_real_shipped_catalog() -> None:
    goal = _goal()

    plan = cmr.select_cold_model_plan(goal, _CATALOG_PAYLOADS, _QUERY_TOOL_PROPERTIES)

    assert plan.tool == "query"
    assert plan.arguments["expression"] == "assertions where kind:decision AND text:coldmodelpagingproof"
    assert plan.arguments["limit"] == goal.forced_page_limit
    assert plan.example_key == "assertions-decisions-about-topic"
    assert "topic" in plan.example_parameters
    assert set(goal.goal_keywords) & set(plan.matched_keywords)
    assert "expression" in plan.discovered_argument_names
    assert "continuation" in plan.discovered_argument_names


def test_select_cold_model_plan_fails_honestly_with_no_keyword_match() -> None:
    goal = _goal(goal_keywords=("zzz-unmatched-keyword-zzz",))

    with pytest.raises(cmr.ColdModelPlanningError, match="no discovery example"):
        cmr.select_cold_model_plan(goal, _CATALOG_PAYLOADS, _QUERY_TOOL_PROPERTIES)


def test_select_cold_model_plan_fails_honestly_when_substitution_is_missing() -> None:
    goal = _goal(substitutions={})

    with pytest.raises(cmr.ColdModelPlanningError, match="substitutions do not supply"):
        cmr.select_cold_model_plan(goal, _CATALOG_PAYLOADS, _QUERY_TOOL_PROPERTIES)


def test_select_cold_model_plan_fails_honestly_when_no_parameterized_examples_exist() -> None:
    goal = _goal()
    literal_only = tuple({**payload, "template": None, "parameters": []} for payload in _CATALOG_PAYLOADS)

    with pytest.raises(cmr.ColdModelPlanningError, match="no parameterized query-route"):
        cmr.select_cold_model_plan(goal, literal_only, _QUERY_TOOL_PROPERTIES)


def test_select_cold_model_plan_fails_honestly_when_schema_hides_a_required_argument() -> None:
    goal = _goal()
    hidden_schema = {key: value for key, value in _QUERY_TOOL_PROPERTIES.items() if key != "continuation"}

    with pytest.raises(cmr.ColdModelPlanningError, match="continuation"):
        cmr.select_cold_model_plan(goal, _CATALOG_PAYLOADS, hidden_schema)


def test_cold_model_module_never_imports_scripted_continuity_scenarios() -> None:
    import ast
    import inspect

    # A real static guard, not just a runtime behavior check: the planner
    # must formulate plans from discovery evidence alone, never from the
    # pre-scripted ContinuityRouteStep declarations t8t's own scenarios use.
    # (Prose mentioning "continuity_scenarios" in the module docstring is
    # fine and expected -- this checks the actual import graph.)
    tree = ast.parse(inspect.getsource(cmr))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    assert not any("continuity_scenarios" in name for name in imported_modules)


# ── Discovery catalog fetch fallback shape ─────────────────────────────


def test_discovery_catalog_fetch_payload_reports_source_and_diagnostic() -> None:
    fetch = cmr.DiscoveryCatalogFetch(
        source="in-process-registry-fallback",
        examples=_CATALOG_PAYLOADS[:1],
        wire_status="response_budget_exceeded",
        wire_original_bytes=82537,
        wire_budget_bytes=25000,
        diagnostic="explain overflowed",
    )

    payload = fetch.to_payload()

    assert payload["source"] == "in-process-registry-fallback"
    assert payload["example_count"] == 1
    assert payload["diagnostic"] == "explain overflowed"
    json.dumps(payload)


# ── Generic pagination proof shape ──────────────────────────────────────


def test_pagination_proof_flags_duplicate_identities_as_not_exact() -> None:
    proof = cmr.PaginationProof()
    proof.pages.append({"page": 1})
    proof.item_identities = ["a", "a"]

    assert proof.unique_identity_count == 1
    assert proof.exact_enumeration_verified is False


def test_pagination_proof_verified_when_counts_and_identities_agree() -> None:
    proof = cmr.PaginationProof()
    proof.pages.append({"page": 1})
    proof.item_identities = ["a", "b", "c"]
    proof.exact_count = 3

    assert proof.exact_enumeration_verified is True


def test_pagination_proof_not_verified_when_exact_count_mismatches() -> None:
    proof = cmr.PaginationProof()
    proof.pages.append({"page": 1})
    proof.item_identities = ["a", "b"]
    proof.exact_count = 3

    assert proof.exact_enumeration_verified is False
