"""Round-trip and parity tests for the canonical query-AST Pydantic schema.

``polylogue.archive.query.query_ast_schema`` is a *validating* Pydantic
projection over the existing hand-rolled ``to_payload()`` dicts produced by
:mod:`polylogue.archive.query.predicate` and
:mod:`polylogue.archive.query.expression`. These tests exercise two
independent claims:

1. The predicate-tree AST round-trips losslessly:
   ``predicate -> predicate_to_ast -> ast_to_predicate == predicate`` for a
   representative spread of predicate shapes (field leaves, Boolean AND/OR
   trees, exists/sequence structural predicates, and the three unit-scoped
   leaves: fts/semantic/lineage).
2. The full ``explain_expression()`` payload for a representative spread of
   DSL surfaces (compact field query, Boolean ``sessions where``, ``near:``
   semantic and ``near:id:`` lineage seeds, terminal unit sources with
   pipeline stages, a durable-reference pipeline, and a raw JSON spec) is
   valid against :class:`QueryExpressionExplanationAst` -- i.e. this schema
   is not aspirational, it is what the parser+lowerer actually emit today.

If a producer ever changes shape without a matching schema update, these
tests fail with a Pydantic ``ValidationError`` (parity gate), not a silent
divergence an agent or generated OpenAPI client would only discover later.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from polylogue.archive.query.expression import explain_expression
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequenceConstraint,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.query.query_ast_schema import (
    QUERY_AST_SCHEMA_VERSION,
    QueryExpressionExplanationAst,
    QueryPredicateAst,
    ast_to_predicate,
    explanation_payload_to_ast,
    predicate_to_ast,
)

_PREDICATE_ROUNDTRIP_CASES: tuple[QueryPredicate, ...] = (
    QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
    QueryFieldPredicate(field="origin", values=("codex-session",), op="=").with_field_ref(
        QueryFieldRef(scope="session", name="origin", source_name="origin")
    ),
    QueryFieldPredicate(field="count", values=("3",), op=">=").with_field_ref(
        QueryFieldRef(scope="unit", name="count", source_name="count", unit="message")
    ),
    QueryNotPredicate(QueryFieldPredicate(field="origin", values=("codex-session",), op="=")),
    QueryBoolPredicate(
        "and",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryFieldPredicate(field="repo", values=("polylogue",), op="="),
        ),
    ),
    QueryBoolPredicate(
        "or",
        (
            QueryFieldPredicate(field="origin", values=("codex-session",), op="="),
            QueryNotPredicate(QueryFieldPredicate(field="repo", values=("polylogue",), op="=")),
        ),
    ),
    QueryExistsPredicate(unit="block", child=QueryFieldPredicate(field="tool_name", values=("Bash",), op="=")),
    QuerySequencePredicate(action_terms=("plan", "edit", "test")),
    QuerySequencePredicate(
        steps=(
            QueryFieldPredicate(field="action", values=("plan",), op="="),
            QueryFieldPredicate(field="action", values=("edit",), op="="),
        ),
        constraints=(QuerySequenceConstraint(kind="within", within_ms=60_000),),
    ),
    QueryTextPredicate(text="deploy with caveats"),
    QuerySemanticPredicate(text="deploy with caveats"),
    QueryLineagePredicate(seed_session_id="codex-session:abc123"),
)


@pytest.mark.parametrize("predicate", _PREDICATE_ROUNDTRIP_CASES, ids=lambda p: type(p).__name__)
def test_predicate_ast_roundtrip_is_lossless(predicate: QueryPredicate) -> None:
    ast = predicate_to_ast(predicate)
    reconstructed = ast_to_predicate(ast)
    assert reconstructed == predicate
    # And the reconstructed predicate must re-validate to an equal AST node,
    # not merely compare equal as a dataclass.
    assert predicate_to_ast(reconstructed) == ast


_EXPRESSION_CASES: tuple[str, ...] = (
    'repo:polylogue since:7d "json envelope"',
    "sessions where exists block(type:code) AND lineage:id:root",
    'sessions where semantic:"query compiler" AND title:hit',
    "sessions where seq(action:file_edit -> action:shell AND output:failed)",
    "messages where role:assistant AND text:timeout",
    "messages where role:assistant | sort by time desc | limit 2 | offset 3",
    "sessions where repo:polylogue | messages where role:assistant | limit 5",
    "from result-set:stable-set | group by model | count",
    '{"repo": "polylogue", "limit": 5}',
    "messages between 5 and 20",
    "context-snapshots where session.repo:polylogue AND boundary:session_start",
    "actions where action:file_edit AND path:polylogue",
    "deploy failed today",
)


@pytest.mark.parametrize("expression", _EXPRESSION_CASES)
def test_explanation_payload_validates_against_canonical_ast(expression: str) -> None:
    explanation = explain_expression(expression)
    payload = explanation.to_payload()

    ast = explanation_payload_to_ast(payload)

    assert ast.schema_version == QUERY_AST_SCHEMA_VERSION
    assert payload["schema_version"] == QUERY_AST_SCHEMA_VERSION
    assert ast.source_text == expression
    assert ast.lowerer == explanation.lowerer

    if explanation.predicate is not None:
        assert ast.predicate is not None
        # The typed predicate sub-tree round-trips back to the exact
        # dataclass the parser produced, not just a structurally similar one.
        assert ast_to_predicate(ast.predicate) == explanation.predicate


def test_canonical_ast_schema_is_json_schema_serializable() -> None:
    """The schema powering ``devtools render openapi`` must build cleanly."""
    schema = QueryExpressionExplanationAst.model_json_schema(mode="serialization")
    assert schema["title"] == "QueryExpressionExplanationAst"
    # A representative nested predicate variant must be reachable from $defs
    # so OpenAPI consumers can resolve the full recursive predicate tree.
    defs = schema.get("$defs", {})
    assert "QueryBoolPredicateAst" in defs
    assert "QueryFieldPredicateAst" in defs


def test_canonical_ast_rejects_unknown_top_level_key() -> None:
    """Schema drift (an added/removed key) must fail loudly, not silently pass."""
    explanation = explain_expression("repo:polylogue")
    payload = dict(explanation.to_payload())
    payload["unexpected_new_field"] = "surprise"
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        explanation_payload_to_ast(payload)


def test_canonical_predicate_ast_rejects_unknown_kind() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(QueryPredicateAst)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "made-up"})
