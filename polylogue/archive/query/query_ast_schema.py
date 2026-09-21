"""Canonical, versioned wire schema for the query DSL's compiled AST.

``polylogue.archive.query.expression.explain_expression`` computes a compiled
predicate tree, a per-branch ``ast`` document, and a ``lowering_plan`` document
for every query DSL expression (fielded compact queries, ``sessions where ...``
Boolean expressions, terminal unit sources such as ``actions where ...``, and
durable-reference pipelines). External consumers -- an MCP client, an
OpenAPI-generated SDK, the generated webui client -- need those documents
described, not handed to them as ``dict[str, object] | None``.

This module publishes that description. It does **not** restate it: every
model here is built from the payload declaration the producing node already
owns (:mod:`polylogue.archive.query.payload_schema`), which is the same
declaration the node serializes itself from. Adding a key to a node's payload
therefore reaches the rendered OpenAPI and the generated client with no second
edit, and a hand-written schema that could disagree with the serializer has
nowhere to live.

Two independent version axes are in play and must not be conflated:

* :data:`polylogue.core.query_identity.QUERY_DEFINITION_PROTOCOL_VERSION`
  (``polylogue.query-definition.v1``) versions the *content-addressed*
  predicate grammar used for query hashing and identity
  (``predicate_from_payload`` / ``QueryPredicate.to_payload``).
  :data:`QueryPredicateAst` is a typed view of that same v1 grammar; it does
  not define a new grammar version.
* :data:`QUERY_AST_SCHEMA_VERSION` (``polylogue.query-explain-ast.v1``)
  versions the broader discovery/explain envelope (clauses, unit sources,
  pipelines, lowering plan). Bump it when that envelope changes
  non-additively.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from polylogue.archive.query.expression import (
    PIPELINE_STAGE_PAYLOAD_SCHEMAS,
    QUERY_AST_PAYLOAD_SCHEMAS,
    QUERY_AST_SCHEMA_VERSION,
)
from polylogue.archive.query.payload_schema import (
    PayloadModelRegistry,
    PayloadUnion,
    union_annotation,
)
from polylogue.archive.query.predicate import (
    PREDICATE_PAYLOAD_SCHEMAS,
    QueryFieldRef,
    QueryPredicate,
    QuerySequenceConstraint,
    predicate_from_payload,
)

_registry = PayloadModelRegistry()

#: Nested declarations that are reached only through another node.
_NESTED_PAYLOAD_SCHEMAS = (QueryFieldRef.PAYLOAD, QuerySequenceConstraint.PAYLOAD)

for _schema in (*_NESTED_PAYLOAD_SCHEMAS, *PREDICATE_PAYLOAD_SCHEMAS, *QUERY_AST_PAYLOAD_SCHEMAS):
    _registry.build(_schema)

_PREDICATE_UNION = PayloadUnion("QueryPredicateAst", PREDICATE_PAYLOAD_SCHEMAS)
_PIPELINE_STAGE_UNION = PayloadUnion("QueryUnitPipelineStageAst", PIPELINE_STAGE_PAYLOAD_SCHEMAS)

QueryPredicateAst: Any = union_annotation(_PREDICATE_UNION, _registry.models)
QueryUnitPipelineStageAst: Any = union_annotation(_PIPELINE_STAGE_UNION, _registry.models)

#: Every published model plus the two union aliases, so recursive forward
#: references resolve without any of them being written down twice.
_NAMESPACE: dict[str, Any] = {
    **_registry.models,
    "QueryPredicateAst": QueryPredicateAst,
    "QueryUnitPipelineStageAst": QueryUnitPipelineStageAst,
}
_registry.rebuild(_NAMESPACE)


def published_ast_models() -> dict[str, type[BaseModel]]:
    """Return every published query-AST model, keyed by its schema name."""

    return dict(_registry.models)


QueryFieldRefAst = _registry.models["QueryFieldRefAst"]
QuerySequenceConstraintAst = _registry.models["QuerySequenceConstraintAst"]
QueryFieldPredicateAst = _registry.models["QueryFieldPredicateAst"]
QueryNotPredicateAst = _registry.models["QueryNotPredicateAst"]
QueryBoolPredicateAst = _registry.models["QueryBoolPredicateAst"]
QueryExistsPredicateAst = _registry.models["QueryExistsPredicateAst"]
QuerySequencePredicateAst = _registry.models["QuerySequencePredicateAst"]
QueryTextPredicateAst = _registry.models["QueryTextPredicateAst"]
QuerySemanticPredicateAst = _registry.models["QuerySemanticPredicateAst"]
QueryLineagePredicateAst = _registry.models["QueryLineagePredicateAst"]
RefOperandAst = _registry.models["RefOperandAst"]
ReferenceQueryPipelineAst = _registry.models["ReferenceQueryPipelineAst"]
QueryUnitSortSpecAst = _registry.models["QueryUnitSortSpecAst"]
QueryUnitAggMetricAst = _registry.models["QueryUnitAggMetricAst"]
QueryUnitSessionScopeStageAst = _registry.models["QueryUnitSessionScopeStageAst"]
QueryUnitSortStageAst = _registry.models["QueryUnitSortStageAst"]
QueryUnitLimitStageAst = _registry.models["QueryUnitLimitStageAst"]
QueryUnitOffsetStageAst = _registry.models["QueryUnitOffsetStageAst"]
QueryUnitGroupStageAst = _registry.models["QueryUnitGroupStageAst"]
QueryUnitCountStageAst = _registry.models["QueryUnitCountStageAst"]
QueryUnitAggStageAst = _registry.models["QueryUnitAggStageAst"]
QueryUnitTransformStageAst = _registry.models["QueryUnitTransformStageAst"]
QueryUnitTerminalStageAst = _registry.models["QueryUnitTerminalStageAst"]
QueryUnitPipelineSourceAst = _registry.models["QueryUnitPipelineSourceAst"]
QueryUnitPipelineResultAst = _registry.models["QueryUnitPipelineResultAst"]
QueryUnitPipelineAst = _registry.models["QueryUnitPipelineAst"]
QueryUnitSourceAst = _registry.models["QueryUnitSourceAst"]
QueryExpressionClauseAst = _registry.models["QueryExpressionClauseAst"]
QueryExpressionAstNodeAst = _registry.models["QueryExpressionAstNodeAst"]
QueryLoweringPlanAst = _registry.models["QueryLoweringPlanAst"]
QueryExpressionExplanationAst = _registry.models["QueryExpressionExplanationAst"]

_predicate_adapter: TypeAdapter[Any] = TypeAdapter(QueryPredicateAst)


def predicate_to_ast(predicate: QueryPredicate) -> Any:
    """Project a compiled predicate node into the canonical, typed AST.

    This validates ``predicate``'s own lossless ``to_payload()`` projection
    against :data:`QueryPredicateAst`. Both sides are produced from the node's
    one payload declaration, so a validation failure here means the payload
    the node actually emitted disagrees with the declaration it emitted it
    from -- a real defect, not a mirror falling behind.
    """
    return _predicate_adapter.validate_python(predicate.to_payload())


def ast_to_predicate(ast: Any) -> QueryPredicate:
    """Invert :func:`predicate_to_ast` back into a typed predicate node."""
    payload = cast("dict[str, object]", ast.model_dump(mode="json", exclude_none=True))
    return predicate_from_payload(payload)


def explanation_payload_to_ast(payload: dict[str, object]) -> Any:
    """Validate an already-built ``QueryExpressionExplanation.to_payload()`` dict."""
    return QueryExpressionExplanationAst.model_validate(payload)


__all__ = [
    "QUERY_AST_SCHEMA_VERSION",
    "QueryBoolPredicateAst",
    "QueryExistsPredicateAst",
    "QueryExpressionAstNodeAst",
    "QueryExpressionClauseAst",
    "QueryExpressionExplanationAst",
    "QueryFieldPredicateAst",
    "QueryFieldRefAst",
    "QueryLineagePredicateAst",
    "QueryLoweringPlanAst",
    "QueryNotPredicateAst",
    "QueryPredicateAst",
    "QuerySemanticPredicateAst",
    "QuerySequenceConstraintAst",
    "QuerySequencePredicateAst",
    "QueryTextPredicateAst",
    "QueryUnitAggMetricAst",
    "QueryUnitAggStageAst",
    "QueryUnitCountStageAst",
    "QueryUnitGroupStageAst",
    "QueryUnitLimitStageAst",
    "QueryUnitOffsetStageAst",
    "QueryUnitPipelineAst",
    "QueryUnitPipelineResultAst",
    "QueryUnitPipelineSourceAst",
    "QueryUnitPipelineStageAst",
    "QueryUnitSessionScopeStageAst",
    "QueryUnitSortSpecAst",
    "QueryUnitSortStageAst",
    "QueryUnitSourceAst",
    "QueryUnitTerminalStageAst",
    "QueryUnitTransformStageAst",
    "RefOperandAst",
    "ReferenceQueryPipelineAst",
    "ast_to_predicate",
    "explanation_payload_to_ast",
    "predicate_to_ast",
    "published_ast_models",
]
