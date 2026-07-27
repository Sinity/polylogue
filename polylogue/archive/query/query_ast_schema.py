"""Canonical, versioned Pydantic projection of the query DSL's compiled AST.

``polylogue/archive/query/expression.py:explain_expression`` already computes
a compiled predicate tree, a per-branch ``ast`` dict, and a ``lowering_plan``
dict for every query DSL expression (fielded compact queries, ``sessions
where ...`` Boolean expressions, terminal unit sources such as ``actions
where ...``, and durable-reference pipelines). Those dicts are produced by
hand-rolled ``to_payload()`` methods scattered across
:mod:`polylogue.archive.query.predicate` and
:mod:`polylogue.archive.query.expression` -- correct, but untyped and
undocumented from an external consumer's point of view (an MCP client or
OpenAPI-generated SDK sees ``dict[str, object] | None``).

This module does **not** introduce a second AST or a parallel intermediate
representation. It defines Pydantic models that mirror the existing
dataclasses' ``to_payload()`` shapes one-to-one, then *validates* the
already-produced payload against them (see :func:`explanation_payload_to_ast`,
:func:`predicate_to_ast`). If a producer's ``to_payload()`` ever drifts from
the shape declared here, validation fails loudly in
``tests/unit/archive/query/test_query_ast_schema.py`` rather than silently
diverging -- that test is the parity gate the bead's design calls for
("adding/removing a field ... fails one actionable check").

Two independent version axes are in play and must not be conflated:

* :data:`polylogue.core.query_identity.QUERY_DEFINITION_PROTOCOL_VERSION`
  (``polylogue.query-definition.v1``) versions the *content-addressed*
  predicate grammar used for query hashing/identity
  (``predicate_from_payload`` / ``QueryPredicate.to_payload``). This module's
  :data:`QueryPredicateAst` union is a typed, schema-validated *view* of that
  same v1 grammar -- it does not define a new grammar version.
* :data:`QUERY_AST_SCHEMA_VERSION` (``polylogue.query-explain-ast.v1``)
  versions the broader *discovery/explain* envelope defined in this module
  (clauses, unit sources, pipelines, lowering plan) that has no prior typed
  home. Bump it when this envelope's shape changes non-additively.

Reused, not reinvented: every leaf/composite predicate kind, pipeline stage
kind, and terminal action here matches an existing closed vocabulary in
:mod:`polylogue.archive.query.predicate` and
:mod:`polylogue.archive.query.expression`. This module adds a stable,
JSON-Schema-capable serialization shape on top of them for external tooling
(OpenAPI generation, MCP-facing typed discovery) -- see
``devtools/render_openapi.py`` and
``polylogue/archive/query/expression.py:QueryExpressionExplanation.to_payload``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from polylogue.archive.query.metadata import QueryUnitName
from polylogue.archive.query.predicate import (
    QueryBoolOp,
    QueryCompareOp,
    QueryExistsUnit,
    QueryPredicate,
    QuerySequenceConstraintKind,
    predicate_from_payload,
)

#: Version stamp for the canonical query-explain AST envelope defined below
#: (:class:`QueryExpressionExplanationAst` and everything it nests). See the
#: module docstring for how this relates to
#: ``QUERY_DEFINITION_PROTOCOL_VERSION``.
QUERY_AST_SCHEMA_VERSION: Literal["polylogue.query-explain-ast.v1"] = "polylogue.query-explain-ast.v1"


class _AstModel(BaseModel):
    """Shared strict base: unknown keys fail validation instead of being dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Predicate tree -- mirrors polylogue.archive.query.predicate.QueryPredicate
# ---------------------------------------------------------------------------


class QueryFieldRefAst(_AstModel):
    """Validated field identity carried by a field-predicate leaf."""

    scope: Literal["session", "unit"]
    name: str
    source_name: str
    unit: str | None = None


class QueryFieldPredicateAst(_AstModel):
    """Leaf predicate over one supported session-query field."""

    kind: Literal["field"] = "field"
    field: str
    op: QueryCompareOp = "="
    values: list[str] = Field(default_factory=list)
    field_ref: QueryFieldRefAst | None = None


class QueryNotPredicateAst(_AstModel):
    """Boolean negation over a predicate subtree."""

    kind: Literal["not"] = "not"
    child: QueryPredicateAst


class QueryBoolPredicateAst(_AstModel):
    """N-ary Boolean operator over predicate subtrees."""

    kind: QueryBoolOp
    children: list[QueryPredicateAst] = Field(default_factory=list)


class QuerySequenceConstraintAst(_AstModel):
    """Constraint on the edge between two action-sequence steps."""

    kind: QuerySequenceConstraintKind = "ordered"
    within_ms: int | None = None


class QueryExistsPredicateAst(_AstModel):
    """Correlated structural predicate over a child archive unit."""

    kind: Literal["exists"] = "exists"
    unit: QueryExistsUnit
    child: QueryPredicateAst


class QuerySequencePredicateAst(_AstModel):
    """Ordered action-sequence predicate over a session."""

    kind: Literal["sequence"] = "sequence"
    unit: Literal["action"] = "action"
    steps: list[QueryPredicateAst] = Field(default_factory=list)
    constraints: list[QuerySequenceConstraintAst] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class QueryTextPredicateAst(_AstModel):
    """Lexical FTS predicate over session message/block text."""

    kind: Literal["fts"] = "fts"
    unit: Literal["session"] = "session"
    text: str


class QuerySemanticPredicateAst(_AstModel):
    """Semantic vector predicate over session message/block text."""

    kind: Literal["semantic"] = "semantic"
    unit: Literal["session"] = "session"
    text: str


class QueryLineagePredicateAst(_AstModel):
    """Session-topology predicate selecting the seed's logical lineage."""

    kind: Literal["lineage"] = "lineage"
    unit: Literal["session"] = "session"
    seed_session_id: str


QueryPredicateAst = Annotated[
    QueryFieldPredicateAst
    | QueryNotPredicateAst
    | QueryBoolPredicateAst
    | QueryExistsPredicateAst
    | QuerySequencePredicateAst
    | QueryTextPredicateAst
    | QuerySemanticPredicateAst
    | QueryLineagePredicateAst,
    Field(discriminator="kind"),
]

for _predicate_model in (
    QueryFieldPredicateAst,
    QueryNotPredicateAst,
    QueryBoolPredicateAst,
    QueryExistsPredicateAst,
    QuerySequencePredicateAst,
    QueryTextPredicateAst,
    QuerySemanticPredicateAst,
    QueryLineagePredicateAst,
):
    _predicate_model.model_rebuild()

_predicate_adapter: TypeAdapter[Any] = TypeAdapter(QueryPredicateAst)


def predicate_to_ast(predicate: QueryPredicate) -> Any:
    """Project a compiled predicate node into the canonical, typed AST.

    This validates ``predicate``'s own lossless ``to_payload()`` projection
    against :data:`QueryPredicateAst` -- it does not re-derive the payload by
    walking the dataclass a second time, so the two shapes cannot drift apart
    without a validation failure surfacing immediately.
    """
    return _predicate_adapter.validate_python(predicate.to_payload())


def ast_to_predicate(ast: Any) -> QueryPredicate:
    """Invert :func:`predicate_to_ast` back into a typed predicate node."""
    payload = cast("dict[str, object]", ast.model_dump(mode="json", exclude_none=True))
    return predicate_from_payload(payload)


# ---------------------------------------------------------------------------
# Explain clause / reference-operand / pipeline projections
# ---------------------------------------------------------------------------


class QueryExpressionClauseAst(_AstModel):
    """Mirrors ``QueryExpressionExplainClause.to_payload()``."""

    kind: Literal["field", "count", "count_range", "date", "date_range", "text", "json"]
    field: str | None = None
    value: str | None = None
    negated: bool = False
    quoted: bool = False
    op: QueryCompareOp | None = None
    number: int | None = None
    min_number: int | None = None
    max_number: int | None = None
    min_value: str | None = None
    max_value: str | None = None


class RefOperandAst(_AstModel):
    """Mirrors ``RefOperand.to_payload()``."""

    kind: Literal["ref_operand"] = "ref_operand"
    reference: str
    reference_kind: str
    evaluation_mode: Literal["re-evaluate", "retained", "resolver-defined"]
    grain: str | None = None


class ReferenceQueryPipelineAst(_AstModel):
    """Mirrors ``ReferenceQueryPipeline.to_payload()``."""

    source: RefOperandAst
    stages: list[str] = Field(default_factory=list)


class QueryUnitSortSpecAst(_AstModel):
    """Mirrors ``QueryUnitSort`` (field/direction)."""

    field: Literal["time", "count", "key"]
    direction: Literal["asc", "desc"] = "asc"


class QueryUnitSessionScopeStageAst(_AstModel):
    kind: Literal["session_scope"] = "session_scope"
    predicate: QueryPredicateAst


class QueryUnitSortStageAst(_AstModel):
    kind: Literal["sort"] = "sort"
    sort: QueryUnitSortSpecAst


class QueryUnitLimitStageAst(_AstModel):
    kind: Literal["limit"] = "limit"
    value: int


class QueryUnitOffsetStageAst(_AstModel):
    kind: Literal["offset"] = "offset"
    value: int


class QueryUnitGroupStageAst(_AstModel):
    kind: Literal["group"] = "group"
    field: str | None = None
    fields: list[str] | None = None


class QueryUnitCountStageAst(_AstModel):
    kind: Literal["count"] = "count"
    metric: Literal["count"] = "count"


class QueryUnitTransformStageAst(_AstModel):
    kind: Literal["transform"] = "transform"
    name: str
    args: dict[str, str] | None = None


class QueryUnitTerminalStageAst(_AstModel):
    kind: Literal["terminal"] = "terminal"
    action: str
    args: dict[str, str] | None = None


QueryUnitPipelineStageAst = Annotated[
    QueryUnitSessionScopeStageAst
    | QueryUnitSortStageAst
    | QueryUnitLimitStageAst
    | QueryUnitOffsetStageAst
    | QueryUnitGroupStageAst
    | QueryUnitCountStageAst
    | QueryUnitTransformStageAst
    | QueryUnitTerminalStageAst,
    Field(discriminator="kind"),
]

QueryUnitSessionScopeStageAst.model_rebuild()


class QueryUnitPipelineSourceAst(_AstModel):
    unit: QueryUnitName
    predicate: QueryPredicateAst


class QueryUnitPipelineResultAst(_AstModel):
    sort: QueryUnitSortSpecAst | None = None
    group_by: str | None = None
    aggregate: Literal["count"] | None = None
    limit: int | None = None
    offset: int | None = None


class QueryUnitPipelineAst(_AstModel):
    """Mirrors ``QueryUnitPipeline.to_payload()``."""

    source: QueryUnitPipelineSourceAst
    stages: list[QueryUnitPipelineStageAst] = Field(default_factory=list)
    session_scope: QueryPredicateAst | None = None
    result: QueryUnitPipelineResultAst | None = None


class QueryUnitSourceAst(_AstModel):
    """Mirrors the ``unit_source`` branch of ``_ast_payload()``."""

    unit: QueryUnitName
    predicate: QueryPredicateAst
    session_predicate: QueryPredicateAst | None = None
    limit: int | None = None
    offset: int | None = None
    sort: QueryUnitSortSpecAst | None = None
    group_by: str | None = None
    aggregate: Literal["count"] | None = None
    pipeline_stages: list[QueryUnitPipelineStageAst] = Field(default_factory=list)
    pipeline: QueryUnitPipelineAst


class QueryExpressionAstNodeAst(_AstModel):
    """Mirrors ``_ast_payload()``'s dict shape: one entry-tagged AST node."""

    entry: Literal["json", "reference_pipeline", "unit_source", "boolean", "compact"]
    clauses: list[QueryExpressionClauseAst] | None = None
    predicate: QueryPredicateAst | None = None
    unit_source: QueryUnitSourceAst | None = None
    reference_pipeline: ReferenceQueryPipelineAst | None = None


class QueryLoweringPlanAst(_AstModel):
    """Mirrors ``_lowering_plan_payload()``."""

    lowerer: str
    selected_units: list[str] = Field(default_factory=list)
    execution_legs: list[str] = Field(default_factory=list)
    plan_description: list[str] = Field(default_factory=list)
    compatibility_selector: str | None = None
    pipeline: QueryUnitPipelineAst | None = None
    pipeline_stages: list[QueryUnitPipelineStageAst] | None = None
    #: Durable-reference ancestry (formatted ``ObjectRef`` strings), present
    #: only on the ``reference-operand-to-planner-relation`` lowerer branch
    #: (``explain_expression``'s ``reference_pipeline`` entry).
    reference_lineage: list[str] | None = None


class QueryExpressionExplanationAst(_AstModel):
    """Canonical, versioned projection of ``QueryExpressionExplanation.to_payload()``.

    This is the schema published in ``docs/openapi/search.yaml`` and the
    documented shape of the MCP ``explain`` operation's ``kind="query"``
    result (``polylogue.mcp.server_cutover:explain`` ->
    ``Polylogue.explain_query_expression``). It is deliberately additive over
    history: every field already existed in the hand-rolled payload dict this
    module validates against; ``schema_version`` is the only new key.
    """

    schema_version: Literal["polylogue.query-explain-ast.v1"] = QUERY_AST_SCHEMA_VERSION
    source_text: str
    clauses: list[QueryExpressionClauseAst] = Field(default_factory=list)
    predicate: QueryPredicateAst | None = None
    ast: QueryExpressionAstNodeAst | None = None
    lowerer: str
    lowering_plan: QueryLoweringPlanAst | None = None
    selected_units: list[str] = Field(default_factory=list)
    execution_legs: list[str] = Field(default_factory=list)
    plan_description: list[str] = Field(default_factory=list)
    unsupported_nodes: list[str] = Field(default_factory=list)


def explanation_payload_to_ast(payload: dict[str, object]) -> QueryExpressionExplanationAst:
    """Validate an already-built ``QueryExpressionExplanation.to_payload()`` dict.

    This is the parity gate: any hand-rolled payload shape that does not
    match this module's declared schema raises a Pydantic ``ValidationError``
    here rather than silently reaching an agent or generated client.
    """
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
]
