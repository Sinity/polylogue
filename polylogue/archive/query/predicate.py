"""Typed Boolean predicates for the query DSL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, ClassVar, ForwardRef, Literal, TypeAlias, cast

from polylogue.archive.query.payload_schema import PayloadField, PayloadSchema, render_payload

QueryBoolOp: TypeAlias = Literal["and", "or"]
QueryCompareOp: TypeAlias = Literal["=", ">", ">=", "<", "<="]
QueryFieldScope: TypeAlias = Literal["session", "unit"]
QueryExistsUnit: TypeAlias = Literal[
    "message",
    "action",
    "block",
    "assertion",
    "file",
    "run",
    "observed-event",
    "context-snapshot",
    "delegation",
]
QuerySequenceConstraintKind: TypeAlias = Literal["ordered", "next", "within"]

_EXISTS_UNITS: frozenset[str] = frozenset(
    {"message", "action", "block", "assertion", "file", "run", "observed-event", "context-snapshot", "delegation"}
)
_COMPARE_OPS: frozenset[str] = frozenset({"=", ">", ">=", "<", "<="})
_FIELD_SCOPES: frozenset[str] = frozenset({"session", "unit"})
_SEQUENCE_CONSTRAINT_KINDS: frozenset[str] = frozenset({"ordered", "next", "within"})


@dataclass(frozen=True)
class QuerySequenceConstraint:
    """Constraint on the edge between two action-sequence steps."""

    kind: QuerySequenceConstraintKind = "ordered"
    within_ms: int | None = None

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


QuerySequenceConstraint.PAYLOAD = PayloadSchema(
    "QuerySequenceConstraintAst",
    (
        PayloadField("kind", lambda: QuerySequenceConstraintKind, default="ordered"),
        PayloadField("within_ms", lambda: int | None, omit="none", default=None),
    ),
    description="Constraint on the edge between two action-sequence steps.",
)


@dataclass(frozen=True)
class QueryFieldRef:
    """Validated field identity carried by executable field predicates."""

    scope: QueryFieldScope
    name: str
    source_name: str
    unit: str | None = None

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


QueryFieldRef.PAYLOAD = PayloadSchema(
    "QueryFieldRefAst",
    (
        PayloadField("scope", lambda: QueryFieldScope),
        PayloadField("name", lambda: str),
        PayloadField("source_name", lambda: str),
        PayloadField("unit", lambda: str | None, omit="none", default=None),
    ),
    description="Validated field identity carried by a field-predicate leaf.",
)


@dataclass(frozen=True)
class QueryFieldPredicate:
    """Leaf predicate over a supported session-query field."""

    field: str
    values: tuple[str, ...] = ()
    op: QueryCompareOp = "="
    field_ref: QueryFieldRef | None = dataclass_field(default=None, compare=False, repr=False)

    def with_field_ref(self, field_ref: QueryFieldRef) -> QueryFieldPredicate:
        """Return this predicate annotated with validated field identity."""

        return QueryFieldPredicate(
            field=self.field,
            values=self.values,
            op=self.op,
            field_ref=field_ref,
        )

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)

    def require_field_ref(self, *, context: str) -> QueryFieldRef:
        """Return the validated field identity or fail at an execution boundary."""

        if self.field_ref is None:
            raise ValueError(
                f"unbound query field predicate {self.field!r}; bind query predicate context before {context}"
            )
        return self.field_ref

    def bound_field_name(self, *, context: str) -> str:
        """Return the validated field name for executable lowerers."""

        return self.require_field_ref(context=context).name


@dataclass(frozen=True)
class QueryNotPredicate:
    """Boolean negation over a predicate subtree."""

    child: QueryPredicate

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


@dataclass(frozen=True)
class QueryBoolPredicate:
    """N-ary Boolean operator over predicate subtrees."""

    op: QueryBoolOp
    children: tuple[QueryPredicate, ...]

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


@dataclass(frozen=True)
class QueryExistsPredicate:
    """Correlated structural predicate over a child archive unit."""

    unit: QueryExistsUnit
    child: QueryPredicate

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


@dataclass(frozen=True)
class QuerySequencePredicate:
    """Ordered action-sequence predicate over a session."""

    steps: tuple[QueryPredicate, ...] = ()
    action_terms: tuple[str, ...] = ()
    constraints: tuple[QuerySequenceConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.steps:
            object.__setattr__(self, "action_terms", _simple_sequence_action_terms(self.steps))
        elif self.action_terms:
            object.__setattr__(
                self,
                "steps",
                tuple(QueryFieldPredicate(field="action", values=(term,), op="=") for term in self.action_terms),
            )
        if self.steps and not self.constraints:
            object.__setattr__(
                self,
                "constraints",
                tuple(QuerySequenceConstraint() for _ in range(len(self.steps) - 1)),
            )
        if len(self.constraints) != max(0, len(self.steps) - 1):
            raise ValueError("sequence constraints must describe every edge between steps")

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


def _simple_sequence_action_terms(steps: tuple[QueryPredicate, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    for step in steps:
        if (
            isinstance(step, QueryFieldPredicate)
            and step.field == "action"
            and step.op == "="
            and len(step.values) == 1
        ):
            terms.append(step.values[0])
            continue
        return ()
    return tuple(terms)


@dataclass(frozen=True)
class QueryTextPredicate:
    """Lexical FTS predicate over session message/block text."""

    text: str

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


QueryTextPredicate.PAYLOAD = PayloadSchema(
    "QueryTextPredicateAst",
    (
        PayloadField("kind", lambda: Literal["fts"], const="fts"),
        PayloadField("unit", lambda: Literal["session"], const="session"),
        PayloadField("text", lambda: str),
    ),
    description="Lexical FTS predicate over session message/block text.",
)


@dataclass(frozen=True)
class QuerySemanticPredicate:
    """Semantic vector predicate over session message/block text."""

    text: str

    PAYLOAD: ClassVar[PayloadSchema]

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


QuerySemanticPredicate.PAYLOAD = PayloadSchema(
    "QuerySemanticPredicateAst",
    (
        PayloadField("kind", lambda: Literal["semantic"], const="semantic"),
        PayloadField("unit", lambda: Literal["session"], const="session"),
        PayloadField("text", lambda: str),
    ),
    description="Semantic vector predicate over session message/block text.",
)


@dataclass(frozen=True)
class QueryLineagePredicate:
    """Session-topology predicate selecting the seed's logical lineage."""

    seed_session_id: str
    logical: bool = False

    PAYLOAD: ClassVar[PayloadSchema]

    @property
    def payload_kind(self) -> Literal["lineage", "logical"]:
        """Wire tag: the lineage leaf publishes which topology it selects."""

        return "logical" if self.logical else "lineage"

    def to_payload(self) -> dict[str, object]:
        return render_payload(self, self.PAYLOAD)


QueryLineagePredicate.PAYLOAD = PayloadSchema(
    "QueryLineagePredicateAst",
    (
        PayloadField("kind", lambda: Literal["lineage", "logical"], source="payload_kind", default="lineage"),
        PayloadField("unit", lambda: Literal["session"], const="session"),
        PayloadField("seed_session_id", lambda: str),
    ),
    description="Session-topology predicate selecting the seed's logical lineage.",
)


QueryPredicate: TypeAlias = (
    QueryFieldPredicate
    | QueryNotPredicate
    | QueryBoolPredicate
    | QueryExistsPredicate
    | QuerySequencePredicate
    | QueryTextPredicate
    | QuerySemanticPredicate
    | QueryLineagePredicate
)


#: Forward references to published models a declaration nests. The models are
#: built in ``query_ast_schema`` and resolved by name once every member
#: exists, so a node can declare a child of its own recursive union.
PREDICATE_AST_REF: Any = ForwardRef("QueryPredicateAst")
FIELD_REF_AST_REF: Any = ForwardRef("QueryFieldRefAst")
SEQUENCE_CONSTRAINT_AST_REF: Any = ForwardRef("QuerySequenceConstraintAst")


QueryFieldPredicate.PAYLOAD = PayloadSchema(
    "QueryFieldPredicateAst",
    (
        PayloadField("kind", lambda: Literal["field"], const="field"),
        PayloadField("field", lambda: str),
        PayloadField("op", lambda: QueryCompareOp, default="="),
        PayloadField("values", lambda: list[str], shape="scalar_list", default=[]),
        PayloadField("field_ref", lambda: FIELD_REF_AST_REF | None, shape="node", omit="none", default=None),
    ),
    description="Leaf predicate over one supported session-query field.",
)

QueryNotPredicate.PAYLOAD = PayloadSchema(
    "QueryNotPredicateAst",
    (
        PayloadField("kind", lambda: Literal["not"], const="not"),
        PayloadField("child", lambda: PREDICATE_AST_REF, shape="node"),
    ),
    description="Boolean negation over a predicate subtree.",
)

QueryBoolPredicate.PAYLOAD = PayloadSchema(
    "QueryBoolPredicateAst",
    (
        PayloadField("kind", lambda: QueryBoolOp, source="op"),
        PayloadField("children", lambda: list[PREDICATE_AST_REF], shape="node_list", default=[]),
    ),
    description="N-ary Boolean operator over predicate subtrees.",
)

QueryExistsPredicate.PAYLOAD = PayloadSchema(
    "QueryExistsPredicateAst",
    (
        PayloadField("kind", lambda: Literal["exists"], const="exists"),
        PayloadField("unit", lambda: QueryExistsUnit),
        PayloadField("child", lambda: PREDICATE_AST_REF, shape="node"),
    ),
    description="Correlated structural predicate over a child archive unit.",
)

QuerySequencePredicate.PAYLOAD = PayloadSchema(
    "QuerySequencePredicateAst",
    (
        PayloadField("kind", lambda: Literal["sequence"], const="sequence"),
        PayloadField("unit", lambda: Literal["action"], const="action"),
        PayloadField("steps", lambda: list[PREDICATE_AST_REF], shape="node_list", default=[]),
        PayloadField(
            "constraints",
            lambda: list[SEQUENCE_CONSTRAINT_AST_REF],
            shape="node_list",
            # Every edge carries a constraint internally; the wire omits them
            # when they are all the default ordering, so an ordinary sequence
            # predicate does not publish a list of empty edges.
            omit_when=lambda node: all(constraint.kind == "ordered" for constraint in node.constraints),
            default=[],
        ),
        PayloadField(
            "actions", lambda: list[str], source="action_terms", shape="scalar_list", omit="falsy", default=[]
        ),
    ),
    description="Ordered action-sequence predicate over a session.",
)


#: Declared predicate leaves and composites, in published union order.
PREDICATE_PAYLOAD_SCHEMAS: tuple[PayloadSchema, ...] = (
    QueryFieldPredicate.PAYLOAD,
    QueryNotPredicate.PAYLOAD,
    QueryBoolPredicate.PAYLOAD,
    QueryExistsPredicate.PAYLOAD,
    QuerySequencePredicate.PAYLOAD,
    QueryTextPredicate.PAYLOAD,
    QuerySemanticPredicate.PAYLOAD,
    QueryLineagePredicate.PAYLOAD,
)


def _field_ref_from_payload(payload: object) -> QueryFieldRef:
    if not isinstance(payload, Mapping):
        raise ValueError("field_ref payload must be an object")
    scope = payload.get("scope")
    name = payload.get("name")
    source_name = payload.get("source_name")
    unit = payload.get("unit")
    if scope not in _FIELD_SCOPES:
        raise ValueError(f"unsupported field_ref scope: {scope!r}")
    if not isinstance(name, str) or not name:
        raise ValueError("field_ref requires a non-empty 'name'")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("field_ref requires a non-empty 'source_name'")
    if unit is not None and not isinstance(unit, str):
        raise ValueError("field_ref 'unit' must be a string when present")
    return QueryFieldRef(
        scope=cast(QueryFieldScope, scope),
        name=name,
        source_name=source_name,
        unit=unit,
    )


def _sequence_constraint_from_payload(payload: object) -> QuerySequenceConstraint:
    if not isinstance(payload, Mapping):
        raise ValueError("sequence constraint payload must be an object")
    kind = payload.get("kind", "ordered")
    within_ms = payload.get("within_ms")
    if kind not in _SEQUENCE_CONSTRAINT_KINDS:
        raise ValueError(f"unsupported sequence constraint kind: {kind!r}")
    if within_ms is not None and (isinstance(within_ms, bool) or not isinstance(within_ms, int)):
        raise ValueError("sequence constraint 'within_ms' must be an integer")
    return QuerySequenceConstraint(kind=cast(QuerySequenceConstraintKind, kind), within_ms=within_ms)


def _payload_list(payload: object, *, field: str) -> Sequence[object]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{field!r} must be a list")
    return payload


def predicate_from_payload(payload: Mapping[str, object]) -> QueryPredicate:
    """Reconstruct a typed predicate from its own ``to_payload()`` projection.

    Every branch below inverts one dataclass's own lossless ``to_payload()``
    mapping (see the corresponding ``to_payload`` above each predicate class
    in this module), so round-tripping a value through ``to_payload`` then
    ``predicate_from_payload`` always reproduces an equal predicate. This is
    deliberately *not* a reverse-compiler over free-form or legacy text: it
    only understands the closed, versioned shape this module itself emits
    (``polylogue.query-definition.v1``). Callers that hold a legacy protocol
    v0 canonical plan (an opaque saved-view JSON request, not this predicate
    grammar) must not route it through this function -- see
    ``polylogue.core.query_identity.require_supported_definition_protocol_version``
    and ``polylogue.archive.query.production_evaluator``, which fails closed
    on v0 identities before reaching here.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("predicate payload must be an object")
    kind = payload.get("kind")
    if kind == "field":
        field = payload.get("field")
        op = payload.get("op", "=")
        values = payload.get("values", ())
        if not isinstance(field, str) or not field:
            raise ValueError("field predicate requires a non-empty 'field'")
        if op not in _COMPARE_OPS:
            raise ValueError(f"unsupported field predicate op: {op!r}")
        raw_values = _payload_list(values, field="values")
        if not all(isinstance(value, str) for value in raw_values):
            raise ValueError("field predicate 'values' must be a list of strings")
        predicate: QueryFieldPredicate = QueryFieldPredicate(
            field=field,
            values=tuple(cast(str, value) for value in raw_values),
            op=cast(QueryCompareOp, op),
        )
        field_ref_payload = payload.get("field_ref")
        if field_ref_payload is not None:
            predicate = predicate.with_field_ref(_field_ref_from_payload(field_ref_payload))
        return predicate
    if kind == "not":
        child = payload.get("child")
        if not isinstance(child, Mapping):
            raise ValueError("not predicate requires a 'child' object")
        return QueryNotPredicate(predicate_from_payload(child))
    if kind in ("and", "or"):
        children = _payload_list(payload.get("children"), field="children")
        parsed_children: list[QueryPredicate] = []
        for child in children:
            if not isinstance(child, Mapping):
                raise ValueError("boolean predicate children must be objects")
            parsed_children.append(predicate_from_payload(child))
        return QueryBoolPredicate(kind, tuple(parsed_children))
    if kind == "exists":
        unit = payload.get("unit")
        child = payload.get("child")
        if unit not in _EXISTS_UNITS:
            raise ValueError(f"unsupported exists unit: {unit!r}")
        if not isinstance(child, Mapping):
            raise ValueError("exists predicate requires a 'child' object")
        return QueryExistsPredicate(unit=cast(QueryExistsUnit, unit), child=predicate_from_payload(child))
    if kind == "sequence":
        steps_payload = _payload_list(payload.get("steps", ()), field="steps")
        parsed_steps: list[QueryPredicate] = []
        for step in steps_payload:
            if not isinstance(step, Mapping):
                raise ValueError("sequence predicate steps must be objects")
            parsed_steps.append(predicate_from_payload(step))
        constraints_payload = payload.get("constraints")
        constraints: tuple[QuerySequenceConstraint, ...] = ()
        if constraints_payload is not None:
            constraints = tuple(
                _sequence_constraint_from_payload(item)
                for item in _payload_list(constraints_payload, field="constraints")
            )
        return QuerySequencePredicate(steps=tuple(parsed_steps), constraints=constraints)
    if kind == "fts":
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("fts predicate requires non-empty 'text'")
        return QueryTextPredicate(text=text)
    if kind == "semantic":
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("semantic predicate requires non-empty 'text'")
        return QuerySemanticPredicate(text=text)
    if kind in ("lineage", "logical"):
        seed = payload.get("seed_session_id")
        if not isinstance(seed, str) or not seed:
            raise ValueError("lineage predicate requires non-empty 'seed_session_id'")
        return QueryLineagePredicate(seed_session_id=seed, logical=kind == "logical")
    raise ValueError(f"unsupported predicate payload kind: {kind!r}")


def lineage_seed_from_predicate(predicate: QueryPredicate | None) -> str | None:
    """Return the ``lineage:id:`` seed session id carried by *predicate*, if any.

    ``lineage:id:<ref>`` compiles to :class:`QueryLineagePredicate` (possibly
    ANDed with other clauses), which the SQL layer uses to filter session rows
    to one shared-root lineage family -- root and every descendant. That is
    already the "explicit projection" a lineage-seeded query asks for
    (polylogue-j8u2), so callers use this to exempt such a query from the
    default top-level-only ``root`` filter: forcing root-only on top of an
    explicit family walk would silently drop the very children the query
    asked for.

    Only descends into ``and`` nodes. A ``lineage:id:X or repo:foo`` result
    set is NOT purely lineage X's family -- rows matched only via the ``or``
    branch would get X's family semantics applied incorrectly. An ``or`` node
    (or a ``not`` wrapping the predicate, which isn't a
    ``QueryBoolPredicate``/``QueryLineagePredicate`` at all) correctly yields
    no seed here.
    """
    if predicate is None:
        return None
    if isinstance(predicate, QueryLineagePredicate):
        return predicate.seed_session_id
    if isinstance(predicate, QueryBoolPredicate) and predicate.op == "and":
        for child in predicate.children:
            seed = lineage_seed_from_predicate(child)
            if seed is not None:
                return seed
    return None


__all__ = [
    "PREDICATE_PAYLOAD_SCHEMAS",
    "QueryBoolOp",
    "QueryBoolPredicate",
    "QueryCompareOp",
    "QueryExistsPredicate",
    "QueryExistsUnit",
    "QueryFieldRef",
    "QueryFieldPredicate",
    "QueryFieldScope",
    "QueryLineagePredicate",
    "QueryNotPredicate",
    "QueryPredicate",
    "QuerySemanticPredicate",
    "QuerySequencePredicate",
    "QuerySequenceConstraint",
    "QuerySequenceConstraintKind",
    "QueryTextPredicate",
    "lineage_seed_from_predicate",
    "predicate_from_payload",
]
