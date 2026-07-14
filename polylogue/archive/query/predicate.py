"""Typed Boolean predicates for the query DSL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal, TypeAlias, cast

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

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": self.kind}
        if self.within_ms is not None:
            payload["within_ms"] = self.within_ms
        return payload


@dataclass(frozen=True)
class QueryFieldRef:
    """Validated field identity carried by executable field predicates."""

    scope: QueryFieldScope
    name: str
    source_name: str
    unit: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope,
            "name": self.name,
            "source_name": self.source_name,
        }
        if self.unit is not None:
            payload["unit"] = self.unit
        return payload


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

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"kind": "field", "field": self.field, "op": self.op, "values": list(self.values)}
        if self.field_ref is not None:
            payload["field_ref"] = self.field_ref.to_payload()
        return payload

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

    def to_payload(self) -> dict[str, object]:
        return {"kind": "not", "child": self.child.to_payload()}


@dataclass(frozen=True)
class QueryBoolPredicate:
    """N-ary Boolean operator over predicate subtrees."""

    op: QueryBoolOp
    children: tuple[QueryPredicate, ...]

    def to_payload(self) -> dict[str, object]:
        return {"kind": self.op, "children": [child.to_payload() for child in self.children]}


@dataclass(frozen=True)
class QueryExistsPredicate:
    """Correlated structural predicate over a child archive unit."""

    unit: QueryExistsUnit
    child: QueryPredicate

    def to_payload(self) -> dict[str, object]:
        return {"kind": "exists", "unit": self.unit, "child": self.child.to_payload()}


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

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": "sequence",
            "unit": "action",
            "steps": [step.to_payload() for step in self.steps],
        }
        if any(constraint.kind != "ordered" for constraint in self.constraints):
            payload["constraints"] = [constraint.to_payload() for constraint in self.constraints]
        if self.action_terms:
            payload["actions"] = list(self.action_terms)
        return payload


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

    def to_payload(self) -> dict[str, object]:
        return {"kind": "fts", "unit": "session", "text": self.text}


@dataclass(frozen=True)
class QuerySemanticPredicate:
    """Semantic vector predicate over session message/block text."""

    text: str

    def to_payload(self) -> dict[str, object]:
        return {"kind": "semantic", "unit": "session", "text": self.text}


@dataclass(frozen=True)
class QueryLineagePredicate:
    """Session-topology predicate selecting the seed's logical lineage."""

    seed_session_id: str

    def to_payload(self) -> dict[str, object]:
        return {"kind": "lineage", "unit": "session", "seed_session_id": self.seed_session_id}


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
    if kind == "lineage":
        seed = payload.get("seed_session_id")
        if not isinstance(seed, str) or not seed:
            raise ValueError("lineage predicate requires non-empty 'seed_session_id'")
        return QueryLineagePredicate(seed_session_id=seed)
    raise ValueError(f"unsupported predicate payload kind: {kind!r}")


__all__ = [
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
    "predicate_from_payload",
]
