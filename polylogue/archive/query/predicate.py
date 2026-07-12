"""Typed Boolean predicates for the query DSL."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal, TypeAlias

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
]
