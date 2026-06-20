"""Typed Boolean predicates for the query DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

QueryBoolOp: TypeAlias = Literal["and", "or"]
QueryCompareOp: TypeAlias = Literal["=", ">=", "<="]


@dataclass(frozen=True)
class QueryFieldPredicate:
    """Leaf predicate over a supported session-query field."""

    field: str
    values: tuple[str, ...] = ()
    op: QueryCompareOp = "="

    def to_payload(self) -> dict[str, object]:
        return {"kind": "field", "field": self.field, "op": self.op, "values": list(self.values)}


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

    unit: Literal["message", "action", "block", "assertion"]
    child: QueryPredicate

    def to_payload(self) -> dict[str, object]:
        return {"kind": "exists", "unit": self.unit, "child": self.child.to_payload()}


@dataclass(frozen=True)
class QuerySequencePredicate:
    """Ordered action-sequence predicate over a session."""

    steps: tuple[QueryPredicate, ...] = ()
    action_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps and self.action_terms:
            object.__setattr__(
                self,
                "steps",
                tuple(QueryFieldPredicate(field="action", values=(term,), op="=") for term in self.action_terms),
            )
        elif self.steps and not self.action_terms:
            object.__setattr__(self, "action_terms", _simple_sequence_action_terms(self.steps))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": "sequence",
            "unit": "action",
            "steps": [step.to_payload() for step in self.steps],
        }
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
    "QueryFieldPredicate",
    "QueryLineagePredicate",
    "QueryNotPredicate",
    "QueryPredicate",
    "QuerySemanticPredicate",
    "QuerySequencePredicate",
    "QueryTextPredicate",
]
