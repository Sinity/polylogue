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

    unit: Literal["message", "action"]
    child: QueryPredicate

    def to_payload(self) -> dict[str, object]:
        return {"kind": "exists", "unit": self.unit, "child": self.child.to_payload()}


@dataclass(frozen=True)
class QuerySequencePredicate:
    """Ordered action-sequence predicate over a session."""

    action_terms: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {"kind": "sequence", "unit": "action", "actions": list(self.action_terms)}


@dataclass(frozen=True)
class QueryTextPredicate:
    """Lexical FTS predicate over session message/block text."""

    text: str

    def to_payload(self) -> dict[str, object]:
        return {"kind": "fts", "unit": "session", "text": self.text}


QueryPredicate: TypeAlias = (
    QueryFieldPredicate
    | QueryNotPredicate
    | QueryBoolPredicate
    | QueryExistsPredicate
    | QuerySequencePredicate
    | QueryTextPredicate
)


__all__ = [
    "QueryBoolOp",
    "QueryBoolPredicate",
    "QueryCompareOp",
    "QueryExistsPredicate",
    "QueryFieldPredicate",
    "QueryNotPredicate",
    "QueryPredicate",
    "QuerySequencePredicate",
    "QueryTextPredicate",
]
