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


QueryPredicate: TypeAlias = QueryFieldPredicate | QueryNotPredicate | QueryBoolPredicate


__all__ = [
    "QueryBoolOp",
    "QueryBoolPredicate",
    "QueryCompareOp",
    "QueryFieldPredicate",
    "QueryNotPredicate",
    "QueryPredicate",
]
