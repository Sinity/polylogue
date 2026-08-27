"""Small, deliberately boring reference model for query differentials.

This is a test oracle, not an archive implementation.  It stores hydrated
domain sessions and evaluates the parser's typed AST directly.  Keeping the
oracle at the domain boundary makes it useful against every public read
surface without introducing a second query grammar or SQL implementation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from polylogue.archive.models import Session
from polylogue.archive.query.expression import (
    QueryExpressionAST,
    _CountRangeToken,
    _CountToken,
    _DateComparisonToken,
    _DateRangeToken,
    _FieldToken,
    _TextToken,
    parse_expression_ast,
)
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryExistsPredicate,
    QueryFieldPredicate,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.semantic.facts import build_session_semantic_facts
from polylogue.archive.semantic.pricing import harmonize_session_cost


def _text(session: Session) -> str:
    return "\n".join((message.text or "") for message in session.messages)


def _value(session: Session, field_name: str) -> object:
    facts = build_session_semantic_facts(session)
    messages = tuple(session.messages)
    values: dict[str, object] = {
        "id": str(session.id),
        "session": str(session.id),
        "title": session.title or "",
        "origin": str(session.origin),
        "messages": facts.total_messages,
        "words": facts.word_count,
        "thinking_messages": facts.thinking_messages,
        "tool_use_messages": facts.tool_messages,
        "has_tool_use": facts.tool_messages > 0,
        "has_thinking": facts.thinking_messages > 0,
        "text": _text(session),
        "date": facts.first_message_at,
        "time": facts.first_message_at,
        "role": tuple(str(message.role) for message in messages),
    }
    return values.get(field_name)


def _compare(actual: object, expected: str, op: str) -> bool:
    if isinstance(actual, bool):
        wanted = expected.lower() in {"true", "yes", "1"}
        left: object = actual
        right: object = wanted
    elif isinstance(actual, (int, float)):
        try:
            left, right = actual, float(expected)
        except ValueError:
            return False
    elif isinstance(actual, datetime):
        try:
            right = datetime.fromisoformat(expected.replace("Z", "+00:00"))
        except ValueError:
            return False
        left = actual
    elif isinstance(actual, tuple):
        return any(_compare(item, expected, op) for item in actual)
    else:
        left, right = str(actual or "").lower(), expected.lower()
    if op == "=":
        return left == right or (isinstance(left, str) and str(right) in left)
    if op == ">":
        return bool(left > right)  # type: ignore[operator]
    if op == ">=":
        return bool(left >= right)  # type: ignore[operator]
    if op == "<":
        return bool(left < right)  # type: ignore[operator]
    if op == "<=":
        return bool(left <= right)  # type: ignore[operator]
    return False


def _field(session: Session, predicate: QueryFieldPredicate) -> bool:
    actual = _value(session, predicate.field.removeprefix("session."))
    return any(_compare(actual, value, predicate.op) for value in predicate.values)


def evaluate_predicate(session: Session, predicate: QueryPredicate, *, lineage: set[str] | None = None) -> bool:
    """Evaluate one production :class:`QueryPredicate` without re-parsing it."""
    if isinstance(predicate, QueryFieldPredicate):
        return _field(session, predicate)
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in _text(session).lower()
    if isinstance(predicate, QuerySemanticPredicate):
        raise NotImplementedError("semantic predicates need a vector oracle")
    if isinstance(predicate, QueryNotPredicate):
        return not evaluate_predicate(session, predicate.child, lineage=lineage)
    if isinstance(predicate, QueryBoolPredicate):
        results = [evaluate_predicate(session, child, lineage=lineage) for child in predicate.children]
        return all(results) if predicate.op == "and" else any(results)
    if isinstance(predicate, QueryExistsPredicate):
        # Structural predicates are intentionally conservative in this first
        # slice: the common message/block text and role fields are covered.
        return any(evaluate_predicate(session, predicate.child, lineage=lineage) for _ in session.messages)
    if isinstance(predicate, QueryLineagePredicate):
        return lineage is not None and predicate.seed_session_id in lineage
    if isinstance(predicate, QuerySequencePredicate):
        terms = predicate.action_terms
        text = _text(session).lower()
        return bool(terms) and all(term.lower() in text for term in terms)
    raise TypeError(f"unsupported query predicate: {predicate!r}")


def _compact_matches(session: Session, token: object) -> bool:
    if isinstance(token, _FieldToken):
        return (
            any(_compare(_value(session, token.field), value, "=") for value in token.raw_value.strip("()").split("|"))
            != token.negated
        )
    if isinstance(token, _TextToken):
        return (token.text.lower() in _text(session).lower()) != token.negated
    if isinstance(token, _CountToken):
        return _compare(_value(session, token.field), str(token.number), token.op)
    if isinstance(token, _CountRangeToken):
        return _compare(_value(session, token.field), str(token.min_number), ">=") and _compare(
            _value(session, token.field), str(token.max_number), "<="
        )
    if isinstance(token, _DateComparisonToken):
        return _compare(_value(session, "date"), token.value, token.op)
    if isinstance(token, _DateRangeToken):
        return _compare(_value(session, "date"), token.min_value, ">=") and _compare(
            _value(session, "date"), token.max_value, "<="
        )
    return False


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    session_ids: tuple[str, ...]
    count: int
    facets: tuple[tuple[str, int], ...] = ()
    costs: tuple[tuple[str, float, bool], ...] = ()


@dataclass(slots=True)
class ReferenceArchive:
    """In-memory archive with canonical session, facet, lineage and cost folds."""

    sessions: dict[str, Session] = field(default_factory=dict)
    parent_by_child: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_sessions(cls, sessions: Iterable[Session]) -> ReferenceArchive:
        """Build an oracle from hydrated production sessions."""
        archive = cls()
        for session in sessions:
            archive.add(session)
        return archive

    def add(self, session: Session, *, parent_id: str | None = None) -> None:
        self.sessions[str(session.id)] = session
        if parent_id is not None:
            self.parent_by_child[str(session.id)] = parent_id

    def lineage(self, session_id: str) -> tuple[Session, ...]:
        result: list[Session] = []
        current = session_id
        seen: set[str] = set()
        while current in self.sessions and current not in seen:
            seen.add(current)
            result.append(self.sessions[current])
            current = self.parent_by_child.get(current, "")
        return tuple(reversed(result))

    def query(self, expression: str | QueryExpressionAST) -> ReferenceResult:
        ast = parse_expression_ast(expression) if isinstance(expression, str) else expression
        selected: list[Session] = []
        for session in self.sessions.values():
            lineage = {str(item.id) for item in self.lineage(str(session.id))}
            if ast.boolean_predicate is not None:
                matches = evaluate_predicate(session, ast.boolean_predicate, lineage=lineage)
            else:
                matches = all(_compact_matches(session, token) for token in ast.clauses)
            if matches:
                selected.append(session)
        selected.sort(key=lambda item: str(item.id))
        origins = sorted(str(session.origin) for session in selected)
        facets = tuple((origin, origins.count(origin)) for origin in sorted(set(origins)))
        costs = tuple((str(session.id), *harmonize_session_cost(session)) for session in selected)
        return ReferenceResult(tuple(str(item.id) for item in selected), len(selected), facets, costs)


__all__ = ["ReferenceArchive", "ReferenceResult", "evaluate_predicate"]
