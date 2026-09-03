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

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.models import Message, Session
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
    QueryFieldRef,
    QueryLineagePredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySemanticPredicate,
    QuerySequencePredicate,
    QueryTextPredicate,
)
from polylogue.archive.query.runtime_matching import matches_action_predicate_sequence
from polylogue.archive.semantic.facts import build_session_semantic_facts
from polylogue.archive.semantic.pricing import harmonize_session_cost


def _text(session: Session) -> str:
    return "\n".join((message.text or "") for message in session.messages)


def _recomposed_session(archive: ReferenceArchive, session: Session) -> Session:
    """Return the session view with its inherited prefix physically composed."""
    lineage = archive.lineage(str(session.id))
    if len(lineage) == 1:
        return session
    messages = [message for ancestor in lineage for message in ancestor.messages]
    return session.model_copy(update={"messages": MessageCollection(messages=messages)})


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


def _unit_field(value: object, predicate: QueryFieldPredicate) -> bool:
    actual = value
    return any(_compare(actual, expected, predicate.op) for expected in predicate.values)


def _message_matches(message: Message, predicate: QueryPredicate) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        field = predicate.field.removeprefix("message.")
        if field == "role":
            return _unit_field(str(message.role), predicate)
        if field == "text":
            return _unit_field(getattr(message, "text", "") or "", predicate)
        if field == "words":
            return _unit_field(len((getattr(message, "text", "") or "").split()), predicate)
        if field == "time":
            return _unit_field(getattr(message, "timestamp", None), predicate)
        return False
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in (getattr(message, "text", "") or "").lower()
    if isinstance(predicate, QueryNotPredicate):
        return not _message_matches(message, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        values = [_message_matches(message, child) for child in predicate.children]
        return any(values) if predicate.op == "or" else all(values)
    return False


def _block_matches(message: Message, predicate: QueryPredicate) -> bool:
    blocks = getattr(message, "blocks", ())
    if isinstance(predicate, QueryFieldPredicate):
        field = predicate.field.removeprefix("block.")
        return any(_unit_field(block.get(field), predicate) for block in blocks if isinstance(block, dict))
    if isinstance(predicate, QueryTextPredicate):
        needle = predicate.text.lower()
        return any(needle in str(block).lower() for block in blocks)
    if isinstance(predicate, QueryNotPredicate):
        return not _block_matches(message, predicate.child)
    if isinstance(predicate, QueryBoolPredicate):
        values = [_block_matches(message, child) for child in predicate.children]
        return any(values) if predicate.op == "or" else all(values)
    return False


def _bind_action_predicate(predicate: QueryPredicate) -> QueryPredicate:
    """Give the shared action matcher the same closed field identity as SQL."""
    if isinstance(predicate, QueryFieldPredicate):
        return predicate.with_field_ref(
            QueryFieldRef(scope="unit", name=predicate.field, source_name=predicate.field, unit="action")
        )
    if isinstance(predicate, QueryNotPredicate):
        return QueryNotPredicate(_bind_action_predicate(predicate.child))
    if isinstance(predicate, QueryBoolPredicate):
        return QueryBoolPredicate(predicate.op, tuple(_bind_action_predicate(child) for child in predicate.children))
    return predicate


def evaluate_predicate(
    session: Session,
    predicate: QueryPredicate,
    *,
    lineage: set[str] | None = None,
    _unit: str = "session",
) -> bool:
    """Evaluate one production :class:`QueryPredicate` without re-parsing it."""
    if isinstance(predicate, QueryFieldPredicate):
        return _field(session, predicate)
    if isinstance(predicate, QueryTextPredicate):
        return predicate.text.lower() in _text(session).lower()
    if isinstance(predicate, QuerySemanticPredicate):
        raise NotImplementedError("semantic predicates need a vector oracle")
    if isinstance(predicate, QueryNotPredicate):
        return not evaluate_predicate(session, predicate.child, lineage=lineage, _unit=_unit)
    if isinstance(predicate, QueryBoolPredicate):
        results = [evaluate_predicate(session, child, lineage=lineage, _unit=_unit) for child in predicate.children]
        return all(results) if predicate.op == "and" else any(results)
    if isinstance(predicate, QueryExistsPredicate):
        if predicate.unit == "message":
            return any(_message_matches(message, predicate.child) for message in session.messages)
        if predicate.unit == "block":
            return any(_block_matches(message, predicate.child) for message in session.messages)
        if predicate.unit == "action":
            return matches_action_predicate_sequence((_bind_action_predicate(predicate.child),), session)
        return False
    if isinstance(predicate, QueryLineagePredicate):
        return lineage is not None and predicate.seed_session_id in lineage
    if isinstance(predicate, QuerySequencePredicate):
        steps = tuple(_bind_action_predicate(step) for step in predicate.steps)
        return matches_action_predicate_sequence(steps, session, predicate.constraints)
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
            effective = _recomposed_session(self, session)
            if ast.boolean_predicate is not None:
                matches = evaluate_predicate(effective, ast.boolean_predicate, lineage=lineage)
            else:
                matches = all(_compact_matches(effective, token) for token in ast.clauses)
            if matches:
                selected.append(effective)
        selected.sort(key=lambda item: str(item.id))
        origins = sorted(str(session.origin) for session in selected)
        facets = tuple((origin, origins.count(origin)) for origin in sorted(set(origins)))
        costs = tuple((str(session.id), *harmonize_session_cost(session)) for session in selected)
        return ReferenceResult(tuple(str(item.id) for item in selected), len(selected), facets, costs)


__all__ = ["ReferenceArchive", "ReferenceResult", "evaluate_predicate"]
