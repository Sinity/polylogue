"""Semantic matching helpers for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryFieldPredicate,
    QueryNotPredicate,
    QueryPredicate,
)

if TYPE_CHECKING:
    from polylogue.archive.actions.actions import Action
    from polylogue.archive.models import Session
    from polylogue.archive.query.plan import SessionQueryPlan


def _actions_for(session: Session) -> tuple[Action, ...]:
    from polylogue.archive.semantic.facts import build_session_semantic_facts

    facts = build_session_semantic_facts(session)
    return facts.actions


def matches_referenced_path(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.referenced_path:
        return True
    affected_paths = tuple(
        path.lower().replace("\\", "/") for action in _actions_for(session) for path in action.affected_paths
    )
    if not affected_paths:
        return False
    return all(any(term.lower().replace("\\", "/") in path for path in affected_paths) for term in plan.referenced_path)


def matches_action_terms(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.action_terms and not plan.excluded_action_terms:
        return True
    categories = {action.kind.value for action in _actions_for(session)}
    required_terms = {term for term in plan.action_terms if term != "none"}
    if "none" in plan.action_terms and categories:
        return False
    if required_terms and not required_terms.issubset(categories):
        return False
    if "none" in plan.excluded_action_terms and not categories:
        return False
    return not ({term for term in plan.excluded_action_terms if term != "none"} & categories)


def matches_tool_terms(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.tool_terms and not plan.excluded_tool_terms:
        return True
    tool_names = {(action.tool_name or "unknown").strip().lower() for action in _actions_for(session)}
    required_terms = {term for term in plan.tool_terms if term != "none"}
    if "none" in plan.tool_terms and tool_names:
        return False
    if required_terms and not required_terms.issubset(tool_names):
        return False
    if "none" in plan.excluded_tool_terms and not tool_names:
        return False
    return not ({term for term in plan.excluded_tool_terms if term != "none"} & tool_names)


def matches_action_sequence(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.action_sequence:
        return True
    actions = _actions_for(session)
    if not actions:
        return False

    index = 0
    target_count = len(plan.action_sequence)
    for action in actions:
        if action.kind.value != plan.action_sequence[index]:
            continue
        index += 1
        if index >= target_count:
            return True
    return False


def matches_action_predicate_sequence(steps: tuple[QueryPredicate, ...], session: Session) -> bool:
    if not steps:
        return True
    actions = _actions_for(session)
    if not actions:
        return False

    index = 0
    target_count = len(steps)
    for action in actions:
        if not _matches_action_predicate(steps[index], action):
            continue
        index += 1
        if index >= target_count:
            return True
    return False


def _matches_action_predicate(predicate: QueryPredicate, action: Action) -> bool:
    if isinstance(predicate, QueryFieldPredicate):
        return _matches_action_field(predicate, action)
    if isinstance(predicate, QueryNotPredicate):
        return not _matches_action_predicate(predicate.child, action)
    if isinstance(predicate, QueryBoolPredicate):
        if predicate.op == "or":
            return any(_matches_action_predicate(child, action) for child in predicate.children)
        return all(_matches_action_predicate(child, action) for child in predicate.children)
    return False


def _matches_action_field(predicate: QueryFieldPredicate, action: Action) -> bool:
    values = tuple(value.strip().lower() for value in predicate.values if value.strip())
    if not values:
        return False
    if predicate.field in {"action", "type"}:
        return _matches_exact_values(action.kind.value, values)
    if predicate.field == "tool":
        return _matches_exact_values(action.normalized_tool_name, values)
    if predicate.field == "command":
        return _matches_text(action.command, values)
    if predicate.field == "path":
        normalized_paths = tuple(path.lower().replace("\\", "/") for path in action.affected_paths)
        return all(any(value.replace("\\", "/") in path for path in normalized_paths) for value in values)
    if predicate.field == "output":
        return _matches_text(action.output_text, values)
    if predicate.field == "text":
        return _matches_text(action.search_text, values)
    return False


def _matches_exact_values(value: str | None, expected: tuple[str, ...]) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in expected


def _matches_text(value: str | None, expected: tuple[str, ...]) -> bool:
    normalized = (value or "").lower().replace("\\", "/")
    return all(term.replace("\\", "/") in normalized for term in expected)


def matches_action_text_terms(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.action_text_terms:
        return True
    searchable_events = [action.search_text.lower() for action in _actions_for(session) if action.search_text]
    if not searchable_events:
        return False
    return all(any(term.lower() in event_text for event_text in searchable_events) for term in plan.action_text_terms)


__all__ = [
    "matches_action_predicate_sequence",
    "matches_action_sequence",
    "matches_action_terms",
    "matches_action_text_terms",
    "matches_referenced_path",
    "matches_tool_terms",
]
