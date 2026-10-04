"""Semantic matching helpers for immutable session query plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING

from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryFieldPredicate,
    QueryNotPredicate,
    QueryPredicate,
    QuerySequenceConstraint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from polylogue.archive.actions.actions import Action
    from polylogue.archive.models import Session
    from polylogue.archive.query.plan import SessionQueryPlan


def _actions_for(session: Session) -> tuple[Action, ...]:
    from polylogue.archive.semantic.facts import build_session_semantic_facts

    facts = build_session_semantic_facts(session)
    return facts.actions


def paths_match_referenced_terms(affected_paths: Iterable[str], terms: tuple[str, ...]) -> bool:
    """The shared referenced-path predicate: every term (AND) must appear in some path.

    This is the single source of truth for referenced-path matching semantics —
    every surface that filters by ``referenced_path`` (the substrate session-query
    filter, the CLI semantic-stats surface, ...) must route through this function
    so they agree on which sessions/actions match a multi-term ``--path`` filter.
    """
    if not terms:
        return True
    normalized_paths = tuple(path.lower().replace("\\", "/") for path in affected_paths)
    if not normalized_paths:
        return False
    return all(any(term.lower().replace("\\", "/") in path for path in normalized_paths) for term in terms)


def matches_referenced_path(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.referenced_path:
        return True
    affected_paths = tuple(path for action in _actions_for(session) for path in action.affected_paths)
    return paths_match_referenced_terms(affected_paths, plan.referenced_path)


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


@dataclass(frozen=True, slots=True)
class SequenceMatchWitness:
    """One ``seq()`` match, bound to the actions that produced it.

    ``action_indices`` are positions in the session's ordered action list, so
    the caller can resolve each step back to a concrete action rather than
    re-deriving which ones matched.
    """

    action_indices: tuple[int, ...]

    @property
    def span(self) -> tuple[int, int]:
        return self.action_indices[0], self.action_indices[-1]


@dataclass(frozen=True, slots=True)
class SequenceMatchWitnesses:
    """A one-pass completed-witness stream with a stable existential answer.

    Peek only the first completed witness, then put it back into the stream.
    Boolean matching never materializes the remaining all-pairs relation.
    """

    witnesses: Iterator[SequenceMatchWitness]
    _matched: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        first = next(self.witnesses, None)
        object.__setattr__(self, "_matched", first is not None)
        if first is not None:
            object.__setattr__(self, "witnesses", chain((first,), self.witnesses))

    def __bool__(self) -> bool:
        return self._matched


def action_predicate_sequence_witnesses(
    steps: tuple[QueryPredicate, ...],
    session: Session,
    constraints: tuple[QuerySequenceConstraint, ...] = (),
) -> SequenceMatchWitnesses:
    """Stream every ordered action tuple satisfying ``steps``.

    Multiplicity is all-pairs, matching the SQL lowering in
    ``archive_query_reads._action_sequence_witness_relation``. Retain only
    one path and its search cursors, rather than an intermediate frontier
    whose size can grow combinatorially. Existence comes from the same
    completed-witness stream.
    """
    if not steps:
        return SequenceMatchWitnesses(witnesses=iter((SequenceMatchWitness(action_indices=()),)))
    actions = _actions_for(session)
    edge_constraints = constraints or tuple(QuerySequenceConstraint() for _ in range(len(steps) - 1))
    if len(edge_constraints) != len(steps) - 1:
        raise ValueError("sequence constraints must describe every edge between steps")
    return SequenceMatchWitnesses(witnesses=_iter_sequence_witnesses(steps, actions, edge_constraints))


def _iter_sequence_witnesses(
    steps: tuple[QueryPredicate, ...],
    actions: tuple[Action, ...],
    constraints: tuple[QuerySequenceConstraint, ...],
) -> Iterator[SequenceMatchWitness]:
    # Iterative depth-first traversal also avoids a recursion-depth ceiling
    # on valid long patterns. Each depth retains just its next candidate.
    prefix: list[int] = []
    next_indices = [0]
    while next_indices:
        depth = len(prefix)
        current_index = next_indices[-1]
        end = len(actions)
        constraint = constraints[depth - 1] if depth else None
        if constraint is not None and constraint.kind == "next":
            end = min(end, prefix[-1] + 2)
        if current_index >= end:
            next_indices.pop()
            if prefix:
                prefix.pop()
            continue
        next_indices[-1] = current_index + 1
        if not _matches_action_predicate(steps[depth], actions[current_index]):
            continue
        if constraint is not None and constraint.kind == "within":
            previous_time = actions[prefix[-1]].timestamp
            current_time = actions[current_index].timestamp
            if previous_time is None or current_time is None:
                continue
            elapsed_ms = int((current_time - previous_time).total_seconds() * 1000)
            if elapsed_ms < 0 or constraint.within_ms is None or elapsed_ms > constraint.within_ms:
                continue
        prefix.append(current_index)
        if len(prefix) == len(steps):
            yield SequenceMatchWitness(action_indices=tuple(prefix))
            prefix.pop()
        else:
            next_indices.append(current_index + 1)


def matches_action_predicate_sequence(
    steps: tuple[QueryPredicate, ...],
    session: Session,
    constraints: tuple[QuerySequenceConstraint, ...] = (),
) -> bool:
    """Whether ``session`` contains the pattern.

    Derived from :func:`action_predicate_sequence_witnesses` rather than
    traversing again, so a reported match always has a witness behind it.
    """
    return bool(action_predicate_sequence_witnesses(steps, session, constraints))


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
    field = predicate.bound_field_name(context="matching action predicates")
    if field in {"action", "type"}:
        return _matches_exact_values(action.kind.value, values)
    if field == "tool":
        return _matches_exact_values(action.normalized_tool_name, values)
    if field == "command":
        return _matches_text(action.command, values)
    if field == "path":
        normalized_paths = tuple(path.lower().replace("\\", "/") for path in action.affected_paths)
        return any(value.replace("\\", "/") in path for value in values for path in normalized_paths)
    if field == "output":
        return _matches_text(action.output_text, values)
    if field == "text":
        return _matches_text(action.search_text, values)
    return False


def _matches_exact_values(value: str | None, expected: tuple[str, ...]) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in expected


def _matches_text(value: str | None, expected: tuple[str, ...]) -> bool:
    normalized = (value or "").lower().replace("\\", "/")
    return any(term.replace("\\", "/") in normalized for term in expected)


def matches_action_text_terms(plan: SessionQueryPlan, session: Session) -> bool:
    if not plan.action_text_terms:
        return True
    searchable_events = [action.search_text.lower() for action in _actions_for(session) if action.search_text]
    if not searchable_events:
        return False
    return all(any(term.lower() in event_text for event_text in searchable_events) for term in plan.action_text_terms)


__all__ = [
    "SequenceMatchWitness",
    "SequenceMatchWitnesses",
    "action_predicate_sequence_witnesses",
    "matches_action_predicate_sequence",
    "matches_action_sequence",
    "matches_action_terms",
    "matches_action_text_terms",
    "matches_referenced_path",
    "matches_tool_terms",
    "paths_match_referenced_terms",
]
