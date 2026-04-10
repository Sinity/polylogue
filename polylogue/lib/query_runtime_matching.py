"""Semantic matching helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.action_events import ActionEvent
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_plan import ConversationQueryPlan


def _action_events_for(conversation: Conversation) -> tuple[ActionEvent, ...]:
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    return facts.action_events


def matches_path_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.path_terms:
        return True
    affected_paths = tuple(
        path.lower() for action in _action_events_for(conversation) for path in action.affected_paths
    )
    if not affected_paths:
        return False
    return all(any(term.lower().replace("\\", "/") in path for path in affected_paths) for term in plan.path_terms)


def matches_action_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.action_terms and not plan.excluded_action_terms:
        return True
    categories = {action.kind.value for action in _action_events_for(conversation)}
    required_terms = {term for term in plan.action_terms if term != "none"}
    if "none" in plan.action_terms and categories:
        return False
    if required_terms and not required_terms.issubset(categories):
        return False
    if "none" in plan.excluded_action_terms and not categories:
        return False
    return not ({term for term in plan.excluded_action_terms if term != "none"} & categories)


def matches_tool_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.tool_terms and not plan.excluded_tool_terms:
        return True
    tool_names = {(action.tool_name or "unknown").strip().lower() for action in _action_events_for(conversation)}
    required_terms = {term for term in plan.tool_terms if term != "none"}
    if "none" in plan.tool_terms and tool_names:
        return False
    if required_terms and not required_terms.issubset(tool_names):
        return False
    if "none" in plan.excluded_tool_terms and not tool_names:
        return False
    return not ({term for term in plan.excluded_tool_terms if term != "none"} & tool_names)


def matches_action_sequence(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.action_sequence:
        return True
    action_events = _action_events_for(conversation)
    if not action_events:
        return False

    index = 0
    target_count = len(plan.action_sequence)
    for action in action_events:
        if action.kind.value != plan.action_sequence[index]:
            continue
        index += 1
        if index >= target_count:
            return True
    return False


def matches_action_text_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.action_text_terms:
        return True
    searchable_events = [
        action.search_text.lower() for action in _action_events_for(conversation) if action.search_text
    ]
    if not searchable_events:
        return False
    return all(any(term.lower() in event_text for event_text in searchable_events) for term in plan.action_text_terms)


__all__ = [
    "matches_action_sequence",
    "matches_action_terms",
    "matches_action_text_terms",
    "matches_path_terms",
    "matches_tool_terms",
]
