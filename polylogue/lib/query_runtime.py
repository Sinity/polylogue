"""Runtime semantic filtering helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from polylogue.lib.query_support import conversation_has_branches, provider_values

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.lib.query_execution import ConversationQueryPlan

_T = TypeVar("_T")


def plan_has_post_filters(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.excluded_providers
        or plan.tags
        or plan.excluded_tags
        or plan.has_types
        or plan.predicates
        or plan.negative_terms
        or plan.path_terms
        or plan.action_terms
        or plan.excluded_action_terms
        or plan.action_sequence
        or plan.action_text_terms
        or plan.tool_terms
        or plan.excluded_tool_terms
        or plan.continuation is not None
        or plan.sidechain is not None
        or plan.root is not None
        or plan.has_branches is not None
    )


def plan_needs_content_loading(plan: ConversationQueryPlan) -> bool:
    if plan.fts_terms and plan.retrieval_lane in {"actions", "hybrid"}:
        return True
    if plan.has_types and any(kind in ("thinking", "tools", "attachments") for kind in plan.has_types):
        return True
    if plan.negative_terms or plan.predicates or plan.similar_text:
        return True
    if plan.path_terms or plan.action_terms or plan.excluded_action_terms:
        return True
    if plan.tool_terms or plan.excluded_tool_terms:
        return True
    if plan.action_sequence:
        return True
    if plan.action_text_terms:
        return True
    if plan.has_branches is not None:
        return True
    return plan.sort in ("messages", "words", "longest", "tokens")


def plan_can_count_in_sql(plan: ConversationQueryPlan) -> bool:
    return not (
        plan.fts_terms
        or plan.conversation_id
        or plan.similar_text
        or plan.path_terms
        or plan.action_terms
        or plan.excluded_action_terms
        or plan.tool_terms
        or plan.excluded_tool_terms
        or plan.action_sequence
        or plan.action_text_terms
        or plan.predicates
        or plan.has_types
        or plan.negative_terms
        or plan.excluded_providers
        or plan.tags
        or plan.excluded_tags
        or plan.continuation is not None
        or plan.sidechain is not None
        or plan.root is not None
        or plan.has_branches is not None
    )


def plan_can_use_action_event_stats(plan: ConversationQueryPlan) -> bool:
    return not (
        plan.fts_terms
        or plan.negative_terms
        or plan.action_sequence
        or plan.action_text_terms
        or plan.predicates
        or plan.has_types
        or plan.tags
        or plan.excluded_tags
        or plan.excluded_providers
        or plan.continuation is not None
        or plan.sidechain is not None
        or plan.root is not None
        or plan.has_branches is not None
        or plan.similar_text
    )


def matches_path_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.path_terms:
        return True
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    affected_paths = tuple(path.lower() for action in facts.action_events for path in action.affected_paths)
    if not affected_paths:
        return False
    return all(
        any(term.lower().replace("\\", "/") in path for path in affected_paths)
        for term in plan.path_terms
    )


def matches_action_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.action_terms and not plan.excluded_action_terms:
        return True
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    categories = {action.kind.value for action in facts.action_events}
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
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    tool_names = {
        (action.tool_name or "unknown").strip().lower()
        for action in facts.action_events
    }
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
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    if not facts.action_events:
        return False

    index = 0
    target_count = len(plan.action_sequence)
    for action in facts.action_events:
        if action.kind.value != plan.action_sequence[index]:
            continue
        index += 1
        if index >= target_count:
            return True
    return False


def matches_action_text_terms(plan: ConversationQueryPlan, conversation: Conversation) -> bool:
    if not plan.action_text_terms:
        return True
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    searchable_events = [action.search_text.lower() for action in facts.action_events if action.search_text]
    if not searchable_events:
        return False
    return all(
        any(term.lower() in event_text for event_text in searchable_events)
        for term in plan.action_text_terms
    )


def apply_common_filters(
    plan: ConversationQueryPlan,
    items: list[_T],
    *,
    sql_pushed: bool,
) -> list[_T]:
    results = list(items)

    if not sql_pushed:
        provider_set = set(provider_values(plan.providers))
        if provider_set:
            results = [item for item in results if str(item.provider) in provider_set]
        if plan.since:
            results = [item for item in results if item.updated_at and item.updated_at >= plan.since]
        if plan.until:
            results = [item for item in results if item.updated_at and item.updated_at <= plan.until]
        if plan.title:
            lowered = plan.title.lower()
            results = [
                item
                for item in results
                if item.display_title and lowered in item.display_title.lower()
            ]
        if plan.parent_id:
            results = [item for item in results if str(item.parent_id or "") == plan.parent_id]

    if plan.excluded_providers:
        excluded = set(provider_values(plan.excluded_providers))
        results = [item for item in results if str(item.provider) not in excluded]
    if plan.tags:
        tag_set = set(plan.tags)
        results = [item for item in results if tag_set.intersection(item.tags)]
    if plan.excluded_tags:
        excluded_tags = set(plan.excluded_tags)
        results = [item for item in results if not excluded_tags.intersection(item.tags)]
    if plan.conversation_id:
        results = [item for item in results if str(item.id).startswith(plan.conversation_id)]
    if "summary" in plan.has_types:
        results = [item for item in results if item.summary]
    if plan.continuation is True:
        results = [item for item in results if item.is_continuation]
    if plan.continuation is False:
        results = [item for item in results if not item.is_continuation]
    if plan.sidechain is True:
        results = [item for item in results if item.is_sidechain]
    if plan.sidechain is False:
        results = [item for item in results if not item.is_sidechain]
    if plan.root is True:
        results = [item for item in results if item.is_root]
    if plan.root is False:
        results = [item for item in results if not item.is_root]

    return results


def apply_full_filters(
    plan: ConversationQueryPlan,
    conversations: list[Conversation],
    *,
    sql_pushed: bool,
) -> list[Conversation]:
    results = apply_common_filters(plan, conversations, sql_pushed=sql_pushed)

    if plan.has_types:
        for content_type in plan.has_types:
            if content_type == "thinking":
                results = [c for c in results if any(m.is_thinking for m in c.messages)]
            elif content_type == "tools":
                results = [c for c in results if any(m.is_tool_use for m in c.messages)]
            elif content_type == "attachments":
                results = [c for c in results if any(m.attachments for m in c.messages)]

    if plan.negative_terms:
        negative_terms = [term.lower() for term in plan.negative_terms]

        def _has_negative_term(conversation: Conversation) -> bool:
            for message in conversation.messages:
                if not message.text:
                    continue
                lowered = message.text.lower()
                for term in negative_terms:
                    if term in lowered:
                        return True
            return False

        results = [conversation for conversation in results if not _has_negative_term(conversation)]

    if plan.has_branches is True:
        results = [conversation for conversation in results if conversation_has_branches(conversation)]
    if plan.has_branches is False:
        results = [conversation for conversation in results if not conversation_has_branches(conversation)]

    for predicate in plan.predicates:
        results = [conversation for conversation in results if predicate(conversation)]

    if plan.path_terms:
        results = [conversation for conversation in results if matches_path_terms(plan, conversation)]
    if plan.action_terms or plan.excluded_action_terms:
        results = [conversation for conversation in results if matches_action_terms(plan, conversation)]
    if plan.action_sequence:
        results = [conversation for conversation in results if matches_action_sequence(plan, conversation)]
    if plan.action_text_terms:
        results = [conversation for conversation in results if matches_action_text_terms(plan, conversation)]
    if plan.tool_terms or plan.excluded_tool_terms:
        results = [conversation for conversation in results if matches_tool_terms(plan, conversation)]

    return results


__all__ = [
    "apply_common_filters",
    "apply_full_filters",
    "matches_action_sequence",
    "matches_action_terms",
    "matches_action_text_terms",
    "matches_path_terms",
    "matches_tool_terms",
    "plan_can_count_in_sql",
    "plan_can_use_action_event_stats",
    "plan_has_post_filters",
    "plan_needs_content_loading",
]
