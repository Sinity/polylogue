"""Description and sizing helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_support import provider_values

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan


def describe_plan(plan: ConversationQueryPlan) -> list[str]:
    parts: list[str] = []
    if plan.fts_terms:
        parts.append(f"contains: {', '.join(plan.fts_terms)}")
    if plan.negative_terms:
        parts.append(f"exclude text: {', '.join(plan.negative_terms)}")
    if plan.retrieval_lane != "auto":
        parts.append(f"retrieval: {plan.retrieval_lane}")
    if plan.path_terms:
        parts.append(f"path: {', '.join(plan.path_terms)}")
    if plan.action_terms:
        parts.append(f"action: {', '.join(plan.action_terms)}")
    if plan.excluded_action_terms:
        parts.append(f"exclude action: {', '.join(plan.excluded_action_terms)}")
    if plan.action_sequence:
        parts.append(f"action sequence: {' -> '.join(plan.action_sequence)}")
    if plan.action_text_terms:
        parts.append(f"action text: {', '.join(plan.action_text_terms)}")
    if plan.tool_terms:
        parts.append(f"tool: {', '.join(plan.tool_terms)}")
    if plan.excluded_tool_terms:
        parts.append(f"exclude tool: {', '.join(plan.excluded_tool_terms)}")
    if plan.providers:
        parts.append(f"provider: {', '.join(provider_values(plan.providers))}")
    if plan.excluded_providers:
        parts.append(f"exclude provider: {', '.join(provider_values(plan.excluded_providers))}")
    if plan.tags:
        parts.append(f"tag: {', '.join(plan.tags)}")
    if plan.excluded_tags:
        parts.append(f"exclude tag: {', '.join(plan.excluded_tags)}")
    if plan.title:
        parts.append(f"title: {plan.title}")
    if plan.has_types:
        parts.append(f"has: {', '.join(plan.has_types)}")
    if plan.filter_has_tool_use:
        parts.append("has_tool_use")
    if plan.filter_has_thinking:
        parts.append("has_thinking")
    if plan.min_messages is not None:
        parts.append(f"min_messages: {plan.min_messages}")
    if plan.max_messages is not None:
        parts.append(f"max_messages: {plan.max_messages}")
    if plan.min_words is not None:
        parts.append(f"min_words: {plan.min_words}")
    if plan.since:
        parts.append(f"since: {plan.since.isoformat()}")
    if plan.until:
        parts.append(f"until: {plan.until.isoformat()}")
    if plan.conversation_id:
        parts.append(f"id: {plan.conversation_id}")
    if plan.parent_id:
        parts.append(f"parent: {plan.parent_id}")
    if plan.continuation is True:
        parts.append("continuation")
    if plan.continuation is False:
        parts.append("not continuation")
    if plan.sidechain is True:
        parts.append("sidechain")
    if plan.sidechain is False:
        parts.append("not sidechain")
    if plan.root is True:
        parts.append("root")
    if plan.root is False:
        parts.append("not root")
    if plan.has_branches is True:
        parts.append("has branches")
    if plan.has_branches is False:
        parts.append("no branches")
    if plan.predicates:
        parts.append(f"custom predicates: {len(plan.predicates)}")
    if plan.similar_text:
        parts.append(f"similar: {plan.similar_text[:30]}")
    return parts


def plan_has_filters(plan: ConversationQueryPlan) -> bool:
    return any(
        (
            plan.fts_terms,
            plan.negative_terms,
            plan.path_terms,
            plan.action_terms,
            plan.excluded_action_terms,
            plan.action_sequence,
            plan.action_text_terms,
            plan.tool_terms,
            plan.excluded_tool_terms,
            plan.providers,
            plan.excluded_providers,
            plan.tags,
            plan.excluded_tags,
            plan.has_types,
            plan.title is not None,
            plan.conversation_id is not None,
            plan.parent_id is not None,
            plan.since is not None,
            plan.until is not None,
            plan.similar_text is not None,
            plan.continuation is not None,
            plan.sidechain is not None,
            plan.root is not None,
            plan.has_branches is not None,
            plan.filter_has_tool_use,
            plan.filter_has_thinking,
            plan.min_messages is not None,
            plan.max_messages is not None,
            plan.min_words is not None,
            plan.predicates,
        )
    )


def effective_fetch_limit(plan: ConversationQueryPlan) -> int | None:
    if plan.limit is None:
        return None
    if plan.has_post_filters():
        return max(plan.limit * 10, 500)
    if plan.sample is not None:
        return max(plan.sample * 3, 200)
    return max(plan.limit * 2, 2)


__all__ = [
    "describe_plan",
    "effective_fetch_limit",
    "plan_has_filters",
]
