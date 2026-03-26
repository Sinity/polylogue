"""Plan-level runtime capability helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan


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


__all__ = [
    "plan_can_count_in_sql",
    "plan_can_use_action_event_stats",
    "plan_has_post_filters",
    "plan_needs_content_loading",
]
