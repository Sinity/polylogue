"""Plan-level runtime capability helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_fields import has_message_content_type_filter, plan_has_fields_matching

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan


def plan_has_post_filters(plan: ConversationQueryPlan) -> bool:
    return plan_has_fields_matching(plan, lambda descriptor: descriptor.requires_post_filter)


def plan_needs_content_loading(plan: ConversationQueryPlan) -> bool:
    if plan.fts_terms and plan.retrieval_lane in {"actions", "hybrid"}:
        return True
    if has_message_content_type_filter(plan):
        return True
    if plan_has_fields_matching(plan, lambda descriptor: descriptor.requires_content_loading):
        return True
    return plan.sort in ("messages", "words", "longest", "tokens")


def plan_can_count_in_sql(plan: ConversationQueryPlan) -> bool:
    return not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_sql_count)


def plan_can_use_action_event_stats(plan: ConversationQueryPlan) -> bool:
    return not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_action_event_stats)


__all__ = [
    "plan_can_count_in_sql",
    "plan_can_use_action_event_stats",
    "plan_has_post_filters",
    "plan_needs_content_loading",
]
