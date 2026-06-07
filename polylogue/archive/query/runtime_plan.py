"""Plan-level runtime capability helpers for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.query.fields import has_message_content_type_filter, plan_has_fields_matching

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan


def plan_has_post_filters(plan: SessionQueryPlan) -> bool:
    return plan_has_fields_matching(plan, lambda descriptor: descriptor.requires_post_filter)


def plan_needs_content_loading(plan: SessionQueryPlan) -> bool:
    if plan.fts_terms and plan.retrieval_lane in {"actions", "hybrid"}:
        return True
    if has_message_content_type_filter(plan):
        return True
    if plan_has_fields_matching(plan, lambda descriptor: descriptor.requires_content_loading):
        return True
    return plan.sort in ("messages", "words", "longest", "tokens")


def plan_can_count_in_sql(plan: SessionQueryPlan) -> bool:
    return not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_sql_count)


def plan_can_use_action_stats(plan: SessionQueryPlan) -> bool:
    return not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_action_stats)


__all__ = [
    "plan_can_count_in_sql",
    "plan_can_use_action_stats",
    "plan_has_post_filters",
    "plan_needs_content_loading",
]
