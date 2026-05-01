"""Execution helpers for immutable conversation-query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.query.plan import ConversationQueryPlan
from polylogue.archive.query.retrieval import (
    fetch_batched_filtered_conversations,
    fetch_candidates,
)
from polylogue.archive.query.support import conversation_to_summary

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation, ConversationSummary
    from polylogue.protocols import ConversationQueryRuntimeStore


async def list_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> list[Conversation]:
    if plan.similar_text:
        candidates = await repository.search_similar(
            plan.similar_text,
            limit=plan.limit or 10,
            vector_provider=plan.vector_provider,
        )
        return plan._finalize(plan._apply_full_filters(candidates, sql_pushed=False))

    if plan._should_batch_post_filter_fetch():
        batched = await fetch_batched_filtered_conversations(plan, repository)
        return plan._finalize(plan._sort_conversations(batched))

    candidate_results, sql_pushed = await fetch_candidates(plan, repository, summaries=False)
    filtered = plan._apply_full_filters(candidate_results, sql_pushed=sql_pushed)
    return plan._finalize(plan._sort_conversations(filtered))


async def list_summaries_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> list[ConversationSummary]:
    can_use_action_stats = await plan.can_use_action_event_stats_with(repository)
    uses_action_read_model = plan._uses_action_read_model()
    if not plan.can_use_summaries() and not uses_action_read_model:
        msg = (
            "Cannot use list_summaries() with content-dependent filters "
            "(regex, has:thinking, has:tools, etc.). Use list() instead."
        )
        raise ValueError(msg)

    if uses_action_read_model and not can_use_action_stats:
        conversations = await list_for_plan(plan, repository)
        return [conversation_to_summary(conversation) for conversation in conversations]

    candidates, sql_pushed = await fetch_candidates(plan, repository, summaries=True)
    filtered = plan._apply_common_filters(candidates, sql_pushed=sql_pushed)
    return plan._finalize(plan._sort_summaries(filtered))


async def first_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> Conversation | None:
    results = await list_for_plan(plan.with_limit(1), repository)
    return results[0] if results else None


async def count_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> int:
    if plan.can_count_in_sql() and await plan._action_event_rows_ready(repository):
        return int(await repository.count_by_query(plan.record_query.for_count()))

    unbounded = plan.with_limit(None)
    if unbounded.can_use_summaries():
        return len(await list_summaries_for_plan(unbounded, repository))
    return len(await list_for_plan(unbounded, repository))


async def delete_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> int:
    results: list[Conversation] | list[ConversationSummary]
    if plan.can_use_summaries():
        results = await list_summaries_for_plan(plan, repository)
    else:
        results = await list_for_plan(plan, repository)

    deleted = 0
    for conversation in results:
        if await repository.delete_conversation(str(conversation.id)):
            deleted += 1
    return deleted


__all__ = [
    "count_for_plan",
    "delete_for_plan",
    "first_for_plan",
    "list_for_plan",
    "list_summaries_for_plan",
]
