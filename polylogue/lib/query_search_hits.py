"""Evidence-bearing search-hit execution over canonical query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query_retrieval import search_limit
from polylogue.lib.query_retrieval_search import search_query_text
from polylogue.lib.query_support import provider_values
from polylogue.lib.search_hits import (
    ConversationSearchHit,
    conversation_search_hit_from_conversation,
)

if TYPE_CHECKING:
    from polylogue.lib.query_plan import ConversationQueryPlan
    from polylogue.protocols import ConversationQueryRuntimeStore


def plan_has_search_hit_evidence(plan: ConversationQueryPlan) -> bool:
    return bool(plan.fts_terms or plan.similar_text)


def _simple_message_hit_plan(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.fts_terms
        and plan.retrieval_lane in {"auto", "dialogue"}
        and plan.conversation_id is None
        and plan.similar_text is None
        and not plan.negative_terms
        and not plan.path_terms
        and not plan.action_terms
        and not plan.excluded_action_terms
        and not plan.action_sequence
        and not plan.action_text_terms
        and not plan.tool_terms
        and not plan.excluded_tool_terms
        and not plan.excluded_providers
        and not plan.tags
        and not plan.excluded_tags
        and not plan.has_types
        and plan.title is None
        and plan.until is None
        and plan.sample is None
        and not plan.filter_has_tool_use
        and not plan.filter_has_thinking
        and plan.min_messages is None
        and plan.max_messages is None
        and plan.min_words is None
    )


def _resolved_retrieval_lane(plan: ConversationQueryPlan) -> str:
    if plan.similar_text:
        return "semantic"
    if plan.retrieval_lane == "auto":
        return "dialogue"
    return plan.retrieval_lane


async def search_hits_for_plan(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> list[ConversationSearchHit]:
    """Return evidence-bearing hits for search-like query plans."""
    if not plan_has_search_hit_evidence(plan):
        return []

    query_text = plan.similar_text or search_query_text(plan)
    if not query_text:
        return []

    if _simple_message_hit_plan(plan):
        provider_names = list(provider_values(plan.providers)) or None
        limit = plan.limit or search_limit(plan)
        return await repository.search_summary_hits(
            query_text,
            limit=limit,
            providers=provider_names,
            since=plan.since.isoformat() if plan.since else None,
        )

    conversations = await plan.list(repository)
    retrieval_lane = _resolved_retrieval_lane(plan)
    query_terms = (query_text,)
    return [
        conversation_search_hit_from_conversation(
            conversation,
            query_terms=query_terms,
            rank=rank,
            retrieval_lane=retrieval_lane,
        )
        for rank, conversation in enumerate(conversations, start=1)
    ]


__all__ = ["plan_has_search_hit_evidence", "search_hits_for_plan"]
