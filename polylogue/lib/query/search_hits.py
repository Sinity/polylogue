"""Evidence-bearing search-hit execution over canonical query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.query.fields import plan_has_fields_matching
from polylogue.lib.query.retrieval import search_limit
from polylogue.lib.query.retrieval_search import search_query_text
from polylogue.lib.query.support import provider_values
from polylogue.lib.search_hits import (
    ConversationSearchHit,
    conversation_search_hit_from_conversation,
)

if TYPE_CHECKING:
    from polylogue.lib.query.plan import ConversationQueryPlan
    from polylogue.protocols import ConversationQueryRuntimeStore


def plan_has_search_hit_evidence(plan: ConversationQueryPlan) -> bool:
    return bool(plan.fts_terms or plan.similar_text)


def _simple_message_hit_plan(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.fts_terms
        and plan.retrieval_lane in {"auto", "dialogue"}
        and not plan_has_fields_matching(plan, lambda descriptor: descriptor.blocks_simple_message_hit)
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
