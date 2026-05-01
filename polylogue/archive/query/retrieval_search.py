"""Search and batched-hydration helpers for query retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING

from polylogue.archive.query.support import provider_values
from polylogue.logging import get_logger
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.archive.query.plan import ConversationQueryPlan
    from polylogue.lib.models import Conversation
    from polylogue.protocols import ConversationQueryRuntimeStore


def search_query_text(plan: ConversationQueryPlan) -> str:
    return " ".join(term.strip() for term in plan.fts_terms if term.strip()).strip()


def search_query_terms(plan: ConversationQueryPlan) -> tuple[str, ...]:
    return tuple(term.lower() for term in plan.fts_terms if term.strip())


def score_action_search_text(
    search_text: str,
    *,
    query_text: str,
    terms: tuple[str, ...],
) -> float:
    haystack = search_text.lower()
    score = 0.0
    if query_text and query_text.lower() in haystack:
        score += 20.0 + len(query_text.split())
    for term in terms:
        if term in haystack:
            score += 4.0
    return score


def conversation_action_search_score(
    conversation: Conversation,
    *,
    query_text: str,
    terms: tuple[str, ...],
) -> float:
    from polylogue.lib.semantic.facts import build_conversation_semantic_facts

    facts = build_conversation_semantic_facts(conversation)
    matches = [
        score_action_search_text(action.search_text, query_text=query_text, terms=terms)
        for action in facts.action_events
        if action.search_text
    ]
    positive = [score for score in matches if score > 0]
    if not positive:
        return 0.0
    return max(positive) + min(len(positive) - 1, 5)


async def search_action_results_fallback(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    limit: int,
) -> list[Conversation]:
    from polylogue.archive.query.retrieval_candidates import (
        candidate_batch_limit,
        candidate_record_query_for,
    )
    from polylogue.archive.query.runtime import apply_common_filters

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    batch_limit = candidate_batch_limit(plan)
    offset = 0
    query_text = search_query_text(plan)
    terms = search_query_terms(plan)
    ranked: list[tuple[float, int, Conversation]] = []
    counter = 0

    while True:
        batch = await repository.list_by_query(request.with_limit(batch_limit).with_offset(offset))
        if not batch:
            break
        for conversation in apply_common_filters(plan, batch, sql_pushed=sql_pushed):
            score = conversation_action_search_score(conversation, query_text=query_text, terms=terms)
            if score <= 0:
                continue
            entry = (score, counter, conversation)
            counter += 1
            if len(ranked) < limit:
                heappush(ranked, entry)
            else:
                heappushpop(ranked, entry)
        if len(batch) < batch_limit:
            break
        offset += batch_limit

    ranked.sort(
        key=lambda item: (
            item[0],
            item[2].updated_at or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return [conversation for _score, _counter, conversation in ranked]


async def search_action_results(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    limit: int,
) -> list[Conversation]:
    from polylogue.archive.query.retrieval_candidates import action_search_ready

    query = search_query_text(plan)
    provider_names = list(provider_values(plan.providers)) or None
    if not await action_search_ready(plan, repository):
        return await search_action_results_fallback(plan, repository, limit=limit)
    try:
        return await repository.search_actions(query, limit=limit, providers=provider_names)
    except Exception as exc:
        logger.warning(
            "action search failed; falling back to slower path",
            error=str(exc),
            error_type=type(exc).__name__,
            query=query,
        )
        return await search_action_results_fallback(plan, repository, limit=limit)


async def search_hybrid_results(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    limit: int,
) -> list[Conversation]:
    query = search_query_text(plan)
    provider_names = list(provider_values(plan.providers)) or None
    text_results = await repository.search(query, limit=limit * 3, providers=provider_names)
    action_results = await search_action_results(plan, repository, limit=limit * 3)
    vector_results: list[Conversation] = []
    if plan.vector_provider is not None:
        try:
            vector_results = await repository.search_similar(
                query,
                limit=limit * 3,
                vector_provider=plan.vector_provider,
            )
        except Exception as exc:
            logger.warning(
                "vector search failed; hybrid results will skip vector lane",
                error=str(exc),
                error_type=type(exc).__name__,
                vector_provider=plan.vector_provider,
            )
            vector_results = []

    text_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(text_results, start=1)]
    action_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(action_results, start=1)]
    vector_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(vector_results, start=1)]
    fused_ids = [
        conversation_id for conversation_id, _score in reciprocal_rank_fusion(text_ranked, action_ranked, vector_ranked)
    ][:limit]

    text_by_id = {str(conversation.id): conversation for conversation in text_results}
    action_by_id = {str(conversation.id): conversation for conversation in action_results}
    vector_by_id = {str(conversation.id): conversation for conversation in vector_results}
    ordered: list[Conversation] = []
    for conversation_id in fused_ids:
        conversation = (
            action_by_id.get(conversation_id) or text_by_id.get(conversation_id) or vector_by_id.get(conversation_id)
        )
        if conversation is not None:
            ordered.append(conversation)
    return ordered


async def fetch_batched_filtered_conversations(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> list[Conversation]:
    from polylogue.archive.query.retrieval_candidates import (
        candidate_batch_limit,
        candidate_record_query_for,
    )
    from polylogue.archive.query.runtime import apply_full_filters

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    batch_limit = candidate_batch_limit(plan)
    offset = 0
    matched: list[Conversation] = []
    seen_ids: set[str] = set()

    while True:
        batch = await repository.list_by_query(request.with_limit(batch_limit).with_offset(offset))
        if not batch:
            break
        filtered_batch = apply_full_filters(plan, batch, sql_pushed=sql_pushed)
        for conversation in filtered_batch:
            conversation_id = str(conversation.id)
            if conversation_id in seen_ids:
                continue
            seen_ids.add(conversation_id)
            matched.append(conversation)
        if plan.limit is not None and len(matched) >= plan.limit:
            break
        if len(batch) < batch_limit:
            break
        offset += batch_limit

    return matched


__all__ = [
    "conversation_action_search_score",
    "fetch_batched_filtered_conversations",
    "score_action_search_text",
    "search_action_results",
    "search_hybrid_results",
    "search_query_terms",
    "search_query_text",
]
