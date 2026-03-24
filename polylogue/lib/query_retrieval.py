"""Retrieval and candidate-selection helpers for immutable conversation query plans."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING

from polylogue.lib.query_support import provider_values
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.lib.query_plan import ConversationQueryPlan
    from polylogue.storage.query_models import ConversationRecordQuery
    from polylogue.storage.repository import ConversationRepository


def candidate_record_query(plan: ConversationQueryPlan) -> tuple[ConversationRecordQuery, bool]:
    record_query = plan.record_query
    return record_query.without_unstable_semantic_filters(), plan.sql_pushed


async def candidate_record_query_for(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> tuple[ConversationRecordQuery, bool]:
    if await action_event_rows_ready(plan, repository):
        return plan.record_query, plan.sql_pushed
    return plan.record_query.without_unstable_semantic_filters(), False


def uses_action_read_model(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.path_terms
        or plan.action_terms
        or plan.excluded_action_terms
        or plan.tool_terms
        or plan.excluded_tool_terms
        or plan.action_text_terms
        or plan.retrieval_lane in {"actions", "hybrid"}
    )


async def action_event_rows_ready(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> bool:
    if not uses_action_read_model(plan):
        return True
    status_reader = getattr(repository, "get_action_event_read_model_status", None)
    if status_reader is None:
        return True
    status = status_reader()
    if inspect.isawaitable(status):
        status = await status
    if not isinstance(status, dict):
        return True
    return bool(status.get("rows_ready", status.get("ready", False)))


async def action_search_ready(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> bool:
    if not uses_action_read_model(plan):
        return True
    status_reader = getattr(repository, "get_action_event_read_model_status", None)
    if status_reader is None:
        return True
    status = status_reader()
    if inspect.isawaitable(status):
        status = await status
    if not isinstance(status, dict):
        return True
    return bool(status.get("ready", False))


async def can_use_action_event_stats_with(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> bool:
    return plan.can_use_action_event_stats() and await action_event_rows_ready(plan, repository)


async def fetch_record_query_for(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> ConversationRecordQuery:
    record_query, _ = await candidate_record_query_for(plan, repository)
    return record_query.with_limit(plan.effective_fetch_limit())


def should_batch_post_filter_fetch(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.limit is not None
        and plan.limit > 0
        and plan.has_post_filters()
        and not plan.fts_terms
        and plan.conversation_id is None
        and plan.sample is None
        and plan.sort == "date"
        and not plan.reverse
    )


def candidate_batch_limit(plan: ConversationQueryPlan) -> int:
    if plan.limit is None:
        return 100
    return min(max(plan.limit * 2, 100), 200)


def search_limit(plan: ConversationQueryPlan) -> int:
    fetch_limit = plan.effective_fetch_limit()
    return max(fetch_limit, 100) if fetch_limit is not None else 10000


async def fetch_direct_id(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
    *,
    summaries: bool,
) -> list[Conversation | ConversationSummary]:
    if not plan.conversation_id or plan.fts_terms:
        return []
    resolved_id = await repository.resolve_id(plan.conversation_id)
    if not resolved_id:
        return []
    if summaries:
        item = await repository.get_summary(str(resolved_id))
    else:
        item = await repository.get(str(resolved_id))
    return [item] if item is not None else []


async def fetch_search_results(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
    *,
    summaries: bool,
) -> tuple[bool, list[Conversation | ConversationSummary]]:
    if not plan.fts_terms:
        return False, []
    if plan.retrieval_lane == "actions":
        if summaries:
            return False, []
        return True, await search_action_results(plan, repository, limit=search_limit(plan))
    if plan.retrieval_lane == "hybrid":
        if summaries:
            return False, []
        return True, await search_hybrid_results(plan, repository, limit=search_limit(plan))
    query = " ".join(plan.fts_terms)
    provider_names = list(provider_values(plan.providers)) or None
    try:
        if summaries:
            results = await repository.search_summaries(query, limit=search_limit(plan), providers=provider_names)
        else:
            results = await repository.search(query, limit=search_limit(plan), providers=provider_names)
        return True, results
    except Exception:
        return False, []


async def fetch_candidates(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
    *,
    summaries: bool,
) -> tuple[list[Conversation | ConversationSummary], bool]:
    direct = await fetch_direct_id(plan, repository, summaries=summaries)
    if direct:
        return direct, False

    used_search, search_results = await fetch_search_results(plan, repository, summaries=summaries)
    if used_search:
        return search_results, False

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    request = request.with_limit(plan.effective_fetch_limit())
    if summaries:
        return await repository.list_summaries_by_query(request), sql_pushed
    return await repository.list_by_query(request), sql_pushed


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
    from polylogue.lib.semantic_facts import build_conversation_semantic_facts

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
    repository: ConversationRepository,
    *,
    limit: int,
) -> list[Conversation]:
    from polylogue.lib.query_runtime import apply_common_filters

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
    repository: ConversationRepository,
    *,
    limit: int,
) -> list[Conversation]:
    query = search_query_text(plan)
    provider_names = list(provider_values(plan.providers)) or None
    if not await action_search_ready(plan, repository):
        return await search_action_results_fallback(plan, repository, limit=limit)
    try:
        return await repository.search_actions(query, limit=limit, providers=provider_names)
    except Exception:
        return await search_action_results_fallback(plan, repository, limit=limit)


async def search_hybrid_results(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
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
        except Exception:
            vector_results = []

    text_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(text_results, start=1)]
    action_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(action_results, start=1)]
    vector_ranked = [(str(conversation.id), float(rank)) for rank, conversation in enumerate(vector_results, start=1)]
    fused_ids = [
        conversation_id
        for conversation_id, _score in reciprocal_rank_fusion(text_ranked, action_ranked, vector_ranked)
    ][:limit]

    text_by_id = {str(conversation.id): conversation for conversation in text_results}
    action_by_id = {str(conversation.id): conversation for conversation in action_results}
    vector_by_id = {str(conversation.id): conversation for conversation in vector_results}
    ordered: list[Conversation] = []
    for conversation_id in fused_ids:
        conversation = (
            action_by_id.get(conversation_id)
            or text_by_id.get(conversation_id)
            or vector_by_id.get(conversation_id)
        )
        if conversation is not None:
            ordered.append(conversation)
    return ordered


async def fetch_batched_filtered_conversations(
    plan: ConversationQueryPlan,
    repository: ConversationRepository,
) -> list[Conversation]:
    from polylogue.lib.query_runtime import apply_full_filters

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    batch_limit = candidate_batch_limit(plan)
    offset = 0
    matched: list[Conversation] = []
    seen_ids: set[str] = set()

    while True:
        batch = await repository.list_by_query(
            request.with_limit(batch_limit).with_offset(offset)
        )
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
    "action_event_rows_ready",
    "action_search_ready",
    "can_use_action_event_stats_with",
    "candidate_record_query",
    "candidate_record_query_for",
    "candidate_batch_limit",
    "fetch_batched_filtered_conversations",
    "fetch_candidates",
    "fetch_record_query_for",
    "search_action_results",
    "search_hybrid_results",
    "search_limit",
    "should_batch_post_filter_fetch",
    "uses_action_read_model",
]
