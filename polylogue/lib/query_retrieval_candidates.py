"""Candidate selection and action-read-model readiness for query retrieval."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from polylogue.lib.query_support import provider_values

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
        from polylogue.lib.query_retrieval_search import search_action_results

        return True, await search_action_results(plan, repository, limit=search_limit(plan))
    if plan.retrieval_lane == "hybrid":
        if summaries:
            return False, []
        from polylogue.lib.query_retrieval_search import search_hybrid_results

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


__all__ = [
    "action_event_rows_ready",
    "action_search_ready",
    "can_use_action_event_stats_with",
    "candidate_batch_limit",
    "candidate_record_query",
    "candidate_record_query_for",
    "fetch_candidates",
    "fetch_direct_id",
    "fetch_record_query_for",
    "fetch_search_results",
    "search_limit",
    "should_batch_post_filter_fetch",
    "uses_action_read_model",
]
