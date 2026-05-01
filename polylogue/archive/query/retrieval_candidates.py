"""Candidate selection and action-read-model readiness for query retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from polylogue.archive.query.support import provider_values

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation, ConversationSummary
    from polylogue.archive.query.plan import ConversationQueryPlan
    from polylogue.protocols import ConversationQueryRuntimeStore
    from polylogue.storage.action_events.artifacts import ActionEventArtifactState
    from polylogue.storage.query_models import ConversationRecordQuery


def candidate_record_query(plan: ConversationQueryPlan) -> tuple[ConversationRecordQuery, bool]:
    record_query = plan.record_query
    return record_query.without_unstable_semantic_filters(), plan.sql_pushed


async def candidate_record_query_for(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> tuple[ConversationRecordQuery, bool]:
    if await action_event_rows_ready(plan, repository):
        return plan.record_query, plan.sql_pushed
    return plan.record_query.without_unstable_semantic_filters(), False


def uses_action_read_model(plan: ConversationQueryPlan) -> bool:
    return bool(
        plan.referenced_path
        or plan.action_terms
        or plan.excluded_action_terms
        or plan.tool_terms
        or plan.excluded_tool_terms
        or plan.action_text_terms
        or plan.retrieval_lane in {"actions", "hybrid"}
    )


async def _action_event_state(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> ActionEventArtifactState | None:
    if not uses_action_read_model(plan):
        return None
    return await repository.get_action_event_artifact_state()


async def action_event_rows_ready(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> bool:
    state = await _action_event_state(plan, repository)
    return True if state is None else state.rows_ready


async def action_search_ready(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> bool:
    state = await _action_event_state(plan, repository)
    return True if state is None else state.ready


async def can_use_action_event_stats_with(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
) -> bool:
    return plan.can_use_action_event_stats() and await action_event_rows_ready(plan, repository)


async def fetch_record_query_for(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
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


@overload
async def fetch_direct_id(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> list[Conversation]: ...


@overload
async def fetch_direct_id(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> list[ConversationSummary]: ...


async def fetch_direct_id(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: bool,
) -> list[Conversation] | list[ConversationSummary]:
    if not plan.conversation_id or plan.fts_terms:
        return []
    resolved_id = await repository.resolve_id(plan.conversation_id)
    if not resolved_id:
        return []
    if summaries:
        summary = await repository.get_summary(str(resolved_id))
        return [summary] if summary is not None else []
    conversation = await repository.get(str(resolved_id))
    return [conversation] if conversation is not None else []


@overload
async def fetch_search_results(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> tuple[bool, list[Conversation]]: ...


@overload
async def fetch_search_results(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> tuple[bool, list[ConversationSummary]]: ...


async def fetch_search_results(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: bool,
) -> tuple[bool, list[Conversation] | list[ConversationSummary]]:
    if not plan.fts_terms:
        return False, []
    if plan.retrieval_lane == "actions":
        if summaries:
            return False, []
        from polylogue.archive.query.retrieval_search import search_action_results

        return True, await search_action_results(plan, repository, limit=search_limit(plan))
    if plan.retrieval_lane == "hybrid":
        if summaries:
            return False, []
        from polylogue.archive.query.retrieval_search import search_hybrid_results

        return True, await search_hybrid_results(plan, repository, limit=search_limit(plan))

    query = " ".join(plan.fts_terms)
    provider_names = list(provider_values(plan.providers)) or None
    if summaries:
        summary_results = await repository.search_summaries(
            query,
            limit=search_limit(plan),
            providers=provider_names,
        )
        return True, summary_results
    conversation_results = await repository.search(
        query,
        limit=search_limit(plan),
        providers=provider_names,
    )
    return True, conversation_results


@overload
async def fetch_candidates(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> tuple[list[Conversation], bool]: ...


@overload
async def fetch_candidates(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> tuple[list[ConversationSummary], bool]: ...


async def fetch_candidates(
    plan: ConversationQueryPlan,
    repository: ConversationQueryRuntimeStore,
    *,
    summaries: bool,
) -> tuple[list[Conversation] | list[ConversationSummary], bool]:
    if summaries:
        direct_summaries = await fetch_direct_id(plan, repository, summaries=True)
        if direct_summaries:
            return direct_summaries, False

        used_search, summary_search_results = await fetch_search_results(plan, repository, summaries=True)
        if used_search:
            return summary_search_results, False

        request, sql_pushed = await candidate_record_query_for(plan, repository)
        request = request.with_limit(plan.effective_fetch_limit())
        return await repository.list_summaries_by_query(request), sql_pushed
    direct_conversations = await fetch_direct_id(plan, repository, summaries=False)
    if direct_conversations:
        return direct_conversations, False

    used_search, conversation_search_results = await fetch_search_results(plan, repository, summaries=False)
    if used_search:
        return conversation_search_results, False

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    request = request.with_limit(plan.effective_fetch_limit())
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
