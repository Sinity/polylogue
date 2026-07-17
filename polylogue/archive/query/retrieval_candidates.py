"""Candidate selection for query retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from polylogue.archive.query.support import _canonical_origins

if TYPE_CHECKING:
    from polylogue.archive.models import Session, SessionSummary
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.core.protocols import SessionQueryRuntimeStore
    from polylogue.storage.query_models import SessionRecordQuery


def candidate_record_query(plan: SessionQueryPlan) -> tuple[SessionRecordQuery, bool]:
    record_query = plan.record_query
    return record_query.without_unstable_semantic_filters(), plan.sql_pushed


async def candidate_record_query_for(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> tuple[SessionRecordQuery, bool]:
    del repository
    return plan.record_query, plan.sql_pushed


def uses_actions(plan: SessionQueryPlan) -> bool:
    return bool(
        plan.referenced_path
        or plan.action_terms
        or plan.excluded_action_terms
        or plan.tool_terms
        or plan.excluded_tool_terms
        or plan.action_text_terms
        or plan.retrieval_lane in {"actions", "hybrid"}
    )


async def actions_ready(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> bool:
    del plan, repository
    return True


async def action_search_ready(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> bool:
    del plan, repository
    return True


async def can_use_action_stats_with(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> bool:
    del repository
    return plan.can_use_action_stats()


async def fetch_record_query_for(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> SessionRecordQuery:
    record_query, _ = await candidate_record_query_for(plan, repository)
    return record_query.with_limit(plan.effective_fetch_limit())


def should_batch_post_filter_fetch(plan: SessionQueryPlan) -> bool:
    return bool(
        plan.limit is not None
        and plan.limit > 0
        and plan.has_post_filters()
        and not plan.fts_terms
        and plan.session_id is None
        and plan.sample is None
        and plan.sort == "date"
        and not plan.reverse
    )


def candidate_batch_limit(plan: SessionQueryPlan) -> int:
    if plan.limit is None:
        return 100
    return min(max(plan.limit * 2, 100), 200)


def search_limit(plan: SessionQueryPlan) -> int:
    # When no explicit --limit is set (e.g. the CLI query-first path with a
    # bare token), effective_fetch_limit() is None. Fall back to the shared
    # MAX_QUERY_LIMIT ceiling rather than an unbounded 10000 fetch (#1749).
    from polylogue.archive.query.spec import MAX_QUERY_LIMIT

    fetch_limit = plan.effective_fetch_limit()
    return max(fetch_limit, 100) if fetch_limit is not None else MAX_QUERY_LIMIT


@overload
async def fetch_direct_id(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> list[Session]: ...


@overload
async def fetch_direct_id(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> list[SessionSummary]: ...


async def fetch_direct_id(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: bool,
) -> list[Session] | list[SessionSummary]:
    if not plan.session_id or plan.fts_terms:
        return []
    resolved_id = await repository.resolve_id(plan.session_id)
    if not resolved_id:
        return []
    if summaries:
        summary = await repository.get_summary(str(resolved_id))
        return [summary] if summary is not None else []
    session = await repository.get(str(resolved_id))
    return [session] if session is not None else []


@overload
async def fetch_search_results(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> tuple[bool, list[Session]]: ...


@overload
async def fetch_search_results(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> tuple[bool, list[SessionSummary]]: ...


async def fetch_search_results(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: bool,
) -> tuple[bool, list[Session] | list[SessionSummary]]:
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

        results, _lane_ranks = await search_hybrid_results(plan, repository, limit=search_limit(plan))
        return True, results

    query = " ".join(plan.fts_terms)
    origins = _canonical_origins(plan.origins)
    if summaries:
        summary_results = await repository.search_summaries(
            query,
            limit=search_limit(plan),
            origins=origins,
        )
        return True, summary_results
    session_results = await repository.search(
        query,
        limit=search_limit(plan),
        origins=origins,
    )
    return True, session_results


@overload
async def fetch_candidates(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[False],
) -> tuple[list[Session], bool]: ...


@overload
async def fetch_candidates(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: Literal[True],
) -> tuple[list[SessionSummary], bool]: ...


async def fetch_candidates(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
    *,
    summaries: bool,
) -> tuple[list[Session] | list[SessionSummary], bool]:
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
    direct_sessions = await fetch_direct_id(plan, repository, summaries=False)
    if direct_sessions:
        return direct_sessions, False

    used_search, session_search_results = await fetch_search_results(plan, repository, summaries=False)
    if used_search:
        return session_search_results, False

    request, sql_pushed = await candidate_record_query_for(plan, repository)
    request = request.with_limit(plan.effective_fetch_limit())
    return await repository.list_by_query(request), sql_pushed


__all__ = [
    "action_search_ready",
    "actions_ready",
    "can_use_action_stats_with",
    "candidate_batch_limit",
    "candidate_record_query",
    "candidate_record_query_for",
    "fetch_candidates",
    "fetch_direct_id",
    "fetch_record_query_for",
    "fetch_search_results",
    "search_limit",
    "should_batch_post_filter_fetch",
    "uses_actions",
]
