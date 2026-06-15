"""Execution helpers for immutable session-query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.retrieval import (
    fetch_batched_filtered_sessions,
    fetch_candidates,
)
from polylogue.archive.query.support import session_to_summary

if TYPE_CHECKING:
    from polylogue.archive.session.domain_models import Session, SessionSummary
    from polylogue.protocols import SessionQueryRuntimeStore


def _reject_session_seed_on_repository_path(plan: SessionQueryPlan) -> None:
    """Fail typed for ``near:id:<ref>`` on the repository runtime store path.

    Session-seeded similarity executes through the archive query path
    (``archive_execution`` / ``search_hits``), which reads the seed session's
    stored vectors via ``VectorProvider.query_by_session``. The repository runtime
    store exposes only text-seeded ``search_similar``; it has no session-seeded
    vector primitive. Rather than silently broadening to an unfiltered listing,
    reject typed and point at the executable path.
    """
    if plan.similar_session_id is None:
        return
    from polylogue.archive.query.expression import ExpressionCompileError

    raise ExpressionCompileError(
        "near:id: session-seeded similarity executes through the archive query path, "
        "not the repository runtime store, which has no session-seeded vector primitive.",
        field="near",
    )


def _plan_sort_is_rank_first(plan: SessionQueryPlan) -> bool:
    """Return True when the plan's effective sort is rank-first.

    When search terms exist and no explicit sort was chosen, results
    should preserve the natural ranking from the search provider.
    """
    return bool(plan.fts_terms and plan.sort is None)


async def list_for_plan(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> list[Session]:
    _reject_session_seed_on_repository_path(plan)
    if plan.similar_text:
        candidates = await repository.search_similar(
            plan.similar_text,
            limit=plan.limit or 10,
            vector_provider=plan.vector_provider,
        )
        return plan._finalize(plan._apply_full_filters(candidates, sql_pushed=False))

    if plan._should_batch_post_filter_fetch():
        batched = await fetch_batched_filtered_sessions(plan, repository)
        return plan._finalize(plan._sort_sessions(batched))

    candidate_results, sql_pushed = await fetch_candidates(plan, repository, summaries=False)
    filtered = plan._apply_full_filters(candidate_results, sql_pushed=sql_pushed)
    if _plan_sort_is_rank_first(plan):
        return plan._finalize(filtered)
    return plan._finalize(plan._sort_sessions(filtered))


async def list_summaries_for_plan(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> list[SessionSummary]:
    _reject_session_seed_on_repository_path(plan)
    can_use_action_stats = await plan.can_use_action_stats_with(repository)
    uses_action_read_model = plan._uses_action_read_model()
    if not plan.can_use_summaries() and not uses_action_read_model:
        msg = (
            "Cannot use list_summaries() with content-dependent filters "
            "(regex, has:thinking, has:tools, etc.). Use list() instead."
        )
        raise ValueError(msg)

    if uses_action_read_model and not can_use_action_stats:
        sessions = await list_for_plan(plan, repository)
        return [session_to_summary(session) for session in sessions]

    candidates, sql_pushed = await fetch_candidates(plan, repository, summaries=True)
    filtered = plan._apply_common_filters(candidates, sql_pushed=sql_pushed)
    if _plan_sort_is_rank_first(plan):
        return plan._finalize(filtered)
    return plan._finalize(plan._sort_summaries(filtered))


async def first_for_plan(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> Session | None:
    results = await list_for_plan(plan.with_limit(1), repository)
    return results[0] if results else None


async def count_for_plan(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> int:
    if plan.can_count_in_sql() and await plan._actions_ready(repository):
        return int(await repository.count_by_query(plan.record_query.for_count()))

    unbounded = plan.with_limit(None)
    if unbounded.can_use_summaries():
        return len(await list_summaries_for_plan(unbounded, repository))
    return len(await list_for_plan(unbounded, repository))


async def delete_for_plan(
    plan: SessionQueryPlan,
    repository: SessionQueryRuntimeStore,
) -> int:
    results: list[Session] | list[SessionSummary]
    if plan.can_use_summaries():
        results = await list_summaries_for_plan(plan, repository)
    else:
        results = await list_for_plan(plan, repository)

    deleted = 0
    for session in results:
        if await repository.delete_session(str(session.id)):
            deleted += 1
    return deleted


__all__ = [
    "count_for_plan",
    "delete_for_plan",
    "first_for_plan",
    "list_for_plan",
    "list_summaries_for_plan",
]
