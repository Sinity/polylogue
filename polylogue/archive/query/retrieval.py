"""Candidate record queries and search limits for immutable session query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.storage.query_models import SessionRecordQuery


def candidate_record_query(plan: SessionQueryPlan) -> tuple[SessionRecordQuery, bool]:
    record_query = plan.record_query
    return record_query.without_unstable_semantic_filters(), plan.sql_pushed


def search_limit(plan: SessionQueryPlan) -> int:
    # When no explicit --limit is set (e.g. the CLI query-first path with a
    # bare token), effective_fetch_limit() is None. Fall back to the shared
    # MAX_QUERY_LIMIT ceiling rather than an unbounded 10000 fetch (#1749).
    from polylogue.archive.query.spec import MAX_QUERY_LIMIT

    fetch_limit = plan.effective_fetch_limit()
    return max(fetch_limit, 100) if fetch_limit is not None else MAX_QUERY_LIMIT


def search_query_text(plan: SessionQueryPlan) -> str:
    return " ".join(term.strip() for term in plan.fts_terms if term.strip()).strip()


__all__ = [
    "candidate_record_query",
    "search_limit",
    "search_query_text",
]
