"""Session search helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage.search.models import SessionSearchEvidenceRow, SessionSearchResult

_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_MESSAGE_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("dangling_fts",), include_run_all=True)
_ACTION_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(
    ("action_event_read_model",),
    include_run_all=True,
)


async def search_session_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> SessionSearchResult:
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_async

    # Search must not silently serve stale FTS results. Status/reporting
    # paths may use bounded structural probes, but retrieval is a hard
    # correctness boundary.
    readiness = await message_fts_search_readiness_async(conn)
    check_fts_readiness(readiness, _MESSAGE_SEARCH_REPAIR_HINT)

    from polylogue.storage.search import build_ranked_session_search_query

    query_spec = build_ranked_session_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
    )
    if query_spec is None:
        return SessionSearchResult(hits=[])

    sql, params = query_spec.sql, query_spec.params
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return SessionSearchResult.from_ids([str(row["session_id"]) for row in rows])


async def search_session_evidence_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
    since: str | None = None,
) -> list[SessionSearchEvidenceRow]:
    from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_async
    from polylogue.storage.search import build_ranked_session_search_query

    # See search_session_hits: retrieval is allowed only against an
    # exactly fresh message FTS surface.
    readiness = await message_fts_search_readiness_async(conn)
    check_fts_readiness(readiness, _MESSAGE_SEARCH_REPAIR_HINT)

    query_spec = build_ranked_session_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
        since=since,
        include_snippet=True,
    )
    if query_spec is None:
        return []

    cursor = await conn.execute(query_spec.sql, query_spec.params)
    rows = await cursor.fetchall()
    from polylogue.storage.search.query_support import extract_match_terms

    matched_terms = extract_match_terms(query)
    return [
        SessionSearchEvidenceRow(
            session_id=str(row["session_id"]),
            rank=rank,
            score=float(row["relevance"]) if row["relevance"] is not None else None,
            message_id=str(row["message_id"]) if row["message_id"] is not None else None,
            snippet=str(row["snippet"]) if row["snippet"] is not None else None,
            match_surface="message",
            retrieval_lane="dialogue",
            matched_terms=matched_terms,
            score_components=({"bm25_raw": float(row["relevance"])} if row["relevance"] is not None else {}),
            score_kind="bm25" if row["relevance"] is not None else None,
            lane_rank=rank,
            raw_score=float(row["relevance"]) if row["relevance"] is not None else None,
        )
        for rank, row in enumerate(rows, start=1)
    ]


async def search_sessions(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    return (await search_session_hits(conn, query, limit, providers)).session_ids()


async def search_action_session_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> SessionSearchResult:
    from polylogue.errors import DatabaseError
    from polylogue.storage.action_events.status import action_event_read_model_status_async
    from polylogue.storage.search import build_ranked_action_search_query

    status = await action_event_read_model_status_async(conn)
    if not bool(status["action_fts_exists"]):
        raise DatabaseError(f"Action search index not built. {_ACTION_SEARCH_REPAIR_HINT}")
    if not bool(status["action_fts_ready"]):
        raise DatabaseError(f"Action search index is incomplete. {_ACTION_SEARCH_REPAIR_HINT}")

    query_spec = build_ranked_action_search_query(
        query=query,
        limit=limit,
        scope_names=providers,
    )
    if query_spec is None:
        return SessionSearchResult(hits=[])

    sql, params = query_spec.sql, query_spec.params
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return SessionSearchResult.from_ids([str(row["session_id"]) for row in rows])


async def search_action_sessions(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
) -> list[str]:
    return (await search_action_session_hits(conn, query, limit, providers)).session_ids()


__all__ = [
    "search_action_session_hits",
    "search_action_sessions",
    "search_session_evidence_hits",
    "search_session_hits",
    "search_sessions",
]
