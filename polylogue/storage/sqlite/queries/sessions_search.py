"""Session search helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_async
from polylogue.storage.search import build_ranked_action_search_query, build_ranked_session_search_query
from polylogue.storage.search.models import SessionSearchEvidenceRow, SessionSearchResult
from polylogue.storage.search.query_support import extract_match_terms


async def search_session_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    origins: list[str] | None = None,
) -> SessionSearchResult:
    # Search must not silently serve stale FTS results. Status/reporting
    # paths may use bounded structural probes, but retrieval is a hard
    # correctness boundary.
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        await conn.execute("BEGIN")
    try:
        readiness = await message_fts_search_readiness_async(conn)
        check_fts_readiness(readiness)

        query_spec = build_ranked_session_search_query(
            query=query,
            limit=limit,
            scope_names=origins,
        )
        if query_spec is None:
            return SessionSearchResult(hits=[])

        sql, params = query_spec.sql, query_spec.params
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return SessionSearchResult.from_ids([str(row["session_id"]) for row in rows])
    finally:
        if owns_snapshot:
            await conn.rollback()


async def search_session_evidence_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    origins: list[str] | None = None,
    since: str | None = None,
) -> list[SessionSearchEvidenceRow]:
    # See search_session_hits: retrieval is allowed only against an
    # exactly fresh message FTS surface.
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        await conn.execute("BEGIN")
    try:
        readiness = await message_fts_search_readiness_async(conn)
        check_fts_readiness(readiness)

        query_spec = build_ranked_session_search_query(
            query=query,
            limit=limit,
            scope_names=origins,
            since=since,
            include_snippet=True,
        )
        if query_spec is None:
            return []

        cursor = await conn.execute(query_spec.sql, query_spec.params)
        rows = await cursor.fetchall()
        matched_terms = extract_match_terms(query)
        return [
            SessionSearchEvidenceRow(
                session_id=str(row["session_id"]),
                rank=rank,
                score=float(row["relevance"]) if row["relevance"] is not None else None,
                message_id=str(row["message_id"]) if row["message_id"] is not None else None,
                snippet=str(row["snippet"] or row["fallback_text"] or ""),
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
    finally:
        if owns_snapshot:
            await conn.rollback()


async def search_sessions(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    origins: list[str] | None = None,
) -> list[str]:
    return (await search_session_hits(conn, query, limit, origins)).session_ids()


async def search_action_session_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    origins: list[str] | None = None,
) -> SessionSearchResult:
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        await conn.execute("BEGIN")
    try:
        readiness = await message_fts_search_readiness_async(conn)
        check_fts_readiness(readiness)

        query_spec = build_ranked_action_search_query(
            query=query,
            limit=limit,
            scope_names=origins,
        )
        if query_spec is None:
            return SessionSearchResult(hits=[])

        sql, params = query_spec.sql, query_spec.params
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return SessionSearchResult.from_ids([str(row["session_id"]) for row in rows])
    finally:
        if owns_snapshot:
            await conn.rollback()


async def search_action_sessions(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    origins: list[str] | None = None,
) -> list[str]:
    return (await search_action_session_hits(conn, query, limit, origins)).session_ids()


__all__ = [
    "search_action_session_hits",
    "search_action_sessions",
    "search_session_evidence_hits",
    "search_session_hits",
    "search_sessions",
]
