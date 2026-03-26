"""Durable session timeline read band for SQLiteQueryStore."""

from __future__ import annotations

from polylogue.storage.backends.queries import (
    session_product_timeline_reads as session_product_timelines_q,
)
from polylogue.storage.store import SessionPhaseRecord, SessionWorkEventRecord


class SQLiteQueryStoreProductTimelinesMixin:
    async def get_session_work_events(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_product_timelines_q.get_work_events(conn, conversation_id)

    async def get_session_phases(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_product_timelines_q.get_session_phases(conn, conversation_id)

    async def list_session_work_events(
        self,
        *,
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_product_timelines_q.list_work_events(
                conn,
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
                query=query,
            )

    async def list_session_phases(
        self,
        *,
        conversation_id: str | None = None,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_product_timelines_q.list_session_phases(
                conn,
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )


__all__ = ["SQLiteQueryStoreProductTimelinesMixin"]
