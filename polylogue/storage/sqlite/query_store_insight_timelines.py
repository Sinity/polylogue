"""Durable session timeline read band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.query_models import SessionTimelineListQuery
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord
from polylogue.storage.sqlite.queries import (
    session_insight_timeline_reads as session_insight_timelines_q,
)


class SQLiteQueryStoreInsightTimelinesMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_session_work_events(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.get_work_events(conn, conversation_id)

    async def get_session_phases(
        self,
        conversation_id: str,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.get_session_phases(conn, conversation_id)

    async def _list_session_work_events_query(
        self,
        query: SessionTimelineListQuery,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.list_work_events(conn, query)

    async def _list_session_phases_query(
        self,
        query: SessionTimelineListQuery,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.list_session_phases(conn, query)

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
        return await self._list_session_work_events_query(
            SessionTimelineListQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
                query=query,
            )
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
        return await self._list_session_phases_query(
            SessionTimelineListQuery(
                conversation_id=conversation_id,
                provider=provider,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )
        )


__all__ = ["SQLiteQueryStoreInsightTimelinesMixin"]
