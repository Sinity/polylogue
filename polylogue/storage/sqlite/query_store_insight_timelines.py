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
        session_id: str,
    ) -> list[SessionWorkEventRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.get_work_events(conn, session_id)

    async def get_session_phases(
        self,
        session_id: str,
    ) -> list[SessionPhaseRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_timelines_q.get_session_phases(conn, session_id)

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


__all__ = ["SQLiteQueryStoreInsightTimelinesMixin"]
