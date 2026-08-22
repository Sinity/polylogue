"""Durable session-profile read band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.query_models import SessionProfileListQuery
from polylogue.storage.runtime import SessionLatencyProfileRecord, SessionProfileRecord
from polylogue.storage.sqlite.queries import (
    session_insight_profile_reads as session_insight_profiles_q,
)
from polylogue.storage.sqlite.queries import (
    session_latency_profile_reads as session_latency_profiles_q,
)


class SQLiteQueryStoreInsightProfilesMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_session_profile(self, session_id: str) -> SessionProfileRecord | None:
        async with self._connection_factory() as conn:
            return await session_insight_profiles_q.get_session_profile(conn, session_id)

    async def get_session_latency_profile(self, session_id: str) -> SessionLatencyProfileRecord | None:
        async with self._connection_factory() as conn:
            return await session_latency_profiles_q.get_session_latency_profile(conn, session_id)

    async def find_stuck_session_latency_profiles(
        self, *, since: str | None = None, limit: int = 50
    ) -> list[SessionLatencyProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_latency_profiles_q.find_stuck_session_latency_profiles(
                conn,
                since=since,
                limit=limit,
            )

    async def list_session_latency_profiles(
        self,
        *,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 500,
    ) -> list[SessionLatencyProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_latency_profiles_q.list_session_latency_profiles(
                conn,
                origin=origin,
                since=since,
                until=until,
                limit=limit,
            )

    async def get_session_profiles_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_profiles_q.get_session_profiles_batch(
                conn,
                session_ids,
            )

    async def _list_session_profiles_query(
        self,
        query: SessionProfileListQuery,
    ) -> list[SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_profiles_q.list_session_profiles(conn, query)


__all__ = ["SQLiteQueryStoreInsightProfilesMixin"]
