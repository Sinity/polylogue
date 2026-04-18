"""Durable session-profile read band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.backends.queries import (
    session_product_profile_reads as session_product_profiles_q,
)
from polylogue.storage.store import SessionProfileRecord


class SQLiteQueryStoreProductProfilesMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def get_session_profile(self, conversation_id: str) -> SessionProfileRecord | None:
        async with self._connection_factory() as conn:
            return await session_product_profiles_q.get_session_profile(conn, conversation_id)

    async def get_session_profiles_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_product_profiles_q.get_session_profiles_batch(
                conn,
                conversation_ids,
            )

    async def list_session_profiles(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_product_profiles_q.list_session_profiles(
                conn,
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                tier=tier,
                limit=limit,
                offset=offset,
                query=query,
            )


__all__ = ["SQLiteQueryStoreProductProfilesMixin"]
