"""Durable product and action-read-model query band for SQLiteQueryStore."""

from __future__ import annotations

from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.backends.queries import session_product_profile_queries as session_product_profiles_q
from polylogue.storage.backends.queries import session_product_summary_queries as session_product_summaries_q
from polylogue.storage.backends.queries import session_product_thread_queries as session_product_threads_q
from polylogue.storage.backends.queries import session_product_timeline_queries as session_product_timelines_q
from polylogue.storage.store import (
    ActionEventRecord,
    DaySessionSummaryRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)


class SQLiteQueryStoreProductsMixin:
    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events(conn, conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events_batch(conn, conversation_ids)

    async def get_action_event_read_model_status(self) -> dict[str, int | bool]:
        from polylogue.storage.action_event_lifecycle import action_event_read_model_status_async

        async with self._connection_factory() as conn:
            return await action_event_read_model_status_async(conn)

    async def get_session_product_status(self) -> dict[str, int | bool]:
        from polylogue.storage.session_product_lifecycle import session_product_status_async

        async with self._connection_factory() as conn:
            return await session_product_status_async(conn)

    async def get_session_profile(self, conversation_id: str) -> SessionProfileRecord | None:
        async with self._connection_factory() as conn:
            return await session_product_profiles_q.get_session_profile(conn, conversation_id)

    async def get_session_profiles_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        async with self._connection_factory() as conn:
            return await session_product_profiles_q.get_session_profiles_batch(conn, conversation_ids)

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
        refined_work_kind: str | None = None,
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
                refined_work_kind=refined_work_kind,
                limit=limit,
                offset=offset,
                query=query,
            )

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

    async def get_work_thread(self, thread_id: str) -> WorkThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_product_threads_q.get_work_thread(conn, thread_id)

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        async with self._connection_factory() as conn:
            return await session_product_threads_q.list_work_threads(
                conn,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )

    async def list_session_tag_rollup_rows(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_session_tag_rollup_rows(
                conn,
                provider=provider,
                since=since,
                until=until,
                query=query,
            )

    async def list_day_session_summaries(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[DaySessionSummaryRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_day_session_summaries(
                conn,
                provider=provider,
                since=since,
                until=until,
            )
