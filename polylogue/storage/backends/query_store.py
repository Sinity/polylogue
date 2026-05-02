"""Low-level SQLite query store composed from explicit concern bands."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.backends.queries import (
    session_insight_summary_queries as session_product_summaries_q,
)
from polylogue.storage.backends.queries import (
    session_insight_thread_queries as session_insight_threads_q,
)
from polylogue.storage.backends.query_store_archive import SQLiteQueryStoreArchiveMixin
from polylogue.storage.backends.query_store_insight_profiles import (
    SQLiteQueryStoreInsightProfilesMixin,
)
from polylogue.storage.backends.query_store_insight_timelines import (
    SQLiteQueryStoreInsightTimelinesMixin,
)
from polylogue.storage.backends.query_store_maintenance import SQLiteQueryStoreMaintenanceMixin
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.query_models import (
    DaySessionSummaryListQuery,
    SessionTagRollupListQuery,
    WorkThreadListQuery,
)
from polylogue.storage.runtime import (
    ActionEventRecord,
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
    WorkThreadRecord,
)

if TYPE_CHECKING:
    from polylogue.storage.action_events.artifacts import ActionEventArtifactState


class SQLiteQueryStore(
    SQLiteQueryStoreArchiveMixin,
    SQLiteQueryStoreInsightProfilesMixin,
    SQLiteQueryStoreInsightTimelinesMixin,
    SQLiteQueryStoreMaintenanceMixin,
):
    """Canonical low-level read/query API for SQLite archive state."""

    def __init__(
        self,
        *,
        connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]],
    ) -> None:
        self._connection_factory = connection_factory

    # -- Insight status (formerly query_store_insight_status.py) ------------

    async def get_action_event_artifact_state(self) -> ActionEventArtifactState:
        from polylogue.storage.action_events.status import action_event_artifact_state_async

        async with self._connection_factory() as conn:
            return await action_event_artifact_state_async(conn)

    async def get_session_insight_status(self) -> SessionInsightStatusSnapshot:
        from polylogue.storage.insights.session.status import session_insight_status_async

        async with self._connection_factory() as conn:
            return await session_insight_status_async(conn)

    # -- Action events (formerly query_store_insight_actions.py) ------------

    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events(conn, conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events_batch(conn, conversation_ids)

    # -- Work threads (formerly query_store_insight_threads.py) -------------

    async def get_work_thread(self, thread_id: str) -> WorkThreadRecord | None:
        async with self._connection_factory() as conn:
            return await session_insight_threads_q.get_work_thread(conn, thread_id)

    async def _list_work_threads_query(
        self,
        query: WorkThreadListQuery,
    ) -> list[WorkThreadRecord]:
        async with self._connection_factory() as conn:
            return await session_insight_threads_q.list_work_threads(conn, query)

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkThreadRecord]:
        return await self._list_work_threads_query(
            WorkThreadListQuery(
                since=since,
                until=until,
                limit=limit,
                offset=offset,
                query=query,
            )
        )

    # -- Summaries (formerly query_store_insight_summaries.py) --------------

    async def _list_session_tag_rollup_rows_query(
        self,
        query: SessionTagRollupListQuery,
    ) -> list[SessionTagRollupRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_session_tag_rollup_rows(conn, query)

    async def list_session_tag_rollup_rows(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ) -> list[SessionTagRollupRecord]:
        return await self._list_session_tag_rollup_rows_query(
            SessionTagRollupListQuery(
                provider=provider,
                since=since,
                until=until,
                query=query,
            )
        )

    async def _list_day_session_summaries_query(
        self,
        query: DaySessionSummaryListQuery,
    ) -> list[DaySessionSummaryRecord]:
        async with self._connection_factory() as conn:
            return await session_product_summaries_q.list_day_session_summaries(conn, query)

    async def list_day_session_summaries(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[DaySessionSummaryRecord]:
        return await self._list_day_session_summaries_query(
            DaySessionSummaryListQuery(
                provider=provider,
                since=since,
                until=until,
            )
        )


__all__ = ["SQLiteQueryStore"]
