"""Hydrated session timeline durable insight reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.phase.extraction import SessionPhase
from polylogue.archive.session.extraction import WorkEvent
from polylogue.storage.insights.insight_read_support import hydrate_sequence
from polylogue.storage.insights.session.timeline_rows import hydrate_session_phase, hydrate_work_event
from polylogue.storage.query_models import SessionTimelineListQuery
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightTimelineReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_work_event_records(
        self,
        session_id: str,
    ) -> list[SessionWorkEventRecord]:
        return await self.queries.get_session_work_events(session_id)

    async def get_session_phase_records(self, session_id: str) -> list[SessionPhaseRecord]:
        return await self.queries.get_session_phases(session_id)

    async def get_session_work_events(self, session_id: str) -> list[WorkEvent]:
        return hydrate_sequence(
            await self.get_session_work_event_records(session_id),
            hydrate_work_event,
        )

    async def get_session_phases(self, session_id: str) -> list[SessionPhase]:
        return hydrate_sequence(
            await self.get_session_phase_records(session_id),
            hydrate_session_phase,
        )

    async def _list_session_work_event_records_query(
        self,
        query: SessionTimelineListQuery,
    ) -> list[SessionWorkEventRecord]:
        return await self.queries._list_session_work_events_query(query)

    async def _list_session_phase_records_query(
        self,
        query: SessionTimelineListQuery,
    ) -> list[SessionPhaseRecord]:
        return await self.queries._list_session_phases_query(query)

    async def list_session_work_events(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        heuristic_label: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[WorkEvent]:
        records = await self._list_session_work_event_records_query(
            SessionTimelineListQuery(
                session_id=session_id,
                origin=origin,
                since=since,
                until=until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                heuristic_label=heuristic_label,
                limit=limit,
                offset=offset,
                query=query,
            )
        )
        return hydrate_sequence(records, hydrate_work_event)

    async def list_session_work_event_records(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        heuristic_label: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionWorkEventRecord]:
        return await self._list_session_work_event_records_query(
            SessionTimelineListQuery(
                session_id=session_id,
                origin=origin,
                since=since,
                until=until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                heuristic_label=heuristic_label,
                limit=limit,
                offset=offset,
                query=query,
            )
        )

    async def list_session_phases(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhase]:
        records = await self._list_session_phase_records_query(
            SessionTimelineListQuery(
                session_id=session_id,
                origin=origin,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )
        )
        return hydrate_sequence(records, hydrate_session_phase)

    async def list_session_phase_records(
        self,
        *,
        session_id: str | None = None,
        origin: str | None = None,
        since: str | None = None,
        until: str | None = None,
        kind: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionPhaseRecord]:
        return await self._list_session_phase_records_query(
            SessionTimelineListQuery(
                session_id=session_id,
                origin=origin,
                since=since,
                until=until,
                kind=kind,
                limit=limit,
                offset=offset,
            )
        )


__all__ = ["RepositoryInsightTimelineReadMixin"]
