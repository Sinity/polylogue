"""Hydrated session timeline durable product reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.phase_extraction import SessionPhase
from polylogue.lib.work_event_extraction import WorkEvent
from polylogue.storage.session_product_timeline_rows import hydrate_session_phase, hydrate_work_event
from polylogue.storage.store import SessionPhaseRecord, SessionWorkEventRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryProductTimelineReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_work_event_records(
        self,
        conversation_id: str,
    ) -> list[SessionWorkEventRecord]:
        return await self.queries.get_session_work_events(conversation_id)

    async def get_session_phase_records(self, conversation_id: str) -> list[SessionPhaseRecord]:
        return await self.queries.get_session_phases(conversation_id)

    async def get_session_work_events(self, conversation_id: str) -> list[WorkEvent]:
        return [hydrate_work_event(record) for record in await self.get_session_work_event_records(conversation_id)]

    async def get_session_phases(self, conversation_id: str) -> list[SessionPhase]:
        return [hydrate_session_phase(record) for record in await self.get_session_phase_records(conversation_id)]

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
    ) -> list[WorkEvent]:
        records = await self.queries.list_session_work_events(
            conversation_id=conversation_id,
            provider=provider,
            since=since,
            until=until,
            kind=kind,
            limit=limit,
            offset=offset,
            query=query,
        )
        return [hydrate_work_event(record) for record in records]

    async def list_session_work_event_records(
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
        return await self.queries.list_session_work_events(
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
    ) -> list[SessionPhase]:
        records = await self.queries.list_session_phases(
            conversation_id=conversation_id,
            provider=provider,
            since=since,
            until=until,
            kind=kind,
            limit=limit,
            offset=offset,
        )
        return [hydrate_session_phase(record) for record in records]

    async def list_session_phase_records(
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
        return await self.queries.list_session_phases(
            conversation_id=conversation_id,
            provider=provider,
            since=since,
            until=until,
            kind=kind,
            limit=limit,
            offset=offset,
        )


__all__ = ["RepositoryProductTimelineReadMixin"]
