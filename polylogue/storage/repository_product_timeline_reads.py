"""Hydrated session timeline durable product reads for the repository."""

from __future__ import annotations

from polylogue.storage.session_product_timeline_rows import hydrate_session_phase, hydrate_work_event


class RepositoryProductTimelineReadMixin:
    async def get_session_work_event_records(self, conversation_id: str):
        return await self.queries.get_session_work_events(conversation_id)

    async def get_session_phase_records(self, conversation_id: str):
        return await self.queries.get_session_phases(conversation_id)

    async def get_session_work_events(self, conversation_id: str):
        return [hydrate_work_event(record) for record in await self.get_session_work_event_records(conversation_id)]

    async def get_session_phases(self, conversation_id: str):
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
    ):
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
    ):
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
    ):
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
    ):
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
