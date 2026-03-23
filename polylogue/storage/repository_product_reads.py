"""Durable session/work/product read methods for the repository."""

from __future__ import annotations

import builtins

from polylogue.storage.session_product_rows import (
    hydrate_session_phase,
    hydrate_session_profile,
    hydrate_work_event,
    hydrate_work_thread,
)


class RepositoryProductReadMixin:
    async def get_session_profile_record(self, conversation_id: str):
        return await self.queries.get_session_profile(conversation_id)

    async def get_session_profile(self, conversation_id: str):
        record = await self.get_session_profile_record(conversation_id)
        return hydrate_session_profile(record) if record is not None else None

    async def get_session_profile_records_batch(
        self,
        conversation_ids: builtins.list[str],
    ):
        return await self.queries.get_session_profiles_batch(conversation_ids)

    async def get_session_profiles_batch(
        self,
        conversation_ids: builtins.list[str],
    ):
        records = await self.get_session_profile_records_batch(conversation_ids)
        return {
            conversation_id: hydrate_session_profile(record)
            for conversation_id, record in records.items()
        }

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
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ):
        records = await self.queries.list_session_profiles(
            provider=provider,
            since=since,
            until=until,
            first_message_since=first_message_since,
            first_message_until=first_message_until,
            session_date_since=session_date_since,
            session_date_until=session_date_until,
            limit=limit,
            offset=offset,
            query=query,
        )
        return [hydrate_session_profile(record) for record in records]

    async def list_session_profile_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ):
        return await self.queries.list_session_profiles(
            provider=provider,
            since=since,
            until=until,
            first_message_since=first_message_since,
            first_message_until=first_message_until,
            session_date_since=session_date_since,
            session_date_until=session_date_until,
            limit=limit,
            offset=offset,
            query=query,
        )

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

    async def get_work_thread_record(self, thread_id: str):
        return await self.queries.get_work_thread(thread_id)

    async def get_work_thread(self, thread_id: str):
        record = await self.get_work_thread_record(thread_id)
        return hydrate_work_thread(record) if record is not None else None

    async def list_work_threads(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ):
        records = await self.queries.list_work_threads(
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            query=query,
        )
        return [hydrate_work_thread(record) for record in records]

    async def list_work_thread_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ):
        return await self.queries.list_work_threads(
            since=since,
            until=until,
            limit=limit,
            offset=offset,
            query=query,
        )

    async def list_session_tag_rollup_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
    ):
        return await self.queries.list_session_tag_rollup_rows(
            provider=provider,
            since=since,
            until=until,
            query=query,
        )

    async def list_day_session_summary_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ):
        return await self.queries.list_day_session_summaries(
            provider=provider,
            since=since,
            until=until,
        )


__all__ = ["RepositoryProductReadMixin"]
