"""Hydrated session-profile durable insight reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.session_profile import SessionProfile
from polylogue.storage.insights.insight_read_support import (
    hydrate_mapping,
    hydrate_optional,
    hydrate_sequence,
)
from polylogue.storage.insights.session.profiles import hydrate_session_profile
from polylogue.storage.query_models import SessionProfileListQuery
from polylogue.storage.runtime import SessionLatencyProfileRecord, SessionProfileRecord

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightProfileReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_profile_record(self, session_id: str) -> SessionProfileRecord | None:
        return await self.queries.get_session_profile(session_id)

    async def get_session_latency_profile_record(self, session_id: str) -> SessionLatencyProfileRecord | None:
        return await self.queries.get_session_latency_profile(session_id)

    async def find_stuck_session_latency_profile_records(
        self,
        *,
        since: str | None = None,
        limit: int = 50,
    ) -> list[SessionLatencyProfileRecord]:
        return await self.queries.find_stuck_session_latency_profiles(since=since, limit=limit)

    async def list_session_latency_profile_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = 500,
    ) -> list[SessionLatencyProfileRecord]:
        return await self.queries.list_session_latency_profiles(
            provider=provider,
            since=since,
            until=until,
            limit=limit,
        )

    async def get_session_profile(self, session_id: str) -> SessionProfile | None:
        record = await self.get_session_profile_record(session_id)
        return hydrate_optional(record, hydrate_session_profile)

    async def get_session_profile_records_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        return await self.queries.get_session_profiles_batch(session_ids)

    async def get_session_profiles_batch(
        self,
        session_ids: list[str],
    ) -> dict[str, SessionProfile]:
        records = await self.get_session_profile_records_batch(session_ids)
        return hydrate_mapping(records, hydrate_session_profile)

    async def _list_session_profile_records_query(
        self,
        query: SessionProfileListQuery,
    ) -> list[SessionProfileRecord]:
        return await self.queries._list_session_profiles_query(query)

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
        min_wallclock_seconds: int | None = None,
        max_wallclock_seconds: int | None = None,
        workflow_shape: str | None = None,
        terminal_state: str | None = None,
        sort: str = "source",
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionProfile]:
        records = await self._list_session_profile_records_query(
            SessionProfileListQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                min_wallclock_seconds=min_wallclock_seconds,
                max_wallclock_seconds=max_wallclock_seconds,
                workflow_shape=workflow_shape,
                terminal_state=terminal_state,
                sort=sort,
                tier=tier,
                limit=limit,
                offset=offset,
                query=query,
            )
        )
        return hydrate_sequence(records, hydrate_session_profile)

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
        min_wallclock_seconds: int | None = None,
        max_wallclock_seconds: int | None = None,
        workflow_shape: str | None = None,
        terminal_state: str | None = None,
        sort: str = "source",
        tier: str = "merged",
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionProfileRecord]:
        return await self._list_session_profile_records_query(
            SessionProfileListQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                min_wallclock_seconds=min_wallclock_seconds,
                max_wallclock_seconds=max_wallclock_seconds,
                workflow_shape=workflow_shape,
                terminal_state=terminal_state,
                sort=sort,
                tier=tier,
                limit=limit,
                offset=offset,
                query=query,
            )
        )

    async def get_session_enrichment_record(self, session_id: str) -> SessionProfileRecord | None:
        return await self.queries.get_session_profile(session_id)

    async def list_session_enrichment_records(
        self,
        *,
        provider: str | None = None,
        since: str | None = None,
        until: str | None = None,
        first_message_since: str | None = None,
        first_message_until: str | None = None,
        session_date_since: str | None = None,
        session_date_until: str | None = None,
        min_wallclock_seconds: int | None = None,
        max_wallclock_seconds: int | None = None,
        sort: str = "source",
        limit: int | None = 50,
        offset: int = 0,
        query: str | None = None,
    ) -> list[SessionProfileRecord]:
        return await self._list_session_profile_records_query(
            SessionProfileListQuery(
                provider=provider,
                since=since,
                until=until,
                first_message_since=first_message_since,
                first_message_until=first_message_until,
                session_date_since=session_date_since,
                session_date_until=session_date_until,
                min_wallclock_seconds=min_wallclock_seconds,
                max_wallclock_seconds=max_wallclock_seconds,
                sort=sort,
                tier="enrichment",
                limit=limit,
                offset=offset,
                query=query,
            )
        )


__all__ = ["RepositoryInsightProfileReadMixin"]
