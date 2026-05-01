"""Hydrated session-profile durable product reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.session.session_profile import SessionProfile
from polylogue.storage.products.product_read_support import (
    hydrate_mapping,
    hydrate_optional,
    hydrate_sequence,
)
from polylogue.storage.products.session.profiles import hydrate_session_profile
from polylogue.storage.query_models import SessionProfileListQuery
from polylogue.storage.runtime import SessionProfileRecord

if TYPE_CHECKING:
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryProductProfileReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_session_profile_record(self, conversation_id: str) -> SessionProfileRecord | None:
        return await self.queries.get_session_profile(conversation_id)

    async def get_session_profile(self, conversation_id: str) -> SessionProfile | None:
        record = await self.get_session_profile_record(conversation_id)
        return hydrate_optional(record, hydrate_session_profile)

    async def get_session_profile_records_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfileRecord]:
        return await self.queries.get_session_profiles_batch(conversation_ids)

    async def get_session_profiles_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, SessionProfile]:
        records = await self.get_session_profile_records_batch(conversation_ids)
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
                sort=sort,
                tier=tier,
                limit=limit,
                offset=offset,
                query=query,
            )
        )

    async def get_session_enrichment_record(self, conversation_id: str) -> SessionProfileRecord | None:
        return await self.queries.get_session_profile(conversation_id)

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


__all__ = ["RepositoryProductProfileReadMixin"]
