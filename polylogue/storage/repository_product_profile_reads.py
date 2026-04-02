"""Hydrated session-profile durable product reads for the repository."""

from __future__ import annotations

import builtins

from polylogue.storage.session_product_profiles import hydrate_session_profile


class RepositoryProductProfileReadMixin:
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
        tier: str = "merged",
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
            tier=tier,
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
        tier: str = "merged",
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
            tier=tier,
            limit=limit,
            offset=offset,
            query=query,
        )

    async def get_session_enrichment_record(self, conversation_id: str):
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
            tier="enrichment",
            limit=limit,
            offset=offset,
            query=query,
        )


__all__ = ["RepositoryProductProfileReadMixin"]
