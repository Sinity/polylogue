"""Archive/query domain methods for the async Polylogue facade."""

from __future__ import annotations

from contextlib import suppress

from polylogue.archive_products import SessionProfileProduct, SessionProfileProductQuery


class PolylogueArchiveMixin:
    async def get_conversation(self, conversation_id: str):
        return await self.operations.get_conversation(conversation_id)

    async def get_conversations(self, conversation_ids: list[str]):
        return await self.operations.get_conversations(conversation_ids)

    async def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ):
        return await self.operations.list_conversations(
            provider=provider,
            limit=limit,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ):
        return await self.operations.search(
            query,
            limit=limit,
            source=source,
            since=since,
        )

    async def get_session_product_status(self) -> dict[str, int | bool]:
        return await self.operations.get_session_product_status()

    async def get_session_profile_product(
        self,
        conversation_id: str,
        *,
        tier: str = "merged",
    ) -> SessionProfileProduct | None:
        return await self.operations.get_session_profile_product(conversation_id, tier=tier)

    async def list_session_profile_products(
        self,
        query: SessionProfileProductQuery | None = None,
    ) -> list[SessionProfileProduct]:
        return await self.operations.list_session_profile_products(query)

    def filter(self):
        from polylogue.lib.filters import ConversationFilter
        from polylogue.storage.search_providers import create_vector_provider

        vector_provider = None
        with suppress(ValueError, ImportError):
            vector_provider = create_vector_provider(self._config)

        return ConversationFilter(self.repository, vector_provider=vector_provider)

    async def stats(self):
        return await self.operations.summary_stats()
