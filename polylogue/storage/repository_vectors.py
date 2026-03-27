"""Vector/search/stats method mixin for the conversation repository."""

from __future__ import annotations

import asyncio
import builtins

from polylogue.logging import get_logger
from polylogue.protocols import VectorProvider
from polylogue.storage.embedding_stats import read_embedding_stats_async


def resolve_optional_vector_provider(
    vector_provider: VectorProvider | None,
) -> VectorProvider | None:
    """Resolve the explicitly supplied provider or create the default one."""
    if vector_provider is not None:
        return vector_provider

    from polylogue.storage.search_providers import create_vector_provider

    return create_vector_provider()

logger = get_logger(__name__)


class RepositoryVectorMixin:
    async def search_similar(
        self,
        text: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[object]:
        if not vector_provider:
            raise ValueError("Semantic search requires a vector provider.")

        results = await asyncio.to_thread(
            vector_provider.query,
            text,
            limit=limit * 3,
        )
        if not results:
            return []

        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = await self._get_message_conversation_mapping(message_ids)

        conv_scores: dict[str, float] = {}
        for msg_id, distance in results:
            conv_id = msg_to_conv.get(msg_id)
            if conv_id:
                conv_scores[conv_id] = min(conv_scores.get(conv_id, float("inf")), distance)

        ranked_ids = sorted(
            conv_scores.keys(),
            key=lambda conversation_id: conv_scores[conversation_id],
        )[:limit]

        return await self.get_many(ranked_ids)

    async def _get_message_conversation_mapping(
        self, message_ids: builtins.list[str]
    ) -> dict[str, str]:
        if not message_ids:
            return {}

        placeholders = ",".join("?" * len(message_ids))
        query = f"SELECT message_id, conversation_id FROM messages WHERE message_id IN ({placeholders})"

        async with self._backend.connection() as conn:
            cursor = await conn.execute(query, message_ids)
            rows = await cursor.fetchall()

        return {row["message_id"]: row["conversation_id"] for row in rows}

    async def embed_conversation(
        self,
        conversation_id: str,
        vector_provider: VectorProvider | None = None,
    ) -> int:
        vector_provider = resolve_optional_vector_provider(vector_provider)

        if vector_provider is None:
            raise ValueError("No vector provider available. Set VOYAGE_API_KEY.")

        messages = await self.queries.get_messages(conversation_id)
        if not messages:
            return 0

        await asyncio.to_thread(
            vector_provider.upsert,
            conversation_id,
            messages,
        )
        return len(messages)

    async def similarity_search(
        self,
        query: str,
        limit: int = 10,
        vector_provider: VectorProvider | None = None,
    ) -> builtins.list[tuple[str, str, float]]:
        vector_provider = resolve_optional_vector_provider(vector_provider)

        if vector_provider is None:
            raise ValueError("No vector provider configured")

        results = await asyncio.to_thread(
            vector_provider.query,
            query,
            limit=limit,
        )
        if not results:
            return []

        message_ids = [msg_id for msg_id, _ in results]
        msg_to_conv = await self._get_message_conversation_mapping(message_ids)

        return [
            (msg_to_conv[msg_id], msg_id, distance)
            for msg_id, distance in results
            if msg_id in msg_to_conv
        ]

    async def get_archive_stats(self):
        from polylogue.lib.stats import ArchiveStats

        async with self._backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM conversations")
            conv_count = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM messages")
            msg_count = (await cursor.fetchone())[0]

            cursor = await conn.execute("SELECT COUNT(*) FROM attachments")
            att_count = (await cursor.fetchone())[0]

            cursor = await conn.execute(
                """
                SELECT provider_name, COUNT(*) as count
                FROM conversations
                GROUP BY provider_name
                """
            )
            provider_rows = await cursor.fetchall()
            providers = {row["provider_name"]: row["count"] for row in provider_rows}

            embedding_stats = await read_embedding_stats_async(conn)

        db_size = 0
        try:
            db_size = self._backend.db_path.stat().st_size
        except Exception as exc:
            logger.warning("DB size check failed: %s", exc)

        return ArchiveStats(
            total_conversations=conv_count,
            total_messages=msg_count,
            total_attachments=att_count,
            providers=providers,
            embedded_conversations=embedding_stats.embedded_conversations,
            embedded_messages=embedding_stats.embedded_messages,
            pending_embedding_conversations=embedding_stats.pending_conversations,
            stale_embedding_messages=embedding_stats.stale_messages,
            messages_missing_embedding_provenance=embedding_stats.messages_missing_provenance,
            embedding_oldest_at=embedding_stats.oldest_embedded_at,
            embedding_newest_at=embedding_stats.newest_embedded_at,
            embedding_models=embedding_stats.model_counts,
            embedding_dimensions=embedding_stats.dimension_counts,
            db_size_bytes=db_size,
        )
