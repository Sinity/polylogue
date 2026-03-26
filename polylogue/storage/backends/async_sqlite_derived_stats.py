"""Derived conversation-stat writes for the async SQLite backend."""

from __future__ import annotations

from polylogue.storage.backends.queries import stats as stats_q
from polylogue.storage.store import MessageRecord


class SQLiteDerivedStatsMixin:
    """Derived aggregate-stat methods for ``SQLiteBackend``."""

    async def upsert_conversation_stats(
        self,
        conversation_id: str,
        provider_name: str,
        messages: list[MessageRecord],
    ) -> None:
        """Upsert precomputed per-conversation aggregate stats."""
        async with self._get_connection() as conn:
            await stats_q.upsert_conversation_stats(
                conn,
                conversation_id,
                provider_name,
                messages,
                self._transaction_depth,
            )


__all__ = ["SQLiteDerivedStatsMixin"]
