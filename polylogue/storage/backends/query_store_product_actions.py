"""Durable action-event read band for SQLiteQueryStore."""

from __future__ import annotations

from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.store import ActionEventRecord


class SQLiteQueryStoreProductActionsMixin:
    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events(conn, conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        async with self._connection_factory() as conn:
            return await action_events_q.get_action_events_batch(conn, conversation_ids)


__all__ = ["SQLiteQueryStoreProductActionsMixin"]
