"""Derived action-event write/query methods for the async SQLite backend."""

from __future__ import annotations

from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.store import ActionEventRecord


class SQLiteDerivedActionsMixin:
    """Derived durable action-event methods for ``SQLiteBackend``."""

    async def replace_action_events(
        self,
        conversation_id: str,
        records: list[ActionEventRecord],
    ) -> None:
        """Replace durable action-event rows for one conversation."""
        async with self._get_connection() as conn:
            await action_events_q.replace_action_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        """Get durable action-event rows for one conversation."""
        return await self.queries.get_action_events(conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        """Get durable action-event rows for multiple conversations."""
        return await self.queries.get_action_events_batch(conversation_ids)


__all__ = ["SQLiteDerivedActionsMixin"]
