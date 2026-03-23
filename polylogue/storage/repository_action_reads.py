"""Durable action-event read methods for the repository."""

from __future__ import annotations

import builtins

from polylogue.storage.action_event_rows import hydrate_action_events
from polylogue.storage.store import ActionEventRecord


class RepositoryActionReadMixin:
    async def get_action_event_records(self, conversation_id: str) -> list[ActionEventRecord]:
        return await self.queries.get_action_events(conversation_id)

    async def get_action_event_records_batch(
        self,
        conversation_ids: builtins.list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        return await self.queries.get_action_events_batch(conversation_ids)

    async def get_action_events(self, conversation_id: str):
        return hydrate_action_events(await self.get_action_event_records(conversation_id))

    async def get_action_events_batch(
        self,
        conversation_ids: builtins.list[str],
    ) -> dict[str, tuple]:
        records_by_conversation = await self.get_action_event_records_batch(conversation_ids)
        return {
            conversation_id: hydrate_action_events(records)
            for conversation_id, records in records_by_conversation.items()
        }


__all__ = ["RepositoryActionReadMixin"]
