"""Durable action-event read methods for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.action_events import ActionEvent
from polylogue.storage.action_events.rows import hydrate_action_events
from polylogue.storage.runtime import ActionEventRecord

if TYPE_CHECKING:
    from polylogue.storage.action_events.artifacts import ActionEventArtifactState
    from polylogue.storage.backends.query_store import SQLiteQueryStore


class RepositoryActionReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_action_event_artifact_state(self) -> ActionEventArtifactState:
        return await self.queries.get_action_event_artifact_state()

    async def get_action_event_records(self, conversation_id: str) -> list[ActionEventRecord]:
        return await self.queries.get_action_events(conversation_id)

    async def get_action_event_records_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        return await self.queries.get_action_events_batch(conversation_ids)

    async def get_action_events(self, conversation_id: str) -> tuple[ActionEvent, ...]:
        return hydrate_action_events(await self.get_action_event_records(conversation_id))

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, tuple[ActionEvent, ...]]:
        records_by_conversation = await self.get_action_event_records_batch(conversation_ids)
        return {
            conversation_id: hydrate_action_events(records)
            for conversation_id, records in records_by_conversation.items()
        }


__all__ = ["RepositoryActionReadMixin"]
