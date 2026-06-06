"""Action-event artifact-state read for the repository.

The materialized action-event *rows* are no longer read through the
repository — every surface derives action events directly from a session's
content blocks (see ``Polylogue.get_action_events``). What remains here is the
artifact-state readout: the materialization-readiness signal consumed by
neighbor discovery, retrieval candidates, repair, and the slow-query notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.action_events.artifacts import ActionEventArtifactState
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryActionReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def get_action_event_artifact_state(self) -> ActionEventArtifactState:
        return await self.queries.get_action_event_artifact_state()


__all__ = ["RepositoryActionReadMixin"]
