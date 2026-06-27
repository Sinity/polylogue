"""Hydrated run-projection durable insight reads for the repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.storage.insights.insight_read_support import hydrate_sequence
from polylogue.storage.insights.session.run_projection_rows import (
    hydrate_context_snapshot,
    hydrate_observed_event,
    hydrate_projected_run,
)
from polylogue.storage.query_models import RunProjectionListQuery
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.query_store import SQLiteQueryStore


class RepositoryInsightRunProjectionReadMixin:
    if TYPE_CHECKING:
        queries: SQLiteQueryStore

    async def query_run_records(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        harness: str | None = None,
        role: str | None = None,
        status: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionRunRecord]:
        return await self.queries.list_session_runs(
            RunProjectionListQuery(
                session_id=session_id,
                run_ref=run_ref,
                harness=harness,
                role=role,
                status=status,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            )
        )

    async def query_runs(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        harness: str | None = None,
        role: str | None = None,
        status: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ProjectedRun]:
        return hydrate_sequence(
            await self.query_run_records(
                session_id=session_id,
                run_ref=run_ref,
                harness=harness,
                role=role,
                status=status,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            ),
            hydrate_projected_run,
        )

    async def query_observed_event_records(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        kind: str | None = None,
        delivery_state: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionObservedEventRecord]:
        return await self.queries.list_session_observed_events(
            RunProjectionListQuery(
                session_id=session_id,
                run_ref=run_ref,
                kind=kind,
                delivery_state=delivery_state,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            )
        )

    async def query_observed_events(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        kind: str | None = None,
        delivery_state: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ObservedEvent]:
        return hydrate_sequence(
            await self.query_observed_event_records(
                session_id=session_id,
                run_ref=run_ref,
                kind=kind,
                delivery_state=delivery_state,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            ),
            hydrate_observed_event,
        )

    async def query_context_snapshot_records(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        boundary: str | None = None,
        inheritance_mode: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[SessionContextSnapshotRecord]:
        return await self.queries.list_session_context_snapshots(
            RunProjectionListQuery(
                session_id=session_id,
                run_ref=run_ref,
                boundary=boundary,
                inheritance_mode=inheritance_mode,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            )
        )

    async def query_context_snapshots(
        self,
        *,
        session_id: str | None = None,
        run_ref: str | None = None,
        boundary: str | None = None,
        inheritance_mode: str | None = None,
        query: str | None = None,
        sort: str = "position",
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[ContextSnapshot]:
        return hydrate_sequence(
            await self.query_context_snapshot_records(
                session_id=session_id,
                run_ref=run_ref,
                boundary=boundary,
                inheritance_mode=inheritance_mode,
                query=query,
                sort=sort,
                limit=limit,
                offset=offset,
            ),
            hydrate_context_snapshot,
        )


__all__ = ["RepositoryInsightRunProjectionReadMixin"]
