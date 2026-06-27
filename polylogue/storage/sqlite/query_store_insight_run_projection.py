"""Materialized run-projection read band for SQLiteQueryStore."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.query_models import RunProjectionListQuery
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)
from polylogue.storage.sqlite.queries import (
    session_insight_run_projection_reads as run_projection_q,
)


class SQLiteQueryStoreInsightRunProjectionMixin:
    if TYPE_CHECKING:
        _connection_factory: Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]

    async def list_session_runs(self, query: RunProjectionListQuery) -> list[SessionRunRecord]:
        async with self._connection_factory() as conn:
            return await run_projection_q.list_runs(conn, query)

    async def list_session_observed_events(
        self,
        query: RunProjectionListQuery,
    ) -> list[SessionObservedEventRecord]:
        async with self._connection_factory() as conn:
            return await run_projection_q.list_observed_events(conn, query)

    async def list_session_context_snapshots(
        self,
        query: RunProjectionListQuery,
    ) -> list[SessionContextSnapshotRecord]:
        async with self._connection_factory() as conn:
            return await run_projection_q.list_context_snapshots(conn, query)


__all__ = ["SQLiteQueryStoreInsightRunProjectionMixin"]
