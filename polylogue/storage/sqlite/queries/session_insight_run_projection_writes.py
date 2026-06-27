"""Run-projection durable session-insight write queries (async)."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.insights.session.storage import (
    _SESSION_CONTEXT_SNAPSHOT_COLUMNS,
    _SESSION_OBSERVED_EVENT_COLUMNS,
    _SESSION_RUN_COLUMNS,
    session_context_snapshot_insert_values,
    session_observed_event_insert_values,
    session_run_insert_values,
)
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)
from polylogue.storage.sqlite.queries._bulk_replace import replace_insight_rows

__all__ = [
    "replace_session_context_snapshots",
    "replace_session_context_snapshots_bulk",
    "replace_session_observed_events",
    "replace_session_observed_events_bulk",
    "replace_session_runs",
    "replace_session_runs_bulk",
]


async def replace_session_runs(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[SessionRunRecord],
    transaction_depth: int,
) -> None:
    await replace_session_runs_bulk(conn, [session_id], records, transaction_depth)


async def replace_session_runs_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionRunRecord],
    transaction_depth: int,
) -> None:
    await replace_insight_rows(
        conn,
        table="session_runs",
        id_column="session_id",
        id_values=session_ids,
        columns=_SESSION_RUN_COLUMNS,
        records=records,
        extractor=session_run_insert_values,
        transaction_depth=transaction_depth,
        or_replace=True,
    )


async def replace_session_observed_events(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[SessionObservedEventRecord],
    transaction_depth: int,
) -> None:
    await replace_session_observed_events_bulk(conn, [session_id], records, transaction_depth)


async def replace_session_observed_events_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionObservedEventRecord],
    transaction_depth: int,
) -> None:
    await replace_insight_rows(
        conn,
        table="session_observed_events",
        id_column="session_id",
        id_values=session_ids,
        columns=_SESSION_OBSERVED_EVENT_COLUMNS,
        records=records,
        extractor=session_observed_event_insert_values,
        transaction_depth=transaction_depth,
        or_replace=True,
    )


async def replace_session_context_snapshots(
    conn: aiosqlite.Connection,
    session_id: str,
    records: list[SessionContextSnapshotRecord],
    transaction_depth: int,
) -> None:
    await replace_session_context_snapshots_bulk(conn, [session_id], records, transaction_depth)


async def replace_session_context_snapshots_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionContextSnapshotRecord],
    transaction_depth: int,
) -> None:
    await replace_insight_rows(
        conn,
        table="session_context_snapshots",
        id_column="session_id",
        id_values=session_ids,
        columns=_SESSION_CONTEXT_SNAPSHOT_COLUMNS,
        records=records,
        extractor=session_context_snapshot_insert_values,
        transaction_depth=transaction_depth,
        or_replace=True,
    )
