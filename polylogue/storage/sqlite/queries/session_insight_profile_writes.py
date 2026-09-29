"""Profile-oriented durable session-insight write queries."""

from __future__ import annotations

from collections.abc import Sequence

import aiosqlite

from polylogue.storage.derived.session.storage import (
    _SESSION_LATENCY_PROFILE_COLUMNS,
    SESSION_PROFILE_INSERT_COLUMNS,
    session_latency_profile_insert_values,
    session_profile_insert_values,
)
from polylogue.storage.runtime import SessionLatencyProfileRecord, SessionProfileRecord
from polylogue.storage.sqlite.queries._bulk_replace import replace_insight_rows

__all__ = [
    "replace_session_latency_profile",
    "replace_session_latency_profiles_bulk",
    "replace_session_profile",
    "replace_session_profiles_bulk",
]


async def replace_session_profiles_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionProfileRecord],
    transaction_depth: int,
) -> None:
    await replace_insight_rows(
        conn,
        table="session_profiles",
        id_column="session_id",
        id_values=session_ids,
        columns=SESSION_PROFILE_INSERT_COLUMNS,
        records=records,
        extractor=session_profile_insert_values,
        transaction_depth=transaction_depth,
    )


async def replace_session_profile(
    conn: aiosqlite.Connection,
    record: SessionProfileRecord,
    transaction_depth: int,
) -> None:
    await replace_session_profiles_bulk(
        conn,
        [record.session_id],
        [record],
        transaction_depth,
    )


async def replace_session_latency_profiles_bulk(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    records: Sequence[SessionLatencyProfileRecord],
    transaction_depth: int,
) -> None:
    await replace_insight_rows(
        conn,
        table="session_latency_profiles",
        id_column="session_id",
        id_values=session_ids,
        columns=_SESSION_LATENCY_PROFILE_COLUMNS,
        records=records,
        extractor=session_latency_profile_insert_values,
        transaction_depth=transaction_depth,
    )


async def replace_session_latency_profile(
    conn: aiosqlite.Connection,
    record: SessionLatencyProfileRecord,
    transaction_depth: int,
) -> None:
    await replace_session_latency_profiles_bulk(
        conn,
        [record.session_id],
        [record],
        transaction_depth,
    )
