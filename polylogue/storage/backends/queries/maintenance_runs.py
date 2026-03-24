"""Queries for durable maintenance lineage records."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.backends.queries.mappers import _row_to_maintenance_run_record
from polylogue.storage.store import MaintenanceRunRecord, _json_array_or_none, _json_or_none

__all__ = [
    "list_maintenance_runs",
    "record_maintenance_run",
    "record_maintenance_run_sync",
]


async def record_maintenance_run(
    conn: aiosqlite.Connection,
    record: MaintenanceRunRecord,
    transaction_depth: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO maintenance_runs (
            maintenance_run_id,
            schema_version,
            executed_at,
            mode,
            preview,
            repair_selected,
            cleanup_selected,
            vacuum_requested,
            target_names_json,
            success,
            manifest_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.maintenance_run_id,
            record.schema_version,
            record.executed_at,
            record.mode,
            int(record.preview),
            int(record.repair_selected),
            int(record.cleanup_selected),
            int(record.vacuum_requested),
            _json_array_or_none(record.target_names),
            int(record.success),
            _json_or_none(record.manifest),
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


def record_maintenance_run_sync(
    conn: sqlite3.Connection,
    record: MaintenanceRunRecord,
) -> None:
    conn.execute(
        """
        INSERT INTO maintenance_runs (
            maintenance_run_id,
            schema_version,
            executed_at,
            mode,
            preview,
            repair_selected,
            cleanup_selected,
            vacuum_requested,
            target_names_json,
            success,
            manifest_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.maintenance_run_id,
            record.schema_version,
            record.executed_at,
            record.mode,
            int(record.preview),
            int(record.repair_selected),
            int(record.cleanup_selected),
            int(record.vacuum_requested),
            _json_array_or_none(record.target_names),
            int(record.success),
            _json_or_none(record.manifest),
        ),
    )
    conn.commit()


async def list_maintenance_runs(
    conn: aiosqlite.Connection,
    *,
    limit: int = 20,
) -> list[MaintenanceRunRecord]:
    cursor = await conn.execute(
        """
        SELECT *
        FROM maintenance_runs
        ORDER BY executed_at DESC, maintenance_run_id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = await cursor.fetchall()
    return [_row_to_maintenance_run_record(row) for row in rows]
