"""Run audit queries."""

from __future__ import annotations

import aiosqlite

from polylogue.core.common import json_object as _json_object
from polylogue.storage.run_state import RunCounts
from polylogue.storage.runtime import RunRecord, _json_or_none
from polylogue.storage.sqlite.queries.mappers import _parse_json

__all__ = [
    "get_latest_run",
    "record_run",
]


async def get_latest_run(conn: aiosqlite.Connection) -> RunRecord | None:
    """Fetch the most recent pipeline run record."""
    cursor = await conn.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1")
    row = await cursor.fetchone()
    if not row:
        return None
    counts_payload = _parse_json(row["counts_json"], field="counts_json", record_id=row["run_id"])
    return RunRecord(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        plan_snapshot=_json_object(_parse_json(row["plan_snapshot"], field="plan_snapshot", record_id=row["run_id"])),
        counts=RunCounts.model_validate(counts_payload).to_payload() if counts_payload is not None else None,
        drift=_json_object(_parse_json(row["drift_json"], field="drift_json", record_id=row["run_id"])),
        indexed=bool(row["indexed"]) if row["indexed"] is not None else None,
        duration_ms=row["duration_ms"],
    )


async def record_run(
    conn: aiosqlite.Connection,
    record: RunRecord,
    transaction_depth: int,
) -> None:
    """Record a pipeline run audit entry."""
    await conn.execute(
        """
        INSERT INTO runs (
            run_id,
            timestamp,
            plan_snapshot,
            counts_json,
            drift_json,
            indexed,
            duration_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO NOTHING
        """,
        (
            record.run_id,
            record.timestamp,
            _json_or_none(record.plan_snapshot),
            _json_or_none(record.counts),
            _json_or_none(record.drift),
            record.indexed,
            record.duration_ms,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()
