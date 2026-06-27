"""Row mappers for materialized run-projection records.

Each row carries the full source model in ``payload_json``; the mapper parses it
back into the pydantic model and re-wraps it with the row's materialization
metadata, so hydration is lossless against the original projection.
"""

from __future__ import annotations

import sqlite3

from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.storage.runtime import (
    SessionContextSnapshotRecord,
    SessionObservedEventRecord,
    SessionRunRecord,
)
from polylogue.storage.sqlite.queries.mappers_support import _parse_json, _row_int, _row_text
from polylogue.types import SessionId


def _row_to_session_run_record(row: sqlite3.Row) -> SessionRunRecord:
    run = ProjectedRun.model_validate(
        _parse_json(row["payload_json"], field="payload_json", record_id=row["run_ref"]) or {}
    )
    return SessionRunRecord(
        session_id=SessionId(row["session_id"]),
        position=int(_row_int(row, "position", 0) or 0),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        run=run,
        search_text=row["search_text"],
    )


def _row_to_session_observed_event_record(row: sqlite3.Row) -> SessionObservedEventRecord:
    event = ObservedEvent.model_validate(
        _parse_json(row["payload_json"], field="payload_json", record_id=row["event_ref"]) or {}
    )
    return SessionObservedEventRecord(
        session_id=SessionId(row["session_id"]),
        position=int(_row_int(row, "position", 0) or 0),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        event=event,
        search_text=row["search_text"],
    )


def _row_to_session_context_snapshot_record(row: sqlite3.Row) -> SessionContextSnapshotRecord:
    snapshot = ContextSnapshot.model_validate(
        _parse_json(row["payload_json"], field="payload_json", record_id=row["snapshot_ref"]) or {}
    )
    return SessionContextSnapshotRecord(
        session_id=SessionId(row["session_id"]),
        position=int(_row_int(row, "position", 0) or 0),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        snapshot=snapshot,
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_session_context_snapshot_record",
    "_row_to_session_observed_event_record",
    "_row_to_session_run_record",
]
