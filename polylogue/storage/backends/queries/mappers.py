"""Public root for row-mapper families and shared helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import TypeVar, cast, overload

from polylogue.errors import DatabaseError
from polylogue.lib.json import JSONValue, loads

_T = TypeVar("_T", bound=object)
_RowValue = str | int | float | bytes | bytearray | None
_JSONText = str | bytes | bytearray

# ---------------------------------------------------------------------------
# Shared row-mapper support helpers (formerly mappers_support.py)
# These MUST be defined before the sub-module imports below, since the
# sub-modules import _parse_json / _row_get from this module.
# ---------------------------------------------------------------------------


def _parse_json(
    raw: _RowValue | None,
    *,
    field: str = "",
    record_id: str = "",
) -> JSONValue | None:
    """Parse a JSON string with diagnostic context on failure."""
    if raw is None:
        return None
    if not isinstance(raw, _JSONText):
        raise DatabaseError(f"Corrupt JSON in {field} for {record_id}: expected JSON text, got {type(raw).__name__}")
    if not raw:
        return None
    raw_preview = raw[:80]
    try:
        return loads(raw)
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"Corrupt JSON in {field} for {record_id}: {exc} (value starts: {raw_preview!r})") from exc


@overload
def _row_get(row: sqlite3.Row, key: str, default: None = None) -> _RowValue | None: ...


@overload
def _row_get(row: sqlite3.Row, key: str, default: _T) -> _T: ...


def _row_get(row: sqlite3.Row, key: str, default: _T | None = None) -> _RowValue | _T | None:
    """Get a column value, returning default if the column doesn't exist."""
    try:
        return cast(_RowValue, row[key])
    except (KeyError, IndexError):
        return default


# ---------------------------------------------------------------------------
# Re-exports from sub-modules
# ---------------------------------------------------------------------------

from polylogue.storage.backends.queries.mappers_archive import (  # noqa: E402
    _row_to_action_event,
    _row_to_artifact_observation,
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
    _row_to_raw_conversation,
)
from polylogue.storage.backends.queries.mappers_product_aggregates import (  # noqa: E402
    _row_to_day_session_summary_record,
    _row_to_session_tag_rollup_record,
)
from polylogue.storage.backends.queries.mappers_product_profiles import (  # noqa: E402
    _row_to_session_profile_record,
)
from polylogue.storage.backends.queries.mappers_product_timelines import (  # noqa: E402
    _row_to_session_phase_record,
    _row_to_session_work_event_record,
    _row_to_work_thread_record,
)

__all__ = [
    "_parse_json",
    "_row_get",
    "_row_to_action_event",
    "_row_to_artifact_observation",
    "_row_to_content_block",
    "_row_to_conversation",
    "_row_to_day_session_summary_record",
    "_row_to_message",
    "_row_to_raw_conversation",
    "_row_to_session_phase_record",
    "_row_to_session_profile_record",
    "_row_to_session_tag_rollup_record",
    "_row_to_session_work_event_record",
    "_row_to_work_thread_record",
]
