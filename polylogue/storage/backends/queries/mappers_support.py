"""Shared row-mapper support helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from polylogue.errors import DatabaseError


def _parse_json(raw: str | None, *, field: str = "", record_id: str = "") -> Any:
    """Parse a JSON string with diagnostic context on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatabaseError(
            f"Corrupt JSON in {field} for {record_id}: {exc} (value starts: {raw[:80]!r})"
        ) from exc


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Get a column value, returning default if the column doesn't exist."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


__all__ = ["_parse_json", "_row_get"]
