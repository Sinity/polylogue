"""Shared row-mapper support helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import TypeVar, cast, overload

from polylogue.errors import DatabaseError
from polylogue.lib.json import JSONValue, json_document, loads

_T = TypeVar("_T", bound=object)
_RowValue = str | int | float | bytes | bytearray | None
_JSONText = str | bytes | bytearray
JSONObject = dict[str, object]


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


def _row_text(row: sqlite3.Row, key: str) -> str | None:
    value = _row_get(row, key)
    if value is None:
        return None
    return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)


def _row_int(row: sqlite3.Row, key: str, default: int | None = None) -> int | None:
    value = _row_get(row, key)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return int(value.decode("utf-8"))
        except ValueError:
            return default
    try:
        return int(value)
    except ValueError:
        return default


def _row_float(row: sqlite3.Row, key: str, default: float | None = None) -> float | None:
    value = _row_get(row, key)
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return float(value.decode("utf-8"))
        except ValueError:
            return default
    try:
        return float(value)
    except ValueError:
        return default


def _json_object(value: JSONValue | None) -> JSONObject | None:
    if value is None:
        return None
    document = json_document(value)
    if not document:
        return None
    result: JSONObject = {}
    for key, item in document.items():
        result[key] = item
    return result


def _json_text_tuple(value: JSONValue | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if item is None:
            continue
        items.append(str(item))
    return tuple(items)


def _json_int_dict(value: JSONValue | None) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, bool):
            result[key] = int(item)
            continue
        if isinstance(item, (int, float)):
            result[key] = int(item)
            continue
        if isinstance(item, str):
            try:
                result[key] = int(item)
            except ValueError:
                continue
    return result
