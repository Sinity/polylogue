"""Typed stored-payload hydration for session-insight row mappers."""

from __future__ import annotations

import sqlite3
from typing import TypeVar

from pydantic import BaseModel

from polylogue.core.errors import DatabaseError
from polylogue.storage.sqlite.queries.mappers_support import _parse_json, _row_get

PayloadModel = TypeVar("PayloadModel", bound=BaseModel)


def parse_payload_model(
    row: sqlite3.Row,
    column: str,
    *,
    record_id: str,
    model: type[PayloadModel],
) -> PayloadModel:
    """Hydrate one stored payload column, refusing a row that has none.

    The canonical session-profile writer stores every payload it declares
    (``SESSION_PROFILE_INSERT_COLUMNS``), so an empty document is not an
    older row shape to rebuild from sibling columns: it is a row no writer
    produced, and reading it as a profile would invent evidence.
    """
    raw_payload = _parse_json(
        _row_get(row, column),
        field=column,
        record_id=record_id,
    )
    if not raw_payload:
        raise DatabaseError(f"Missing stored payload in {column} for {record_id}")
    return model.model_validate(raw_payload)


__all__ = ["parse_payload_model"]
