"""Byte-bounded evidence rows, with lossless fragments for an oversized row.

Concatenate each field's base64-decoded bytes, then decode UTF-8 (and JSON
for ``encoding=json``). SQLite supplies bounded byte slices, using incremental
blob handles where supported and SQL substr for tables with generated columns.
Resuming never materializes or reserializes the entire field in Python.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from polylogue.archive.query.transaction import QueryContinuationInvalidError

DEFAULT_EVIDENCE_PAGE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EvidencePayloadPage:
    rows: list[dict[str, object]]
    total: int
    fragment: dict[str, object] | None = None
    cursor: Mapping[str, object] | None = None

    @property
    def completed_rows(self) -> int:
        return len(self.rows) + int(self.fragment is not None and self.fragment["complete"] is True)


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _coordinates(cursor: Mapping[str, object] | None, column_count: int) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    if set(cursor) != {"field", "byte"}:
        raise QueryContinuationInvalidError("invalid evidence field cursor")
    field, byte = cursor["field"], cursor["byte"]
    if type(field) is not int or type(byte) is not int or not 0 <= field < column_count or byte < 0:
        raise QueryContinuationInvalidError("invalid evidence field coordinates")
    return field, byte


def _field_fragment(
    conn: sqlite3.Connection,
    *,
    table: str,
    columns: tuple[str, ...],
    row_id: int,
    row_offset: int,
    cursor: Mapping[str, object] | None,
    budget: int,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    field, byte = _coordinates(cursor, len(columns))
    types = conn.execute(
        f"SELECT {', '.join(f'typeof({column})' for column in columns)} FROM {table} WHERE rowid = ?",
        (row_id,),
    ).fetchone()
    chunks: list[dict[str, object]] = []
    fragment: dict[str, object] = {"row_offset": row_offset, "fields": chunks, "complete": False}
    remaining = budget - _json_size(fragment)
    while field < len(columns):
        column = columns[field]
        encoding = "utf-8" if types[field] in {"text", "blob"} and column != "structured_patch_json" else "json"
        name = "structured_patch" if column == "structured_patch_json" else column
        blob = None
        try:
            if types[field] in {"text", "blob"}:
                if table == "web_content_constructs":
                    # SQLite refuses blobopen on the entire table because
                    # construct_id is generated, even for ordinary columns.
                    size = int(
                        conn.execute(
                            f"SELECT length(CAST({column} AS BLOB)) FROM {table} WHERE rowid = ?", (row_id,)
                        ).fetchone()[0]
                    )
                else:
                    blob = conn.blobopen(table, column, row_id, readonly=True)
                    size = len(blob)
                scalar = b""
            else:
                value = conn.execute(f"SELECT {column} FROM {table} WHERE rowid = ?", (row_id,)).fetchone()[0]
                if column in {"replace_all", "user_modified"} and value is not None:
                    value = bool(value)
                scalar = json.dumps(value, separators=(",", ":")).encode("utf-8")
                size = len(scalar)
            if byte > size or (byte == size and size != 0):
                raise QueryContinuationInvalidError("evidence byte cursor is outside its field")
            chunk: dict[str, object] = {
                "field": name,
                "encoding": encoding,
                "offset": byte,
                "total_bytes": size,
                "data_base64": "",
            }
            overhead = _json_size(chunk) + 1
            capacity = max(0, (remaining - overhead) // 4 * 3)
            if remaining < overhead or (capacity == 0 and byte < size):
                if not chunks:
                    raise ValueError("evidence byte budget cannot hold a field fragment")
                break
            count = min(capacity, size - byte)
            if count == 0:
                # SQLite substr(X'', 1, 0) yields NULL, not an empty blob.
                # An empty text field still has a real, zero-byte payload.
                data = b""
            elif blob is not None:
                blob.seek(byte)
                data = blob.read(count)
            elif types[field] in {"text", "blob"}:
                data = conn.execute(
                    f"SELECT substr(CAST({column} AS BLOB), ?, ?) FROM {table} WHERE rowid = ?",
                    (byte + 1, count, row_id),
                ).fetchone()[0]
            else:
                data = scalar[byte : byte + count]
            chunk["data_base64"] = base64.b64encode(data).decode("ascii")
            chunks.append(chunk)
            remaining -= _json_size(chunk) + 1
            byte += len(data)
            if byte < size:
                break
            field, byte = field + 1, 0
        finally:
            if blob is not None:
                blob.close()
    complete = field == len(columns)
    fragment["complete"] = complete
    return fragment, None if complete else {"field": field, "byte": byte}


def read_evidence_payload_page(
    conn: sqlite3.Connection,
    *,
    kind: str,
    session_id: str,
    limit: int,
    offset: int,
    cursor: Mapping[str, object] | None,
    budget: int,
    read_rows: Callable[[int, int], tuple[list[dict[str, object]], int]],
) -> EvidencePayloadPage:
    """Read a bounded prefix, or one advancing fragment of the first large row.

    Identifiers come only from the declared projections, never from a token.
    Preflight transfers only ids and byte counts into Python. Its conservative
    JSON bound accounts for escaping every byte and for column names.
    """
    if kind == "file-edits":
        from polylogue.storage.sqlite.queries.file_edits import _SELECT_COLUMNS

        table, order = "file_edits", "message_id, tool_use_block_id"
    elif kind == "web-content":
        from polylogue.storage.sqlite.queries.web_content_constructs import _SELECT_COLUMNS

        table, order = "web_content_constructs", "message_id, block_id, position"
    else:
        raise ValueError(f"not a fragmentable evidence relation: {kind!r}")
    columns = tuple(column.strip() for column in _SELECT_COLUMNS.split(",") if column.strip() != "session_id")
    _coordinates(cursor, len(columns))
    total = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0])
    sizes = (
        "0" if cursor is not None else " + ".join(f"COALESCE(length(CAST({column} AS BLOB)), 4)" for column in columns)
    )
    preflight = conn.execute(
        f"SELECT rowid, ({sizes}) FROM {table} WHERE session_id = ? ORDER BY {order} LIMIT ? OFFSET ?",
        (session_id, 1 if cursor is not None else limit, offset),
    )
    accepted = 0
    remaining = budget
    first_row_id: int | None = None
    try:
        for row_id, raw_size in preflight:
            if first_row_id is None:
                first_row_id = int(row_id)
            estimate = int(raw_size) * 6 + len(columns) * 128
            if cursor is not None or estimate > remaining:
                break
            remaining -= estimate
            accepted += 1
    finally:
        preflight.close()
    if accepted:
        rows, observed_total = read_rows(accepted, offset)
        return EvidencePayloadPage(rows=rows, total=observed_total)
    if first_row_id is None:
        if cursor is not None:
            raise QueryContinuationInvalidError("evidence cursor names a missing row")
        return EvidencePayloadPage(rows=[], total=total)
    fragment, next_cursor = _field_fragment(
        conn, table=table, columns=columns, row_id=first_row_id, row_offset=offset, cursor=cursor, budget=budget
    )
    return EvidencePayloadPage(rows=[], total=total, fragment=fragment, cursor=next_cursor)
