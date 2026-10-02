"""Exact SQLite cell framing with bounded Python transfers.

SQLite can allocate a complete cell for expressions, generated-column
tables and WITHOUT ROWID lookups. Incremental rowid reads avoid that
allocation; chunking the fallback only bounds transfers to Python, not
SQLite's internal memory.
"""

from __future__ import annotations

import sqlite3
import struct
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal

StorageClass = Literal["null", "integer", "real", "text", "blob"]
LITERAL_CHUNK_BYTES = 64 * 1024


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@dataclass(frozen=True, slots=True)
class SQLiteLiteralCell:
    """Only fixed-width metadata; a variable cell remains on its native owner."""

    storage_class: StorageClass
    byte_length: int
    number: int | float | None = None

    def fixed_bytes(self) -> bytes:
        if self.storage_class == "null":
            return b""
        if self.storage_class == "integer":
            assert isinstance(self.number, int)
            return self.number.to_bytes(8, "big", signed=True)
        if self.storage_class == "real":
            assert isinstance(self.number, (int, float))
            return struct.pack(">d", float(self.number))
        raise ValueError("variable-width SQLite cell requires its literal native stream")


def literal_metadata(storage_class: str | bytes, value: object, byte_length: int | None) -> SQLiteLiteralCell:
    """Use the existing migration proof's storage-class and numeric framing."""
    kind = storage_class.decode("ascii") if isinstance(storage_class, bytes) else storage_class
    if kind == "null":
        return SQLiteLiteralCell("null", 0)
    if kind == "integer" and isinstance(value, int):
        return SQLiteLiteralCell("integer", 8, value)
    if kind == "real" and isinstance(value, (int, float)):
        return SQLiteLiteralCell("real", 8, float(value))
    if kind in {"text", "blob"} and byte_length is not None:
        return SQLiteLiteralCell("text" if kind == "text" else "blob", byte_length)
    raise ValueError("unrecognized SQLite literal storage class")


def cell_projection(expression: str) -> str:
    """Read metadata without converting a variable-width cell to Python."""
    return (
        f"typeof({expression}), "
        f"CASE WHEN typeof({expression}) IN ('integer','real') THEN {expression} END, "
        f"CASE WHEN typeof({expression}) IN ('text','blob') THEN length(CAST({expression} AS BLOB)) END"
    )


def stream_literal_blob(
    blob: sqlite3.Blob, length: int, check_cancel: Callable[[], None]
) -> Generator[bytes, None, None]:
    """Read an actual caller-owned readonly handle without owning its close."""
    if len(blob) != length:
        raise ValueError("SQLite literal length changed within its retained snapshot")
    remaining = length
    while remaining:
        check_cancel()
        chunk = blob.read(min(LITERAL_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError("SQLite literal ended before its declared length")
        remaining -= len(chunk)
        yield chunk


@contextmanager
def owned_literal_stream(stream: Generator[bytes, None, None]) -> Iterator[Generator[bytes, None, None]]:
    """Settle this exact borrowed native stream without losing either failure."""
    primary: BaseException | None = None
    try:
        yield stream
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            stream.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("Native literal read and settlement failed", [primary, cleanup]) from cleanup
            raise


def stream_literal_cell(
    connection: sqlite3.Connection,
    cell: SQLiteLiteralCell,
    *,
    expression: str,
    source_sql: str,
    parameters: tuple[object, ...],
    incremental: Callable[[], AbstractContextManager[sqlite3.Blob]] | None,
    close_cursor: Callable[[sqlite3.Cursor], None],
    check_cancel: Callable[[], None],
) -> Generator[bytes, None, None]:
    """Yield exact literal bytes; callers retain and settle the actual owner.

    The incremental factory must return the original creator's readonly Blob
    context. Its close failure remains attached to that original SQL owner.
    Fallback locators and their native snapshot are caller-owned as well.
    """
    if cell.storage_class not in {"text", "blob"}:
        yield cell.fixed_bytes()
        return
    if incremental is not None:
        with incremental() as blob:
            yield from stream_literal_blob(blob, cell.byte_length, check_cancel)
        return
    for offset in range(0, cell.byte_length, LITERAL_CHUNK_BYTES):
        check_cancel()
        count = min(LITERAL_CHUNK_BYTES, cell.byte_length - offset)
        cursor = connection.cursor()
        primary: BaseException | None = None
        try:
            cursor.execute(
                f"SELECT substr(CAST({expression} AS BLOB), ?, ?) {source_sql}",
                (offset + 1, count, *parameters),
            )
            row = cursor.fetchone()
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                close_cursor(cursor)
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "Literal read and cursor settlement failed", [primary, cleanup]
                    ) from cleanup
                raise
        if row is None or not isinstance(row[0], bytes) or len(row[0]) != count:
            raise ValueError("SQLite literal locator changed within its retained snapshot")
        yield row[0]
