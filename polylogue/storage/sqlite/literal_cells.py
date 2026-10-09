"""Exact SQLite cell framing with bounded Python transfers.

SQLite can allocate a complete cell for expressions, generated-column
tables and WITHOUT ROWID lookups. Incremental rowid reads avoid that
allocation; chunking the fallback only bounds transfers to Python, not
SQLite's internal memory.
"""

from __future__ import annotations

import codecs
import json
import sqlite3
import struct
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal

StorageClass = Literal["null", "integer", "real", "text", "blob"]
LITERAL_CHUNK_BYTES = 64 * 1024


def canonical_json_text_chunks(stream: Generator[bytes, None, None]) -> Generator[bytes, None, None]:
    """Quote exact UTF-8 TEXT with the canonical serializer's escaping.

    The caller supplies bounded native chunks. Decoder state holds only an
    incomplete UTF-8 sequence; closing this reader closes the original stream.
    Invalid UTF-8 has the same strict decoding failure as the ordinary adapter.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    with owned_literal_stream(stream):
        yield b'"'
        for chunk in stream:
            text = decoder.decode(chunk, final=False)
            if text:
                yield json.dumps(text, ensure_ascii=False, separators=(",", ":"))[1:-1].encode("utf-8")
        final = decoder.decode(b"", final=True)
        if final:
            yield json.dumps(final, ensure_ascii=False, separators=(",", ":"))[1:-1].encode("utf-8")
        yield b'"'


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


def inline_cell_projection(expression: str) -> str:
    """A text or blob cell's bytes when they fit one literal chunk, else NULL."""
    return (
        f"CASE WHEN typeof({expression}) IN ('text','blob') "
        f"AND length(CAST({expression} AS BLOB)) <= {LITERAL_CHUNK_BYTES} "
        f"THEN CAST({expression} AS BLOB) END"
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
            if primary is not None and cleanup is not primary:
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
                if primary is not None and cleanup is not primary:
                    raise BaseExceptionGroup(
                        "Literal read and cursor settlement failed", [primary, cleanup]
                    ) from cleanup
                raise
        if row is None or not isinstance(row[0], bytes) or len(row[0]) != count:
            raise ValueError("SQLite literal locator changed within its retained snapshot")
        yield row[0]


class SQLiteLiteralWriteError(RuntimeError):
    """A streamed literal cannot be published in its declared SQLite cell."""

    def __init__(self, message: str, *, physical_limit: bool = False) -> None:
        super().__init__(message)
        self.physical_limit = physical_limit


def write_literal_text(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    rowid: int,
    *,
    byte_length: int,
    chunks: Callable[[], Generator[bytes, None, None]],
    schema: Literal["main", "temp"] = "main",
) -> None:
    """Write TEXT incrementally under the actual creator's native transaction."""
    from polylogue.core.compute_cancel import check_compute_cancelled
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        native_sql_owner_for_connection,
    )

    owner = native_sql_owner_for_connection(connection)
    if owner is None or not connection.in_transaction:
        raise SQLiteLiteralWriteError("literal write requires its existing native transaction owner")
    owner.require_connection()
    if schema not in {"main", "temp"}:
        raise SQLiteLiteralWriteError("literal write has no declared schema")
    if byte_length > connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH):
        raise SQLiteLiteralWriteError("literal exceeds SQLite's physical value limit", physical_limit=True)
    from polylogue.storage.io_phase_metrics import connection_cursor

    try:
        with connection_cursor(
            connection,
            f"UPDATE {quote_identifier(schema)}.{quote_identifier(table)} SET {quote_identifier(column)}=CAST(zeroblob(?) AS TEXT) WHERE rowid=?",
            (byte_length, rowid),
        ) as cursor:
            if cursor.rowcount != 1:
                raise SQLiteLiteralWriteError("literal write lost its original row")
    except sqlite3.DataError as error:
        raise SQLiteLiteralWriteError("literal exceeds SQLite's physical record limit", physical_limit=True) from error
    blob = connection.blobopen(table, column, rowid, name=schema)
    owner.retain_incremental_blob(blob)
    primary: BaseException | None = None
    try:
        length = 0
        with owned_literal_stream(chunks()) as stream:
            for chunk in stream:
                for offset in range(0, len(chunk), LITERAL_CHUNK_BYTES):
                    check_compute_cancelled()
                    part = chunk[offset : offset + LITERAL_CHUNK_BYTES]
                    length += len(part)
                    if length > byte_length:
                        raise SQLiteLiteralWriteError("literal exceeds declared length")
                    blob.write(part)
        if length != byte_length:
            raise SQLiteLiteralWriteError("literal did not fill declared length")
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            owner.close_incremental_blob(blob)
        except BaseException as cleanup:
            owner.close_required = True
            failure = (
                cleanup
                if primary is None
                else BaseExceptionGroup("Literal write and native Blob close failed", [primary, cleanup])
            )
            raise NativeConnectionSettlementError(owner, failure) from cleanup
