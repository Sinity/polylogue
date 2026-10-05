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
import sys
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.retained_values import TextOperand
from polylogue.storage.io_phase_metrics import connection_cursor, native_connection_physically_closed
from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection

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


class Utf8Tape(Protocol):
    @property
    def byte_length(self) -> int: ...
    def iter_bytes(self) -> Generator[bytes, None, None]: ...


def fill_original_text_cell(
    connection: sqlite3.Connection, *, table: str, column: str, rowid: int, tape: Utf8Tape, schema: str = "main"
) -> None:
    """Fill a preallocated TEXT cell without binding a complete Python value.

    The calling artifact owns its rowid staging table and transaction. SQLite
    enforces its actual cell/row limit at INSERT/step; the source adapter maps
    that native typed refusal through its existing value-bound contract.
    """
    owner = native_sql_owner_for_connection(connection)
    if owner is None or owner.require_connection() is not connection:
        raise RuntimeError("TEXT transfer requires its original native SQL creator")
    check_compute_cancelled()
    with connection_cursor(
        connection,
        f"SELECT typeof({quote_identifier(column)}) FROM {quote_identifier(schema)}.{quote_identifier(table)} WHERE rowid=?",
        (rowid,),
    ) as cursor:
        row = cursor.fetchone()
    if row != ("text",):
        raise ValueError("TEXT transfer requires its exact preallocated original cell")
    blob = connection.blobopen(table, column, rowid, readonly=False, name=schema)
    owner.retain_incremental_blob(blob)
    chunks: Generator[bytes, None, None] | None = None
    try:
        try:
            chunks = tape.iter_bytes()
            if len(blob) != tape.byte_length:
                raise ValueError("TEXT transfer requires its exact preallocated original length")
            remaining = tape.byte_length
            for chunk in chunks:
                check_compute_cancelled()
                if len(chunk) > remaining:
                    raise ValueError("UTF-8 tape exceeds its original declared length")
                blob.write(chunk)
                remaining -= len(chunk)
            if remaining:
                raise ValueError("UTF-8 tape ended before its original declared length")
        finally:
            primary = sys.exception()
            try:
                try:
                    if chunks is not None:
                        chunks.close()
                except BaseException as cleanup:
                    if primary is not None and cleanup is not primary:
                        raise BaseExceptionGroup(
                            "TEXT tape read and iterator close failed", [primary, cleanup]
                        ) from primary
                    raise
            finally:
                primary = None
    finally:
        primary = sys.exception()
        try:
            try:
                owner.close_incremental_blob(blob)
            except BaseException as cleanup:
                # A healthy transfer adds no per-record dependency. Failed native
                # close retains this same original tape on its existing creator.
                try:
                    owner.retain_lifetime(tape)
                except BaseException as retention:
                    cleanup = BaseExceptionGroup(
                        "Native close and original tape retention failed", [cleanup, retention]
                    )
                if primary is not None and cleanup is not primary:
                    raise BaseExceptionGroup("TEXT transfer and native close failed", [primary, cleanup]) from primary
                raise cleanup
        finally:
            primary = None


class NativeTextInput:
    """Borrow bytes from one exact creator-owned TEXT cell."""

    def __init__(
        self, connection: sqlite3.Connection, *, table: str, column: str, rowid: int, schema: str = "main"
    ) -> None:
        owner = native_sql_owner_for_connection(connection)
        if owner is None or owner.require_connection() is not connection:
            raise RuntimeError("TEXT input requires its original native SQL creator")
        check_compute_cancelled()
        with connection_cursor(
            connection,
            f"SELECT typeof({quote_identifier(column)}), length(CAST({quote_identifier(column)} AS BLOB)) FROM {quote_identifier(schema)}.{quote_identifier(table)} WHERE rowid=?",
            (rowid,),
        ) as cursor:
            metadata = cursor.fetchone()
        if metadata is None or metadata[0] != "text":
            raise ValueError("TEXT input requires its exact original TEXT cell")
        self._owner = owner
        self._connection = connection
        self._blob = connection.blobopen(table, column, rowid, readonly=True, name=schema)
        self._closed = False
        owner.retain_incremental_blob(self._blob)
        try:
            if len(self._blob) != metadata[1]:
                raise ValueError("TEXT input length changed within its original reader")
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("TEXT input admission and close failed", [primary, cleanup]) from primary
            raise

    def read(self, size: int) -> bytes:
        if self._closed:
            raise ValueError("TEXT input is closed")
        if size < 0:
            raise ValueError("TEXT input requires an explicit bounded read size")
        check_compute_cancelled()
        return self._blob.read(min(size, LITERAL_CHUNK_BYTES))

    def close(self) -> None:
        if self._closed:
            return
        if native_connection_physically_closed(self._connection):
            self._closed = True
            return
        try:
            self._owner.close_incremental_blob(self._blob)
        except BaseException as cleanup:
            try:
                self._owner.retain_lifetime(self)
            except BaseException as retention:
                raise BaseExceptionGroup("TEXT input close and retention failed", [cleanup, retention]) from cleanup
            raise
        self._closed = True


def open_native_text_input(
    connection: sqlite3.Connection, *, table: str, column: str, rowid: int, schema: str = "main"
) -> NativeTextInput:
    """Open the original cell stream accepted by the lexical record reader."""
    return NativeTextInput(connection, table=table, column=column, rowid=rowid, schema=schema)


@dataclass(frozen=True, slots=True)
class NativeCellText:
    """Repeatable text view; each traversal settles its own native Blob child."""

    connection: sqlite3.Connection
    table: str
    column: str
    rowid: int
    schema: str = "main"

    def iter_text_chunks(self) -> Generator[str, None, None]:
        stream = open_native_text_input(
            self.connection, table=self.table, column=self.column, rowid=self.rowid, schema=self.schema
        )
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            while chunk := stream.read(LITERAL_CHUNK_BYTES):
                text = decoder.decode(chunk, final=False)
                if text:
                    yield text
            final = decoder.decode(b"", final=True)
            if final:
                yield final
        finally:
            primary = sys.exception()
            try:
                try:
                    stream.close()
                except BaseException as cleanup:
                    if primary is not None and cleanup is not primary:
                        raise BaseExceptionGroup("TEXT input read and close failed", [primary, cleanup]) from primary
                    raise
            finally:
                primary = None

    def equals(self, value: str) -> bool:
        offset = 0
        chunks = self.iter_text_chunks()
        try:
            for chunk in chunks:
                if not value.startswith(chunk, offset):
                    return False
                offset += len(chunk)
            return offset == len(value)
        finally:
            primary = sys.exception()
            try:
                try:
                    chunks.close()
                except BaseException as cleanup:
                    if primary is not None and cleanup is not primary:
                        raise BaseExceptionGroup("TEXT comparison and close failed", [primary, cleanup]) from primary
                    raise
            finally:
                primary = None


def native_cell_text(
    connection: sqlite3.Connection, *, table: str, column: str, rowid: int, schema: str = "main"
) -> NativeCellText:
    """Borrow an exact original rowid/column without reading its full value."""
    return NativeCellText(connection, table, column, rowid, schema)


class OwnedTextTape:
    """Independent transfer bytes retained only by an actual failed native child."""

    def __init__(
        self, operand: TextOperand, directory: Path, *, retain_failed: Callable[[OwnedTextTape], None]
    ) -> None:
        self._handle = tempfile.TemporaryFile(mode="w+b", dir=directory)  # noqa: SIM115 -- original builder owns this tape
        self._byte_length = 0
        iterator: Generator[str, None, None] | None = None
        try:
            iterator = operand.iter_text_chunks()
            try:
                for chunk in iterator:
                    # Raw bound TEXT has strict SQLite UTF-8 semantics. Declared
                    # prose replacement or JSON escaping belongs to its producer.
                    for offset in range(0, len(chunk), 65536):
                        check_compute_cancelled()
                        encoded = chunk[offset : offset + 65536].encode("utf-8")
                        self._handle.write(encoded)
                        self._byte_length += len(encoded)
            finally:
                primary = sys.exception()
                try:
                    try:
                        if iterator is not None:
                            iterator.close()
                    except BaseException as cleanup:
                        if primary is not None and cleanup is not primary:
                            raise BaseExceptionGroup(
                                "shard text read and close failed", [primary, cleanup]
                            ) from primary
                        raise
                finally:
                    primary = None
        except BaseException as primary:
            try:
                self._handle.close()
            except BaseException as cleanup:
                failures = [primary] if cleanup is primary else [primary, cleanup]
                try:
                    retain_failed(self)
                except BaseException as retention:
                    if all(retention is not failure for failure in failures):
                        failures.append(retention)
                if len(failures) == 1:
                    raise
                raise BaseExceptionGroup("shard text capture and close failed", failures) from primary
            raise

    @property
    def byte_length(self) -> int:
        return self._byte_length

    def iter_bytes(self) -> Generator[bytes, None, None]:
        self._handle.seek(0)
        while chunk := self._handle.read(65536):
            check_compute_cancelled()
            yield chunk

    def close(self) -> None:
        from polylogue.storage.sqlite.connection_profile import (
            NativeConnectionSettlementError,
            retained_native_sql_owners_for_lifetime,
        )

        pending = tuple(
            owner for owner in retained_native_sql_owners_for_lifetime(self) if not owner.physical_resources_settled()
        )
        if pending:
            raise NativeConnectionSettlementError(pending[0], RuntimeError("shard text tape requires native drain"))
        self._handle.close()
