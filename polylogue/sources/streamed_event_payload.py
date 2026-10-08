"""Replayable, disk-backed JSON arrays in parsed session-event payloads.

Antigravity can attach an arbitrarily long parent-reference array to one
session event.  A normal Python list makes the parser, prepared artifact,
hashing and writer each retain the complete array.  This value keeps only an
array id and replays its JSON items from the preparation store instead.

The store connection is borrowed from ``SqliteMessageStore``.  Its owner must
keep that connection alive until the prepared session has been hashed and
written to its artifact.  The marker is an explicit protocol value; callers
must use :func:`iter_json_value` or :meth:`StreamedJsonArray.iter_values`
instead of handing it to ``json.dumps``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from polylogue.sources.value_bounds import (
    MAX_STORABLE_VALUE_BYTES,
    ValueBoundRefusedError,
    require_storable_string,
)

_TABLE = "prepared_streamed_json_array_value"
_META = "prepared_streamed_json_array"
_LONE_SURROGATE = re.compile("[\\ud800-\\udfff]")


def ensure_streamed_json_array_table(connection: sqlite3.Connection) -> None:
    """Create the scratch-only item table in the caller-owned preparation DB."""
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {_META} (array_id TEXT PRIMARY KEY, item_count INTEGER NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
        "array_id TEXT NOT NULL, item_ordinal INTEGER NOT NULL, value_json TEXT NOT NULL, "
        "PRIMARY KEY (array_id, item_ordinal)) WITHOUT ROWID"
    )


class StreamedJsonArray:
    """A replayable JSON array backed by the live prepared-artifact database."""

    __slots__ = ("_connection", "_path", "array_id", "count")

    def __init__(self, connection: sqlite3.Connection | None, path: Path, array_id: str, count: int) -> None:
        self._connection = connection
        self._path = path
        self.array_id = array_id
        self.count = count

    def __len__(self) -> int:
        return self.count

    def iter_values(self) -> Iterator[object]:
        """Decode one element at a time from the owning prepared database."""
        if self._connection is not None:
            yield from _iter_values(self._connection, self.array_id)
            return
        with _open_reader(self._path) as connection:
            yield from _iter_values(connection, self.array_id)

    def iter_json_chunks(self, *, ensure_ascii: bool, sort_keys: bool = False) -> Iterator[str]:
        yield from iter_json_value(self, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


class SqliteJsonArrayWriter:
    """Append array values to a caller-owned prepared-artifact connection."""

    __slots__ = ("_connection", "_array_id", "_count", "_finished")

    def __init__(self, connection: sqlite3.Connection) -> None:
        ensure_streamed_json_array_table(connection)
        self._connection = connection
        self._array_id = uuid.uuid4().hex
        self._count = 0
        self._finished = False

    def append(self, value: object) -> None:
        if self._finished:
            raise RuntimeError("streamed JSON array is already finished")
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if not encoded.isascii() and _LONE_SURROGATE.search(encoded):
            encoded = _LONE_SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", encoded)
        encoded = require_storable_string(encoded, kind="streamed event payload item")
        try:
            self._connection.execute(
                f"INSERT INTO {_TABLE} (array_id, item_ordinal, value_json) VALUES (?, ?, ?)",
                (self._array_id, self._count, encoded),
            )
        except sqlite3.DataError as exc:
            if "too big" not in str(exc):
                raise
            observed = len(encoded.encode("utf-8", "surrogatepass"))
            raise ValueBoundRefusedError("streamed event payload item", observed, MAX_STORABLE_VALUE_BYTES) from exc
        self._count += 1

    def extend(self, values: Iterable[object]) -> None:
        for value in values:
            self.append(value)

    def finish(self) -> StreamedJsonArray:
        if self._finished:
            raise RuntimeError("streamed JSON array is already finished")
        self._finished = True
        self._connection.execute(
            f"INSERT INTO {_META} (array_id, item_count) VALUES (?, ?)", (self._array_id, self._count)
        )
        row = self._connection.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            raise RuntimeError("streamed event arrays require a file-backed preparation database")
        return StreamedJsonArray(self._connection, Path(str(row[2])), self._array_id, self._count)


def iter_json_value(value: object, *, ensure_ascii: bool, sort_keys: bool = False) -> Iterator[str]:
    """Emit compact JSON incrementally, recognizing only the explicit marker.

    This deliberately avoids ``JSONEncoder.default``: the standard encoder's
    C implementation can re-enter an arbitrary fallback and collect a value
    before the caller has a chance to stream it.
    """
    if isinstance(value, StreamedJsonArray):
        yield "["
        first = True
        for item in value.iter_values():
            if not first:
                yield ","
            first = False
            yield from iter_json_value(item, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        yield "]"
        return
    if isinstance(value, Mapping):
        yield "{"
        first = True
        keys = sorted(value) if sort_keys else value.keys()
        for key in keys:
            if not isinstance(key, str):
                key = str(key)
            if not first:
                yield ","
            first = False
            yield json.dumps(key, ensure_ascii=ensure_ascii, separators=(",", ":"))
            yield ":"
            yield from iter_json_value(value[key], ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        yield "}"
        return
    if isinstance(value, (list, tuple)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from iter_json_value(item, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        yield "]"
        return
    encoded = json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":"))
    if not ensure_ascii and not encoded.isascii() and _LONE_SURROGATE.search(encoded):
        encoded = _LONE_SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", encoded)
    yield encoded


def _iter_values(connection: sqlite3.Connection, array_id: str) -> Iterator[object]:
    cursor = connection.execute(
        f"SELECT value_json FROM {_TABLE} WHERE array_id = ? ORDER BY item_ordinal", (array_id,)
    )
    try:
        for (encoded,) in cursor:
            yield json.loads(encoded)
    finally:
        cursor.close()


@contextmanager
def _open_reader(path: Path) -> Iterator[sqlite3.Connection]:
    from polylogue.core.sql_settlement import current_native_sql_lifetimes
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    connection = connect_measured(f"file:{quote(str(path))}?mode=ro", uri=True)
    owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        yield owner.require_connection()
    except BaseException as primary:
        from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction

        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


__all__ = [
    "SqliteJsonArrayWriter",
    "StreamedJsonArray",
    "ensure_streamed_json_array_table",
    "iter_json_value",
]
