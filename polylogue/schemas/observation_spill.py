"""A disk-backed JSON tree for schema inspection of large documents.

The input is decoded once into a private SQLite tree. Mapping and sequence
views read that tree lazily, so existing schema observation code can traverse
the complete document without retaining its decoded values in Python memory.
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Callable, Generator, ItemsView, Iterable, Iterator, KeysView, Mapping, Sequence, ValuesView
from contextlib import AbstractContextManager, ExitStack, closing, suppress
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, SupportsIndex, TypeVar, cast, overload

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.value_bounds import require_storable_string
from polylogue.storage.sqlite.connection_profile import scratch_connection_context

_Default = TypeVar("_Default")


class StreamedJSONReadError(sqlite3.DatabaseError):
    """A lazy JSON view failed while executing or stepping its own SQLite read."""

    def __init__(self, failure: sqlite3.Error) -> None:
        super().__init__("streamed_json_read_failed")
        for attribute in ("sqlite_errorcode", "sqlite_errorname"):
            if hasattr(failure, attribute):
                setattr(self, attribute, getattr(failure, attribute))


def _read_rows(
    connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()
) -> Generator[Any, None, None]:
    cursor = None
    try:
        cursor = connection.execute(sql, parameters)
        yield from cursor
    except sqlite3.Error as exc:
        raise StreamedJSONReadError(exc) from exc
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except sqlite3.Error as exc:
                raise StreamedJSONReadError(exc) from exc


def _read_row(connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()) -> Any:
    rows = _read_rows(connection, sql, parameters)
    try:
        return next(rows, None)
    finally:
        rows.close()


class SpilledObject(dict[str, JSONValue]):
    def __init__(self, connection: sqlite3.Connection, node_id: int) -> None:
        dict.__init__(self)
        self._connection = connection
        self._node_id = node_id

    def __iter__(self) -> Iterator[str]:
        with closing(
            _read_rows(
                self._connection,
                "SELECT key_bytes FROM json_object_members WHERE parent_id = ? ORDER BY ordinal",
                (self._node_id,),
            )
        ) as cursor:
            for (key_bytes,) in cursor:
                yield bytes(key_bytes).decode("utf-8", "surrogatepass")

    def __len__(self) -> int:
        row = _read_row(
            self._connection, "SELECT COUNT(*) FROM json_object_members WHERE parent_id = ?", (self._node_id,)
        )
        return int(row[0])

    def __getitem__(self, key: str) -> JSONValue:
        row = _read_row(
            self._connection,
            "SELECT child_id FROM json_object_members WHERE parent_id = ? AND key_bytes = ?",
            (self._node_id, key.encode("utf-8", "surrogatepass")),
        )
        if row is None:
            raise KeyError(key)
        return _load_node(self._connection, int(row[0]))

    @overload
    def get(self, key: str, default: None = None) -> JSONValue: ...

    @overload
    def get(self, key: str, default: _Default) -> JSONValue | _Default: ...

    def get(self, key: str, default: _Default | None = None) -> JSONValue | _Default:
        try:
            return self[key]
        except KeyError:
            return default

    def sorted_keys(self) -> Iterator[str]:
        with closing(
            _read_rows(
                self._connection,
                "SELECT key_bytes FROM json_object_members WHERE parent_id = ? ORDER BY key_bytes",
                (self._node_id,),
            )
        ) as _owned_rows:
            for (key,) in _owned_rows:
                yield bytes(key).decode("utf-8", "surrogatepass")

    def normalized_sorted_items(self, normalize_key: Callable[[str], str]) -> Iterator[tuple[str, JSONValue]]:
        """Sort normalized keys on disk; refuse collisions without a Python key set."""
        connection = self._connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS json_normalized_keys (parent_id INTEGER, normalized BLOB, child_id INTEGER, PRIMARY KEY(parent_id, normalized)) WITHOUT ROWID"
        )
        connection.execute("DELETE FROM json_normalized_keys WHERE parent_id=?", (self._node_id,))
        cursor = _read_rows(
            connection, "SELECT key_bytes, child_id FROM json_object_members WHERE parent_id=?", (self._node_id,)
        )
        try:
            for key, child in cursor:
                check_compute_cancelled()
                normalized = normalize_key(bytes(key).decode("utf-8", "surrogatepass"))
                try:
                    connection.execute(
                        "INSERT INTO json_normalized_keys VALUES (?, ?, ?)",
                        (self._node_id, normalized.encode("utf-8", "surrogatepass"), child),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("normalized_json_key_collision") from error
        finally:
            cursor.close()
        cursor = _read_rows(
            connection,
            "SELECT normalized, child_id FROM json_normalized_keys WHERE parent_id=? ORDER BY normalized",
            (self._node_id,),
        )
        try:
            for key, child in cursor:
                yield (bytes(key).decode("utf-8", "surrogatepass"), _load_node(connection, int(child)))
        finally:
            cursor.close()

    def key_union(self, extra: set[str]) -> KeysView[str]:
        """Join a bounded preceding key set without copying this object's keys."""
        connection = self._connection
        node = cast(int, connection.execute("INSERT INTO json_nodes(kind) VALUES ('object')").lastrowid)
        connection.execute(
            "INSERT INTO json_object_members SELECT ?, key_bytes, ordinal, child_id FROM json_object_members WHERE parent_id = ?",
            (node, self._node_id),
        )
        ordinal = len(self)
        null = cast(
            int, connection.execute("INSERT INTO json_nodes(kind, scalar_json) VALUES ('scalar', 'null')").lastrowid
        )
        for key in extra:
            connection.execute(
                "INSERT OR IGNORE INTO json_object_members VALUES (?, ?, ?, ?)",
                (node, key.encode("utf-8", "surrogatepass"), ordinal, null),
            )
            ordinal += 1
        return KeysView(SpilledObject(connection, node))

    def record_profile_groups(
        self, samples: Iterable[JSONDocument], *, record_type_key: str | None, coarse_type: Callable[[object], str]
    ) -> Iterator[tuple[str, Iterator[tuple[str, tuple[str, ...]]]]]:
        """Spill sampled field unions, preserving the existing profile ordering."""
        from polylogue.archive.raw_payload import record_bucket_key

        connection = self._connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS profile_fields (bucket BLOB, field BLOB, kind TEXT, PRIMARY KEY (bucket, field, kind)) WITHOUT ROWID"
        )
        connection.execute("DELETE FROM profile_fields")
        for sample in samples:
            bucket = record_bucket_key(sample, record_type_key).encode("utf-8", "surrogatepass")
            for key, value in sample.items():
                connection.execute(
                    "INSERT OR IGNORE INTO profile_fields VALUES (?, ?, ?)",
                    (bucket, key.encode("utf-8", "surrogatepass"), coarse_type(value)),
                )

        def fields(bucket: bytes) -> Iterator[tuple[str, tuple[str, ...]]]:
            with closing(
                _read_rows(
                    connection,
                    "SELECT DISTINCT field FROM profile_fields WHERE bucket = ? ORDER BY field LIMIT 24",
                    (bucket,),
                )
            ) as _owned_rows:
                for (key,) in _owned_rows:
                    with closing(
                        _read_rows(
                            connection,
                            "SELECT kind FROM profile_fields WHERE bucket = ? AND field = ? ORDER BY kind",
                            (bucket, key),
                        )
                    ) as _kind_rows:
                        kinds = tuple(row[0] for row in _kind_rows)
                    yield (bytes(key).decode("utf-8", "surrogatepass"), kinds)

        with closing(
            _read_rows(connection, "SELECT DISTINCT bucket FROM profile_fields ORDER BY bucket")
        ) as _owned_rows:
            for (bucket,) in _owned_rows:
                yield (bytes(bucket).decode("utf-8", "surrogatepass"), fields(bucket))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return (
            _read_row(
                self._connection,
                "SELECT 1 FROM json_object_members WHERE parent_id = ? AND key_bytes = ?",
                (self._node_id, key.encode("utf-8", "surrogatepass")),
            )
            is not None
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Mapping)
            and len(self) == len(other)
            and all(key in other and self[key] == other[key] for key in self)
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    # The returned views implement the Mapping protocol against the SQLite
    # tree; ``dict_keys``' static type cannot describe a dict subclass whose
    # values live outside its inherited empty storage.
    def keys(self) -> Any:
        return KeysView(self)

    def values(self) -> Any:
        return ValuesView(self)

    def items(self) -> Any:
        return _SpilledItemsView(self)


class _SpilledItemsView(ItemsView[str, JSONValue]):
    def __init__(self, mapping: SpilledObject) -> None:
        super().__init__(mapping)
        self._spill_mapping = mapping

    def __iter__(self) -> Iterator[tuple[str, JSONValue]]:
        mapping = self._spill_mapping
        with closing(
            _read_rows(
                mapping._connection,
                "SELECT key_bytes, child_id FROM json_object_members WHERE parent_id = ? ORDER BY ordinal",
                (mapping._node_id,),
            )
        ) as cursor:
            for key_bytes, child_id in cursor:
                yield (
                    bytes(key_bytes).decode("utf-8", "surrogatepass"),
                    _load_node(mapping._connection, int(child_id)),
                )


class SpilledArray(list[JSONValue], Sequence[JSONValue]):
    def __init__(self, connection: sqlite3.Connection, node_id: int) -> None:
        list.__init__(self)
        self._connection = connection
        self._node_id = node_id

    def __len__(self) -> int:
        row = _read_row(self._connection, "SELECT COUNT(*) FROM json_array_items WHERE parent_id = ?", (self._node_id,))
        return int(row[0])

    def __iter__(self) -> Iterator[JSONValue]:
        with closing(
            _read_rows(
                self._connection,
                "SELECT child_id FROM json_array_items WHERE parent_id = ? ORDER BY ordinal",
                (self._node_id,),
            )
        ) as cursor:
            for (child_id,) in cursor:
                yield _load_node(self._connection, int(child_id))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, list)
            and len(self) == len(other)
            and all((left == right for left, right in zip(self, other, strict=True)))
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)

    @overload
    def __getitem__(self, index: SupportsIndex) -> JSONValue: ...

    @overload
    def __getitem__(self, index: slice) -> list[JSONValue]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> JSONValue | list[JSONValue]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                with closing(
                    _read_rows(
                        self._connection,
                        "SELECT child_id FROM json_array_items WHERE parent_id = ? AND ordinal >= ? AND ordinal < ? ORDER BY ordinal",
                        (self._node_id, start, stop),
                    )
                ) as cursor:
                    return [_load_node(self._connection, int(row[0])) for row in cursor]
            return list(self)[index]
        index = index.__index__()
        item_index = index if index >= 0 else len(self) + index
        if item_index < 0:
            raise IndexError(index)
        row = _read_row(
            self._connection,
            "SELECT child_id FROM json_array_items WHERE parent_id = ? AND ordinal = ?",
            (self._node_id, item_index),
        )
        if row is None:
            raise IndexError(index)
        return _load_node(self._connection, int(row[0]))


def _load_node(connection: sqlite3.Connection, node_id: int) -> JSONValue:
    row = _read_row(connection, "SELECT kind, scalar_json FROM json_nodes WHERE id = ?", (node_id,))
    if row is None:
        raise ValueError("streamed JSON tree lost a referenced node")
    kind, scalar_json = row
    if kind == "object":
        return SpilledObject(connection, node_id)
    if kind == "array":
        return SpilledArray(connection, node_id)
    if kind == "string":
        return bytes(scalar_json).decode("utf-8", "surrogatepass")
    return cast(JSONValue, json.loads(str(scalar_json)))


def _store_schema_node(connection: sqlite3.Connection, child: object) -> int:
    if isinstance(child, (SpilledObject, SpilledArray)):
        if child._connection is not connection:
            raise ValueError("schema node belongs to a different spill")
        return child._node_id
    if isinstance(child, dict):
        kind = "object"
    elif isinstance(child, list):
        kind = "array"
    elif isinstance(child, str):
        kind = "string"
    else:
        kind = "scalar"
    cursor = connection.execute(
        "INSERT INTO json_nodes(kind, scalar_json) VALUES (?, ?)",
        (
            kind,
            child.encode("utf-8", "surrogatepass")
            if isinstance(child, str)
            else json.dumps(child, ensure_ascii=True)
            if kind == "scalar"
            else None,
        ),
    )
    node = cast(int, cursor.lastrowid)
    if kind == "object":
        for ordinal, (key, item) in enumerate(cast(dict[str, object], child).items()):
            connection.execute(
                "INSERT INTO json_object_members VALUES (?, ?, ?, ?)",
                (node, key.encode("utf-8", "surrogatepass"), ordinal, _store_schema_node(connection, item)),
            )
    elif kind == "array":
        for ordinal, item in enumerate(cast(list[object], child)):
            connection.execute(
                "INSERT INTO json_array_items VALUES (?, ?, ?)",
                (node, ordinal, _store_schema_node(connection, item)),
            )
    return node


_MISSING = object()


@dataclass
class _Frame:
    node_id: int
    kind: str
    key: str | None = None
    ordinal: int = 0


class StreamedJSONDocument(AbstractContextManager[JSONValue]):
    """Decode JSON to private disk and expose a lazy view.

    With ``jsonl=True``, every whitespace-separated root value is instead
    exposed as an item in one synthetic lazy array.
    """

    def __init__(self, path: Path, *, jsonl: bool = False) -> None:
        self._path = path
        self._jsonl = jsonl
        self._owned = ExitStack()
        self._connection: sqlite3.Connection | None = None
        self._root_id: int | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the live spill connection to work owned by this context."""
        if self._connection is None:
            raise RuntimeError("schema spill is closed")
        return self._connection

    def __enter__(self) -> JSONValue:
        connection = self._owned.enter_context(
            scratch_connection_context(prefix="polylogue-schema-json-", filename="document.sqlite")
        )
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-4096")
            connection.executescript(
                """
                CREATE TABLE json_nodes (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER REFERENCES json_nodes(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    scalar_json TEXT
                );
                CREATE TABLE json_object_members (
                    parent_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    key_bytes BLOB NOT NULL,
                    ordinal INTEGER NOT NULL,
                    child_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    PRIMARY KEY (parent_id, key_bytes),
                    UNIQUE (parent_id, ordinal)
                );
                CREATE TABLE json_array_items (
                    parent_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    child_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    PRIMARY KEY (parent_id, ordinal)
                );
                """
            )
            connection.execute("PRAGMA foreign_keys=ON")
            self._connection = connection
            self._root_id = self._decode(connection)
            return _load_node(connection, self._root_id)
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._owned.__exit__(kind, value, traceback)
        self._connection = None

    def store_schema(self, value: object) -> JSONDocument:
        """Store a newly built schema node while borrowing existing child nodes."""
        connection = self.connection
        return cast(JSONDocument, _load_node(connection, _store_schema_node(connection, value)))

    def _decode(self, connection: sqlite3.Connection) -> int:
        stack: list[_Frame] = []
        root_id: int | None = None
        if self._jsonl:
            root_id = cast(int, connection.execute("INSERT INTO json_nodes(kind) VALUES ('array')").lastrowid)
            stack.append(_Frame(root_id, "array"))

        def add_node(kind: str, scalar: object = None) -> int:
            if isinstance(scalar, str):
                require_storable_string(scalar)
            if isinstance(scalar, Decimal):
                scalar = float(scalar)
            scalar_json: str | bytes | None
            if kind == "scalar" and isinstance(scalar, str):
                # Store literal UTF-8 bytes, so JSON escaping cannot impose a
                # smaller physical cell bound than the retained value itself.
                kind = "string"
                scalar_json = scalar.encode("utf-8", "surrogatepass")
            else:
                scalar_json = json.dumps(scalar, ensure_ascii=True, separators=(",", ":")) if kind == "scalar" else None
            parent_id = stack[-1].node_id if stack else None
            cursor = connection.execute(
                "INSERT INTO json_nodes(parent_id, kind, scalar_json) VALUES (?, ?, ?)",
                (parent_id, kind, scalar_json),
            )
            node_id = cast(int, cursor.lastrowid)
            if not stack:
                return node_id
            parent = stack[-1]
            parent_kind = parent.kind
            if parent_kind == "object":
                key = parent.key
                if key is None:
                    raise ValueError("streamed JSON object value has no key")
                key_bytes = key.encode("utf-8", "surrogatepass")
                existing = connection.execute(
                    "SELECT ordinal, child_id FROM json_object_members WHERE parent_id = ? AND key_bytes = ?",
                    (parent.node_id, key_bytes),
                ).fetchone()
                if existing is None:
                    ordinal = parent.ordinal
                    parent.ordinal = ordinal + 1
                    connection.execute(
                        "INSERT INTO json_object_members(parent_id,key_bytes,ordinal,child_id) VALUES (?,?,?,?)",
                        (parent.node_id, key_bytes, ordinal, node_id),
                    )
                else:
                    old_node_id = int(existing[1])
                    connection.execute(
                        "UPDATE json_object_members SET child_id = ? WHERE parent_id = ? AND key_bytes = ?",
                        (node_id, parent.node_id, key_bytes),
                    )
                    connection.execute("DELETE FROM json_nodes WHERE id = ?", (old_node_id,))
                parent.key = None
            else:
                ordinal = parent.ordinal
                parent.ordinal = ordinal + 1
                connection.execute(
                    "INSERT INTO json_array_items(parent_id,ordinal,child_id) VALUES (?,?,?)",
                    (parent.node_id, ordinal, node_id),
                )
            return node_id

        from ijson.backends import python as exact_backend

        from polylogue.core.json_envelope import LexemeAlignedReader
        from polylogue.sources.detection_projection import _DetectionText

        with self._path.open("rb") as stream:
            encoding = json.detect_encoding(stream.read(4))
            stream.seek(0)
            with io.BufferedReader(_DetectionText(stream, encoding)) as reader:
                events = exact_backend.basic_parse(LexemeAlignedReader(reader), multiple_values=self._jsonl)
                try:
                    parsed_root = self._consume_events(events, connection, stack, add_node)
                except BaseException:
                    close = getattr(events, "close", None)
                    if callable(close):
                        with suppress(BaseException):
                            close()
                    raise
        if self._jsonl:
            if len(stack) != 1 or stack[0].node_id != root_id:
                raise ValueError("streamed JSON lines ended inside a container")
            stack.pop()
        elif parsed_root is not None:
            root_id = parsed_root
        if stack:
            raise ValueError("streamed JSON ended inside a container")
        if root_id is None:
            raise ValueError("streamed JSON document has no root value")
        connection.commit()
        return root_id

    @staticmethod
    def _consume_events(
        events: Iterable[tuple[str, object]],
        connection: sqlite3.Connection,
        stack: list[_Frame],
        add_node: Callable[[str, object], int],
    ) -> int | None:
        from polylogue.core.work_progress import advance_work_progress, utf8_byte_length

        root_id = None
        for event, value in events:
            check_compute_cancelled()
            # Count decoded JSON content, not parser tokens as messages.
            # Whitespace is intentionally excluded from this work counter.
            if isinstance(value, str):
                advance_work_progress(bytes=utf8_byte_length(value))
            elif event == "number":
                advance_work_progress(bytes=len(str(value).encode("ascii")))
            elif event == "null":
                advance_work_progress(bytes=4)
            elif event == "boolean":
                advance_work_progress(bytes=4 if value else 5)
            if event == "map_key":
                if not stack or stack[-1].kind != "object":
                    raise ValueError("streamed JSON key is outside an object")
                require_storable_string(cast(str, value), kind="object key")
                stack[-1].key = cast(str, value)
            elif event == "start_map":
                node_id = add_node("object", None)
                if root_id is None:
                    root_id = node_id
                stack.append(_Frame(node_id, "object"))
            elif event == "start_array":
                node_id = add_node("array", None)
                if root_id is None:
                    root_id = node_id
                stack.append(_Frame(node_id, "array"))
            elif event in {"end_map", "end_array"}:
                if not stack:
                    raise ValueError("streamed JSON container ended without a start")
                stack.pop()
            elif event in {"string", "number", "boolean", "null"}:
                node_id = add_node("scalar", value)
                if root_id is None:
                    root_id = node_id
        return root_id
