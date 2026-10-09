"""A disk-backed JSON tree for schema inspection of large documents.

The input is decoded once into a private SQLite tree. Mapping and sequence
views read that tree lazily. Exact string and number tokens live in chunks;
structural consumers read their kinds, and selected scalar reads reconstruct
complete values without keeping the decoded document in Python memory.
"""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import re
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, ItemsView, Iterable, Iterator, KeysView, Mapping, Sequence, ValuesView
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager, suppress
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Never, SupportsIndex, TypeVar, cast, overload

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDocument, JSONValue, _ValidatedJSONContainer
from polylogue.core.json_envelope import _Readable
from polylogue.core.work_progress import advance_work_progress
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


@dataclass(frozen=True)
class SpilledKey:
    """An exact completed key token, borrowed from the existing JSON owner."""

    connection: sqlite3.Connection
    token: int
    _owner: _ScalarTokenStore | None = field(default=None, repr=False, compare=False)

    def startswith(self, prefix: str) -> bool:
        expected = prefix.encode("utf-8", "surrogatepass")
        offset = 0
        with closing(self.iter_utf8_chunks()) as chunks:
            for chunk in chunks:
                shared = min(len(chunk), len(expected) - offset)
                if chunk[:shared] != expected[offset : offset + shared]:
                    return False
                offset += shared
                if offset == len(expected):
                    return True
        return offset == len(expected)

    def iter_utf8_chunks(self) -> Generator[bytes, None, None]:
        with closing(
            _read_rows(
                self.connection,
                "SELECT data FROM json_scalar_chunks WHERE kind='string' AND token=? ORDER BY ordinal",
                (self.token,),
            )
        ) as rows:
            for (chunk,) in rows:
                check_compute_cancelled()
                yield bytes(chunk)

    def read(self) -> str:
        return "".join(chunk.decode("utf-8", "surrogatepass") for chunk in self.iter_utf8_chunks())

    @property
    def small_name(self) -> str | None:
        row = _read_row(
            self.connection, "SELECT short_chars,small_bytes FROM json_key_meta WHERE token=?", (self.token,)
        )
        if row is None:
            raise ValueError("JSON key token is incomplete")
        return bytes(row[1]).decode("utf-8", "surrogatepass") if row[0] <= 128 else None

    @property
    def digest(self) -> bytes:
        row = _read_row(self.connection, "SELECT digest FROM json_key_meta WHERE token=?", (self.token,))
        if row is None:
            raise ValueError("JSON key token is incomplete")
        return bytes(row[0])

    def compare(self, other: SpilledKey) -> int:
        return _compare_key_chunks(self.iter_utf8_chunks(), other.iter_utf8_chunks())

    def matches(self, value: str) -> bool:
        def chunks() -> Generator[bytes, None, None]:
            for offset in range(0, len(value), 1024):
                yield value[offset : offset + 1024].encode("utf-8", "surrogatepass")

        return _compare_key_chunks(self.iter_utf8_chunks(), chunks()) == 0


@dataclass(frozen=True, eq=False)
class SpilledProfileToken:
    """One exact profile name assembled from existing scalar-token references."""

    parts: tuple[str | SpilledKey, ...]

    def iter_utf8_chunks(self) -> Generator[bytes, None, None]:
        for part in self.parts:
            if isinstance(part, SpilledKey):
                with closing(part.iter_utf8_chunks()) as chunks:
                    yield from chunks
            else:
                for offset in range(0, len(part), 1024):
                    yield part[offset : offset + 1024].encode("utf-8", "surrogatepass")

    def read(self) -> str:
        return "".join(chunk.decode("utf-8", "surrogatepass") for chunk in self.iter_utf8_chunks())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str | SpilledProfileToken) and compare_profile_tokens(self, other) == 0

    __hash__ = None  # type: ignore[assignment]


def profile_token_chunks(token: str | SpilledProfileToken) -> Generator[bytes, None, None]:
    if isinstance(token, SpilledProfileToken):
        with closing(token.iter_utf8_chunks()) as chunks:
            yield from chunks
    else:
        for offset in range(0, len(token), 1024):
            yield token[offset : offset + 1024].encode("utf-8", "surrogatepass")


def compare_profile_tokens(left: str | SpilledProfileToken, right: str | SpilledProfileToken) -> int:
    if isinstance(left, str) and isinstance(right, str):
        return (left > right) - (left < right)
    return _compare_key_chunks(profile_token_chunks(left), profile_token_chunks(right))


def profile_token_text(token: str | SpilledProfileToken) -> str:
    """Materialize a token only at an explicitly selected literal-output boundary."""
    return token.read() if isinstance(token, SpilledProfileToken) else token


def profile_token_repr_chunks(token: str | SpilledProfileToken) -> Generator[bytes, None, None]:
    """Emit exactly Python's string repr, including its global quote choice."""
    single = double = False
    for chunk in profile_token_chunks(token):
        single = single or b"'" in chunk
        double = double or b'"' in chunk
    quote = '"' if single and not double else "'"
    yield quote.encode()
    for chunk in profile_token_chunks(token):
        text = chunk.decode("utf-8", "surrogatepass")
        escaped = "".join("\\" + character if character == quote else repr(character)[1:-1] for character in text)
        yield escaped.encode("utf-8")
    yield quote.encode()


def _profile_name(prefix: str, key: str | SpilledKey, suffix: str = "") -> str | SpilledProfileToken:
    if isinstance(key, SpilledKey):
        name = key.small_name
        if name is None:
            return SpilledProfileToken((prefix, key, suffix))
        key = name
    return prefix + key + suffix


def _compare_key_chunks(left_chunks: Generator[bytes, None, None], right_chunks: Generator[bytes, None, None]) -> int:
    """Settle exact key identity/order even when chunk boundaries differ."""
    left = right = b""
    with closing(left_chunks), closing(right_chunks):
        while True:
            if not left:
                left = next(left_chunks, b"")
            if not right:
                right = next(right_chunks, b"")
            if not left or not right:
                return bool(left) - bool(right)
            shared = min(len(left), len(right))
            a, b = left[:shared], right[:shared]
            if a != b:
                return -1 if a < b else 1
            left, right = left[shared:], right[shared:]


def _literal_key(connection: sqlite3.Connection, key: str) -> SpilledKey:
    ordinal = int(
        connection.execute("SELECT COALESCE(MAX(token),0)+1 FROM json_scalar_tokens WHERE kind='string'").fetchone()[0]
    )
    store = _ScalarTokenStore(connection)
    for offset in range(0, len(key), 1024):
        store._store("string", ordinal, key[offset : offset + 1024].encode("utf-8", "surrogatepass"))
    store._finish("string", ordinal)
    return SpilledKey(connection, ordinal)


class SpilledObject(dict[str, JSONValue], _ValidatedJSONContainer):
    def __init__(self, connection: sqlite3.Connection, node_id: int) -> None:
        dict.__init__(self)
        self._connection = connection
        self._node_id = node_id

    def key_entries(self, *, sorted_keys: bool = False) -> Generator[tuple[SpilledKey, int], None, None]:
        order = "CAST(key_token AS TEXT) COLLATE json_key_order" if sorted_keys else "ordinal"
        with closing(
            _read_rows(
                self._connection,
                f"SELECT key_token,child_id FROM json_object_members WHERE parent_id=? ORDER BY {order}",
                (self._node_id,),
            )
        ) as rows:
            for token, child in rows:
                check_compute_cancelled()
                yield SpilledKey(self._connection, int(token)), int(child)

    def __iter__(self) -> Iterator[str]:
        with closing(self.key_entries()) as entries:
            for key, _child in entries:
                yield key.read()

    def __len__(self) -> int:
        row = _read_row(
            self._connection, "SELECT COUNT(*) FROM json_object_members WHERE parent_id=?", (self._node_id,)
        )
        return int(row[0])

    def _member(self, key: str) -> int | None:
        digest = hashlib.sha256()
        for offset in range(0, len(key), 1024):
            digest.update(key[offset : offset + 1024].encode("utf-8", "surrogatepass"))
        with closing(
            _read_rows(
                self._connection,
                "SELECT 1,m.key_token,m.child_id,k.small_bytes FROM json_object_members m "
                "JOIN json_key_meta k ON k.token=m.key_token WHERE m.parent_id=? AND m.key_digest=?",
                (self._node_id, digest.digest()),
            )
        ) as rows:
            for _exists, token, child, small_bytes in rows:
                if small_bytes is not None:
                    equal = len(key) <= 128 and bytes(small_bytes) == key.encode("utf-8", "surrogatepass")
                else:
                    equal = SpilledKey(self._connection, int(token)).matches(key)
                if equal:
                    return int(child)
        return None

    def __getitem__(self, key: str) -> JSONValue:
        child = self._member(key)
        if child is None:
            raise KeyError(key)
        return _load_node(self._connection, child)

    @overload
    def get(self, key: str, default: None = None) -> JSONValue: ...

    @overload
    def get(self, key: str, default: _Default) -> JSONValue | _Default: ...

    def get(self, key: str, default: _Default | None = None) -> JSONValue | _Default:
        try:
            return self[key]
        except KeyError:
            return default

    def value_for_key(self, key: SpilledKey) -> JSONValue:
        """Read an internally selected member without reconstructing its name."""
        if key.connection is not self._connection:
            raise ValueError("JSON key belongs to a different owner")
        row = _read_row(
            self._connection,
            "SELECT child_id FROM json_object_members WHERE parent_id=? AND key_token=?",
            (self._node_id, key.token),
        )
        if row is None:
            raise KeyError(key.token)
        return _load_node(self._connection, int(row[0]))

    def structure_value(self, key: str) -> JSONValue:
        child = self._member(key)
        if child is None:
            raise KeyError(key)
        return _load_structure_node(self._connection, child)

    def string_prefix(self, key: str, *, characters: int) -> str:
        """Read a declared discriminator prefix without reconstructing its value."""
        child = self._member(key)
        if child is None:
            raise KeyError(key)
        kind, token, scalar = _read_row(
            self._connection, "SELECT kind, token, scalar_json FROM json_nodes WHERE id=?", (child,)
        )
        if kind != "string":
            raise TypeError("declared JSON discriminator is not a string")
        if token is None:
            return bytes(scalar).decode("utf-8", "surrogatepass")[:characters]
        result: list[str] = []
        remaining = characters
        with closing(
            _read_rows(
                self._connection,
                "SELECT data FROM json_scalar_chunks WHERE kind='string' AND token=? ORDER BY ordinal",
                (token,),
            )
        ) as chunks:
            for (chunk,) in chunks:
                piece = bytes(chunk).decode("utf-8", "surrogatepass")[:remaining]
                result.append(piece)
                remaining -= len(piece)
                if remaining == 0:
                    break
        return "".join(result)

    def structure_key_items(self, *, sorted_keys: bool = False) -> Generator[tuple[SpilledKey, JSONValue], None, None]:
        """Keep original key tokens while reading only each child's structural kind."""
        with closing(self.key_entries(sorted_keys=sorted_keys)) as entries:
            for key, child in entries:
                yield key, _load_structure_node(self._connection, child)

    def collapse_observed_keys(self) -> bool:
        """Apply the existing key policy without reconstructing content-sized names."""
        from polylogue.schemas.field_stats.detection import (
            _HIGH_CARDINALITY_KEY_THRESHOLD,
            should_collapse_observed_keys,
        )

        if len(self) >= _HIGH_CARDINALITY_KEY_THRESHOLD:
            return True
        names: list[str] = []
        with closing(self.key_entries()) as entries:
            for key, _child in entries:
                name = key.small_name
                if name is None:
                    return True
                names.append(name)
        return should_collapse_observed_keys(names)

    def structure_items(self) -> Iterator[tuple[str, JSONValue]]:
        with closing(self.key_entries()) as entries:
            for key, child in entries:
                yield key.read(), _load_structure_node(self._connection, child)

    def sorted_keys(self) -> Iterator[str]:
        with closing(self.key_entries(sorted_keys=True)) as entries:
            for key, _child in entries:
                yield key.read()

    def normalized_sorted_items(self, normalize_key: Callable[[str], str]) -> Iterator[tuple[str, JSONValue]]:
        """Sort normalized keys on disk; refuse collisions without a Python key set."""
        connection = self._connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS json_normalized_keys (parent_id INTEGER, normalized BLOB, child_id INTEGER, PRIMARY KEY(parent_id, normalized)) WITHOUT ROWID"
        )
        connection.execute("DELETE FROM json_normalized_keys WHERE parent_id=?", (self._node_id,))
        cursor = _read_rows(
            connection, "SELECT key_token, child_id FROM json_object_members WHERE parent_id=?", (self._node_id,)
        )
        try:
            for key, child in cursor:
                check_compute_cancelled()
                normalized = normalize_key(SpilledKey(connection, int(key)).read())
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

    def key_union(self, extra: set[str]) -> SpilledObject:
        """Join a bounded preceding key set without copying this object's keys."""
        connection = self._connection
        node = cast(int, connection.execute("INSERT INTO json_nodes(kind) VALUES ('object')").lastrowid)
        connection.execute(
            "INSERT INTO json_object_members SELECT ?, key_token, key_digest, ordinal, child_id FROM json_object_members WHERE parent_id = ?",
            (node, self._node_id),
        )
        ordinal = len(self)
        null = cast(
            int, connection.execute("INSERT INTO json_nodes(kind, scalar_json) VALUES ('scalar', 'null')").lastrowid
        )
        mapping = SpilledObject(connection, node)
        for key in extra:
            if key not in mapping:
                token = _literal_key(connection, key)
                connection.execute(
                    "INSERT INTO json_object_members VALUES (?,?,?,?,?)",
                    (node, token.token, token.digest, ordinal, null),
                )
                ordinal += 1
        return SpilledObject(connection, node)

    def record_profile_groups(
        self, samples: Iterable[JSONDocument], *, record_type_key: str | None, coarse_type: Callable[[object], str]
    ) -> Iterator[tuple[str, Iterator[tuple[SpilledKey, tuple[str, ...]]]]]:
        """Keep exact field unions and their original order in the scalar owner."""
        from polylogue.archive.raw_payload import record_bucket_key

        connection = self._connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS profile_fields (bucket BLOB, key_token INTEGER, key_digest BLOB, kind TEXT, "
            "PRIMARY KEY(bucket,key_token,kind)) WITHOUT ROWID"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS profile_field_digest ON profile_fields(bucket,key_digest)")
        connection.execute("DELETE FROM profile_fields")
        for sample in samples:
            bucket = record_bucket_key(sample, record_type_key).encode("utf-8", "surrogatepass")
            if isinstance(sample, SpilledObject):
                entries = sample.structure_key_items()
            else:
                entries = ((_literal_key(connection, key), value) for key, value in sample.items())
            with closing(entries) as sample_fields:
                for key, value in sample_fields:
                    canonical = key.token
                    with closing(
                        _read_rows(
                            connection,
                            "SELECT DISTINCT key_token FROM profile_fields WHERE bucket=? AND key_digest=?",
                            (bucket, key.digest),
                        )
                    ) as candidates:
                        for (token,) in candidates:
                            if key.compare(SpilledKey(connection, int(token))) == 0:
                                canonical = int(token)
                                break
                    connection.execute(
                        "INSERT OR IGNORE INTO profile_fields VALUES (?,?,?,?)",
                        (bucket, canonical, key.digest, coarse_type(value)),
                    )

        def fields(bucket: bytes) -> Iterator[tuple[SpilledKey, tuple[str, ...]]]:
            with closing(
                _read_rows(
                    connection,
                    "SELECT DISTINCT key_token FROM profile_fields WHERE bucket=? "
                    "ORDER BY CAST(key_token AS TEXT) COLLATE json_key_order LIMIT 24",
                    (bucket,),
                )
            ) as keys:
                for (token,) in keys:
                    with closing(
                        _read_rows(
                            connection,
                            "SELECT kind FROM profile_fields WHERE bucket=? AND key_token=? ORDER BY kind",
                            (bucket, token),
                        )
                    ) as kinds:
                        value_kinds = tuple(row[0] for row in kinds)
                    yield SpilledKey(connection, int(token)), value_kinds

        with closing(_read_rows(connection, "SELECT DISTINCT bucket FROM profile_fields ORDER BY bucket")) as buckets:
            for (bucket,) in buckets:
                yield bytes(bucket).decode("utf-8", "surrogatepass"), fields(bucket)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._member(key) is not None

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Mapping)
            and len(self) == len(other)
            and all(key in other and self[key] == other[key] for key in self)
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("streamed JSON container is read-only")

    clear = pop = popitem = setdefault = update = _immutable
    __setitem__ = __delitem__ = __ior__ = _immutable

    def copy(self) -> dict[str, JSONValue]:
        return dict(self.items())

    # The returned views implement the Mapping protocol against the SQLite
    # tree; ``dict_keys``' static type cannot describe a dict subclass whose
    # values live outside its inherited empty storage.
    def keys(self) -> Any:
        return KeysView(self)

    def values(self) -> Any:
        return _SpilledValuesView(self)

    def items(self) -> Any:
        return _SpilledItemsView(self)


class _SpilledValuesView(ValuesView[JSONValue]):
    def __init__(self, mapping: SpilledObject) -> None:
        super().__init__(mapping)
        self._spill_mapping = mapping

    def __iter__(self) -> Iterator[JSONValue]:
        mapping = self._spill_mapping
        with closing(
            _read_rows(
                mapping._connection,
                "SELECT child_id FROM json_object_members WHERE parent_id = ? ORDER BY ordinal",
                (mapping._node_id,),
            )
        ) as cursor:
            for (child_id,) in cursor:
                yield _load_node(mapping._connection, int(child_id))


class _SpilledItemsView(ItemsView[str, JSONValue]):
    def __init__(self, mapping: SpilledObject) -> None:
        super().__init__(mapping)
        self._spill_mapping = mapping

    def __iter__(self) -> Iterator[tuple[str, JSONValue]]:
        mapping = self._spill_mapping
        with closing(
            _read_rows(
                mapping._connection,
                "SELECT key_token, child_id FROM json_object_members WHERE parent_id = ? ORDER BY ordinal",
                (mapping._node_id,),
            )
        ) as cursor:
            for key_token, child_id in cursor:
                yield (
                    SpilledKey(mapping._connection, int(key_token)).read(),
                    _load_node(mapping._connection, int(child_id)),
                )


class SpilledArray(list[JSONValue], Sequence[JSONValue], _ValidatedJSONContainer):
    def __init__(self, connection: sqlite3.Connection, node_id: int) -> None:
        list.__init__(self)
        self._connection = connection
        self._node_id = node_id

    def structure_values(self) -> Iterator[JSONValue]:
        with closing(
            _read_rows(
                self._connection,
                "SELECT child_id FROM json_array_items WHERE parent_id=? ORDER BY ordinal",
                (self._node_id,),
            )
        ) as rows:
            for (child_id,) in rows:
                yield _load_structure_node(self._connection, int(child_id))

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

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("streamed JSON container is read-only")

    append = clear = extend = insert = pop = remove = reverse = sort = _immutable
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable

    def copy(self) -> list[JSONValue]:
        return list(self)

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


@dataclass(frozen=True, slots=True)
class _ScalarToken:
    kind: str
    ordinal: int


@dataclass(frozen=True)
class _ScalarTokenReference:
    """A completed event value borrowing its exact lexical token owner."""

    store: _ScalarTokenStore
    kind: str
    ordinal: int

    def read(self) -> JSONValue:
        if (
            _read_row(
                self.store.connection,
                "SELECT 1 FROM json_scalar_tokens WHERE kind=? AND token=?",
                (self.kind, self.ordinal),
            )
            is None
        ):
            raise ValueError("incomplete JSON scalar token cannot supply record evidence")
        return self.store.read(self.kind, self.ordinal)


class _ScalarTokenStore:
    """Exact scalar chunks owned by the existing private JSON tree."""

    def __init__(self, connection: sqlite3.Connection, *, allow_nonfinite: bool = True) -> None:
        self.connection = connection
        self.allow_nonfinite = allow_nonfinite
        self.pending = bytearray()
        self.chunk = 0
        self.decoded_bytes = 0
        self.number_integer = True
        self.short_chars = 0
        self.small_bytes = bytearray()
        self.key_digest = hashlib.sha256()
        self.failure: ValueError | UnicodeError | None = None
        self.failure_token: tuple[str, int] | None = None
        self.projection_scope = 0

    def string(self, ordinal: int, content: bytes, final: bool) -> None:
        from polylogue.core.json_envelope import _prefix_cut

        # Chunk sizes are a buffering choice, never a support limit. A chunk
        # boundary cannot split an escape, surrogate pair or UTF-8 character.
        for start in range(0, len(content), 4096):
            self.pending.extend(content[start : start + 4096])
            while len(self.pending) > 4096:
                cut = _prefix_cut(bytes(self.pending[:4096]))
                if not cut:
                    break
                self._string_chunk(ordinal, bytes(self.pending[:cut]))
                del self.pending[:cut]
        if final:
            self._string_chunk(ordinal, bytes(self.pending))
            self.pending.clear()
            self._finish("string", ordinal)

    def _string_chunk(self, ordinal: int, raw: bytes) -> None:
        try:
            decoded = json.loads(b'"' + raw + b'"').encode("utf-8", "surrogatepass")
        except (ValueError, UnicodeError) as error:
            # The structural reader validates this token's complete syntax.
            # Let it reject the stream through the tokenizer so its coroutine
            # chain closes normally; never expose a tree after a sink failure.
            self.failure = error
            if self.failure_token is None:
                self.failure_token = ("string", ordinal)
            return
        self._store("string", ordinal, decoded)

    def number(self, ordinal: int, content: bytes, final: bool) -> None:
        if not self.allow_nonfinite and any(byte in content for byte in (b"N", b"I")):
            raise ValueError("non-finite JSON constant")
        self.number_integer &= not any(byte in content for byte in (b".", b"e", b"E", b"N", b"I"))
        for start in range(0, len(content), 4096):
            self._store("number", ordinal, content[start : start + 4096])
        if final:
            self._finish("number", ordinal)

    def _store(self, kind: str, ordinal: int, content: bytes) -> None:
        check_compute_cancelled()
        self.connection.execute(
            "INSERT INTO json_scalar_chunks VALUES (?, ?, ?, ?)", (kind, ordinal, self.chunk, content)
        )
        if kind == "string":
            self.short_chars = min(129, self.short_chars + len(content.decode("utf-8", "surrogatepass")))
            if self.short_chars <= 128:
                self.small_bytes.extend(content)
            else:
                self.small_bytes.clear()
            self.key_digest.update(content)
        self.chunk += 1
        self.decoded_bytes += len(content)
        advance_work_progress(bytes=len(content))

    def _finish(self, kind: str, ordinal: int) -> None:
        if self.failure_token != (kind, ordinal):
            self.connection.execute(
                "INSERT INTO json_scalar_tokens VALUES (?, ?, ?, ?)",
                (kind, ordinal, self.decoded_bytes, "integer" if kind == "number" and self.number_integer else kind),
            )
            if kind == "string":
                self.connection.execute(
                    "INSERT INTO json_key_meta VALUES (?,?,?,?)",
                    (
                        ordinal,
                        self.short_chars,
                        self.key_digest.digest(),
                        bytes(self.small_bytes) if self.short_chars <= 128 else None,
                    ),
                )
        self.chunk = 0
        self.decoded_bytes = 0
        self.number_integer = True
        self.short_chars = 0
        self.small_bytes = bytearray()
        self.key_digest = hashlib.sha256()

    def read(self, kind: str, ordinal: int) -> JSONValue:
        rows = _read_rows(
            self.connection,
            "SELECT data FROM json_scalar_chunks WHERE kind=? AND token=? ORDER BY ordinal",
            (kind, ordinal),
        )

        def chunks() -> Generator[bytes, None, None]:
            for (content,) in rows:
                check_compute_cancelled()
                yield bytes(content)

        with closing(rows):
            if kind == "string":
                return "".join(chunk.decode("utf-8", "surrogatepass") for chunk in chunks())
            from polylogue.core.json import _decode_integer

            return cast(JSONValue, json.loads(b"".join(chunks()), parse_int=_decode_integer))


def _load_structure_node(connection: sqlite3.Connection, node_id: int) -> JSONValue:
    """Resolve structure-only scalar evidence without decoding scalar content."""
    row = _read_row(
        connection,
        "SELECT node.kind, tokens.json_kind FROM json_nodes node LEFT JOIN json_scalar_tokens tokens "
        "ON tokens.kind=node.kind AND tokens.token=node.token WHERE node.id=?",
        (node_id,),
    )
    if row is None:
        raise ValueError("streamed JSON tree lost a referenced node")
    kind, json_kind = row
    if kind == "string":
        return ""
    if kind == "number":
        return 0 if json_kind == "integer" else 0.0
    return _load_node(connection, node_id)


def _load_node(connection: sqlite3.Connection, node_id: int) -> JSONValue:
    row = _read_row(connection, "SELECT kind, scalar_json, token FROM json_nodes WHERE id = ?", (node_id,))
    if row is None:
        raise ValueError("streamed JSON tree lost a referenced node")
    kind, scalar_json, token = row
    if kind == "object":
        return SpilledObject(connection, node_id)
    if kind == "array":
        return SpilledArray(connection, node_id)
    if token is not None:
        return _ScalarTokenStore(connection).read(kind, int(token))
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
                "INSERT INTO json_object_members VALUES (?, ?, ?, ?, ?)",
                (
                    node,
                    (token := _literal_key(connection, key)).token,
                    token.digest,
                    ordinal,
                    _store_schema_node(connection, item),
                ),
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
    key: SpilledKey | None = None
    ordinal: int = 0


_DIRECT_SURROGATE_PAIR = re.compile("([\ud800-\udbff])([\udc00-\udfff])")


class _ExactJSONText(io.RawIOBase):
    """Transcode bytes without turning direct surrogate units into escapes.

    The exact scalar sink must distinguish direct and escaped units: changing
    a direct high unit to an escape can pair it with an originally escaped
    low unit. Only adjacent directly encoded provider units pair here.
    """

    def __init__(self, handle: Any, encoding: str, *, strip_bom: bool = True, provider_utf8: bool = True) -> None:
        self.handle = handle
        self.decoder = codecs.getincrementaldecoder(encoding)(errors="surrogatepass")
        self.provider_utf8 = provider_utf8 and encoding in {"utf-8", "utf-8-sig"}
        self.strip_bom = strip_bom
        self.pending = bytearray()
        self.held_high = ""
        self.ended = False
        self.started = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer)  # type: ignore[arg-type]
        while not self.pending and not self.ended:
            check_compute_cancelled()
            chunk = self.handle.read(1024 * 1024)
            self.ended = not chunk
            text = self.held_high + self.decoder.decode(chunk, final=self.ended)
            self.held_high = ""
            if self.provider_utf8:
                if not self.ended and text and 0xD800 <= ord(text[-1]) <= 0xDBFF:
                    text, self.held_high = text[:-1], text[-1]
                text = _DIRECT_SURROGATE_PAIR.sub(
                    lambda pair: chr(0x10000 + ((ord(pair[1]) - 0xD800) << 10) + ord(pair[2]) - 0xDC00), text
                )
            if self.strip_bom and not self.started:
                text = text.lstrip("\ufeff")
                self.started = bool(text)
            self.pending.extend(text.encode("utf-8", "surrogatepass"))
        count = min(len(view), len(self.pending))
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count


class StreamedJSONDocument(AbstractContextManager[JSONValue]):
    """Decode JSON to private disk and expose a lazy view.

    With ``jsonl=True``, every whitespace-separated root value is instead
    exposed as an item in one synthetic lazy array.
    """

    def __init__(self, path: Path | None, *, jsonl: bool = False) -> None:
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
            connection.execute("PRAGMA journal_mode=DELETE" if self._path is None else "PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-4096")
            connection.executescript(
                """
                CREATE TABLE json_nodes (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER REFERENCES json_nodes(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    scalar_json TEXT,
                    token INTEGER
                );
                CREATE TABLE json_scalar_tokens (
                    kind TEXT NOT NULL, token INTEGER NOT NULL, decoded_bytes INTEGER NOT NULL, json_kind TEXT NOT NULL,
                    PRIMARY KEY(kind,token)
                ) WITHOUT ROWID;
                CREATE TABLE json_scalar_chunks (
                    kind TEXT NOT NULL, token INTEGER NOT NULL, ordinal INTEGER NOT NULL, data BLOB NOT NULL,
                    PRIMARY KEY(kind,token,ordinal)
                ) WITHOUT ROWID;
                CREATE TABLE json_key_meta(token INTEGER PRIMARY KEY, short_chars INTEGER NOT NULL, digest BLOB NOT NULL, small_bytes BLOB);
                CREATE TABLE json_object_members (
                    parent_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    key_token INTEGER NOT NULL,
                    key_digest BLOB NOT NULL,
                    ordinal INTEGER NOT NULL,
                    child_id INTEGER NOT NULL REFERENCES json_nodes(id) ON DELETE CASCADE,
                    PRIMARY KEY (parent_id, key_token),
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
            connection.execute("CREATE INDEX json_member_digest ON json_object_members(parent_id,key_digest)")
            connection.create_collation(
                "json_key_order",
                lambda left, right: SpilledKey(connection, int(left)).compare(SpilledKey(connection, int(right))),
            )
            connection.execute("PRAGMA foreign_keys=ON")
            self._connection = connection
            self._root_id = (
                self._decode(connection, self._path)
                if self._path is not None
                else cast(int, connection.execute("INSERT INTO json_nodes(kind) VALUES ('array')").lastrowid)
            )
            connection.commit()
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

    def append_document(
        self,
        path: Path,
        *,
        allow_nonfinite: bool = True,
        strip_bom: bool = True,
        provider_utf8: bool = True,
        text_encoding: str | None = None,
    ) -> int:
        """Append one complete record to an empty-path owner's lazy tape.

        A failed attempt leaves no nodes or tokens. The caller may replay
        the same physical line with its decoder's next text policy.
        """
        if self._path is not None or self._jsonl or self._root_id is None:
            raise RuntimeError("record append requires a live empty-path tape")
        connection = self.connection
        connection.execute("SAVEPOINT json_record")
        try:
            node = self._decode(
                connection,
                path,
                allow_nonfinite=allow_nonfinite,
                strip_bom=strip_bom,
                provider_utf8=provider_utf8,
                text_encoding=text_encoding,
            )
            ordinal = int(
                connection.execute(
                    "SELECT COUNT(*) FROM json_array_items WHERE parent_id=?", (self._root_id,)
                ).fetchone()[0]
            )
            connection.execute("UPDATE json_nodes SET parent_id=? WHERE id=?", (self._root_id, node))
            connection.execute("INSERT INTO json_array_items VALUES (?,?,?)", (self._root_id, ordinal, node))
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            for statement in ("ROLLBACK TO json_record", "RELEASE json_record"):
                try:
                    connection.execute(statement)
                except BaseException as cleanup:
                    failures.append(cleanup)
            if len(failures) > 1:
                raise BaseExceptionGroup("JSON record decode and attempt rollback failed", failures) from None
            raise
        connection.execute("RELEASE json_record")
        return node

    def append_value(self, value: JSONValue) -> None:
        """Retain a value already decoded at an explicitly eager boundary."""
        if self._path is not None or self._jsonl or self._root_id is None:
            raise RuntimeError("value append requires a live empty-path tape")
        connection = self.connection
        node = _store_schema_node(connection, value)
        ordinal = int(
            connection.execute("SELECT COUNT(*) FROM json_array_items WHERE parent_id=?", (self._root_id,)).fetchone()[
                0
            ]
        )
        connection.execute("UPDATE json_nodes SET parent_id=? WHERE id=?", (self._root_id, node))
        connection.execute("INSERT INTO json_array_items VALUES (?,?,?)", (self._root_id, ordinal, node))

    def _decode(
        self,
        connection: sqlite3.Connection,
        path: Path,
        *,
        allow_nonfinite: bool = True,
        strip_bom: bool = True,
        provider_utf8: bool = True,
        text_encoding: str | None = None,
    ) -> int:
        stack: list[_Frame] = []
        tokens = _ScalarTokenStore(connection, allow_nonfinite=allow_nonfinite)
        token_offsets = {
            kind: int(
                connection.execute(
                    "SELECT COALESCE(MAX(token),0) FROM json_scalar_tokens WHERE kind=?", (kind,)
                ).fetchone()[0]
            )
            for kind in ("string", "number")
        }
        root_id: int | None = None
        if self._jsonl:
            root_id = cast(int, connection.execute("INSERT INTO json_nodes(kind) VALUES ('array')").lastrowid)
            stack.append(_Frame(root_id, "array"))

        def add_node(kind: str, scalar: object = None) -> int:
            token_id = None
            if isinstance(scalar, _ScalarToken):
                kind = scalar.kind
                token_id = scalar.ordinal
                scalar = None
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
                "INSERT INTO json_nodes(parent_id, kind, scalar_json, token) VALUES (?, ?, ?, ?)",
                (parent_id, kind, scalar_json, token_id),
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
                existing = None
                for candidate in connection.execute(
                    "SELECT key_token,ordinal,child_id FROM json_object_members WHERE parent_id=? AND key_digest=?",
                    (parent.node_id, key.digest),
                ):
                    if key.compare(SpilledKey(connection, int(candidate[0]))) == 0:
                        existing = candidate
                        break
                if existing is None:
                    ordinal = parent.ordinal
                    parent.ordinal = ordinal + 1
                    connection.execute(
                        "INSERT INTO json_object_members VALUES (?,?,?,?,?)",
                        (parent.node_id, key.token, key.digest, ordinal, node_id),
                    )
                else:
                    old_node_id = int(existing[2])
                    connection.execute(
                        "UPDATE json_object_members SET child_id=? WHERE parent_id=? AND key_token=?",
                        (node_id, parent.node_id, existing[0]),
                    )
                    connection.execute("DELETE FROM json_nodes WHERE id=?", (old_node_id,))
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

        from polylogue.core.json_envelope import LexemeAlignedReader, _PrefixStringReader

        with path.open("rb") as stream:
            encoding = text_encoding or json.detect_encoding(stream.read(4))
            stream.seek(0)
            with io.BufferedReader(
                _ExactJSONText(stream, encoding, strip_bom=strip_bom, provider_utf8=provider_utf8)
            ) as reader:
                scalar_reader = _PrefixStringReader(
                    reader,
                    scalar_values=True,
                    string_sink=lambda ordinal, chunk, final: tokens.string(
                        ordinal + token_offsets["string"], chunk, final
                    ),
                    number_sink=lambda ordinal, chunk, final: tokens.number(
                        ordinal + token_offsets["number"], chunk, final
                    ),
                )

                def parsed_events() -> Iterator[tuple[str, object]]:
                    from ijson import sendable_list

                    pending = sendable_list()
                    parser = exact_backend.basic_parse_coro(pending, multiple_values=self._jsonl)
                    aligned = LexemeAlignedReader(scalar_reader)
                    try:
                        while True:
                            chunk = aligned.read(65536)
                            try:
                                parser.send(chunk)
                            except StopIteration:
                                yield from pending
                                return
                            yield from pending
                            pending.clear()
                            if not chunk:
                                return
                    finally:
                        # Own the coroutine even when reading or cancellation
                        # fails upstream, before ijson gets its next chunk.
                        with suppress(BaseException):
                            parser.close()
                        pending.clear()

                events = parsed_events()

                def exact_events() -> Iterator[tuple[str, object]]:
                    string_ordinal = token_offsets["string"]
                    number_ordinal = token_offsets["number"]
                    for event, value in events:
                        if event in {"map_key", "string"}:
                            string_ordinal += 1
                            value = (
                                SpilledKey(connection, string_ordinal)
                                if event == "map_key"
                                else _ScalarToken("string", string_ordinal)
                            )
                        elif event == "number":
                            number_ordinal += 1
                            value = _ScalarToken("number", number_ordinal)
                        yield event, value

                try:
                    parsed_root = self._consume_events(exact_events(), connection, stack, add_node)
                except BaseException:
                    close = getattr(events, "close", None)
                    if callable(close):
                        with suppress(BaseException):
                            close()
                    raise
        if tokens.failure is not None:
            raise tokens.failure
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
        return root_id

    @staticmethod
    def _consume_events(
        events: Iterable[tuple[str, object]],
        connection: sqlite3.Connection,
        stack: list[_Frame],
        add_node: Callable[[str, object], int],
    ) -> int | None:
        root_id = None
        for event, value in events:
            check_compute_cancelled()
            # Scalar chunks report byte progress while they are written, so
            # a long token remains visible and cancellable before its event.
            # Boolean and null literals have no chunk transport.
            if event == "null":
                advance_work_progress(bytes=4)
            elif event == "boolean":
                advance_work_progress(bytes=4 if value else 5)
            if event == "map_key":
                if not stack or stack[-1].kind != "object":
                    raise ValueError("streamed JSON key is outside an object")
                if not isinstance(value, SpilledKey):
                    raise ValueError("streamed JSON key has no exact token")
                stack[-1].key = value
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


@contextmanager
def owned_scalar_events(handle: _Readable) -> Generator[Iterator[tuple[str, object]], None, None]:
    """Borrow exact scalar tokens for a selected streaming projection."""
    from ijson.backends import python as exact_backend

    from polylogue.core.json_envelope import LexemeAlignedReader, _PrefixStringReader

    owner = StreamedJSONDocument(None)
    with owner:
        tokens = _ScalarTokenStore(owner.connection)
        reader = _PrefixStringReader(handle, scalar_values=True, string_sink=tokens.string, number_sink=tokens.number)
        with closing(iter(exact_backend.basic_parse(LexemeAlignedReader(reader)))) as events:

            def exact_events() -> Iterator[tuple[str, object]]:
                strings = numbers = 0
                for event, value in events:
                    if event in {"map_key", "string"}:
                        strings += 1
                        value = (
                            SpilledKey(owner.connection, strings, tokens)
                            if event == "map_key"
                            else _ScalarTokenReference(tokens, "string", strings)
                        )
                    elif event == "number":
                        numbers += 1
                        value = _ScalarTokenReference(tokens, "number", numbers)
                    yield event, value
                if tokens.failure is not None:
                    raise tokens.failure

            with closing(exact_events()) as exact:
                yield exact
