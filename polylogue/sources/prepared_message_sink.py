"""Disk-backed parsed messages for worker preparation and sealed publication."""

from __future__ import annotations

import base64
import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, MutableSequence, Set
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, overload
from urllib.parse import quote

import ijson

from polylogue.sources.decoder_json import _json_subtree, normalize_ijson_stdlib_numbers
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent

_ACTIVE_PARENT_LOOKUP_SQL = (
    "SELECT parent_id FROM prepared_message INDEXED BY prepared_message_provider "
    "WHERE session_ordinal = ? AND provider_id = ? ORDER BY message_ordinal DESC LIMIT 1"
)


def _read_uri(path: Path) -> str:
    return f"file:{quote(str(path))}?mode=ro"


def _message_json(value: ParsedMessage) -> str:
    payload = value.model_dump(mode="json")
    payload["parent_message_position"] = value.parent_message_position
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    return json.dumps(payload, ensure_ascii=False)


def _event_json(value: ParsedSessionEvent) -> str:
    payload = value.model_dump(mode="json")
    payload["boundary_message_position"] = value.boundary_message_position
    return json.dumps(payload, ensure_ascii=False)


def _attachment_json(value: ParsedAttachment) -> str:
    payload = value.model_dump(mode="json")
    payload["message_position"] = value.message_position
    payload["message_variant_index"] = value.message_variant_index
    payload["owner_coordinate"] = asdict(value.owner_coordinate) if value.owner_coordinate is not None else None
    payload["precomputed_blob"] = value.precomputed_blob
    payload["_prepared_inline_bytes"] = (
        base64.b64encode(value.inline_bytes).decode("ascii") if value.inline_bytes is not None else None
    )
    return json.dumps(payload, ensure_ascii=False)


def _decode_attachment(encoded: str, path: Path, session_ordinal: int, attachment_ordinal: int) -> ParsedAttachment:
    payload = json.loads(encoded)
    inline = payload.pop("_prepared_inline_bytes", None)
    if inline is not None:
        payload["inline_bytes"] = base64.b64decode(inline, validate=True)
    return ParsedAttachment.model_validate(payload).model_copy(
        update={"prepared_carrier_key": (str(path), session_ordinal, attachment_ordinal)}
    )


class SqliteMessageSink(MutableSequence[ParsedMessage]):
    """One session's messages, in stable ordinal order without a resident list."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedMessage: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedMessage]: ...

    def __getitem__(self, index: int | slice) -> ParsedMessage | list[ParsedMessage]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        if self._writer is not None:
            row = self._writer.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? AND message_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        else:
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
                row = conn.execute(
                    "SELECT message_json FROM prepared_message WHERE session_ordinal = ? AND message_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        if row is None:
            raise ValueError("prepared message row disappeared")
        return ParsedMessage.model_validate_json(row[0])

    @overload
    def __setitem__(self, index: int, value: ParsedMessage) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedMessage]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedMessage | Iterable[ParsedMessage]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        if isinstance(index, slice):
            raise TypeError("prepared message sink does not support slice replacement")
        if not isinstance(value, ParsedMessage):
            raise TypeError("prepared message replacement must be a ParsedMessage")
        ordinal = self._ordinal(index)
        self._writer.execute(
            "UPDATE prepared_message SET message_json = ?, provider_id = ?, parent_id = ?, active_leaf = ? "
            "WHERE session_ordinal = ? AND message_ordinal = ?",
            (
                _message_json(value),
                value.provider_message_id,
                value.parent_message_provider_id,
                int(bool(value.is_active_leaf)),
                self.session_ordinal,
                ordinal,
            ),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared messages cannot be deleted")

    def insert(self, index: int, value: ParsedMessage) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        if index != self._count:
            raise TypeError("prepared messages can only be appended")
        self._writer.execute(
            "INSERT INTO prepared_message VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.session_ordinal,
                self._count,
                _message_json(value),
                value.provider_message_id,
                value.parent_message_provider_id,
                int(bool(value.is_active_leaf)),
            ),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedMessage]:
        yield from self.iter_from(0)

    def provider_message_ids(self, *, include_none: bool) -> Set[str | None]:
        """A disk-backed set for membership comparison of large sessions."""
        return SqliteProviderMessageIds(self, include_none=include_none)

    def iter_from(self, start: int) -> Iterator[ParsedMessage]:
        """Stream a suffix without decoding or scanning its inherited prefix."""
        if start < 0 or start > self._count:
            raise IndexError(start)
        if self._writer is not None:
            cursor = self._writer.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? "
                "AND message_ordinal >= ? ORDER BY message_ordinal",
                (self.session_ordinal, start),
            )
            for row in cursor:
                yield ParsedMessage.model_validate_json(row[0])
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            cursor = conn.execute(
                "SELECT message_json FROM prepared_message WHERE session_ordinal = ? "
                "AND message_ordinal >= ? ORDER BY message_ordinal",
                (self.session_ordinal, start),
            )
            for row in cursor:
                yield ParsedMessage.model_validate_json(row[0])

    def normalize_active_path(self) -> SqliteMessageSink:
        """Apply the writer's leaf/path normalization without a message list."""
        if self._writer is None:
            # Publication artifacts are immutable. The worker has already
            # normalized them before sealing.
            return self
        leaf_count = self._writer.execute(
            "SELECT COUNT(*) FROM prepared_message WHERE session_ordinal = ? AND active_leaf = 1",
            (self.session_ordinal,),
        ).fetchone()[0]
        if not self._count:
            return self
        if leaf_count != 1:
            for ordinal, active in self._writer.execute(
                "SELECT message_ordinal, active_leaf FROM prepared_message WHERE session_ordinal = ?",
                (self.session_ordinal,),
            ):
                expected = ordinal == self._count - 1
                if bool(active) != expected:
                    message = self[ordinal]
                    self[ordinal] = message.model_copy(update={"is_active_leaf": expected})
            return self
        leaf = self._writer.execute(
            "SELECT provider_id FROM prepared_message WHERE session_ordinal = ? AND active_leaf = 1",
            (self.session_ordinal,),
        ).fetchone()[0]
        if not leaf:
            return self
        self._writer.execute("DROP TABLE IF EXISTS temp.prepared_active_path")
        self._writer.execute("CREATE TEMP TABLE prepared_active_path (provider_id TEXT PRIMARY KEY)")
        cursor: str | None = leaf
        while cursor:
            result = self._writer.execute("INSERT OR IGNORE INTO prepared_active_path VALUES (?)", (cursor,))
            if result.rowcount == 0:
                break
            parent = self._writer.execute(
                _ACTIVE_PARENT_LOOKUP_SQL,
                (self.session_ordinal, cursor),
            ).fetchone()
            cursor = parent[0] if parent is not None else None
        for (ordinal,) in self._writer.execute(
            "SELECT message_ordinal FROM prepared_message WHERE session_ordinal = ? "
            "AND provider_id IN (SELECT provider_id FROM prepared_active_path)",
            (self.session_ordinal,),
        ):
            message = self[ordinal]
            self[ordinal] = message.model_copy(update={"is_active_path": True})
        self._writer.execute("DROP TABLE temp.prepared_active_path")
        return self

    @contextmanager
    def atomic_edit(self) -> Iterator[None]:
        if self._writer is None:
            raise TypeError("sealed prepared messages are immutable")
        name = f"prepared_edit_{uuid.uuid4().hex}"
        count_before = self._count
        self._writer.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self._writer.execute(f"ROLLBACK TO {name}")
            self._writer.execute(f"RELEASE {name}")
            self._count = count_before
            raise
        else:
            self._writer.execute(f"RELEASE {name}")


class SqliteProviderMessageIds(Set[str | None]):
    """Native IDs backed by the prepared-message provider index."""

    def __init__(self, messages: SqliteMessageSink, *, include_none: bool) -> None:
        self.messages = messages
        self.include_none = include_none

    def _connection(self) -> sqlite3.Connection:
        return self.messages._writer or sqlite3.connect(_read_uri(self.messages.path), uri=True)

    def _where(self, alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f"{prefix}session_ordinal = ?" + ("" if self.include_none else f" AND {prefix}provider_id IS NOT NULL")

    def __contains__(self, value: object) -> bool:
        if value is None and not self.include_none:
            return False
        if value is not None and not isinstance(value, str):
            return False
        conn = self._connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM prepared_message WHERE session_ordinal = ? AND provider_id IS ? LIMIT 1",
                (self.messages.session_ordinal, value),
            ).fetchone()
            return row is not None
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __iter__(self) -> Iterator[str | None]:
        conn = self._connection()
        try:
            for (provider_id,) in conn.execute(
                f"SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()} ORDER BY provider_id",
                (self.messages.session_ordinal,),
            ):
                yield provider_id
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __len__(self) -> int:
        conn = self._connection()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT provider_id FROM prepared_message WHERE {self._where()})",
                (self.messages.session_ordinal,),
            ).fetchone()
            return int(row[0])
        finally:
            if conn is not self.messages._writer:
                conn.close()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        if isinstance(other, SqliteProviderMessageIds):
            conn = sqlite3.connect(_read_uri(self.messages.path), uri=True)
            try:
                other_table = "prepared_message"
                if other.messages.path != self.messages.path:
                    conn.execute("ATTACH DATABASE ? AS other_prepared", (_read_uri(other.messages.path),))
                    other_table = "other_prepared.prepared_message"
                left = self._where()
                right = other._where()
                row = conn.execute(
                    f"SELECT 1 FROM (SELECT provider_id FROM prepared_message WHERE {left} "
                    f"EXCEPT SELECT provider_id FROM {other_table} WHERE {right}) LIMIT 1",
                    (self.messages.session_ordinal, other.messages.session_ordinal),
                ).fetchone()
                return row is None
            finally:
                conn.close()
        return all(value in other for value in self)


class SqliteAttachmentSink(MutableSequence[ParsedAttachment]):
    """Ordered attachments retained in the same sealed carrier as messages."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedAttachment: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedAttachment]: ...

    def __getitem__(self, index: int | slice) -> ParsedAttachment | list[ParsedAttachment]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        if self._writer is not None:
            row = self._writer.execute(
                "SELECT attachment_json FROM prepared_attachment WHERE session_ordinal = ? AND attachment_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        else:
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
                row = conn.execute(
                    "SELECT attachment_json FROM prepared_attachment WHERE session_ordinal = ? AND attachment_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        if row is None:
            raise ValueError("prepared attachment row disappeared")
        return _decode_attachment(row[0], self.path, self.session_ordinal, ordinal)

    @overload
    def __setitem__(self, index: int, value: ParsedAttachment) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedAttachment]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedAttachment | Iterable[ParsedAttachment]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared attachments are immutable")
        if isinstance(index, slice) or not isinstance(value, ParsedAttachment):
            raise TypeError("prepared attachment replacement needs one attachment")
        self._writer.execute(
            "UPDATE prepared_attachment SET attachment_json = ? WHERE session_ordinal = ? AND attachment_ordinal = ?",
            (_attachment_json(value), self.session_ordinal, self._ordinal(index)),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared attachments cannot be deleted")

    def insert(self, index: int, value: ParsedAttachment) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared attachments are immutable")
        if index != self._count:
            raise TypeError("prepared attachments can only be appended")
        self._writer.execute(
            "INSERT INTO prepared_attachment VALUES (?, ?, ?)",
            (self.session_ordinal, self._count, _attachment_json(value)),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedAttachment]:
        if self._writer is not None:
            rows = self._writer.execute(
                "SELECT attachment_ordinal, attachment_json FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            for ordinal, encoded in rows:
                yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            rows = conn.execute(
                "SELECT attachment_ordinal, attachment_json FROM prepared_attachment WHERE session_ordinal = ? ORDER BY attachment_ordinal",
                (self.session_ordinal,),
            )
            for ordinal, encoded in rows:
                yield _decode_attachment(encoded, self.path, self.session_ordinal, ordinal)


class SqliteSessionEventSink(MutableSequence[ParsedSessionEvent]):
    """One session's semantic events with disk-backed deterministic ordering."""

    def __init__(
        self,
        path: Path,
        session_ordinal: int,
        *,
        writer: sqlite3.Connection | None = None,
        count: int = 0,
    ) -> None:
        self.path = path
        self.session_ordinal = session_ordinal
        self._writer = writer
        self._count = count

    def __len__(self) -> int:
        return self._count

    def _ordinal(self, index: int) -> int:
        ordinal = index + self._count if index < 0 else index
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError(index)
        return ordinal

    @overload
    def __getitem__(self, index: int) -> ParsedSessionEvent: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedSessionEvent]: ...

    def __getitem__(self, index: int | slice) -> ParsedSessionEvent | list[ParsedSessionEvent]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        ordinal = self._ordinal(index)
        conn = self._writer
        if conn is None:
            with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as reader:
                row = reader.execute(
                    "SELECT event_json FROM prepared_event WHERE session_ordinal = ? AND event_ordinal = ?",
                    (self.session_ordinal, ordinal),
                ).fetchone()
        else:
            row = conn.execute(
                "SELECT event_json FROM prepared_event WHERE session_ordinal = ? AND event_ordinal = ?",
                (self.session_ordinal, ordinal),
            ).fetchone()
        if row is None:
            raise ValueError("prepared event row disappeared")
        return ParsedSessionEvent.model_validate_json(row[0])

    @overload
    def __setitem__(self, index: int, value: ParsedSessionEvent) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ParsedSessionEvent]) -> None: ...

    def __setitem__(self, index: int | slice, value: ParsedSessionEvent | Iterable[ParsedSessionEvent]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        if isinstance(index, slice) or not isinstance(value, ParsedSessionEvent):
            raise TypeError("prepared event replacement needs one event")
        ordinal = self._ordinal(index)
        self._writer.execute(
            "UPDATE prepared_event SET timestamp = ?, event_type = ?, event_json = ? WHERE session_ordinal = ? AND event_ordinal = ?",
            (value.timestamp, value.event_type, _event_json(value), self.session_ordinal, ordinal),
        )

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError("prepared events cannot be deleted")

    def insert(self, index: int, value: ParsedSessionEvent) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        if index < 0:
            index = max(0, self._count + index)
        index = min(index, self._count)
        if index < self._count:
            # A two-phase ordinal shift avoids transient primary-key
            # collisions and keeps Codex's late compaction event insertion
            # on disk rather than copying the complete event list.
            self._writer.execute(
                "UPDATE prepared_event SET event_ordinal = -1 - event_ordinal "
                "WHERE session_ordinal = ? AND event_ordinal >= ?",
                (self.session_ordinal, index),
            )
            self._writer.execute(
                "UPDATE prepared_event SET event_ordinal = -event_ordinal "
                "WHERE session_ordinal = ? AND event_ordinal < 0",
                (self.session_ordinal,),
            )
        self._writer.execute(
            "INSERT INTO prepared_event (session_ordinal, event_ordinal, timestamp, event_type, event_json) VALUES (?, ?, ?, ?, ?)",
            (self.session_ordinal, index, value.timestamp, value.event_type, _event_json(value)),
        )
        self._count += 1

    def __iter__(self) -> Iterator[ParsedSessionEvent]:
        yield from self._iter_query("ORDER BY event_ordinal")

    def _iter_query(self, order_sql: str, parameters: tuple[object, ...] = ()) -> Iterator[ParsedSessionEvent]:
        sql = "SELECT event_json FROM prepared_event WHERE session_ordinal = ? " + order_sql
        if self._writer is not None:
            cursor = self._writer.execute(sql, (self.session_ordinal, *parameters))
            for row in cursor:
                yield ParsedSessionEvent.model_validate_json(row[0])
            return
        with closing(sqlite3.connect(_read_uri(self.path), uri=True)) as conn:
            for row in conn.execute(sql, (self.session_ordinal, *parameters)):
                yield ParsedSessionEvent.model_validate_json(row[0])

    def iter_ordered(self, type_order_tier: Mapping[str, int]) -> Iterator[ParsedSessionEvent]:
        clauses = " ".join("WHEN ? THEN ?" for _ in type_order_tier)
        order = "ORDER BY COALESCE(timestamp, ''), CASE event_type " + clauses + " ELSE 0 END, event_ordinal"
        parameters: tuple[object, ...] = tuple(item for pair in type_order_tier.items() for item in pair)
        yield from self._iter_query(order, parameters)

    def sort_in_place(self, type_order_tier: Mapping[str, int]) -> None:
        if self._writer is None:
            raise TypeError("sealed prepared events are immutable")
        self._writer.execute(
            "UPDATE prepared_event SET sort_tier = 0 WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        for event_type, tier in type_order_tier.items():
            self._writer.execute(
                "UPDATE prepared_event SET sort_tier = ? WHERE session_ordinal = ? AND event_type = ?",
                (tier, self.session_ordinal, event_type),
            )
        # Reorder the core prefix once. Later sidecar events append after it,
        # even when their timestamps are earlier than the core's last event.
        self._writer.execute("DROP TABLE IF EXISTS temp.prepared_event_order")
        self._writer.execute(
            "CREATE TEMP TABLE prepared_event_order AS "
            "SELECT session_ordinal, event_ordinal AS old_ordinal, "
            "ROW_NUMBER() OVER (PARTITION BY session_ordinal "
            "ORDER BY COALESCE(timestamp, ''), sort_tier, event_ordinal) - 1 AS new_ordinal "
            "FROM prepared_event WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute(
            "UPDATE prepared_event SET event_ordinal = -1 - ("
            "SELECT new_ordinal FROM prepared_event_order AS o "
            "WHERE o.session_ordinal = prepared_event.session_ordinal "
            "AND o.old_ordinal = prepared_event.event_ordinal) "
            "WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute(
            "UPDATE prepared_event SET event_ordinal = -1 - event_ordinal WHERE session_ordinal = ?",
            (self.session_ordinal,),
        )
        self._writer.execute("DROP TABLE temp.prepared_event_order")


class SqliteMessageStore:
    """Own the unsealed scratch transaction until its producer has finished."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = DELETE")
        self.conn.execute(
            "CREATE TABLE prepared_message (session_ordinal INTEGER NOT NULL, message_ordinal INTEGER NOT NULL, message_json TEXT NOT NULL, provider_id TEXT, parent_id TEXT, active_leaf INTEGER NOT NULL, PRIMARY KEY (session_ordinal, message_ordinal)) WITHOUT ROWID"
        )
        self.conn.execute(
            "CREATE INDEX prepared_message_provider ON prepared_message(session_ordinal, provider_id, message_ordinal)"
        )
        self.conn.execute(
            "CREATE TABLE prepared_event (session_ordinal INTEGER NOT NULL, event_ordinal INTEGER NOT NULL, timestamp TEXT, event_type TEXT NOT NULL, event_json TEXT NOT NULL, sort_tier INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_ordinal, event_ordinal)) WITHOUT ROWID"
        )
        self.conn.execute(
            "CREATE TABLE prepared_attachment (session_ordinal INTEGER NOT NULL, attachment_ordinal INTEGER NOT NULL, attachment_json TEXT NOT NULL, PRIMARY KEY (session_ordinal, attachment_ordinal)) WITHOUT ROWID"
        )
        self.conn.execute("BEGIN IMMEDIATE")
        self._next_session_ordinal = 0
        self._next_event_ordinal = 0
        self._next_attachment_ordinal = 0

    def new_sink(self) -> SqliteMessageSink:
        sink = SqliteMessageSink(self.path, self._next_session_ordinal, writer=self.conn)
        self._next_session_ordinal += 1
        return sink

    def new_event_sink(self) -> SqliteSessionEventSink:
        sink = SqliteSessionEventSink(self.path, self._next_event_ordinal, writer=self.conn)
        self._next_event_ordinal += 1
        return sink

    def new_attachment_sink(self) -> SqliteAttachmentSink:
        sink = SqliteAttachmentSink(self.path, self._next_attachment_ordinal, writer=self.conn)
        self._next_attachment_ordinal += 1
        return sink

    def close(self) -> None:
        self.conn.close()


class ChatGPTNodeMapping(Mapping[str, object]):
    """Keep node bytes, insertion order, and duplicate-key resolution in scratch."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute(
            "CREATE TABLE chatgpt_node (node_key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, node_json TEXT NOT NULL)"
        )

    def put(self, key: str, node: object, ordinal: int) -> None:
        encoded = json.dumps(node, ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO chatgpt_node VALUES (?, ?, ?) "
            "ON CONFLICT(node_key) DO UPDATE SET node_json = excluded.node_json",
            (key, ordinal, encoded),
        )

    def __getitem__(self, key: str) -> object:
        row = self.conn.execute("SELECT node_json FROM chatgpt_node WHERE node_key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __iter__(self) -> Iterator[str]:
        for (key,) in self.conn.execute("SELECT node_key FROM chatgpt_node ORDER BY ordinal"):
            yield key

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chatgpt_node").fetchone()[0])


class _SingleChatGPTNode(Mapping[str, object]):
    """Expose one node to the canonical normalizer without collecting its peers."""

    def __init__(self, key: str, node: dict[str, object]) -> None:
        self.key = key
        self.node = node

    def __getitem__(self, key: str) -> object:
        if key != self.key:
            raise KeyError(key)
        return self.node

    def __iter__(self) -> Iterator[str]:
        yield self.key

    def __len__(self) -> int:
        return 1


def _simple_chatgpt_node(key: str, node: object) -> bool:
    """A deliberately small shape with no attachment, timing, or event carriers."""
    if not isinstance(node, dict) or set(node) - {"id", "parent", "children", "message"}:
        return False
    if node.get("id") != key or not isinstance(node.get("parent"), (str, type(None))):
        return False
    children = node.get("children", [])
    if not isinstance(children, list) or not all(isinstance(child, str) for child in children):
        return False
    message = node.get("message")
    if not isinstance(message, dict) or set(message) - {
        "id",
        "author",
        "create_time",
        "update_time",
        "content",
        "metadata",
        "status",
        "end_turn",
        "weight",
        "recipient",
    }:
        return False
    if not isinstance(message.get("id"), str) or not message["id"]:
        return False
    author = message.get("author")
    if not isinstance(author, dict) or set(author) - {"role", "name", "metadata"}:
        return False
    if author.get("role") not in {"user", "assistant"} or author.get("metadata", {}) != {}:
        return False
    if not isinstance(author.get("name"), (str, type(None))):
        return False
    if message.get("metadata", {}) != {}:
        return False
    if not isinstance(message.get("create_time"), (int, float, type(None))):
        return False
    if not isinstance(message.get("update_time"), (int, float, type(None))):
        return False
    if not isinstance(message.get("status"), (str, type(None))):
        return False
    if not isinstance(message.get("end_turn"), (bool, type(None))):
        return False
    if message.get("recipient") not in (None, "all"):
        return False
    if message.get("weight", 1) != 1:
        return False
    content = message.get("content")
    if not isinstance(content, dict) or set(content) != {"content_type", "parts"}:
        return False
    parts = content.get("parts")
    if isinstance(parts, list) and any(isinstance(part, str) and "sandbox:" in part for part in parts):
        # The canonical normalizer reports truncated sandbox-link evidence.
        # These links also create attachments, so they must go directly to
        # the collecting fallback without a speculative emitting pass.
        return False
    return (
        content.get("content_type") == "text"
        and isinstance(parts, list)
        and bool(parts)
        and all(isinstance(part, str) for part in parts)
        and any(part for part in parts)
    )


def prepare_simple_chatgpt_mapping(
    envelope: dict[str, object], mapping: ChatGPTNodeMapping, store: SqliteMessageStore, fallback_id: str
) -> ParsedSession | None:
    """Spill a conservative text-only ChatGPT mapping into the ordinary sink.

    Return None for any shape that requires the full parser. The initial scan
    changes only scratch tables; a fallback can ignore them safely.
    """
    from polylogue.sources.parsers import chatgpt
    from polylogue.sources.parsers.base import AdmissionLedger, AdmissionUnit

    if any(
        key != "mapping" and not isinstance(value, (str, int, float, bool, type(None)))
        for key, value in envelope.items()
    ):
        return None
    current_node = envelope.get("current_node")
    if not isinstance(current_node, str) or not current_node or current_node not in mapping:
        return None
    conn = store.conn
    conn.execute(
        "CREATE TABLE chatgpt_simple_node (node_key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, "
        "parent_key TEXT, sibling INTEGER NOT NULL, timestamp REAL, message_id TEXT NOT NULL UNIQUE)"
    )
    conn.execute("CREATE TABLE chatgpt_simple_sibling (parent_key TEXT PRIMARY KEY, next_ordinal INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE chatgpt_simple_child (parent_key TEXT NOT NULL, child_key TEXT NOT NULL, "
        "sibling INTEGER NOT NULL, PRIMARY KEY (parent_key, child_key)) WITHOUT ROWID"
    )
    for ordinal, key in enumerate(mapping):
        node = mapping[key]
        if not _simple_chatgpt_node(key, node):
            return None
        assert isinstance(node, dict)
        parent = node.get("parent")
        sibling_key = parent if isinstance(parent, str) and parent else ""
        children = node.get("children", [])
        assert isinstance(children, list)
        for child_ordinal, child in enumerate(children):
            conn.execute(
                "INSERT OR IGNORE INTO chatgpt_simple_child VALUES (?, ?, ?)",
                (key, child, child_ordinal),
            )
        row = conn.execute(
            "SELECT next_ordinal FROM chatgpt_simple_sibling WHERE parent_key = ?", (sibling_key,)
        ).fetchone()
        sibling = row[0] if row else 0
        conn.execute(
            "INSERT INTO chatgpt_simple_sibling VALUES (?, 1) "
            "ON CONFLICT(parent_key) DO UPDATE SET next_ordinal = next_ordinal + 1",
            (sibling_key,),
        )
        message = node["message"]
        assert isinstance(message, dict)
        if conn.execute("SELECT 1 FROM chatgpt_simple_node WHERE message_id = ?", (message["id"],)).fetchone():
            return None
        conn.execute(
            "INSERT INTO chatgpt_simple_node VALUES (?, ?, ?, ?, ?, ?)",
            (key, ordinal, parent, sibling, chatgpt._coerce_float(message.get("create_time")), message["id"]),
        )

    # The full parser treats a missing current node as no active path. Keep
    # cycle detection in SQLite rather than a set proportional to path depth.
    conn.execute("CREATE TABLE chatgpt_simple_active (node_key TEXT PRIMARY KEY, depth INTEGER NOT NULL)")
    current = envelope.get("current_node")
    depth = 0
    while isinstance(current, str):
        row = conn.execute("SELECT parent_key FROM chatgpt_simple_node WHERE node_key = ?", (current,)).fetchone()
        if row is None or conn.execute("SELECT 1 FROM chatgpt_simple_active WHERE node_key = ?", (current,)).fetchone():
            break
        conn.execute("INSERT INTO chatgpt_simple_active VALUES (?, ?)", (current, depth))
        depth += 1
        current = row[0]
    leaf_row = conn.execute("SELECT node_key FROM chatgpt_simple_active ORDER BY depth LIMIT 1").fetchone()
    active_leaf_node = leaf_row[0] if leaf_row else None
    has_active_path = depth > 0
    has_timestamp = (
        conn.execute("SELECT 1 FROM chatgpt_simple_node WHERE timestamp IS NOT NULL LIMIT 1").fetchone() is not None
    )
    conn.execute(
        "CREATE TABLE chatgpt_simple_message (node_key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, "
        "timestamp REAL, message_json TEXT NOT NULL, provider_id TEXT NOT NULL, parent_key TEXT)"
    )
    conn.execute("CREATE INDEX chatgpt_simple_provider ON chatgpt_simple_message(provider_id)")
    ledger = AdmissionLedger()
    ledger.expect(AdmissionUnit.OUTER_RECORD, 1)
    ledger.materialized(AdmissionUnit.OUTER_RECORD, 0, "conversation")
    default_model = chatgpt._string_value(envelope, "default_model_slug")
    for key in mapping:
        node = mapping[key]
        assert isinstance(node, dict)
        normalized, attachments = chatgpt.extract_messages_from_mapping(
            _SingleChatGPTNode(key, node),
            default_model_slug=default_model,
        )
        if len(normalized) != 1 or attachments:
            return None
        message = normalized[0]
        row = conn.execute(
            "SELECT ordinal, parent_key, sibling, timestamp FROM chatgpt_simple_node WHERE node_key = ?", (key,)
        ).fetchone()
        assert row is not None
        ordinal, parent_key, sibling, timestamp = row
        if parent_key:
            declared = conn.execute(
                "SELECT sibling FROM chatgpt_simple_child WHERE parent_key = ? AND child_key = ?",
                (parent_key, key),
            ).fetchone()
            if declared is not None:
                sibling = declared[0]
        message = message.model_copy(
            update={
                "position": ordinal,
                "branch_index": sibling if parent_key else 0,
                "variant_index": sibling if parent_key else 0,
                "is_active_path": (
                    conn.execute("SELECT 1 FROM chatgpt_simple_active WHERE node_key = ?", (key,)).fetchone()
                    is not None
                    if has_active_path
                    else None
                ),
                "is_active_leaf": key == active_leaf_node if active_leaf_node is not None else None,
                "parent_message_provider_id": parent_key or None,
            }
        )
        conn.execute(
            "INSERT INTO chatgpt_simple_message VALUES (?, ?, ?, ?, ?, ?)",
            (key, ordinal, timestamp, _message_json(message), message.provider_message_id, parent_key),
        )
        content = node["message"]["content"]
        parts = content["parts"]
        ledger.expect(AdmissionUnit.MESSAGE, 1)
        ledger.materialized(AdmissionUnit.MESSAGE, ordinal, key)
        ledger.expect(AdmissionUnit.PART, len(parts))
        for _ in parts:
            ledger.materialized(AdmissionUnit.PART, ledger.next_ordinal(AdmissionUnit.PART), "text")
        ledger.expect(AdmissionUnit.BLOCK, len(message.blocks))
        for block in message.blocks:
            ledger.materialized(AdmissionUnit.BLOCK, ledger.next_ordinal(AdmissionUnit.BLOCK), block.type.value)

    sink = store.new_sink()
    order = "ORDER BY timestamp IS NULL, timestamp, ordinal" if has_timestamp else "ORDER BY ordinal"
    for _node_key, encoded, parent_key in conn.execute(
        f"SELECT node_key, message_json, parent_key FROM chatgpt_simple_message {order}"
    ):
        message = ParsedMessage.model_validate_json(encoded)
        if parent_key:
            owner = conn.execute(
                "SELECT provider_id FROM chatgpt_simple_message WHERE node_key = ?", (parent_key,)
            ).fetchone()
            if owner is None:
                owner = conn.execute(
                    "SELECT provider_id FROM chatgpt_simple_message WHERE provider_id = ? LIMIT 1", (parent_key,)
                ).fetchone()
            message = message.model_copy(update={"parent_message_provider_id": owner[0] if owner else None})
        sink.append(message)
    shell = chatgpt.parse({**envelope, "mapping": {}}, fallback_id)
    return shell.model_copy(
        update={
            "messages": sink,
            "active_leaf_message_provider_id": (
                conn.execute(
                    "SELECT provider_id FROM chatgpt_simple_message WHERE node_key = ?", (active_leaf_node,)
                ).fetchone()[0]
                if active_leaf_node is not None
                else None
            ),
            "unit_accounting": ledger.close(),
        }
    )


def read_chatgpt_mapping_object(
    handle: BinaryIO, conn: sqlite3.Connection
) -> tuple[dict[str, object], ChatGPTNodeMapping] | None:
    """Consume a complete native object while writing each mapping node immediately."""
    events = iter(ijson.parse(handle))
    mapping = ChatGPTNodeMapping(conn)
    if next(events, None) != ("", "start_map", None):
        return None
    envelope: dict[str, object] = {}
    mapping_count = 0
    for prefix, event, value in events:
        if prefix != "" or event != "map_key":
            continue
        key = str(value)
        next_event = next(events, None)
        if next_event is None:
            raise ValueError("incomplete ChatGPT object")
        child_prefix, child_event, child_value = next_event
        if key != "mapping" or child_event != "start_map":
            envelope[key] = normalize_ijson_stdlib_numbers(_json_subtree(events, child_event, child_value))
            continue
        mapping_count += 1
        if mapping_count != 1:
            return None
        ordinal = 0
        for node_prefix, node_event, node_value in events:
            if node_prefix == "mapping" and node_event == "end_map":
                break
            if node_prefix != "mapping" or node_event != "map_key":
                raise ValueError("invalid ChatGPT mapping structure")
            node_key = str(node_value)
            node_start = next(events, None)
            if node_start is None:
                raise ValueError("incomplete ChatGPT mapping node")
            node = normalize_ijson_stdlib_numbers(_json_subtree(events, node_start[1], node_start[2]))
            mapping.put(node_key, node, ordinal)
            ordinal += 1  # noqa: SIM113  (nested value events are not node ordinals)
        else:
            raise ValueError("incomplete ChatGPT mapping")
    if (
        mapping_count != 1
        or not mapping
        or not isinstance(envelope.get("current_node"), str)
        or not isinstance(envelope.get("create_time"), (int, float))
        or not isinstance(envelope.get("conversation_id"), str)
        and not isinstance(envelope.get("id"), str)
    ):
        return None
    envelope["mapping"] = mapping
    return envelope, mapping


__all__ = [
    "SqliteMessageSink",
    "SqliteMessageStore",
    "SqliteAttachmentSink",
    "SqliteSessionEventSink",
    "ChatGPTNodeMapping",
    "read_chatgpt_mapping_object",
]
