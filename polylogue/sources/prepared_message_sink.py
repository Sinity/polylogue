"""Disk-backed parsed messages for worker preparation and sealed publication."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, MutableSequence, Set
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import overload
from urllib.parse import quote

from polylogue.sources.parsers.base import ParsedMessage, ParsedSessionEvent

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
        self.conn.execute("BEGIN IMMEDIATE")
        self._next_session_ordinal = 0
        self._next_event_ordinal = 0

    def new_sink(self) -> SqliteMessageSink:
        sink = SqliteMessageSink(self.path, self._next_session_ordinal, writer=self.conn)
        self._next_session_ordinal += 1
        return sink

    def new_event_sink(self) -> SqliteSessionEventSink:
        sink = SqliteSessionEventSink(self.path, self._next_event_ordinal, writer=self.conn)
        self._next_event_ordinal += 1
        return sink

    def close(self) -> None:
        self.conn.close()


__all__ = ["SqliteMessageSink", "SqliteMessageStore", "SqliteSessionEventSink"]
