"""Disposable disk state for a streamed Claude Code normalization pass."""

from __future__ import annotations

import pickle
import sqlite3
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from polylogue.sources.pickle_spool import PickleSpool
from polylogue.sources.providers.claude_code_models import ClaudeCodeBackgroundTaskNotification


class SqliteStringSet:
    def __init__(self, scratch: ClaudeStreamScratch, scope: str) -> None:
        self._scratch = scratch
        self._scope = scope

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        return (
            self._scratch.db.execute(
                "SELECT 1 FROM strings WHERE scope = ? AND value = ?", (self._scope, value)
            ).fetchone()
            is not None
        )

    def add(self, value: str) -> None:
        self._scratch.db.execute("INSERT OR IGNORE INTO strings (scope, value) VALUES (?, ?)", (self._scope, value))


class SqliteNotifications:
    def __init__(self, scratch: ClaudeStreamScratch, scope: str) -> None:
        self._scratch = scratch
        self._scope = scope

    def append(self, item: tuple[ClaudeCodeBackgroundTaskNotification, str | None, str | None]) -> None:
        notification, source_id, timestamp = item
        self._scratch.db.execute(
            "INSERT INTO notifications (scope, task, tool, payload, source_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._scope,
                notification.task_id,
                notification.tool_use_id or "",
                notification.model_dump_json(),
                source_id,
                timestamp,
            ),
        )

    def __iter__(self) -> Iterator[tuple[ClaudeCodeBackgroundTaskNotification, str | None, str | None]]:
        rows = self._scratch.db.execute(
            "SELECT payload, source_id, timestamp FROM notifications WHERE scope = ? ORDER BY seq", (self._scope,)
        )
        for payload, source_id, timestamp in rows:
            yield ClaudeCodeBackgroundTaskNotification.model_validate_json(payload), source_id, timestamp

    def iter_final(self) -> Iterator[tuple[ClaudeCodeBackgroundTaskNotification, str | None, str | None]]:
        rows = self._scratch.db.execute(
            "SELECT n.payload, n.source_id, n.timestamp FROM notifications AS n "
            "JOIN (SELECT task, tool, MIN(seq) AS first_seq, MAX(seq) AS last_seq "
            "FROM notifications WHERE scope = ? GROUP BY task, tool) AS latest ON n.seq = latest.last_seq "
            "ORDER BY latest.first_seq",
            (self._scope,),
        )
        for payload, source_id, timestamp in rows:
            yield ClaudeCodeBackgroundTaskNotification.model_validate_json(payload), source_id, timestamp


class SqlitePickleMap(MutableMapping[str, Any]):
    def __init__(self, scratch: ClaudeStreamScratch, scope: str) -> None:
        self._scratch = scratch
        self._scope = scope

    def __getitem__(self, key: str) -> Any:
        row = self._scratch.db.execute(
            "SELECT payload FROM mapped WHERE scope = ? AND key = ?", (self._scope, key)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return pickle.loads(row[0])

    def __setitem__(self, key: str, value: Any) -> None:
        self._scratch.db.execute(
            "INSERT OR REPLACE INTO mapped (scope, key, payload) VALUES (?, ?, ?)",
            (self._scope, key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)),
        )

    def __delitem__(self, key: str) -> None:
        cursor = self._scratch.db.execute("DELETE FROM mapped WHERE scope = ? AND key = ?", (self._scope, key))
        if cursor.rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for (key,) in self._scratch.db.execute("SELECT key FROM mapped WHERE scope = ? ORDER BY key", (self._scope,)):
            yield key

    def __len__(self) -> int:
        row = self._scratch.db.execute("SELECT COUNT(*) FROM mapped WHERE scope = ?", (self._scope,)).fetchone()
        return int(row[0]) if row else 0

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._scratch.db.execute(
            "INSERT OR IGNORE INTO mapped (scope, key, payload) VALUES (?, ?, ?)",
            (self._scope, key, pickle.dumps(default, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        return self[key]


@dataclass(frozen=True, slots=True)
class _BorrowedPrefix:
    connection_identity: int
    node_id: int


class ClaudeStreamScratch:
    def __init__(self) -> None:
        scratch_root = Path("/realm/tmp/work")
        self._directory = TemporaryDirectory(
            prefix="polylogue-claude-", dir=scratch_root if scratch_root.is_dir() else None
        )
        self.db = sqlite3.connect(Path(self._directory.name) / "stream.sqlite")
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript(
            "CREATE TABLE strings (scope TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (scope, value));"
            "CREATE TABLE notifications (seq INTEGER PRIMARY KEY, scope TEXT NOT NULL, task TEXT NOT NULL, "
            "tool TEXT NOT NULL, payload TEXT NOT NULL, source_id TEXT, timestamp TEXT);"
            "CREATE INDEX notifications_scope ON notifications (scope, task, tool);"
            "CREATE TABLE starts (scope TEXT NOT NULL, task TEXT NOT NULL, tool TEXT NOT NULL, "
            "message_index INTEGER NOT NULL, block_index INTEGER NOT NULL);"
            "CREATE INDEX starts_exact ON starts (scope, task, tool);"
            "CREATE INDEX starts_task ON starts (scope, task);"
            "CREATE TABLE mapped (scope TEXT NOT NULL, key TEXT NOT NULL, payload BLOB NOT NULL, "
            "PRIMARY KEY (scope, key)) WITHOUT ROWID;"
        )
        self._prefix_owners: dict[int, sqlite3.Connection] = {}
        try:
            self._prefix: PickleSpool[tuple[int, object]] = PickleSpool()
        except BaseException:
            self.db.close()
            self._directory.cleanup()
            raise

    def string_set(self, scope: str) -> SqliteStringSet:
        return SqliteStringSet(self, scope)

    def notifications(self, scope: str) -> SqliteNotifications:
        return SqliteNotifications(self, scope)

    def mapped(self, scope: str) -> SqlitePickleMap:
        return SqlitePickleMap(self, scope)

    def add_prefix(self, position: int, item: object) -> None:
        from polylogue.schemas.observation_spill import SpilledArray, SpilledObject

        if isinstance(item, SpilledObject | SpilledArray):
            # The parser caller retains its completed tape through replay and
            # session finalization. Borrow its owner once, never pickle a view
            # or reconstruct unknown values merely to remember a prefix.
            identity = id(item._connection)
            self._prefix_owners[identity] = item._connection
            item = _BorrowedPrefix(identity, item._node_id)
        self._prefix.append((position, item))

    def prefix_count(self) -> int:
        return len(self._prefix)

    def iter_prefix(self) -> Iterator[tuple[int, object]]:
        from polylogue.schemas.observation_spill import _load_node

        for position, item in self._prefix:
            if isinstance(item, _BorrowedPrefix):
                item = _load_node(self._prefix_owners[item.connection_identity], item.node_id)
            yield position, item

    def clear_prefix(self) -> None:
        self._prefix.close()
        self._prefix = PickleSpool()
        self._prefix_owners.clear()

    def add_start(self, scope: str, task: str, tool: str, message_index: int, block_index: int) -> None:
        self.db.execute("INSERT INTO starts VALUES (?, ?, ?, ?, ?)", (scope, task, tool, message_index, block_index))

    def unique_start(self, scope: str, task: str, tool: str | None) -> tuple[int, int] | None:
        if tool is None:
            rows = self.db.execute(
                "SELECT message_index, block_index FROM starts WHERE scope = ? AND task = ? LIMIT 2",
                (scope, task),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT message_index, block_index FROM starts WHERE scope = ? AND task = ? AND tool = ? LIMIT 2",
                (scope, task, tool),
            ).fetchall()
        return (int(rows[0][0]), int(rows[0][1])) if len(rows) == 1 else None

    def close(self) -> None:
        try:
            self._prefix.close()
        finally:
            self._prefix_owners.clear()
            try:
                self.db.close()
            finally:
                self._directory.cleanup()

    def __enter__(self) -> ClaudeStreamScratch:
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()
