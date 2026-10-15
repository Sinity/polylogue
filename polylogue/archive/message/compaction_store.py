"""Request-owned disk rows and exact streaming JSON measures for compaction."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict
from typing import Any, cast

from polylogue.archive.message.token_measure import TextMeasure, measure


def compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class CompactionStore:
    """One input row at a time; only a fitting final pack becomes Python tuples.

    Payload chunks avoid imposing SQLite's single-value length on a valid
    transcript row. Ranking, run membership and omission prefix measures stay
    private to this request; this connection never attaches archive tiers.
    """

    def __init__(self, connection: sqlite3.Connection, checkpoint: Callable[[], None]) -> None:
        self.connection = connection
        self.checkpoint = checkpoint
        # No multi-row atomicity is needed for disposable request rows.
        # Autocommit bounds the native MEMORY journal as well as Python memory.
        connection.isolation_level = None
        with closing(connection.execute("PRAGMA temp_store=FILE")):
            pass
        self._payload = 0
        self.omission_count = 0
        self.included_count = 0
        self.dropped_count = 0
        self._candidate_measure = TextMeasure.text("[")
        self._active_count = 0
        with closing(
            connection.executescript(
                "CREATE TABLE payload_chunks(payload INTEGER,part INTEGER,text BLOB NOT NULL,PRIMARY KEY(payload,part));"
                "CREATE TABLE compact_sessions(id BLOB PRIMARY KEY,parent_id BLOB,branch BLOB,unresolved INTEGER NOT NULL);"
                "CREATE TABLE compact_messages(session_id BLOB,position INTEGER,message_id BLOB,payload INTEGER NOT NULL,"
                "PRIMARY KEY(session_id,position));"
                "CREATE INDEX compact_message_identity ON compact_messages(session_id,message_id);"
                "CREATE TABLE compact_candidates(id INTEGER PRIMARY KEY,ref BLOB NOT NULL,session_id BLOB NOT NULL,"
                "origin BLOB NOT NULL,score REAL NOT NULL,run INTEGER NOT NULL,original INTEGER NOT NULL,"
                "payload INTEGER NOT NULL,reference INTEGER NOT NULL,retained INTEGER NOT NULL,active INTEGER NOT NULL DEFAULT 1,prefix BLOB);"
                "CREATE INDEX compact_rank ON compact_candidates(active,score DESC,ref,id);"
                "CREATE INDEX compact_runs ON compact_candidates(run,id);"
                "CREATE INDEX compact_candidate_session ON compact_candidates(session_id,active);"
                "CREATE TABLE compact_omissions(position INTEGER PRIMARY KEY,payload INTEGER NOT NULL,prefix BLOB NOT NULL);"
                "CREATE TABLE compact_totals(session_id BLOB PRIMARY KEY,included INTEGER NOT NULL DEFAULT 0,"
                "dropped INTEGER NOT NULL DEFAULT 0,has_drop INTEGER NOT NULL DEFAULT 0,inc_order INTEGER,drop_order INTEGER,"
                "inc_prefix BLOB,drop_prefix BLOB);"
            )
        ):
            pass

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.checkpoint()
        with closing(self.connection.execute(sql, self._parameters(params))):
            pass

    @staticmethod
    def _parameters(params: tuple[object, ...]) -> tuple[object, ...]:
        # Opaque native keys may contain lone surrogates. A local BLOB
        # representation preserves them without a process-global SQL adapter
        # or any change to the public JSON encoding policy.
        return tuple(value.encode("utf-8", "surrogatepass") if isinstance(value, str) else value for value in params)

    @staticmethod
    def _value(value: Any) -> Any:
        return value.decode("utf-8", "surrogatepass") if isinstance(value, bytes) else value

    def rows(self, sql: str, params: tuple[object, ...] = ()) -> Iterator[tuple[Any, ...]]:
        self.checkpoint()
        with closing(self.connection.execute(sql, self._parameters(params))) as cursor:
            for row in cursor:
                self.checkpoint()
                yield tuple(self._value(value) for value in row)

    def scalar(self, sql: str, params: tuple[object, ...] = ()) -> Any:
        with closing(self.connection.execute(sql, self._parameters(params))) as cursor:
            row = cursor.fetchone()
        return self._value(row[0]) if row is not None else None

    def store(self, value: object) -> int:
        self._payload += 1
        identifier = self._payload
        text = compact_json(value)
        for part, offset in enumerate(range(0, len(text), 16_384)):
            self.execute("INSERT INTO payload_chunks VALUES(?,?,?)", (identifier, part, text[offset : offset + 16_384]))
        return identifier

    def load(self, identifier: int) -> object:
        return cast(
            object,
            json.loads(
                "".join(
                    str(row[0])
                    for row in self.rows("SELECT text FROM payload_chunks WHERE payload=? ORDER BY part", (identifier,))
                )
            ),
        )

    def payload_parts(self, identifier: int) -> Iterator[str]:
        for row in self.rows("SELECT text FROM payload_chunks WHERE payload=? ORDER BY part", (identifier,)):
            yield str(row[0])

    def add_session(self, identifier: str, parent: str | None) -> None:
        self.execute(
            "INSERT INTO compact_sessions VALUES(?,?,NULL,?) ON CONFLICT(id) DO UPDATE SET "
            "parent_id=excluded.parent_id,branch=NULL,unresolved=excluded.unresolved",
            (identifier, None, int(bool(parent))),
        )
        self.execute("DELETE FROM compact_messages WHERE session_id=?", (identifier,))
        self.execute("INSERT OR IGNORE INTO compact_totals(session_id) VALUES(?)", (identifier,))

    def add_message(self, session_id: str, position: int, identifier: str, value: Mapping[str, object]) -> None:
        payload = self.store(value)
        self.execute("INSERT INTO compact_messages VALUES(?,?,?,?)", (session_id, position, identifier, payload))

    def add_link(self, child: str, parent: str, branch: str | None) -> None:
        self.execute(
            "UPDATE compact_sessions SET parent_id=?,branch=?,unresolved=0 WHERE id=?", (parent, branch, child)
        )

    def parent_has(self, parent: str | None, message: str) -> bool:
        return bool(
            parent
            and self.scalar(
                "SELECT 1 FROM compact_messages WHERE session_id=? AND message_id=? LIMIT 1", (parent, message)
            )
        )

    def add_candidate(self, value: dict[str, object], *, reference: str, run: int | None, tokens: int) -> int:
        identifier = int(self.scalar("SELECT COALESCE(MAX(id),0)+1 FROM compact_candidates") or 1)
        payload = self.store(value)
        anchor = value["anchor"]
        assert isinstance(anchor, dict)
        score = value["score"]
        assert isinstance(score, float)
        self.execute(
            "INSERT INTO compact_candidates(id,ref,session_id,origin,score,run,original,payload,reference,retained) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                reference,
                str(value["session_id"]),
                str(value["material_origin"]),
                score,
                run or identifier,
                payload,
                payload,
                self.store(anchor["ref"]),
                tokens,
            ),
        )
        return run or identifier

    def candidates(self, *, reverse: bool = False) -> Iterator[tuple[int, dict[str, object]]]:
        order = "score,ref DESC,id DESC" if reverse else "score DESC,ref,id"
        for row in self.rows(f"SELECT id,payload FROM compact_candidates WHERE active=1 ORDER BY {order}"):
            value = self.load(int(row[1]))
            assert isinstance(value, dict)
            yield int(row[0]), value

    def update_candidate(self, identifier: int, value: dict[str, object]) -> None:
        self.execute("UPDATE compact_candidates SET payload=? WHERE id=?", (self.store(value), identifier))

    def add_omission(self, value: dict[str, object]) -> None:
        previous = self.scalar("SELECT prefix FROM compact_omissions WHERE position=?", (self.omission_count - 1,))
        prefix = TextMeasure(**json.loads(str(previous))) if previous is not None else TextMeasure.text("[")
        if self.omission_count:
            prefix = prefix.then(TextMeasure.text(","))
        prefix = prefix.then(TextMeasure.text(compact_json(value)))
        self.execute(
            "INSERT OR REPLACE INTO compact_omissions VALUES(?,?,?)",
            (self.omission_count, self.store(value), compact_json(asdict(prefix))),
        )
        self.omission_count += 1

    def transfer(self, identifier: int, session: str, remaining: int) -> int:
        previous = int(self.scalar("SELECT retained FROM compact_candidates WHERE id=?", (identifier,)) or 0)
        self.execute("UPDATE compact_candidates SET retained=? WHERE id=?", (remaining, identifier))
        self.drop_tokens(session, previous - remaining)
        return previous - remaining

    def drop_tokens(self, session: str, tokens: int) -> None:
        self.execute("UPDATE compact_totals SET dropped=dropped+?,has_drop=1 WHERE session_id=?", (tokens, session))

    def item_parts(self, identifier: int, value: dict[str, object]) -> Iterator[str]:
        yield "{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ","
            yield compact_json(key) + ":"
            occurrences = value["occurrence_count"]
            assert isinstance(occurrences, int)
            if key == "refs" and occurrences > 1:
                yield "["
                run = self.scalar("SELECT run FROM compact_candidates WHERE id=?", (identifier,))
                for position, row in enumerate(
                    self.rows("SELECT reference FROM compact_candidates WHERE run=? ORDER BY id", (run,))
                ):
                    if position:
                        yield ","
                    yield from self.payload_parts(int(row[0]))
                yield "]"
            else:
                yield compact_json(value[key])
        yield "}"

    def refresh_totals(self) -> None:
        self.execute(
            "UPDATE compact_totals SET included=COALESCE((SELECT SUM(retained) FROM compact_candidates "
            "WHERE active=1 AND session_id=compact_totals.session_id),0),inc_order=NULL,drop_order=NULL"
        )
        for column, predicate in (
            (
                "inc",
                "EXISTS(SELECT 1 FROM compact_candidates c WHERE c.active=1 AND "
                "c.session_id=compact_totals.session_id)",
            ),
            ("drop", "has_drop=1"),
        ):
            prefix = TextMeasure.text("{")
            count = 0
            field = "included" if column == "inc" else "dropped"
            for row in self.rows(
                f"SELECT session_id,{field} FROM compact_totals WHERE {predicate} ORDER BY session_id"
            ):
                if count:
                    prefix = prefix.then(TextMeasure.text(","))
                prefix = prefix.then(TextMeasure.text(compact_json(str(row[0])) + ":" + compact_json(row[1])))
                self.execute(
                    f"UPDATE compact_totals SET {column}_order=?,{column}_prefix=? WHERE session_id=?",
                    (count, compact_json(asdict(prefix)), str(row[0])),
                )
                count += 1
            if column == "inc":
                self.included_count = count
            else:
                self.dropped_count = count

    def refresh(self) -> None:
        self.refresh_totals()
        prefix = TextMeasure.text("[")
        count = 0
        for identifier, value in self.candidates():
            if count:
                prefix = prefix.then(TextMeasure.text(","))
            prefix = prefix.then(measure(self.item_parts(identifier, value)))
            self.execute(
                "UPDATE compact_candidates SET prefix=? WHERE id=?", (compact_json(asdict(prefix)), identifier)
            )
            count += 1
        self._candidate_measure = prefix
        self._active_count = count

    def pop_candidate(self) -> tuple[int, dict[str, object]]:
        iterator = self.candidates(reverse=True)
        try:
            identifier, value = next(iterator)
        finally:
            iterator.close()
        self.execute("UPDATE compact_candidates SET active=0 WHERE id=?", (identifier,))
        self._active_count -= 1
        prefix = self.scalar(
            "SELECT prefix FROM compact_candidates WHERE active=1 ORDER BY score,ref DESC,id DESC LIMIT 1"
        )
        self._candidate_measure = TextMeasure(**json.loads(str(prefix))) if prefix else TextMeasure.text("[")
        return identifier, value

    @property
    def count(self) -> int:
        return self._active_count

    def array_measure(self, name: str) -> TextMeasure:
        if name == "items":
            prefix = self._candidate_measure
        else:
            raw = self.scalar("SELECT prefix FROM compact_omissions WHERE position=?", (self.omission_count - 1,))
            prefix = TextMeasure(**json.loads(str(raw))) if raw else TextMeasure.text("[")
        return prefix.then(TextMeasure.text("]"))

    def totals_measure(self, name: str) -> TextMeasure:
        column = "inc" if name == "included_tokens_by_session" else "drop"
        count = self.included_count if column == "inc" else self.dropped_count
        raw = self.scalar(f"SELECT {column}_prefix FROM compact_totals WHERE {column}_order=?", (count - 1,))
        prefix = TextMeasure(**json.loads(str(raw))) if raw else TextMeasure.text("{")
        return prefix.then(TextMeasure.text("}"))

    def array_parts(self, name: str) -> Iterator[str]:
        yield "["
        if name == "items":
            for index, (identifier, value) in enumerate(self.candidates()):
                if index:
                    yield ","
                yield from self.item_parts(identifier, value)
        else:
            for index, row in enumerate(
                self.rows(
                    "SELECT payload FROM compact_omissions WHERE position<? ORDER BY position", (self.omission_count,)
                )
            ):
                if index:
                    yield ","
                yield from self.payload_parts(int(row[0]))
        yield "]"

    def totals_parts(self, name: str) -> Iterator[str]:
        column = "inc" if name == "included_tokens_by_session" else "drop"
        field = "included" if column == "inc" else "dropped"
        count = self.included_count if column == "inc" else self.dropped_count
        yield "{"
        for index, row in enumerate(
            self.rows(
                f"SELECT session_id,{field} FROM compact_totals WHERE {column}_order<? ORDER BY {column}_order",
                (count,),
            )
        ):
            if index:
                yield ","
            yield compact_json(str(row[0])) + ":" + compact_json(row[1])
        yield "}"


@contextmanager
def compaction_store(checkpoint: Callable[[], None]) -> Iterator[CompactionStore]:
    """Keep scratch SQL and its directory on the actual native creator."""
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    with scratch_connection_context(prefix="polylogue-compaction-", filename="compaction.sqlite") as connection:
        yield CompactionStore(connection, checkpoint)
