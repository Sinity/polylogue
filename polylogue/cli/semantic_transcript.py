"""Private semantic dependencies borrowing one CLI transcript descriptor."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Generator, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import overload

from polylogue.cli.transcript import TranscriptRecords
from polylogue.core.enums import BlockType
from polylogue.rendering.semantic_card_models import LineageDescriptor, SemanticTranscriptEntry, TranscriptNoticeSource
from polylogue.rendering.semantic_cards import (
    SemanticBlockCoordinate,
    SemanticResultMatch,
    iter_semantic_entries,
    normalized_semantic_messages,
    semantic_tool_occurrences,
)


class _Ordinals(MutableMapping[str, int]):
    def __init__(self, owner: SemanticTranscriptRecords) -> None:
        self.owner = owner

    def __getitem__(self, key: str) -> int:
        self.owner.check()
        row = self.owner.connection.execute("SELECT value FROM counters WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return int(row[0])

    def __setitem__(self, key: str, value: int) -> None:
        self.owner.check()
        self.owner.connection.execute("INSERT OR REPLACE INTO counters VALUES(?,?)", (key, value))

    def __delitem__(self, key: str) -> None:
        self.owner.check()
        self.owner.connection.execute("DELETE FROM counters WHERE key=?", (key,))

    def __iter__(self) -> Iterator[str]:
        self.owner.check()
        with self.owner.retain_cursor(self.owner.connection.execute("SELECT key FROM counters")) as cursor:
            for row in cursor:
                self.owner.check()
                yield str(row[0])

    def __len__(self) -> int:
        self.owner.check()
        return int(self.owner.connection.execute("SELECT COUNT(*) FROM counters").fetchone()[0])


class _NoticeRange(Sequence[TranscriptNoticeSource]):
    def __init__(self, owner: SemanticTranscriptRecords, indices: range) -> None:
        self.owner, self.indices = owner, indices

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> TranscriptNoticeSource: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[TranscriptNoticeSource]: ...
    def __getitem__(self, index: int | slice) -> TranscriptNoticeSource | Sequence[TranscriptNoticeSource]:
        if isinstance(index, slice):
            return _NoticeRange(self.owner, self.indices[index])
        self.owner.check()
        row = self.owner.connection.execute(
            "SELECT payload FROM notices WHERE position=?", (self.indices[index],)
        ).fetchone()
        return TranscriptNoticeSource(**json.loads(row[0]))

    def __iter__(self) -> Generator[TranscriptNoticeSource, None, None]:
        if self.indices.step != 1:
            for index in range(len(self)):
                yield self[index]
            return
        self.owner.check()
        with self.owner.retain_cursor(
            self.owner.connection.execute(
                "SELECT payload FROM notices WHERE position>=? AND position<? ORDER BY position",
                (self.indices.start, self.indices.stop),
            )
        ) as cursor:
            for row in cursor:
                self.owner.check()
                yield TranscriptNoticeSource(**json.loads(row[0]))


class _PendingNotices:
    def __init__(self, owner: SemanticTranscriptRecords) -> None:
        self.owner, self.start, self.stop = owner, 0, 0

    def append(self, source: TranscriptNoticeSource) -> None:
        self.owner.check()
        self.owner.connection.execute("INSERT INTO notices VALUES(?,?)", (self.stop, json.dumps(asdict(source))))
        self.stop += 1

    def drain(self) -> Sequence[TranscriptNoticeSource]:
        result = _NoticeRange(self.owner, range(self.start, self.stop))
        self.start = self.stop
        return result


class SemanticTranscriptRecords:
    """Index dependency coordinates without retaining message bodies or notice runs."""

    def __init__(
        self, records: TranscriptRecords, connection: sqlite3.Connection, checkpoint: Callable[[], None]
    ) -> None:
        self.records_owner, self.connection, self.checkpoint = records, connection, checkpoint
        self.closed = False
        self.cursors: set[sqlite3.Cursor] = set()
        connection.execute("PRAGMA temp_store=FILE")
        connection.executescript(
            "CREATE TEMP TABLE messages(position INTEGER PRIMARY KEY,offset INTEGER NOT NULL);"
            "CREATE TEMP TABLE counters(key TEXT PRIMARY KEY,value INTEGER NOT NULL);"
            "CREATE TEMP TABLE occurrences(kind TEXT NOT NULL,tool TEXT NOT NULL,ordinal INTEGER NOT NULL,"
            "message_index INTEGER NOT NULL,block_index INTEGER NOT NULL,PRIMARY KEY(kind,tool,ordinal));"
            "CREATE UNIQUE INDEX temp.occurrence_coordinate ON occurrences(message_index,block_index);"
            "CREATE TEMP TABLE notices(position INTEGER PRIMARY KEY,payload TEXT NOT NULL);"
        )

    @property
    def count(self) -> int:
        return self.records_owner.count

    def check(self) -> None:
        if self.closed:
            raise RuntimeError("semantic transcript owner is closed")
        self.checkpoint()

    def append(self, record: Mapping[str, object]) -> None:
        self.check()
        stream = self.records_owner.stream
        stream.seek(0, 2)
        offset = stream.tell()
        self.connection.execute("INSERT INTO messages VALUES(?,?)", (self.count, offset))
        self.records_owner.append(record)

    def records(self) -> Generator[dict[str, object], None, None]:
        self.check()
        for record in self.records_owner.records():
            self.check()
            yield record

    @contextmanager
    def retain_cursor(self, cursor: sqlite3.Cursor) -> Iterator[sqlite3.Cursor]:
        self.cursors.add(cursor)
        try:
            yield cursor
        finally:
            if cursor in self.cursors:
                self.cursors.remove(cursor)
                cursor.close()

    def entries(
        self, *, session_id: str, provider_family: str | None, lineage: LineageDescriptor | None
    ) -> Generator[SemanticTranscriptEntry, None, None]:
        self.check()
        self.connection.execute("DELETE FROM occurrences")
        self.connection.execute("DELETE FROM counters")
        self.connection.execute("DELETE FROM notices")
        for kind, tool, ordinal, coordinate, _ in semantic_tool_occurrences(
            normalized_semantic_messages(self.records(), session_id=session_id, provider_family=provider_family),
            _Ordinals(self),
        ):
            self.connection.execute(
                "INSERT INTO occurrences VALUES(?,?,?,?,?)",
                (kind, tool, ordinal, coordinate.message_index, coordinate.block_index),
            )

        def counterpart(coordinate: SemanticBlockCoordinate, kind: str) -> tuple[int, int] | None:
            self.check()
            row = self.connection.execute(
                "SELECT other.message_index,other.block_index FROM occurrences own JOIN occurrences other "
                "ON other.tool=own.tool AND other.ordinal=own.ordinal AND other.kind=? "
                "WHERE own.message_index=? AND own.block_index=?",
                (kind, coordinate.message_index, coordinate.block_index),
            ).fetchone()
            return None if row is None else (int(row[0]), int(row[1]))

        def result_for_use(coordinate: SemanticBlockCoordinate) -> SemanticResultMatch | None:
            result = counterpart(coordinate, BlockType.TOOL_RESULT.value)
            if result is None:
                return None
            offset = self.connection.execute("SELECT offset FROM messages WHERE position=?", (result[0],)).fetchone()[0]
            stream = self.records_owner.stream
            stream.seek(offset)
            record = json.loads(stream.readline())
            message = next(
                normalized_semantic_messages((record,), session_id=session_id, provider_family=provider_family)
            )
            return SemanticResultMatch(SemanticBlockCoordinate(*result), message, message.blocks[result[1]])

        yield from iter_semantic_entries(
            normalized_semantic_messages(self.records(), session_id=session_id, provider_family=provider_family),
            session_id=session_id,
            lineage=lineage,
            result_for_use=result_for_use,
            result_is_paired=lambda coordinate: counterpart(coordinate, BlockType.TOOL_USE.value) is not None,
            pending_empty_thinking=_PendingNotices(self),
        )
        self.check()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for cursor in tuple(self.cursors):
            cursor.close()
        self.cursors.clear()
        self.connection.close()


@contextmanager
def semantic_transcript_records(
    *, checkpoint: Callable[[], None] = lambda: None
) -> Iterator[SemanticTranscriptRecords]:
    with (
        TemporaryFile(mode="w+t", encoding="utf-8") as stream,
        TemporaryDirectory(prefix="semantic-transcript-") as directory,
    ):
        connection = sqlite3.connect(Path(directory) / "dependencies.db")
        try:
            owner = SemanticTranscriptRecords(TranscriptRecords(stream), connection, checkpoint)
            try:
                yield owner
            finally:
                owner.close()
        finally:
            connection.close()
