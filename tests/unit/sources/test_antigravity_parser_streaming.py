"""Antigravity trajectory parsing stays bounded across rows and sessions."""

from __future__ import annotations

import json
import sqlite3
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from polylogue.sources.parse_accounting_spool import SqliteParseAccountingWriter
from polylogue.sources.parsers.antigravity import parse_trajectory_db
from polylogue.sources.parsers.base import AdmissionUnit, ParseAccounting, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.sources.streamed_event_payload import StreamedJsonArray


class _AccountingBuilder:
    def __init__(self, expected: dict[AdmissionUnit, int], writer: SqliteParseAccountingWriter) -> None:
        self.expected = expected
        self.writer = writer

    def append(self, outcome: object) -> None:
        self.writer.append(outcome)

    def finish(self) -> ParseAccounting:
        return ParseAccounting.model_construct(expected=self.expected, outcomes=self.writer.finish())


def _accounting_factory(
    store: SqliteMessageStore,
) -> Callable[[dict[AdmissionUnit, int]], _AccountingBuilder]:
    def create(expected: dict[AdmissionUnit, int]) -> _AccountingBuilder:
        writer_expected: dict[object, int] = {}
        for unit, count in expected.items():
            writer_expected[unit] = count
        return _AccountingBuilder(expected, SqliteParseAccountingWriter(store.conn, writer_expected))

    return create


def _source(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE trajectory_meta(trajectory_id TEXT, cascade_id TEXT); "
        "CREATE TABLE steps(idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT, trajectory_id TEXT); "
        "CREATE TABLE parent_references(cascade_id TEXT, parent_id TEXT);"
    )
    return connection


def _parse(path: Path, store: SqliteMessageStore) -> Iterator[ParsedSession]:
    return parse_trajectory_db(
        path,
        grouping=store.conn,
        message_sink_factory=store.new_sink,
        event_sink_factory=store.new_event_sink,
        accounting_factory=_accounting_factory(store),
    )


def test_one_trajectory_spills_many_unsupported_steps(tmp_path: Path) -> None:
    source_path = tmp_path / "one.db"
    source = _source(source_path)
    count = 5000
    source.execute("INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1')")
    source.executemany(
        "INSERT INTO steps VALUES (?, 'future_step', 'v1', ?, 'trajectory-1')",
        ((ordinal, json.dumps({"text": f"unsupported {ordinal}"})) for ordinal in range(count)),
    )
    source.commit()
    source.close()

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        tracemalloc.start()
        [session] = list(_parse(source_path, store))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(session.messages) == 0
        assert len(session.session_events) == count + 1
        accounting = session.unit_accounting
        assert accounting is not None
        assert len(accounting.outcomes) == count
        accounting.assert_conserved()
        # All per-step events and exceptional admission rows live in SQLite.
        assert peak < 20 * 1024 * 1024, f"peak traced memory: {peak} bytes"
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        store.close()


def test_many_trajectory_metadata_rows_are_spilled(tmp_path: Path) -> None:
    source_path = tmp_path / "many.db"
    source = _source(source_path)
    count = 2000
    source.executemany(
        "INSERT INTO trajectory_meta VALUES (?, ?)",
        ((f"trajectory-{ordinal}", f"cascade-{ordinal}") for ordinal in range(count)),
    )
    source.commit()
    source.close()

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        tracemalloc.start()
        parsed = 0
        for session in _parse(source_path, store):
            assert len(session.messages) == 0
            parsed += 1
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert parsed == count
        assert peak < 20 * 1024 * 1024, f"peak traced memory: {peak} bytes"
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        store.close()


def test_parent_reference_event_arrays_are_spilled(tmp_path: Path) -> None:
    source_path = tmp_path / "parents.db"
    source = _source(source_path)
    count = 4000
    source.execute("INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1')")
    source.executemany(
        "INSERT INTO parent_references VALUES ('cascade-1', ?)",
        ((f"parent-{ordinal:05d}",) for ordinal in range(count)),
    )
    source.commit()
    source.close()

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        tracemalloc.start()
        [session] = list(_parse(source_path, store))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        parent_event = next(
            event for event in session.session_events if event.event_type == "antigravity_parent_reference"
        )
        references = parent_event.payload["references"]
        parent_ids = parent_event.payload["parent_provider_ids"]
        assert isinstance(references, StreamedJsonArray)
        assert isinstance(parent_ids, StreamedJsonArray)
        assert len(references) == count
        assert len(parent_ids) == count
        assert sum(1 for _ in references.iter_values()) == count
        assert sum(1 for _ in parent_ids.iter_values()) == count
        assert peak < 20 * 1024 * 1024
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        store.close()


def test_long_tool_call_run_keeps_only_count_and_sole_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "tool-run.db"
    source = _source(source_path)
    count = 2000
    source.execute("INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1')")
    source.executemany(
        "INSERT INTO steps VALUES (?, 'terminal_command', 'v1', ?, 'trajectory-1')",
        ((ordinal, json.dumps({"tool_name": "shell", "command": f"command-{ordinal}"})) for ordinal in range(count)),
    )
    source.execute(
        "INSERT INTO steps VALUES (?, 'tool_result', 'v1', ?, 'trajectory-1')",
        (count, json.dumps({"tool_name": "shell", "output": "done", "status": "success"})),
    )
    source.commit()
    source.close()

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        [session] = list(_parse(source_path, store))
        result = session.messages[-1].blocks[0]
        assert len(session.messages) == count + 1
        assert result.type.value == "tool_result"
        assert result.tool_id is None
        accounting = session.unit_accounting
        assert accounting is not None
        assert accounting.expected[AdmissionUnit.PART] == count + 1
        accounting.assert_conserved()
    finally:
        store.close()


def test_cancelled_trajectory_parse_releases_source_transaction(tmp_path: Path) -> None:
    source_path = tmp_path / "cancel.db"
    source = _source(source_path)
    source.execute("INSERT INTO trajectory_meta VALUES ('trajectory-1','cascade-1')")
    source.executemany(
        "INSERT INTO steps VALUES (?, 'future_step', 'v1', ?, 'trajectory-1')",
        ((ordinal, json.dumps({"text": f"unsupported {ordinal}"})) for ordinal in range(500)),
    )
    source.commit()
    source.close()

    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    calls = 0

    class CancelledError(Exception):
        pass

    def check_cancelled() -> None:
        nonlocal calls
        calls += 1
        if calls == 20:
            raise CancelledError

    try:
        with pytest.raises(CancelledError):
            list(
                parse_trajectory_db(
                    source_path,
                    grouping=store.conn,
                    message_sink_factory=store.new_sink,
                    event_sink_factory=store.new_event_sink,
                    accounting_factory=_accounting_factory(store),
                    check_cancelled=check_cancelled,
                )
            )

        # The parser's logical-source context has exited even though the
        # step cursor was interrupted partway through its read.
        source = sqlite3.connect(source_path)
        try:
            source.execute("BEGIN EXCLUSIVE")
            source.rollback()
        finally:
            source.close()
    finally:
        store.close()
