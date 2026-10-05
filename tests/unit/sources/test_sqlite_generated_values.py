"""Readable generated values survive actual export and untyped reconstruction."""

from __future__ import annotations

import sqlite3
import subprocess
from builtins import BaseExceptionGroup
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.sources import sqlite_export
from polylogue.sources.parsers import antigravity, hermes_state, hermes_verification
from polylogue.sources.parsers.codex_state import (
    CodexThreadRecord,
    classify_codex_sqlite_path,
    iter_codex_state_parts,
    iter_codex_state_records,
)
from tests.infra.generated_sqlite import GeneratedStorage, generated_state_parts, generated_thread_state


@pytest.mark.parametrize("storage", ["VIRTUAL", "STORED"])
def test_codex_generated_title_survives_acquisition_and_retained_parser(
    tmp_path: Path, storage: GeneratedStorage
) -> None:
    """table_info in the export drops title and breaks the retained SELECT."""
    source = generated_thread_state(tmp_path / "state_5.sqlite", storage)
    expected = list(iter_codex_state_records(source))
    assert len(expected) == 1 and isinstance(expected[0], CodexThreadRecord)
    assert expected[0].title == "neutral curated"
    export = tmp_path / "retained.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source, tables=("threads", "thread_spawn_edges")))
    assert classify_codex_sqlite_path(source) == classify_codex_sqlite_path(export) == "thread_state"
    assert list(iter_codex_state_records(export)) == expected
    assert sqlite_export.logical_source_shape(source) == sqlite_export.logical_source_shape(export)
    with sqlite_export.logical_source_context(export) as conn:
        assert conn.execute("SELECT rowid,title FROM threads").fetchall() == [(17, "neutral curated")]
        ddl = conn.execute("SELECT sql FROM sqlite_schema WHERE name='threads'").fetchone()[0]
        assert "GENERATED" not in ddl
    header = sqlite_export.read_export_header(export)
    assert any("GENERATED" in str(row[3]) for row in header.schema if row[1] == "threads")


@pytest.mark.parametrize("storage", ["VIRTUAL", "STORED"])
@pytest.mark.parametrize("kind", ["goals", "memories"])
def test_selected_generated_codex_material_reaches_retained_parts(
    tmp_path: Path, storage: GeneratedStorage, kind: Literal["goals", "memories"]
) -> None:
    """Acquired objective/memory cannot become empty in the retained reader."""
    source = generated_state_parts(tmp_path / f"{kind}.sqlite", storage, kind=kind)
    table = "thread_goals" if kind == "goals" else "stage1_outputs"
    fields = ("objective",) if kind == "goals" else ("raw_memory", "rollout_summary")
    with closing(sqlite3.connect(source)) as conn:
        expected = dict(zip(fields, conn.execute(f"SELECT {','.join(fields)} FROM {table}").fetchone(), strict=True))
    export = tmp_path / "retained.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source, tables=(table,)))
    live_parts = list(iter_codex_state_parts(source, state_kind=kind, text_chars=4))
    parts = list(iter_codex_state_parts(export, state_kind=kind, text_chars=4))
    assert parts == live_parts
    assert classify_codex_sqlite_path(source) == classify_codex_sqlite_path(export) == kind
    assert sqlite_export.logical_source_shape(source) == sqlite_export.logical_source_shape(export)
    with sqlite_export.logical_source_context(export) as conn:
        assert conn.execute(f"SELECT rowid FROM {table}").fetchall() == [(19 if kind == "goals" else 23,)]
    record = next(part.payload for part in parts if part.part_kind == "record")
    actual: dict[str, str] = {}
    for field in fields:
        value = record[field]
        assert isinstance(value, str)
        actual[field] = value
    for part in parts:
        if part.part_kind == "text_chunk":
            field = str(part.payload["field"])
            assert part.payload["offset_chars"] == len(actual[field])
            text = part.payload["text"]
            assert isinstance(text, str)
            actual[field] += text
    assert actual == expected


@pytest.mark.parametrize("storage", ["VIRTUAL", "STORED"])
@pytest.mark.parametrize("without_rowid", [False, True])
def test_generated_values_keep_native_storage_classes_bytes_and_row_identity(
    tmp_path: Path, storage: GeneratedStorage, without_rowid: bool
) -> None:
    """Replaying source affinity or omitting evaluated columns loses typed bytes."""
    source = tmp_path / "typed.sqlite"
    suffix = " WITHOUT ROWID" if without_rowid else ""
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(f"CREATE TABLE typed (key INTEGER PRIMARY KEY, seed, acquired AS(seed) {storage}){suffix}").close()
        conn.executemany(
            "INSERT INTO typed(key,seed) VALUES (?,?)",
            [(7, None), (11, 37), (19, 1.25), (23, "Ω\x00NFD e\u0301"), (29, b"\x00\xff")],
        ).close()
        conn.execute("INSERT INTO typed(key,seed) VALUES (31,CAST(x'ff00' AS TEXT))").close()
        conn.text_factory = bytes
        expected = conn.execute("SELECT key,typeof(acquired),CAST(acquired AS BLOB) FROM typed ORDER BY key").fetchall()
    export = tmp_path / "typed.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source))
    shape = sqlite_export.logical_source_shape(source)
    assert shape == {"typed": ("key", "seed", "acquired")}
    assert sqlite_export.logical_source_shape(export) == shape
    with sqlite_export.logical_source_context(export) as conn:
        conn.text_factory = bytes
        assert (
            conn.execute("SELECT key,typeof(acquired),CAST(acquired AS BLOB) FROM typed ORDER BY key").fetchall()
            == expected
        )
        assert [row[2] for row in sqlite_export.readable_table_info(conn, "typed")] == [b"", b"", b""]
        if not without_rowid:
            assert conn.execute("SELECT rowid FROM typed ORDER BY rowid").fetchall() == [
                (7,),
                (11,),
                (19,),
                (23,),
                (29,),
                (31,),
            ]


@pytest.mark.parametrize("reader", [hermes_state._columns, hermes_verification._columns, antigravity._sqlite_columns])
@pytest.mark.parametrize("storage", ["VIRTUAL", "STORED"])
def test_provider_column_presence_includes_readable_generated_values(
    tmp_path: Path, reader: Any, storage: GeneratedStorage
) -> None:
    """A direct column-presence reader cannot treat acquired generated data as absent."""
    source = tmp_path / "presence.sqlite"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(f"CREATE TABLE fields (seed TEXT, content AS(seed) {storage})").close()
        conn.execute("INSERT INTO fields(seed) VALUES ('retained')").close()
        assert reader(conn, "fields") == {"seed", "content"}
    export = tmp_path / "presence.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source))
    with sqlite_export.logical_source_context(export) as conn:
        assert reader(conn, "fields") == {"seed", "content"}
        assert conn.execute("SELECT content FROM fields").fetchone()[0] == "retained"


def test_hidden_virtual_table_columns_do_not_enter_readable_shape_or_export(tmp_path: Path) -> None:
    """table_xinfo alone would widen the existing virtual-table contract."""
    source = tmp_path / "virtual.sqlite"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE VIRTUAL TABLE documents USING fts5(body)").close()
        assert [row[1] for row in sqlite_export.readable_table_info(conn, "documents")] == ["body"]
    assert sqlite_export.logical_source_shape(source)["documents"] == ("body",)
    export = tmp_path / "virtual.export"
    export.write_bytes(sqlite_export.logical_export_bytes(source, tables=("documents",)))
    header = sqlite_export.read_export_header(export)
    assert header.tables == ()
    assert header.missing == ("documents",)


@pytest.mark.parametrize("failure", [OSError, DaemonOperationCancelled])
def test_generated_export_callback_failure_preserves_original_error_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Generated acquisition keeps original cancellation/fault and settles the worker."""
    source = generated_thread_state(tmp_path / "state.sqlite", "VIRTUAL")
    children: list[subprocess.Popen[bytes]] = []
    operations: list[str] = []
    original = subprocess.Popen
    original_exchange = sqlite_export._exchange_source_worker

    def exchange(request: dict[str, Any], handle: Any = None) -> dict[str, Any]:
        operations.append(request["operation"])
        return original_exchange(request, handle)

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original(*args, **kwargs)
        children.append(child)
        return child

    error = failure("synthetic generated export refusal")

    class Sink:
        def write(self, payload: bytes) -> int:
            if b"neutral curated" in payload:
                raise error
            return len(payload)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(sqlite_export, "_exchange_source_worker", exchange)
    with pytest.raises(failure) as caught:
        sqlite_export.write_logical_export(source, Sink())
    assert caught.value is error
    # Canonical input binding settles its worker before the export worker starts.
    assert operations == ["binding", "export"]
    assert len(children) == len(operations)
    for child in children:
        assert child.poll() is not None
        assert child.stdin is not None and child.stdin.closed
        assert child.stdout is not None and child.stdout.closed


@pytest.mark.parametrize("fail_close", [False, True])
def test_readable_metadata_execute_failure_retains_original_statement(fail_close: bool) -> None:
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor

    primary = sqlite3.OperationalError("synthetic metadata refusal")
    cursors: list[ControlledCursor] = []

    class MetadataCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = ()) -> MetadataCursor:
            super().execute(sql, parameters)
            if sql.startswith("PRAGMA table_xinfo"):
                cursors.append(self)
                if fail_close:
                    self.allow_cleanup.clear()
                raise primary
            return self

    CursorT = TypeVar("CursorT", bound=sqlite3.Cursor)

    class MetadataConnection(ControlledConnection):
        @overload
        def cursor(self, factory: None = None) -> sqlite3.Cursor: ...

        @overload
        def cursor(self, factory: Callable[[sqlite3.Connection], CursorT]) -> CursorT: ...

        def cursor(self, factory: Callable[[sqlite3.Connection], CursorT] | None = None) -> sqlite3.Cursor:
            return super().cursor(factory=MetadataCursor if factory is None else factory)

    conn = sqlite3.connect(":memory:", factory=MetadataConnection)
    try:
        conn.execute("CREATE TABLE evidence(value TEXT)").close()
        with pytest.raises(BaseException) as caught:
            sqlite_export.readable_table_info(conn, "evidence")
        assert len(cursors) == 1
        cursor = cursors[0]
        if fail_close:
            assert isinstance(caught.value, BaseExceptionGroup)
            assert caught.value.exceptions == (primary, cursor.cleanup_failure)
            assert any(item is cursor for item in conn.live_cursors())
            cursor.allow_cleanup.set()
            conn.close_cursor(cursor)
        else:
            assert caught.value is primary
        assert cursor.close_attempts == (2 if fail_close else 1)
        assert not any(item is cursor for item in conn.live_cursors())
    finally:
        for cursor in cursors:
            cursor.allow_cleanup.set()
        conn.settle_cursors()
        conn.close()
