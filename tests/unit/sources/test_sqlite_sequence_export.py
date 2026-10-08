"""SQLite-owned sequence evidence hashes its exact stored cell classes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.sources.sqlite_export import logical_export_bytes, logical_source_context, read_export_header
from polylogue.sources.sqlite_snapshot import sqlite_logical_revision


@pytest.mark.parametrize(
    "storage_class,value,encoded",
    [
        ("integer", 1, ["i", 1]),
        ("real", 1.0, ["f", 1.0]),
        ("text", "1", ["t", "1"]),
        ("blob", b"1", ["b", "31"]),
        ("null", None, None),
    ],
)
def test_sequence_high_water_storage_class_changes_retained_identity(
    tmp_path: Path, storage_class: str, value: int | float | str | bytes | None, encoded: list[object] | None
) -> None:
    source = tmp_path / "sequence.db"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT)")
        conn.execute("INSERT INTO item (label) VALUES ('neutral')")
    before = sqlite_logical_revision(source)
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name='item'", (value,))
        assert conn.execute("SELECT typeof(seq) FROM sqlite_sequence").fetchone() == (storage_class,)
    payload = logical_export_bytes(source)
    assert json.loads(payload.splitlines()[0])["sqlite_sequence"] == [[["t", "item"], encoded]]
    after = sqlite_logical_revision(source)
    assert (after == before) is (storage_class == "integer")
    assert sqlite_logical_revision(source) == after
    assert logical_export_bytes(source) == payload

    # The header's system-table evidence does not change declared-table replay.
    export = tmp_path / "sequence.export"
    export.write_bytes(payload)
    assert read_export_header(export).tables == ("item",)
    with logical_source_context(export) as replay:
        assert replay.execute("SELECT id, label FROM item").fetchall() == [(1, "neutral")]


def test_sequence_invalid_utf8_text_keeps_its_storage_class(tmp_path: Path) -> None:
    source = tmp_path / "sequence.db"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        conn.execute("INSERT INTO item DEFAULT VALUES")
        conn.execute("UPDATE sqlite_sequence SET seq=CAST(x'ff' AS TEXT)")
    payload = logical_export_bytes(source)
    assert json.loads(payload.splitlines()[0])["sqlite_sequence"] == [[["t", "item"], ["tx", "ff"]]]
    text_revision = sqlite_logical_revision(source)
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("UPDATE sqlite_sequence SET seq=x'ff'")
    assert json.loads(logical_export_bytes(source).splitlines()[0])["sqlite_sequence"] == [[["t", "item"], ["b", "ff"]]]
    assert sqlite_logical_revision(source) != text_revision


@pytest.mark.parametrize(
    "value,encoded",
    [(1, ["i", 1]), ("1", ["t", "1"]), (b"1", ["b", "31"]), (None, None)],
)
def test_sequence_names_are_typed_and_declared_scope_matches_only_text(
    tmp_path: Path, value: int | str | bytes | None, encoded: list[object] | None
) -> None:
    source = tmp_path / "sequence.db"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute('CREATE TABLE "1" (id INTEGER PRIMARY KEY AUTOINCREMENT)')
        conn.execute('INSERT INTO "1" DEFAULT VALUES')
        conn.execute("UPDATE sqlite_sequence SET name=?", (value,))
    whole = logical_export_bytes(source)
    assert json.loads(whole.splitlines()[0])["sqlite_sequence"] == [[encoded, ["i", 1]]]
    scoped = logical_export_bytes(source, tables=("1",))
    expected = [[encoded, ["i", 1]]] if isinstance(value, str) else []
    assert json.loads(scoped.splitlines()[0])["sqlite_sequence"] == expected
    assert logical_export_bytes(source) == whole
    assert logical_export_bytes(source, tables=("1",)) == scoped
