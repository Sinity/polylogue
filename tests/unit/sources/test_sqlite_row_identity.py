"""Catalog row identity survives canonical export and retained reconstruction."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.sources.sqlite_export import logical_export_bytes, logical_source_context


def test_quoted_without_rowid_text_does_not_hide_native_row_identity(tmp_path: Path) -> None:
    source = tmp_path / "neutral.sqlite"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("CREATE TABLE threads (id TEXT, title TEXT DEFAULT 'WITHOUT ROWID')")
        conn.execute("INSERT INTO threads(rowid,id,title) VALUES (5,'neutral-session','neutral')")
    before = logical_export_bytes(source)
    export = tmp_path / "retained.export"
    export.write_bytes(before)
    with logical_source_context(export) as replay:
        assert replay.execute("SELECT rowid,id,title FROM threads").fetchall() == [(5, "neutral-session", "neutral")]
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute("UPDATE threads SET rowid=7")
    after = logical_export_bytes(source)
    assert after != before
    export.write_bytes(after)
    with logical_source_context(export) as replay:
        assert replay.execute("SELECT rowid,id,title FROM threads").fetchall() == [(7, "neutral-session", "neutral")]


@pytest.mark.parametrize("option", ["WITHOUT\nROWID", "WITHOUT /* catalog option */ ROWID"])
def test_without_rowid_catalog_option_accepts_sql_whitespace(tmp_path: Path, option: str) -> None:
    source = tmp_path / "neutral.sqlite"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(f"CREATE TABLE threads(id TEXT PRIMARY KEY,title TEXT) {option}")
        conn.execute("INSERT INTO threads VALUES ('neutral-session','neutral')")
    payload = logical_export_bytes(source)
    table = json.loads(payload.splitlines()[1])
    assert table["rowid"] is False
    assert table["columns"] == ["id", "title"]
    export = tmp_path / "retained.export"
    export.write_bytes(payload)
    with logical_source_context(export) as replay:
        assert replay.execute("SELECT id,title FROM threads").fetchall() == [("neutral-session", "neutral")]


@pytest.mark.parametrize("alias", ["rowid", "RowID", "ROWID"])
def test_declared_rowid_alias_shadows_case_insensitively(tmp_path: Path, alias: str) -> None:
    source = tmp_path / "neutral.sqlite"
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(f'CREATE TABLE threads("{alias}" TEXT,title TEXT)')
        conn.execute("INSERT INTO threads VALUES ('declared-cell','neutral')")
    payload = logical_export_bytes(source)
    table = json.loads(payload.splitlines()[1])
    assert table["rowid"] is False
    assert table["columns"] == [alias, "title"]
    export = tmp_path / "retained.export"
    export.write_bytes(payload)
    with logical_source_context(export) as replay:
        assert replay.execute("SELECT rowid,title FROM threads").fetchall() == [("declared-cell", "neutral")]
