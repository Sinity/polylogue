"""Original Index generated-column cells keep exact literal and cursor custody."""

from __future__ import annotations

import sqlite3
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path
from typing import Any, Self

import pytest

from polylogue.storage.io_phase_metrics import _MeasuredConnection, connection_cursor
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    native_sql_children,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.sqlite_cursor_settlement import ControlledCursor
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path, title: str | None) -> str:
    with write_lease("test.generated-cell-seed", archive_root=root):
        bootstrap_archive_root(root)
        builder = SessionBuilder(root / "index.db", "generated-cells").provider("codex").add_message(text="Neutral")
        builder.save()
        session_id = builder.native_session_id()
        with closing(
            open_isolated_write_connection(root / "index.db", purpose="test.generated-cell-seed", archive_root=root)
        ) as index:
            with connection_cursor(index, "UPDATE sessions SET title=? WHERE session_id=?", (title, session_id)):
                pass
            index.commit()
    return session_id


@pytest.mark.parametrize("title", [None, "", "λ\x00中" * 40000])
def test_generated_table_exact_original_cells_have_bounded_transfers(
    tmp_path: Path, title: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = _seed(tmp_path, title)
    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
        index = seal.observer("index")
        assert isinstance(index, _MeasuredConnection)
        original_cursor = index.cursor
        transfers: list[int] = []

        class GuardedCursor(sqlite3.Cursor):
            def fetchone(self) -> sqlite3.Row | None:
                row: sqlite3.Row | None = super().fetchone()
                if row is not None:
                    for value in row:
                        assert not isinstance(value, str) or len(value.encode("utf-8")) <= 65536
                        if isinstance(value, bytes):
                            assert len(value) <= 65536
                            transfers.append(len(value))
                return row

        monkeypatch.setattr(index, "cursor", lambda: original_cursor(factory=GuardedCursor))
        with seal.original_read_snapshot():
            with seal.original_rows("index", "SELECT rowid FROM sessions WHERE session_id=?", (session_id,)) as cursor:
                rowid = cursor.fetchone()[0]
            image = seal.retain_tier_row("index", "sessions", rowid)
            assert image is not None
            cell = image.cells[image.columns.index("title")]
            expected = None if title is None else title.encode("utf-8")
            kind, length, _fixed = seal._literal_cell_metadata(cell)
            assert (kind, length) == ("null", 0) if expected is None else (kind, length) == ("text", len(expected))
            if expected is not None:
                assert b"".join(seal._literal_cell_chunks(cell)) == expected
            assert seal._matches_retained_row(index, image)
            changed = seal.overlay_tier_row(image, {"title": seal.retain_literal_scalar("different")}, rowid=rowid)
            assert not seal._matches_retained_row(index, changed)
        assert not index.live_cursors()
        assert transfers and max(transfers) <= 65536


@pytest.mark.parametrize("execute_failure", [False, True])
def test_generated_cell_failed_cursor_settlement_retains_original_creator(
    tmp_path: Path, execute_failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = _seed(tmp_path, "λ" * 40000)
    seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
    index = seal.observer("index")
    assert isinstance(index, _MeasuredConnection)
    owner = next(child for child in native_sql_children(seal) if child.connection is index)
    original_cursor = index.cursor
    blocked: list[ControlledCursor] = []
    completed: list[bool] = []
    primary = OSError("synthetic generated-cell read failure")
    owner.retain_settlement_callback(lambda: completed.append(True))

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = (), /) -> Self:
            result = super().execute(sql, parameters)
            if sql.startswith('SELECT substr(CAST("title" AS BLOB)'):
                self.allow_cleanup.clear()
                blocked.append(self)
                if execute_failure:
                    raise primary
            return result

    monkeypatch.setattr(index, "cursor", lambda: original_cursor(factory=FaultCursor))
    try:
        with pytest.raises((NativeConnectionSettlementError, BaseExceptionGroup)) as failure:
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "index", "SELECT rowid FROM sessions WHERE session_id=?", (session_id,)
                ) as cursor:
                    rowid = cursor.fetchone()[0]
                seal.retain_tier_row("index", "sessions", rowid)
        assert blocked and all(cursor in index.live_cursors() for cursor in blocked)
        assert all(id(cursor) in index._unsettled_native_cursors for cursor in blocked)
        assert owner.connection is index and owner.close_required
        assert completed == []
        if execute_failure:
            visited: set[int] = set()

            def contains(error: BaseException) -> bool:
                if error is primary:
                    return True
                if id(error) in visited:
                    return False
                visited.add(id(error))
                nested = getattr(error, "exceptions", ())
                return any(contains(item) for item in nested) or any(
                    isinstance(item, BaseException) and contains(item)
                    for item in (getattr(error, "failure", None), error.__cause__, error.__context__)
                )

            assert contains(failure.value)
        with pytest.raises((NativeConnectionSettlementError, BaseExceptionGroup)):
            seal.close()
        assert completed == [] and owner.connection is index
    finally:
        for cursor in blocked:
            cursor.allow_cleanup.set()
        seal.close()
    assert owner.connection is None and completed == [True]
