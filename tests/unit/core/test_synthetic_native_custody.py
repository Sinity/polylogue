"""Synthetic row stores retain failed cleanup on their original creator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.schemas.synthetic.relations import ForeignKeyGraph
from polylogue.schemas.workload_profile_rows import DepthGrammars
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    retained_native_sql_owners_on_current_thread,
)


@pytest.mark.parametrize("kind", ["depth", "generated"])
def test_synthetic_rows_failed_close_retains_exact_handle_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    store: Any = DepthGrammars() if kind == "depth" else ForeignKeyGraph()
    if kind == "depth":
        store.append("1", {"record_kinds": {"assistant_text": 1}})
    else:
        store.capture_schema({"type": "object", "properties": {"neutral": {"type": "string"}}})
    connection = store._connection
    owner = store._sql_owner
    path: Path | None = None
    cursor = connection.execute("PRAGMA database_list")
    try:
        filename = cursor.fetchone()[2]
        if filename:
            path = Path(filename)
    finally:
        cursor.close()
    original_close = connection.close
    failure = OSError("neutral synthetic row close failure")

    def fail_close() -> None:
        raise failure

    monkeypatch.setattr(connection, "close", fail_close)
    with pytest.raises(NativeConnectionSettlementError) as refused:
        store.close()
    assert refused.value.owner is owner
    assert refused.value.failure is failure
    assert store._connection is connection
    assert owner in retained_native_sql_owners_on_current_thread()
    if path is not None:
        assert path.is_file()
    monkeypatch.setattr(connection, "close", original_close)
    store.close()
    assert owner not in retained_native_sql_owners_on_current_thread()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    if path is not None:
        assert not path.exists()
