from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, cast

from polylogue.operations.source_item_settlement import _matching_raw_members


class _FetchoneOnlyCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> Any:
        raise AssertionError("raw membership comparison must remain cursor-streamed")

    def close(self) -> None:
        self._cursor.close()


class _FetchoneOnlyConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _FetchoneOnlyCursor:
        return _FetchoneOnlyCursor(self._connection.execute(sql, parameters))


def test_raw_membership_comparison_streams_large_exact_population(tmp_path: Path) -> None:
    source = sqlite3.connect(":memory:")
    source.executescript(
        "CREATE TABLE source_item_raw_members (source_generation_id TEXT, source_item_id TEXT, "
        "raw_id TEXT, raw_blob_hash BLOB);"
        "CREATE TABLE raw_sessions (raw_id TEXT, blob_hash BLOB);"
    )
    receipt_path = tmp_path / "receipt.sqlite"
    receipt = sqlite3.connect(receipt_path)
    receipt.execute("CREATE TABLE item_raws (source_item_id TEXT, raw_id TEXT, raw_blob_hash BLOB, complete INTEGER)")
    member_count = 2049
    source_members: list[tuple[str, str, str, bytes]] = []
    receipt_members: list[tuple[str, str, bytes, int]] = []
    for ordinal in range(member_count):
        raw_id = f"raw:{ordinal:05d}"
        blob_hash = hashlib.sha256(raw_id.encode()).digest()
        source_members.append(("generation", "item", raw_id, blob_hash))
        receipt_members.append(("item", raw_id, blob_hash, 1))
    source.executemany("INSERT INTO source_item_raw_members VALUES (?, ?, ?, ?)", source_members)
    source.executemany("INSERT INTO raw_sessions VALUES (?, ?)", [(row[2], row[3]) for row in source_members])
    receipt.executemany("INSERT INTO item_raws VALUES (?, ?, ?, ?)", receipt_members)
    source.commit()
    receipt.commit()

    stop_checks: list[int] = []
    complete = _matching_raw_members(
        cast(sqlite3.Connection, _FetchoneOnlyConnection(source)),
        cast(sqlite3.Connection, _FetchoneOnlyConnection(receipt)),
        source_generation_id="generation",
        source_item_id="item",
        expected_count=member_count,
        check_stop=lambda: stop_checks.append(len(stop_checks)),
    )
    assert complete == (True, True)
    assert len(stop_checks) > 10

    receipt.execute(
        "UPDATE item_raws SET raw_blob_hash=? WHERE source_item_id='item' AND raw_id=?",
        (b"x" * 32, "raw:01024"),
    )
    receipt.commit()
    mismatch = _matching_raw_members(
        cast(sqlite3.Connection, _FetchoneOnlyConnection(source)),
        cast(sqlite3.Connection, _FetchoneOnlyConnection(receipt)),
        source_generation_id="generation",
        source_item_id="item",
        expected_count=member_count,
        check_stop=None,
    )
    assert mismatch == (False, False)
    receipt.close()
    source.close()
