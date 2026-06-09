"""Unit tests for the shared ``replace_insight_rows`` helper."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from polylogue.storage.sqlite.queries._bulk_replace import replace_insight_rows

_DDL = """
    CREATE TABLE insight_demo (
        session_id TEXT NOT NULL,
        slot INTEGER NOT NULL,
        payload TEXT,
        PRIMARY KEY (session_id, slot)
    )
"""

_COLUMNS = ("session_id", "slot", "payload")


def _record(session_id: str, slot: int, payload: str) -> tuple[str, int, str]:
    return (session_id, slot, payload)


def _extract(record: tuple[str, int, str]) -> tuple[str, int, str]:
    return record


async def _open(path: Path) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await conn.execute(_DDL)
    await conn.commit()
    return conn


async def _all_rows(conn: aiosqlite.Connection) -> list[tuple[str, int, str]]:
    cursor = await conn.execute(
        "SELECT session_id, slot, payload FROM insight_demo ORDER BY session_id, slot",
    )
    rows = await cursor.fetchall()
    return [(r["session_id"], r["slot"], r["payload"]) for r in rows]


async def test_empty_input_is_noop(tmp_path: Path) -> None:
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=(),
            columns=_COLUMNS,
            records=(),
            extractor=_extract,
            transaction_depth=0,
        )
        assert await _all_rows(conn) == []
    finally:
        await conn.close()


async def test_single_record_insert(tmp_path: Path) -> None:
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        records = [_record("c1", 0, "hello")]
        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=("c1",),
            columns=_COLUMNS,
            records=records,
            extractor=_extract,
            transaction_depth=0,
        )
        assert await _all_rows(conn) == [("c1", 0, "hello")]
    finally:
        await conn.close()


async def test_multi_record_replace_replaces_prior_rows(tmp_path: Path) -> None:
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        # Seed prior rows for c1 and c2 (and a sibling c3 that must survive).
        await conn.executemany(
            "INSERT INTO insight_demo (session_id, slot, payload) VALUES (?, ?, ?)",
            [
                ("c1", 0, "old-c1-0"),
                ("c1", 1, "old-c1-1"),
                ("c2", 0, "old-c2-0"),
                ("c3", 0, "keep-c3"),
            ],
        )
        await conn.commit()

        new_records = [
            _record("c1", 0, "new-c1-0"),
            _record("c2", 0, "new-c2-0"),
            _record("c2", 1, "new-c2-1"),
        ]
        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=("c1", "c2"),
            columns=_COLUMNS,
            records=new_records,
            extractor=_extract,
            transaction_depth=0,
        )
        assert await _all_rows(conn) == [
            ("c1", 0, "new-c1-0"),
            ("c2", 0, "new-c2-0"),
            ("c2", 1, "new-c2-1"),
            ("c3", 0, "keep-c3"),
        ]
    finally:
        await conn.close()


async def test_delete_only_when_no_records(tmp_path: Path) -> None:
    """DELETE runs even when records is empty (tombstone semantics)."""
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        await conn.executemany(
            "INSERT INTO insight_demo (session_id, slot, payload) VALUES (?, ?, ?)",
            [("c1", 0, "old"), ("c2", 0, "keep")],
        )
        await conn.commit()

        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=("c1",),
            columns=_COLUMNS,
            records=(),
            extractor=_extract,
            transaction_depth=0,
        )
        assert await _all_rows(conn) == [("c2", 0, "keep")]
    finally:
        await conn.close()


async def test_idempotent_reinsert(tmp_path: Path) -> None:
    """Running the same replace twice yields the same end state."""
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        records = [
            _record("c1", 0, "v"),
            _record("c1", 1, "v"),
            _record("c2", 0, "v"),
        ]
        for _ in range(2):
            await replace_insight_rows(
                conn,
                table="insight_demo",
                id_column="session_id",
                id_values=("c1", "c2"),
                columns=_COLUMNS,
                records=records,
                extractor=_extract,
                transaction_depth=0,
            )
        assert await _all_rows(conn) == [
            ("c1", 0, "v"),
            ("c1", 1, "v"),
            ("c2", 0, "v"),
        ]
    finally:
        await conn.close()


async def test_transaction_depth_nonzero_does_not_commit(tmp_path: Path) -> None:
    """When transaction_depth > 0, helper does not commit; caller controls it."""
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        records = [_record("c1", 0, "uncommitted")]
        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=("c1",),
            columns=_COLUMNS,
            records=records,
            extractor=_extract,
            transaction_depth=1,
        )
        # Row is visible inside the open transaction on the same connection.
        assert await _all_rows(conn) == [("c1", 0, "uncommitted")]
        # ROLLBACK should discard it because no commit happened.
        await conn.rollback()
        assert await _all_rows(conn) == []
    finally:
        await conn.close()


async def test_batch_boundary_with_many_ids(tmp_path: Path) -> None:
    """Large id_values list builds a single IN (...) with all placeholders."""
    conn = await _open(tmp_path / "demo.sqlite")
    try:
        seed = [(f"c{i:03d}", 0, "old") for i in range(50)]
        await conn.executemany(
            "INSERT INTO insight_demo (session_id, slot, payload) VALUES (?, ?, ?)",
            seed,
        )
        await conn.commit()

        ids = tuple(f"c{i:03d}" for i in range(50))
        records = [_record(cid, 0, "new") for cid in ids]
        await replace_insight_rows(
            conn,
            table="insight_demo",
            id_column="session_id",
            id_values=ids,
            columns=_COLUMNS,
            records=records,
            extractor=_extract,
            transaction_depth=0,
        )
        rows = await _all_rows(conn)
        assert len(rows) == 50
        assert all(payload == "new" for _, _, payload in rows)
    finally:
        await conn.close()
