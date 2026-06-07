from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.storage.sqlite.connection import open_connection


def test_write_gateway_commits_effects_on_caller_owned_connection(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": (),
            },
        )

        assert result.operation is WriteOperation.INGEST
        assert result.status == "committed"
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_write_gateway_normal_commit_does_not_drop_fts_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    ensured: list[bool] = []

    def ensure_only(_conn: object) -> None:
        ensured.append(True)

    def fail_restore(_conn: object) -> None:
        raise AssertionError("normal archive writes must not drop/recreate FTS triggers")

    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", ensure_only)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.restore_fts_triggers_sync", fail_restore)

    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": (),
            },
        )

    assert result.status == "committed"
    assert ensured == [True]


def test_write_gateway_can_skip_fts_repairs_when_triggers_maintained_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    repaired: list[str] = []

    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids: repaired.append("messages"),
    )

    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": ("c1",),
                "repair_message_fts": False,
            },
        )

    assert result.status == "committed"
    assert repaired == []


def test_write_gateway_repairs_fts_when_requested_even_if_live_triggers_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    repaired: list[str] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids: repaired.append("messages"),
    )

    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": ("c1",),
            },
        )

    assert result.status == "committed"
    assert repaired == ["messages"]


def test_write_gateway_repairs_fts_when_live_triggers_were_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    repaired: list[str] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids: repaired.append("messages"),
    )

    with open_connection(db_path) as conn:
        conn.execute("DROP TRIGGER messages_fts_ai")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": ("c1",),
            },
        )

    assert result.status == "committed"
    assert repaired == ["messages"]


@pytest.mark.asyncio
async def test_write_gateway_async_commit_uses_same_local_effects_path(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        result = await ArchiveWriteGateway(db_path).commit_write(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_session_ids": (),
            },
        )

        assert result.operation is WriteOperation.INGEST
        assert result.status == "committed"
