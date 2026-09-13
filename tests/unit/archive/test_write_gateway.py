from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.write_gateway import (
    WRITE_OPERATION_POLICIES,
    ArchiveWriteGateway,
    WriteOperation,
    WriteResultStatus,
    write_operation_policy_for,
)
from polylogue.storage.sqlite.connection import open_connection

COMMITTED: WriteResultStatus = "committed"


def test_declared_write_operations_have_exhaustive_production_effect_policies() -> None:
    """Every enum value names its production writer and transaction policy."""
    assert set(WRITE_OPERATION_POLICIES) == set(WriteOperation)
    assert write_operation_policy_for(WriteOperation.RESET, "archive-index").run_archive_effects is True
    assert write_operation_policy_for(WriteOperation.RESET, "user-overlay").run_archive_effects is False
    assert write_operation_policy_for(WriteOperation.DELETE, "archive-index").actuator == "ArchiveStore.delete_sessions"
    assert write_operation_policy_for(WriteOperation.TAG_UPDATE, "user-overlay").run_archive_effects is False
    assert write_operation_policy_for(WriteOperation.METADATA_UPDATE, "user-overlay").run_archive_effects is False


def test_user_overlay_gateway_commits_without_index_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag/metadata policy commits its real transaction but cannot schedule FTS/cache work.

    Anti-vacuity: changing this route to the archive-index policy invokes the
    patched FTS function and fails, while removing the gateway call leaves the
    insert uncommitted after the connection closes.
    """
    db_path = tmp_path / "user.db"
    cache_invalidations: list[bool] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync",
        lambda _conn: pytest.fail("user overlays must not touch index FTS"),
    )
    monkeypatch.setattr(
        "polylogue.storage.search.cache.invalidate_search_cache",
        lambda: cache_invalidations.append(True),
    )

    with open_connection(db_path) as conn:
        conn.execute("CREATE TABLE writes (value TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO writes VALUES ('tag')")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.TAG_UPDATE,
            {"_connection": conn, "changed_session_ids": (), "effect_scope": "user-overlay"},
        )

    with open_connection(db_path) as conn:
        assert conn.execute("SELECT value FROM writes").fetchone()[0] == "tag"
    assert result.effect_receipts == ()
    assert cache_invalidations == []


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
        assert result.status == COMMITTED
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

    assert result.status == COMMITTED
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
        lambda _conn, _ids, **_kwargs: repaired.append("messages"),
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

    assert result.status == COMMITTED
    assert repaired == []


def test_write_gateway_repairs_fts_when_requested_even_if_live_triggers_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    repaired: list[tuple[tuple[str, ...], bool | None]] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, ids, **kwargs: repaired.append((tuple(ids), kwargs.get("record_exact_snapshot"))),
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

    assert result.status == COMMITTED
    assert repaired == [(("c1",), False)]


def test_write_gateway_repairs_fts_when_live_triggers_were_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    repaired: list[str] = []

    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync",
        lambda _conn, _ids, **_kwargs: repaired.append("messages"),
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

    assert result.status == COMMITTED
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
        assert result.status == COMMITTED
