from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.daemon.convergence_stages import (
    make_default_convergence_stages,
    make_embed_stage,
    make_fts_stage,
    make_insights_stage,
)
from polylogue.storage.insights.session.runtime import SessionInsightCounts


def test_insights_stage_rebuilds_sync_against_configured_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    opened_paths: list[Path] = []
    rebuilt = False

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            if "sqlite_master" in sql:
                return _FakeCursor([(1,)])
            if "raw_conversations" in sql:
                return _FakeCursor([(params[0], "conv-1")])
            if "session_profiles" in sql:
                return _FakeCursor([])
            raise AssertionError(f"unexpected SQL: {sql} {params}")

        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeCursor:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def fetchone(self) -> object | None:
            return self._rows[0] if self._rows else None

        def fetchall(self) -> list[object]:
            return self._rows

    @contextmanager
    def fake_open_connection(path: Path) -> Iterator[FakeConnection]:
        opened_paths.append(path)
        yield FakeConnection()

    def fake_rebuild(
        conn: FakeConnection,
        *,
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        nonlocal rebuilt
        del conn
        rebuilt = True
        assert conversation_ids == ["conv-1"]
        assert page_size == 10
        return SessionInsightCounts(
            profiles=1,
            work_events=2,
            phases=3,
            threads=4,
            tag_rollups=5,
            day_summaries=6,
        )

    def fail_if_used(coro: object) -> object:
        raise AssertionError("insights stage should not open an asyncio runner")

    monkeypatch.setattr(asyncio, "run", fail_if_used)
    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)

    assert make_insights_stage(db_path).execute(tmp_path / "source.jsonl") is True
    assert opened_paths == [db_path]
    assert rebuilt is True


def test_fts_stage_repairs_only_missing_action_index_when_messages_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    repaired_messages: list[list[str]] = []
    inserted_actions: list[list[str]] = []
    rebuilt = False
    committed = False

    class FakeConnection:
        def close(self) -> None:
            pass

        def commit(self) -> None:
            nonlocal committed
            committed = True

    def fake_open_connection(path: Path, *, timeout: float) -> FakeConnection:
        assert path == db_path
        assert timeout == 30.0
        return FakeConnection()

    def fake_repair_messages(conn: FakeConnection, conversation_ids: list[str]) -> None:
        repaired_messages.append(conversation_ids)

    def fake_insert_missing_actions(conn: FakeConnection, conversation_ids: list[str]) -> None:
        inserted_actions.append(conversation_ids)

    def fake_rebuild(conn: FakeConnection) -> None:
        nonlocal rebuilt
        rebuilt = True

    needs_calls: list[list[str]] = []

    def fake_repair_needs(_conn: FakeConnection, conversation_ids: list[str]) -> stages._FtsRepairNeeds:
        needs_calls.append(conversation_ids)
        return stages._FtsRepairNeeds(actions=len(needs_calls) == 1)

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.repair_message_fts_index_sync", fake_repair_messages)
    monkeypatch.setattr(
        "polylogue.storage.fts.fts_lifecycle.insert_missing_action_fts_index_sync",
        fake_insert_missing_actions,
    )
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_fts_repair_needs_for_conversations", fake_repair_needs)
    monkeypatch.setattr(stages, "_action_events_exist_for_conversations", lambda _conn, _ids: False)

    stage = make_fts_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert repaired_messages == []
    assert inserted_actions == [["conv-a", "conv-b"]]
    assert needs_calls == [["conv-a", "conv-b"], ["conv-a", "conv-b"]]
    assert committed is True
    assert rebuilt is False


def test_action_event_probe_uses_indexed_base_table() -> None:
    queries: list[tuple[str, tuple[str, ...]]] = []

    class FakeConnection:
        def execute(self, sql: str, params: tuple[str, ...] = ()) -> object:
            queries.append((sql, params))
            if "sqlite_master" in sql:
                return _FakeCursor([("action_events",)])
            if "action_events" in sql:
                return _FakeCursor([(1,)])
            raise AssertionError(f"unexpected SQL: {sql}")

    class _FakeCursor:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchone(self) -> tuple[object, ...] | None:
            return self._rows[0] if self._rows else None

    conn = cast(sqlite3.Connection, FakeConnection())
    assert stages._action_events_exist_for_conversations(conn, ["conv-a", "conv-b"]) is True
    probe_sql = queries[-1][0]
    assert "FROM action_events\n" in probe_sql
    assert "action_events_fts" not in probe_sql
    assert queries[-1][1] == ("conv-a", "conv-b")


def test_embed_stage_scopes_changed_conversations_without_asyncio_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    embedded_calls: list[list[str]] = []

    class FakeConnection:
        def close(self) -> None:
            pass

        def commit(self) -> None:
            pass

    def fake_open_connection(path: Path, *, timeout: float) -> FakeConnection:
        assert path == db_path
        assert timeout == 5.0
        return FakeConnection()

    def fail_if_used(coro: object) -> object:
        raise AssertionError("embed stage should not open an asyncio runner")

    def fake_pending(_conn: FakeConnection, conversation_ids: list[str]) -> list[str]:
        return [conversation_id for conversation_id in conversation_ids if conversation_id in {"conv-a", "conv-c"}]

    def fake_embed(_db_path: Path, conversation_ids: list[str]) -> bool:
        embedded_calls.append(list(conversation_ids))
        return True

    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "1")
    monkeypatch.setattr(asyncio, "run", fail_if_used)
    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a", "conv-b"], Path(paths[1]): ["conv-c"]},
    )
    monkeypatch.setattr(stages, "_pending_embedding_conversation_ids", fake_pending)
    monkeypatch.setattr(stages, "_embed_conversations_sync", fake_embed)
    monkeypatch.setattr(stages, "_reconcile_embedding_config_change", lambda _conn: None)

    stage = make_embed_stage(db_path)
    assert stage.check_many is not None
    assert stage.execute_many is not None
    assert stage.check_many([path_a, path_b]) == {path_a, path_b}
    assert stage.execute_many([path_a, path_b]) is True
    assert embedded_calls == [["conv-a", "conv-c"]]


def test_default_convergence_stages_do_not_embed_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.delenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", raising=False)

    stage_names = [stage.name for stage in make_default_convergence_stages(tmp_path / "archive.sqlite")]

    assert stage_names == ["fts", "insights"]


def test_default_convergence_stages_include_embed_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "true")

    stage_names = [stage.name for stage in make_default_convergence_stages(tmp_path / "archive.sqlite")]

    assert stage_names == ["fts", "embed", "insights"]


def test_insights_stage_batches_sync_rebuild_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    rebuild_calls: list[tuple[list[str], int]] = []

    class FakeConnection:
        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    @contextmanager
    def fake_open_connection(path: Path) -> Iterator[FakeConnection]:
        assert path == db_path
        yield FakeConnection()

    def fake_rebuild(
        conn: FakeConnection,
        *,
        conversation_ids: list[str],
        page_size: int,
    ) -> SessionInsightCounts:
        del conn
        rebuild_calls.append((conversation_ids, page_size))
        return SessionInsightCounts(profiles=2, work_events=0, phases=0, threads=0, tag_rollups=0, day_summaries=0)

    monkeypatch.setattr("polylogue.storage.sqlite.connection.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.insights.session.rebuild.rebuild_session_insights_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )

    stage = make_insights_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert rebuild_calls == [(["conv-a", "conv-b"], 10)]


def test_embedding_config_enabled_with_key() -> None:
    """Embedding is enabled when config has both enabled flag and API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is True


def test_embedding_config_disabled_without_key() -> None:
    """Embedding is disabled when config has enabled flag but no API key."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = None
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False


def test_embedding_config_disabled_explicitly() -> None:
    """Embedding is disabled when config has key but enabled flag is False."""
    from unittest.mock import patch

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = False
        mock_cfg.return_value.voyage_api_key = "test-key"
        from polylogue.daemon.convergence_stages import _embedding_config_enabled

        assert _embedding_config_enabled() is False
