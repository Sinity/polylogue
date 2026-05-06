from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.daemon.convergence_stages import make_fts_stage, make_insights_stage
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

    def fake_rebuild(conn: FakeConnection, *, conversation_ids: list[str]) -> SessionInsightCounts:
        nonlocal rebuilt
        rebuilt = True
        assert conversation_ids == ["conv-1"]
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


def test_fts_stage_repairs_changed_conversations_without_full_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    repaired: list[list[str]] = []
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

    def fake_repair(conn: FakeConnection, conversation_ids: list[str]) -> None:
        repaired.append(conversation_ids)

    def fake_rebuild(conn: FakeConnection) -> None:
        nonlocal rebuilt
        rebuilt = True

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_connection", fake_open_connection)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.repair_fts_index_sync", fake_repair)
    monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.rebuild_fts_index_sync", fake_rebuild)
    monkeypatch.setattr(
        stages,
        "_conversation_ids_for_source_paths",
        lambda _conn, paths: {Path(paths[0]): ["conv-a"], Path(paths[1]): ["conv-b"]},
    )
    monkeypatch.setattr(stages, "_fts_needs_repair_for_conversations", lambda _conn, _ids: False)

    stage = make_fts_stage(db_path)
    assert stage.execute_many is not None
    assert stage.execute_many([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is True
    assert repaired == [["conv-a", "conv-b"]]
    assert committed is True
    assert rebuilt is False
