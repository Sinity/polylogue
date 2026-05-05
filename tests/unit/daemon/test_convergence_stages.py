from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from polylogue.daemon.convergence_stages import make_insights_stage
from polylogue.storage.insights.session.runtime import SessionInsightCounts


def test_insights_stage_rebuilds_sync_against_configured_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "archive.sqlite"
    opened_paths: list[Path] = []
    rebuilt = False

    class FakeConnection:
        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    @contextmanager
    def fake_open_connection(path: Path) -> Iterator[FakeConnection]:
        opened_paths.append(path)
        yield FakeConnection()

    def fake_rebuild(conn: FakeConnection) -> SessionInsightCounts:
        nonlocal rebuilt
        rebuilt = True
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
