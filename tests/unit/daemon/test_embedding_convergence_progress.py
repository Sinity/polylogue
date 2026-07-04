from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import convergence_stages, embedding_backlog
from polylogue.daemon.status import format_daemon_status_lines
from polylogue.storage.embeddings.materialization import EmbedSessionOutcome, PendingSession


def test_periodic_embedding_backlog_waits_for_catch_up_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def fake_to_thread(_func: object, *_args: object, **_kwargs: object) -> object:
        calls.append("drain")
        raise asyncio.CancelledError

    async def exercise() -> None:
        catch_up_complete = asyncio.Event()
        monkeypatch.setattr("polylogue.paths.active_index_db_path", lambda: tmp_path / "index.db")
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(embedding_backlog, "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS", 0)
        task = asyncio.create_task(
            embedding_backlog.periodic_embedding_backlog_check(catch_up_complete=catch_up_complete)
        )
        await asyncio.sleep(0)
        assert calls == []
        catch_up_complete.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert calls == ["drain"]


class _EmbeddingConfig:
    embedding_enabled = True
    voyage_api_key = "pa-test"
    embedding_model = "voyage-4"
    embedding_dimension = 1024

    def get(self, key: str, default: object = None) -> object:
        values: dict[str, object] = {
            "voyage_api_key": "pa-test",
            "embedding_max_cost_usd": 5.0,
        }
        return values.get(key, default)


class _FakeRepository:
    def __init__(self, *, backend: object) -> None:
        self.backend = backend

    async def close(self) -> None:
        return None


def _seed_embedding_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                updated_at TEXT
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT
            );
            """
        )
        conn.execute("INSERT INTO sessions VALUES ('conv-a', 'A', '2026-01-01')")
        conn.execute("INSERT INTO sessions VALUES ('conv-b', 'B', '2026-01-02')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-1', 'conv-a', 'alpha')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-2', 'conv-a', 'beta')")
        conn.execute("INSERT INTO messages VALUES ('msg-b-1', 'conv-b', 'gamma')")
        conn.commit()


def test_daemon_embedding_backlog_drain_processes_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor_path = tmp_path / "index.db"
    db_anchor_path.touch()
    archive_db = tmp_path / "index.db"
    with sqlite3.connect(archive_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            INSERT INTO sessions VALUES ('codex-session:v1-b', 'Archive B', 1, 2);
            """
        )

    embedded_calls: list[tuple[Path, str]] = []

    def fake_embed(db_path: Path, _provider: object, session_id: str) -> EmbedSessionOutcome:
        embedded_calls.append((db_path, session_id))
        return EmbedSessionOutcome(
            status="embedded",
            session_id=session_id,
            embedded_message_count=2,
        )

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    processed = embedding_backlog.drain_embedding_backlog_once(db_anchor_path)

    assert processed == 2
    assert set(embedded_calls) == {(archive_db, "codex-session:v1-a"), (archive_db, "codex-session:v1-b")}
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        runs = list_embedding_catchup_runs(conn)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].scanned_sessions == 2
    assert runs[0].embedded_sessions == 2
    assert runs[0].skipped_sessions == 0
    assert runs[0].error_count == 0
    assert runs[0].embedded_messages == 4


def test_daemon_embedding_backlog_uses_bounded_pending_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    db_path.touch()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            """
        )

    observed_kwargs: dict[str, object] = {}

    def fake_select(*args: object, **kwargs: object) -> list[PendingSession]:
        observed_kwargs.update(kwargs)
        return [PendingSession(session_id="codex-session:v1-a", title="Archive A", message_count=2)]

    def fake_embed(_db_path: Path, _provider: object, session_id: str) -> EmbedSessionOutcome:
        return EmbedSessionOutcome(status="embedded", session_id=session_id, embedded_message_count=2)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.select_pending_archive_session_window", fake_select
    )
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 1
    assert observed_kwargs["include_stale_checks"] is False


def test_daemon_embedding_backlog_records_skipped_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor_path = tmp_path / "index.db"
    db_anchor_path.touch()
    with sqlite3.connect(db_anchor_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:skip-a', 'Skip A', 1, 2);
            INSERT INTO sessions VALUES ('codex-session:skip-b', 'Skip B', 1, 1);
            """
        )

    def fake_embed(_db_path: Path, _provider: object, session_id: str) -> EmbedSessionOutcome:
        return EmbedSessionOutcome(status="no_embeddable_messages", session_id=session_id)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    processed = embedding_backlog.drain_embedding_backlog_once(db_anchor_path)

    assert processed == 2
    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        runs = list_embedding_catchup_runs(conn)
    assert len(runs) == 1
    assert runs[0].scanned_sessions == 2
    assert runs[0].embedded_sessions == 0
    assert runs[0].skipped_sessions == 2
    assert runs[0].error_count == 0
    assert runs[0].embedded_messages == 0


def test_archive_convergence_embedding_uses_embeddings_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_db = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                message_count INTEGER NOT NULL,
                sort_key_ms INTEGER
            );
            INSERT INTO sessions VALUES ('codex-session:v1-a', 'Archive A', 2, 1);
            """
        )
    observed_vector_db_paths: list[Path] = []
    embedded_calls: list[tuple[Path, str]] = []
    fake_provider = MagicMock()

    def fake_create_vector_provider(**kwargs: object) -> object:
        observed_vector_db_paths.append(Path(str(kwargs["db_path"])))
        return fake_provider

    def fake_embed(db_path: Path, provider: object, session_id: str) -> EmbedSessionOutcome:
        assert provider is fake_provider
        embedded_calls.append((db_path, session_id))
        return EmbedSessionOutcome(
            status="embedded",
            session_id=session_id,
            embedded_message_count=2,
        )

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", fake_create_vector_provider)
    monkeypatch.setattr("polylogue.storage.embeddings.materialization.embed_archive_session_sync", fake_embed)

    ok = convergence_stages._embed_archive_sessions_sync(index_db, ("codex-session:v1-a",))

    assert ok is True
    assert observed_vector_db_paths == [embeddings_db]
    assert embedded_calls == [(index_db, "codex-session:v1-a")]


def test_daemon_embedding_backlog_drain_is_noop_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _seed_embedding_db(db_path)

    class DisabledConfig(_EmbeddingConfig):
        embedding_enabled = False

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: DisabledConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: DisabledConfig())

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 0


def test_daemon_embedding_backlog_drain_pauses_when_monthly_cap_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    _seed_embedding_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE embedding_catchup_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                stop_reason TEXT,
                rebuild INTEGER NOT NULL DEFAULT 0,
                max_sessions INTEGER,
                max_messages INTEGER,
                stop_after_seconds INTEGER,
                max_errors INTEGER,
                planned_sessions INTEGER NOT NULL DEFAULT 0,
                planned_messages INTEGER NOT NULL DEFAULT 0,
                processed_sessions INTEGER NOT NULL DEFAULT 0,
                embedded_sessions INTEGER NOT NULL DEFAULT 0,
                skipped_sessions INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                embedded_messages INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                last_session_id TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO embedding_catchup_runs (
                run_id, started_at, updated_at, status, estimated_cost_usd
            ) VALUES ('spent', datetime('now'), datetime('now'), 'completed', 5.0)
            """
        )
        conn.commit()

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 0


def test_daemon_status_lines_include_latest_embedding_catchup() -> None:
    lines = format_daemon_status_lines(
        {
            "embedding_readiness": {
                "embedding_enabled": True,
                "embedding_coverage_percent": 12.5,
                "embedding_pending_count": 10,
                "embedding_pending_message_count": 200,
                "embedding_stale_count": 0,
                "embedding_failure_count": 0,
                "embedding_estimated_cost_usd": 0.02,
                "embedding_model": "voyage-4",
                "embedding_dimension": 1024,
                "embedding_latest_catchup_run": {
                    "status": "running",
                    "processed_sessions": 3,
                    "planned_sessions": 10,
                    "embedded_messages": 42,
                },
            }
        }
    )

    assert "  latest catch-up: running, 3/10 convs, 42 msgs embedded" in lines
