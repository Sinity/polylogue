from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import convergence_stages, embedding_backlog
from polylogue.daemon.status import format_daemon_status_lines
from polylogue.storage.embeddings.materialization import EmbedConversationOutcome
from polylogue.storage.embeddings.progress import latest_embedding_catchup_run


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
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT,
                updated_at TEXT
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                text TEXT
            );
            """
        )
        conn.execute("INSERT INTO conversations VALUES ('conv-a', 'A', '2026-01-01')")
        conn.execute("INSERT INTO conversations VALUES ('conv-b', 'B', '2026-01-02')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-1', 'conv-a', 'alpha')")
        conn.execute("INSERT INTO messages VALUES ('msg-a-2', 'conv-a', 'beta')")
        conn.execute("INSERT INTO messages VALUES ('msg-b-1', 'conv-b', 'gamma')")
        conn.commit()


def test_daemon_embedding_batch_records_catchup_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "polylogue.db"
    _seed_embedding_db(db_path)
    outcomes = iter(
        [
            EmbedConversationOutcome(status="embedded", conversation_id="conv-a", embedded_message_count=2),
            EmbedConversationOutcome(status="no_messages", conversation_id="conv-b"),
        ]
    )

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.repository.ConversationRepository", _FakeRepository)
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.embed_conversation_sync",
        lambda *_args, **_kwargs: next(outcomes),
    )

    ok = convergence_stages._embed_conversations_sync(db_path, ("conv-a", "conv-b"))

    assert ok is True
    with sqlite3.connect(db_path) as conn:
        payload = latest_embedding_catchup_run(conn)
    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["planned_conversations"] == 2
    assert payload["planned_messages"] == 3
    assert payload["processed_conversations"] == 2
    assert payload["embedded_conversations"] == 1
    assert payload["skipped_conversations"] == 1
    assert payload["error_count"] == 0
    assert payload["embedded_messages"] == 2


def test_daemon_embedding_batch_marks_error_stop_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "polylogue.db"
    _seed_embedding_db(db_path)

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr("polylogue.storage.repository.ConversationRepository", _FakeRepository)
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda **_: MagicMock())
    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.embed_conversation_sync",
        lambda *_args, **_kwargs: EmbedConversationOutcome(
            status="error",
            conversation_id="conv-a",
            error="provider 429",
        ),
    )

    ok = convergence_stages._embed_conversations_sync(db_path, ("conv-a", "conv-b"), max_errors=1)

    assert ok is False
    with sqlite3.connect(db_path) as conn:
        payload = latest_embedding_catchup_run(conn)
    assert payload is not None
    assert payload["status"] == "stopped"
    assert payload["stop_reason"] == "max errors reached (1)"
    assert payload["processed_conversations"] == 1
    assert payload["error_count"] == 1


def test_daemon_embedding_backlog_drain_processes_bounded_pending_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "polylogue.db"
    _seed_embedding_db(db_path)
    embedded_calls: list[tuple[str, ...]] = []
    cost_caps: list[float | None] = []

    def fake_embed(
        _db_path: Path,
        conversation_ids: tuple[str, ...],
        *,
        max_cost_usd: float | None = None,
        **_kwargs: object,
    ) -> bool:
        embedded_calls.append(tuple(conversation_ids))
        cost_caps.append(max_cost_usd)
        return True

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(convergence_stages, "_embed_conversations_sync", fake_embed)

    processed = embedding_backlog.drain_embedding_backlog_once(db_path)

    assert processed == 2
    assert embedded_calls == [("conv-a", "conv-b")]
    assert cost_caps == [5.0]


def test_daemon_embedding_backlog_drain_is_noop_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "polylogue.db"
    _seed_embedding_db(db_path)

    class DisabledConfig(_EmbeddingConfig):
        embedding_enabled = False

    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: DisabledConfig())
    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: DisabledConfig())
    monkeypatch.setattr(
        convergence_stages,
        "_embed_conversations_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("embedding must not run")),
    )

    assert embedding_backlog.drain_embedding_backlog_once(db_path) == 0


def test_daemon_embedding_backlog_drain_pauses_when_monthly_cap_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "polylogue.db"
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
                max_conversations INTEGER,
                max_messages INTEGER,
                stop_after_seconds INTEGER,
                max_errors INTEGER,
                planned_conversations INTEGER NOT NULL DEFAULT 0,
                planned_messages INTEGER NOT NULL DEFAULT 0,
                processed_conversations INTEGER NOT NULL DEFAULT 0,
                embedded_conversations INTEGER NOT NULL DEFAULT 0,
                skipped_conversations INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                embedded_messages INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                last_conversation_id TEXT
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
    monkeypatch.setattr(
        convergence_stages,
        "_embed_conversations_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("embedding must not run")),
    )

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
                    "processed_conversations": 3,
                    "planned_conversations": 10,
                    "embedded_messages": 42,
                },
            }
        }
    )

    assert "  latest catch-up: running, 3/10 convs, 42 msgs embedded" in lines
