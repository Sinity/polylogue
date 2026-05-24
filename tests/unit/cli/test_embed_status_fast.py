"""Fast-path coverage for ``polylogue embed status``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.embed import embed_command
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot


class _Cfg:
    def __init__(self, *, embedding_enabled: bool, voyage_api_key: str | None) -> None:
        self.embedding_enabled = embedding_enabled
        self.voyage_api_key = voyage_api_key


def _env(db_path: Path) -> Any:
    env = MagicMock()
    env.config.db_path = db_path
    return env


def _seed_archive_without_embedding_ledgers(db_path: Path, *, vec_table: bool = False) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, conversation_id TEXT, content_hash TEXT)")
        conn.executemany(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.executemany(
            "INSERT INTO messages (message_id, conversation_id, content_hash) VALUES (?, ?, ?)",
            [("msg-1", "conv-1", "h1"), ("msg-2", "conv-2", "h2")],
        )
        if vec_table:
            conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.commit()


def _payload(result_output: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(result_output))


def _run_status(db_path: Path, *args: str, cfg: _Cfg | None = None) -> dict[str, Any]:
    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=cfg or _Cfg(embedding_enabled=False, voyage_api_key=None),
    ):
        result = runner.invoke(
            embed_command,
            ["status", "--format", "json", *args],
            obj=_env(db_path),
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    return _payload(result.output)


def test_status_json_fast_path_handles_absent_embedding_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path)

    assert payload["status"] == "none"
    assert payload["total_conversations"] == 2
    assert payload["embedded_conversations"] == 0
    assert payload["pending_conversations"] == 2
    assert payload["pending_messages"] == 0
    assert payload["retrieval_bands"] == {}


def test_status_json_counts_empty_vec0_rowids_as_zero_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path, vec_table=True)

    payload = _run_status(db_path)

    assert payload["embedded_messages"] == 0
    assert payload["retrieval_ready"] is False
    assert payload["freshness_status"] == "none"


@pytest.mark.parametrize(
    ("cfg", "config_enabled", "has_key", "stage_enabled"),
    [
        (_Cfg(embedding_enabled=False, voyage_api_key="vk-live"), False, True, False),
        (_Cfg(embedding_enabled=True, voyage_api_key=None), True, False, False),
    ],
)
def test_status_json_reports_config_gate_combinations(
    tmp_path: Path,
    cfg: _Cfg,
    config_enabled: bool,
    has_key: bool,
    stage_enabled: bool,
) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path, cfg=cfg)

    assert payload["config_enabled"] is config_enabled
    assert payload["has_voyage_api_key"] is has_key
    assert payload["daemon_stage_enabled"] is stage_enabled


def test_status_json_detail_mode_runs_exact_retrieval_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    monkeypatch.setattr(
        "polylogue.storage.embeddings.embedding_stats.action_event_read_model_status_sync",
        lambda _conn: {
            "count": 0,
            "action_fts_count": 0,
            "action_fts_ready": True,
            "stale_count": 0,
        },
    )
    monkeypatch.setattr(
        "polylogue.storage.embeddings.embedding_stats.session_insight_status_sync",
        lambda _conn: SessionInsightStatusSnapshot(),
    )

    payload = _run_status(db_path, "--detail")

    assert payload["pending_conversations"] == 2
    assert payload["pending_messages"] == 2
    assert set(payload["retrieval_bands"]) == {
        "transcript_embeddings",
        "evidence_retrieval",
        "inference_retrieval",
        "enrichment_retrieval",
    }
