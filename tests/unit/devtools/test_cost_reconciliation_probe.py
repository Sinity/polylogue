from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from devtools.cost_reconciliation_probe import main
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    initialize_archive_database(root / "index.db", ArchiveTier.INDEX)
    return root


def _insert_session(
    conn: sqlite3.Connection,
    *,
    origin: str,
    native_id: str,
    model: str = "model",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_tokens: int = 3,
    cache_write_tokens: int = 2,
) -> str:
    conn.execute(
        "INSERT INTO sessions (origin, native_id, content_hash) VALUES (?, ?, ?)",
        (origin, native_id, b"s" * 32),
    )
    session_id = f"{origin}:{native_id}"
    conn.execute(
        """
        INSERT INTO session_model_usage (
            session_id, model_name, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, cost_provenance
        ) VALUES (?, ?, ?, ?, ?, ?, 'priced')
        """,
        (session_id, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
    )
    return session_id


def _write_codex_state(path: Path, *, tokens: int = 100) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE threads(id TEXT PRIMARY KEY, tokens_used INTEGER, model TEXT)")
        conn.execute("INSERT INTO threads(id, tokens_used, model) VALUES ('thread-1', ?, 'gpt-5-codex')", (tokens,))


def test_codex_probe_compares_copied_state_to_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _archive_root(tmp_path)
    with sqlite3.connect(archive / "index.db") as conn:
        session_id = _insert_session(
            conn,
            origin="codex-session",
            native_id="thread-1",
            model="gpt-5-codex",
            input_tokens=20,
            output_tokens=10,
            cache_read_tokens=60,
            cache_write_tokens=10,
        )
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
              session_id, position, provider_event_type, model_name,
              total_input_tokens, total_output_tokens, total_cached_input_tokens,
              total_cache_write_tokens, total_tokens
            ) VALUES (?, 0, 'token_count', 'gpt-5-codex', 800, 200, 700, 100, 1800)
            """,
            (session_id,),
        )
    codex_state = tmp_path / "state_5.sqlite"
    _write_codex_state(codex_state, tokens=100)

    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--codex-state",
                str(codex_state),
                "--scratch-dir",
                str(tmp_path / "scratch"),
                "--json",
                "--check",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    codex = payload["sections"][0]
    assert payload["ok"] is True
    assert codex["name"] == "codex"
    assert codex["status"] == "pass"
    assert codex["comparison"]["compared"] == 1
    assert codex["comparison"]["median_ratio"] == 1.0
    assert "session_model_usage disjoint lanes" in codex["details"]["lane_contract"]
    assert Path(codex["details"]["copied_path"]).exists()


def test_codex_probe_check_fails_outside_tolerance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _archive_root(tmp_path)
    with sqlite3.connect(archive / "index.db") as conn:
        session_id = _insert_session(conn, origin="codex-session", native_id="thread-1", model="gpt-5-codex")
        conn.execute(
            """
            INSERT INTO session_provider_usage_events (
              session_id, position, provider_event_type, model_name, total_tokens
            ) VALUES (?, 0, 'token_count', 'gpt-5-codex', 200)
            """,
            (session_id,),
        )
    codex_state = tmp_path / "state_5.sqlite"
    _write_codex_state(codex_state, tokens=100)

    assert main(["--archive-root", str(archive), "--codex-state", str(codex_state), "--json", "--check"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed_sections"] == ["codex"]
    assert payload["sections"][0]["comparison"]["outside_tolerance"] == 1


def test_claude_probe_compares_model_usage_lanes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _archive_root(tmp_path)
    with sqlite3.connect(archive / "index.db") as conn:
        _insert_session(conn, origin="claude-code-session", native_id="claude-1", model="claude-sonnet-4-5")
    stats = tmp_path / "stats-cache.json"
    stats.write_text(
        json.dumps(
            {
                "modelUsage": {
                    "claude-sonnet-4-5": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "cacheReadInputTokens": 3,
                        "cacheCreationInputTokens": 2,
                        "costUSD": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["--archive-root", str(archive), "--claude-stats-cache", str(stats), "--json", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    claude = payload["sections"][1]
    assert claude["status"] == "pass"
    assert claude["comparison"]["compared"] == 4
    assert claude["details"]["lane_contract"].startswith("stats-cache modelUsage lanes")


def test_missing_optional_and_required_external_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _archive_root(tmp_path)

    assert main(["--archive-root", str(archive), "--codex-state", str(tmp_path / "missing.sqlite"), "--json"]) == 0
    optional_payload = json.loads(capsys.readouterr().out)
    assert optional_payload["ok"] is True
    assert optional_payload["sections"][0]["status"] == "skip"

    assert (
        main(
            [
                "--archive-root",
                str(archive),
                "--codex-state",
                str(tmp_path / "missing.sqlite"),
                "--require-codex",
                "--json",
                "--check",
            ]
        )
        == 1
    )
    required_payload = json.loads(capsys.readouterr().out)
    assert required_payload["ok"] is False
    assert required_payload["failed_sections"] == ["codex"]


def test_malformed_claude_stats_cache_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive = _archive_root(tmp_path)
    stats = tmp_path / "stats-cache.json"
    stats.write_text("{}", encoding="utf-8")

    assert main(["--archive-root", str(archive), "--claude-stats-cache", str(stats), "--json", "--check"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["sections"][1]["status"] == "fail"
    assert "modelUsage" in payload["sections"][1]["summary"]
