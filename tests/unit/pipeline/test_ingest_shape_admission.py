"""Classifier-admitted shapes materialize through retained publication."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.artifact_taxonomy import classify_artifact
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from tests.infra.retained_replay import publish_retained_payload


def _publish(tmp_path: Path, provider: Provider, source_path: str, content: bytes) -> tuple[str, ...]:
    archive_root = tmp_path / "archive"
    import asyncio

    _, session_ids = asyncio.run(
        publish_retained_payload(
            archive_root,
            provider=provider,
            payload=content,
            source_path=source_path,
            acquired_at_ms=1,
        )
    )
    return session_ids


def test_claude_project_export_reaches_parser_through_retained_publication(tmp_path: Path) -> None:
    payload: JSONValue = {"uuid": "p1", "docs": [{"uuid": "d1", "content": "x"}], "prompt_template": "t"}
    assert classify_artifact(payload, provider=Provider.CLAUDE_AI).parse_as_session

    assert _publish(tmp_path, Provider.CLAUDE_AI, "projects/p1.json", json.dumps(payload).encode())
    with sqlite3.connect(tmp_path / "archive" / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("project:p1",)]


def test_claude_account_memory_export_reaches_parser_through_retained_publication(tmp_path: Path) -> None:
    payload: JSONValue = [{"account_uuid": "acct-1", "conversations_memory": "Prefers short answers."}]
    assert classify_artifact(payload, provider=Provider.CLAUDE_AI).parse_as_session

    assert _publish(tmp_path, Provider.CLAUDE_AI, "memories.json", json.dumps(payload).encode())
    with sqlite3.connect(tmp_path / "archive" / "index.db") as conn:
        assert conn.execute("SELECT native_id FROM sessions").fetchall() == [("account-memory:acct-1",)]


def test_project_signature_alone_does_not_admit_a_bare_uuid_document() -> None:
    assert not classify_artifact({"uuid": "p1", "docs": []}, provider=Provider.CLAUDE_AI).parse_as_session


def _codex_rollout(*extra: JSONValue) -> list[JSONValue]:
    return [
        {
            "type": "session_meta",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"id": "rollout-usage", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/workspace"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-01T00:00:01Z",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        },
        *extra,
    ]


def test_codex_rollout_with_bare_token_usage_record_retains_its_usage(tmp_path: Path) -> None:
    records = _codex_rollout({"type": "token_usage_record", "input_tokens": 10})
    content = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    assert classify_artifact(records, provider=Provider.CODEX).parse_as_session

    assert _publish(tmp_path, Provider.CODEX, "sessions/rollout-usage.jsonl", content)
    with sqlite3.connect(tmp_path / "archive" / "index.db") as conn:
        events = conn.execute(
            "SELECT payload_json FROM session_events WHERE event_type = 'token_usage_record'"
        ).fetchall()
    assert len(events) == 1
    assert json.loads(events[0][0])["usage"] == {"input_tokens": 10}


def test_codex_bare_token_usage_record_needs_a_known_counter() -> None:
    assert not classify_artifact(
        _codex_rollout({"type": "token_usage_record", "note": "no counters"}), provider=Provider.CODEX
    ).parse_as_session
