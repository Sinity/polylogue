from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.archive.raw_payload import build_raw_payload_envelope
from polylogue.config import Source
from polylogue.core.enums import BlockType, Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.live.batch_support import _detect_provider_from_path_sample, _parse_path_as_session_artifact
from polylogue.sources.parsers import antigravity, hermes_state
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions, iter_source_sessions_with_raw


def _write_hermes_state_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (17);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cwd TEXT,
                git_branch TEXT,
                git_repo_root TEXT,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                handoff_state TEXT,
                handoff_platform TEXT,
                handoff_error TEXT,
                compression_failure_cooldown_until REAL,
                compression_failure_error TEXT,
                rewind_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                platform_message_id TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, model, model_config, system_prompt, parent_session_id,
                started_at, ended_at, end_reason, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, reasoning_tokens, cwd,
                git_branch, git_repo_root, estimated_cost_usd, actual_cost_usd,
                title, api_call_count, rewind_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-root",
                "nous-hermes-test",
                "{}",
                "be precise",
                None,
                1_775_000_000.0,
                1_775_000_100.0,
                "completed",
                10,
                20,
                3,
                4,
                5,
                "/realm/project/polylogue",
                "feature/hermes",
                "/realm/project/polylogue",
                0.002,
                0.0015,
                "Hermes parser work",
                2,
                1,
            ),
        )
        conn.execute(
            """
            INSERT INTO sessions (
                id, model, model_config, parent_session_id, started_at,
                ended_at, end_reason, title
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-child",
                "nous-hermes-test",
                json.dumps({"_delegate_from": "hermes-root"}),
                "hermes-root",
                1_775_000_101.0,
                1_775_000_150.0,
                "completed",
                "Delegate child",
            ),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp, token_count) VALUES (?, ?, ?, ?, ?)",
            ("hermes-root", "user", "run pytest", 1_775_000_001.0, 10),
        )
        conn.execute(
            """
            INSERT INTO messages(
                session_id, role, content, tool_calls, timestamp, token_count,
                finish_reason, reasoning_content, reasoning_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hermes-root",
                "assistant",
                "running",
                json.dumps([{"id": "call-1", "function": {"name": "shell", "arguments": '{"cmd":"pytest"}'}}]),
                1_775_000_002.0,
                20,
                "tool_calls",
                "need test proof",
                json.dumps({"effort": "medium"}),
            ),
        )
        conn.execute(
            """
            INSERT INTO messages(session_id, role, content, tool_call_id, tool_name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("hermes-root", "tool", "passed", "call-1", "shell", 1_775_000_003.0),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("hermes-child", "assistant", "delegated work", 1_775_000_102.0),
        )


def test_gemini_cli_session_document_parses_through_dispatch() -> None:
    payload: JSONDocument = {
        "sessionId": "gemini-session-1",
        "projectHash": "project-hash",
        "startTime": "2026-04-08T20:45:00.000Z",
        "lastUpdated": "2026-04-08T20:47:00.000Z",
        "kind": "chat",
        "summary": "Parser work",
        "messages": [
            {"id": "u1", "timestamp": "2026-04-08T20:45:01.000Z", "type": "user", "content": ["hello"]},
            {
                "id": "a1",
                "timestamp": "2026-04-08T20:45:02.000Z",
                "type": "gemini",
                "content": "response",
                "model": "gemini-test",
                "durationMs": 900,
                "tokens": {"total": 10},
                "thoughts": [{"text": "reasoned"}],
                "toolCalls": [{"id": "tool-1", "name": "read_file", "arguments": {"path": "README.md"}}],
            },
        ],
    }

    assert detect_provider(payload) is Provider.GEMINI_CLI
    classification = classify_artifact(payload, provider=Provider.GEMINI_CLI, source_path="chats/session.json")
    assert classification.parse_as_session is True

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert session.source_name is Provider.GEMINI_CLI
    assert session.provider_session_id == "gemini-session-1"
    assert session.created_at == "2026-04-08T20:45:00.000Z"
    assert session.updated_at == "2026-04-08T20:47:00.000Z"
    assert session.title == "Parser work"
    assert session.messages[1].role == "assistant"
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == "a1"
    assert session.messages[1].model_name == "gemini-test"
    assert session.messages[1].duration_ms == 900
    assert session.messages[1].input_tokens == 0
    assert session.messages[1].output_tokens == 10
    assert {block.type for block in session.messages[1].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }


def test_hermes_session_document_parses_through_dispatch() -> None:
    payload: JSONDocument = {
        "session_id": "hermes-session-1",
        "model": "local-model",
        "base_url": "http://localhost",
        "platform": "linux",
        "session_start": "2026-05-07T08:39:43.000000",
        "last_updated": "2026-05-07T08:46:00.000000",
        "system_prompt": "be concise",
        "tools": [{"name": "shell"}],
        "message_count": 3,
        "messages": [
            {"role": "user", "content": "run checks"},
            {
                "role": "assistant",
                "content": "running",
                "model": "local-override",
                "durationMs": 1250,
                "usage": {"input_tokens": 4, "output_tokens": 6},
                "reasoning_content": "need tests",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "run_shell_command", "arguments": '{"cmd":"pytest"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "passed"},
        ],
    }

    assert detect_provider(payload) is Provider.HERMES
    classification = classify_artifact(payload, provider=Provider.HERMES, source_path="session_1.json")
    assert classification.parse_as_session is True

    [session] = parse_payload("hermes", payload, "fallback")

    assert session.source_name is Provider.HERMES
    assert session.provider_session_id == "hermes-session-1"
    assert session.created_at == "2026-05-07T08:39:43.000000"
    assert session.updated_at == "2026-05-07T08:46:00.000000"
    assert session.messages[0].role == "system"
    assert [message.position for message in session.messages] == [0, 1, 2, 3]
    assert session.messages[0].model_name == "local-model"
    assert session.messages[2].role == "assistant"
    assert session.messages[2].model_name == "local-override"
    assert session.messages[2].duration_ms == 1250
    assert session.messages[2].input_tokens == 4
    assert session.messages[2].output_tokens == 6
    assert {block.type for block in session.messages[2].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }
    assert session.messages[3].role == "tool"
    assert session.messages[3].is_active_leaf is True
    assert session.active_leaf_message_provider_id == "call-1"
    assert any(block.type is BlockType.TOOL_RESULT for block in session.messages[3].blocks)


def test_hermes_state_db_parses_authoritative_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    sessions = hermes_state.parse_state_db(db_path, fallback_id="fallback")

    assert [session.provider_session_id for session in sessions] == ["hermes-root", "hermes-child"]
    root = sessions[0]
    assert root.source_name is Provider.HERMES
    assert root.title == "Hermes parser work"
    assert root.instructions_text == "be precise"
    assert root.working_directories == ["/realm/project/polylogue"]
    assert root.git_branch == "feature/hermes"
    assert root.reported_cost_usd == 0.0015
    assert root.messages[0].role == "system"
    assert root.messages[2].provider_message_id == "hermes-root:message:2"
    assert root.messages[2].output_tokens == 20
    assert {block.type for block in root.messages[2].blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.TOOL_USE,
    }
    assert any(block.type is BlockType.TOOL_RESULT and block.tool_id == "call-1" for block in root.messages[3].blocks)
    usage_events = [event for event in root.session_events if event.event_type == "token_count"]
    assert usage_events
    assert usage_events[0].payload["total_token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 3,
        "cache_write_tokens": 4,
        "reasoning_output_tokens": 5,
        "total_tokens": 42,
    }

    child = sessions[1]
    assert child.parent_session_provider_id == "hermes-root"
    assert child.branch_type is not None
    assert child.branch_type.value == "subagent"


def test_hermes_state_db_dispatch_marker_parses_multiple_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)
    payload = hermes_state.marker_payload(db_path)

    assert detect_provider(payload) is Provider.HERMES
    sessions = parse_payload("hermes", payload, "fallback", source_path=str(db_path))

    assert [session.provider_session_id for session in sessions] == ["hermes-root", "hermes-child"]


def test_hermes_state_db_source_iterator_captures_raw_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    blob_root = tmp_path / "blob"
    _write_hermes_state_db(db_path)

    rows = list(
        iter_source_sessions_with_raw(
            Source(name="hermes", path=db_path),
            capture_raw=True,
            blob_root=blob_root,
        )
    )

    assert [session.provider_session_id for _raw, session in rows] == ["hermes-root", "hermes-child"]
    raw = rows[0][0]
    assert raw is not None
    assert raw.raw_bytes == b""
    assert raw.blob_hash
    assert raw.blob_size and raw.blob_size > 0


def test_hermes_state_db_raw_payload_envelope_uses_marker(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    envelope = build_raw_payload_envelope(db_path, source_path=db_path, fallback_provider="inbox")

    assert envelope.provider is Provider.HERMES
    assert envelope.artifact.parse_as_session is True
    assert envelope.payload == hermes_state.marker_payload(db_path)


def test_hermes_state_db_live_batch_classifies_as_session_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    _write_hermes_state_db(db_path)

    assert _detect_provider_from_path_sample(db_path, Provider.UNKNOWN) is Provider.HERMES
    assert _parse_path_as_session_artifact(db_path, provider=Provider.HERMES) is True


def test_agent_sidecars_are_classified_as_non_session() -> None:
    logs = classify_artifact_path("~/.gemini/tmp/polylogue/logs.json", provider=Provider.GEMINI_CLI)
    request_dump = classify_artifact_path("~/.hermes/sessions/request_dump_1.json", provider=Provider.HERMES)
    antigravity_pb = classify_artifact_path(
        "~/.gemini/antigravity/sessions/session.pb",
        provider=Provider.ANTIGRAVITY,
    )
    antigravity_resolved = classify_artifact_path(
        "~/.gemini/antigravity/brain/session/task.md.resolved.1",
        provider=Provider.ANTIGRAVITY,
    )

    assert logs is not None
    assert logs.parse_as_session is False
    assert request_dump is not None
    assert request_dump.parse_as_session is False
    assert antigravity_pb is not None
    assert antigravity_pb.parse_as_session is False
    assert antigravity_resolved is not None
    assert antigravity_resolved.parse_as_session is False


def test_runtime_provider_identity_keeps_gemini_surfaces_distinct() -> None:
    assert Provider.from_string("gemini") is Provider.GEMINI
    assert Provider.from_string("aistudio") is Provider.GEMINI
    assert Provider.from_string("gemini-cli") is Provider.GEMINI_CLI
    assert Provider.from_string("antigravity") is Provider.ANTIGRAVITY


def test_antigravity_brain_artifact_metadata_parses_sibling_markdown(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "03c22aa3-8b7f-438d-baa8-d12567249cd9"
    session_dir.mkdir(parents=True)
    artifact = session_dir / "implementation_plan.md"
    metadata = session_dir / "implementation_plan.md.metadata.json"
    artifact.write_text("# Implementation Plan\n\nDo the work.\n", encoding="utf-8")
    payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Implementation plan for source ingestion",
        "updatedAt": "2026-01-07T19:08:15.216541610Z",
    }
    metadata.write_text(
        (
            '{"artifactType":"ARTIFACT_TYPE_OTHER",'
            '"summary":"Implementation plan for source ingestion",'
            '"updatedAt":"2026-01-07T19:08:15.216541610Z"}'
        ),
        encoding="utf-8",
    )

    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY, source_path=metadata)
    assert classification.parse_as_session is True

    [session] = parse_payload(
        Provider.ANTIGRAVITY,
        payload,
        "fallback",
        source_path=str(metadata),
    )

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "03c22aa3-8b7f-438d-baa8-d12567249cd9:implementation_plan.md"
    assert session.updated_at == "2026-01-07T19:08:15.216541610Z"
    assert session.messages[0].text == "# Implementation Plan\n\nDo the work.\n"
    assert session.source_name is Provider.ANTIGRAVITY


def test_antigravity_language_server_markdown_export_parses_turns() -> None:
    markdown = """# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Run the checks.

*User accepted the command `pytest -q`*

### Planner Response

Checks passed.
"""
    summary = antigravity.AntigravitySessionSummary(
        cascade_id="e85783e3-f047-49b8-9035-4029f58dd04a",
        title="Refactoring and Executing Plan",
        workspace_name="Sinity/sinex",
        snippet="Run the checks.",
        last_modified_time="2026-03-05T04:21:34.468316671Z",
    )

    session = antigravity.parse_markdown_export(markdown, summary)

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "e85783e3-f047-49b8-9035-4029f58dd04a"
    assert session.title == "Refactoring and Executing Plan"
    assert session.updated_at == "2026-03-05T04:21:34.468316671Z"
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert "pytest -q" in (session.messages[0].text or "")
    assert session.messages[1].text == "Checks passed."
    assert session.provider_session_id == "e85783e3-f047-49b8-9035-4029f58dd04a"


def test_antigravity_language_server_export_dispatches_as_session() -> None:
    payload = antigravity.markdown_export_payload(
        antigravity.AntigravitySessionSummary(
            cascade_id="cascade-1",
            title="Session",
            last_modified_time="2026-03-05T04:21:34Z",
        ),
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
    )

    assert detect_provider(payload) is Provider.ANTIGRAVITY
    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY)
    assert classification.parse_as_session is True

    [session] = parse_payload(Provider.ANTIGRAVITY, payload, "fallback")

    assert session.provider_session_id == "cascade-1"
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.text for message in session.messages] == ["hello", "hi"]


def test_antigravity_source_walk_prefers_language_server_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sessions").mkdir()
    exported = antigravity.parse_markdown_export(
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
        antigravity.AntigravitySessionSummary(cascade_id="cascade-1", title="Session"),
    )

    def fake_exports(root: Path) -> list[ParsedSession]:
        assert root == tmp_path
        return [exported]

    monkeypatch.setattr("polylogue.sources.source_parsing.antigravity.iter_language_server_exports", fake_exports)

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert [session.provider_session_id for session in sessions] == ["cascade-1"]
    assert sessions[0].messages[0].text == "hello"


def test_antigravity_source_walk_ingests_metadata_not_config(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "task.md").write_text("Task artifact", encoding="utf-8")
    (session_dir / "task.md.metadata.json").write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Task","updatedAt":"2026-01-07T00:00:00Z"}',
        encoding="utf-8",
    )
    (tmp_path / "mcp_config.json").write_text("{}", encoding="utf-8")

    sessions = list(iter_source_sessions(Source(name="antigravity", path=tmp_path)))

    assert len(sessions) == 1
    assert sessions[0].source_name is Provider.ANTIGRAVITY
