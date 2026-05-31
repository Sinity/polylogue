from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.artifact_taxonomy import classify_artifact, classify_artifact_path
from polylogue.config import Source
from polylogue.core.json import JSONDocument
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.base import ParsedConversation
from polylogue.sources.source_parsing import iter_source_conversations
from polylogue.types import ContentBlockType, Provider


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
                "tokens": {"total": 10},
                "thoughts": [{"text": "reasoned"}],
                "toolCalls": [{"id": "tool-1", "name": "read_file", "arguments": {"path": "README.md"}}],
            },
        ],
    }

    assert detect_provider(payload) is Provider.GEMINI_CLI
    classification = classify_artifact(payload, provider=Provider.GEMINI_CLI, source_path="chats/session.json")
    assert classification.parse_as_conversation is True

    [conversation] = parse_payload("gemini-cli", payload, "fallback")

    assert conversation.source_name is Provider.GEMINI_CLI
    assert conversation.provider_conversation_id == "gemini-session-1"
    assert conversation.created_at == "2026-04-08T20:45:00.000Z"
    assert conversation.updated_at == "2026-04-08T20:47:00.000Z"
    assert conversation.title == "Parser work"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].provider_meta == {"model": "gemini-test", "tokens": {"total": 10}}
    assert {block.type for block in conversation.messages[1].content_blocks} >= {
        ContentBlockType.TEXT,
        ContentBlockType.THINKING,
        ContentBlockType.TOOL_USE,
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
    assert classification.parse_as_conversation is True

    [conversation] = parse_payload("hermes", payload, "fallback")

    assert conversation.source_name is Provider.HERMES
    assert conversation.provider_conversation_id == "hermes-session-1"
    assert conversation.created_at == "2026-05-07T08:39:43.000000"
    assert conversation.updated_at == "2026-05-07T08:46:00.000000"
    assert conversation.provider_meta == {
        "source_family": "hermes",
        "model": "local-model",
        "base_url": "http://localhost",
        "platform": "linux",
        "tools": [{"name": "shell"}],
    }
    assert conversation.messages[0].role == "system"
    assert conversation.messages[2].role == "assistant"
    assert {block.type for block in conversation.messages[2].content_blocks} >= {
        ContentBlockType.TEXT,
        ContentBlockType.THINKING,
        ContentBlockType.TOOL_USE,
    }
    assert conversation.messages[3].role == "tool"
    assert any(block.type is ContentBlockType.TOOL_RESULT for block in conversation.messages[3].content_blocks)


def test_agent_sidecars_are_classified_as_non_conversation() -> None:
    logs = classify_artifact_path("~/.gemini/tmp/polylogue/logs.json", provider=Provider.GEMINI_CLI)
    request_dump = classify_artifact_path("~/.hermes/sessions/request_dump_1.json", provider=Provider.HERMES)
    antigravity_pb = classify_artifact_path(
        "~/.gemini/antigravity/conversations/session.pb",
        provider=Provider.ANTIGRAVITY,
    )
    antigravity_resolved = classify_artifact_path(
        "~/.gemini/antigravity/brain/session/task.md.resolved.1",
        provider=Provider.ANTIGRAVITY,
    )

    assert logs is not None
    assert logs.parse_as_conversation is False
    assert request_dump is not None
    assert request_dump.parse_as_conversation is False
    assert antigravity_pb is not None
    assert antigravity_pb.parse_as_conversation is False
    assert antigravity_resolved is not None
    assert antigravity_resolved.parse_as_conversation is False


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
    assert classification.parse_as_conversation is True

    [conversation] = parse_payload(
        Provider.ANTIGRAVITY,
        payload,
        "fallback",
        source_path=str(metadata),
    )

    assert conversation.source_name is Provider.ANTIGRAVITY
    assert conversation.provider_conversation_id == "03c22aa3-8b7f-438d-baa8-d12567249cd9:implementation_plan.md"
    assert conversation.updated_at == "2026-01-07T19:08:15.216541610Z"
    assert conversation.messages[0].text == "# Implementation Plan\n\nDo the work.\n"
    assert conversation.provider_meta is not None
    assert conversation.provider_meta["source_family"] == "antigravity"


def test_antigravity_language_server_markdown_export_parses_turns() -> None:
    markdown = """# Chat Conversation

Note: _This is purely the output of the chat conversation._

### User Input

Run the checks.

*User accepted the command `pytest -q`*

### Planner Response

Checks passed.
"""
    summary = antigravity.AntigravityConversationSummary(
        cascade_id="e85783e3-f047-49b8-9035-4029f58dd04a",
        title="Refactoring and Executing Plan",
        workspace_name="Sinity/sinex",
        snippet="Run the checks.",
        last_modified_time="2026-03-05T04:21:34.468316671Z",
    )

    conversation = antigravity.parse_markdown_export(markdown, summary)

    assert conversation.source_name is Provider.ANTIGRAVITY
    assert conversation.provider_conversation_id == "e85783e3-f047-49b8-9035-4029f58dd04a"
    assert conversation.title == "Refactoring and Executing Plan"
    assert conversation.updated_at == "2026-03-05T04:21:34.468316671Z"
    assert [message.role for message in conversation.messages] == ["user", "assistant"]
    assert "pytest -q" in (conversation.messages[0].text or "")
    assert conversation.messages[1].text == "Checks passed."
    assert conversation.provider_meta is not None
    assert conversation.provider_meta["source_format"] == "language_server_markdown_export"
    assert conversation.provider_meta["workspace_name"] == "Sinity/sinex"


def test_antigravity_language_server_export_dispatches_as_conversation() -> None:
    payload = antigravity.markdown_export_payload(
        antigravity.AntigravityConversationSummary(
            cascade_id="cascade-1",
            title="Conversation",
            last_modified_time="2026-03-05T04:21:34Z",
        ),
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
    )

    assert detect_provider(payload) is Provider.ANTIGRAVITY
    classification = classify_artifact(payload, provider=Provider.ANTIGRAVITY)
    assert classification.parse_as_conversation is True

    [conversation] = parse_payload(Provider.ANTIGRAVITY, payload, "fallback")

    assert conversation.provider_conversation_id == "cascade-1"
    assert conversation.updated_at == "2026-03-05T04:21:34Z"
    assert [message.text for message in conversation.messages] == ["hello", "hi"]


def test_antigravity_source_walk_prefers_language_server_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "conversations").mkdir()
    exported = antigravity.parse_markdown_export(
        "### User Input\n\nhello\n\n### Planner Response\n\nhi",
        antigravity.AntigravityConversationSummary(cascade_id="cascade-1", title="Conversation"),
    )

    def fake_exports(root: Path) -> list[ParsedConversation]:
        assert root == tmp_path
        return [exported]

    monkeypatch.setattr("polylogue.sources.source_parsing.antigravity.iter_language_server_exports", fake_exports)

    conversations = list(iter_source_conversations(Source(name="antigravity", path=tmp_path)))

    assert [conversation.provider_conversation_id for conversation in conversations] == ["cascade-1"]
    assert conversations[0].messages[0].text == "hello"


def test_antigravity_source_walk_ingests_metadata_not_config(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "task.md").write_text("Task artifact", encoding="utf-8")
    (session_dir / "task.md.metadata.json").write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Task","updatedAt":"2026-01-07T00:00:00Z"}',
        encoding="utf-8",
    )
    (tmp_path / "mcp_config.json").write_text("{}", encoding="utf-8")

    conversations = list(iter_source_conversations(Source(name="antigravity", path=tmp_path)))

    assert len(conversations) == 1
    assert conversations[0].source_name is Provider.ANTIGRAVITY
