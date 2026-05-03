from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import write_capture_envelope
from polylogue.config import Source, get_config
from polylogue.pipeline.runner import run_sources
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import Provider


def _capture_payload() -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "chatgpt:conv-123",
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "conv-123",
            "title": "Work plan",
            "updated_at": "2026-04-24T00:00:01+00:00",
            "model": "gpt-5.4",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Draft the plan", "ordinal": 0},
                {
                    "provider_turn_id": "a1",
                    "role": "assistant",
                    "text": "Here is the plan",
                    "ordinal": 1,
                    "attachments": [
                        {
                            "provider_attachment_id": "att-1",
                            "name": "plan.md",
                            "mime_type": "text/markdown",
                            "url": "https://chatgpt.com/attachment/1",
                        }
                    ],
                },
                {"provider_turn_id": "a1", "role": "assistant", "text": "duplicate", "ordinal": 2},
            ],
        },
    }


def test_browser_capture_detects_inner_provider() -> None:
    assert detect_provider(_capture_payload()) is Provider.CHATGPT


def test_browser_capture_parses_session_metadata_and_deduplicates_turns() -> None:
    parsed = parse_payload(Provider.CHATGPT, _capture_payload(), "fallback")

    assert len(parsed) == 1
    conversation = parsed[0]
    assert conversation.provider_name is Provider.CHATGPT
    assert conversation.provider_conversation_id == "conv-123"
    assert conversation.title == "Work plan"
    assert [message.provider_message_id for message in conversation.messages] == ["u1", "a1"]
    assert conversation.provider_meta is not None
    assert conversation.provider_meta["browser_capture"] is True
    assert conversation.provider_meta["model"] == "gpt-5.4"
    assert len(conversation.attachments) == 1
    assert conversation.attachments[0].message_provider_id == "a1"
    assert conversation.attachments[0].provider_meta == {"url": "https://chatgpt.com/attachment/1"}


def test_browser_capture_supports_claude_ai_provider() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-session"

    assert detect_provider(payload) is Provider.CLAUDE_AI
    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")
    assert parsed[0].provider_name is Provider.CLAUDE_AI
    assert parsed[0].provider_conversation_id == "claude-session"


@pytest.mark.asyncio
async def test_browser_capture_receiver_artifact_lands_in_archive(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_capture_payload())
    artifact = write_capture_envelope(envelope, spool_path=tmp_path / "browser-capture").path
    config = get_config()
    config.sources = [Source(name="inbox", path=artifact)]

    await run_sources(config=config, stage="acquire")
    await run_sources(config=config, stage="parse")

    with open_connection(None) as conn:
        row = conn.execute(
            "SELECT provider_name, provider_conversation_id, provider_meta FROM conversations"
        ).fetchone()

    assert row is not None
    assert row["provider_name"] == "chatgpt"
    assert row["provider_conversation_id"] == "conv-123"
    assert "browser_capture" in row["provider_meta"]
