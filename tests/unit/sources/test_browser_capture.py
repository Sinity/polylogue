from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import write_capture_envelope
from polylogue.config import Source, get_config
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import detect_provider, parse_payload
from tests.infra.archive_scenarios import open_index_db


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


def test_browser_capture_detects_list_wrapped_inner_provider() -> None:
    assert detect_provider([_capture_payload()]) is Provider.CHATGPT


def test_browser_capture_parses_session_metadata_and_deduplicates_turns() -> None:
    parsed = parse_payload(Provider.CHATGPT, _capture_payload(), "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    assert session.source_name is Provider.CHATGPT
    assert session.provider_session_id == "conv-123"
    assert session.title == "Work plan"
    assert [message.provider_message_id for message in session.messages] == ["u1", "a1"]
    assert len(session.attachments) == 1
    assert session.attachments[0].message_provider_id == "a1"
    assert session.attachments[0].source_url == "https://chatgpt.com/attachment/1"


def test_browser_capture_prefers_raw_chatgpt_payload_when_present() -> None:
    payload = _capture_payload()
    payload["raw_provider_payload"] = {
        "id": "native-conv",
        "title": "Native ChatGPT title",
        "create_time": 1781442866.0,
        "update_time": 1781442966.0,
        "current_node": "assistant-node",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["user-node"]},
            "user-node": {
                "id": "user-node",
                "parent": "root",
                "children": ["assistant-node"],
                "message": {
                    "id": "native-u1",
                    "author": {"role": "user"},
                    "create_time": 1781442870.0,
                    "content": {"content_type": "text", "parts": ["Native user text"]},
                    "metadata": {},
                },
            },
            "assistant-node": {
                "id": "assistant-node",
                "parent": "user-node",
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "create_time": 1781442880.0,
                    "content": {"content_type": "code", "text": "print('native')"},
                    "metadata": {"model_slug": "gpt-native"},
                },
            },
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    assert session.provider_session_id == "native-conv"
    assert session.title == "Native ChatGPT title"
    assert [message.provider_message_id for message in session.messages] == ["native-u1", "native-a1"]
    assert [message.text for message in session.messages] == ["Native user text", "print('native')"]
    assert session.messages[1].model_name == "gpt-native"
    assert session.messages[1].blocks[0].type.value == "code"


def test_browser_capture_raw_chatgpt_without_id_uses_envelope_session_id() -> None:
    payload = _capture_payload()
    payload["raw_provider_payload"] = {
        "title": "Native ChatGPT title",
        "current_node": "assistant-node",
        "mapping": {
            "user-node": {
                "id": "user-node",
                "parent": None,
                "children": ["assistant-node"],
                "message": {
                    "id": "native-u1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Native user text"]},
                    "metadata": {},
                },
            },
            "assistant-node": {
                "id": "assistant-node",
                "parent": "user-node",
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Native answer text"]},
                    "metadata": {},
                },
            },
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "file-fallback")

    assert len(parsed) == 1
    session = parsed[0]
    assert session.provider_session_id == "conv-123"
    assert session.title == "Native ChatGPT title"
    assert [message.provider_message_id for message in session.messages] == ["native-u1", "native-a1"]


def test_browser_capture_parses_list_wrapped_live_decoder_shape() -> None:
    parsed = parse_payload(Provider.CHATGPT, [_capture_payload()], "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    assert session.provider_session_id == "conv-123"
    assert session.title == "Work plan"
    assert [message.provider_message_id for message in session.messages] == ["u1", "a1"]


def test_browser_capture_supports_claude_ai_provider() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-session"

    assert detect_provider(payload) is Provider.CLAUDE_AI
    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")
    assert parsed[0].source_name is Provider.CLAUDE_AI
    assert parsed[0].provider_session_id == "claude-session"


@pytest.mark.asyncio
async def test_browser_capture_receiver_artifact_lands_in_archive(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_capture_payload())
    artifact = write_capture_envelope(envelope, spool_path=tmp_path / "browser-capture").path
    config = get_config()
    config.sources = [Source(name="inbox", path=artifact)]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(config.sources)

    # The archive ingest path persists the captured session into the archive
    # ``index.db`` ``sessions`` table: ``origin`` carries the source family
    # (chatgpt -> chatgpt-export) and ``native_id`` the provider session id.
    # Per #1743 there is no metadata escape hatch on ``sessions`` (the former
    # ``origin_meta`` JSON column is gone); capture provenance survives only in
    # the raw source blob, not as a queryable session column.
    with open_index_db(config.archive_root / "index.db") as conn:
        row = conn.execute("SELECT origin, native_id, title FROM sessions").fetchone()

    assert row is not None
    assert row["origin"] == "chatgpt-export"
    assert row["native_id"] == "conv-123"
    assert row["title"] == "Work plan"


@pytest.mark.asyncio
async def test_browser_capture_raw_payload_coalesces_with_chatgpt_export(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    del workspace_env
    gdpr_export = tmp_path / "chatgpt-export.json"
    gdpr_export.write_text(
        json.dumps(
            {
                "id": "conv-123",
                "title": "GDPR title",
                "current_node": "gdpr-assistant",
                "mapping": {
                    "gdpr-user": {
                        "id": "gdpr-user",
                        "parent": None,
                        "children": ["gdpr-assistant"],
                        "message": {
                            "id": "gdpr-u1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["GDPR user"]},
                            "metadata": {},
                        },
                    },
                    "gdpr-assistant": {
                        "id": "gdpr-assistant",
                        "parent": "gdpr-user",
                        "children": [],
                        "message": {
                            "id": "gdpr-a1",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["GDPR answer"]},
                            "metadata": {},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    capture_payload = _capture_payload()
    capture_payload["raw_provider_payload"] = {
        "title": "Browser title",
        "current_node": "browser-assistant",
        "mapping": {
            "browser-user": {
                "id": "browser-user",
                "parent": None,
                "children": ["browser-assistant"],
                "message": {
                    "id": "browser-u1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Browser user"]},
                    "metadata": {},
                },
            },
            "browser-assistant": {
                "id": "browser-assistant",
                "parent": "browser-user",
                "children": [],
                "message": {
                    "id": "browser-a1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Browser answer"]},
                    "metadata": {},
                },
            },
        },
    }
    artifact = write_capture_envelope(
        BrowserCaptureEnvelope.model_validate(capture_payload),
        spool_path=tmp_path / "browser-capture",
    ).path
    config = get_config()
    sources = [
        Source(name="chatgpt", path=gdpr_export),
        Source(name="browser-capture", path=artifact),
    ]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(sources)

    with open_index_db(config.archive_root / "index.db") as conn:
        rows = conn.execute(
            """
            SELECT session_id, origin, native_id, title, message_count
            FROM sessions
            WHERE origin = 'chatgpt-export'
            ORDER BY native_id
            """
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["session_id"] == "chatgpt-export:conv-123"
    assert rows[0]["native_id"] == "conv-123"
    assert rows[0]["title"] == "Browser title"
    assert rows[0]["message_count"] == 2
