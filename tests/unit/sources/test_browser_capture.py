from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import write_capture_envelope
from polylogue.config import Source, get_config
from polylogue.core.enums import Provider, TitleSource
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.browser_capture import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
    TEMPORARY_CHAT_INGEST_FLAG,
)
from polylogue.sources.parsers.browser_capture import (
    parse as parse_browser_capture,
)
from polylogue.storage.blob_store import BlobStore
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
    assert session.title_source is TitleSource.ORIGIN


def test_browser_capture_page_title_fallback_is_not_provider_title() -> None:
    payload = json.loads(json.dumps(_capture_payload()))
    payload["session"].pop("title")

    session = parse_browser_capture(payload, "fallback")

    assert session.title == "ChatGPT - Work plan"
    assert session.title_source is None
    assert session.updated_at == "2026-04-24T00:00:01+00:00"
    assert [message.provider_message_id for message in session.messages] == ["u1", "a1"]
    assert len(session.attachments) == 1
    assert session.attachments[0].message_provider_id == "a1"
    assert session.attachments[0].source_url == "https://chatgpt.com/attachment/1"
    assert DOM_FALLBACK_INGEST_FLAG in session.ingest_flags
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG not in session.ingest_flags


def test_browser_capture_does_not_launder_capture_time_as_provider_update() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    del session_payload["updated_at"]

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert parsed[0].updated_at is None


def test_browser_capture_embedded_attachment_payloads_become_inline_bytes() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    turns = session_payload["turns"]
    assert isinstance(turns, list)
    assistant_turn = turns[1]
    assert isinstance(assistant_turn, dict)
    assistant_turn["attachments"] = [
        {
            "provider_attachment_id": "att-text",
            "name": "notes.md",
            "mime_type": "text/markdown",
            "extracted_content": "hello notes",
        },
        {
            "provider_attachment_id": "att-b64",
            "name": "payload.bin",
            "mime_type": "application/octet-stream",
            "provider_meta": {"base64_data": "AAECAw=="},
        },
        {
            "provider_attachment_id": "att-remote",
            "name": "remote.pdf",
            "mime_type": "application/pdf",
            "url": "https://chatgpt.com/attachment/remote",
        },
    ]

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    by_name = {attachment.name: attachment for attachment in parsed[0].attachments}
    assert by_name["notes.md"].inline_bytes == b"hello notes"
    assert by_name["notes.md"].size_bytes == len(b"hello notes")
    assert by_name["notes.md"].upload_origin == "paste"
    assert by_name["payload.bin"].inline_bytes == b"\x00\x01\x02\x03"
    assert by_name["payload.bin"].size_bytes == 4
    assert by_name["payload.bin"].upload_origin == "paste"
    assert by_name["remote.pdf"].inline_bytes is None
    assert by_name["remote.pdf"].source_url == "https://chatgpt.com/attachment/remote"
    assert by_name["remote.pdf"].upload_origin == "url"


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
    # bd polylogue-4fm3: code-interpreter calls are TOOL_USE (paired with
    # their execution_output via tool_id), not CODE.
    assert session.messages[1].blocks[0].type.value == "tool_use"
    assert DOM_FALLBACK_INGEST_FLAG not in session.ingest_flags
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG in session.ingest_flags


def test_browser_capture_compact_chatgpt_projection_uses_envelope_turns() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    session_payload["provider_meta"] = {"capture_fidelity": "native_compact"}
    payload["provider_meta"] = {"capture_fidelity": "native_compact"}
    payload["raw_provider_payload"] = {
        "polylogue_bridge_projection": "chatgpt-native-compact-v1",
        "mapping": {
            "untrusted-native-shape": {
                "id": "untrusted-native-shape",
                "message": {
                    "id": "wrong-native-message",
                    "author": {"role": "assistant"},
                    "content": {"parts": ["must not override envelope turns"]},
                },
            }
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")[0]

    assert [message.provider_message_id for message in parsed.messages] == ["u1", "a1"]
    assert [message.text for message in parsed.messages] == ["Draft the plan", "Here is the plan"]
    assert COMPACT_BROWSER_CAPTURE_INGEST_FLAG in parsed.ingest_flags
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG not in parsed.ingest_flags
    assert DOM_FALLBACK_INGEST_FLAG not in parsed.ingest_flags


def test_browser_capture_raw_chatgpt_payload_matches_direct_import_identity() -> None:
    raw_payload = {
        "conversation_id": "native-conv",
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
                    "content": {"content_type": "text", "parts": ["Native answer text"]},
                    "metadata": {},
                },
            },
        },
    }
    payload = _capture_payload()
    payload["raw_provider_payload"] = raw_payload

    direct_session = parse_payload(Provider.CHATGPT, raw_payload, "direct-fallback")[0]
    captured_session = parse_payload(Provider.CHATGPT, payload, "capture-fallback")[0]

    assert captured_session.provider_session_id == direct_session.provider_session_id == "native-conv"
    assert captured_session.title == direct_session.title == "Native ChatGPT title"
    assert (
        [message.provider_message_id for message in captured_session.messages]
        == [message.provider_message_id for message in direct_session.messages]
        == ["native-u1", "native-a1"]
    )


def test_browser_capture_preserves_live_and_native_generation_measurements() -> None:
    """Browser parsing keeps three duration meanings; collapsing them or naming compute fails."""
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    session_payload["provider_meta"] = {
        "generation_observations": [
            {
                "observation_id": "conv-123:a1:started:1",
                "state": "started",
                "observed_at": "2026-07-16T00:00:00Z",
                "evidence_source": "dom_control",
                "fidelity": "observed",
                "duration_semantics": "dom_observed_wall",
                "turn_provider_id": "native-a1",
                "wall_elapsed_ms": 0,
                "trigger": "initial_scan",
            },
            {
                "observation_id": "conv-123:a1:completed:worked-for",
                "state": "completed",
                "observed_at": "2026-07-16T01:26:30Z",
                "evidence_source": "dom_duration_control",
                "fidelity": "observed",
                "duration_semantics": "provider_ui_elapsed",
                "turn_provider_id": "native-a1",
                "displayed_elapsed_ms": 5_190_000,
                "raw_label": "Worked for 86m 30s",
                "trigger": "dom_mutation",
            },
        ]
    }
    payload["raw_provider_payload"] = {
        "conversation_id": "conv-123",
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
                    "content": {"content_type": "text", "parts": ["Do the work"]},
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
                    "content": {"content_type": "reasoning_recap", "parts": ["Done"]},
                    "metadata": {
                        "reasoning_start_time": 1784164541.690012,
                        "reasoning_end_time": 1784169732.588194,
                        "finished_duration_sec": 5190,
                    },
                },
            },
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")[0]
    lifecycle = [event for event in parsed.session_events if event.event_type == "generation_lifecycle"]

    assert [event.payload["evidence_source"] for event in lifecycle] == [
        "provider_native",
        "dom_control",
        "dom_duration_control",
    ]
    assert [event.payload["duration_semantics"] for event in lifecycle] == [
        "provider_reported_elapsed",
        "dom_observed_wall",
        "provider_ui_elapsed",
    ]
    assert all(event.payload["duration_semantics"] != "model_compute" for event in lifecycle)
    assert lifecycle[0].payload["elapsed_duration_ms"] == 5_190_000
    assert lifecycle[1].payload["wall_elapsed_ms"] == 0
    assert lifecycle[2].payload["displayed_elapsed_ms"] == 5_190_000
    assert parsed.reported_duration_ms == 5_190_000


def test_browser_capture_rejects_malformed_or_duplicate_generation_observations() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    valid = {
        "observation_id": "conv-123:a1:started:1",
        "state": "started",
        "observed_at": "2026-07-16T00:00:00Z",
        "evidence_source": "dom_control",
        "fidelity": "observed",
        "duration_semantics": "dom_observed_wall",
        "wall_elapsed_ms": 0,
    }
    session_payload["provider_meta"] = {
        "generation_observations": [
            valid,
            valid,
            {**valid, "observation_id": "negative", "wall_elapsed_ms": -1},
            {**valid, "observation_id": "bad-time", "observed_at": "not-a-timestamp"},
            {**valid, "observation_id": "bad-state", "state": "thinking-ish"},
        ]
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")[0]
    lifecycle = [event for event in parsed.session_events if event.event_type == "generation_lifecycle"]

    assert len(lifecycle) == 1
    assert lifecycle[0].payload["observation_id"] == valid["observation_id"]


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


def test_browser_capture_non_native_raw_payload_keeps_envelope_turns() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    session_payload["provider_session_id"] = "temporary:abc123"
    payload["raw_provider_payload"] = {
        "dom_transcript": {
            "turn_count": 2,
            "note": "preservation metadata, not a ChatGPT native mapping payload",
        }
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "file-fallback")

    assert len(parsed) == 1
    session = parsed[0]
    assert session.provider_session_id == "temporary:abc123"
    assert [message.provider_message_id for message in session.messages] == ["u1", "a1"]
    assert [message.text for message in session.messages] == ["Draft the plan", "Here is the plan"]
    assert session.ingest_flags == [TEMPORARY_CHAT_INGEST_FLAG, DOM_FALLBACK_INGEST_FLAG]


def test_browser_capture_session_kind_marks_temporary_chat() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    session_payload["provider_meta"] = {"session_kind": "temporary"}

    parsed = parse_payload(Provider.CHATGPT, payload, "file-fallback")

    assert parsed[0].provider_session_id == "conv-123"
    assert parsed[0].session_kind == "temporary"
    assert parsed[0].ingest_flags == [TEMPORARY_CHAT_INGEST_FLAG, DOM_FALLBACK_INGEST_FLAG]


def test_browser_capture_typed_session_kind_marks_temporary_chat() -> None:
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    session_payload["session_kind"] = "temporary"

    envelope = BrowserCaptureEnvelope.model_validate(payload)
    parsed = parse_payload(Provider.CHATGPT, payload, "file-fallback")

    assert envelope.session.session_kind == "temporary"
    assert parsed[0].provider_session_id == "conv-123"
    assert parsed[0].session_kind == "temporary"
    assert parsed[0].ingest_flags == [TEMPORARY_CHAT_INGEST_FLAG, DOM_FALLBACK_INGEST_FLAG]


def test_browser_capture_session_kind_defaults_to_standard() -> None:
    envelope = BrowserCaptureEnvelope.model_validate(_capture_payload())

    assert envelope.session.session_kind == "standard"


def test_browser_capture_raw_chatgpt_normalizes_legacy_synthetic_fallback_id() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider_session_id"] = "chatgpt:2fcf01f9-beb1-474f-bd1c-84e1bec712ca:9f658806"
    payload["raw_provider_payload"] = {
        "title": "Native ChatGPT title",
        "current_node": "assistant-node",
        "mapping": {
            "assistant-node": {
                "id": "assistant-node",
                "parent": None,
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
    assert parsed[0].provider_session_id == "2fcf01f9-beb1-474f-bd1c-84e1bec712ca"


def test_browser_capture_prefers_raw_claude_ai_payload_when_present() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-conv-123"
    payload["raw_provider_payload"] = {
        "uuid": "claude-native-conv",
        "name": "Native Claude title",
        "created_at": "2026-04-24T00:00:00+00:00",
        "updated_at": "2026-04-24T00:00:01+00:00",
        "chat_messages": [
            {
                "uuid": "claude-u1",
                "sender": "human",
                "text": "Native Claude user",
                "created_at": "2026-04-24T00:00:00+00:00",
            },
            {
                "uuid": "claude-a1",
                "sender": "assistant",
                "text": "Native Claude answer",
                "created_at": "2026-04-24T00:00:01+00:00",
                "model": "claude-native",
            },
        ],
    }

    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")

    assert len(parsed) == 1
    parsed_session = parsed[0]
    assert parsed_session.source_name is Provider.CLAUDE_AI
    assert parsed_session.provider_session_id == "claude-native-conv"
    assert parsed_session.title == "Native Claude title"
    assert [message.provider_message_id for message in parsed_session.messages] == ["claude-u1", "claude-a1"]
    assert [message.text for message in parsed_session.messages] == ["Native Claude user", "Native Claude answer"]
    assert parsed_session.messages[1].model_name == "claude-native"


def test_browser_capture_raw_claude_ai_uses_content_when_text_empty() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-conv-123"
    payload["raw_provider_payload"] = {
        "uuid": "claude-native-conv",
        "name": "Native Claude title",
        "created_at": "2026-04-24T00:00:00+00:00",
        "updated_at": "2026-04-24T00:00:01+00:00",
        "chat_messages": [
            {
                "uuid": "claude-u1",
                "sender": "human",
                "text": "",
                "content": [{"type": "text", "text": "Native Claude user from content"}],
                "created_at": "2026-04-24T00:00:00+00:00",
            },
            {
                "uuid": "claude-a1",
                "sender": "assistant",
                "text": "",
                "content": [{"type": "text", "text": "Native Claude answer from content"}],
                "created_at": "2026-04-24T00:00:01+00:00",
            },
        ],
    }

    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")

    assert len(parsed) == 1
    parsed_session = parsed[0]
    assert [message.provider_message_id for message in parsed_session.messages] == ["claude-u1", "claude-a1"]
    assert [message.text for message in parsed_session.messages] == [
        "Native Claude user from content",
        "Native Claude answer from content",
    ]


def test_browser_capture_raw_claude_ai_attachment_content_stays_acquirable() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-conv-123"
    payload["raw_provider_payload"] = {
        "uuid": "claude-native-conv",
        "name": "Native Claude title",
        "chat_messages": [
            {
                "uuid": "claude-u1",
                "sender": "human",
                "text": "Native Claude user",
                "attachments": [
                    {
                        "file_name": "embedded.md",
                        "file_type": "text/markdown",
                        "extracted_content": "embedded notes",
                    }
                ],
                "files": [
                    {
                        "file_uuid": "remote-file-1",
                        "file_name": "remote.tar.gz",
                        "size_bytes": 1024,
                    }
                ],
            },
        ],
    }

    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")

    by_name = {attachment.name: attachment for attachment in parsed[0].attachments}
    assert by_name["embedded.md"].inline_bytes == b"embedded notes"
    assert by_name["embedded.md"].size_bytes == len(b"embedded notes")
    assert by_name["remote.tar.gz"].inline_bytes is None


def test_browser_capture_raw_claude_ai_without_id_uses_envelope_session_id() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-conv-123"
    payload["raw_provider_payload"] = {
        "title": "Loose Claude title",
        "chat_messages": [
            {"id": "claude-u1", "role": "user", "text": "Loose Claude user"},
            {"id": "claude-a1", "role": "assistant", "text": "Loose Claude answer"},
        ],
    }

    parsed = parse_payload(Provider.CLAUDE_AI, payload, "file-fallback")

    assert len(parsed) == 1
    parsed_session = parsed[0]
    assert parsed_session.source_name is Provider.CLAUDE_AI
    assert parsed_session.provider_session_id == "claude-conv-123"
    assert parsed_session.title == "Loose Claude title"
    assert [message.provider_message_id for message in parsed_session.messages] == ["claude-u1", "claude-a1"]


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


@pytest.mark.parametrize(
    ("provider", "legacy_id", "expected_id"),
    [
        ("chatgpt", "chatgpt:2fcf01f9-beb1-474f-bd1c-84e1bec712ca:9f658806", "2fcf01f9-beb1-474f-bd1c-84e1bec712ca"),
        (
            "claude-ai",
            "claude-ai:6a590003-3e69-4eb7-aed3-fbb75fb800c0:ce1a9248",
            "6a590003-3e69-4eb7-aed3-fbb75fb800c0",
        ),
        (
            "chatgpt",
            "chatgpt:WEB:b04a9756-2c83-40b6-97d2-afd5e283f7fe:bdb32387",
            "WEB:b04a9756-2c83-40b6-97d2-afd5e283f7fe",
        ),
        (
            "chatgpt",
            "chatgpt-2fcf01f9-beb1-474f-bd1c-84e1bec712ca-20105879-0",
            "2fcf01f9-beb1-474f-bd1c-84e1bec712ca",
        ),
        (
            "chatgpt",
            "chatgpt-WEB-b04a9756-2c83-40b6-97d2-afd5e283f7fe-bdb32387-0",
            "WEB:b04a9756-2c83-40b6-97d2-afd5e283f7fe",
        ),
    ],
)
def test_browser_capture_normalizes_legacy_synthetic_dom_session_ids(
    provider: str,
    legacy_id: str,
    expected_id: str,
) -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider"] = provider
    session["provider_session_id"] = legacy_id

    parsed = parse_payload(Provider.CHATGPT if provider == "chatgpt" else Provider.CLAUDE_AI, payload, "fallback")

    assert parsed[0].provider_session_id == expected_id


def test_browser_capture_does_not_normalize_legacy_root_route_synthetic_id() -> None:
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider_session_id"] = "chatgpt:/:e4af3c6b"

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert parsed[0].provider_session_id == "chatgpt:/:e4af3c6b"


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
async def test_browser_capture_embedded_attachments_are_acquired_in_archive(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    del workspace_env
    payload = _capture_payload()
    session_payload = payload["session"]
    assert isinstance(session_payload, dict)
    turns = session_payload["turns"]
    assert isinstance(turns, list)
    assistant_turn = turns[1]
    assert isinstance(assistant_turn, dict)
    assistant_turn["attachments"] = [
        {
            "provider_attachment_id": "att-embedded",
            "name": "embedded.md",
            "mime_type": "text/markdown",
            "extracted_content": "archive notes",
        },
        {
            "provider_attachment_id": "att-remote",
            "name": "remote.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 4096,
            "url": "https://chatgpt.com/attachment/remote",
        },
    ]
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    artifact = write_capture_envelope(envelope, spool_path=tmp_path / "browser-capture").path
    config = get_config()
    blob_store = BlobStore(config.archive_root / "blob")
    config.sources = [Source(name="browser-capture", path=artifact)]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(config.sources)

    with open_index_db(config.archive_root / "index.db") as conn:
        rows = conn.execute(
            """
            SELECT display_name, blob_hash, byte_count, acquisition_status
            FROM attachments
            ORDER BY display_name
            """
        ).fetchall()

    by_name = {str(row["display_name"]): row for row in rows}
    acquired = by_name["embedded.md"]
    expected_hash = hashlib.sha256(b"archive notes").digest()
    assert acquired["acquisition_status"] == "acquired"
    assert acquired["byte_count"] == len(b"archive notes")
    assert bytes(acquired["blob_hash"]) == expected_hash
    assert blob_store.read_all(expected_hash.hex()) == b"archive notes"

    remote = by_name["remote.pdf"]
    assert remote["acquisition_status"] == "unfetched"
    assert remote["blob_hash"] is None
    assert remote["byte_count"] == 4096


@pytest.mark.parametrize("source_order", ["export-first", "browser-first"])
@pytest.mark.asyncio
async def test_browser_capture_raw_payload_coalesces_with_chatgpt_export(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    source_order: str,
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
    export_first_sources = [
        Source(name="chatgpt", path=gdpr_export),
        Source(name="browser-capture", path=artifact),
    ]
    sources = export_first_sources if source_order == "export-first" else list(reversed(export_first_sources))

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
    assert rows[0]["title"] == "GDPR title"
    assert rows[0]["message_count"] == 2


@pytest.mark.asyncio
async def test_browser_capture_raw_payload_coalesces_with_claude_ai_export(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    del workspace_env
    gdpr_export = tmp_path / "claude-export.json"
    gdpr_export.write_text(
        json.dumps(
            {
                "uuid": "claude-conv-123",
                "name": "Claude GDPR title",
                "created_at": "2026-04-24T00:00:00+00:00",
                "updated_at": "2026-04-24T00:00:01+00:00",
                "chat_messages": [
                    {
                        "uuid": "gdpr-u1",
                        "sender": "human",
                        "text": "Claude GDPR user",
                        "created_at": "2026-04-24T00:00:00+00:00",
                    },
                    {
                        "uuid": "gdpr-a1",
                        "sender": "assistant",
                        "text": "Claude GDPR answer",
                        "created_at": "2026-04-24T00:00:01+00:00",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    capture_payload = _capture_payload()
    session = capture_payload["session"]
    assert isinstance(session, dict)
    session["provider"] = "claude-ai"
    session["provider_session_id"] = "claude-conv-123"
    capture_payload["raw_provider_payload"] = {
        "name": "Claude browser title",
        "created_at": "2026-04-24T00:00:00+00:00",
        "updated_at": "2026-04-24T00:00:02+00:00",
        "chat_messages": [
            {
                "uuid": "browser-u1",
                "sender": "human",
                "text": "Claude browser user",
                "created_at": "2026-04-24T00:00:00+00:00",
            },
            {
                "uuid": "browser-a1",
                "sender": "assistant",
                "text": "Claude browser answer",
                "created_at": "2026-04-24T00:00:02+00:00",
            },
        ],
    }
    artifact = write_capture_envelope(
        BrowserCaptureEnvelope.model_validate(capture_payload),
        spool_path=tmp_path / "browser-capture",
    ).path
    config = get_config()
    sources = [
        Source(name="claude-ai", path=gdpr_export),
        Source(name="browser-capture", path=artifact),
    ]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(sources)

    with open_index_db(config.archive_root / "index.db") as conn:
        rows = conn.execute(
            """
            SELECT session_id, origin, native_id, title, message_count
            FROM sessions
            WHERE origin = 'claude-ai-export'
            ORDER BY native_id
            """
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["session_id"] == "claude-ai-export:claude-conv-123"
    assert rows[0]["native_id"] == "claude-conv-123"
    assert rows[0]["title"] == "Claude GDPR title"
    assert rows[0]["message_count"] == 2


def test_native_payload_delegation_keeps_envelope_acquired_assets() -> None:
    """Extension-acquired bytes must survive the native-payload parse path.

    The session structure comes from raw_provider_payload, but the envelope
    carries attachments the extension fetched through the authenticated page
    (sandbox deliverables). Regression: these were silently dropped.
    """

    import base64 as _b64

    payload = _capture_payload()
    payload["session"]["attachments"] = [  # type: ignore[index]
        {
            "provider_attachment_id": "sandbox:native-a1:/mnt/data/kit.zip",
            "message_provider_id": "native-a1",
            "name": "kit.zip",
            "mime_type": "application/zip",
            "inline_base64": _b64.b64encode(b"PK\x03\x04demo-bytes").decode("ascii"),
            "provider_meta": {"capture_source": "chatgpt_page_asset_fetch", "asset_kind": "sandbox"},
        }
    ]
    payload["raw_provider_payload"] = {
        "id": "native-conv",
        "title": "Native ChatGPT title",
        "mapping": {
            "assistant-node": {
                "id": "assistant-node",
                "parent": None,
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "create_time": 1781442880.0,
                    "content": {
                        "content_type": "text",
                        "parts": ["Kit ready: [zip](sandbox:/mnt/data/kit.zip)"],
                    },
                    "metadata": {},
                },
            },
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    by_id = {attachment.provider_attachment_id: attachment for attachment in session.attachments}
    acquired = by_id["sandbox:native-a1:/mnt/data/kit.zip"]
    assert acquired.inline_bytes == b"PK\x03\x04demo-bytes"
    assert acquired.name == "kit.zip"
    # The parser-derived sandbox row and the envelope row share one id: the
    # envelope's byte-carrying version wins while keeping the parser's kind.
    assert acquired.attachment_kind == "sandbox_file"
    assert len([a for a in session.attachments if a.provider_attachment_id.startswith("sandbox:")]) == 1


def test_native_payload_delegation_without_envelope_attachments_is_unchanged() -> None:
    payload = _capture_payload()
    payload["session"]["attachments"] = []  # type: ignore[index]
    for turn in payload["session"]["turns"]:  # type: ignore[index]
        turn.pop("attachments", None)
    payload["raw_provider_payload"] = {
        "id": "native-conv",
        "mapping": {
            "assistant-node": {
                "id": "assistant-node",
                "parent": None,
                "children": [],
                "message": {
                    "id": "native-a1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["plain answer"]},
                    "metadata": {},
                },
            },
        },
    }

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(parsed) == 1
    assert parsed[0].attachments == []


# --- polylogue-ah21: BrowserCaptureTurn typed blocks channel -----------------
#
# These tests drive the real capture -> archive route
# (BrowserCaptureEnvelope.model_validate -> parse_payload/parse ->
# ParsedSession/ParsedMessage -> Polylogue.parse_sources -> index.db) on
# synthesized wire payloads shaped like what browser-extension/src/backfill/
# providers.js's ChatGptBackfillAdapter.normalizeCapture now emits for a
# code-interpreter turn pair: a tool_use turn whose own provider_turn_id is
# the tool_id, and a tool_result turn whose parent_turn_id (the call node's
# id in the ChatGPT mapping tree) is echoed back as its tool_id. Before this
# change BrowserCaptureTurn had no blocks field at all, so this structure had
# nowhere to go except turn.text prose -- every one of these assertions fails
# if BrowserCaptureTurn.blocks or the parser's projection of it is removed.


def test_browser_capture_turn_accepts_blocks_only_content() -> None:
    """A tool turn with only structured blocks and no prose text is valid.

    Regression guard for require_content: before adding `blocks`, a turn with
    no `text` and no `attachments` always raised -- there was no other
    channel for structure to arrive through.
    """
    from polylogue.browser_capture.models import BrowserCaptureTurn

    turn = BrowserCaptureTurn.model_validate(
        {
            "provider_turn_id": "call-1",
            "role": "tool",
            "blocks": [{"type": "tool_use", "tool_name": "python", "tool_id": "call-1", "tool_input": {"code": "1+1"}}],
        }
    )
    assert turn.text is None
    assert turn.blocks[0].type.value == "tool_use"


def test_browser_capture_turn_rejects_truly_empty_content() -> None:
    from pydantic import ValidationError

    from polylogue.browser_capture.models import BrowserCaptureTurn

    with pytest.raises(ValidationError):
        BrowserCaptureTurn.model_validate({"provider_turn_id": "empty-1", "role": "assistant"})


def _chatgpt_code_interpreter_pair_payload(*, compact: bool) -> dict[str, object]:
    """A ChatGPT capture whose two tool turns arrive with typed blocks.

    Mirrors the compact/backfill wire shape: the capture never carries a
    trusted native `raw_provider_payload` mapping (either because it is the
    size-bounded compact projection, or because there simply is none), so
    the parser must build session structure from `session.turns` alone -- the
    exact path that used to flatten tool call/result turns into text.
    """
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["provider_meta"] = {"capture_fidelity": "native_compact"} if compact else {}
    session["turns"] = [
        {"provider_turn_id": "u1", "role": "user", "text": "Compute 6 * 7", "ordinal": 0},
        {
            "provider_turn_id": "call-1",
            "role": "tool",
            "ordinal": 1,
            "parent_turn_id": "u1",
            "blocks": [
                {
                    "type": "tool_use",
                    "tool_name": "python",
                    "tool_id": "call-1",
                    "tool_input": {"code": "6 * 7"},
                }
            ],
        },
        {
            "provider_turn_id": "result-1",
            "role": "tool",
            "ordinal": 2,
            "parent_turn_id": "call-1",
            "blocks": [
                {
                    "type": "tool_result",
                    "tool_id": "call-1",
                    "text": "42",
                }
            ],
        },
        {
            "provider_turn_id": "a1",
            "role": "assistant",
            "text": "6 * 7 = 42",
            "ordinal": 3,
            "parent_turn_id": "result-1",
        },
    ]
    if compact:
        payload["provider_meta"] = {"capture_fidelity": "native_compact"}
        payload["raw_provider_payload"] = {
            "polylogue_bridge_projection": "chatgpt-native-compact-v1",
            "mapping": {"untrusted": {"id": "untrusted", "message": None}},
        }
    return payload


@pytest.mark.parametrize("compact", [False, True])
def test_browser_capture_tool_turn_blocks_land_as_typed_tool_use_and_tool_result(compact: bool) -> None:
    payload = _chatgpt_code_interpreter_pair_payload(compact=compact)

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    by_id = {message.provider_message_id: message for message in session.messages}
    call_message = by_id["call-1"]
    result_message = by_id["result-1"]

    assert len(call_message.blocks) == 1
    assert call_message.blocks[0].type.value == "tool_use"
    assert call_message.blocks[0].tool_name == "python"
    assert call_message.blocks[0].tool_id == "call-1"
    assert call_message.blocks[0].tool_input == {"code": "6 * 7"}

    assert len(result_message.blocks) == 1
    assert result_message.blocks[0].type.value == "tool_result"
    assert result_message.blocks[0].tool_id == "call-1"
    assert result_message.blocks[0].text == "42"

    # The regression signal named in polylogue-ah21: tool_use and tool_result
    # counts (and pairing by tool_id) must be 1:1, not reconstructed prose.
    tool_use_ids = [b.tool_id for m in session.messages for b in m.blocks if b.type.value == "tool_use"]
    tool_result_ids = [b.tool_id for m in session.messages for b in m.blocks if b.type.value == "tool_result"]
    assert tool_use_ids == ["call-1"]
    assert tool_result_ids == ["call-1"]

    # parent_turn_id survives into the archive's DAG-linking field for every
    # turn on this path, not just the plain user/assistant ones.
    assert call_message.parent_message_provider_id == "u1"
    assert result_message.parent_message_provider_id == "call-1"
    assert by_id["a1"].parent_message_provider_id == "result-1"

    if compact:
        assert COMPACT_BROWSER_CAPTURE_INGEST_FLAG in session.ingest_flags
        assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG not in session.ingest_flags
    else:
        assert DOM_FALLBACK_INGEST_FLAG in session.ingest_flags


@pytest.mark.asyncio
async def test_browser_capture_tool_turn_blocks_land_in_archive_with_consistent_ratio(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Full receiver -> parser -> materialize -> index.db round trip.

    Proves the fix on the real production write path (write_capture_envelope
    + Polylogue.parse_sources), not just the in-memory parser: the archive's
    blocks table must show one tool_use row and one tool_result row sharing a
    tool_id, matching the 1:1 pairing produced above.
    """
    payload = _chatgpt_code_interpreter_pair_payload(compact=True)
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    artifact = write_capture_envelope(envelope, spool_path=tmp_path / "browser-capture").path
    config = get_config()
    config.sources = [Source(name="inbox", path=artifact)]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(config.sources)

    with open_index_db(config.archive_root / "index.db") as conn:
        rows = conn.execute(
            "SELECT block_type, tool_id, tool_name, text FROM blocks "
            "WHERE block_type IN ('tool_use', 'tool_result') ORDER BY position"
        ).fetchall()

    assert [dict(row) for row in rows] == [
        {"block_type": "tool_use", "tool_id": "call-1", "tool_name": "python", "text": None},
        {"block_type": "tool_result", "tool_id": "call-1", "tool_name": None, "text": "42"},
    ]


def test_browser_capture_block_metadata_routes_to_session_events() -> None:
    """``BrowserCaptureBlock.metadata`` must reach ``session_events``, not just the block.

    Exercises the generic (no-native-payload) turn loop's
    ``_block_metadata_evidence_events`` wiring. The ``blocks`` table has no
    metadata column and the write path only reads a ``language`` key back out
    of it (bd polylogue-9x22), so whatever the capture extension attaches to
    a block's ``metadata`` (here: an opaque ``dom_selector`` the extension
    used to locate the tool call in the page) would otherwise be silently
    dropped at write time. Deleting the
    ``block_metadata_events.extend(_block_metadata_evidence_events(...))``
    call in ``parse()`` makes this fail.
    """
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["turns"] = [
        {"provider_turn_id": "u1", "role": "user", "text": "run a search", "ordinal": 0},
        {
            "provider_turn_id": "a1",
            "role": "assistant",
            "text": "calling search",
            "ordinal": 1,
            "blocks": [
                {
                    "type": "tool_use",
                    "tool_name": "web_search",
                    "tool_id": "call-1",
                    "tool_input": {"query": "polylogue"},
                    "metadata": {"dom_selector": "#tool-call-1"},
                },
                {"type": "text", "text": "no metadata here"},
            ],
        },
    ]

    parsed = parse_payload(Provider.CHATGPT, payload, "fallback")

    assert len(parsed) == 1
    events = [event for event in parsed[0].session_events if event.event_type == "browser_capture_block_metadata"]
    assert len(events) == 1
    event = events[0]
    assert event.source_message_provider_id == "a1"
    assert event.payload == {"block_index": 0, "dom_selector": "#tool-call-1"}


@pytest.mark.asyncio
async def test_browser_capture_block_metadata_lands_in_archive_session_events(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Full receiver -> parser -> materialize -> index.db round trip.

    Proves the fix on the real production write path
    (write_capture_envelope + Polylogue.parse_sources), not just the
    in-memory parser: the archive's session_events table must carry the
    block's metadata dict even though the blocks table has no metadata
    column to hold it.
    """
    payload = _capture_payload()
    session = payload["session"]
    assert isinstance(session, dict)
    session["turns"] = [
        {"provider_turn_id": "u1", "role": "user", "text": "run a search", "ordinal": 0},
        {
            "provider_turn_id": "a1",
            "role": "assistant",
            "text": "calling search",
            "ordinal": 1,
            "blocks": [
                {
                    "type": "tool_use",
                    "tool_name": "web_search",
                    "tool_id": "call-1",
                    "tool_input": {"query": "polylogue"},
                    "metadata": {"dom_selector": "#tool-call-1"},
                }
            ],
        },
    ]
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    artifact = write_capture_envelope(envelope, spool_path=tmp_path / "browser-capture").path
    config = get_config()
    config.sources = [Source(name="inbox", path=artifact)]

    async with Polylogue(archive_root=config.archive_root, db_path=config.db_path) as polylogue:
        await polylogue.parse_sources(config.sources)

    with open_index_db(config.archive_root / "index.db") as conn:
        rows = conn.execute(
            "SELECT event_type, source_message_provider_id, payload_json FROM session_events "
            "WHERE event_type = 'browser_capture_block_metadata'"
        ).fetchall()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["source_message_provider_id"] == "a1"
    assert json.loads(row["payload_json"]) == {"block_index": 0, "dom_selector": "#tool-call-1"}


def test_browser_capture_claude_fallback_turn_blocks_produce_typed_tool_blocks() -> None:
    """Same structural gap on the Claude fallback path (no native chat_messages payload)."""
    payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://claude.ai/chat/conv-99",
            "captured_at": "2026-07-29T00:00:00+00:00",
            "adapter_name": "claude-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "claude-ai",
            "provider_session_id": "conv-99",
            "title": "Tool demo",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Run ls", "ordinal": 0},
                {
                    "provider_turn_id": "a1",
                    "role": "assistant",
                    "ordinal": 1,
                    "parent_turn_id": "u1",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "tool_name": "bash",
                            "tool_id": "toolu_1",
                            "tool_input": {"command": "ls"},
                        }
                    ],
                },
                {
                    "provider_turn_id": "t1",
                    "role": "tool",
                    "ordinal": 2,
                    "parent_turn_id": "a1",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "tool_id": "toolu_1",
                            "text": "file.txt",
                            "is_error": False,
                        }
                    ],
                },
            ],
        },
    }

    parsed = parse_payload(Provider.CLAUDE_AI, payload, "fallback")

    assert len(parsed) == 1
    session = parsed[0]
    by_id = {message.provider_message_id: message for message in session.messages}
    assistant_message = by_id["a1"]
    tool_message = by_id["t1"]

    assert [block.type.value for block in assistant_message.blocks] == ["tool_use"]
    assert assistant_message.blocks[0].tool_id == "toolu_1"
    assert assistant_message.blocks[0].tool_name == "bash"

    assert [block.type.value for block in tool_message.blocks] == ["tool_result"]
    assert tool_message.blocks[0].tool_id == "toolu_1"
    assert tool_message.blocks[0].is_error is False
    assert tool_message.parent_message_provider_id == "a1"
