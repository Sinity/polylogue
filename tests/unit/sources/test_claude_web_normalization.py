"""Real-route laws for Claude web export and browser-capture normalization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from polylogue.core.enums import Provider
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.browser_capture import (
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
)
from polylogue.sources.parsers.claude.common import CLAUDE_LINEAGE_CYCLE_INGEST_FLAG
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.pipeline_roundtrip import parse_payload_roundtrip, write_and_hydrate
from tests.infra.storage_records import db_setup


def _native_claude_payload(*, order: tuple[str, ...] | None = None) -> dict[str, Any]:
    messages: dict[str, dict[str, Any]] = {
        "u1": {
            "uuid": "u1",
            "sender": "human",
            "text": "Build a durable normalization law.",
            "created_at": "2026-07-01T10:00:00Z",
        },
        "a-old": {
            "uuid": "a-old",
            "sender": "assistant",
            "text": "An obsolete branch.",
            "parent_message_uuid": "u1",
            "created_at": "2026-07-01T10:00:01Z",
            "updated_at": "2026-07-01T10:00:01.500Z",
            "status": "completed",
            "end_turn": True,
            "duration_ms": 1000,
        },
        "a-new": {
            "uuid": "a-new",
            "sender": "assistant",
            "parent_message_uuid": "u1",
            "created_at": "2026-07-01T10:00:02Z",
            "updated_at": "2026-07-01T10:00:03Z",
            "version_uuid": "revision-a-new-2",
            "status": "completed",
            "end_turn": True,
            "model": "claude-opus-4-1",
            "model_effort": "high",
            "durationMs": 2500,
            "metadata": {"thinking_config": {"type": "enabled", "budget_tokens": 8192}},
            "content": [
                {"type": "thinking", "thinking": "Compare the two production routes."},
                {
                    "type": "tool_use",
                    "id": "tool-artifact-1",
                    "name": "artifacts",
                    "input": {
                        "id": "artifact-1",
                        "version_uuid": "artifact-version-2",
                        "type": "application/vnd.ant.code",
                        "title": "normalizer.py",
                        "content": "def normalize():\n    return True\n",
                    },
                },
                {
                    "type": "text",
                    "text": "Use native identity and lineage.",
                    "citations": [
                        {
                            "type": "webpage",
                            "title": "Provider evidence",
                            "url": "https://evidence.invalid/claude",
                            "start_index": 0,
                            "end_index": 12,
                        }
                    ],
                },
                {"type": "future_native_block", "opaque": {"preserve": True}},
            ],
        },
        "u2": {
            "uuid": "u2",
            "sender": "human",
            "parent_message_uuid": "a-new",
            "created_at": "2026-07-01T10:00:04Z",
            "attachments": [
                {
                    "id": "att-native",
                    "file_uuid": "provider-file-native",
                    "file_name": "evidence.json",
                    "file_type": "application/json",
                    "file_size": 27,
                }
            ],
        },
    }
    sequence = order or ("u1", "a-old", "a-new", "u2")
    return {
        "uuid": "claude-web-conv-1",
        "name": "Claude normalization survivor",
        "created_at": "2026-07-01T10:00:00Z",
        "updated_at": "2026-07-01T10:00:04Z",
        "model": "claude-sonnet-4",
        "model_effort": "medium",
        "thinking_config": {"type": "enabled", "budget_tokens": 4096},
        "current_leaf_message_uuid": "u2",
        "status": "completed",
        "chat_messages": [copy.deepcopy(messages[key]) for key in sequence],
        "files": [
            {
                "file_uuid": "session-file-1",
                "file_name": "session-context.json",
                "file_type": "application/json",
            }
        ],
    }


def _message_facts(session: ParsedSession) -> dict[str, tuple[object, ...]]:
    return {
        message.provider_message_id: (
            message.parent_message_provider_id,
            message.position,
            message.branch_index,
            message.variant_index,
            message.is_active_path,
            message.is_active_leaf,
            message.model_name,
            message.model_effort,
            message.delivery_status,
            message.end_turn,
            message.timestamp,
            tuple(str(block.type) for block in message.blocks),
        )
        for message in session.messages
    }


def _parse_real_route(payload: dict[str, Any]) -> ParsedSession:
    detected = detect_provider(payload)
    assert detected is Provider.CLAUDE_AI
    parsed = parse_payload(detected, payload, "claude-web-fallback")
    assert len(parsed) == 1
    return parsed[0]


def _native_browser_envelope(raw_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "claude-ai:claude-web-conv-1",
        # Deliberately ambiguous top-level shape. If browser-envelope detection
        # moves behind the looser ChatGPT mapping detector, this fixture is stolen.
        "mapping": {"decoy": {"message": {"content": {"parts": ["wrong provider"]}}}},
        "provenance": {
            "source_url": "https://claude.ai/chat/claude-web-conv-1",
            "page_title": "Claude - normalization survivor",
            "captured_at": "2026-07-01T10:00:05+00:00",
            "adapter_name": "claude-native-api-v1",
            "capture_mode": "snapshot",
        },
        "provider_meta": {"capture_fidelity": "native_full"},
        "session": {
            "provider": "claude-ai",
            "provider_session_id": "claude-web-conv-1",
            "title": "Envelope title must not replace native title",
            "created_at": "2026-07-01T10:00:00+00:00",
            "updated_at": "2026-07-01T10:00:04+00:00",
            "model": "claude-sonnet-4",
            "turns": [
                {
                    "provider_turn_id": "u1",
                    "role": "user",
                    "text": "STALE DOM USER TEXT",
                    "ordinal": 0,
                },
                {
                    "provider_turn_id": "a-old",
                    "role": "assistant",
                    "text": "STALE DOM OLD BRANCH",
                    "ordinal": 1,
                    "parent_turn_id": "u1",
                },
                {
                    "provider_turn_id": "a-new",
                    "role": "assistant",
                    "text": "STALE DOM ACTIVE BRANCH",
                    "ordinal": 2,
                    "parent_turn_id": "u1",
                },
                {
                    "provider_turn_id": "u2",
                    "role": "user",
                    "ordinal": 3,
                    "parent_turn_id": "a-new",
                    "attachments": [
                        {
                            "provider_attachment_id": "att-native",
                            "message_provider_id": "u2",
                            "name": "evidence.json",
                            "mime_type": "application/json",
                            "extracted_content": '{"captured":true}',
                        }
                    ],
                },
            ],
        },
        "raw_provider_payload": raw_payload,
    }


def _dom_browser_envelope() -> dict[str, Any]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": "claude-ai:claude-web-conv-1",
        "mapping": {"decoy": {"message": {"content": {"parts": ["wrong provider"]}}}},
        "provenance": {
            "source_url": "https://claude.ai/chat/claude-web-conv-1",
            "page_title": "Claude - normalization survivor",
            "captured_at": "2026-07-01T10:00:05+00:00",
            "adapter_name": "claude-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "claude-ai",
            "provider_session_id": "claude-web-conv-1",
            "title": "Claude normalization survivor",
            "created_at": "2026-07-01T10:00:00+00:00",
            "updated_at": "2026-07-01T10:00:04+00:00",
            "model": "claude-sonnet-4",
            "provider_meta": {
                "current_leaf_message_uuid": "u2",
                "model_effort": "medium",
                "thinking_config": {"type": "enabled", "budget_tokens": 4096},
            },
            "turns": [
                {
                    "provider_turn_id": "u1",
                    "role": "user",
                    "text": "Build a durable normalization law.",
                    "timestamp": "2026-07-01T10:00:00Z",
                    "ordinal": 0,
                },
                {
                    "provider_turn_id": "a-old",
                    "role": "assistant",
                    "text": "An obsolete branch.",
                    "timestamp": "2026-07-01T10:00:01Z",
                    "ordinal": 1,
                    "parent_turn_id": "u1",
                    "provider_meta": {"status": "completed", "end_turn": True},
                },
                {
                    "provider_turn_id": "a-new",
                    "role": "assistant",
                    "text": "DOM fallback cannot expose native thinking or artifact blocks.",
                    "timestamp": "2026-07-01T10:00:02Z",
                    "ordinal": 2,
                    "parent_turn_id": "u1",
                    "provider_meta": {
                        "model": "claude-opus-4-1",
                        "model_effort": "high",
                        "status": "completed",
                        "end_turn": True,
                    },
                },
                {
                    "provider_turn_id": "u2",
                    "role": "user",
                    "timestamp": "2026-07-01T10:00:04Z",
                    "ordinal": 3,
                    "parent_turn_id": "a-new",
                    "attachments": [
                        {
                            "provider_attachment_id": "att-native",
                            "message_provider_id": "u2",
                            "name": "evidence.json",
                            "mime_type": "application/json",
                            "extracted_content": '{"captured":true}',
                        }
                    ],
                },
            ],
        },
    }


def test_claude_export_normalization_survives_record_reordering() -> None:
    """Mutations: branch flattening or thinking-metadata loss must fail."""
    chronological = _parse_real_route(_native_claude_payload())
    shuffled = _parse_real_route(_native_claude_payload(order=("u2", "a-new", "u1", "a-old")))

    assert chronological.provider_session_id == "claude-web-conv-1"
    assert chronological.title == "Claude normalization survivor"
    assert chronological.active_leaf_message_provider_id == "u2"
    assert chronological.created_at == "2026-07-01T10:00:00+00:00"
    assert chronological.updated_at == "2026-07-01T10:00:04+00:00"
    assert [message.provider_message_id for message in chronological.messages] == [
        "u1",
        "a-old",
        "a-new",
        "u2",
    ]
    assert _message_facts(shuffled) == _message_facts(chronological)

    by_id = {message.provider_message_id: message for message in chronological.messages}
    assert by_id["u1"].position == 0
    assert by_id["a-old"].parent_message_provider_id == "u1"
    assert by_id["a-old"].position == 1
    assert by_id["a-old"].variant_index == 0
    assert by_id["a-old"].is_active_path is False
    assert by_id["a-new"].parent_message_provider_id == "u1"
    assert by_id["a-new"].position == 1
    assert by_id["a-new"].branch_index == 1
    assert by_id["a-new"].variant_index == 1
    assert by_id["a-new"].is_active_path is True
    assert by_id["u2"].parent_message_provider_id == "a-new"
    assert by_id["u2"].position == 2
    assert by_id["u2"].text is None
    assert by_id["u2"].blocks == []
    assert by_id["u2"].is_active_leaf is True

    assert by_id["u1"].model_name == "claude-sonnet-4"
    assert by_id["a-new"].model_name == "claude-opus-4-1"
    assert by_id["a-new"].model_effort == "high"
    assert by_id["a-new"].delivery_status == "completed"
    assert by_id["a-new"].end_turn is True
    assert chronological.reported_duration_ms == 3500
    assert chronological.models_used == ["claude-sonnet-4", "claude-opus-4-1"]

    block_types = [str(block.type) for block in by_id["a-new"].blocks]
    assert block_types == ["thinking", "tool_use", "text", "text"]
    artifact_constructs = by_id["a-new"].blocks[1].web_constructs
    assert len(artifact_constructs) == 1
    assert artifact_constructs[0].construct_type == "canvas"
    assert artifact_constructs[0].provider_key == "application/vnd.ant.code"
    assert artifact_constructs[0].source_id == "artifact-version-2"
    citation_constructs = by_id["a-new"].blocks[2].web_constructs
    assert len(citation_constructs) == 1
    assert citation_constructs[0].construct_type == "content_reference"
    assert citation_constructs[0].url == "https://evidence.invalid/claude"
    assert citation_constructs[0].start_index == 0
    assert citation_constructs[0].end_index == 12
    unknown_block = by_id["a-new"].blocks[3]
    assert unknown_block.text is None
    assert unknown_block.metadata == {
        "provider_type": "future_native_block",
        "raw_preserved_in_source": True,
    }

    attachments = {attachment.provider_attachment_id: attachment for attachment in chronological.attachments}
    assert attachments["att-native"].message_provider_id == "u2"
    assert attachments["session-file-1"].message_provider_id is None
    assert {event.event_type for event in chronological.session_events} >= {
        "model_configuration",
        "message_revision",
        "provider_message_update",
        "provider_session_status",
    }
    session_config = next(
        event
        for event in chronological.session_events
        if event.event_type == "model_configuration" and event.source_message_provider_id is None
    )
    assert session_config.payload == {
        "model": "claude-sonnet-4",
        "effort": "medium",
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }
    message_config = next(
        event
        for event in chronological.session_events
        if event.event_type == "model_configuration" and event.source_message_provider_id == "a-new"
    )
    assert message_config.payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    revision = next(
        event
        for event in chronological.session_events
        if event.event_type == "message_revision" and event.source_message_provider_id == "a-new"
    )
    assert revision.source_message_provider_id == "a-new"
    assert revision.payload["revision_id"] == "revision-a-new-2"


def test_authenticated_browser_capture_uses_native_payload_and_enriches_attachment() -> None:
    """Mutations: detector-order theft or attachment replacement loss must fail."""
    direct = _parse_real_route(_native_claude_payload())
    envelope = _native_browser_envelope(_native_claude_payload())
    envelope["capture_id"] = "claude-ai:envelope-decoy"
    session_payload = envelope["session"]
    assert isinstance(session_payload, dict)
    session_payload["provider_session_id"] = "envelope-decoy"

    assert detect_provider(envelope) is Provider.CLAUDE_AI
    parsed = parse_payload(Provider.CLAUDE_AI, envelope, "browser-fallback")
    assert len(parsed) == 1
    native = parsed[0]

    assert native.provider_session_id == direct.provider_session_id == "claude-web-conv-1"
    assert native.provider_session_id != session_payload["provider_session_id"]
    assert native.title == direct.title
    assert _message_facts(native) == _message_facts(direct)
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG in native.ingest_flags
    assert DOM_FALLBACK_INGEST_FLAG not in native.ingest_flags
    assert all("STALE DOM" not in (message.text or "") for message in native.messages)

    attachment = next(row for row in native.attachments if row.provider_attachment_id == "att-native")
    assert attachment.message_provider_id == "u2"
    assert attachment.inline_bytes == b'{"captured":true}'
    assert attachment.name == "evidence.json"
    assert attachment.size_bytes == 27
    assert attachment.provider_file_id == "provider-file-native"
    assert attachment.upload_origin == "oauth"


def test_native_browser_envelope_supplies_only_missing_optional_metadata() -> None:
    """Envelope projection fills omissions but never replaces native conversation identity."""
    raw = _native_claude_payload()
    for key in ("name", "created_at", "updated_at", "model", "model_effort", "thinking_config"):
        raw.pop(key)
    for message in raw["chat_messages"]:
        if isinstance(message, dict) and message.get("uuid") != "a-new":
            message.pop("model", None)
    envelope = _native_browser_envelope(raw)
    session_payload = envelope["session"]
    assert isinstance(session_payload, dict)
    session_payload["title"] = "Envelope metadata fallback"
    session_payload["model"] = "claude-sonnet-4"

    parsed = parse_payload(Provider.CLAUDE_AI, envelope, "browser-fallback")[0]

    assert parsed.provider_session_id == "claude-web-conv-1"
    assert parsed.title == "Envelope metadata fallback"
    assert parsed.created_at == "2026-07-01T10:00:00+00:00"
    assert parsed.updated_at == "2026-07-01T10:00:04+00:00"
    by_id = {message.provider_message_id: message for message in parsed.messages}
    assert by_id["u1"].model_name == "claude-sonnet-4"
    assert by_id["a-new"].model_name == "claude-opus-4-1"
    assert set(parsed.models_used) == {"claude-sonnet-4", "claude-opus-4-1"}
    session_config = next(
        event
        for event in parsed.session_events
        if event.event_type == "model_configuration" and event.source_message_provider_id is None
    )
    assert session_config.payload == {"model": "claude-sonnet-4"}


def test_claude_dom_fallback_keeps_identity_lineage_and_declares_degraded_fidelity() -> None:
    """Mutation witness: replacing DOM-fallback flags or flattening lineage must fail."""
    envelope = _dom_browser_envelope()
    assert detect_provider(envelope) is Provider.CLAUDE_AI

    fallback = parse_payload(Provider.CLAUDE_AI, envelope, "browser-fallback")[0]

    assert fallback.provider_session_id == "claude-web-conv-1"
    assert fallback.active_leaf_message_provider_id == "u2"
    assert DOM_FALLBACK_INGEST_FLAG in fallback.ingest_flags
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG not in fallback.ingest_flags
    by_id = {message.provider_message_id: message for message in fallback.messages}
    assert by_id["a-old"].position == by_id["a-new"].position == 1
    assert by_id["a-old"].variant_index == 0
    assert by_id["a-new"].variant_index == 1
    assert by_id["a-old"].is_active_path is False
    assert by_id["a-new"].is_active_path is True
    assert by_id["u2"].text is None
    assert by_id["u2"].is_active_leaf is True
    assert by_id["a-new"].delivery_status == "completed"
    assert by_id["a-new"].end_turn is True
    assert [str(block.type) for block in by_id["a-new"].blocks] == ["text"]
    assert next(row for row in fallback.attachments if row.provider_attachment_id == "att-native").inline_bytes == (
        b'{"captured":true}'
    )

    native = parse_payload(
        Provider.CLAUDE_AI,
        _native_browser_envelope(_native_claude_payload()),
        "browser-fallback",
    )[0]
    native_blocks = {str(block.type) for message in native.messages for block in message.blocks}
    fallback_blocks = {str(block.type) for message in fallback.messages for block in message.blocks}
    assert {"thinking", "tool_use"}.issubset(native_blocks)
    assert "thinking" not in fallback_blocks
    assert "tool_use" not in fallback_blocks


def test_claude_lineage_walk_is_iterative_and_cycle_diagnostic() -> None:
    """Dependency: recursive lineage traversal or a removed cycle diagnostic must fail."""
    long_messages: list[dict[str, Any]] = []
    for index in range(1100):
        message: dict[str, Any] = {
            "uuid": f"long-{index}",
            "sender": "human" if index % 2 == 0 else "assistant",
            "text": f"turn {index}",
        }
        if index:
            message["parent_message_uuid"] = f"long-{index - 1}"
        long_messages.append(message)

    long_session = _parse_real_route(
        {
            "uuid": "claude-long-lineage",
            "current_leaf_message_uuid": "long-1099",
            "chat_messages": long_messages,
        }
    )
    assert len(long_session.messages) == 1100
    assert long_session.messages[-1].provider_message_id == "long-1099"
    assert long_session.messages[-1].position == 1099
    assert long_session.messages[-1].is_active_leaf is True

    cyclic = _parse_real_route(
        {
            "uuid": "claude-cyclic-lineage",
            "chat_messages": [
                {
                    "uuid": "cycle-a",
                    "sender": "human",
                    "text": "cycle A",
                    "parent_message_uuid": "cycle-b",
                },
                {
                    "uuid": "cycle-b",
                    "sender": "assistant",
                    "text": "cycle B",
                    "parent_message_uuid": "cycle-a",
                },
            ],
        }
    )
    assert CLAUDE_LINEAGE_CYCLE_INGEST_FLAG in cyclic.ingest_flags
    assert {message.provider_message_id for message in cyclic.messages} == {"cycle-a", "cycle-b"}
    assert cyclic.active_leaf_message_provider_id is None


def test_claude_normalization_survives_archive_write_and_hydration(
    workspace_env: dict[str, Path],
) -> None:
    """Real write witness: branch/status/config rows must reach archive storage."""
    payload = _native_claude_payload(order=("u2", "a-new", "u1", "a-old"))
    raw_bytes = json.dumps(payload).encode()
    db_path = db_setup(workspace_env)

    with open_connection(db_path) as conn:
        roundtrip = parse_payload_roundtrip("claude-ai", raw_bytes, unique_id="claude-web-normalization")
        hydrated = write_and_hydrate(roundtrip, conn)
        session_id = str(hydrated.id)
        rows = conn.execute(
            """
            SELECT native_id, position, variant_index, is_active_path, is_active_leaf,
                   parent_message_id, delivery_status, end_turn, model_name
            FROM messages
            WHERE session_id = ?
            ORDER BY position, variant_index
            """,
            (session_id,),
        ).fetchall()
        event_types = {
            str(row[0])
            for row in conn.execute(
                "SELECT event_type FROM session_events WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }
        block_types = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT b.block_type
                FROM blocks AS b
                JOIN messages AS m ON m.message_id = b.message_id
                WHERE m.session_id = ?
                """,
                (session_id,),
            ).fetchall()
        }

    assert hydrated.title == "Claude normalization survivor"
    assert len(hydrated.messages) == 4
    by_native = {str(row[0]): row for row in rows}
    assert tuple(by_native) == ("u1", "a-old", "a-new", "u2")
    assert by_native["a-old"][1:5] == (1, 0, 0, 0)
    assert by_native["a-new"][1:5] == (1, 1, 1, 0)
    assert by_native["u2"][1:5] == (2, 0, 1, 1)
    assert str(by_native["a-new"][5]).endswith(":u1")
    assert by_native["a-new"][6] == "completed"
    assert by_native["a-new"][7] == 1
    assert by_native["a-new"][8] == "claude-opus-4-1"
    assert {"model_configuration", "message_revision", "provider_session_status"}.issubset(event_types)
    assert {"thinking", "tool_use", "text"}.issubset(block_types)
    attachment_only = next(message for message in hydrated.messages if str(message.id).endswith(":u2"))
    assert attachment_only.text is None
    assert [attachment.name for attachment in attachment_only.attachments] == ["evidence.json"]
