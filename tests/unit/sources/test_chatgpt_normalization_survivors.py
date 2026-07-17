from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.ingest_flags import DOM_FALLBACK_INGEST_FLAG, NATIVE_BROWSER_CAPTURE_INGEST_FLAG
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONDocument
from polylogue.sources import dispatch
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import browser_capture, chatgpt
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveRawParsedWriteResult, ArchiveStore

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "chatgpt"
_NATIVE_FIXTURE = _FIXTURE_DIR / "native-conversation-v1.json"
_BROWSER_FIXTURE = _FIXTURE_DIR / "native-browser-capture-v1.json"


def _load_fixture(path: Path) -> JSONDocument:
    return cast("JSONDocument", json.loads(path.read_text(encoding="utf-8")))


def _parse_one(payload: JSONDocument) -> ParsedSession:
    provider = detect_provider(payload)
    assert provider is Provider.CHATGPT
    sessions = parse_payload(provider, payload, "fixture-fallback")
    assert len(sessions) == 1
    return sessions[0]


def test_chatgpt_native_wire_fixture_survives_dispatch_and_semantic_normalization() -> None:
    """Actual detect -> lower -> parse keeps the provider's full structured turn graph.

    Production dependencies: dispatch.detect_provider, dispatch._lower_payload_specs,
    dispatch._parse_lowered_spec, and chatgpt.parse. Removing any rich-field
    projection below (authoredness, branch identity, blocks, attachments, timing,
    effort, status) makes this survivor fail rather than falling back to prose.
    """

    payload = _load_fixture(_NATIVE_FIXTURE)
    session = _parse_one(payload)
    by_id = {message.provider_message_id: message for message in session.messages}

    assert session.provider_session_id == "chatgpt-fixture-conversation"
    assert session.title == "Privacy-safe ChatGPT native fixture"
    assert session.created_at == "1784164500.0"
    assert session.updated_at == "1784169740.0"
    assert session.active_leaf_message_provider_id == "answer-active-message"
    assert len(session.messages) == 8

    context = by_id["context-message"]
    user = by_id["user-message"]
    inactive = by_id["answer-inactive-message"]
    thought = by_id["thought-active-message"]
    tool_call = by_id["tool-call-message"]
    tool_result = by_id["tool-result-message"]
    recap = by_id["recap-active-message"]
    answer = by_id["answer-active-message"]

    assert context.message_type is MessageType.CONTEXT
    assert context.material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert user.material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert thought.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert tool_call.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert tool_result.material_origin is MaterialOrigin.TOOL_RESULT
    assert answer.material_origin is MaterialOrigin.ASSISTANT_AUTHORED

    assert inactive.variant_index == 0
    assert inactive.is_active_path is False
    assert thought.variant_index == 1
    assert thought.is_active_path is True
    assert answer.is_active_path is True
    assert answer.is_active_leaf is True
    assert answer.parent_message_provider_id == "recap-active-message"

    assert [block.type for block in user.blocks] == [BlockType.TEXT, BlockType.IMAGE]
    assert user.blocks[1].metadata == {"asset_pointer": "file-service://fixture-input-image"}
    assert [block.type for block in thought.blocks] == [BlockType.THINKING]
    assert thought.blocks[0].text == "Privacy-safe reasoning trace."
    assert [block.type for block in tool_call.blocks] == [BlockType.TOOL_USE]
    assert tool_call.blocks[0].tool_name == "browser.search"
    assert tool_call.blocks[0].tool_input == {"query": "fixture evidence"}
    assert [block.type for block in tool_result.blocks] == [BlockType.TOOL_RESULT]
    assert tool_result.blocks[0].text == "Privacy-safe tool result."
    assert [block.type for block in recap.blocks] == [BlockType.THINKING]
    assert [block.type for block in answer.blocks] == [BlockType.TEXT, BlockType.IMAGE]
    assert answer.blocks[1].metadata == {"asset_pointer": "file-service://fixture-output-image"}

    attachments = {attachment.provider_attachment_id: attachment for attachment in session.attachments}
    upload = attachments["fixture-upload-1"]
    sandbox = attachments["sandbox:answer-active-message:/mnt/data/fixture-result.zip"]
    assert upload.message_provider_id == "user-message"
    assert upload.provider_file_id == "file-fixture-upload-1"
    assert upload.upload_origin == "oauth"
    assert sandbox.message_provider_id == "answer-active-message"
    assert sandbox.name == "fixture-result.zip"
    assert sandbox.attachment_kind == "sandbox_file"
    assert sandbox.source_url == "sandbox:/mnt/data/fixture-result.zip"

    assert inactive.model_name == "gpt-5-6-pro"
    assert inactive.model_effort == "standard"
    assert thought.model_name == "gpt-5-6-pro"
    assert thought.model_effort == "extended"
    assert answer.model_effort == "extended"
    assert answer.timestamp == "1784169733.0"
    assert answer.delivery_status == "finished_successfully"
    assert answer.end_turn is True

    # Native lifecycle fields and both legacy duration spellings are repeated
    # across the branch. One recap owns one semantic duration; no duplicate row
    # contributes a second 5,190,000 ms to the session total.
    assert recap.duration_ms == 5_190_000
    assert all(message.duration_ms is None for message in (thought, tool_call, tool_result, answer))
    assert session.reported_duration_ms == 5_190_000
    lifecycle = [event for event in session.session_events if event.event_type == "generation_lifecycle"]
    assert len(lifecycle) == 1
    assert lifecycle[0].source_message_provider_id == "recap-active-message"
    assert lifecycle[0].timestamp == "1784169732.588194"
    assert lifecycle[0].payload == {
        "state": "completed",
        "evidence_source": "provider_native",
        "fidelity": "exact",
        "duration_semantics": "provider_reported_elapsed",
        "elapsed_duration_ms": 5_190_000,
        "started_at_ms": 1_784_164_541_690,
        "ended_at_ms": 1_784_169_732_588,
    }


def test_timing_only_provider_node_rehomes_duration_to_the_last_emitted_branch_message() -> None:
    """A metadata-only winning node must not make its generation duration disappear.

    Production dependency: chatgpt.parse resolves the selected lifecycle owner
    after message emission. Removing that rehoming projects no message duration
    and leaves the event attached to an unmaterialized provider message id.
    """

    payload = _load_fixture(_NATIVE_FIXTURE)
    mapping = cast("dict[str, dict[str, object]]", payload["mapping"])
    recap_message = cast("dict[str, object]", mapping["recap-active"]["message"])
    recap_metadata = cast("dict[str, object]", recap_message["metadata"])
    recap_metadata.pop("finished_duration_sec")
    answer_node = mapping["answer-active"]
    answer_message = cast("dict[str, object]", answer_node["message"])
    answer_metadata = cast("dict[str, object]", answer_message["metadata"])
    answer_metadata.pop("finished_duration_sec")
    answer_node["children"] = ["timing-only-node"]
    mapping["timing-only-node"] = {
        "id": "timing-only-node",
        "parent": "answer-active",
        "children": [],
        "message": {
            "id": "timing-only-message",
            "author": {"role": "assistant"},
            "create_time": 1784169734.0,
            "content": {"content_type": "text", "parts": []},
            "status": "finished_successfully",
            "end_turn": True,
            "metadata": {
                "reasoning_start_time": 1784164541.690012,
                "reasoning_end_time": 1784169732.588194,
                "finished_duration_sec": 5190,
                "durationMs": 5190000,
            },
        },
    }
    payload["current_node"] = "timing-only-node"

    session = _parse_one(payload)
    by_id = {message.provider_message_id: message for message in session.messages}
    lifecycle = [event for event in session.session_events if event.event_type == "generation_lifecycle"]

    assert len(session.messages) == 8
    assert by_id["answer-active-message"].duration_ms == 5_190_000
    assert by_id["recap-active-message"].duration_ms is None
    assert session.reported_duration_ms == 5_190_000
    assert len(lifecycle) == 1
    assert lifecycle[0].source_message_provider_id == "answer-active-message"


def test_generation_dedup_preserves_a_distinct_message_local_legacy_duration() -> None:
    """Only a legacy value equal to the selected lifecycle duration is a duplicate."""

    payload = _load_fixture(_NATIVE_FIXTURE)
    mapping = cast("dict[str, dict[str, object]]", payload["mapping"])
    tool_result_message = cast("dict[str, object]", mapping["tool-result"]["message"])
    tool_result_metadata = cast("dict[str, object]", tool_result_message["metadata"])
    tool_result_metadata["durationMs"] = 250

    session = _parse_one(payload)
    by_id = {message.provider_message_id: message for message in session.messages}

    assert by_id["recap-active-message"].duration_ms == 5_190_000
    assert by_id["tool-result-message"].duration_ms == 250
    assert session.reported_duration_ms == 5_190_250


def test_bare_legacy_duration_remains_message_local_without_a_synthetic_lifecycle_event() -> None:
    """Legacy-only exports retain the pre-existing message-duration contract."""

    payload = cast(
        "JSONDocument",
        {
            "id": "legacy-duration-only",
            "current_node": "answer-node",
            "mapping": {
                "answer-node": {
                    "id": "answer-node",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "answer-message",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Legacy answer"]},
                        "metadata": {"duration_ms": 750},
                    },
                }
            },
        },
    )

    session = _parse_one(payload)

    assert session.messages[0].duration_ms == 750
    assert session.reported_duration_ms == 750
    assert session.session_events == []


def test_malformed_native_timing_does_not_promote_a_legacy_duration_to_lifecycle() -> None:
    """An invalid native spelling cannot confer lifecycle meaning on legacy data."""

    payload = cast(
        "JSONDocument",
        {
            "id": "malformed-native-with-legacy-duration",
            "current_node": "answer-node",
            "mapping": {
                "answer-node": {
                    "id": "answer-node",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "answer-message",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Legacy answer"]},
                        "metadata": {
                            "finished_duration_sec": "pending",
                            "durationMs": 750,
                        },
                    },
                }
            },
        },
    )

    session = _parse_one(payload)

    assert session.messages[0].duration_ms == 750
    assert session.reported_duration_ms == 750
    assert session.session_events == []


def test_browser_capture_detector_short_circuits_the_weaker_chatgpt_mapping_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser envelope detector must run before the weak mapping detector."""

    payload = _load_fixture(_BROWSER_FIXTURE)
    payload["mapping"] = {"decoy-chatgpt-node": {}}
    assert browser_capture.looks_like(payload)
    assert chatgpt.looks_like(payload)

    def fail_if_weaker_detector_runs(_payload: object) -> bool:
        pytest.fail("ChatGPT mapping detector ran before the browser-capture envelope detector")

    monkeypatch.setattr(chatgpt, "looks_like", fail_if_weaker_detector_runs)
    assert dispatch.detect_provider(payload) is Provider.CHATGPT


def test_native_browser_capture_delegates_to_chatgpt_and_merges_acquired_assets() -> None:
    """Browser capture stays a transport route, not a second ChatGPT protocol."""

    native = _parse_one(_load_fixture(_NATIVE_FIXTURE))
    captured = _parse_one(_load_fixture(_BROWSER_FIXTURE))

    assert captured.messages == native.messages
    assert captured.session_events == native.session_events
    assert captured.reported_duration_ms == native.reported_duration_ms == 5_190_000
    assert captured.title == native.title
    assert DOM_FALLBACK_INGEST_FLAG not in captured.ingest_flags
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG in captured.ingest_flags
    assert {message.provider_message_id for message in captured.messages}.isdisjoint(
        {"dom-user-message", "dom-answer-message"}
    )

    attachments = {attachment.provider_attachment_id: attachment for attachment in captured.attachments}
    acquired = attachments["sandbox:answer-active-message:/mnt/data/fixture-result.zip"]
    assert acquired.inline_bytes == b"PK\x03\x04privacy-safe-fixture"
    assert acquired.size_bytes == len(b"PK\x03\x04privacy-safe-fixture")
    assert acquired.mime_type == "application/zip"
    assert acquired.upload_origin == "paste"
    assert acquired.attachment_kind == "sandbox_file"
    assert acquired.source_url == "sandbox:/mnt/data/fixture-result.zip"
    assert (
        len(
            [
                attachment
                for attachment in captured.attachments
                if attachment.provider_attachment_id.startswith("sandbox:")
            ]
        )
        == 1
    )


def _long_dom_fallback_payload() -> JSONDocument:
    payload = deepcopy(_load_fixture(_BROWSER_FIXTURE))
    payload.pop("raw_provider_payload", None)
    payload["provider_meta"] = {}
    session = cast("dict[str, object]", payload["session"])
    session["provider_meta"] = {}
    session["title"] = "Longer DOM fallback must not win"
    session["updated_at"] = "2026-07-17T23:59:59+00:00"
    turns = cast("list[dict[str, object]]", session["turns"])
    for index in range(2, 12):
        turns.append(
            {
                "provider_turn_id": f"dom-extra-{index}",
                "role": "assistant" if index % 2 else "user",
                "text": f"DOM-only extra turn {index}",
                "timestamp": f"2026-07-17T{index:02d}:00:00+00:00",
                "ordinal": index,
            }
        )
    return payload


@pytest.mark.parametrize("arrival_order", [("dom", "native"), ("native", "dom")])
def test_native_full_fidelity_replaces_or_resists_longer_dom_fallback_and_keeps_raw_fields(
    tmp_path: Path,
    arrival_order: tuple[str, str],
) -> None:
    """Real parser flags drive archive replacement in both arrival orders.

    Production dependencies: browser_capture.parse emits fidelity flags and
    ArchiveStore.write_raw_and_parsed_result applies them. Removing native
    replacement, allowing fallback replacement, or relinking the retained
    session to fallback raw bytes makes this survivor fail.
    """

    payloads = {
        "native": _load_fixture(_BROWSER_FIXTURE),
        "dom": _long_dom_fallback_payload(),
    }
    parsed = {kind: _parse_one(payload) for kind, payload in payloads.items()}
    assert len(parsed["native"].messages) == 8
    assert len(parsed["dom"].messages) == 12
    assert NATIVE_BROWSER_CAPTURE_INGEST_FLAG in parsed["native"].ingest_flags
    assert DOM_FALLBACK_INGEST_FLAG in parsed["dom"].ingest_flags

    root = tmp_path / "archive"
    outcomes: dict[str, ArchiveRawParsedWriteResult] = {}
    with ArchiveStore(root) as archive:
        for acquired_at_ms, kind in enumerate(arrival_order, start=1_800_000_000_000):
            outcomes[kind] = archive.write_raw_and_parsed_result(
                parsed[kind],
                payload=json.dumps(payloads[kind], sort_keys=True, separators=(",", ":")).encode(),
                source_path=f"/fixture/{kind}-capture.json",
                acquired_at_ms=acquired_at_ms,
            )

        # A still-newer, longer fallback must not relink or replace native data.
        final_fallback = archive.write_raw_and_parsed_result(
            parsed["dom"],
            payload=json.dumps(payloads["dom"], indent=2).encode(),
            source_path="/fixture/dom-capture-latest.json",
            acquired_at_ms=1_800_000_000_100,
        )
        native_outcome = outcomes["native"]
        retained = archive.read_session(native_outcome.session_id)
        provider, raw_material, raw_source_path, _revision_kind = archive.raw_revision_material(native_outcome.raw_id)

    with sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True) as conn:
        retained_session_row = conn.execute(
            "SELECT raw_id, reported_duration_ms FROM sessions WHERE session_id = ?",
            (native_outcome.session_id,),
        ).fetchone()
        retained_message_rows = conn.execute(
            "SELECT native_id, model_effort, duration_ms FROM messages WHERE session_id = ?",
            (native_outcome.session_id,),
        ).fetchall()
    assert retained_session_row is not None
    retained_raw_id = str(retained_session_row[0])
    retained_reported_duration_ms = int(retained_session_row[1])
    retained_message_facts = {
        str(native_id): (model_effort, duration_ms) for native_id, model_effort, duration_ms in retained_message_rows
    }

    assert retained.title == "Privacy-safe ChatGPT native fixture"
    assert len(retained.messages) == 8
    assert {message.native_id for message in retained.messages} == {
        "context-message",
        "user-message",
        "answer-inactive-message",
        "thought-active-message",
        "tool-call-message",
        "tool-result-message",
        "recap-active-message",
        "answer-active-message",
    }
    assert final_fallback.content_changed is False
    assert final_fallback.counts["skipped_sessions"] == 1
    assert retained_raw_id == native_outcome.raw_id
    assert retained_reported_duration_ms == 5_190_000
    assert retained_message_facts["thought-active-message"] == ("extended", 5_190_000)
    assert retained_message_facts["answer-active-message"] == ("extended", 5_190_000)
    assert provider is Provider.CHATGPT
    assert raw_source_path == "/fixture/native-capture.json"
    assert b'"reasoning_start_time":1784164541.690012' in raw_material
    assert b'"reasoning_end_time":1784169732.588194' in raw_material
    assert b'"finished_duration_sec":5190' in raw_material
    assert b'"durationMs":5190000' in raw_material
    assert b'"duration_ms":5190000' in raw_material
    assert b'"thinking_effort":"extended"' in raw_material
