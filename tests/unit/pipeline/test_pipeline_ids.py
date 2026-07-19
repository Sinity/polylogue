"""Focused tests for pipeline ID and session content-hash helpers."""

from __future__ import annotations

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.hashing import hash_payload
from polylogue.pipeline.ids import (
    _attachment_hash_payload,
    _message_hash_payload,
    _normalize_for_hash,
    _session_hash_payload,
    session_content_hash,
    session_id,
    session_revision_projection,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import ParsedSessionEvent


def _parsed_message(provider_message_id: str, role: str, text: str, timestamp: str | None) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.normalize(role),
        text=text,
        timestamp=timestamp,
    )


def _parsed_session(
    provider_session_id: str,
    title: str,
    messages: list[ParsedMessage],
    *,
    created_at: str | None,
    updated_at: str | None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=provider_session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        attachments=[],
    )


class TestSessionIdValidation:
    def test_rejects_empty_provider(self) -> None:
        # provider→source_name rename: the empty-first-arg error now names source_name.
        with pytest.raises(ValueError, match="source_name"):
            session_id("", "conv-123")

    def test_rejects_empty_provider_session_id(self) -> None:
        with pytest.raises(ValueError, match="session"):
            session_id("chatgpt", "")


def test_session_content_hash_with_missing_message_ids() -> None:
    session = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("", "user", "Hello", "2024-01-01T00:00:00Z")],
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    digest = session_content_hash(session)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_message_none_vs_empty_timestamp_changes_session_hash() -> None:
    none_ts = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("msg-1", "user", "hello", None)],
        created_at="2024-01-01",
        updated_at=None,
    )
    empty_ts = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("msg-1", "user", "hello", "")],
        created_at="2024-01-01",
        updated_at=None,
    )
    assert session_content_hash(none_ts) != session_content_hash(empty_ts)


def test_session_hash_is_deterministic() -> None:
    session = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("msg-1", "user", "", "2024-01-01")],
        created_at="2024-01-01",
        updated_at=None,
    )
    first = session_content_hash(session)
    second = session_content_hash(session)
    assert first == second
    assert len(first) == 64


def test_message_id_change_changes_session_hash() -> None:
    a = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("msg-1", "user", "hello", "2024-01-01")],
        created_at="2024-01-01",
        updated_at=None,
    )
    b = _parsed_session(
        "conv-1",
        "Test",
        [_parsed_message("msg-2", "user", "hello", "2024-01-01")],
        created_at="2024-01-01",
        updated_at=None,
    )
    assert session_content_hash(a) != session_content_hash(b)


def test_session_hash_empty_messages_is_valid() -> None:
    session = _parsed_session(
        "conv-1",
        "Empty Conv",
        [],
        created_at=None,
        updated_at=None,
    )
    assert len(session_content_hash(session)) == 64


def test_session_hash_timestamps_affect_hash() -> None:
    message = _parsed_message("m1", "user", "hi", None)
    session_one = _parsed_session("conv-1", "Test", [message], created_at="2024-01-01", updated_at=None)
    session_two = _parsed_session("conv-1", "Test", [message], created_at="2024-01-02", updated_at=None)

    assert session_content_hash(session_one) != session_content_hash(session_two)


def _golden_session() -> ParsedSession:
    """Fixed session covering every hashed shape: text, tool_use/tool_result
    blocks (the nested tool_input path), an attachment, and a session event.
    """
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="golden-1",
        title="Golden Session",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:05:00Z",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello", timestamp="2026-01-01T00:00:00Z"),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                text=None,
                timestamp="2026-01-01T00:01:00Z",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id="t1",
                        tool_input={"command": "echo hi", "nested": {"a": 1, "b": [1, 2, 3]}},
                    ),
                    ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="t1", text="hi"),
                ],
            ),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m1",
                name="f.txt",
                mime_type="text/plain",
                size_bytes=3,
            )
        ],
        session_events=[
            ParsedSessionEvent(event_type="turn_context", timestamp="2026-01-01T00:00:30Z", payload={"k": "v"})
        ],
    )


def test_session_revision_projection_golden_hashes() -> None:
    """Pin exact hash bytes (polylogue-fqp0's dedup refactor must not change them).

    These digests were captured from the pre-refactor double-serialization
    implementation. ``session_revision_projection`` now builds each hash-stable
    payload once and reuses it for both the whole-tree hash and the per-item
    hashes -- a pure elimination of redundant computation, not an identity-hash
    change. If this test ever needs updating, that is an evidence epoch and
    requires the migration/fingerprint-bump story in polylogue-fqp0's design,
    not a routine test-fixture update.
    """
    session = _golden_session()
    projection = session_revision_projection(session)

    assert projection.session_hash.hex() == "4d768f0de553d68aa26dd2a3edc137cebbbcc147ec51ab75a5b55e05bd563f87"
    assert [h.hex() for h in projection.message_hashes] == [
        "65a99313c3ed8b81e69ecc0b36f314b3bf7848bc10fcea11415ddb9b07188941",
        "8efbf7b5b70bef4d73ec3550fe97e6f4d456cd724550bc5621a36766fe7f1f8b",
    ]
    assert {h.hex() for h in projection.attachment_hashes} == {
        "f0a9ad8518ab2426ed33575cf5eb0ee0a55d852d3c40c7a2f8f600f4f2f8e89b"
    }
    assert [h.hex() for h in projection.event_hashes] == [
        "8f6539c2bc89ff2c78e183cda534a04f4d14823a0416df54d73fbee6f1f0824f"
    ]


def test_session_revision_projection_session_hash_matches_session_content_hash() -> None:
    """The projection's session_hash must always equal session_content_hash's output."""
    for session in (_golden_session(), _parsed_session("conv-2", "Empty", [], created_at=None, updated_at=None)):
        projection = session_revision_projection(session)
        assert projection.session_hash.hex() == session_content_hash(session)


def test_session_revision_projection_matches_independent_recomputation() -> None:
    """Dedup sharing must not drift from recomputing each payload independently.

    Directly reproduces the pre-refactor "build payloads twice" shape inline
    (not by calling pipeline internals) and asserts identical output --
    guards the dedup refactor (polylogue-fqp0) against silently changing what
    gets hashed while eliminating the redundant construction.
    """
    session = _golden_session()
    projection = session_revision_projection(session)

    independent_message_payloads = [
        _message_hash_payload(m, m.provider_message_id or f"msg-{i}") for i, m in enumerate(session.messages, start=1)
    ]
    independent_attachment_payloads = [_attachment_hash_payload(a) for a in session.attachments]
    independent_event_payloads = [
        {
            "event_index": idx,
            "event_type": _normalize_for_hash(e.event_type),
            "timestamp": _normalize_for_hash(e.timestamp),
            "source_message_provider_id": _normalize_for_hash(e.source_message_provider_id),
            "payload": hash_payload(e.payload),
        }
        for idx, e in enumerate(session.session_events)
    ]
    independent_session_hash = hash_payload(
        _session_hash_payload(
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            messages=independent_message_payloads,
            attachments=independent_attachment_payloads,
            session_events=independent_event_payloads,
        )
    )

    assert projection.session_hash.hex() == independent_session_hash
    assert list(projection.message_hashes) == [bytes.fromhex(hash_payload(p)) for p in independent_message_payloads]
    assert projection.attachment_hashes == frozenset(
        bytes.fromhex(hash_payload(p)) for p in independent_attachment_payloads
    )
    assert list(projection.event_hashes) == [bytes.fromhex(hash_payload(p)) for p in independent_event_payloads]
