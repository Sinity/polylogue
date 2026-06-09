"""Focused tests for pipeline ID and session content-hash helpers."""

from __future__ import annotations

import pytest

from polylogue.archive.message.roles import Role
from polylogue.pipeline.ids import (
    session_content_hash,
    session_id,
)
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.types import Provider


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
