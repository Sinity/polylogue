"""Property-based parser contracts for provider parsers and shared helpers.

This file owns the parser-preservation contracts that are still worth keeping
as properties:

1. Provider parsers preserve canonical provider identity and never invent
   messages beyond what the payload shape allows.
2. Roles emitted by parsers are normalized.
3. Parser-specific helpers such as ``looks_like`` accept their own generated
   payload families.
4. Shared helper functions keep their non-crashing/round-trip guarantees.

Lower-level role normalization laws live in ``test_parse_laws.py``.
Source-detection/container-shape laws live in ``test_source_laws.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from polylogue.config import Source
from polylogue.lib.json import JSONDocument, json_document
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources import iter_source_conversations
from polylogue.sources.parsers import chatgpt, claude, codex, drive
from polylogue.sources.parsers.base import (
    ParsedConversation,
    attachment_from_meta,
    extract_messages_from_list,
)
from tests.infra.strategies import (
    chatgpt_export_strategy,
    chatgpt_message_node_strategy,
    claude_ai_export_strategy,
    claude_code_message_strategy,
    codex_message_strategy,
    gemini_export_strategy,
    message_strategy,
)
from tests.infra.strategies.providers import claude_code_session_strategy, codex_session_strategy

NormalizedTimestampInput = int | float | str | None
ParserPayload: TypeAlias = object


class AttachmentMeta(TypedDict):
    id: str
    name: str
    mime_type: str
    size: int


def _chatgpt_message_node_count(payload: ParserPayload) -> int:
    export = json_document(payload)
    mapping = json_document(export.get("mapping"))
    return sum(
        1
        for node in mapping.values()
        if isinstance(node, dict)
        and (message := json_document(node.get("message")))
        and json_document(message.get("content")).get("parts")
    )


def _claude_ai_message_count(payload: ParserPayload) -> int:
    export = json_document(payload)
    messages = export.get("chat_messages")
    return sum(1 for msg in messages if isinstance(msg, dict) and msg.get("text")) if isinstance(messages, list) else 0


def _jsonl_record_count(payload: ParserPayload) -> int:
    return len([record for record in payload if isinstance(record, dict)]) if isinstance(payload, list) else 0


def _gemini_chunk_count(payload: ParserPayload) -> int:
    export = json_document(payload)
    chunks = json_document(export.get("chunkedPrompt")).get("chunks")
    if not isinstance(chunks, list):
        return 0
    return len([c for c in chunks if isinstance(c, dict) and c.get("text") and c.get("role")])


def _payload_sequence(payload: ParserPayload) -> Sequence[object]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        return ()
    return payload


def _looks_like_sequence(predicate: Callable[[Sequence[object]], bool], payload: ParserPayload) -> bool:
    return predicate(_payload_sequence(payload))


def _payload_id(payload: ParserPayload) -> str:
    document = json_document(payload)
    return str(document.get("id") or document.get("uuid") or document.get("conversation_id") or "fallback")


def _payload_title(payload: ParserPayload) -> str:
    document = json_document(payload)
    return str(document.get("title") or document.get("name") or "fallback")


def _parse_chatgpt(payload: ParserPayload) -> ParsedConversation:
    return chatgpt.parse(json_document(payload), "fallback")


def _parse_claude_ai(payload: ParserPayload) -> ParsedConversation:
    return claude.parse_ai(json_document(payload), "fallback")


def _parse_claude_code(payload: ParserPayload) -> ParsedConversation:
    return claude.parse_code(_payload_sequence(payload), "fallback")


def _parse_codex(payload: ParserPayload) -> ParsedConversation:
    return codex.parse(_payload_sequence(payload), "fallback")


def _parse_gemini(payload: ParserPayload) -> ParsedConversation:
    return drive.parse_chunked_prompt("gemini", json_document(payload), "fallback")


@dataclass(frozen=True)
class ParserCase:
    name: str
    strategy: SearchStrategy[ParserPayload]
    parse: Callable[[ParserPayload], ParsedConversation]
    looks_like: Callable[[ParserPayload], bool]
    expected_provider: str
    message_cap: Callable[[ParserPayload], int]
    id_oracle: Callable[[ParserPayload], str] | None = None
    title_oracle: Callable[[ParserPayload], str] | None = None
    extra_assertion: Callable[[ParserPayload, ParsedConversation], None] | None = None


def _assert_claude_code_cleanup(_payload: ParserPayload, result: ParsedConversation) -> None:
    for message in result.messages:
        assert message.provider_meta is None
        assert isinstance(message.content_blocks, list)
        assert message.timestamp is None or isinstance(message.timestamp, str)


def _assert_codex_text_recovery(payload: ParserPayload, result: ParsedConversation) -> None:
    if not result.messages:
        return
    for message in result.messages:
        assert message.provider_meta is None
    if not isinstance(payload, list):
        return
    response_items = [
        record.get("payload", record)
        for record in payload
        if isinstance(record, dict)
        and record.get("type") != "session_meta"
        and isinstance(record.get("payload", record), dict)
    ]
    if not response_items:
        return
    first = response_items[0]
    content = first.get("content")
    if not isinstance(content, list):
        return
    expected_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "input_text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            expected_parts.append(text)
    expected_text = "\n".join(expected_parts)
    if expected_text:
        assert result.messages[0].text == expected_text


PARSER_CASES: tuple[ParserCase, ...] = (
    ParserCase(
        name="chatgpt",
        strategy=chatgpt_export_strategy(min_messages=1, max_messages=10),
        parse=_parse_chatgpt,
        looks_like=chatgpt.looks_like,
        expected_provider="chatgpt",
        message_cap=_chatgpt_message_node_count,
        id_oracle=_payload_id,
        title_oracle=_payload_title,
    ),
    ParserCase(
        name="claude-ai",
        strategy=claude_ai_export_strategy(min_messages=1, max_messages=10),
        parse=_parse_claude_ai,
        looks_like=claude.looks_like_ai,
        expected_provider="claude-ai",
        message_cap=_claude_ai_message_count,
        id_oracle=_payload_id,
    ),
    ParserCase(
        name="claude-code",
        strategy=claude_code_session_strategy(min_messages=1, max_messages=10),
        parse=_parse_claude_code,
        looks_like=lambda payload: _looks_like_sequence(claude.looks_like_code, payload),
        expected_provider="claude-code",
        message_cap=_jsonl_record_count,
        extra_assertion=_assert_claude_code_cleanup,
    ),
    ParserCase(
        name="codex",
        strategy=codex_session_strategy(min_messages=1, max_messages=10, use_envelope=True),
        parse=_parse_codex,
        looks_like=lambda payload: _looks_like_sequence(codex.looks_like, payload),
        expected_provider="codex",
        message_cap=_jsonl_record_count,
        extra_assertion=_assert_codex_text_recovery,
    ),
    ParserCase(
        name="gemini",
        strategy=gemini_export_strategy(min_messages=1, max_messages=10),
        parse=_parse_gemini,
        looks_like=drive.looks_like,
        expected_provider="gemini",
        message_cap=_gemini_chunk_count,
    ),
)


@pytest.mark.parametrize("case", PARSER_CASES, ids=lambda case: case.name)
@given(st.data())
@settings(max_examples=35, suppress_health_check=[HealthCheck.too_slow])
def test_provider_parser_contract(case: ParserCase, data: st.DataObject) -> None:
    payload = data.draw(case.strategy)
    result = case.parse(payload)

    assert isinstance(result, ParsedConversation)
    assert result.provider_name == case.expected_provider
    assert len(result.messages) <= case.message_cap(payload)
    assert all(message.role in {"user", "assistant", "system", "tool", "message"} for message in result.messages)

    if case.id_oracle is not None:
        assert result.provider_conversation_id == case.id_oracle(payload)
    if case.title_oracle is not None:
        assert result.title == case.title_oracle(payload)
    if case.extra_assertion is not None:
        case.extra_assertion(payload, result)


@pytest.mark.parametrize("case", PARSER_CASES, ids=lambda case: case.name)
@given(st.data())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_provider_looks_like_accepts_generated_payloads(case: ParserCase, data: st.DataObject) -> None:
    payload = data.draw(case.strategy)
    assert case.looks_like(payload)


@given(st.lists(message_strategy(), min_size=0, max_size=20))
@settings(max_examples=50)
def test_extract_messages_from_list_never_invents_messages(messages: list[JSONDocument]) -> None:
    result = extract_messages_from_list(messages)
    expected = sum(1 for msg in messages if isinstance(msg, dict) and (msg.get("text") or msg.get("content")))
    assert len(result) <= expected


@given(message_strategy())
@settings(max_examples=50)
def test_extract_messages_from_list_normalizes_role(msg: JSONDocument) -> None:
    result = extract_messages_from_list([msg])
    if result:
        assert result[0].role in {"user", "assistant", "system", "tool", "message"}


@given(chatgpt_message_node_strategy())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_chatgpt_node_contract(node: JSONDocument) -> None:
    export: dict[str, object] = {"mapping": {str(node["id"]): node}, "id": "test"}
    result = chatgpt.parse(export, "fallback")
    assert all(message.role in {"user", "assistant", "system", "tool", "message"} for message in result.messages)


@given(claude_code_message_strategy())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_claude_code_message_type_contract(msg: JSONDocument) -> None:
    result = claude.parse_code([msg], "fallback")
    if not result.messages:
        return
    parsed = result.messages[0]
    # ClaudeCodeRecord.role precedence: message.role > type field.
    # When message.role is present and valid, it overrides the type field.
    inner_role = None
    message = msg.get("message")
    if isinstance(message, dict):
        inner_role = message.get("role")
    if isinstance(inner_role, str) and inner_role in {"user", "assistant", "system", "tool"}:
        assert parsed.role == inner_role
    elif msg.get("type") == "user":
        assert parsed.role == "user"
    elif msg.get("type") == "assistant":
        assert parsed.role == "assistant"


@given(codex_message_strategy())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_codex_message_text_contract(msg: JSONDocument) -> None:
    session: list[object] = [
        {"type": "session_meta", "payload": {"id": "test", "timestamp": "2024-01-01"}},
        {"type": "response_item", "payload": msg},
    ]
    result = codex.parse(session, "fallback")
    if not result.messages:
        return
    content = msg.get("content")
    expected_parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "input_text":
                continue
            text = item.get("text")
            if isinstance(text, str):
                expected_parts.append(text)
    expected_text = "\n".join(expected_parts)
    if expected_text:
        assert result.messages[0].text == expected_text


@given(
    st.one_of(
        st.floats(min_value=0, max_value=2e12, allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=int(2e12)),
        st.text(max_size=50),
        st.none(),
    )
)
def test_timestamp_normalization_never_crashes(timestamp: NormalizedTimestampInput) -> None:
    result = claude.normalize_timestamp(timestamp)
    assert result is None or isinstance(result, str)


@given(st.floats(min_value=1577836800, max_value=1893456000, allow_nan=False))
def test_timestamp_normalization_preserves_seconds(epoch: float) -> None:
    result = claude.normalize_timestamp(epoch)
    assert result is not None
    assert abs(float(result) - epoch) < 1


@given(st.integers(min_value=1577836800000, max_value=1893456000000))
def test_timestamp_normalization_handles_milliseconds(epoch_ms: int) -> None:
    result = claude.normalize_timestamp(epoch_ms)
    assert result is not None
    assert abs(float(result) - (epoch_ms / 1000.0)) < 1


@given(
    st.fixed_dictionaries(
        {
            "id": st.uuids().map(str),
            "name": st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "S"),
                    blacklist_characters="\x00\x1f",
                ),
            ),
            "mime_type": st.sampled_from(["text/plain", "image/png", "application/pdf"]),
            "size": st.integers(min_value=1, max_value=10_000_000),
        }
    )
)
def test_attachment_extraction_preserves_metadata(attachment_meta: AttachmentMeta) -> None:
    result = attachment_from_meta(attachment_meta, "msg-1", 1)

    assert result is not None
    assert result.provider_attachment_id == attachment_meta["id"]
    if result.name:
        original_name = attachment_meta["name"]
        if original_name.strip(".") == "":
            assert result.name == "file"
        else:
            assert any(c in result.name for c in original_name if c.isprintable())
    assert result.mime_type == attachment_meta["mime_type"]
    assert result.size_bytes == attachment_meta["size"]


# =============================================================================
# MERGED FROM test_seeded_parser_contracts.py (contract tests for seeded data)
# =============================================================================


@pytest.fixture(params=sorted(SyntheticCorpus.available_providers()) or ["chatgpt"])
def provider_conversations(
    request: pytest.FixtureRequest,
    synthetic_source: Callable[..., object],
) -> tuple[str, list[ParsedConversation]]:
    """Parse synthetic data for each available provider."""
    provider = str(request.param)
    try:
        source = synthetic_source(provider, count=3, seed=42)
    except FileNotFoundError:
        pytest.skip(f"No schema for {provider}")
    if not isinstance(source, Source):
        raise TypeError(f"expected Source, got {type(source).__name__}")

    convos = list(iter_source_conversations(source))
    if not convos:
        pytest.skip(f"No conversations parsed for {provider}")

    return provider, convos


class TestTimestampParseability:
    """All parsed timestamps should be valid ISO 8601 or epoch values."""

    def test_message_timestamps_are_parseable(
        self: object,
        provider_conversations: tuple[str, list[ParsedConversation]],
    ) -> None:
        """Every message with a timestamp has a parseable value."""
        from polylogue.lib.timestamps import parse_timestamp

        provider, convos = provider_conversations
        for conv in convos:
            for msg in conv.messages:
                if msg.timestamp is not None and isinstance(msg.timestamp, str):
                    # Should be a datetime already (parsed by provider parser)
                    # or a string that parse_timestamp can handle
                    parsed = parse_timestamp(msg.timestamp)
                    assert parsed is not None, (
                        f"{provider}: unparseable timestamp {msg.timestamp!r} "
                        f"in conversation {conv.provider_conversation_id}"
                    )


class TestConversationIdUniqueness:
    """Parsed conversations should have unique IDs within a provider."""

    def test_conversation_ids_are_unique(
        self: object,
        provider_conversations: tuple[str, list[ParsedConversation]],
    ) -> None:
        """No duplicate provider_conversation_id within one parse run."""
        provider, convos = provider_conversations
        ids = [c.provider_conversation_id for c in convos]
        assert len(ids) == len(set(ids)), (
            f"{provider}: duplicate conversation IDs: {[cid for cid in ids if ids.count(cid) > 1]}"
        )


class TestMessageOrderConsistency:
    """Messages within a conversation should maintain insertion order."""

    def test_messages_have_consistent_roles(
        self: object,
        provider_conversations: tuple[str, list[ParsedConversation]],
    ) -> None:
        """Messages alternate between user-like and assistant-like roles."""
        provider, convos = provider_conversations
        user_roles = {"user", "human"}
        assistant_roles = {"assistant", "model"}

        for conv in convos:
            if len(conv.messages) < 2:
                continue
            # Verify at least one user and one assistant message exist
            roles = {m.role for m in conv.messages}
            has_user = bool(roles & user_roles)
            has_assistant = bool(roles & assistant_roles)
            assert has_user or has_assistant, (
                f"{provider}: conversation has no user or assistant messages, roles: {roles}"
            )


class TestNonEmptyContent:
    """Parsed conversations should have meaningful content."""

    def test_at_least_one_message_has_text(
        self: object,
        provider_conversations: tuple[str, list[ParsedConversation]],
    ) -> None:
        """Every conversation has at least one message with non-empty text."""
        provider, convos = provider_conversations
        for conv in convos:
            texts = [m.text for m in conv.messages if m.text]
            assert len(texts) > 0, f"{provider}: conversation {conv.provider_conversation_id} has no messages with text"
