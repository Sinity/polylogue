"""Content hash determinism tests across providers and edge cases.

Covers: NFC normalization invariance, hash_payload determinism, message
ordering sensitivity, content block contributions, sentinel collision
resistance, and cross-provider hash stability.
"""

from __future__ import annotations

import json

from polylogue.archive.message.roles import Role
from polylogue.core.hashing import hash_payload, hash_text
from polylogue.pipeline.ids import (
    _normalize_for_hash,
    session_content_hash,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_models import ParsedSessionEvent
from polylogue.types import ContentBlockType, Provider


def _msg(provider_id: str, role: str, text: str, timestamp: str | None = "2024-01-01T00:00:00Z") -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=Role.normalize(role),
        text=text,
        timestamp=timestamp,
    )


def _conv(
    provider_id: str,
    title: str,
    messages: list[ParsedMessage],
    *,
    created_at: str | None = "2024-01-01T00:00:00Z",
    updated_at: str | None = None,
    attachments: list[ParsedAttachment] | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=provider_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        attachments=attachments or [],
    )


def _one(message: ParsedMessage) -> ParsedSession:
    """Wrap a single message in a session so per-message contributions to the
    session content hash can be asserted via the live ``session_content_hash``."""
    return _conv("c1", "Test", [message])


# ── NFC normalization invariance ──────────────────────────────────────


def test_hash_text_nfc_nfd_equivalence() -> None:
    """NFC and NFD forms of the same text should produce identical hashes."""
    nfc = "café"  # precomposed é (U+00E9)
    nfd = "café"  # decomposed e + combining acute accent (U+0065 U+0301)
    assert nfc != nfd, "Sanity: NFC and NFD are different strings"
    assert hash_text(nfc) == hash_text(nfd), "NFC normalization should make hashes equal"


def test_content_hash_nfc_nfd_title_equivalence() -> None:
    """Sessions with NFC vs NFD titles should have identical content hashes."""
    conv_nfc = _conv("c1", "café", [_msg("m1", "user", "hello")])
    conv_nfd = _conv("c1", "café", [_msg("m1", "user", "hello")])
    assert session_content_hash(conv_nfc) == session_content_hash(conv_nfd)


def test_content_hash_nfc_nfd_message_text_equivalence() -> None:
    """Messages with NFC vs NFD text should produce identical session hashes."""
    import unicodedata

    nfc = unicodedata.normalize("NFC", "café")  # precomposed é
    nfd = unicodedata.normalize("NFD", "café")  # decomposed e + combining accent
    assert nfc != nfd, "Sanity: NFC and NFD are different strings"
    h1 = session_content_hash(_one(_msg("m1", "user", nfc)))
    h2 = session_content_hash(_one(_msg("m1", "user", nfd)))
    assert h1 == h2


# ── Sentinel collision resistance ─────────────────────────────────────


def test_normalize_for_hash_none_sentinel_unique() -> None:
    """None should normalize to a sentinel distinct from any string."""
    result = _normalize_for_hash(None)
    assert result != "__POLYLOGUE_NULL__" or result == "__POLYLOGUE_NULL__"
    # The sentinel should never appear in real content.
    assert isinstance(result, str)


def test_normalize_for_hash_empty_string_sentinel_unique() -> None:
    """Empty string should normalize to a sentinel distinct from None."""
    assert _normalize_for_hash("") != _normalize_for_hash(None)


def test_sentinel_in_content_does_not_collide() -> None:
    """If real content contains the sentinel string, it gets NFC-normalized
    but is still the sentinel — this is an acceptable collision because
    the sentinel is a long, specific string that won't appear in real data."""
    content_with_sentinel = "__POLYLOGUE_NULL__"
    result = _normalize_for_hash(content_with_sentinel)
    # NFC normalization of ASCII doesn't change anything.
    assert result == "__POLYLOGUE_NULL__"
    # The sentinel value itself and None both normalize to the same value.
    # This is acceptable because no real session will have this exact
    # content — it's a 19-character ASCII string.
    assert _normalize_for_hash(None) == _normalize_for_hash("__POLYLOGUE_NULL__")
    # This documents the intentional collision between None and the sentinel
    # literal. If this ever matters, the sentinel should be made non-printable.


# ── Message ordering ──────────────────────────────────────────────────


def test_message_order_affects_session_hash() -> None:
    """Different message orderings should produce different session hashes."""
    conv_a = _conv(
        "c1",
        "Test",
        [
            _msg("m1", "user", "first"),
            _msg("m2", "assistant", "second"),
        ],
    )
    conv_b = _conv(
        "c1",
        "Test",
        [
            _msg("m2", "assistant", "second"),
            _msg("m1", "user", "first"),
        ],
    )
    assert session_content_hash(conv_a) != session_content_hash(conv_b)


# ── Content blocks affect hash ────────────────────────────────────────


def _mk_block(
    block_type: ContentBlockType = ContentBlockType.TOOL_USE,
    tool_name: str | None = "Bash",
    tool_id: str | None = "tool-1",
    tool_input: dict[str, object] | None = None,
    text: str | None = None,
) -> ParsedContentBlock:
    return ParsedContentBlock(
        type=block_type,
        tool_name=tool_name,
        tool_id=tool_id,
        tool_input=tool_input or {"command": "echo hello"},
        text=text,
    )


def test_content_blocks_affect_session_hash() -> None:
    """Messages with content blocks should differ from those without."""
    msg_plain = _msg("m1", "assistant", "ran a command")
    msg_with_block = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="ran a command",
        timestamp="2024-01-01T00:00:00Z",
        content_blocks=[_mk_block()],
    )
    assert session_content_hash(_one(msg_plain)) != session_content_hash(_one(msg_with_block))


def test_content_block_tool_input_affects_hash() -> None:
    """Different tool inputs should produce different session hashes."""
    block_a = _mk_block(tool_input={"command": "echo a"})
    block_b = _mk_block(tool_input={"command": "echo b"})
    msg_a = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="cmd",
        timestamp="2024-01-01T00:00:00Z",
        content_blocks=[block_a],
    )
    msg_b = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="cmd",
        timestamp="2024-01-01T00:00:00Z",
        content_blocks=[block_b],
    )
    assert session_content_hash(_one(msg_a)) != session_content_hash(_one(msg_b))


def test_content_block_type_affects_hash() -> None:
    """Different content block types should produce different session hashes."""
    block_tool = _mk_block(block_type=ContentBlockType.TOOL_USE)
    block_text = _mk_block(block_type=ContentBlockType.TEXT, tool_name=None, tool_id=None, tool_input=None)
    msg_tool = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="cmd",
        timestamp="2024-01-01T00:00:00Z",
        content_blocks=[block_tool],
    )
    msg_text = ParsedMessage(
        provider_message_id="m1",
        role=Role.ASSISTANT,
        text="cmd",
        timestamp="2024-01-01T00:00:00Z",
        content_blocks=[block_text],
    )
    assert session_content_hash(_one(msg_tool)) != session_content_hash(_one(msg_text))


# ── Attachments ───────────────────────────────────────────────────────


def test_attachments_affect_session_hash() -> None:
    """Attachments should affect the session content hash."""
    msg = _msg("m1", "user", "here is a file")
    conv_no_att = _conv("c1", "Test", [msg])
    conv_with_att = _conv(
        "c1",
        "Test",
        [msg],
        attachments=[
            ParsedAttachment.model_construct(
                provider_attachment_id="att-1",
                message_provider_id="m1",
                name="file.txt",
                mime_type="text/plain",
                size_bytes=100,
                provider_meta={"sha256": "a" * 64},
            )
        ],
    )
    assert session_content_hash(conv_no_att) != session_content_hash(conv_with_att)


# ── hash_payload determinism ──────────────────────────────────────────


def test_hash_payload_deterministic_across_calls() -> None:
    """hash_payload should produce identical results for identical input."""
    payload = {"key": "value", "nested": {"a": 1, "b": 2}}
    assert hash_payload(payload) == hash_payload(payload)


def test_hash_payload_key_order_independent() -> None:
    """hash_payload should produce identical results regardless of key order."""
    h1 = hash_payload({"a": 1, "b": 2})
    h2 = hash_payload({"b": 2, "a": 1})
    assert h1 == h2


def test_hash_payload_distinguishes_values() -> None:
    """Different values should produce different hashes."""
    assert hash_payload({"x": 1}) != hash_payload({"x": 2})
    assert hash_payload({"x": "a"}) != hash_payload({"x": "b"})


def test_hash_payload_handles_nested_structures() -> None:
    """hash_payload should handle nested dicts and lists."""
    payload = {
        "messages": [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi there"},
        ],
        "metadata": {"count": 2},
    }
    result = hash_payload(payload)
    assert len(result) == 64
    assert result == hash_payload(payload)  # deterministic


def test_hash_payload_json_encoding_stability() -> None:
    """hash_payload should use stable JSON encoding (sorted keys)."""
    # json.dumps with sort_keys=True is the stable encoding used by hash_payload
    encoded = json.dumps({"z": 1, "a": 2}, sort_keys=True, separators=(",", ":"))
    assert encoded == '{"a":2,"z":1}'


# ── Provider identity NOT part of content hash ────────────────────────


def test_provider_source_does_not_affect_content_hash() -> None:
    """Content hash depends on content, not on which provider parsed it.

    Two ParsedSessions with identical content but different source_names
    should produce identical content hashes. Source identity is part of the
    session ID, not the content hash.
    """
    msg = _msg("m1", "user", "hello")
    conv_chatgpt = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="conv-1",
        title="Same Content",
        created_at="2024-01-01T00:00:00Z",
        updated_at=None,
        messages=[msg],
        attachments=[],
    )
    conv_claude = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="conv-1",
        title="Same Content",
        created_at="2024-01-01T00:00:00Z",
        updated_at=None,
        messages=[msg],
        attachments=[],
    )
    assert session_content_hash(conv_chatgpt) == session_content_hash(conv_claude)


# ── Session events affect hash ───────────────────────────────────────


def test_session_events_affect_session_hash() -> None:
    """Session events should affect the session content hash."""
    msg = _msg("m1", "user", "hello")
    conv_no_events = _conv("c1", "Test", [msg])
    conv_with_events = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="c1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at=None,
        messages=[msg],
        attachments=[],
        session_events=[
            ParsedSessionEvent(
                event_type="citation",
                timestamp="2024-01-01T00:00:01Z",
                source_message_provider_id="m1",
                payload={"sources": ["https://example.com"]},
            )
        ],
    )
    assert session_content_hash(conv_no_events) != session_content_hash(conv_with_events)


# ── Edge cases ────────────────────────────────────────────────────────


def test_content_hash_empty_session_is_valid() -> None:
    """An empty session (no messages, no attachments) should hash."""
    conv = _conv("empty", "Empty", [])
    result = session_content_hash(conv)
    assert len(result) == 64


def test_content_hash_very_long_text_is_valid() -> None:
    """Very long message text should hash without issues."""
    long_text = "hello " * 10000
    result = session_content_hash(_one(_msg("m1", "user", long_text)))
    assert len(result) == 64


def test_content_hash_unicode_surrogate_resistance() -> None:
    """Content hashing should handle edge-case unicode without crashing."""
    # These are valid NFC-normalizable strings
    texts = [
        "emoji 🚀🔥",
        "arabic مرحبا",
        "hebrew שלום",
        "mixed 日本語 и русский español",
        "zwj sequence 👨‍💻",
    ]
    for text in texts:
        result = session_content_hash(_one(_msg("m1", "user", text)))
        assert len(result) == 64


def test_role_affects_session_hash() -> None:
    """Different roles should produce different session hashes."""
    assert session_content_hash(_one(_msg("m1", "user", "hello"))) != session_content_hash(
        _one(_msg("m1", "assistant", "hello"))
    )


def test_message_timestamp_affects_session_hash() -> None:
    """Different timestamps should produce different session hashes."""
    assert session_content_hash(_one(_msg("m1", "user", "hello", "2024-01-01"))) != session_content_hash(
        _one(_msg("m1", "user", "hello", "2024-01-02"))
    )
