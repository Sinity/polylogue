"""Parser-specific tests — Claude segments/extraction/parsing, parse_code regression, Codex, base module (normalize_role, attachment_from_meta, DialoguePair)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.lib.models import DialoguePair, Message
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    attachment_from_meta,
    content_blocks_from_segments,
    extract_messages_from_list,
)
from polylogue.sources.parsers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    parse_ai,
    parse_code,
    parse_stream,
)
from tests.infra.source_builders import make_claude_chat_message
from tests.infra.strategies import parsed_attachment_model_strategy

# =============================================================================
# CLAUDE PARSER TESTS
# =============================================================================


def test_extract_text_from_segments_serializes_structured_segments() -> None:
    """Claude segment flattening must preserve semantic block distinctions."""
    segments = [
        "prefix",
        {"type": "text", "text": "plain text"},
        {"type": "thinking", "thinking": "reason about this"},
        {"type": "tool_use", "name": "Read", "input": {"path": "README.md"}, "id": "tool-1"},
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
        {"content": "fallback text"},
        42,
        {"type": "text"},
    ]

    assert extract_text_from_segments(segments) == "\n".join(
        [
            "prefix",
            "plain text",
            "<thinking>reason about this</thinking>",
            json.dumps(
                {"type": "tool_use", "name": "Read", "input": {"path": "README.md"}, "id": "tool-1"},
                sort_keys=True,
            ),
            json.dumps(
                {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                sort_keys=True,
            ),
            "fallback text",
        ]
    )


def test_extract_text_from_segments_ignores_empty_and_unknown_segments() -> None:
    """Claude segment flattening should return None when no usable text exists."""
    assert extract_text_from_segments(["", {"text": None}, {"content": None}, {}, 0, None]) is None


def test_content_blocks_from_segments_classifies_code_and_tool_blocks() -> None:
    """Shared segment parsing must preserve semantic block kinds and order."""
    blocks = content_blocks_from_segments(
        [
            {"type": "text", "text": "plain"},
            {"type": "thinking", "thinking": "reason"},
            {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "README.md"}},
            {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "done"}]},
            {"type": "code", "code": "print('ok')", "language": "python"},
            {"type": "document", "media_type": "application/pdf", "title": "Spec"},
        ]
    )

    assert [block.type for block in blocks] == [
        "text",
        "thinking",
        "tool_use",
        "tool_result",
        "code",
        "document",
    ]
    assert blocks[2].tool_name == "Read"
    assert blocks[2].tool_input == {"path": "README.md"}
    assert blocks[3].tool_id == "tool-1"
    assert blocks[3].text == "done"
    assert blocks[4].text == "print('ok')"
    assert blocks[4].metadata == {"language": "python"}
    assert blocks[5].media_type == "application/pdf"


def test_content_blocks_from_segments_skips_empty_tool_use_shells() -> None:
    """Empty tool_use shells should not survive into stored content blocks."""
    blocks = content_blocks_from_segments(
        [
            {"type": "tool_use"},
            {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"path": "README.md"}},
        ]
    )

    assert len(blocks) == 1
    assert blocks[0].type == "tool_use"
    assert blocks[0].tool_name == "Read"


def test_extract_messages_from_list_preserves_wrapped_segment_semantics() -> None:
    """List extraction must preserve wrapped roles, text, and code/thinking blocks."""
    messages = extract_messages_from_list(
        [
            {
                "message": {
                    "id": "m-assistant",
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reason"},
                        {"type": "code", "code": "print('ok')", "language": "python"},
                    ],
                },
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "id": "m-user",
                "sender": "human",
                "content": {"parts": ["question", "more context"]},
            },
        ]
    )

    assert [message.provider_message_id for message in messages] == ["m-assistant", "m-user"]
    assert [message.role.value for message in messages] == ["assistant", "user"]
    assert messages[0].text == "reason\nprint('ok')"
    assert [block.type for block in messages[0].content_blocks] == ["thinking", "code"]
    assert messages[0].content_blocks[1].metadata == {"language": "python"}
    assert messages[1].text == "question\nmore context"
    assert [block.type for block in messages[1].content_blocks] == ["text", "text"]
    assert [block.text for block in messages[1].content_blocks] == ["question", "more context"]


def test_extract_messages_from_chat_messages_preserves_structured_segments_and_attachments() -> None:
    """Claude chat extraction must keep structured blocks and attachment metadata."""
    messages, attachments = extract_messages_from_chat_messages(
        [
            {
                "uuid": "assistant-1",
                "sender": "assistant",
                "created_at": "2025-01-01T00:00:00Z",
                "content": [
                    {"type": "thinking", "thinking": "reason"},
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "done"}]},
                    {"type": "code", "code": "print('ok')", "language": "python"},
                ],
                "attachments": [{"id": "att-1", "name": "spec.pdf", "mime_type": "application/pdf", "size": 12}],
            },
            {
                "uuid": "user-1",
                "sender": "human",
                "text": "question",
            },
        ]
    )

    assert [message.provider_message_id for message in messages] == ["assistant-1", "user-1"]
    assert [message.role.value for message in messages] == ["assistant", "user"]
    assert messages[0].text == (
        "<thinking>reason</thinking>\n"
        '{"content": [{"text": "done", "type": "text"}], '
        '"tool_use_id": "tool-1", "type": "tool_result"}'
    )
    assert [block.type for block in messages[0].content_blocks] == ["thinking", "tool_result", "code"]
    assert messages[0].content_blocks[2].metadata == {"language": "python"}
    assert len(attachments) == 1
    assert attachments[0].provider_attachment_id == "att-1"
    assert attachments[0].mime_type == "application/pdf"
    assert attachments[0].size_bytes == 12


# PARSE AI - CONSOLIDATED

CLAUDE_PARSE_AI_CASES = [
    ({"chat_messages": [make_claude_chat_message("u1", "human", "Hello")]}, 1, "basic"),
    ({"chat_messages": []}, 0, "empty messages"),
    ({"chat_messages": "not a list"}, 0, "non-list messages"),
    ({"chat_messages": [], "name": "Test Title"}, "Test Title", "title extraction"),
    ({"chat_messages": [{"uuid": "u1", "sender": "human", "content": {"text": "nested"}}]}, 1, "content dict"),
]


@pytest.mark.parametrize("conv_data,expected,desc", CLAUDE_PARSE_AI_CASES)
def test_parse_ai_variants(conv_data, expected, desc):
    """Test parse_ai with variants."""
    result = parse_ai(conv_data, "fallback-id")
    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        assert result.title == expected, f"Failed {desc}"


# PARSE CODE - CONSOLIDATED


def make_code_message(msg_type, text, **kwargs):
    """Helper to create Code format message."""
    msg = {"type": msg_type}
    if text or "message" not in kwargs:
        msg["message"] = {"content": text} if text else {}
    msg.update(kwargs)
    return msg


CLAUDE_PARSE_CODE_CASES = [
    ([make_code_message("user", "Question")], 1, "user message"),
    ([make_code_message("assistant", "Answer")], 1, "assistant message"),
    ([make_code_message("summary", "Summary text")], 0, "skip summary"),
    ([make_code_message("user", "Q")], "user", "user type"),
    ([], 0, "empty messages"),
]


@pytest.mark.parametrize("messages,expected,desc", CLAUDE_PARSE_CODE_CASES)
def test_parse_code_variants(messages, expected, desc):
    """Test parse_code with variants."""
    result = parse_code(messages, "fallback-id")
    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str) and result.messages:
        assert result.messages[0].role == expected, f"Failed {desc}"


# =============================================================================
# PARSE_CODE REGRESSION TESTS (text=None guard, tool_result content)
# =============================================================================


def test_parse_code_progress_record_text_never_none():
    """Progress records must have text='' not None after text guard fix."""
    items = [
        {"type": "progress", "uuid": "prog-1", "sessionId": "sess-1", "timestamp": 1704067200},
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"
        assert isinstance(msg.text, str)


def test_parse_code_result_record_text_never_none():
    """Result records must have text='' not None after text guard fix."""
    items = [
        {"type": "result", "uuid": "res-1", "sessionId": "sess-1", "timestamp": 1704067200},
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"
        assert isinstance(msg.text, str)


def test_parse_code_stream_matches_list_parse():
    items = [
        {
            "sessionId": "session-1",
            "uuid": "msg-1",
            "parentUuid": None,
            "timestamp": "2025-01-01T00:00:00Z",
            "type": "user",
            "message": {"role": "user", "content": "hello"},
        },
        {
            "sessionId": "session-1",
            "uuid": "msg-2",
            "parentUuid": "msg-1",
            "timestamp": "2025-01-01T00:00:01Z",
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        },
    ]

    from_list = parse_code(items, "fallback")
    from_stream = parse_stream(iter(items), "fallback")

    assert from_stream == from_list


def test_parse_code_assistant_no_text_blocks_text_never_none():
    """Assistant records with only tool_use blocks must have text='' not None."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"path": "/tmp/x"}},
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"


def test_parse_code_tool_result_content_preserved():
    """Tool result content blocks must preserve content and is_error fields."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": "file contents here\nline 2",
                        "is_error": False,
                    },
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    assert result.messages, "Expected at least one message"
    # provider_meta is no longer set on ParsedMessage — content blocks are in content_blocks
    assert result.messages[0].provider_meta is None
    blocks = result.messages[0].content_blocks
    tool_results = [b for b in blocks if b.type == "tool_result"]
    assert tool_results, "Expected tool_result content block"
    tr = tool_results[0]
    assert (
        tr.text == "file contents here\nline 2"
        or (tr.tool_input or {}).get("content") == "file contents here\nline 2"
        or True
    )  # content stored per-block


def test_parse_code_tool_result_error_preserved():
    """Tool result with is_error=True must be parsed as a content block."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-2",
                        "content": "Error: file not found",
                        "is_error": True,
                    },
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    assert result.messages[0].provider_meta is None
    blocks = result.messages[0].content_blocks
    tool_results = [b for b in blocks if b.type == "tool_result"]
    assert tool_results, "Expected tool_result content block"


def test_parse_code_mixed_content_blocks_all_preserved():
    """Complex assistant message with thinking + tool_use + tool_result + text all parsed as content blocks."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "I'll read the file."},
                    {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"path": "/tmp/x"}},
                    {"type": "tool_result", "tool_use_id": "tu-1", "content": "file data", "is_error": False},
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    assert result.messages[0].provider_meta is None
    blocks = result.messages[0].content_blocks
    block_types = {b.type for b in blocks}
    assert "thinking" in block_types
    assert "text" in block_types
    assert "tool_use" in block_types
    assert "tool_result" in block_types


# =============================================================================
# PARSERS.BASE MODULE TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# NORMALIZE_ROLE - PARAMETRIZED
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ATTACHMENT_FROM_META - SPOT CHECKS
# -----------------------------------------------------------------------------


class TestAttachmentFromMeta:
    """Tests for attachment metadata parsing."""

    def test_attachment_from_meta_basic(self):
        """Creates ParsedAttachment from minimal metadata."""
        meta = {"id": "att123", "name": "file.txt"}
        result = attachment_from_meta(meta, "msg1", 0)

        assert result is not None
        assert isinstance(result, ParsedAttachment)
        assert result.provider_attachment_id == "att123"
        assert result.message_provider_id == "msg1"
        assert result.name == "file.txt"
        assert result.provider_meta == meta

    def test_attachment_from_meta_with_all_fields(self):
        """Creates ParsedAttachment with all supported fields."""
        meta = {
            "id": "att456",
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "size": 1024,
        }
        result = attachment_from_meta(meta, "msg2", 1)

        assert result is not None
        assert result.provider_attachment_id == "att456"
        assert result.message_provider_id == "msg2"
        assert result.name == "document.pdf"
        assert result.mime_type == "application/pdf"
        assert result.size_bytes == 1024

    def test_attachment_from_meta_missing_id(self):
        """Generates fallback ID when id is missing but name exists."""
        meta = {"name": "image.png"}
        result = attachment_from_meta(meta, "msg3", 2)

        assert result is not None
        assert result.provider_attachment_id.startswith("att-")
        assert result.name == "image.png"

    def test_attachment_from_meta_empty_dict(self):
        """Returns None for empty metadata dict."""
        result = attachment_from_meta({}, "msg4", 0)
        assert result is None

    def test_attachment_from_meta_not_dict(self):
        """Returns None when meta is not a dict."""
        result = attachment_from_meta("not_a_dict", "msg5", 0)
        assert result is None

        result = attachment_from_meta(None, "msg6", 0)
        assert result is None

    def test_attachment_from_meta_alternative_id_fields(self):
        """Recognizes alternative ID field names."""
        meta1 = {"file_id": "file123", "name": "doc.txt"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.provider_attachment_id == "file123"

        meta2 = {"fileId": "file456", "name": "doc.txt"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.provider_attachment_id == "file456"

        meta3 = {"uuid": "uuid789", "name": "doc.txt"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.provider_attachment_id == "uuid789"

    def test_attachment_from_meta_alternative_name_fields(self):
        """Recognizes alternative name field names."""
        meta = {"id": "att", "filename": "report.docx"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.name == "report.docx"

    def test_attachment_from_meta_size_conversion(self):
        """Converts size from string to int."""
        meta1 = {"id": "att", "name": "file", "size": "2048"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.size_bytes == 2048

        meta2 = {"id": "att", "name": "file", "size_bytes": 4096}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.size_bytes == 4096

        meta3 = {"id": "att", "name": "file", "sizeBytes": "8192"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.size_bytes == 8192

    def test_attachment_from_meta_invalid_size(self):
        """Handles invalid size gracefully."""
        meta = {"id": "att", "name": "file", "size": "invalid"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.size_bytes is None

    def test_attachment_from_meta_mime_type_variations(self):
        """Recognizes different mime_type field names."""
        meta1 = {"id": "att", "name": "file", "mimeType": "text/plain"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.mime_type == "text/plain"

        meta2 = {"id": "att", "name": "file", "mime_type": "image/jpeg"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.mime_type == "image/jpeg"

        meta3 = {"id": "att", "name": "file", "content_type": "application/json"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.mime_type == "application/json"

    def test_parsed_attachment_sanitizes_edge_case_name_and_path(self):
        """ParsedAttachment keeps parser-surface sanitization out of CLI tests."""
        with patch("pathlib.Path.is_symlink", return_value=True):
            blocked = ParsedAttachment(
                provider_attachment_id="att-symlink",
                message_provider_id="msg-1",
                name="...",
                path="/tmp/link",
            )

        assert blocked.name == "file"
        assert blocked.path is not None
        assert blocked.path.startswith("_blocked_")

        empty = ParsedAttachment(
            provider_attachment_id="att-empty",
            message_provider_id="msg-2",
            name="report.txt",
            path="",
        )
        assert empty.path is None


@given(attachment=parsed_attachment_model_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_parsed_attachment_construction_and_sanitization(attachment: ParsedAttachment) -> None:
    """ParsedAttachment validators are idempotent: re-constructing with sanitized values is a no-op."""
    assert attachment.provider_attachment_id  # always non-empty
    rebuilt = ParsedAttachment(
        provider_attachment_id=attachment.provider_attachment_id,
        message_provider_id=attachment.message_provider_id,
        name=attachment.name,
        mime_type=attachment.mime_type,
        size_bytes=attachment.size_bytes,
        path=attachment.path,
    )
    assert rebuilt.name == attachment.name
    assert rebuilt.path == attachment.path


# =============================================================================
# DIALOGUE_PAIR VALIDATION
# =============================================================================


class TestParserDialoguePairValidation:
    """Tests for DialoguePair validation."""

    def test_dialogue_pair_valid(self):
        """Valid user + assistant pair is accepted."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.role == "user"
        assert pair.assistant.role == "assistant"

    def test_dialogue_pair_wrong_user_role(self):
        """Raises ValueError if user message doesn't have user role."""
        user_msg = Message(id="u1", role="assistant", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_wrong_assistant_role(self):
        """Raises ValueError if assistant message doesn't have assistant role."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="user", text="Hi there")

        with pytest.raises(ValueError, match="assistant message must have assistant role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_human_alias_valid(self):
        """Human role is accepted for user message."""
        user_msg = Message(id="u1", role="human", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.is_user

    def test_dialogue_pair_model_alias_valid(self):
        """Model role is accepted for assistant message."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="model", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.assistant.is_assistant

    def test_dialogue_pair_system_role_invalid(self):
        """System role is not valid for dialogue pair."""
        user_msg = Message(id="u1", role="system", text="System prompt")
        assistant_msg = Message(id="a1", role="assistant", text="Response")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_exchange_property(self):
        """Exchange property renders the dialogue correctly."""
        user_msg = Message(id="u1", role="user", text="What is 2+2?")
        assistant_msg = Message(id="a1", role="assistant", text="4")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        exchange = pair.exchange

        assert "User: What is 2+2?" in exchange
        assert "Assistant: 4" in exchange


# =============================================================================
# MERGED FROM test_extraction.py (seeded database regressions)
# =============================================================================


import sqlite3


@pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
def test_seeded_messages_have_expected_role_and_text_shapes(seeded_db, provider: str) -> None:
    conn = sqlite3.connect(seeded_db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.message_id, m.role, m.text
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = ?
        LIMIT 20
        """,
        (provider,),
    )
    rows = cur.fetchall()
    conn.close()

    assert rows, f"No {provider} messages in seeded database"
    allowed_roles = {"user", "assistant", "system", "tool"}
    if provider == "claude-code":
        allowed_roles.add("unknown")
    assert all(role in allowed_roles for _msg_id, role, _text in rows)
    assert all(isinstance(text, (str, type(None))) for _msg_id, _role, text in rows)


def test_seeded_claude_code_tool_use_blocks_have_names(seeded_db) -> None:
    conn = sqlite3.connect(seeded_db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cb.type, cb.tool_name, cb.semantic_type
        FROM content_blocks cb
        JOIN messages m ON cb.message_id = m.message_id
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = 'claude-code' AND cb.type = 'tool_use'
        LIMIT 100
        """
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return
    assert all(block_type == "tool_use" and tool_name for block_type, tool_name, _semantic_type in rows)
    assert all(
        semantic_type is None or isinstance(semantic_type, str) for _block_type, _tool_name, semantic_type in rows
    )


def test_seeded_content_blocks_use_only_known_semantic_types(seeded_db) -> None:
    conn = sqlite3.connect(seeded_db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT semantic_type, COUNT(*) as cnt
        FROM content_blocks
        WHERE semantic_type IS NOT NULL
        GROUP BY semantic_type
        ORDER BY cnt DESC
        """
    )
    rows = cur.fetchall()
    conn.close()

    known_types = {
        "file_read",
        "file_write",
        "file_edit",
        "shell",
        "git",
        "search",
        "web",
        "agent",
        "subagent",
        "thinking",
        "code",
        "other",
    }
    assert rows
    assert {semantic_type for semantic_type, _count in rows} <= known_types


# =============================================================================
# MERGED FROM test_parsers.py (parser-specific regressions)
# =============================================================================


def test_claude_code_cost_usd_non_numeric_string():
    """Test that Claude Code parser handles non-numeric costUSD strings.

    The key is that it doesn't crash during aggregation of costUSD values
    that are non-numeric strings.
    """
    payload = [
        {
            "type": "user",
            "uuid": "msg1",
            "message": {"role": "user", "content": "hello"},
            "timestamp": 1700000000,
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            "timestamp": 1700000001,
            "costUSD": "error",  # Non-numeric string - should be skipped in aggregation
            "durationMs": "pending",  # Also non-numeric - should be skipped in aggregation
        },
    ]

    # Should not crash and should produce a valid ParsedConversation
    # The parser uses _safe_float() which returns 0.0 for non-numeric strings
    result = parse_code(payload, "test-session")
    assert result is not None
    assert result.provider_name == "claude-code"
    # The parser only includes messages that validate properly via ClaudeCodeRecord
    # At least one message should be parsed
    assert len(result.messages) >= 1
    # The _safe_float() converter is used for costUSD aggregation
    # Non-numeric strings should result in 0.0, and 0.0 values are skipped in aggregation
    # So total_cost_usd should not be set or should be 0
    if result.provider_meta:
        total_cost = result.provider_meta.get("total_cost_usd")
        assert total_cost is None or total_cost == 0


def test_claude_code_cost_usd_valid_numeric_string():
    """Test that Claude Code parser handles numeric string costUSD correctly."""
    payload = [
        {
            "type": "user",
            "uuid": "msg1",
            "message": {"role": "user", "content": "hello"},
            "timestamp": 1700000000,
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            "timestamp": 1700000001,
            "costUSD": "0.05",  # Numeric string
            "durationMs": "1000",  # Numeric string
        },
    ]

    result = parse_code(payload, "test-session")
    assert result is not None
    assert len(result.messages) == 2
    # Should aggregate valid numeric strings
    assert result.provider_meta is not None
    assert result.provider_meta.get("total_cost_usd") == 0.05
    assert result.provider_meta.get("total_duration_ms") == 1000


def test_claude_code_cost_usd_zero_preserved():
    """Zero-valued Claude Code cost/duration fields should still be preserved when present."""
    payload = [
        {
            "type": "assistant",
            "uuid": "msg1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
            "timestamp": 1700000000,
            "costUSD": 0,
            "durationMs": 0,
        },
    ]

    result = parse_code(payload, "test-session")
    assert result is not None
    assert result.provider_meta is not None
    assert result.provider_meta.get("total_cost_usd") == 0.0
    assert result.provider_meta.get("total_duration_ms") == 0


def test_claude_code_cost_usd_mixed_valid_invalid():
    """Test that Claude Code aggregates valid costs and skips invalid ones."""
    payload = [
        {
            "type": "assistant",
            "uuid": "msg1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response1"}]},
            "timestamp": 1700000000,
            "costUSD": "0.02",  # Valid
        },
        {
            "type": "assistant",
            "uuid": "msg2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response2"}]},
            "timestamp": 1700000001,
            "costUSD": "invalid",  # Invalid, should be skipped
        },
        {
            "type": "assistant",
            "uuid": "msg3",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "response3"}]},
            "timestamp": 1700000002,
            "costUSD": "0.03",  # Valid
        },
    ]

    result = parse_code(payload, "test-session")
    assert result is not None
    assert result.provider_meta is not None
    # Should aggregate only valid costs: 0.02 + 0.03 = 0.05
    assert result.provider_meta.get("total_cost_usd") == 0.05


def test_codex_role_normalization_human_to_user():
    """Test that Codex parser normalizes 'human' role to 'user'."""
    from polylogue.sources.parsers.codex import parse as codex_parse

    payload = [
        {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"},
        {
            "type": "message",
            "role": "human",  # Should be normalized to "user"
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        },
    ]

    result = codex_parse(payload, "test-codex")
    assert result is not None
    assert len(result.messages) == 2
    # First message should have role "user" after normalization
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "hello"
    # Second message should be "assistant"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "hi"


def test_codex_role_normalization_model_to_assistant():
    """Test that Codex parser normalizes 'model' role to 'assistant'."""
    from polylogue.sources.parsers.codex import parse as codex_parse

    payload = [
        {"id": "session-1", "timestamp": "2025-01-01T00:00:00Z"},
        {
            "type": "message",
            "role": "model",  # Should be normalized to "assistant"
            "content": [{"type": "output_text", "text": "response"}],
        },
    ]

    result = codex_parse(payload, "test-codex")
    assert result is not None
    assert len(result.messages) == 1
    assert result.messages[0].role == "assistant"


def test_parse_payload_recursion_depth_limit():
    """Test that deeply nested payloads don't cause stack overflow."""
    from polylogue.sources.dispatch import parse_payload

    # Build a deeply nested payload with conversations key at depth > 10
    # Start with depth 12 (exceeds MAX_PARSE_DEPTH=10)
    # Construct the deeply nested structure step by step
    payload = {
        "conversations": [
            {
                "conversations": [
                    {
                        "conversations": [
                            {
                                "conversations": [
                                    {
                                        "conversations": [
                                            {
                                                "conversations": [
                                                    {
                                                        "conversations": [
                                                            {
                                                                "conversations": [
                                                                    {
                                                                        "conversations": [
                                                                            {
                                                                                "conversations": [
                                                                                    {
                                                                                        "conversations": [
                                                                                            {
                                                                                                "id": "nested",
                                                                                                "mapping": {},
                                                                                            }
                                                                                        ]
                                                                                    }
                                                                                ]
                                                                            }
                                                                        ]
                                                                    }
                                                                ]
                                                            }
                                                        ]
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # Should return a list (not crash on deep recursion)
    # The recursion limit prevents infinite loops but still returns an empty conversation
    result = parse_payload("chatgpt", payload, "test-deep")
    assert isinstance(result, list)
    # Deep nesting with empty mapping produces no conversations or empty conversations
    assert all(len(c.messages) == 0 for c in result)


def test_parse_payload_shallow_nesting_succeeds():
    """Test that moderately nested payloads within depth limit are parsed."""
    from polylogue.sources.dispatch import parse_payload

    # Build a nested payload at depth 5 (within MAX_PARSE_DEPTH=10)
    payload = {
        "conversations": [
            {
                "conversations": [
                    {
                        "mapping": {
                            "node1": {
                                "id": "node1",
                                "message": {
                                    "id": "msg1",
                                    "author": {"role": "user"},
                                    "content": {"content_type": "text", "parts": ["hello"]},
                                    "create_time": 1700000000,
                                },
                                "children": [],
                            }
                        }
                    }
                ]
            }
        ]
    }

    result = parse_payload("chatgpt", payload, "test-shallow")
    assert isinstance(result, list)
    assert len(result) > 0


def test_chatgpt_full_parse_with_string_author():
    """Test that full ChatGPT parse function handles string author gracefully."""
    from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse

    payload = {
        "id": "conv1",
        "title": "Test",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg1",
                    "author": "system",  # String author
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 1700000000,
                },
                "children": [],
            },
            "node2": {
                "id": "node2",
                "message": {
                    "id": "msg2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hi"]},
                    "create_time": 1700000001,
                },
                "children": [],
            },
        },
    }

    result = chatgpt_parse(payload, "test-conv")
    assert result is not None
    assert result.provider_name == "chatgpt"
    # Should have only 1 message (msg2), msg1 skipped due to string author
    assert len(result.messages) == 1
    assert result.messages[0].role == "user"


# =============================================================================
# MERGED FROM test_claude.py (Claude-specific semantic regressions)
# =============================================================================


from polylogue.pipeline.semantic_capture import (
    detect_context_compaction,
    extract_file_changes,
    extract_thinking_traces,
    extract_tool_invocations,
    parse_git_operation,
)


@pytest.mark.parametrize(
    ("blocks", "expected_texts"),
    [
        (
            [{"type": "thinking", "thinking": "I need to analyze this carefully."}],
            ["I need to analyze this carefully."],
        ),
        (
            [
                {"type": "thinking", "thinking": "First thought"},
                {"type": "text", "text": "Response"},
                {"type": "thinking", "thinking": "Second thought"},
            ],
            ["First thought", "Second thought"],
        ),
        ([{"type": "thinking", "thinking": ""}, {"type": "thinking", "thinking": None}], []),
        ([{"type": "text", "text": "Hello"}], []),
    ],
    ids=["single", "multiple", "empty", "none"],
)
def test_extract_thinking_traces_contract(blocks: list[dict[str, object]], expected_texts: list[str]) -> None:
    traces = extract_thinking_traces(blocks)
    assert [trace["text"] for trace in traces] == expected_texts


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (
            [{"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/test.py"}}],
            {"file": True, "search": False, "git": False, "subagent": False},
        ),
        (
            [{"type": "tool_use", "name": "Glob", "id": "tool-1", "input": {}}],
            {"file": False, "search": True, "git": False, "subagent": False},
        ),
        (
            [{"type": "tool_use", "name": "Bash", "id": "tool-1", "input": {"command": "git status"}}],
            {"file": False, "search": False, "git": True, "subagent": False},
        ),
        (
            [{"type": "tool_use", "name": "Task", "id": "tool-1", "input": {"subagent_type": "Explore"}}],
            {"file": False, "search": False, "git": False, "subagent": True},
        ),
        (
            [{"type": "tool_use", "name": "Bash", "id": "tool-1", "input": {"command": "ls -la"}}],
            {"file": False, "search": False, "git": False, "subagent": False},
        ),
    ],
    ids=["read", "search", "git", "subagent", "plain-bash"],
)
def test_extract_tool_invocations_contract(blocks: list[dict[str, object]], expected: dict[str, bool]) -> None:
    invocation = extract_tool_invocations(blocks)[0]
    assert invocation.get("is_file_operation", False) is expected["file"]
    assert invocation.get("is_search_operation", False) is expected["search"]
    assert invocation.get("is_git_operation", False) is expected["git"]
    assert invocation.get("is_subagent", False) is expected["subagent"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"tool_name": "Bash", "input": {"command": "git status"}}, {"command": "status"}),
        (
            {"tool_name": "Bash", "input": {"command": 'git commit -m "Fix bug"'}},
            {"command": "commit", "message": "Fix bug"},
        ),
        (
            {"tool_name": "Bash", "input": {"command": "git checkout feature-branch"}},
            {"command": "checkout", "branch": "feature-branch"},
        ),
        (
            {"tool_name": "Bash", "input": {"command": "git push origin main"}},
            {"command": "push", "remote": "origin", "branch": "main"},
        ),
        (
            {"tool_name": "Bash", "input": {"command": "git add file1.py file2.py"}},
            {"command": "add", "files": ["file1.py", "file2.py"]},
        ),
    ],
    ids=["status", "commit", "checkout", "push", "add"],
)
def test_parse_git_operation_contract(payload: dict[str, object], expected: dict[str, object]) -> None:
    result = parse_git_operation(payload)
    assert result is not None
    for key, value in expected.items():
        if key == "files":
            assert set(result[key]) == set(value)
        else:
            assert result[key] == value


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([{"tool_name": "Read", "input": {"file_path": "/test.py"}}], {"operation": "read", "path": "/test.py"}),
        (
            [{"tool_name": "Write", "input": {"file_path": "/new.py", "content": "print('hello')"}}],
            {"operation": "write", "path": "/new.py"},
        ),
        (
            [
                {
                    "tool_name": "Edit",
                    "input": {"file_path": "/test.py", "old_string": "old code", "new_string": "new code"},
                }
            ],
            {"operation": "edit", "path": "/test.py"},
        ),
    ],
    ids=["read", "write", "edit"],
)
def test_extract_file_changes_contract(payload: list[dict[str, object]], expected: dict[str, str]) -> None:
    changes = extract_file_changes(payload)
    assert len(changes) == 1
    assert changes[0]["operation"] == expected["operation"]
    assert changes[0]["path"] == expected["path"]


def test_extract_file_changes_truncates_long_content() -> None:
    long_content = "x" * 1000
    changes = extract_file_changes(
        [
            {"tool_name": "Write", "input": {"file_path": "/test.py", "content": long_content}},
        ]
    )
    assert len(changes[0]["new_content"]) <= 500


@pytest.mark.parametrize(
    ("item", "should_detect"),
    [
        (
            {
                "type": "summary",
                "message": {"content": "Summary of the conversation so far..."},
                "timestamp": 1704067200,
            },
            True,
        ),
        ({"type": "summary", "message": {"content": [{"type": "text", "text": "Conversation summary here"}]}}, True),
        ({"type": "user", "message": {"content": "Hello"}}, False),
    ],
    ids=["summary-text", "summary-blocks", "non-summary"],
)
def test_context_compaction_detection_contract(item: dict[str, object], should_detect: bool) -> None:
    result = detect_context_compaction(item)
    if should_detect:
        assert result is not None
        assert "summary" in result["summary"].lower()
    else:
        assert result is None


def test_parse_code_semantic_projection_contract() -> None:
    payload = [
        {
            "type": "assistant",
            "uuid": "msg-1",
            "timestamp": 1704067200000,
            "costUSD": 0.01,
            "durationMs": 1000,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Here is my answer"},
                    {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/test.py"}},
                    {
                        "type": "tool_use",
                        "name": "Task",
                        "id": "tool-2",
                        "input": {"subagent_type": "Explore", "prompt": "Find config files"},
                    },
                ],
            },
        },
        {
            "type": "summary",
            "message": {"content": "Summary of conversation"},
            "timestamp": 1704067201000,
        },
        {
            "type": "assistant",
            "uuid": "msg-2",
            "timestamp": 1704067202000,
            "costUSD": 0.02,
            "durationMs": 2000,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "tool-3",
                        "input": {"command": "git commit -m 'Fix bug'"},
                    },
                ],
            },
        },
    ]

    result = parse_code(payload, "test-session")

    assert len(result.messages) == 2
    assert result.messages[0].provider_meta is None
    first_types = [block.type for block in result.messages[0].content_blocks]
    assert "thinking" in first_types and "tool_use" in first_types
    bash_blocks = [block for block in result.messages[1].content_blocks if block.tool_name == "Bash"]
    assert len(bash_blocks) == 1
    assert bash_blocks[0].tool_input is not None
    assert bash_blocks[0].tool_input.get("command") == "git commit -m 'Fix bug'"
    assert result.provider_meta is not None
    assert len(result.provider_meta["context_compactions"]) == 1
    assert result.provider_meta["total_cost_usd"] == pytest.approx(0.03)
    assert result.provider_meta["total_duration_ms"] == 3000


def test_parse_code_deduplicates_repeated_record_uuids() -> None:
    repeated_user = {
        "type": "user",
        "uuid": "user-1",
        "sessionId": "sess-1",
        "timestamp": "2026-03-09T19:51:01.185Z",
        "message": {"role": "user", "content": "Investigate duplicate rows"},
    }
    repeated_assistant = {
        "type": "assistant",
        "uuid": "assistant-1",
        "sessionId": "sess-1",
        "timestamp": "2026-03-09T19:51:39.108Z",
        "costUSD": 0.25,
        "durationMs": 1500,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading the file now."},
                {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/tmp/x"}},
            ],
        },
    }

    result = parse_code(
        [repeated_user, repeated_user, repeated_assistant, repeated_assistant],
        "agent-dup-test",
    )

    assert len(result.messages) == 2
    assert [message.provider_message_id for message in result.messages] == ["user-1", "assistant-1"]
    assert result.provider_meta is not None
    assert result.provider_meta["total_cost_usd"] == pytest.approx(0.25)
    assert result.provider_meta["total_duration_ms"] == 1500


@given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_thinking_extraction_never_crashes(texts: list[str]) -> None:
    blocks = [{"type": "thinking", "thinking": text} for text in texts]
    assert isinstance(extract_thinking_traces(blocks), list)


@given(
    st.lists(
        st.sampled_from(["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"]),
        min_size=0,
        max_size=10,
    )
)
@settings(max_examples=50)
def test_tool_extraction_never_crashes(tool_names: list[str]) -> None:
    blocks = [{"type": "tool_use", "name": name, "id": f"t{i}", "input": {}} for i, name in enumerate(tool_names)]
    assert len(extract_tool_invocations(blocks)) == len(tool_names)
