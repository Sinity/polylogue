"""Parser-specific tests — Claude segments/extraction/parsing, parse_code regression, Codex, base module (normalize_role, attachment_from_meta, DialoguePair)."""

from __future__ import annotations

import pytest

from polylogue.lib.models import DialoguePair, Message
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    attachment_from_meta,
    normalize_role,
)
from polylogue.sources.parsers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    parse_ai,
    parse_code,
)
from polylogue.sources.parsers.codex import looks_like as codex_looks_like
from polylogue.sources.parsers.codex import parse as codex_parse
from tests.infra.helpers import make_claude_chat_message

# =============================================================================
# CLAUDE PARSER TESTS
# =============================================================================

CLAUDE_SEGMENT_CASES = [
    (["plain text"], "plain text", "string segment"),
    ([{"text": "dict with text"}], "dict with text", "dict with text"),
    ([{"content": "dict with content"}], "dict with content", "dict with content"),
    (["text1", {"text": "text2"}, "text3"], "text1", "mixed segments"),
    ([{}, "", None], None, "empty/None segments"),
]

@pytest.mark.parametrize("segments,expected_contains,desc", CLAUDE_SEGMENT_CASES)
def test_extract_text_from_segments(segments, expected_contains, desc):
    """Test segment extraction variants."""
    result = extract_text_from_segments(segments)
    if expected_contains:
        assert expected_contains in (result or ""), f"Failed {desc}"
    else:
        assert result is None or result == "", f"Failed {desc}"


CLAUDE_EXTRACT_CHAT_MESSAGES_CASES = [
    # Basic
    ([make_claude_chat_message("u1", "human", "Hello")], 1, "basic message"),

    # Attachments variants
    ([make_claude_chat_message("u1", "human", "Hi", attachments=[{"file_name": "doc.pdf"}])], 1, "attachments field"),
    ([make_claude_chat_message("u1", "human", "Hi", files=[{"file_name": "doc.pdf"}])], 1, "files field"),

    # Role variants
    ([make_claude_chat_message("u1", "human", "Hi")], "user", "human role"),
    ([make_claude_chat_message("u1", "assistant", "Hi")], "assistant", "assistant role"),
    ([make_claude_chat_message("u1", None, "Hi")], 0, "missing sender skipped"),

    # Timestamp variants (with role)
    ([make_claude_chat_message("u1", "human", "Hi", timestamp="2024-01-01T00:00:00Z")], 1, "created_at"),
    ([{"uuid": "u1", "sender": "human", "text": "Hi", "create_time": 1704067200}], 1, "create_time"),
    ([{"uuid": "u1", "sender": "human", "text": "Hi", "timestamp": 1704067200}], 1, "timestamp field"),

    # ID variants (with role)
    ([{"uuid": "u1", "sender": "human", "text": "Hi"}], "u1", "uuid field"),
    ([{"id": "i1", "sender": "human", "text": "Hi"}], "i1", "id field"),
    ([{"message_id": "m1", "sender": "human", "text": "Hi"}], "m1", "message_id field"),

    # Content variants (with role)
    ([{"uuid": "u1", "sender": "human", "text": ["list", "of", "parts"]}], 0, "text as list skipped"),
    ([{"uuid": "u1", "sender": "human", "content": {"text": "nested text"}}], 1, "content dict with text"),
    ([{"uuid": "u1", "sender": "human", "content": {"parts": ["part1", "part2"]}}], 1, "content dict with parts"),

    # Missing text (with role)
    ([{"uuid": "u1", "sender": "human"}], 0, "missing text skipped"),
    ([{"uuid": "u1", "sender": "human", "text": ""}], 0, "empty text skipped"),
    ([{"uuid": "u1", "sender": "human", "text": None}], 0, "None text skipped"),

    # Non-dict items (valid one has role)
    (["not a dict", {"uuid": "u1", "sender": "human", "text": "Valid"}], 1, "skip non-dict"),

    # Empty list
    ([], 0, "empty list"),
]


@pytest.mark.parametrize("chat_messages,expected,desc", CLAUDE_EXTRACT_CHAT_MESSAGES_CASES)
def test_claude_extract_chat_messages_comprehensive(chat_messages, expected, desc):
    """Comprehensive chat_messages extraction.

    Replaces 15 extraction tests.
    """
    messages, attachments = extract_messages_from_chat_messages(chat_messages)

    if isinstance(expected, int):
        assert len(messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str) and messages:
        # ID field tests check provider_message_id, role tests check role
        if "field" in desc and desc not in ["attachments field", "files field", "timestamp field"]:
            assert messages[0].provider_message_id == expected, f"Failed {desc}"
        else:
            # Expected role
            assert messages[0].role == expected, f"Failed {desc}"


# PARSE AI - CONSOLIDATED

CLAUDE_PARSE_AI_CASES = [
    ({"chat_messages": [make_claude_chat_message("u1", "human", "Hello")]}, 1, "basic"),
    ({"chat_messages": []}, 0, "empty messages"),
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
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    assert tool_results, "Expected tool_result in content_blocks"
    tr = tool_results[0]
    assert "content" in tr, "tool_result must have content field"
    assert tr["content"] == "file contents here\nline 2"
    assert "is_error" in tr, "tool_result must have is_error field"
    assert tr["is_error"] is False


def test_parse_code_tool_result_error_preserved():
    """Tool result with is_error=True must preserve the flag."""
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
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    assert tool_results[0]["is_error"] is True
    assert tool_results[0]["content"] == "Error: file not found"


def test_parse_code_mixed_content_blocks_all_preserved():
    """Complex assistant message with thinking + tool_use + tool_result + text all preserved."""
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
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    types = [b["type"] for b in blocks]
    assert "thinking" in types
    assert "text" in types
    assert "tool_use" in types
    assert "tool_result" in types
    # Verify tool_result has all fields
    tr = next(b for b in blocks if b["type"] == "tool_result")
    assert tr["content"] == "file data"
    assert tr["is_error"] is False
    assert tr["tool_use_id"] == "tu-1"


# =============================================================================
# CODEX PARSER TESTS
# =============================================================================


def test_codex_looks_like_envelope_format():
    """Test looks_like returns True for envelope format."""
    valid_payload = [
        {"type": "session_meta", "payload": {"id": "test-123", "timestamp": "2025-01-01"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
    ]
    assert codex_looks_like(valid_payload) is True


def test_codex_looks_like_intermediate_format():
    """Test looks_like returns True for intermediate format."""
    valid_payload = [
        {"id": "test-123", "timestamp": "2025-01-01", "git": {}},
        {"type": "message", "role": "user", "content": []},
    ]
    assert codex_looks_like(valid_payload) is True


def test_codex_looks_like_empty_list():
    """Test looks_like returns False for empty list."""
    assert codex_looks_like([]) is False


def test_codex_looks_like_not_list():
    """Test looks_like returns False for non-list types."""
    assert codex_looks_like({}) is False
    assert codex_looks_like("string") is False
    assert codex_looks_like(None) is False
    assert codex_looks_like(123) is False


def test_codex_parse_empty_list():
    """Test parsing an empty list returns conversation with no messages."""
    result = codex_parse([], fallback_id="test-empty-list")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "test-empty-list"
    assert len(result.messages) == 0


def test_codex_parse_envelope_format():
    """Test parsing envelope format with session_meta and response_item."""
    payload = [
        {"type": "session_meta", "payload": {"id": "session-123", "timestamp": "2025-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "timestamp": "2025-01-01T00:00:01Z",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Hi there!"}],
                "timestamp": "2025-01-01T00:00:02Z",
            },
        },
    ]
    result = codex_parse(payload, fallback_id="fallback-id")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "session-123"
    assert result.created_at == "2025-01-01T00:00:00Z"
    assert len(result.messages) == 2

    assert result.messages[0].provider_message_id == "msg-1"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Hello"
    assert result.messages[0].timestamp == "2025-01-01T00:00:01Z"

    assert result.messages[1].provider_message_id == "msg-2"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "Hi there!"


def test_codex_parse_intermediate_format():
    """Test parsing intermediate format with direct records."""
    payload = [
        {"id": "session-456", "timestamp": "2025-01-02T00:00:00Z", "git": {}},
        {"record_type": "state"},
        {
            "type": "message",
            "id": "msg-3",
            "role": "user",
            "content": [{"type": "input_text", "text": "Test message"}],
            "timestamp": "2025-01-02T00:00:01Z",
        },
    ]
    result = codex_parse(payload, fallback_id="fallback-id")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "session-456"
    assert result.created_at == "2025-01-02T00:00:00Z"
    assert len(result.messages) == 1

    assert result.messages[0].provider_message_id == "msg-3"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Test message"


# =============================================================================
# PARSERS.BASE MODULE TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# NORMALIZE_ROLE - PARAMETRIZED
# -----------------------------------------------------------------------------


NORMALIZE_ROLE_STANDARD_CASES = [
    ("user", "user", "user stays user"),
    ("assistant", "assistant", "assistant stays assistant"),
    ("system", "system", "system stays system"),
    ("human", "user", "human aliased to user"),
    ("model", "assistant", "model aliased to assistant"),
]

NORMALIZE_ROLE_CASE_INSENSITIVE_CASES = [
    ("USER", "user", "upper USER"),
    ("Human", "user", "mixed Human"),
    ("ASSISTANT", "assistant", "upper ASSISTANT"),
    ("Model", "assistant", "mixed Model"),
    ("SYSTEM", "system", "upper SYSTEM"),
]

NORMALIZE_ROLE_EDGE_CASES = [
    ("  user  ", "user", "whitespace stripped user"),
    ("\tassistant\n", "assistant", "whitespace stripped assistant"),
    ("custom_role", "unknown", "unrecognized returns unknown"),
    ("BOT", "unknown", "unrecognized BOT returns unknown"),
]

NORMALIZE_ROLE_STRICT_CASES = [
    (None, "None raises ValueError"),
    ("", "empty raises ValueError"),
    ("   ", "whitespace only raises ValueError"),
    ("\t\n", "tabs and newlines raise ValueError"),
]


class TestNormalizeRole:
    """Parametrized tests for role normalization."""

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_STANDARD_CASES)
    def test_normalize_role_standard(self, role: str, expected: str, description: str):
        """Standard role mappings."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_CASE_INSENSITIVE_CASES)
    def test_normalize_role_case_insensitive(self, role: str, expected: str, description: str):
        """Role normalization is case-insensitive."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_EDGE_CASES)
    def test_normalize_role_edge_cases(self, role: str, expected: str, description: str):
        """Edge cases: whitespace stripping, unrecognized roles."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,description", NORMALIZE_ROLE_STRICT_CASES)
    def test_normalize_role_rejects_empty(self, role: str | None, description: str):
        """Empty/None roles raise ValueError - role is required at parse time."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            normalize_role(role)  # type: ignore


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
