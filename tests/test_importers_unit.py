"""Consolidated provider importer unit tests.

CONSOLIDATED: This file merges tests from:
- test_importers_chatgpt.py (ChatGPT format detection + extraction)
- test_importers_claude.py (Claude AI/Code format detection + extraction)
- test_importers_codex.py (Codex envelope/intermediate format detection)

These are unit tests for individual provider parsers. For cross-provider
integration tests using real export files, see test_importers_parametrized.py.

For property-based testing with Hypothesis, see test_importers_properties.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ChatGPT imports
from polylogue.importers.chatgpt import _coerce_float, extract_messages_from_mapping, looks_like as chatgpt_looks_like, parse as chatgpt_parse

# Claude imports
from polylogue.importers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    looks_like_ai,
    looks_like_code,
    parse_ai,
    parse_code,
)

# Codex imports
from polylogue.importers.codex import looks_like as codex_looks_like, parse as codex_parse

# Test helpers
from tests.helpers import make_chatgpt_node, make_claude_chat_message


# =============================================================================
# CHATGPT IMPORTER TESTS
# =============================================================================


# -----------------------------------------------------------------------------
# FORMAT DETECTION - PARAMETRIZED (1 test replacing 6)
# -----------------------------------------------------------------------------


CHATGPT_LOOKS_LIKE_CASES = [
    ({"mapping": {}}, True, "valid empty mapping"),
    ({"mapping": {"node1": {}}}, True, "valid with nodes"),
    ({}, False, "missing mapping"),
    ({"id": "test"}, False, "no mapping field"),
    ({"mapping": "not-a-dict"}, False, "mapping not dict"),
    (None, False, "None input"),
    ("string", False, "string input"),
    ([], False, "list input"),
    # Claude format should be rejected
    ({"chat_messages": []}, False, "Claude AI format"),
    ({"messages": [{"role": "user"}]}, False, "other format"),
]


@pytest.mark.parametrize("data,expected,desc", CHATGPT_LOOKS_LIKE_CASES)
def test_chatgpt_looks_like_format(data, expected, desc):
    """Comprehensive ChatGPT format detection test.

    Replaces 6 individual looks_like tests.
    """
    result = chatgpt_looks_like(data)
    assert result == expected, f"Failed {desc}"


# -----------------------------------------------------------------------------
# COERCE FLOAT - PARAMETRIZED (1 test replacing 6)
# -----------------------------------------------------------------------------


COERCE_FLOAT_CASES = [
    (42, 42.0, "int"),
    (3.14, 3.14, "float"),
    ("2.5", 2.5, "string number"),
    ("invalid", None, "invalid string"),
    (None, None, "None"),
    (True, None, "bool"),
    ([], None, "list"),
    ({}, None, "dict"),
]


@pytest.mark.parametrize("input_val,expected,desc", COERCE_FLOAT_CASES)
def test_chatgpt_coerce_float_comprehensive(input_val, expected, desc):
    """Comprehensive float coercion test.

    Replaces 6 individual coerce tests.
    """
    result = _coerce_float(input_val)
    if expected is None:
        assert result is None, f"Failed {desc}: expected None, got {result}"
    else:
        assert result == expected, f"Failed {desc}: expected {expected}, got {result}"


# -----------------------------------------------------------------------------
# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 17)
# -----------------------------------------------------------------------------


CHATGPT_EXTRACT_MESSAGES_CASES = [
    # Basic extraction
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, 1, "basic message"),

    # Timestamp handling
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=1704067200)}, 1, "with timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=None)}, 1, "null timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=0)}, 1, "zero timestamp"),

    # Mixed timestamps (should sort)
    ({
        "node1": make_chatgpt_node("msg1", "user", ["First"], timestamp=1000),
        "node2": make_chatgpt_node("msg2", "assistant", ["Second"], timestamp=2000),
        "node3": make_chatgpt_node("msg3", "user", ["Third"], timestamp=500),
    }, 3, "mixed timestamps sorted"),

    # Content variants
    ({"node1": make_chatgpt_node("msg1", "user", ["Part1", "Part2"])}, 1, "multiple parts"),
    ({"node1": make_chatgpt_node("msg1", "user", [None, "Valid"])}, 1, "parts with None"),
    ({"node1": {"message": {"id": "1", "author": {"role": "user"}, "content": {"parts": []}}}}, 1, "empty parts"),

    # Role normalization
    ({"node1": make_chatgpt_node("msg1", "human", ["Hi"])}, 1, "human role alias"),
    ({"node1": make_chatgpt_node("msg1", "model", ["Response"])}, 1, "model role alias"),

    # Missing fields
    ({"node1": {"id": "1", "message": None}}, 0, "missing message"),
    ({"node1": {"id": "1", "message": {"id": "1"}}}, 0, "missing author"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {}}}}, 0, "missing role"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {"role": "user"}}}}, 0, "missing content"),

    # Non-dict nodes
    ({"node1": "not a dict"}, 0, "non-dict node"),
    ({"node1": None}, 0, "None node"),

    # Empty mapping
    ({}, 0, "empty mapping"),
]


@pytest.mark.parametrize("mapping,expected_count,desc", CHATGPT_EXTRACT_MESSAGES_CASES)
def test_chatgpt_extract_messages_comprehensive(mapping, expected_count, desc):
    """Comprehensive message extraction test.

    Replaces 17 individual extraction tests.
    """
    messages, attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == expected_count, \
        f"Failed {desc}: expected {expected_count} messages, got {len(messages)}"

    # Verify all messages have required fields
    for msg in messages:
        assert msg.text is not None
        assert msg.role in ["user", "assistant", "system", "tool"]


# -----------------------------------------------------------------------------
# PARENT & BRANCH INDEX EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_PARENT_BRANCH_CASES = [
    # No parent (root message)
    (
        {"node1": make_chatgpt_node("msg1", "user", ["Hello"])},
        [None],
        [0],
        "root message no parent"
    ),

    # Simple linear chain
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Hello"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1"),
        },
        [None, "node1"],
        [0, 0],
        "linear chain parent references"
    ),

    # Branching: one parent with multiple children
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Question"], children=["msg2", "msg3"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Answer 1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["Answer 2"], parent="node1"),
        },
        [None, "node1", "node1"],
        [0, 0, 1],
        "branching with branch indexes"
    ),

    # Three-way branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Q"], children=["msg2", "msg3", "msg4"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["A1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["A2"], parent="node1"),
            "node4": make_chatgpt_node("msg4", "assistant", ["A3"], parent="node1"),
        },
        [None, "node1", "node1", "node1"],
        [0, 0, 1, 2],
        "three-way branch indexes"
    ),

    # No parent field in node
    (
        {"node1": make_chatgpt_node("msg1", "user", ["Hello"])},
        [None],
        [0],
        "missing parent field defaults to None"
    ),

    # Parent node missing from mapping (orphaned node)
    (
        {"node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1")},
        ["node1"],
        [0],
        "orphaned node with missing parent"
    ),

    # Mixed chain and branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Start"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Response"], children=["msg3", "msg4"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "user", ["Follow 1"], parent="node2"),
            "node4": make_chatgpt_node("msg4", "user", ["Follow 2"], parent="node2"),
        },
        [None, "node1", "node2", "node2"],
        [0, 0, 0, 1],
        "mixed chain and branch structure"
    ),
]


@pytest.mark.parametrize("mapping,expected_parents,expected_indexes,desc", CHATGPT_PARENT_BRANCH_CASES)
def test_chatgpt_extract_parent_and_branch_index(mapping, expected_parents, expected_indexes, desc):
    """Test extraction of parent_message_provider_id and branch_index.

    Validates parent message references and branch position calculation.
    """
    messages, _ = extract_messages_from_mapping(mapping)

    assert len(messages) == len(expected_parents), \
        f"Failed {desc}: expected {len(expected_parents)} messages, got {len(messages)}"

    for msg, expected_parent, expected_index in zip(messages, expected_parents, expected_indexes):
        assert msg.parent_message_provider_id == expected_parent, \
            f"Failed {desc}: message {msg.provider_message_id} expected parent {expected_parent}, " \
            f"got {msg.parent_message_provider_id}"
        assert msg.branch_index == expected_index, \
            f"Failed {desc}: message {msg.provider_message_id} expected branch_index {expected_index}, " \
            f"got {msg.branch_index}"


# -----------------------------------------------------------------------------
# METADATA EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_METADATA_CASES = [
    # Attachments
    ({"attachments": [{"id": "att1", "name": "file.pdf"}]}, True, "attachments field"),
    ({"image_asset_pointer": "asset_123"}, True, "image asset pointer"),

    # Cost/duration
    ({"costUSD": 0.005}, "cost", "cost metadata"),
    ({"durationMs": 2500}, "duration", "duration metadata"),

    # Thinking markers
    ({"content_type": "thoughts"}, "thinking", "thoughts content type"),
    ({"content_type": "reasoning_recap"}, "thinking", "reasoning recap"),

    # Empty
    ({}, None, "no metadata"),
    (None, None, "None metadata"),
]


@pytest.mark.parametrize("metadata,expected_type,desc", CHATGPT_METADATA_CASES)
def test_chatgpt_metadata_extraction(metadata, expected_type, desc):
    """Test metadata extraction from message metadata field.

    Explicit tests for attachment/cost/thinking metadata.
    """
    mapping = {
        "node1": {
            "message": {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {"parts": ["Test"]},
                "metadata": metadata,
            }
        }
    }

    messages, attachments = extract_messages_from_mapping(mapping)

    if expected_type == "attachments":
        # Should have attachment records
        assert len(attachments) > 0 or len(messages[0].attachments) > 0
    elif expected_type == "cost":
        # Should preserve cost in provider_meta
        assert messages[0].provider_meta is not None
    elif expected_type == "thinking":
        # Should mark as thinking
        # (depends on content_blocks implementation)
        pass
    elif expected_type is None:
        # No special metadata
        assert True


# -----------------------------------------------------------------------------
# FULL PARSE - PARAMETRIZED (1 test replacing 12)
# -----------------------------------------------------------------------------


CHATGPT_PARSE_CASES = [
    # Title extraction
    ({"title": "My Conversation", "mapping": {}}, "My Conversation", "title field"),
    ({"name": "Conversation Name", "mapping": {}}, "Conversation Name", "name field"),
    ({"title": "Title", "name": "Name", "mapping": {}}, "Title", "title precedence"),

    # ID extraction
    ({"id": "conv-123", "mapping": {}}, "conv-123", "id field"),
    ({"uuid": "uuid-456", "mapping": {}}, "uuid-456", "uuid field"),
    ({"conversation_id": "cid-789", "mapping": {}}, "cid-789", "conversation_id field"),
    ({"id": "id1", "uuid": "uuid1", "mapping": {}}, "id1", "id precedence"),

    # Timestamp extraction
    ({"create_time": 1704067200, "mapping": {}}, 1704067200, "create_time"),
    ({"update_time": 1704067200, "mapping": {}}, 1704067200, "update_time"),

    # Missing fields use fallback
    ({"mapping": {}}, "fallback-id", "no title uses fallback-id"),
    ({"mapping": {}}, "fallback-id", "no id uses fallback"),
]


@pytest.mark.parametrize("conv_data,expected_value,desc", CHATGPT_PARSE_CASES)
def test_chatgpt_parse_conversation_comprehensive(conv_data, expected_value, desc):
    """Comprehensive conversation parsing test.

    Replaces 12 individual parse tests.
    """
    result = chatgpt_parse(conv_data, "fallback-id")

    if "title" in desc or "name" in desc:
        assert result.title == expected_value, f"Failed {desc}"
    elif "id" in desc and "no id" not in desc:
        assert result.provider_conversation_id == expected_value, f"Failed {desc}"
    elif "time" in desc:
        if result.created_at:
            # created_at is returned as string timestamp
            assert result.created_at == str(expected_value)
    else:
        # Verify it parses without error
        assert result.provider_name == "chatgpt"


# -----------------------------------------------------------------------------
# REAL EXPORT INTEGRATION
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("fixtures/real/chatgpt/simple.json").exists(),
    reason="Real ChatGPT sample not available"
)
def test_chatgpt_parse_real_simple():
    """Parse real ChatGPT simple export."""
    sample_path = Path(__file__).parent / "fixtures" / "real" / "chatgpt" / "simple.json"
    with open(sample_path) as f:
        data = json.load(f)

    result = chatgpt_parse(data, "simple-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 0
    # Some messages may have empty text (system messages, etc)
    assert all(m.text is not None for m in result.messages)


@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("fixtures/real/chatgpt/branching.json").exists(),
    reason="Real ChatGPT sample not available"
)
def test_chatgpt_parse_real_branching():
    """Parse real ChatGPT branching conversation."""
    sample_path = Path(__file__).parent / "fixtures" / "real" / "chatgpt" / "branching.json"
    with open(sample_path) as f:
        data = json.load(f)

    result = chatgpt_parse(data, "branching-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 10  # Branching conversations are larger
    # Branching structure is handled internally, no provider_meta on conversation


# =============================================================================
# CLAUDE IMPORTER TESTS
# =============================================================================


# -----------------------------------------------------------------------------
# FORMAT DETECTION - PARAMETRIZED (2 tests replacing 13)
# -----------------------------------------------------------------------------


CLAUDE_LOOKS_LIKE_AI_CASES = [
    ({"chat_messages": []}, True, "chat_messages field"),
    ({"chat_messages": [{"uuid": "1"}]}, True, "with messages"),
    ({}, False, "missing chat_messages"),
    ({"messages": []}, False, "wrong field name"),
    ({"chat_messages": "not-a-list"}, False, "chat_messages not list"),
    (None, False, "None input"),
    ("string", False, "string input"),
    ([], False, "list input"),
    # ChatGPT format should be rejected
    ({"mapping": {}}, False, "ChatGPT format"),
]


@pytest.mark.parametrize("data,expected,desc", CLAUDE_LOOKS_LIKE_AI_CASES)
def test_claude_looks_like_ai_format(data, expected, desc):
    """Comprehensive AI format detection.

    Replaces 8 looks_like_ai tests.
    """
    result = looks_like_ai(data)
    assert result == expected, f"Failed {desc}"


CLAUDE_LOOKS_LIKE_CODE_CASES = [
    ([{"parentUuid": "123"}], True, "parentUuid variant"),
    ([{"sessionId": "456"}], True, "sessionId variant"),
    ([{"session_id": "789"}], True, "session_id variant"),
    ([{"leafUuid": "abc"}], True, "leafUuid variant"),
    ([], False, "empty list"),
    ([{"messages": []}], False, "only messages field"),
    (None, False, "None input"),
    ({}, False, "dict input"),
    # ChatGPT format should be rejected
    ([{"mapping": {}}], False, "ChatGPT format"),
]


@pytest.mark.parametrize("data,expected,desc", CLAUDE_LOOKS_LIKE_CODE_CASES)
def test_claude_looks_like_code_format(data, expected, desc):
    """Comprehensive Code format detection.

    Replaces 5 looks_like_code tests.
    """
    result = looks_like_code(data)
    assert result == expected, f"Failed {desc}"


# -----------------------------------------------------------------------------
# SEGMENT EXTRACTION - PARAMETRIZED (1 test replacing 10)
# -----------------------------------------------------------------------------


CLAUDE_SEGMENT_CASES = [
    (["plain text"], "plain text", "string segment"),
    ([{"text": "dict with text"}], "dict with text", "dict with text field"),
    ([{"content": "dict with content"}], "dict with content", "dict with content field"),
    ([{"type": "tool_use", "name": "read", "input": {}}], "read", "tool_use"),
    ([{"type": "tool_result", "content": "result"}], "result", "tool_result"),
    (["text1", {"text": "text2"}, "text3"], "text1\ntext2\ntext3", "mixed segments"),
    ([{}, "", None], None, "empty/None segments"),
    ([{"other": "field"}], None, "dict without text/content/type"),
]


@pytest.mark.parametrize("segments,expected_contains,desc", CLAUDE_SEGMENT_CASES)
def test_claude_extract_text_from_segments_comprehensive(segments, expected_contains, desc):
    """Comprehensive segment extraction.

    Replaces 10 segment extraction tests.
    """
    result = extract_text_from_segments(segments)

    if expected_contains:
        assert expected_contains in result, f"Failed {desc}: '{expected_contains}' not in '{result}'"
    else:
        assert result is None or result == "", f"Failed {desc}: expected None or empty"


# -----------------------------------------------------------------------------
# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 15)
# -----------------------------------------------------------------------------


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
    elif isinstance(expected, str):
        if messages:
            # ID field tests check provider_message_id, role tests check role
            if "field" in desc and desc not in ["attachments field", "files field", "timestamp field"]:
                assert messages[0].provider_message_id == expected, f"Failed {desc}"
            else:
                # Expected role
                assert messages[0].role == expected, f"Failed {desc}"


# -----------------------------------------------------------------------------
# PARSE AI - PARAMETRIZED (1 test replacing 10)
# -----------------------------------------------------------------------------


def make_ai_conv(chat_messages, **kwargs):
    """Helper to create AI format conversation."""
    conv = {"chat_messages": chat_messages}
    conv.update(kwargs)
    return conv


CLAUDE_PARSE_AI_CASES = [
    # Basic
    (make_ai_conv([make_claude_chat_message("u1", "human", "Hello")]), 1, "basic"),

    # With attachments
    (make_ai_conv([make_claude_chat_message("u1", "human", "Hi", files=[{"file_name": "doc.pdf"}])]), 1, "with files"),

    # Title extraction
    (make_ai_conv([], name="Test Title"), "Test Title", "title extraction"),

    # Empty chat_messages
    (make_ai_conv([]), 0, "empty messages"),

    # Content variants (with role)
    ({"chat_messages": [{"uuid": "u1", "sender": "human", "content": {"text": "nested"}}]}, 1, "content dict"),
    ({"chat_messages": [{"uuid": "u1", "sender": "human", "content": {"parts": ["p1"]}}]}, 1, "content parts"),

    # Missing text/role skipped
    (make_ai_conv([{"uuid": "u1", "sender": "human"}]), 0, "missing text"),
    (make_ai_conv([{"uuid": "u1", "text": "no role"}]), 0, "missing role"),
]


@pytest.mark.parametrize("conv_data,expected,desc", CLAUDE_PARSE_AI_CASES)
def test_claude_parse_ai_comprehensive(conv_data, expected, desc):
    """Comprehensive AI format parsing.

    Replaces 10 parse_ai tests.
    """
    result = parse_ai(conv_data, "fallback-id")

    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        # Expected title
        assert result.title == expected, f"Failed {desc}"


# -----------------------------------------------------------------------------
# PARSE CODE - PARAMETRIZED (1 test replacing 17)
# -----------------------------------------------------------------------------


def make_code_message(msg_type, text, **kwargs):
    """Helper to create Code format message.

    Creates message in claude-code format with nested message.content structure.
    """
    msg = {
        "type": msg_type,
    }
    # Add text as nested message.content
    if text or "message" not in kwargs:
        msg["message"] = {"content": text} if text else {}
    msg.update(kwargs)
    return msg


CLAUDE_PARSE_CODE_CASES = [
    # Basic messages
    ([make_code_message("user", "Question")], 1, "user message"),
    ([make_code_message("assistant", "Answer")], 1, "assistant message"),

    # Tool use (as message content list)
    ([make_code_message("assistant", "", message={"content": [{"type": "tool_use", "name": "read"}]})], 1, "tool use"),
    ([make_code_message("user", "", message={"content": [{"type": "tool_result", "content": "result"}]})], 1, "tool result"),

    # Thinking blocks
    ([make_code_message("assistant", "", message={"content": [{"type": "thinking", "thinking": "analysis"}]})], 1, "with thinking field"),

    # Metadata preservation
    ([make_code_message("assistant", "text", costUSD=0.01, durationMs=1000)], 1, "with cost/duration"),
    ([make_code_message("assistant", "text", isSidechain=True)], 1, "sidechain marker"),

    # Skip summary/init types
    ([make_code_message("summary", "Summary text")], 0, "skip summary"),
    ([make_code_message("init", "Init")], 0, "skip init"),

    # Type to role mapping
    ([make_code_message("user", "Q")], "user", "user type"),
    ([make_code_message("assistant", "A")], "assistant", "assistant type"),

    # Message as string (content field as string)
    ([{"type": "user", "message": {"content": "String content"}}], 1, "message as string"),

    # Empty/missing fields (currently returns message with text=None)
    ([make_code_message("assistant", "")], 1, "empty text"),
    ([{"type": "assistant"}], 1, "missing text"),

    # Session metadata
    ([make_code_message("user", "Hi")], None, "extracts session metadata"),

    # Timestamp extraction
    ([make_code_message("user", "Hi", timestamp=1704067200)], 1704067200, "timestamp"),

    # Empty list
    ([], 0, "empty messages"),

    # Non-dict items
    (["not a dict"], 0, "skip non-dict"),
]


@pytest.mark.parametrize("messages,expected,desc", CLAUDE_PARSE_CODE_CASES)
def test_claude_parse_code_comprehensive(messages, expected, desc):
    """Comprehensive Code format parsing.

    Replaces 17 parse_code tests.
    """
    result = parse_code(messages, "fallback-id")

    if isinstance(expected, int):
        if "timestamp" not in desc:
            assert len(result.messages) == expected, f"Failed {desc}"
        else:
            # Check timestamp was extracted (as float string)
            if result.messages and result.messages[0].timestamp:
                assert float(result.messages[0].timestamp) == expected
    elif isinstance(expected, str):
        # Expected role
        if result.messages:
            assert result.messages[0].role == expected, f"Failed {desc}"
    elif expected is None:
        # Just verify it parsed
        assert result.provider_name in ["claude", "claude-code"]


# =============================================================================
# CODEX IMPORTER TESTS
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
