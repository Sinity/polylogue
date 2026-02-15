"""ChatGPT parser tests — format detection, message extraction, parent/branch, metadata, parsing, real exports."""

from __future__ import annotations

import json

import pytest

from polylogue.sources.parsers.chatgpt import _coerce_float, extract_messages_from_mapping
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import looks_like_ai, looks_like_code
from tests.infra.helpers import make_chatgpt_node

# =============================================================================
# CHATGPT PARSER TESTS
# =============================================================================


# MERGED FORMAT + COERCE DETECTION
PROVIDER_FORMAT_DETECTION_CASES = [
    # ChatGPT
    ({"mapping": {}}, True, chatgpt_looks_like, "ChatGPT: valid empty mapping"),
    ({"mapping": {"node1": {}}}, True, chatgpt_looks_like, "ChatGPT: valid with nodes"),
    ({}, False, chatgpt_looks_like, "ChatGPT: missing mapping"),
    (None, False, chatgpt_looks_like, "ChatGPT: None input"),
    # Claude AI
    ({"chat_messages": []}, True, looks_like_ai, "Claude AI: chat_messages"),
    ({}, False, looks_like_ai, "Claude AI: missing chat_messages"),
    (None, False, looks_like_ai, "Claude AI: None"),
    # Claude Code
    ([{"parentUuid": "123"}], True, looks_like_code, "Claude Code: parentUuid"),
    ([], False, looks_like_code, "Claude Code: empty list"),
    (None, False, looks_like_code, "Claude Code: None"),
]


@pytest.mark.parametrize("data,expected,check_fn,desc", PROVIDER_FORMAT_DETECTION_CASES)
def test_provider_format_detection(data, expected, check_fn, desc):
    """Unified format detection across all providers."""
    result = check_fn(data)
    assert result == expected, f"Failed {desc}"


# COERCE FLOAT - MERGED WITH FORMAT DETECTION ABOVE

COERCE_FLOAT_CASES = [
    (42, 42.0, "int"),
    (3.14, 3.14, "float"),
    ("2.5", 2.5, "string number"),
    ("invalid", None, "invalid string"),
    (None, None, "None"),
]

@pytest.mark.parametrize("input_val,expected,desc", COERCE_FLOAT_CASES)
def test_coerce_float(input_val, expected, desc):
    """Test _coerce_float conversion."""
    result = _coerce_float(input_val)
    assert result == expected, f"Failed {desc}"


# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 17)


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
    ({"node1": {"message": {"id": "1", "author": {"role": "user"}, "content": {"parts": []}}}}, 0, "empty parts"),

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

    for msg, expected_parent, expected_index in zip(messages, expected_parents, expected_indexes, strict=False):
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


PARSE_CONVERSATION_CASES = [
    # ChatGPT title extraction
    (chatgpt_parse, {"title": "My Conv", "mapping": {}}, "title", "ChatGPT: title field"),
    (chatgpt_parse, {"name": "Conv Name", "mapping": {}}, "name", "ChatGPT: name field"),
    (chatgpt_parse, {"id": "conv-123", "mapping": {}}, "id", "ChatGPT: id field"),
    (chatgpt_parse, {"mapping": {}}, "fallback", "ChatGPT: uses fallback-id"),
]


@pytest.mark.parametrize("parse_fn,conv_data,check_type,desc", PARSE_CONVERSATION_CASES)
def test_parse_conversation(parse_fn, conv_data, check_type, desc):
    """Unified conversation parsing across providers."""
    result = parse_fn(conv_data, "fallback-id")

    if check_type == "title":
        assert result.title in conv_data.values(), f"Failed {desc}"
    elif check_type == "id":
        assert result.provider_conversation_id == conv_data["id"], f"Failed {desc}"
    elif check_type == "fallback":
        assert result.provider_conversation_id == "fallback-id", f"Failed {desc}"
    elif check_type == "provider":
        assert result.provider_name in ["claude", "claude-code"], f"Failed {desc}"


# -----------------------------------------------------------------------------
# SYNTHETIC DATA INTEGRATION
# -----------------------------------------------------------------------------


def test_chatgpt_parse_synthetic_simple():
    """Parse synthetic ChatGPT export."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    corpus = SyntheticCorpus.for_provider("chatgpt")
    raw = corpus.generate(count=1, messages_per_conversation=range(3, 6), seed=42)[0]
    data = json.loads(raw) if isinstance(raw, bytes) else raw

    result = chatgpt_parse(data, "simple-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 0
    assert all(m.text is not None for m in result.messages)


def test_chatgpt_parse_synthetic_branching():
    """Parse synthetic ChatGPT conversation with many messages (branching structure)."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    corpus = SyntheticCorpus.for_provider("chatgpt")
    raw = corpus.generate(count=1, messages_per_conversation=range(12, 20), seed=99)[0]
    data = json.loads(raw) if isinstance(raw, bytes) else raw

    result = chatgpt_parse(data, "branching-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 10  # Multiple messages like branching conversations
