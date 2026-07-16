"""ChatGPT parser tests — format detection, message extraction, parent/branch, metadata, parsing, real exports."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import pytest

from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin
from polylogue.scenarios import CorpusSpec
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.chatgpt import (
    SHARED_CONVERSATION_INDEX_INGEST_FLAG,
    _coerce_float,
    extract_messages_from_mapping,
)
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import looks_like_ai, looks_like_code
from tests.infra.source_builders import make_chatgpt_node

ProviderCheck: TypeAlias = Callable[[object], bool]
ProviderDetectionCase: TypeAlias = tuple[object, bool, ProviderCheck, str]
CoerceFloatCase: TypeAlias = tuple[object, float | None, str]
ChatGPTMapping: TypeAlias = dict[str, object]
ExtractMessagesCase: TypeAlias = tuple[ChatGPTMapping, int, str]
ParentBranchCase: TypeAlias = tuple[ChatGPTMapping, list[str | None], list[int], str]
MetadataCase: TypeAlias = tuple[object, str | None, str]
ParseFn: TypeAlias = Callable[[Mapping[str, object], str], ParsedSession]
ParseSessionCase: TypeAlias = tuple[ParseFn, ChatGPTMapping, str, str]


def _looks_like_code_payload(data: object) -> bool:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        return False
    return looks_like_code(data)


# =============================================================================
# CHATGPT PARSER TESTS
# =============================================================================


# MERGED FORMAT + COERCE DETECTION
PROVIDER_FORMAT_DETECTION_CASES: list[ProviderDetectionCase] = [
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
    ([{"parentUuid": "123"}], True, _looks_like_code_payload, "Claude Code: parentUuid"),
    ([], False, _looks_like_code_payload, "Claude Code: empty list"),
    (None, False, _looks_like_code_payload, "Claude Code: None"),
]


@pytest.mark.parametrize("data,expected,check_fn,desc", PROVIDER_FORMAT_DETECTION_CASES)
def test_provider_format_detection(data: object, expected: bool, check_fn: ProviderCheck, desc: str) -> None:
    """Unified format detection across all providers."""
    result = check_fn(data)
    assert result == expected, f"Failed {desc}"


def test_chatgpt_rich_web_metadata_is_promoted_to_typed_constructs() -> None:
    messages, _ = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant", "name": "research_kickoff_tool.start_research_task"},
                    "recipient": "canmore.update_textdoc",
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Research answer"]},
                    "metadata": {
                        "canvas": {"textdoc_id": "canvas-1"},
                        "content_references": [{"url": "https://example.test/ref"}],
                        "search_result_groups": [{"query": "polylogue"}],
                        "search_queries": ["polylogue browser capture"],
                        "selected_sources": [{"title": "Source"}],
                        "async_task_type": "deep_research",
                        "async_task_id": "task-1",
                        "citations": [{"start_ix": 0, "end_ix": 8}],
                    },
                },
            }
        }
    )

    constructs = messages[0].blocks[0].web_constructs
    assert [construct.construct_type.value for construct in constructs] == [
        "canvas",
        "content_reference",
        "content_reference",
        "search_query",
        "selected_source",
        "async_task",
    ]
    assert constructs[0].source_id == "canvas-1"
    assert constructs[1].url == "https://example.test/ref"
    assert constructs[2].start_index == 0
    assert constructs[3].query == "polylogue browser capture"
    assert constructs[4].title == "Source"
    assert constructs[5].task_type == "deep_research"
    assert constructs[5].task_id == "task-1"
    assert messages[0].blocks[0].metadata is None


def test_chatgpt_keeps_image_asset_only_nodes() -> None:
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "image-msg",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-service://image-asset-1",
                            }
                        ],
                    },
                    "metadata": {"model_slug": "gpt-image"},
                },
            }
        }
    )

    assert len(messages) == 1
    assert messages[0].provider_message_id == "image-msg"
    assert messages[0].blocks[0].type is BlockType.IMAGE
    assert messages[0].blocks[0].metadata == {"asset_pointer": "file-service://image-asset-1"}


def test_chatgpt_shared_conversation_index_shell_is_tagged() -> None:
    session = chatgpt_parse(
        {
            "conversation_id": "shared-conv",
            "id": "share-row",
            "is_anonymous": True,
            "title": "Shared title only",
        },
        "fallback",
    )

    assert session.messages == []
    assert SHARED_CONVERSATION_INDEX_INGEST_FLAG in session.ingest_flags


def test_chatgpt_temporary_payload_sets_session_kind() -> None:
    session = chatgpt_parse(
        {
            "id": "temporary-native",
            "title": "Temporary",
            "is_temporary": True,
            "mapping": {},
        },
        "fallback",
    )

    assert session.session_kind == "temporary"
    assert "capture:temporary-chat" in session.ingest_flags


# COERCE FLOAT - MERGED WITH FORMAT DETECTION ABOVE

COERCE_FLOAT_CASES: list[CoerceFloatCase] = [
    (42, 42.0, "int"),
    (3.14, 3.14, "float"),
    ("2.5", 2.5, "string number"),
    ("2024-01-15T10:30:00Z", 1705314600.0, "ISO datetime string"),
    ("invalid", None, "invalid string"),
    (None, None, "None"),
]


@pytest.mark.parametrize("input_val,expected,desc", COERCE_FLOAT_CASES)
def test_coerce_float(input_val: object, expected: float | None, desc: str) -> None:
    """Test _coerce_float conversion."""
    result = _coerce_float(input_val)
    assert result == expected, f"Failed {desc}"


def test_chatgpt_message_extraction_sorts_iso_timestamps() -> None:
    mapping = {
        "late": {
            "id": "late",
            "message": {
                "id": "late",
                "author": {"role": "assistant"},
                "content": {"parts": ["later"]},
                "create_time": "2024-01-15T10:31:00Z",
            },
        },
        "early": {
            "id": "early",
            "message": {
                "id": "early",
                "author": {"role": "user"},
                "content": {"parts": ["earlier"]},
                "create_time": "2024-01-15T10:30:00Z",
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [message.provider_message_id for message in messages] == ["early", "late"]


# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 17)


CHATGPT_EXTRACT_MESSAGES_CASES: list[ExtractMessagesCase] = [
    # Basic extraction
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, 1, "basic message"),
    # Timestamp handling
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=1704067200)}, 1, "with timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=None)}, 1, "null timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=0)}, 1, "zero timestamp"),
    # Mixed timestamps (should sort)
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["First"], timestamp=1000),
            "node2": make_chatgpt_node("msg2", "assistant", ["Second"], timestamp=2000),
            "node3": make_chatgpt_node("msg3", "user", ["Third"], timestamp=500),
        },
        3,
        "mixed timestamps sorted",
    ),
    # Content variants
    ({"node1": make_chatgpt_node("msg1", "user", ["Part1", "Part2"])}, 1, "multiple parts"),
    (
        {
            "node1": {
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"parts": [None, "Valid"]},
                }
            }
        },
        1,
        "parts with None",
    ),
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
def test_chatgpt_extract_messages_comprehensive(mapping: ChatGPTMapping, expected_count: int, desc: str) -> None:
    """Comprehensive message extraction test.

    Replaces 17 individual extraction tests.
    """
    messages, attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == expected_count, f"Failed {desc}: expected {expected_count} messages, got {len(messages)}"

    # Verify all messages have required fields
    for msg in messages:
        assert msg.text is not None
        assert msg.role in ["user", "assistant", "system", "tool"]


# -----------------------------------------------------------------------------
# PARENT & BRANCH INDEX EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_PARENT_BRANCH_CASES: list[ParentBranchCase] = [
    # No parent (root message)
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, [None], [0], "root message no parent"),
    # Simple linear chain
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Hello"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1"),
        },
        [None, "msg1"],
        [0, 0],
        "linear chain parent references",
    ),
    # Branching: one parent with multiple children
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Question"], children=["msg2", "msg3"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Answer 1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["Answer 2"], parent="node1"),
        },
        [None, "msg1", "msg1"],
        [0, 0, 1],
        "branching with branch indexes",
    ),
    # Three-way branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Q"], children=["msg2", "msg3", "msg4"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["A1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["A2"], parent="node1"),
            "node4": make_chatgpt_node("msg4", "assistant", ["A3"], parent="node1"),
        },
        [None, "msg1", "msg1", "msg1"],
        [0, 0, 1, 2],
        "three-way branch indexes",
    ),
    # No parent field in node
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, [None], [0], "missing parent field defaults to None"),
    # Parent node missing from the emitted message set: keep the message, drop
    # the dangling parent edge so storage does not reject the session.
    (
        {"node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1")},
        [None],
        [0],
        "orphaned node drops missing parent",
    ),
    # Mixed chain and branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Start"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Response"], children=["msg3", "msg4"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "user", ["Follow 1"], parent="node2"),
            "node4": make_chatgpt_node("msg4", "user", ["Follow 2"], parent="node2"),
        },
        [None, "msg1", "msg2", "msg2"],
        [0, 0, 0, 1],
        "mixed chain and branch structure",
    ),
]


@pytest.mark.parametrize("mapping,expected_parents,expected_indexes,desc", CHATGPT_PARENT_BRANCH_CASES)
def test_chatgpt_extract_parent_and_branch_index(
    mapping: ChatGPTMapping,
    expected_parents: list[str | None],
    expected_indexes: list[int],
    desc: str,
) -> None:
    """Test extraction of parent_message_provider_id and branch_index.

    Validates parent message references and branch position calculation.
    """
    messages, _ = extract_messages_from_mapping(mapping)

    assert len(messages) == len(expected_parents), (
        f"Failed {desc}: expected {len(expected_parents)} messages, got {len(messages)}"
    )

    for msg, expected_parent, expected_index in zip(messages, expected_parents, expected_indexes, strict=False):
        assert msg.parent_message_provider_id == expected_parent, (
            f"Failed {desc}: message {msg.provider_message_id} expected parent {expected_parent}, "
            f"got {msg.parent_message_provider_id}"
        )
        assert msg.branch_index == expected_index, (
            f"Failed {desc}: message {msg.provider_message_id} expected branch_index {expected_index}, "
            f"got {msg.branch_index}"
        )


def test_chatgpt_drops_parent_links_to_filtered_messages() -> None:
    messages, _ = extract_messages_from_mapping(
        {
            "empty-parent": make_chatgpt_node("parent-msg", "user", [], children=["child"]),
            "child": make_chatgpt_node("child-msg", "assistant", ["survives"], parent="parent-msg"),
            "root": make_chatgpt_node("root-msg", "user", ["root"], children=["valid-child-msg"]),
            "valid-child": make_chatgpt_node("valid-child-msg", "assistant", ["valid"], parent="root-msg"),
        }
    )

    parents = {message.provider_message_id: message.parent_message_provider_id for message in messages}

    assert parents["child-msg"] is None
    assert parents["valid-child-msg"] == "root-msg"


# -----------------------------------------------------------------------------
# METADATA EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_METADATA_CASES: list[MetadataCase] = [
    # Attachments
    ({"attachments": [{"id": "att1", "name": "file.pdf"}]}, "attachments", "attachments field"),
    ({"image_asset_pointer": "asset_123"}, None, "image asset pointer metadata ignored"),
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
def test_chatgpt_metadata_extraction(metadata: object, expected_type: str | None, desc: str) -> None:
    """Test metadata extraction from message metadata field.

    Explicit tests for attachment/cost/thinking metadata.
    """
    mapping: dict[str, object] = {
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
        assert len(attachments) > 0
    elif expected_type == "cost":
        # Cost metadata is not retained per-message (cost lives in
        # session_model_usage / reported_cost_usd); the parser must still
        # extract the message rather than drop or crash on the metadata.
        assert len(messages) == 1
    elif expected_type == "thinking":
        # Should mark as thinking
        # (depends on content_blocks implementation)
        pass
    elif expected_type is None:
        pass  # No special metadata expected


# -----------------------------------------------------------------------------
# FULL PARSE - PARAMETRIZED (1 test replacing 12)
# -----------------------------------------------------------------------------


PARSE_SESSION_CASES: list[ParseSessionCase] = [
    # ChatGPT title extraction
    (chatgpt_parse, {"title": "My Conv", "mapping": {}}, "title", "ChatGPT: title field"),
    (chatgpt_parse, {"name": "Conv Name", "mapping": {}}, "name", "ChatGPT: name field"),
    (chatgpt_parse, {"id": "conv-123", "mapping": {}}, "id", "ChatGPT: id field"),
    (chatgpt_parse, {"mapping": {}}, "fallback", "ChatGPT: uses fallback-id"),
]


@pytest.mark.parametrize("parse_fn,conv_data,check_type,desc", PARSE_SESSION_CASES)
def test_parse_session(parse_fn: ParseFn, conv_data: ChatGPTMapping, check_type: str, desc: str) -> None:
    """Unified session parsing across providers."""
    result = parse_fn(conv_data, "fallback-id")

    if check_type == "title":
        assert result.title in conv_data.values(), f"Failed {desc}"
    elif check_type == "id":
        assert result.provider_session_id == conv_data["id"], f"Failed {desc}"
    elif check_type == "fallback":
        assert result.provider_session_id == "fallback-id", f"Failed {desc}"
    elif check_type == "provider":
        assert result.source_name in ["claude-ai", "claude-code"], f"Failed {desc}"


# -----------------------------------------------------------------------------
# SYNTHETIC DATA INTEGRATION
# -----------------------------------------------------------------------------


def test_chatgpt_parse_synthetic_simple() -> None:
    """Parse synthetic ChatGPT export."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=3,
            messages_max=5,
            seed=42,
            origin="generated.test-chatgpt-parser",
            tags=("synthetic", "test", "chatgpt-parser"),
        )
    )[0]
    data = json.loads(raw)

    result = chatgpt_parse(data, "simple-test")

    assert result.source_name == "chatgpt"
    assert len(result.messages) > 0
    assert all(m.text is not None for m in result.messages)


def test_chatgpt_parse_synthetic_branching() -> None:
    """Parse synthetic ChatGPT session with many messages (branching structure)."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=12,
            messages_max=19,
            seed=99,
            origin="generated.test-chatgpt-parser",
            tags=("synthetic", "test", "chatgpt-parser"),
        )
    )[0]
    data = json.loads(raw)

    result = chatgpt_parse(data, "branching-test")

    assert result.source_name == "chatgpt"
    assert len(result.messages) > 10  # Multiple messages like branching sessions


# -----------------------------------------------------------------------------
# METADATA ROUNDTRIP: parser → materialization → hydration
# -----------------------------------------------------------------------------


def test_chatgpt_metadata_extracted_into_content_blocks() -> None:
    """ChatGPT message metadata is extracted into typed parser fields."""
    # Build a ChatGPT mapping payload with rich message-level metadata.
    payload: dict[str, object] = {
        "title": "Metadata Roundtrip Test",
        "id": "conv-roundtrip-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant", "name": "dalle"},
                    "content": {"parts": ["Generated an image of a cat"], "content_type": "text"},
                    "create_time": 1717430400.0,
                    "recipient": "dalle.text2im",
                    "status": "finished_successfully",
                    "end_turn": True,
                    "metadata": {
                        "model_slug": "gpt-4",
                        "citations": [{"title": "Example", "url": "https://example.com"}],
                        "aggregate_result": {"exit_code": 0, "output": "hello world"},
                        "user_context_message_data": {"about_user_message": "I like cats"},
                    },
                },
            },
            "node2": {
                "id": "node2",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "user"},
                    "content": {"parts": ["Show me a cat"], "content_type": "text"},
                    "create_time": 1717430300.0,
                    "metadata": {
                        "model_slug": "gpt-4",
                    },
                },
                "parent": "node1",
            },
        },
    }

    # --- Stage 1: Parse ---
    parsed = chatgpt_parse(payload, "roundtrip-test")
    assert parsed.source_name == "chatgpt"

    assistant_msg = next(m for m in parsed.messages if m.role == "assistant")
    assert len(assistant_msg.blocks) >= 1
    assert assistant_msg.model_name == "gpt-4"
    assert assistant_msg.sender_name == "dalle"
    assert assistant_msg.recipient == "dalle.text2im"
    assert assistant_msg.delivery_status == "finished_successfully"
    assert assistant_msg.end_turn is True
    assert assistant_msg.user_context_text == "I like cats"
    constructs = assistant_msg.blocks[0].web_constructs
    assert any(construct.construct_type.value == "content_reference" for construct in constructs)
    assert any(construct.construct_type.value == "async_task" for construct in constructs)
    assert assistant_msg.blocks[0].metadata is None

    user_msg = next(m for m in parsed.messages if m.role == "user")
    assert len(user_msg.blocks) >= 1
    assert user_msg.model_name == "gpt-4"
    assert user_msg.sender_name is None
    assert user_msg.recipient is None
    assert user_msg.blocks[0].metadata is None


def test_chatgpt_recipient_addressed_json_payload_parses_as_tool_use() -> None:
    """Regression for polylogue-e2yk.

    A recipient-addressed message (e.g. the web-search tool) whose text is a
    JSON-encoded payload must parse as a BlockType.TOOL_USE block, not raw
    BlockType.TEXT -- the reader already folds tool_use blocks by default, so
    this alone fixes the raw-JSON-dumped-in-transcript symptom.
    """
    payload: dict[str, object] = {
        "title": "Web Search Tool Call",
        "id": "conv-tool-call-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": ['{"search_query":[{"q":"Hetzner Cloud prices"}],"response_length":"medium"}'],
                        "content_type": "text",
                    },
                    "create_time": 1717430400.0,
                    "recipient": "web",
                },
            },
        },
    }

    messages, _ = extract_messages_from_mapping(payload["mapping"])  # type: ignore[arg-type]
    assistant_msg = next(m for m in messages if m.role == "assistant")

    assert assistant_msg.recipient == "web"
    assert len(assistant_msg.blocks) == 1
    block = assistant_msg.blocks[0]
    assert block.type == BlockType.TOOL_USE
    assert block.tool_name == "web"
    assert block.tool_input == {
        "search_query": [{"q": "Hetzner Cloud prices"}],
        "response_length": "medium",
    }


def test_chatgpt_recipient_addressed_non_json_text_stays_text() -> None:
    """A recipient-addressed message whose text is NOT JSON stays BlockType.TEXT.

    Only a JSON-parseable payload is reinterpreted as a tool call -- plain
    prose directed at a recipient (e.g. dalle image-gen captions) must not
    be misclassified.
    """
    payload: dict[str, object] = {
        "title": "Non-JSON Recipient Text",
        "id": "conv-non-json-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "content": {"parts": ["a plain caption, not JSON"], "content_type": "text"},
                    "create_time": 1717430400.0,
                    "recipient": "dalle.text2im",
                },
            },
        },
    }

    messages, _ = extract_messages_from_mapping(payload["mapping"])  # type: ignore[arg-type]
    assistant_msg = next(m for m in messages if m.role == "assistant")

    assert assistant_msg.recipient == "dalle.text2im"
    assert len(assistant_msg.blocks) == 1
    assert assistant_msg.blocks[0].type == BlockType.TEXT


# =============================================================================
# CATALOG-DRIVEN METADATA ROUNDTRIP PERMUTATIONS
# =============================================================================

# Each entry tests a distinct metadata field or combination surviving the
# parser → materialization → hydration pipeline. The catalog provides the
# message-level metadata dict and the expected assertions keyed by field name.

_METADATA_PERMUTATION_CASES: list[tuple[dict[str, object], str, dict[str, object]]] = [
    # --- Single-field permutations ---
    (
        {"model_slug": "gpt-4"},
        "single: model_slug",
        {"chatgpt_model": "gpt-4"},
    ),
    (
        {"model_slug": "gpt-4o"},
        "single: model_slug variant gpt-4o",
        {"chatgpt_model": "gpt-4o"},
    ),
    (
        {"model_slug": "o1"},
        "single: model_slug variant o1",
        {"chatgpt_model": "o1"},
    ),
    # --- author metadata (tool use messages) ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True},
        "combined: model + tool author metadata",
        {"chatgpt_model": "gpt-4", "chatgpt_author_name": "dalle", "chatgpt_recipient": "dalle.text2im"},
    ),
    # --- status ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "message_status": "finished_successfully"},
        "combined: model + author + status finished",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "finished_successfully",
        },
    ),
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "message_status": "failed"},
        "combined: model + author + status failed",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "failed",
        },
    ),
    # --- end_turn ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "end_turn": True},
        "combined: model + author + end_turn True",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_end_turn": True,
        },
    ),
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "end_turn": False},
        "combined: model + author + end_turn False",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_end_turn": False,
        },
    ),
    # --- citations ---
    (
        {"model_slug": "gpt-4", "citations": [{"title": "Ref", "url": "https://example.com"}]},
        "single: citations list",
        {"chatgpt_model": "gpt-4", "chatgpt_citations": [{"title": "Ref", "url": "https://example.com"}]},
    ),
    # --- code execution ---
    (
        {"model_slug": "gpt-4", "aggregate_result": {"exit_code": 0, "output": "ok"}},
        "single: code_execution aggregate_result",
        {"chatgpt_model": "gpt-4", "chatgpt_code_execution": {"exit_code": 0, "output": "ok"}},
    ),
    # --- user context ---
    (
        {"model_slug": "gpt-4", "user_context_message_data": {"about_user_message": "likes cats"}},
        "single: user_context_message_data",
        {"chatgpt_model": "gpt-4", "chatgpt_user_context": {"about_user_message": "likes cats"}},
    ),
    # --- full combination ---
    (
        {
            "model_slug": "gpt-4",
            "is_tool_message": True,
            "message_status": "finished_successfully",
            "end_turn": True,
            "citations": [{"title": "A", "url": "https://a.com"}],
            "aggregate_result": {"exit_code": 0, "output": "done"},
            "user_context_message_data": {"about_user_message": "needs help"},
        },
        "full: all metadata fields combined",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "finished_successfully",
            "chatgpt_end_turn": True,
            "chatgpt_citations": [{"title": "A", "url": "https://a.com"}],
            "chatgpt_code_execution": {"exit_code": 0, "output": "done"},
            "chatgpt_user_context": {"about_user_message": "needs help"},
        },
    ),
]


def _build_chatgpt_message_metadata_payload(
    meta_spec: dict[str, object],
) -> dict[str, object]:
    """Build a ChatGPT mapping payload with the given metadata spec."""
    author: dict[str, object] = {"role": "assistant"}
    if meta_spec.get("is_tool_message"):
        author["name"] = "dalle"

    metadata: dict[str, object] = {}
    if "model_slug" in meta_spec:
        metadata["model_slug"] = meta_spec["model_slug"]
    if "citations" in meta_spec:
        metadata["citations"] = meta_spec["citations"]
    if "aggregate_result" in meta_spec:
        metadata["aggregate_result"] = meta_spec["aggregate_result"]
    if "user_context_message_data" in meta_spec:
        metadata["user_context_message_data"] = meta_spec["user_context_message_data"]

    message: dict[str, object] = {
        "id": "msg-1",
        "author": author,
        "content": {"parts": ["Test message"]},
        "create_time": 1717430400.0,
        "metadata": metadata,
    }

    if "is_tool_message" in meta_spec and meta_spec["is_tool_message"]:
        message["recipient"] = "dalle.text2im"
    if "message_status" in meta_spec:
        message["status"] = meta_spec["message_status"]
    if "end_turn" in meta_spec:
        message["end_turn"] = meta_spec["end_turn"]

    return {
        "title": "Metadata Permutation Test",
        "id": "conv-perm-001",
        "mapping": {"node1": {"id": "node1", "message": message}},
    }


@pytest.mark.parametrize("meta_spec,desc,expected_fields", _METADATA_PERMUTATION_CASES)
def test_chatgpt_metadata_permutation_extracted_by_parser(
    meta_spec: dict[str, object],
    desc: str,
    expected_fields: dict[str, object],
) -> None:
    """Catalog-driven: each metadata field lands in a typed parser field."""
    _ = expected_fields
    payload = _build_chatgpt_message_metadata_payload(meta_spec)

    parsed = chatgpt_parse(payload, "permutation-test")
    assert parsed.source_name == "chatgpt"
    assert len(parsed.messages) >= 1

    blocks = parsed.messages[0].blocks
    assert len(blocks) >= 1
    message = parsed.messages[0]
    constructs = blocks[0].web_constructs
    assert blocks[0].metadata is None

    if "model_slug" in meta_spec:
        assert message.model_name == meta_spec["model_slug"], desc
    if meta_spec.get("is_tool_message"):
        assert message.sender_name == "dalle", desc
        assert message.recipient == "dalle.text2im", desc
    else:
        assert message.sender_name is None, desc
        assert message.recipient is None, desc
    if "message_status" in meta_spec:
        assert message.delivery_status == meta_spec["message_status"], desc
    if "end_turn" in meta_spec:
        assert message.end_turn is meta_spec["end_turn"], desc
    if "citations" in meta_spec:
        reference = next(construct for construct in constructs if construct.construct_type.value == "content_reference")
        assert reference.title == "Ref" or reference.title == "A", desc
    if "aggregate_result" in meta_spec:
        task = next(construct for construct in constructs if construct.construct_type.value == "async_task")
        assert task.text in {"ok", "done"}, desc
    if "user_context_message_data" in meta_spec:
        assert message.user_context_text in {"likes cats", "needs help"}, desc


# ---------------------------------------------------------------------------
# #1743 — branch graph preservation with active-path metadata
# ---------------------------------------------------------------------------


def _branch_node(
    msg_id: str,
    role: str,
    text: str,
    *,
    parent: str | None = None,
    children: list[str] | None = None,
    content_type: str = "text",
) -> dict[str, Any]:
    """Mapping node helper that does not force a create_time (graph order is truth)."""
    return {
        "id": msg_id,
        "message": {
            "id": msg_id,
            "author": {"role": role},
            "content": {"content_type": content_type, "parts": [text]},
            "create_time": None,
        },
        "parent": parent,
        "children": children or [],
    }


def test_regeneration_preserves_all_branches_and_marks_active_leaf() -> None:
    """A regenerated assistant turn keeps every branch and marks the active leaf."""
    nodes = [
        _branch_node("root", "system", "", parent=None, children=["u1"]),
        _branch_node("u1", "user", "question", parent="root", children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD wrong answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW correct answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Regenerated",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_new",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "OLD wrong answer", "NEW correct answer"]
    by_id = {m.provider_message_id: m for m in conv.messages}
    assert by_id["u1"].is_active_path is True
    assert by_id["a_old"].is_active_path is False
    assert by_id["a_new"].is_active_path is True
    assert by_id["a_old"].is_active_leaf is False
    assert by_id["a_new"].is_active_leaf is True
    assert conv.active_leaf_message_provider_id == "a_new"


def test_chatgpt_position_stays_mapping_order_when_active_path_timestamps_are_scrambled() -> None:
    """Archive row positions remain unique while active-path membership stays explicit."""
    root = _branch_node("root", "system", "", parent=None, children=["u1"])
    user = _branch_node("u1", "user", "first active", parent="root", children=["a1"])
    assistant = _branch_node("a1", "assistant", "second active", parent="u1", children=["u2"])
    followup = _branch_node("u2", "user", "third active", parent="a1", children=[])
    user["message"]["create_time"] = 300.0
    assistant["message"]["create_time"] = 100.0
    followup["message"]["create_time"] = 200.0
    payload = {
        "title": "Scrambled active path",
        "mapping": {n["id"]: n for n in [assistant, followup, root, user]},
        "current_node": "u2",
    }

    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    assert by_id["a1"].position == 0
    assert by_id["u2"].position == 1
    assert by_id["u1"].position == 3
    assert by_id["a1"].is_active_path is True
    assert by_id["u2"].is_active_path is True
    assert by_id["u1"].is_active_path is True
    assert [(m.provider_message_id, m.timestamp) for m in conv.messages] == [
        ("a1", "100.0"),
        ("u2", "200.0"),
        ("u1", "300.0"),
    ]


def test_chatgpt_transport_rows_are_classified_as_protocol_material() -> None:
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["q1"]),
        _branch_node("q1", "assistant", '{"queries":["find context"]}', parent="u1", children=["cmd"]),
        _branch_node("cmd", "assistant", "bash -lc ls -lah", parent="q1", children=["a1"]),
        _branch_node("a1", "assistant", "human-readable answer", parent="cmd", children=[]),
    ]
    payload = {
        "title": "Transport rows",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a1",
        "create_time": 1700000000.0,
    }

    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    assert by_id["q1"].message_type is MessageType.PROTOCOL
    assert by_id["q1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["cmd"].message_type is MessageType.PROTOCOL
    assert by_id["cmd"].material_origin is MaterialOrigin.OPERATOR_COMMAND
    assert by_id["a1"].message_type is MessageType.MESSAGE
    assert by_id["a1"].material_origin is MaterialOrigin.ASSISTANT_AUTHORED


def test_no_current_node_preserves_all_nodes_losslessly() -> None:
    """Without current_node, every node is preserved (lossless fallback) (#1744).

    Real ChatGPT exports always carry current_node; synthetic/edge inputs may
    not. With no active-leaf pointer there is no way to know which branch was
    active, so the fallback keeps all nodes rather than silently dropping one.
    """
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Regenerated no current_node",
        "mapping": {n["id"]: n for n in nodes},
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "OLD answer", "NEW answer"]
    assert [m.is_active_path for m in conv.messages] == [None, None, None]
    assert [m.is_active_leaf for m in conv.messages] == [None, None, None]
    assert conv.active_leaf_message_provider_id is None


def test_current_node_pointing_at_old_branch_marks_that_leaf() -> None:
    """current_node is authoritative active-path metadata, not an emission filter."""
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "kept answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "discarded answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Old branch active",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_old",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "kept answer", "discarded answer"]
    by_id = {m.provider_message_id: m for m in conv.messages}
    assert by_id["a_old"].is_active_path is True
    assert by_id["a_old"].is_active_leaf is True
    assert by_id["a_new"].is_active_path is False
    assert by_id["a_new"].is_active_leaf is False
    assert conv.active_leaf_message_provider_id == "a_old"


def test_chatgpt_archive_contract_fields() -> None:
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW answer", parent="u1", children=[]),
    ]
    nodes[2]["message"]["metadata"] = {"model_slug": "gpt-4o", "durationMs": 2500}
    payload = {
        "title": "Regenerated archive contract",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_new",
        "create_time": 1700000000.0,
    }

    conv = chatgpt_parse(payload, "fallback-id")

    assert [m.position for m in conv.messages] == [0, 1, 2]
    assert [m.variant_index for m in conv.messages] == [0, 0, 1]
    active = next(m for m in conv.messages if m.provider_message_id == "a_new")
    assert active.model_name == "gpt-4o"
    assert active.duration_ms == 2500


def test_chatgpt_generation_timing_uses_one_authoritative_reasoning_owner() -> None:
    """Repeated run-wide metadata must not multiply one Pro generation."""
    nodes = [
        _branch_node("u1", "user", "Do the work", parent=None, children=["thought"]),
        _branch_node("thought", "assistant", "working", parent="u1", children=["recap"]),
        _branch_node("recap", "assistant", "reasoning summary", parent="thought", children=["answer"]),
        _branch_node("answer", "assistant", "substantive answer", parent="recap", children=[]),
    ]
    nodes[1]["message"]["content"]["content_type"] = "thoughts"
    nodes[1]["message"]["metadata"] = {"reasoning_start_time": 1784164544.946}
    nodes[2]["message"]["content"]["content_type"] = "reasoning_recap"
    nodes[2]["message"]["metadata"] = {
        "reasoning_start_time": 1784164541.690012,
        "reasoning_end_time": 1784169732.588194,
        "finished_duration_sec": 5190,
    }
    nodes[3]["message"]["metadata"] = {
        "reasoning_start_time": 1784164544.946,
        "finished_duration_sec": 5190,
    }
    payload = {
        "id": "pro-generation",
        "title": "Long Pro generation",
        "mapping": {node["id"]: node for node in nodes},
        "current_node": "answer",
    }

    session = chatgpt_parse(payload, "fallback-id")
    by_id = {message.provider_message_id: message for message in session.messages}
    lifecycle = [event for event in session.session_events if event.event_type == "generation_lifecycle"]

    assert by_id["recap"].duration_ms == 5_190_000
    assert by_id["thought"].duration_ms is None
    assert by_id["answer"].duration_ms is None
    assert session.reported_duration_ms == 5_190_000
    assert len(lifecycle) == 1
    assert lifecycle[0].source_message_provider_id == "recap"
    assert lifecycle[0].payload == {
        "state": "completed",
        "evidence_source": "provider_native",
        "fidelity": "exact",
        "duration_semantics": "provider_reported_elapsed",
        "elapsed_duration_ms": 5_190_000,
        "started_at_ms": 1_784_164_541_690,
        "ended_at_ms": 1_784_169_732_588,
    }


@pytest.mark.parametrize(
    ("metadata", "expected_duration_ms"),
    [
        ({"reasoning_start_time": 10.0, "reasoning_end_time": 12.25}, 2250),
        ({"reasoning_start_time": 12.0, "reasoning_end_time": 10.0}, None),
        ({"finished_duration_sec": -3}, None),
        ({"finished_duration_sec": "pending"}, None),
    ],
)
def test_chatgpt_generation_timing_fallback_rejects_malformed_values(
    metadata: dict[str, object], expected_duration_ms: int | None
) -> None:
    node = _branch_node("recap", "assistant", "summary", parent=None, children=[])
    node["message"]["content"]["content_type"] = "reasoning_recap"
    node["message"]["metadata"] = metadata

    session = chatgpt_parse(
        {"id": "timing-edge", "mapping": {"recap": node}, "current_node": "recap"},
        "fallback-id",
    )

    assert session.messages[0].duration_ms == expected_duration_ms
    assert session.reported_duration_ms == expected_duration_ms
    assert len(session.session_events) == (1 if expected_duration_ms is not None else 0)


# ---------------------------------------------------------------------------
# #1744 — non-`parts` content is preserved (code interpreter, execution output)
# ---------------------------------------------------------------------------


def test_code_interpreter_content_is_preserved() -> None:
    """A code node carries top-level content.text (no parts) — must not drop (#1744)."""
    nodes = [
        _branch_node("u1", "user", "run this", parent=None, children=["tool"]),
        {
            "id": "tool",
            "message": {
                "id": "tool",
                "author": {"role": "assistant", "name": "python"},
                "content": {"content_type": "code", "text": "print(1)"},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool"},
                "content": {"content_type": "execution_output", "text": "1\n"},
                "create_time": None,
            },
            "parent": "tool",
            "children": [],
        },
    ]
    payload = {
        "title": "Code interpreter",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert "print(1)" in texts
    assert "1\n" in texts
    # Content-block types reflect the code-interpreter semantics.
    from polylogue.core.enums import BlockType

    code_msg = next(m for m in conv.messages if m.text == "print(1)")
    assert any(b.type == BlockType.CODE for b in code_msg.blocks)
    out_msg = next(m for m in conv.messages if m.text == "1\n")
    assert any(b.type == BlockType.TOOL_RESULT for b in out_msg.blocks)


# SANDBOX FILE LINKS (assistant-generated downloadable deliverables)


def test_sandbox_links_become_unfetchable_attachments() -> None:
    text = (
        "Kit delivered.\n\n"
        "**[Download the ZIP](sandbox:/mnt/data/compiler-kit.zip)**\n"
        "[Checksum](sandbox:/mnt/data/compiler-kit.zip.sha256): `abc`\n"
        "Also see [the prompts dir](sandbox:/mnt/data/compiler-kit/prompts/) "
        "and again [the ZIP](sandbox:/mnt/data/compiler-kit.zip)."
    )
    mapping = {
        "node1": make_chatgpt_node("msg1", "assistant", [text]),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    sandbox = [a for a in attachments if a.attachment_kind == "sandbox_file"]
    assert [a.name for a in sandbox] == [
        "compiler-kit.zip",
        "compiler-kit.zip.sha256",
        None,  # directory link keeps trailing slash; no file name
    ]
    assert [a.source_url for a in sandbox] == [
        "sandbox:/mnt/data/compiler-kit.zip",
        "sandbox:/mnt/data/compiler-kit.zip.sha256",
        "sandbox:/mnt/data/compiler-kit/prompts/",
    ]
    assert all(a.message_provider_id == "msg1" for a in sandbox)
    # Duplicate link in the same message is recorded once.
    assert len(sandbox) == 3


def test_sandbox_links_in_user_messages_are_not_attachments() -> None:
    mapping = {
        "node1": make_chatgpt_node("msg1", "user", ["please regenerate sandbox:/mnt/data/old.zip"]),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    assert not [a for a in attachments if a.attachment_kind == "sandbox_file"]


def test_sandbox_link_trailing_punctuation_is_stripped() -> None:
    mapping = {
        "node1": make_chatgpt_node(
            "msg1",
            "assistant",
            ["Saved to sandbox:/mnt/data/report.md. Enjoy, or see sandbox:/mnt/data/data.csv,"],
        ),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    sandbox = [a for a in attachments if a.attachment_kind == "sandbox_file"]
    assert [a.source_url for a in sandbox] == [
        "sandbox:/mnt/data/report.md",
        "sandbox:/mnt/data/data.csv",
    ]


# CITATION MARKER HYGIENE + SYSTEM-INJECTED CONTEXT


def test_citation_markers_are_stripped_but_citations_survive() -> None:
    marked = "Per the brief. \ue200filecite\ue202turn0file0\ue201 More text \ue200cite\ue202turn1search2\ue202L10-L20\ue201 end."
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": [marked]},
                "metadata": {
                    "citations": [
                        {
                            "citation_format_type": "berry_file_search",
                            "start_ix": 15,
                            "end_ix": 43,
                            "metadata": {"title": "brief.md", "url": "https://example.test/brief"},
                        }
                    ]
                },
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == 1
    message = messages[0]
    text = message.text
    assert text is not None
    assert "\ue200" not in text
    assert "\ue201" not in text
    assert "\ue202" not in text
    assert "filecite" not in text
    assert text.startswith("Per the brief.")
    assert text.endswith("end.")
    assert all("\ue200" not in (block.text or "") for block in message.blocks)
    constructs = [c for block in message.blocks for c in block.web_constructs]
    assert any(c.provider_key == "citations" for c in constructs)


def test_user_editable_context_becomes_runtime_context_message() -> None:
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {
                    "content_type": "user_editable_context",
                    "user_profile": "Profile: local-first archivist.",
                    "user_instructions": "Always answer with evidence refs.",
                },
                "metadata": {"is_visually_hidden_from_conversation": True},
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == 1
    message = messages[0]
    assert message.message_type is MessageType.CONTEXT
    assert message.material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert message.text is not None
    assert "Profile: local-first archivist." in message.text
    assert "Always answer with evidence refs." in message.text
    assert message.blocks[0].metadata == {"content_type": "user_editable_context"}


def test_model_editable_context_memory_payload_is_kept_and_empty_is_dropped() -> None:
    def node(msg_id: str, model_set_context: str) -> dict[str, object]:
        return {
            "id": msg_id,
            "message": {
                "id": msg_id,
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "model_editable_context",
                    "model_set_context": model_set_context,
                },
            },
        }

    mapping = {
        "node1": node("msg1", "1. Prefers rigorous verification.\n2. Runs Polylogue."),
        "node2": node("msg2", ""),
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [m.provider_message_id for m in messages] == ["msg1"]
    assert messages[0].message_type is MessageType.CONTEXT
    assert messages[0].text is not None
    assert "Prefers rigorous verification" in messages[0].text


def test_file_citation_nested_metadata_is_surfaced() -> None:
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": ["Cited claim."]},
                "metadata": {
                    "citations": [
                        {
                            "citation_format_type": "berry_file_search",
                            "start_ix": 0,
                            "end_ix": 12,
                            "metadata": {
                                "id": "file_00000000b074",
                                "name": "06-strategy-falsification.md",
                                "source": "my_files",
                                "type": "file",
                                "extra": {
                                    "cited_message_id": "bd889688",
                                    "library_file_id": "libfile_9eba",
                                    "source_url": None,
                                },
                            },
                        }
                    ]
                },
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    constructs = [c for b in messages[0].blocks for c in b.web_constructs if c.provider_key == "citations"]
    assert len(constructs) == 1
    citation = constructs[0]
    # Source identity lives one level down (metadata) and two levels down
    # (metadata.extra); losing it reduces a file citation to a bare span.
    assert citation.title == "06-strategy-falsification.md"
    assert citation.source_id == "file_00000000b074"
    assert citation.start_index == 0
    assert citation.end_index == 12


def test_inline_citation_marker_tokens_become_anchored_constructs() -> None:
    marked = "Claim text. \ue200filecite\ue202turn3file14\ue202L180-L293\ue201 More."
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": [marked]},
                "metadata": {},
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    message = messages[0]
    assert message.text is not None
    assert "\ue200" not in message.text
    markers = [c for b in message.blocks for c in b.web_constructs if c.provider_key == "inline_citation_marker"]
    assert len(markers) == 1
    marker = markers[0]
    # Line ranges often exist ONLY in the marker tokens (metadata line_range
    # is frequently null) — the construct must retain them.
    assert marker.text == "filecite turn3file14 L180-L293"
    # Anchored in ORIGINAL-text coordinates, matching citation start_ix/end_ix.
    assert marker.start_index == marked.index("\ue200")
    assert marker.end_index == marked.index("\ue201") + 1
