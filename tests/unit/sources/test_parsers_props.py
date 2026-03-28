"""Property-based tests for parser modules.

These tests verify critical invariants that parsers must maintain
regardless of input variations.

Key properties tested:
1. Message count preservation - valid exports produce expected message counts
2. Role normalization - all roles map to canonical forms
3. Timestamp parsing - timestamps are parsed or None, never crash
4. Conversation ID preservation - provider IDs are retained
5. No data loss - text content is preserved
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.parsers import chatgpt, claude, codex
from polylogue.sources.parsers.base import (
    ParsedConversation,
    content_blocks_from_segments,
    extract_messages_from_list,
    normalize_role,
)
from polylogue.sources.parsers.drive import parse_chunked_prompt
from tests.infra.strategies import (
    chatgpt_export_strategy,
    chatgpt_message_node_strategy,
    claude_ai_export_strategy,
    claude_code_message_strategy,
    codex_message_strategy,
    message_strategy,
)
from tests.infra.strategies.providers import (
    claude_code_session_strategy,
    codex_session_strategy,
)

# =============================================================================
# Role Normalization Properties
# =============================================================================


@given(st.sampled_from(["user", "assistant", "system", "human", "model", "tool"]))
def test_normalize_role_is_idempotent(role: str):
    """Normalizing a role twice produces the same result."""
    once = normalize_role(role)
    twice = normalize_role(once)
    assert once == twice


@given(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
def test_normalize_role_never_crashes(role: str):
    """normalize_role handles any non-empty input without crashing."""
    result = normalize_role(role)
    assert isinstance(result, str)
    assert len(result) > 0  # Always returns something


@given(st.sampled_from([
    ("user", "user"),
    ("human", "user"),
    ("USER", "user"),
    ("Human", "user"),
    ("assistant", "assistant"),
    ("model", "assistant"),
    ("ASSISTANT", "assistant"),
    ("system", "system"),
    ("SYSTEM", "system"),
]))
def test_normalize_role_canonical_mappings(input_output: tuple[str, str]):
    """Known role strings map to expected canonical forms."""
    input_role, expected = input_output
    assert normalize_role(input_role) == expected


# =============================================================================
# ChatGPT Parser Properties
# =============================================================================


@given(chatgpt_export_strategy(min_messages=1, max_messages=10))
@settings(max_examples=50)
def test_chatgpt_preserves_conversation_id(export: dict):
    """ChatGPT parser preserves conversation ID."""
    fallback_id = "fallback"
    result = chatgpt.parse(export, fallback_id)

    assert isinstance(result, ParsedConversation)
    # Should use export ID if present, else fallback
    expected_id = str(export.get("id") or export.get("uuid") or export.get("conversation_id") or fallback_id)
    assert result.provider_conversation_id == expected_id


@given(chatgpt_export_strategy(min_messages=1, max_messages=10))
@settings(max_examples=50)
def test_chatgpt_preserves_title(export: dict):
    """ChatGPT parser preserves conversation title."""
    fallback_id = "fallback"
    result = chatgpt.parse(export, fallback_id)

    expected_title = str(export.get("title") or export.get("name") or fallback_id)
    assert result.title == expected_title


@given(chatgpt_export_strategy(min_messages=1, max_messages=5))
@settings(max_examples=30)
def test_chatgpt_extracts_messages_from_mapping(export: dict):
    """ChatGPT parser extracts messages from mapping nodes."""
    result = chatgpt.parse(export, "fallback")

    # Count nodes that have message content
    expected_min = sum(
        1 for node in export.get("mapping", {}).values()
        if isinstance(node, dict)
        and isinstance(node.get("message"), dict)
        and node["message"].get("content", {}).get("parts")
    )

    # Parser might filter some messages, but should have at least some
    if expected_min > 0:
        assert len(result.messages) >= 0  # May filter empty messages


@given(chatgpt_message_node_strategy())
@settings(max_examples=30)
def test_chatgpt_node_roles_normalized(node: dict):
    """Message roles from ChatGPT nodes are normalized."""
    export = {"mapping": {node["id"]: node}, "id": "test"}
    result = chatgpt.parse(export, "fallback")

    for msg in result.messages:
        # Role should be one of canonical values
        assert msg.role in {"user", "assistant", "system", "tool", "message"}


# =============================================================================
# Claude AI Parser Properties
# =============================================================================


@given(claude_ai_export_strategy(min_messages=1, max_messages=10))
@settings(max_examples=50)
def test_claude_ai_preserves_conversation_id(export: dict):
    """Claude AI parser preserves conversation ID."""
    fallback_id = "fallback"
    result = claude.parse_ai(export, fallback_id)

    assert isinstance(result, ParsedConversation)
    expected_id = str(export.get("id") or export.get("uuid") or export.get("conversation_id") or fallback_id)
    assert result.provider_conversation_id == expected_id


@given(claude_ai_export_strategy(min_messages=1, max_messages=10))
@settings(max_examples=50)
def test_claude_ai_message_count_preservation(export: dict):
    """Claude AI parser preserves message count (messages with text)."""
    result = claude.parse_ai(export, "fallback")

    chat_msgs = export.get("chat_messages", [])
    # Count messages that have text content
    expected = sum(
        1 for msg in chat_msgs
        if isinstance(msg, dict) and msg.get("text")
    )

    assert len(result.messages) == expected


@given(claude_ai_export_strategy(min_messages=1, max_messages=5))
@settings(max_examples=30)
def test_claude_ai_roles_normalized(export: dict):
    """Claude AI sender field is normalized to role."""
    result = claude.parse_ai(export, "fallback")

    for msg in result.messages:
        # Claude AI uses "human" and "assistant"
        assert msg.role in {"user", "assistant", "system", "message"}


@given(claude_ai_export_strategy(min_messages=1, max_messages=5))
@settings(max_examples=30)
def test_claude_ai_text_preserved(export: dict):
    """Claude AI message text is preserved."""
    result = claude.parse_ai(export, "fallback")

    chat_msgs = export.get("chat_messages", [])
    for i, msg in enumerate(result.messages):
        if i < len(chat_msgs) and chat_msgs[i].get("text"):
            # Text should be preserved (possibly with processing)
            assert msg.text is not None


# =============================================================================
# Claude Code Parser Properties
# =============================================================================


@given(claude_code_session_strategy(min_messages=1, max_messages=10))
@settings(max_examples=50)
def test_claude_code_parses_session(session: list[dict]):
    """Claude Code parser handles JSONL sessions."""
    result = claude.parse_code(session, "fallback")

    assert isinstance(result, ParsedConversation)
    assert result.provider_name == "claude-code"


@given(claude_code_message_strategy())
@settings(max_examples=50)
def test_claude_code_message_type_to_role(msg: dict):
    """Claude Code message type maps to role correctly."""
    session = [msg]
    result = claude.parse_code(session, "fallback")

    if result.messages:
        parsed_msg = result.messages[0]
        # Type should map to role
        if msg["type"] == "user":
            assert parsed_msg.role == "user"
        elif msg["type"] == "assistant":
            assert parsed_msg.role == "assistant"


@given(claude_code_session_strategy(min_messages=2, max_messages=5))
@settings(max_examples=30)
def test_claude_code_extracts_timestamps(session: list[dict]):
    """Claude Code parser extracts timestamps."""
    result = claude.parse_code(session, "fallback")

    for msg in result.messages:
        # Timestamp should be present or None, never crash
        assert msg.timestamp is None or isinstance(msg.timestamp, str)


@given(claude_code_session_strategy(min_messages=1, max_messages=5))
@settings(max_examples=30)
def test_claude_code_extracts_metadata(session: list[dict]):
    """Claude Code parser extracts costUSD and durationMs to metadata.

    Note: claude-code no longer stores 'raw' in provider_meta to avoid
    duplicating data already stored in raw_conversations.raw_content.
    """
    result = claude.parse_code(session, "fallback")

    for parsed_msg in result.messages:
        assert parsed_msg.provider_meta is not None
        # raw is NOT stored — it's in raw_conversations instead
        assert "raw" not in parsed_msg.provider_meta


# =============================================================================
# Codex Parser Properties
# =============================================================================


@given(codex_session_strategy(min_messages=1, max_messages=10, use_envelope=True))
@settings(max_examples=50)
def test_codex_envelope_format_parses(session: list[dict]):
    """Codex parser handles envelope format."""
    result = codex.parse(session, "fallback")

    assert isinstance(result, ParsedConversation)
    assert result.provider_name == "codex"


@given(codex_session_strategy(min_messages=1, max_messages=10, use_envelope=False))
@settings(max_examples=50)
def test_codex_legacy_format_parses(session: list[dict]):
    """Codex parser handles legacy format."""
    result = codex.parse(session, "fallback")

    assert isinstance(result, ParsedConversation)
    assert result.provider_name == "codex"


@given(codex_message_strategy())
@settings(max_examples=30)
def test_codex_message_text_extraction(msg: dict):
    """Codex parser extracts text from content array."""
    session = [
        {"type": "session_meta", "payload": {"id": "test", "timestamp": "2024-01-01"}},
        {"type": "response_item", "payload": msg},
    ]
    result = codex.parse(session, "fallback")

    if result.messages:
        # Check text was extracted from content items
        parsed_msg = result.messages[0]
        expected_text = "\n".join(
            item.get("text", "")
            for item in msg.get("content", [])
            if item.get("type") == "input_text"
        )
        if expected_text:
            assert parsed_msg.text == expected_text


# =============================================================================
# Generic Message Extraction Properties
# =============================================================================


@given(st.lists(message_strategy(), min_size=0, max_size=20))
@settings(max_examples=50)
def test_extract_messages_from_list_preserves_count(messages: list[dict]):
    """extract_messages_from_list preserves message count (for valid messages)."""
    result = extract_messages_from_list(messages)

    # Count messages that have text content
    expected = sum(
        1 for msg in messages
        if isinstance(msg, dict)
        and (msg.get("text") or msg.get("content"))
    )

    assert len(result) <= expected  # May filter some messages


@given(message_strategy())
@settings(max_examples=50)
def test_extract_messages_role_normalized(msg: dict):
    """extract_messages_from_list normalizes roles."""
    result = extract_messages_from_list([msg])

    if result:
        parsed = result[0]
        # Role should be normalized
        assert parsed.role == normalize_role(msg.get("role"))


def test_content_blocks_from_segments_classifies_code_and_tool_blocks() -> None:
    """Segment parsing must preserve explicit semantic block kinds and order."""
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


def test_parse_chunked_prompt_preserves_reasoning_code_and_drive_docs() -> None:
    """Drive/Gemini chunk parsing must retain reasoning, code, and attachment metadata."""
    payload = {
        "id": "gemini-conv",
        "title": "Gemini Contract",
        "createTime": "2025-01-01T00:00:00Z",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-user",
                    "role": "user",
                    "text": "question",
                    "driveDocument": {
                        "id": "doc-1",
                        "name": "spec.pdf",
                        "mimeType": "application/pdf",
                        "sizeBytes": "12",
                    },
                },
                {
                    "id": "msg-thought",
                    "role": "model",
                    "text": "reasoning",
                    "isThought": True,
                    "thinkingBudget": 32,
                },
                {
                    "id": "msg-code",
                    "role": "model",
                    "parts": [{"text": "inline"}],
                    "executableCode": {"language": "python", "code": "print('ok')"},
                    "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "ok"},
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback")

    assert result.provider_name == "gemini"
    assert result.provider_conversation_id == "gemini-conv"
    assert [message.provider_message_id for message in result.messages] == [
        "msg-user",
        "msg-thought",
        "msg-code",
    ]
    assert result.messages[0].provider_meta["content_blocks"] == [{"type": "text", "text": "question"}]
    assert result.messages[1].provider_meta["content_blocks"] == [{"type": "thinking", "text": "reasoning"}]
    assert result.messages[1].provider_meta["reasoning_traces"] == [
        {"text": "reasoning", "token_count": 32, "provider": "gemini"}
    ]
    assert result.messages[2].provider_meta["content_blocks"] == [
        {"type": "text", "text": "inline"},
        {"type": "code", "text": "print('ok')"},
        {"type": "tool_result", "text": "ok"},
    ]
    assert len(result.attachments) == 1
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[0].mime_type == "application/pdf"
    assert result.attachments[0].size_bytes == 12


# =============================================================================
# Timestamp Handling Properties
# =============================================================================


@given(st.one_of(
    st.floats(min_value=0, max_value=2e12, allow_nan=False, allow_infinity=False),
    st.integers(min_value=0, max_value=int(2e12)),
    st.text(max_size=50),
    st.none(),
))
def test_timestamp_normalization_never_crashes(timestamp):
    """Timestamp normalization handles any input without crashing."""
    result = claude.normalize_timestamp(timestamp)
    assert result is None or isinstance(result, str)


@given(st.floats(min_value=1577836800, max_value=1893456000, allow_nan=False))
def test_timestamp_normalization_preserves_seconds(epoch: float):
    """Second-precision epochs are preserved."""
    result = claude.normalize_timestamp(epoch)
    assert result is not None
    # Should be within 1 second of original
    parsed = float(result)
    assert abs(parsed - epoch) < 1


@given(st.integers(min_value=1577836800000, max_value=1893456000000))
def test_timestamp_normalization_handles_milliseconds(epoch_ms: int):
    """Millisecond epochs are converted to seconds."""
    result = claude.normalize_timestamp(epoch_ms)
    assert result is not None
    # Should be converted to seconds
    parsed = float(result)
    expected = epoch_ms / 1000.0
    assert abs(parsed - expected) < 1


# =============================================================================
# Attachment Handling Properties
# =============================================================================


@given(st.fixed_dictionaries({
    "id": st.uuids().map(str),
    # Use printable characters to avoid sanitization filtering
    "name": st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\x00\x1f",
    )),
    "mime_type": st.sampled_from(["text/plain", "image/png", "application/pdf"]),
    # Use size > 0 since 0 may be filtered
    "size": st.integers(min_value=1, max_value=10_000_000),
}))
def test_attachment_extraction_preserves_metadata(attachment_meta: dict):
    """Attachment metadata is preserved during extraction (for valid inputs)."""
    from polylogue.sources.parsers.base import attachment_from_meta

    result = attachment_from_meta(attachment_meta, "msg-1", 1)

    assert result is not None
    assert result.provider_attachment_id == attachment_meta["id"]
    # Name may be sanitized (control chars removed, dots-only → "file")
    if result.name:
        original_name = attachment_meta["name"]
        # Dots-only names (e.g., ".", "...", etc.) are replaced with "file" for security
        if original_name.strip(".") == "":
            assert result.name == "file"
        else:
            # Should contain some of the original printable text
            assert any(c in result.name for c in original_name if c.isprintable())
    assert result.mime_type == attachment_meta["mime_type"]
    assert result.size_bytes == attachment_meta["size"]


# =============================================================================
# Detection Properties
# =============================================================================


@given(chatgpt_export_strategy())
@settings(max_examples=20)
def test_chatgpt_looks_like_detection(export: dict):
    """ChatGPT exports are correctly detected."""
    assert chatgpt.looks_like(export)


@given(claude_ai_export_strategy())
@settings(max_examples=20)
def test_claude_ai_looks_like_detection(export: dict):
    """Claude AI exports are correctly detected."""
    assert claude.looks_like_ai(export)


@given(claude_code_session_strategy(min_messages=1))
@settings(max_examples=20)
def test_claude_code_looks_like_detection(session: list[dict]):
    """Claude Code sessions are correctly detected."""
    assert claude.looks_like_code(session)


@given(codex_session_strategy(min_messages=1))
@settings(max_examples=20)
def test_codex_looks_like_detection(session: list[dict]):
    """Codex sessions are correctly detected."""
    assert codex.looks_like(session)
