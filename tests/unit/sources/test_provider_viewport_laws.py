"""Law-based contracts for provider viewport adapters."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import assume, given, settings

from polylogue.lib.provider_semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_codex_text,
)
from polylogue.lib.viewports import ContentType
from polylogue.schemas.unified import (
    extract_content_blocks,
    extract_from_provider_meta,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.sources.providers.chatgpt import ChatGPTMessage
from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.codex import CodexRecord
from polylogue.sources.providers.gemini import GeminiMessage
from polylogue.types import Provider
from tests.infra.strategies import (
    chatgpt_message_node_strategy,
    claude_ai_message_strategy,
    claude_code_message_strategy,
    codex_message_strategy,
    gemini_message_strategy,
)

_CANONICAL_ROLES = {"user", "assistant", "system", "tool", "unknown"}


def _assert_harmonized_matches_record(provider_name: str, raw: dict[str, object], record: object) -> None:
    harmonized = extract_harmonized_message(provider_name, raw)
    meta = record.to_meta()

    assert harmonized.provider == Provider.from_string(provider_name)
    assert harmonized.id == meta.id
    assert harmonized.role.value == meta.role
    assert harmonized.text == record.text_content
    assert harmonized.timestamp == meta.timestamp
    assert harmonized.model == meta.model
    assert harmonized.tokens == meta.tokens
    assert harmonized.cost == meta.cost
    assert harmonized.duration_ms == meta.duration_ms
    assert harmonized.content_blocks == record.extract_content_blocks()
    assert harmonized.reasoning_traces == record.extract_reasoning_traces()
    assert harmonized.tool_calls == record.extract_tool_calls()


def _assert_structured_provider_meta_round_trip(provider_name: str, record: object) -> None:
    meta = record.to_meta()
    provider_meta = {
        "content_blocks": [block.model_dump(mode="json") for block in record.extract_content_blocks()],
        "reasoning_traces": [trace.model_dump(mode="json") for trace in record.extract_reasoning_traces()],
        "tool_calls": [call.model_dump(mode="json") for call in record.extract_tool_calls()],
    }
    if meta.model is not None:
        provider_meta["model"] = meta.model
    if meta.tokens is not None:
        provider_meta["tokens"] = meta.tokens.model_dump(mode="json")
    if meta.cost is not None:
        provider_meta["cost"] = meta.cost.model_dump(mode="json")
    if meta.duration_ms is not None:
        provider_meta["duration_ms"] = meta.duration_ms

    harmonized = extract_from_provider_meta(
        provider_name,
        provider_meta,
        message_id=meta.id,
        role=meta.role,
        text=record.text_content,
        timestamp=meta.timestamp,
    )

    assert harmonized.provider == Provider.from_string(provider_name)
    assert harmonized.id == meta.id
    assert harmonized.role.value == meta.role
    assert harmonized.text == record.text_content
    assert harmonized.timestamp == meta.timestamp
    assert harmonized.content_blocks == record.extract_content_blocks()
    assert harmonized.reasoning_traces == record.extract_reasoning_traces()
    assert harmonized.tool_calls == record.extract_tool_calls()


@given(claude_ai_message_strategy())
@settings(max_examples=40, deadline=None)
def test_claude_ai_message_viewports_preserve_text_and_role(raw: dict[str, object]) -> None:
    message = ClaudeAIChatMessage.model_validate(raw)
    meta = message.to_meta()
    blocks = message.to_content_blocks()

    assert message.role_normalized in _CANONICAL_ROLES
    assert meta.role == message.role_normalized
    assert meta.provider == "claude-ai"
    assert len(blocks) == 1
    assert blocks[0].type == ContentType.TEXT
    assert blocks[0].text == message.text
    assert message.parsed_timestamp is None or isinstance(message.parsed_timestamp, datetime)
    _assert_harmonized_matches_record("claude-ai", raw, message)
    _assert_structured_provider_meta_round_trip("claude-ai", message)


@given(chatgpt_message_node_strategy())
@settings(max_examples=40, deadline=None)
def test_chatgpt_message_viewports_are_self_consistent(node: dict[str, object]) -> None:
    raw_message = node.get("message")
    assume(isinstance(raw_message, dict))
    message = ChatGPTMessage.model_validate(raw_message)

    meta = message.to_meta()
    blocks = message.extract_content_blocks()

    assert meta.role == message.role_normalized
    assert meta.provider == "chatgpt"
    assert message.role_normalized in _CANONICAL_ROLES
    assert message.text_content == extract_chatgpt_text(message.content.model_dump(mode="python"))
    assert blocks == message.to_content_blocks()
    assert all(block.raw for block in blocks)
    _assert_harmonized_matches_record("chatgpt", raw_message, message)
    _assert_structured_provider_meta_round_trip("chatgpt", message)


@given(claude_code_message_strategy())
@settings(max_examples=40, deadline=None)
def test_claude_code_record_viewports_match_unified_extractors(raw: dict[str, object]) -> None:
    record = ClaudeCodeRecord.model_validate(raw)
    meta = record.to_meta()

    assert meta.provider == "claude-code"
    assert meta.role == record.role
    assert record.parsed_timestamp is None or isinstance(record.parsed_timestamp, datetime)
    if record.content_blocks_raw:
        assert record.text_content == extract_claude_code_text(record.content_blocks_raw)
    assert record.extract_content_blocks() == extract_content_blocks(record.content_blocks_raw)
    assert record.extract_reasoning_traces() == extract_reasoning_traces(record.content_blocks_raw, "claude-code")
    assert record.extract_tool_calls() == extract_tool_calls(record.content_blocks_raw, "claude-code")
    _assert_harmonized_matches_record("claude-code", raw, record)
    _assert_structured_provider_meta_round_trip("claude-code", record)


@given(codex_message_strategy())
@settings(max_examples=40, deadline=None)
def test_codex_record_meta_and_block_classification_are_consistent(raw: dict[str, object]) -> None:
    record = CodexRecord.model_validate(raw)
    meta = record.to_meta()
    blocks = record.extract_content_blocks()

    assert meta.provider == "codex"
    assert meta.role == record.role_normalized
    assert record.role_normalized in _CANONICAL_ROLES
    assert record.text_content == extract_codex_text(record.effective_content)
    assert len(blocks) == len(record.effective_content)

    expected_types = []
    for block in record.effective_content:
        block_type = block.get("type", "")
        if block_type in {"input_text", "output_text", "text"}:
            expected_types.append(ContentType.TEXT)
        elif "code" in block_type:
            expected_types.append(ContentType.CODE)
        else:
            expected_types.append(ContentType.UNKNOWN)
    assert [block.type for block in blocks] == expected_types
    _assert_harmonized_matches_record("codex", raw, record)
    _assert_structured_provider_meta_round_trip("codex", record)


@given(gemini_message_strategy())
@settings(max_examples=40, deadline=None)
def test_gemini_message_viewports_are_consistent(raw: dict[str, object]) -> None:
    message = GeminiMessage.model_validate(raw)
    meta = message.to_meta()
    blocks = message.extract_content_blocks()
    traces = message.extract_reasoning_traces()

    assert meta.provider == "gemini"
    assert meta.role == message.role_normalized
    assert message.role_normalized in _CANONICAL_ROLES
    if message.tokenCount is None:
        assert meta.tokens is None
    else:
        assert meta.tokens is not None
        assert meta.tokens.output_tokens == message.tokenCount

    if message.isThought and message.text:
        assert traces
        assert blocks
        assert blocks[0].type == ContentType.THINKING
        assert traces[0].text == message.text
    else:
        assert not traces
    _assert_harmonized_matches_record("gemini", raw, message)
    _assert_structured_provider_meta_round_trip("gemini", message)


@pytest.mark.parametrize(
    ("content_type", "parts", "language", "expected_type", "expected_text"),
    [
        ("text", ["plain"], None, ContentType.TEXT, "plain"),
        ("code", ["print('hi')"], "python", ContentType.CODE, "print('hi')"),
        ("tether_browsing_display", ["search hit"], None, ContentType.TOOL_RESULT, "search hit"),
        ("multimodal_image", ["opaque"], None, ContentType.UNKNOWN, "opaque"),
    ],
)
def test_chatgpt_to_content_blocks_preserve_type_text_and_raw_contract(
    content_type: str,
    parts: list[str],
    language: str | None,
    expected_type: ContentType,
    expected_text: str,
) -> None:
    message = ChatGPTMessage.model_validate(
        {
            "id": "chatgpt-blocks",
            "author": {"role": "assistant"},
            "content": {"content_type": content_type, "parts": parts, "language": language},
            "create_time": 1700000000.0,
        }
    )

    blocks = message.to_content_blocks()
    meta = message.to_meta()

    assert len(blocks) == 1
    assert blocks[0].type == expected_type
    assert blocks[0].text == expected_text
    assert blocks[0].raw == message.content.model_dump()
    assert blocks[0].language == language
    assert meta.id == "chatgpt-blocks"
    assert meta.role == "assistant"
    assert meta.provider == "chatgpt"


def test_codex_viewport_blocks_preserve_text_language_and_unknown_contract() -> None:
    record = CodexRecord.model_validate(
        {
            "id": "codex-blocks",
            "role": "assistant",
            "timestamp": "2024-06-15T10:30:00Z",
            "content": [
                {"type": "output_text", "output_text": "result"},
                {"type": "code_output", "code": "print('hi')", "language": "python"},
                {"type": "image_data", "text": "opaque"},
            ],
        }
    )

    blocks = record.extract_content_blocks()
    meta = record.to_meta()

    assert [block.type for block in blocks] == [ContentType.TEXT, ContentType.CODE, ContentType.UNKNOWN]
    assert blocks[0].text == "result"
    assert blocks[1].text == "print('hi')"
    assert blocks[1].language == "python"
    assert blocks[2].text == "opaque"
    assert [block.raw for block in blocks] == record.effective_content
    assert meta.id == "codex-blocks"
    assert meta.role == "assistant"
    assert meta.provider == "codex"
