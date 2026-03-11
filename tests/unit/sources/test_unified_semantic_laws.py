"""Focused contracts for unified/provider semantic equivalence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import given, settings

from polylogue.lib.provider_semantics import extract_chatgpt_text, extract_claude_code_text
from polylogue.lib.viewports import ContentType
from polylogue.schemas.unified import (
    _missing_role,
    bulk_harmonize,
    extract_content_blocks,
    extract_from_provider_meta,
    extract_harmonized_message,
    extract_reasoning_traces,
    extract_token_usage,
    extract_tool_calls,
    harmonize_parsed_message,
    is_message_record,
)
from polylogue.sources.providers.chatgpt import ChatGPTConversation, ChatGPTMessage
from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage
from polylogue.sources.providers.claude_code import ClaudeCodeRecord
from polylogue.sources.providers.codex import CodexRecord
from polylogue.sources.providers.gemini import GeminiMessage
from tests.infra.strategies import (
    chatgpt_semantic_message_strategy,
    claude_ai_semantic_message_strategy,
    claude_code_semantic_record_strategy,
    codex_semantic_record_strategy,
    gemini_semantic_message_strategy,
    provider_semantic_case_strategy,
)


def _build_viewport_record(provider: str, raw: dict[str, object]) -> object:
    if provider == "chatgpt":
        return ChatGPTMessage.model_validate(raw)
    if provider == "claude-ai":
        return ClaudeAIChatMessage.model_validate(raw)
    if provider == "claude-code":
        return ClaudeCodeRecord.model_validate(raw)
    if provider == "codex":
        return CodexRecord.model_validate(raw)
    if provider == "gemini":
        return GeminiMessage.model_validate(raw)
    raise AssertionError(f"unexpected provider {provider}")


def _structured_provider_meta(record: object) -> dict[str, object]:
    meta = record.to_meta()
    provider_meta: dict[str, object] = {
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
    return provider_meta


def _expected_block_types(provider: str, raw: dict[str, object]) -> list[ContentType]:
    if provider == "chatgpt":
        content_type = raw.get("content", {}).get("content_type")
        if content_type == "text":
            return [ContentType.TEXT]
        if content_type == "code":
            return [ContentType.CODE]
        if isinstance(content_type, str) and ("tether" in content_type or "browse" in content_type):
            return [ContentType.TOOL_RESULT]
        return [ContentType.UNKNOWN]
    if provider == "claude-ai":
        return [ContentType.TEXT]
    if provider == "claude-code":
        raw_blocks = raw.get("message", {}).get("content", [])
        mapping = {
            "text": ContentType.TEXT,
            "thinking": ContentType.THINKING,
            "tool_use": ContentType.TOOL_USE,
            "tool_result": ContentType.TOOL_RESULT,
            "code": ContentType.CODE,
        }
        return [mapping[block["type"]] for block in raw_blocks if isinstance(block, dict) and block.get("type") in mapping]
    if provider == "codex":
        content = raw.get("payload", {}).get("content") if raw.get("type") == "response_item" else raw.get("content", [])
        mapping = {"input_text": ContentType.TEXT, "output_text": ContentType.TEXT, "text": ContentType.TEXT}
        types: list[ContentType] = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type in mapping:
                types.append(mapping[block_type])
            elif "code" in block_type:
                types.append(ContentType.CODE)
            else:
                types.append(ContentType.UNKNOWN)
        return types
    if provider == "gemini":
        types: list[ContentType] = []
        if raw.get("isThought"):
            types.append(ContentType.THINKING)
        elif raw.get("text"):
            types.append(ContentType.TEXT)
        for part in raw.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                types.append(ContentType.TEXT)
            elif isinstance(part, dict) and ("inlineData" in part or "fileData" in part):
                types.append(ContentType.FILE)
        if raw.get("executableCode"):
            types.append(ContentType.CODE)
        if raw.get("codeExecutionResult"):
            types.append(ContentType.TOOL_RESULT)
        return types
    raise AssertionError(provider)


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_unified_semantic_equivalence_for_rich_provider_cases(case: tuple[str, dict[str, object]]) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()

    from_raw = extract_harmonized_message(provider, raw)
    from_provider_meta = extract_from_provider_meta(
        provider,
        {"raw": raw},
        message_id=meta.id,
        role=meta.role,
        text=record.text_content,
        timestamp=meta.timestamp,
    )
    from_structured = extract_from_provider_meta(
        provider,
        _structured_provider_meta(record),
        message_id=meta.id,
        role=meta.role,
        text=record.text_content,
        timestamp=meta.timestamp,
    )

    assert from_raw.role == from_provider_meta.role == from_structured.role
    assert from_raw.text == from_provider_meta.text == from_structured.text == record.text_content
    assert from_raw.content_blocks == from_provider_meta.content_blocks == from_structured.content_blocks
    assert from_raw.reasoning_traces == from_provider_meta.reasoning_traces == from_structured.reasoning_traces
    assert from_raw.tool_calls == from_provider_meta.tool_calls == from_structured.tool_calls


@given(provider_semantic_case_strategy())
@settings(max_examples=30, deadline=None)
def test_harmonize_parsed_message_matches_bulk_harmonize(case: tuple[str, dict[str, object]]) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()
    parsed = SimpleNamespace(
        provider_meta={"raw": raw},
        provider_message_id=meta.id,
        role=meta.role,
        text=record.text_content,
        timestamp=meta.timestamp,
    )

    individual = harmonize_parsed_message(
        provider,
        parsed.provider_meta,
        message_id=parsed.provider_message_id,
        role=parsed.role,
        text=parsed.text,
        timestamp=parsed.timestamp,
    )
    bulk = bulk_harmonize(provider, [parsed])

    assert individual is not None
    assert bulk == [individual]


@pytest.mark.parametrize(
    ("provider", "raw", "expected"),
    [
        ("claude-code", {"type": "user"}, True),
        ("claude-code", {"type": "assistant"}, True),
        ("claude-code", {"type": "system"}, True),
        ("claude-code", {"type": "progress"}, False),
        ("claude-code", {"type": "summary"}, False),
        ("claude-code", {}, True),
        ("chatgpt", {"anything": "goes"}, True),
        ("gemini", {"role": "model"}, True),
    ],
)
def test_is_message_record_contract(provider: str, raw: dict[str, object], expected: bool) -> None:
    assert is_message_record(provider, raw) is expected


def test_missing_role_contract() -> None:
    with pytest.raises(ValueError, match="Message has no role"):
        _missing_role()


@given(
    chatgpt_semantic_message_strategy(),
    claude_ai_semantic_message_strategy(),
    claude_code_semantic_record_strategy(),
    codex_semantic_record_strategy(),
    gemini_semantic_message_strategy(),
)
@settings(max_examples=20, deadline=None)
def test_rich_semantic_cases_are_message_records(
    chatgpt_raw: dict[str, object],
    claude_ai_raw: dict[str, object],
    claude_code_raw: dict[str, object],
    codex_raw: dict[str, object],
    gemini_raw: dict[str, object],
) -> None:
    assert is_message_record("chatgpt", chatgpt_raw)
    assert is_message_record("claude-ai", claude_ai_raw)
    assert is_message_record("claude-code", claude_code_raw)
    assert is_message_record("codex", codex_raw)
    assert is_message_record("gemini", gemini_raw)


def test_chatgpt_iter_user_assistant_pairs_contract() -> None:
    conversation = ChatGPTConversation.model_validate(
        {
            "id": "conv-pairs",
            "conversation_id": "conv-pairs",
            "title": "Pairs",
            "create_time": 1700000000.0,
            "update_time": 1700000001.0,
            "current_node": "n4",
            "mapping": {
                "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                "n1": {
                    "id": "n1",
                    "parent": "n0",
                    "children": ["n2"],
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["q1"]},
                    },
                },
                "n2": {
                    "id": "n2",
                    "parent": "n1",
                    "children": ["n3"],
                    "message": {
                        "id": "m2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["a1"]},
                    },
                },
                "n3": {
                    "id": "n3",
                    "parent": "n2",
                    "children": ["n4"],
                    "message": {
                        "id": "m3",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["q2"]},
                    },
                },
                "n4": {
                    "id": "n4",
                    "parent": "n3",
                    "children": [],
                    "message": {
                        "id": "m4",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["a2"]},
                    },
                },
            },
        }
    )

    pairs = list(conversation.iter_user_assistant_pairs())

    assert [(user.id, assistant.id) for user, assistant in pairs] == [("m1", "m2"), ("m3", "m4")]


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_rich_provider_block_classification_contract(case: tuple[str, dict[str, object]]) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)

    assert [block.type for block in record.extract_content_blocks()] == _expected_block_types(provider, raw)


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_rich_provider_meta_contract(case: tuple[str, dict[str, object]]) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()

    assert meta.provider == provider
    if provider == "chatgpt":
        assert meta.id == raw["id"]
        assert meta.model == raw.get("metadata", {}).get("model_slug")
    elif provider == "claude-ai":
        assert meta.id == raw["uuid"]
        assert meta.role in {"assistant", "user"}
        assert meta.timestamp is not None
    elif provider == "claude-code":
        assert meta.id == raw["uuid"]
        assert meta.duration_ms == raw.get("durationMs")
        if raw.get("costUSD") is None:
            assert meta.cost is None
        else:
            assert meta.cost is not None and meta.cost.total_usd == raw.get("costUSD")
        usage = raw.get("message", {}).get("usage")
        if usage is None:
            assert meta.tokens is None
        else:
            assert meta.tokens is not None
            assert meta.tokens.input_tokens == usage["input_tokens"]
            assert meta.tokens.output_tokens == usage["output_tokens"]
    elif provider == "codex":
        assert meta.id == raw["id"]
        assert meta.role in {"assistant", "user"}
    elif provider == "gemini":
        if raw.get("tokenCount") is None:
            assert meta.tokens is None
        else:
            assert meta.tokens is not None
            assert meta.tokens.output_tokens == raw["tokenCount"]


@given(provider_semantic_case_strategy())
@settings(max_examples=30, deadline=None)
def test_structured_provider_meta_overlay_contract(case: tuple[str, dict[str, object]]) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()

    structured = _structured_provider_meta(record)
    structured.pop("model", None)
    structured.pop("duration_ms", None)

    harmonized = extract_from_provider_meta(
        provider,
        structured,
        message_id=meta.id,
        role=meta.role,
        text=record.text_content,
        timestamp=meta.timestamp,
    )

    assert harmonized.id == meta.id
    assert harmonized.role.value == meta.role
    assert harmonized.text == record.text_content
    assert harmonized.timestamp == meta.timestamp


def test_generic_unified_extractors_preserve_nested_text_and_tool_metadata() -> None:
    content = [
        {"type": "text", "text": "plain"},
        {"type": "thinking", "thinking": "reason"},
        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "README.md"}},
        {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "done"}, {"text": "more"}]},
        {"type": "code", "code": "print('ok')", "language": "python"},
    ]

    blocks = extract_content_blocks(content)
    traces = extract_reasoning_traces(content, "claude-code")
    calls = extract_tool_calls(content, "claude-code")

    assert [block.type for block in blocks] == [
        ContentType.TEXT,
        ContentType.THINKING,
        ContentType.TOOL_USE,
        ContentType.TOOL_RESULT,
        ContentType.CODE,
    ]
    assert blocks[3].text == "done\nmore"
    assert traces[0].text == "reason"
    assert len(calls) == 1
    assert calls[0].name == "Read"
    assert calls[0].input == {"path": "README.md"}


def test_extract_claude_code_text_excludes_non_text_blocks_contract() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "reason"},
        {"type": "tool_result", "content": "done"},
        {"type": "text", "text": "world"},
        "ignored",
    ]

    assert extract_claude_code_text(content) == "hello\nworld"


def test_extract_token_usage_maps_cache_fields_contract() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 12,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4,
        "total_tokens": 29,
    }

    tokens = extract_token_usage(usage)

    assert tokens is not None
    assert tokens.input_tokens == 10
    assert tokens.output_tokens == 12
    assert tokens.cache_read_tokens == 3
    assert tokens.cache_write_tokens == 4
    assert tokens.total_tokens == 29


def test_extract_from_provider_meta_rebuilds_text_and_overlay_metadata_contract() -> None:
    provider_meta = {
        "content_blocks": [
            {"type": "text", "text": "plain text"},
            {"type": "code", "text": "print('x')", "language": "python"},
            {"type": "tool_result", "text": "tool output"},
        ],
        "tokens": {"output_tokens": 5},
        "cost": {"total_usd": 0.25},
        "duration_ms": 12,
    }

    harmonized = extract_from_provider_meta(
        "claude-code",
        provider_meta,
        message_id="msg-1",
        role="assistant",
        timestamp="2025-01-01T00:00:00Z",
    )

    assert harmonized.id == "msg-1"
    assert harmonized.role.value == "assistant"
    assert harmonized.text == "plain text\nprint('x')\ntool output"
    assert harmonized.tokens is not None and harmonized.tokens.output_tokens == 5
    assert harmonized.cost is not None and harmonized.cost.total_usd == 0.25
    assert harmonized.duration_ms == 12


def test_extract_harmonized_message_chatgpt_fallback_preserves_dict_text_parts() -> None:
    raw = {
        "id": "chatgpt-fallback",
        "author": {"role": "assistant"},
        "create_time": 1700000000.0,
        "content": {"parts": ["hello", {"text": "world"}]},
        "metadata": {"model_slug": "gpt-4o"},
    }

    harmonized = extract_harmonized_message("chatgpt", raw)

    assert harmonized.id == "chatgpt-fallback"
    assert harmonized.role.value == "assistant"
    assert harmonized.text == "hello\nworld"


def test_extract_chatgpt_text_prefers_direct_text_contract() -> None:
    content = {
        "content_type": "code",
        "text": "print('ok')",
        "parts": ["ignored", {"text": "still ignored"}],
    }

    assert extract_chatgpt_text(content) == "print('ok')"
