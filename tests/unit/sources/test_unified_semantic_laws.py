"""Focused contracts for unified/provider semantic equivalence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from types import SimpleNamespace
from typing import Protocol, TypeAlias

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.archive.provider.semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_codex_text,
)
from polylogue.lib.json import JSONDocument, json_document, json_document_list
from polylogue.lib.roles import Role
from polylogue.lib.viewport.viewports import (
    ContentBlock,
    ContentType,
    CostInfo,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)
from polylogue.schemas.unified.unified import (
    HarmonizedMessage,
    _coerce_content_blocks,
    _coerce_reasoning_traces,
    _coerce_tool_calls,
    _extract_generic_cost,
    _extract_generic_tokens,
    _harmonize_extracted_provider_meta,
    _has_extracted_viewports,
    _missing_role,
    _overlay_message_context,
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
from polylogue.types import Provider
from tests.infra.strategies import (
    chatgpt_semantic_message_strategy,
    claude_ai_semantic_message_strategy,
    claude_code_semantic_record_strategy,
    code_block_strategy,
    codex_semantic_record_strategy,
    content_block_strategy,
    gemini_semantic_message_strategy,
    provider_semantic_case_strategy,
    text_content_strategy,
    thinking_block_strategy,
    tool_use_block_strategy,
)

_VALID_VIEWPORT_ROLES = {"user", "assistant", "system", "tool", "unknown", "model"}


RawPayload: TypeAlias = JSONDocument
RawContent: TypeAlias = list[JSONDocument]
SemanticCase: TypeAlias = tuple[str, RawPayload]


class ViewportRecord(Protocol):
    @property
    def text_content(self) -> str: ...

    @property
    def role_normalized(self) -> str | Role: ...

    @property
    def parsed_timestamp(self) -> datetime | None: ...

    def to_meta(self) -> MessageMeta: ...

    def extract_content_blocks(self) -> list[ContentBlock]: ...

    def extract_reasoning_traces(self) -> list[ReasoningTrace]: ...

    def extract_tool_calls(self) -> list[ToolCall]: ...


def _provider(value: str | Provider) -> Provider:
    return value if isinstance(value, Provider) else Provider.from_string(value)


def _as_dict(value: object) -> RawPayload:
    return json_document(value)


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _content_blocks(value: list[object]) -> RawContent:
    return json_document_list(value)


def _doc(value: object) -> JSONDocument:
    return json_document(value)


def _build_viewport_record(provider: str, raw: RawPayload) -> ViewportRecord:
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


def _structured_provider_meta(record: ViewportRecord) -> RawPayload:
    meta = record.to_meta()
    provider_meta: RawPayload = {
        "content_blocks": [json_document(block.model_dump(mode="json")) for block in record.extract_content_blocks()],
        "reasoning_traces": [
            json_document(trace.model_dump(mode="json")) for trace in record.extract_reasoning_traces()
        ],
        "tool_calls": [json_document(call.model_dump(mode="json")) for call in record.extract_tool_calls()],
    }
    if meta.model is not None:
        provider_meta["model"] = meta.model
    if meta.tokens is not None:
        provider_meta["tokens"] = json_document(meta.tokens.model_dump(mode="json"))
    if meta.cost is not None:
        provider_meta["cost"] = json_document(meta.cost.model_dump(mode="json"))
    if meta.duration_ms is not None:
        provider_meta["duration_ms"] = meta.duration_ms
    return provider_meta


def _expected_block_types(provider: str, raw: RawPayload) -> list[ContentType]:
    if provider == "chatgpt":
        content_type = _as_dict(raw.get("content")).get("content_type")
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
        raw_block_items = _as_list(_as_dict(raw.get("message")).get("content"))
        mapping = {
            "text": ContentType.TEXT,
            "thinking": ContentType.THINKING,
            "tool_use": ContentType.TOOL_USE,
            "tool_result": ContentType.TOOL_RESULT,
            "code": ContentType.CODE,
        }
        return [
            mapping[block["type"]]
            for block in raw_block_items
            if isinstance(block, dict) and isinstance(block.get("type"), str) and block["type"] in mapping
        ]
    if provider == "codex":
        payload = _as_dict(raw.get("payload"))
        content = payload.get("content") if raw.get("type") == "response_item" else raw.get("content", [])
        mapping = {"input_text": ContentType.TEXT, "output_text": ContentType.TEXT, "text": ContentType.TEXT}
        codex_types: list[ContentType] = []
        for block in _as_list(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type in mapping:
                codex_types.append(mapping[block_type])
            elif "code" in block_type:
                codex_types.append(ContentType.CODE)
            else:
                codex_types.append(ContentType.UNKNOWN)
        return codex_types
    if provider == "gemini":
        gemini_types: list[ContentType] = []
        if raw.get("isThought"):
            gemini_types.append(ContentType.THINKING)
        elif raw.get("text"):
            gemini_types.append(ContentType.TEXT)
        for part in _as_list(raw.get("parts")):
            if isinstance(part, dict) and part.get("text"):
                gemini_types.append(ContentType.TEXT)
            elif isinstance(part, dict) and ("inlineData" in part or "fileData" in part):
                gemini_types.append(ContentType.FILE)
        if raw.get("executableCode"):
            gemini_types.append(ContentType.CODE)
        if raw.get("codeExecutionResult"):
            gemini_types.append(ContentType.TOOL_RESULT)
        return gemini_types
    raise AssertionError(provider)


def _expected_reasoning_texts(content: list[object]) -> list[str]:
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            text = block.get("thinking") or block.get("text")
            if text:
                texts.append(str(text))
        elif block.get("isThought") and block.get("text"):
            texts.append(str(block["text"]))
    return texts


def _expected_tool_blocks(content: list[object]) -> RawContent:
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_use"]


def _expected_content_types(content: list[object]) -> list[ContentType]:
    mapping = {
        "text": ContentType.TEXT,
        "thinking": ContentType.THINKING,
        "tool_use": ContentType.TOOL_USE,
        "tool_result": ContentType.TOOL_RESULT,
        "code": ContentType.CODE,
    }
    return [
        mapping[block_type]
        for block in content
        if isinstance(block, dict) and (block_type := block.get("type", "text")) in mapping
    ]


# =============================================================================
# Property-based laws (@given) — kept unchanged
# =============================================================================


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_unified_semantic_equivalence_for_rich_provider_cases(case: SemanticCase) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()

    from_raw = extract_harmonized_message(provider, raw)
    from_provider_meta = extract_from_provider_meta(
        provider,
        {"raw": raw},
        message_id=meta.id,
        role=meta.role.value,
        text=record.text_content,
        timestamp=meta.timestamp,
    )
    from_structured = extract_from_provider_meta(
        provider,
        _structured_provider_meta(record),
        message_id=meta.id,
        role=meta.role.value,
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
def test_harmonize_parsed_message_matches_bulk_harmonize(case: SemanticCase) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()
    parsed = SimpleNamespace(
        provider_meta={"raw": raw},
        provider_message_id=meta.id,
        role=meta.role.value,
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


@given(
    st.sampled_from(["user", "assistant", "system", "tool", "unknown", "human", "model", "ASSISTANT", "SYSTEM", "USER"])
)
def test_harmonized_message_role_coercion_contract(role_str: str) -> None:
    msg = HarmonizedMessage(role=Role.normalize(role_str), text="test", provider=Provider.CLAUDE_CODE)
    assert msg.role.value in _VALID_VIEWPORT_ROLES


@given(st.sampled_from(["chatgpt", "claude-ai", "claude-code", "gemini", "codex"]))
def test_harmonized_message_provider_coercion_contract(provider_str: str) -> None:
    msg = HarmonizedMessage(role=Role.USER, text="test", provider=_provider(provider_str))
    assert msg.provider.value == provider_str


@given(
    st.lists(
        st.one_of(
            thinking_block_strategy(),
            tool_use_block_strategy(),
            code_block_strategy(),
            text_content_strategy(),
        ),
        max_size=10,
    ),
    st.sampled_from(["claude-code", "claude-ai", "chatgpt", "gemini", "codex"]),
)
@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
def test_typed_content_blocks_extract_without_crash(
    content: RawContent,
    provider: str,
) -> None:
    """Typed content block strategies always produce extractable blocks."""
    blocks = extract_content_blocks(content)
    traces = extract_reasoning_traces(content, provider)
    calls = extract_tool_calls(content, provider)
    assert isinstance(blocks, list)
    assert isinstance(traces, list)
    assert isinstance(calls, list)


@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.fixed_dictionaries({"isThought": st.just(True), "text": st.text(min_size=1, max_size=100)}),
            st.just("not a dict"),
            st.just(42),
        ),
        max_size=10,
    ),
    st.sampled_from(["claude-code", "claude-ai", "chatgpt", "gemini", "codex"]),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_extract_reasoning_traces_preserve_reasoning_blocks_contract(
    content: list[object],
    provider: str,
) -> None:
    traces = extract_reasoning_traces(_content_blocks(content), provider)
    assert [trace.text for trace in traces] == _expected_reasoning_texts(content)
    assert all(str(trace.provider) == provider for trace in traces)


@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.just("not a dict"),
            st.just(None),
        ),
        max_size=10,
    ),
    st.sampled_from(["claude-code", "claude-ai", "chatgpt", "gemini", "codex"]),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_extract_tool_calls_preserve_tool_use_blocks_contract(
    content: list[object],
    provider: str,
) -> None:
    calls = extract_tool_calls(_content_blocks(content), provider)
    expected = _expected_tool_blocks(content)
    assert [call.name for call in calls] == [str(block.get("name", "")) for block in expected]
    assert [call.id for call in calls] == [block.get("id") for block in expected]
    assert [call.input for call in calls] == [
        block.get("input", {}) if isinstance(block.get("input"), dict) else {} for block in expected
    ]
    assert all(str(call.provider) == provider for call in calls)


@given(
    st.lists(
        st.one_of(
            content_block_strategy(),
            st.fixed_dictionaries({"type": st.just("unknown"), "text": st.text(max_size=50)}),
            st.just("not a dict"),
        ),
        max_size=10,
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_extract_content_blocks_preserve_recognized_block_order_contract(
    content: list[object],
) -> None:
    blocks = extract_content_blocks(_content_blocks(content))
    assert [block.type for block in blocks] == _expected_content_types(content)


@given(
    st.lists(
        st.one_of(
            st.fixed_dictionaries(
                {
                    "text": st.text(max_size=40),
                    "input_text": st.text(max_size=40),
                    "output_text": st.text(max_size=40),
                }
            ),
            st.just({"type": "image", "url": "https://example.com"}),
            st.just("not a dict"),
        ),
        max_size=8,
    )
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_extract_codex_text_prefers_first_available_text_field_contract(content: list[object]) -> None:
    expected: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if isinstance(text, str) and text:
            expected.append(text)
    assert extract_codex_text(_content_blocks(content)) == "\n".join(expected)


@given(
    chatgpt_semantic_message_strategy(),
    claude_ai_semantic_message_strategy(),
    claude_code_semantic_record_strategy(),
    codex_semantic_record_strategy(),
    gemini_semantic_message_strategy(),
)
@settings(max_examples=20, deadline=None)
def test_rich_semantic_cases_are_message_records(
    chatgpt_raw: RawPayload,
    claude_ai_raw: RawPayload,
    claude_code_raw: RawPayload,
    codex_raw: RawPayload,
    gemini_raw: RawPayload,
) -> None:
    assert is_message_record("chatgpt", chatgpt_raw)
    assert is_message_record("claude-ai", claude_ai_raw)
    assert is_message_record("claude-code", claude_code_raw)
    assert is_message_record("codex", codex_raw)
    assert is_message_record("gemini", gemini_raw)


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_rich_provider_block_classification_contract(case: SemanticCase) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)

    assert [block.type for block in record.extract_content_blocks()] == _expected_block_types(provider, raw)


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_provider_adapter_viewport_contract(case: SemanticCase) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()
    blocks = record.extract_content_blocks()
    traces = record.extract_reasoning_traces()
    calls = record.extract_tool_calls()

    assert isinstance(record.text_content, str)
    assert isinstance(record.role_normalized, str)
    assert record.role_normalized in _VALID_VIEWPORT_ROLES
    assert record.parsed_timestamp is None or isinstance(record.parsed_timestamp, datetime)
    assert isinstance(meta, MessageMeta)
    assert isinstance(blocks, list)
    assert isinstance(traces, list)
    assert isinstance(calls, list)
    assert meta.provider == _provider(provider)
    assert meta.role.value == record.role_normalized
    assert all(block.raw is not None for block in blocks)

    if isinstance(record, ChatGPTMessage):
        assert record.role_normalized in {"user", "assistant", "system", "tool", "unknown"}
        content = record.content
        assert content is not None
        assert record.text_content == extract_chatgpt_text(_doc(content.model_dump(mode="python")))
        assert blocks == record.to_content_blocks()
        assert not traces
        assert not calls
    elif isinstance(record, ClaudeAIChatMessage):
        assert record.role_normalized in {"user", "assistant", "system", "tool", "unknown"}
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT
        assert blocks[0].text == record.text
        assert record.parsed_timestamp is None or meta.timestamp is not None
        assert not traces
        assert not calls
    elif isinstance(record, ClaudeCodeRecord):
        assert meta.role.value == record.role
        if record.content_blocks_raw:
            assert record.text_content == extract_claude_code_text(record.content_blocks_raw)
        assert blocks == extract_content_blocks(record.content_blocks_raw)
        assert traces == extract_reasoning_traces(record.content_blocks_raw, Provider.CLAUDE_CODE)
        assert calls == extract_tool_calls(record.content_blocks_raw, Provider.CLAUDE_CODE)
    elif isinstance(record, CodexRecord):
        assert record.role_normalized in {"user", "assistant", "system", "tool", "unknown"}
        assert record.text_content == extract_codex_text(record.effective_content)
        assert len(blocks) == len(record.effective_content)
        assert not traces
        assert not calls
    elif isinstance(record, GeminiMessage):
        assert record.role_normalized in {"user", "assistant", "system", "tool", "unknown"}
        if record.tokenCount is None:
            assert meta.tokens is None
        else:
            assert meta.tokens is not None
            assert meta.tokens.output_tokens == record.tokenCount
        if record.isThought and record.text:
            assert traces
            assert blocks
            assert blocks[0].type == ContentType.THINKING
            assert traces[0].text == record.text
        else:
            assert not traces
    else:  # pragma: no cover - guarded by provider_semantic_case_strategy
        raise AssertionError(provider)


@given(provider_semantic_case_strategy())
@settings(max_examples=40, deadline=None)
def test_rich_provider_meta_contract(case: SemanticCase) -> None:
    provider, raw = case
    record = _build_viewport_record(provider, raw)
    meta = record.to_meta()

    assert meta.provider == _provider(provider)
    if provider == "chatgpt":
        assert meta.id == raw["id"]
        assert meta.model == _as_dict(raw.get("metadata")).get("model_slug")
    elif provider == "claude-ai":
        assert meta.id == raw["uuid"]
        assert meta.role.value in {"assistant", "user"}
        assert meta.timestamp is not None
    elif provider == "claude-code":
        assert meta.id == raw["uuid"]
        assert meta.duration_ms == raw.get("durationMs")
        if raw.get("costUSD") is None:
            assert meta.cost is None
        else:
            assert meta.cost is not None and meta.cost.total_usd == raw.get("costUSD")
        usage = _as_dict(_as_dict(raw.get("message")).get("usage"))
        if not usage:
            assert meta.tokens is None
        else:
            assert meta.tokens is not None
            assert meta.tokens.input_tokens == usage["input_tokens"]
            assert meta.tokens.output_tokens == usage["output_tokens"]
    elif provider == "codex":
        assert meta.id == raw["id"]
        assert meta.role.value in {"assistant", "user"}
    elif provider == "gemini":
        if raw.get("tokenCount") is None:
            assert meta.tokens is None
        else:
            assert meta.tokens is not None
            assert meta.tokens.output_tokens == raw["tokenCount"]


@given(provider_semantic_case_strategy())
@settings(max_examples=30, deadline=None)
def test_structured_provider_meta_overlay_contract(case: SemanticCase) -> None:
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
        role=meta.role.value,
        text=record.text_content,
        timestamp=meta.timestamp,
    )

    assert harmonized.id == meta.id
    assert harmonized.role.value == meta.role.value
    assert harmonized.text == record.text_content
    assert harmonized.timestamp == meta.timestamp


# =============================================================================
# Unique hand-written tests (non-consolidatable)
# =============================================================================


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
def test_is_message_record_contract(provider: str, raw: RawPayload, expected: bool) -> None:
    assert is_message_record(provider, raw) is expected


@pytest.mark.parametrize(
    ("label", "action", "match"),
    [
        ("missing-role", lambda: _missing_role(), "Message has no role"),
        ("unknown-provider", lambda: extract_harmonized_message("unknown_provider", _doc({})), "Unknown provider"),
        (
            "claude-code-empty-message",
            lambda: extract_harmonized_message("claude-code", _doc({"uuid": "m1", "type": "human", "message": {}})),
            "no role",
        ),
    ],
)
def test_invalid_semantic_surface_contract(label: str, action: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        action()


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


# =============================================================================
# Consolidated hand-written tests
# =============================================================================


def test_coercion_and_token_cost_internals_contract() -> None:
    """Consolidated: coerce helpers, extract_token_usage, _extract_generic_tokens/cost."""
    # --- Coercion: skip invalid items, inject provider ---
    trace = ReasoningTrace(text="kept-trace", provider=Provider.CLAUDE_CODE)
    call = ToolCall(name="Read", input={"path": "README.md"}, provider=Provider.CLAUDE_CODE)
    block = ContentBlock(type=ContentType.TEXT, text="kept-block", raw={"text": "kept-block"})

    coerced_traces = _coerce_reasoning_traces(
        [trace, {"text": "dict-trace"}, {"text": None}, "ignored"],
        Provider.CLAUDE_CODE,
    )
    coerced_calls = _coerce_tool_calls(
        [call, {"name": "Write", "input": {"path": "notes.txt"}}, {"name": None}, "ignored"],
        Provider.CLAUDE_CODE,
    )
    coerced_blocks = _coerce_content_blocks(
        [block, {"type": "text", "text": "dict-block", "raw": {"text": "dict-block"}}, {"type": "text"}, 123]
    )

    assert [item.text for item in coerced_traces] == ["kept-trace", "dict-trace"]
    assert all(item.provider == Provider.CLAUDE_CODE for item in coerced_traces)
    assert [item.name for item in coerced_calls] == ["Read", "Write"]
    assert all(item.provider == Provider.CLAUDE_CODE for item in coerced_calls)
    assert [item.text for item in coerced_blocks] == ["kept-block", "dict-block", None]

    # --- extract_token_usage with cache fields ---
    usage = _doc(
        {
            "input_tokens": 10,
            "output_tokens": 12,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
            "total_tokens": 29,
        }
    )
    tokens = extract_token_usage(usage)
    assert tokens is not None
    assert tokens.input_tokens == 10
    assert tokens.output_tokens == 12
    assert tokens.cache_read_tokens == 3
    assert tokens.cache_write_tokens == 4
    assert tokens.total_tokens == 29

    # --- _extract_generic_tokens ---
    assert _extract_generic_tokens(_doc({"usage": usage})) == extract_token_usage(usage)
    assert _extract_generic_tokens(_doc({"tokens": _doc({"output_tokens": 7})})) == TokenUsage(output_tokens=7)
    assert _extract_generic_tokens(_doc({"tokenCount": 9})) == TokenUsage(output_tokens=9)
    assert _extract_generic_tokens(_doc({"tokens": _doc({"output_tokens": "bad"})})) is None

    # --- _extract_generic_cost ---
    assert _extract_generic_cost(_doc({"cost": _doc({"total_usd": 0.25})})) == CostInfo(total_usd=0.25)
    assert _extract_generic_cost(_doc({"costUSD": 1.5})) == CostInfo(total_usd=1.5)
    assert _extract_generic_cost(_doc({"cost": _doc({"total_usd": "bad"})})) is None


def test_harmonization_pipeline_internals_contract() -> None:
    """Consolidated: _harmonize_extracted_provider_meta, _has_extracted_viewports, _overlay_message_context."""
    # --- _harmonize_extracted_provider_meta ---
    provider_meta = _doc(
        {
            "content_blocks": [
                {"type": "text", "text": "plain text", "raw": {"text": "plain text"}},
                {"type": "thinking", "text": "private reasoning", "raw": {"thinking": "private reasoning"}},
                {
                    "type": "tool_use",
                    "tool_call": {"name": "Read", "id": "tool-1", "input": {"path": "README.md"}},
                    "raw": {"type": "tool_use"},
                },
            ],
            "tokens": {"output_tokens": 5},
            "costUSD": 0.25,
            "durationMs": 12,
            "sender": "assistant",
            "created_at": "2025-01-01T00:00:00Z",
        }
    )

    harmonized = _harmonize_extracted_provider_meta(
        Provider.CLAUDE_CODE,
        provider_meta,
        message_id="msg-1",
        text=None,
        role=None,
        timestamp=None,
    )

    assert harmonized.id == "msg-1"
    assert harmonized.role.value == "assistant"
    assert harmonized.text == "plain text\nprivate reasoning"
    assert [trace.text for trace in harmonized.reasoning_traces] == ["private reasoning"]
    assert [call.name for call in harmonized.tool_calls] == ["Read"]
    assert harmonized.tokens == TokenUsage(output_tokens=5)
    assert harmonized.cost == CostInfo(total_usd=0.25)
    assert harmonized.duration_ms == 12
    assert harmonized.timestamp is not None

    # --- _has_extracted_viewports ---
    assert _has_extracted_viewports(_doc({"content_blocks": [], "duration_ms": 1})) is True
    assert _has_extracted_viewports(_doc({"sender": "assistant", "text": "hello"})) is False

    # --- _overlay_message_context ---
    message = extract_harmonized_message(
        "claude-ai",
        _doc({"uuid": "m1", "sender": "assistant", "text": "hello", "created_at": "2025-01-01T00:00:00Z"}),
    )
    unknown = message.model_copy(update={"id": None, "role": "unknown", "text": "", "timestamp": None})

    overlaid = _overlay_message_context(
        unknown,
        message_id="db-id",
        role="assistant",
        text="db text",
        timestamp="2025-01-02T00:00:00Z",
    )
    untouched = _overlay_message_context(
        message,
        message_id="db-id",
        role="user",
        text="db text",
        timestamp="2025-01-02T00:00:00Z",
    )

    assert overlaid.id == "db-id"
    assert overlaid.role.value == "assistant"
    assert overlaid.text == "db text"
    assert overlaid.timestamp is not None
    assert untouched == message


def test_generic_content_extraction_contract() -> None:
    """Consolidated: generic extractors with nested text/tool metadata + claude-code text filtering."""
    # --- Generic unified extractors ---
    content = [
        {"type": "text", "text": "plain"},
        {"type": "thinking", "thinking": "reason"},
        {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "README.md"}},
        {
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": [{"type": "text", "text": "done"}, {"text": "more"}],
        },
        {"type": "code", "code": "print('ok')", "language": "python"},
    ]

    blocks = extract_content_blocks(_content_blocks(content))
    traces = extract_reasoning_traces(_content_blocks(content), Provider.CLAUDE_CODE)
    calls = extract_tool_calls(_content_blocks(content), Provider.CLAUDE_CODE)

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

    # --- Claude Code text: excludes non-text blocks ---
    cc_content: list[object] = [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "reason"},
        {"type": "tool_result", "content": "done"},
        {"type": "text", "text": "world"},
        "ignored",
    ]
    assert extract_claude_code_text(_content_blocks(cc_content)) == "hello\nworld"


def test_extract_from_provider_meta_integration_contract() -> None:
    """Consolidated: extract_from_provider_meta with plain blocks, structured blocks+tools, and multi-provider overlay."""
    # --- Plain content blocks with tokens/cost/duration ---
    provider_meta_plain = _doc(
        {
            "content_blocks": [
                {"type": "text", "text": "plain text"},
                {"type": "code", "text": "print('x')", "language": "python"},
                {"type": "tool_result", "text": "tool output"},
            ],
            "tokens": {"output_tokens": 5},
            "cost": {"total_usd": 0.25},
            "duration_ms": 12,
        }
    )

    harmonized = extract_from_provider_meta(
        "claude-code",
        provider_meta_plain,
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

    # --- Structured blocks with tool_calls and reasoning_traces ---
    provider_meta_structured = _doc(
        {
            "content_blocks": [
                {"type": "text", "text": "plain text"},
                {"type": "tool_use", "tool_call": {"name": "Read", "id": "tool-1", "input": {"path": "README.md"}}},
                {"type": "tool_result", "text": "tool output"},
                {"type": "code", "text": "print('x')", "language": "python"},
            ],
            "reasoning_traces": [{"text": "reason", "provider": "claude-code"}],
            "tool_calls": [{"name": "Read", "id": "tool-1", "input": {"path": "README.md"}, "provider": "claude-code"}],
            "tokens": {"output_tokens": 5},
            "cost": {"total_usd": 0.25},
            "duration_ms": 12,
        }
    )

    harmonized2 = extract_from_provider_meta(
        "claude-code",
        provider_meta_structured,
        message_id="msg-1",
        role="assistant",
        timestamp="2025-01-01T00:00:00Z",
    )

    assert harmonized2.id == "msg-1"
    assert harmonized2.role.value == "assistant"
    assert harmonized2.text == "plain text\ntool output\nprint('x')"
    assert [block.type for block in harmonized2.content_blocks] == [
        ContentType.TEXT,
        ContentType.TOOL_USE,
        ContentType.TOOL_RESULT,
        ContentType.CODE,
    ]

    # --- Multi-provider overlay: DB fields fill in when provider_meta lacks them ---
    overlay_cases: list[tuple[str, RawPayload, str, str, str, str]] = [
        ("claude-ai", {"sender": "human", "text": ""}, "db-message-id", "user", "DB fallback text", "claude-ai"),
        (
            "gemini",
            {"role": "model", "text": "", "tokenCount": {"invalid": True}},
            "db-gemini-id",
            "assistant",
            "DB Gemini fallback text",
            "gemini",
        ),
        (
            "codex",
            {"payload": "not-a-dict", "content": []},
            "db-codex-id",
            "assistant",
            "DB Codex fallback text",
            "codex",
        ),
        (
            "codex",
            {"raw": {"payload": "not-a-dict", "content": []}},
            "db-codex-raw-id",
            "assistant",
            "DB raw fallback text",
            "codex",
        ),
    ]
    for provider, provider_meta, message_id, role, text, expected_provider in overlay_cases:
        msg = extract_from_provider_meta(
            provider,
            provider_meta,
            message_id=message_id,
            role=role,
            text=text,
            timestamp="2024-01-15T10:30:00Z",
        )
        assert msg.id == message_id
        assert msg.role.value == role
        assert msg.text == text
        assert msg.timestamp is not None
        assert msg.provider.value == expected_provider


def test_extract_harmonized_message_fallback_contract() -> None:
    """Consolidated: extract_harmonized_message edge cases across providers.

    Covers ChatGPT dict-text-parts fallback, ChatGPT direct text preference,
    Claude Code full semantic fields, Claude Code type fallback,
    and malformed payloads for claude-ai/gemini/codex.
    """
    # --- ChatGPT: dict text parts fallback ---
    chatgpt_raw = {
        "id": "chatgpt-fallback",
        "author": {"role": "assistant"},
        "create_time": 1700000000.0,
        "content": {"parts": ["hello", {"text": "world"}]},
        "metadata": {"model_slug": "gpt-4o"},
    }
    msg_cg = extract_harmonized_message("chatgpt", _doc(chatgpt_raw))
    assert msg_cg.id == "chatgpt-fallback"
    assert msg_cg.role.value == "assistant"
    assert msg_cg.text == "hello\nworld"

    # --- ChatGPT: direct text preferred over parts ---
    content_direct = {
        "content_type": "code",
        "text": "print('ok')",
        "parts": ["ignored", {"text": "still ignored"}],
    }
    assert extract_chatgpt_text(_doc(content_direct)) == "print('ok')"

    # --- Claude Code: full semantic fields preserved ---
    cc_raw = {
        "uuid": "claude-fallback",
        "type": "assistant",
        "timestamp": "2025-01-01T00:00:00Z",
        "durationMs": 99,
        "costUSD": 0.25,
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 12,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
            },
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "reason"},
                {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "README.md"}},
                {"type": "tool_result", "content": [{"type": "text", "text": "done"}]},
                {"type": "code", "code": "print('ok')", "language": "python"},
            ],
        },
    }
    msg_cc = extract_harmonized_message("claude-code", _doc(cc_raw))
    assert msg_cc.id == "claude-fallback"
    assert msg_cc.role.value == "assistant"
    assert msg_cc.text == "hello"
    assert msg_cc.timestamp is not None
    assert msg_cc.duration_ms == 99
    assert msg_cc.model == "claude-sonnet-4"
    assert msg_cc.cost is not None and msg_cc.cost.total_usd == 0.25
    assert msg_cc.tokens is not None
    assert msg_cc.tokens.input_tokens == 10
    assert msg_cc.tokens.output_tokens == 12
    assert msg_cc.tokens.cache_read_tokens == 2
    assert msg_cc.tokens.cache_write_tokens == 3
    assert [block.type for block in msg_cc.content_blocks] == [
        ContentType.TEXT,
        ContentType.THINKING,
        ContentType.TOOL_USE,
        ContentType.TOOL_RESULT,
        ContentType.CODE,
    ]
    assert msg_cc.reasoning_traces and msg_cc.reasoning_traces[0].text == "reason"
    assert msg_cc.tool_calls and msg_cc.tool_calls[0].name == "Read"
    assert msg_cc.tool_calls[0].input == {"path": "README.md"}

    # --- Claude Code: type field fallback for role ---
    claude_code_role_fallbacks: list[tuple[RawPayload, str]] = [
        ({"uuid": "m1", "type": "human", "message": "not-a-dict"}, "user"),
        ({"uuid": "m1", "type": "assistant", "message": "not-a-dict"}, "assistant"),
    ]
    for raw, expected_role in claude_code_role_fallbacks:
        msg = extract_harmonized_message("claude-code", _doc(raw))
        assert msg.role.value == expected_role
        assert msg.id == "m1"

    # --- Malformed payloads across providers ---
    malformed_cases: list[tuple[str, RawPayload, str, str, str, str | None]] = [
        (
            "claude-ai",
            {"sender": "human", "text": "Fallback Claude AI text", "created_at": "2024-01-15T10:30:00Z"},
            "user",
            "Fallback Claude AI text",
            "claude-ai",
            None,
        ),
        (
            "gemini",
            {
                "role": "model",
                "text": "Fallback Gemini text",
                "tokenCount": {"invalid": True},
                "isThought": True,
                "thinkingBudget": 256,
            },
            "assistant",
            "Fallback Gemini text",
            "gemini",
            "Fallback Gemini text",
        ),
        (
            "codex",
            {
                "id": "codex-fallback",
                "timestamp": "2024-01-15T10:30:00Z",
                "payload": "not-a-dict",
                "content": [{"type": "input_text", "text": "Fallback Codex text"}],
            },
            "unknown",
            "Fallback Codex text",
            "codex",
            None,
        ),
    ]
    for provider, raw, expected_role, expected_text, expected_provider, reasoning_text in malformed_cases:
        msg = extract_harmonized_message(provider, raw)
        assert msg.role.value == expected_role
        assert msg.text == expected_text
        assert msg.provider.value == expected_provider
        if reasoning_text is None:
            assert not msg.reasoning_traces
        else:
            assert [trace.text for trace in msg.reasoning_traces] == [reasoning_text]


def test_provider_content_block_detail_contract() -> None:
    """Consolidated: ChatGPT and Codex content block field details (type, text, language, raw)."""
    # --- ChatGPT content types ---
    for content_type, parts, language, expected_type, expected_text in [
        ("text", ["plain"], None, ContentType.TEXT, "plain"),
        ("code", ["print('hi')"], "python", ContentType.CODE, "print('hi')"),
        ("tether_browsing_display", ["search hit"], None, ContentType.TOOL_RESULT, "search hit"),
        ("multimodal_image", ["opaque"], None, ContentType.UNKNOWN, "opaque"),
    ]:
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
        assert message.content is not None
        assert blocks[0].raw == message.content.model_dump()
        assert blocks[0].language == language
        assert meta.id == "chatgpt-blocks"
        assert meta.role == "assistant"
        assert meta.provider == "chatgpt"

    # --- Codex content blocks ---
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
