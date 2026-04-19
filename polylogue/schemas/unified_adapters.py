"""Typed provider-adapter routing for harmonized messages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeAlias, cast

from polylogue.lib.raw_payload_decode import JSONRecord
from polylogue.lib.viewports import ContentBlock, MessageMeta, ReasoningTrace, ToolCall
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role
from polylogue.types import Provider


class ViewportRecord(Protocol):
    @property
    def text_content(self) -> str: ...

    def to_meta(self) -> MessageMeta: ...

    def extract_reasoning_traces(self) -> list[ReasoningTrace]: ...

    def extract_tool_calls(self) -> list[ToolCall]: ...

    def extract_content_blocks(self) -> list[ContentBlock]: ...


AdapterBuilder: TypeAlias = Callable[[JSONRecord], ViewportRecord]


def _harmonize_viewport_message(
    provider: Provider,
    raw: JSONRecord,
    message: ViewportRecord,
) -> HarmonizedMessage:
    """Build a harmonized message from a typed provider adapter."""
    meta = message.to_meta()
    return HarmonizedMessage(
        id=meta.id,
        role=meta.role or _missing_role(),
        text=message.text_content,
        timestamp=meta.timestamp,
        reasoning_traces=message.extract_reasoning_traces(),
        tool_calls=message.extract_tool_calls(),
        content_blocks=message.extract_content_blocks(),
        model=meta.model,
        tokens=meta.tokens,
        cost=meta.cost,
        duration_ms=meta.duration_ms,
        provider=provider,
        raw=raw,
    )


def _validate_claude_code_record(raw: JSONRecord) -> ViewportRecord:
    from polylogue.sources.providers.claude_code import ClaudeCodeRecord

    if raw.get("message") == {}:
        raise ValueError("Message has no role. Data should be validated at import time.")
    return cast(ViewportRecord, ClaudeCodeRecord.model_validate(raw))


def _validate_claude_ai_message(raw: JSONRecord) -> ViewportRecord:
    from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage

    return cast(ViewportRecord, ClaudeAIChatMessage.model_validate(raw))


def _validate_chatgpt_message(raw: JSONRecord) -> ViewportRecord:
    from polylogue.sources.providers.chatgpt import ChatGPTMessage

    return cast(ViewportRecord, ChatGPTMessage.model_validate(raw))


def _validate_gemini_message(raw: JSONRecord) -> ViewportRecord:
    from polylogue.sources.providers.gemini import GeminiMessage

    return cast(ViewportRecord, GeminiMessage.model_validate(raw))


def _validate_codex_record(raw: JSONRecord) -> ViewportRecord:
    from polylogue.sources.providers.codex import CodexRecord

    return cast(ViewportRecord, CodexRecord.model_validate(raw))


_ADAPTER_BUILDERS: dict[Provider, AdapterBuilder] = {
    Provider.CLAUDE_CODE: _validate_claude_code_record,
    Provider.CLAUDE_AI: _validate_claude_ai_message,
    Provider.CHATGPT: _validate_chatgpt_message,
    Provider.GEMINI: _validate_gemini_message,
    Provider.CODEX: _validate_codex_record,
}


def extract_with_adapter(provider: Provider, raw: JSONRecord) -> HarmonizedMessage:
    """Extract via the canonical typed provider adapter for valid raw records."""
    try:
        builder = _ADAPTER_BUILDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc
    return _harmonize_viewport_message(provider, raw, builder(raw))


__all__ = ["extract_with_adapter"]
