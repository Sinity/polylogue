"""Fallback semantic extraction for malformed or partial provider payloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from polylogue.lib.provider_semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.lib.raw_payload_decode import JSONRecord
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import CostInfo, ReasoningTrace, TokenUsage
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role, extract_token_usage
from polylogue.types import Provider


def _record(value: object) -> JSONRecord:
    return cast(JSONRecord, value) if isinstance(value, dict) else {}


def _content_blocks(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    return [dict(item) for item in value if isinstance(item, dict)]


def _payload_record(raw: JSONRecord) -> dict[str, Any]:
    return dict(raw)


FallbackExtractor = Callable[[JSONRecord], HarmonizedMessage]


def _fallback_extract_claude_code(raw: JSONRecord) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude Code records."""
    payload = _payload_record(raw)
    if payload.get("message") == {}:
        _missing_role()
    msg = _payload_record(_record(payload.get("message")))
    content = _content_blocks(msg.get("content"))

    return HarmonizedMessage(
        id=payload.get("uuid"),
        role=Role.normalize((msg.get("role") or payload.get("type")) or _missing_role()),
        text=extract_claude_code_text(content),
        timestamp=parse_timestamp(payload.get("timestamp")),
        reasoning_traces=extract_reasoning_traces(content, Provider.CLAUDE_CODE),
        tool_calls=extract_tool_calls(content, Provider.CLAUDE_CODE),
        content_blocks=extract_content_blocks(content),
        model=msg.get("model"),
        tokens=extract_token_usage(dict(usage) if (usage := _record(msg.get("usage"))) else None),
        cost=CostInfo(total_usd=payload.get("costUSD")) if payload.get("costUSD") else None,
        duration_ms=payload.get("durationMs"),
        provider=Provider.CLAUDE_CODE,
        raw=payload,
    )


def _fallback_extract_claude_ai(raw: JSONRecord) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude AI records."""
    payload = _payload_record(raw)
    return HarmonizedMessage(
        id=payload.get("uuid"),
        role=Role.normalize(payload.get("sender") or _missing_role()),
        text=payload.get("text", ""),
        timestamp=parse_timestamp(payload.get("created_at")),
        provider=Provider.CLAUDE_AI,
        raw=payload,
    )


def _fallback_extract_chatgpt(raw: JSONRecord) -> HarmonizedMessage:
    """Fallback extraction for malformed ChatGPT records."""
    payload = _payload_record(raw)
    author = _payload_record(_record(payload.get("author")))
    content = _payload_record(_record(payload.get("content")))
    metadata = _payload_record(_record(payload.get("metadata")))

    return HarmonizedMessage(
        id=payload.get("id"),
        role=Role.normalize(author.get("role") or _missing_role()),
        text=extract_chatgpt_text(content),
        timestamp=parse_timestamp(payload.get("create_time")),
        model=metadata.get("model_slug"),
        provider=Provider.CHATGPT,
        raw=payload,
    )


def _fallback_extract_gemini(raw: JSONRecord) -> HarmonizedMessage:
    """Fallback extraction for malformed Gemini records."""
    payload = _payload_record(raw)
    is_thinking = payload.get("isThought", False)
    thinking_budget = payload.get("thinkingBudget")
    token_count = payload.get("tokenCount")
    output_tokens = token_count if isinstance(token_count, int) else None

    return HarmonizedMessage(
        id=None,
        role=Role.normalize(payload.get("role") or _missing_role()),
        text=payload.get("text", ""),
        timestamp=None,
        reasoning_traces=[
            ReasoningTrace(
                text=payload.get("text", ""),
                token_count=thinking_budget if isinstance(thinking_budget, int) else None,
                provider=Provider.GEMINI,
                raw=payload,
            )
        ]
        if is_thinking
        else [],
        tokens=TokenUsage(output_tokens=output_tokens) if output_tokens is not None else None,
        provider=Provider.GEMINI,
        raw=payload,
    )


def _fallback_extract_codex(raw: JSONRecord) -> HarmonizedMessage:
    """Fallback extraction for malformed Codex records."""
    raw_payload = _payload_record(raw)
    payload = _payload_record(_record(raw_payload.get("payload")))
    if payload:
        role = payload.get("role", "unknown")
        content = payload.get("content", [])
    else:
        role = raw_payload.get("role", "unknown")
        content = raw_payload.get("content", [])

    text_parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if text:
            text_parts.append(text)

    return HarmonizedMessage(
        id=raw_payload.get("id"),
        role=Role.normalize(role),
        text="\n".join(text_parts),
        timestamp=parse_timestamp(raw_payload.get("timestamp")),
        provider=Provider.CODEX,
        raw=raw_payload,
    )


_FALLBACK_EXTRACTORS: dict[Provider, FallbackExtractor] = {
    Provider.CLAUDE_CODE: _fallback_extract_claude_code,
    Provider.CLAUDE_AI: _fallback_extract_claude_ai,
    Provider.CHATGPT: _fallback_extract_chatgpt,
    Provider.GEMINI: _fallback_extract_gemini,
    Provider.CODEX: _fallback_extract_codex,
}


def extract_fallback_message(provider: Provider, raw: JSONRecord) -> HarmonizedMessage:
    """Fallback harmonization for malformed raw provider payloads."""
    try:
        extractor = _FALLBACK_EXTRACTORS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc
    return extractor(raw)


__all__ = ["extract_fallback_message"]
