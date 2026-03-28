"""Fallback semantic extraction for malformed or partial provider payloads."""

from __future__ import annotations

from typing import Any

from polylogue.lib.provider_semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import CostInfo, ReasoningTrace, TokenUsage
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role, extract_token_usage
from polylogue.types import Provider


def _fallback_extract_claude_code(raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude Code records."""
    msg = raw.get("message", {})
    content = msg.get("content", []) if isinstance(msg, dict) else []

    return HarmonizedMessage(
        id=raw.get("uuid"),
        role=Role.normalize((msg.get("role") if isinstance(msg, dict) else raw.get("type")) or _missing_role()),
        text=extract_claude_code_text(content),
        timestamp=parse_timestamp(raw.get("timestamp")),
        reasoning_traces=extract_reasoning_traces(content, Provider.CLAUDE_CODE),
        tool_calls=extract_tool_calls(content, Provider.CLAUDE_CODE),
        content_blocks=extract_content_blocks(content),
        model=msg.get("model") if isinstance(msg, dict) else None,
        tokens=extract_token_usage(msg.get("usage") if isinstance(msg, dict) else None),
        cost=CostInfo(total_usd=raw.get("costUSD")) if raw.get("costUSD") else None,
        duration_ms=raw.get("durationMs"),
        provider=Provider.CLAUDE_CODE,
        raw=raw,
    )


def _fallback_extract_claude_ai(raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude AI records."""
    return HarmonizedMessage(
        id=raw.get("uuid"),
        role=Role.normalize(raw.get("sender") or _missing_role()),
        text=raw.get("text", ""),
        timestamp=parse_timestamp(raw.get("created_at")),
        provider=Provider.CLAUDE_AI,
        raw=raw,
    )


def _fallback_extract_chatgpt(raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback extraction for malformed ChatGPT records."""
    author = raw.get("author", {})
    content = raw.get("content", {})
    metadata = raw.get("metadata", {})

    return HarmonizedMessage(
        id=raw.get("id"),
        role=Role.normalize((author.get("role") if isinstance(author, dict) else None) or _missing_role()),
        text=extract_chatgpt_text(content) if isinstance(content, dict) else "",
        timestamp=parse_timestamp(raw.get("create_time")),
        model=metadata.get("model_slug") if isinstance(metadata, dict) else None,
        provider=Provider.CHATGPT,
        raw=raw,
    )


def _fallback_extract_gemini(raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback extraction for malformed Gemini records."""
    is_thinking = raw.get("isThought", False)
    thinking_budget = raw.get("thinkingBudget")
    token_count = raw.get("tokenCount")
    output_tokens = token_count if isinstance(token_count, int) else None

    return HarmonizedMessage(
        id=None,
        role=Role.normalize(raw.get("role") or _missing_role()),
        text=raw.get("text", ""),
        timestamp=None,
        reasoning_traces=[
            ReasoningTrace(
                text=raw.get("text", ""),
                token_count=thinking_budget if isinstance(thinking_budget, int) else None,
                provider=Provider.GEMINI,
                raw=raw,
            )
        ]
        if is_thinking
        else [],
        tokens=TokenUsage(output_tokens=output_tokens) if output_tokens is not None else None,
        provider=Provider.GEMINI,
        raw=raw,
    )


def _fallback_extract_codex(raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback extraction for malformed Codex records."""
    if isinstance(raw.get("payload"), dict):
        payload = raw["payload"]
        role = payload.get("role", "unknown")
        content = payload.get("content", [])
    else:
        role = raw.get("role", "unknown")
        content = raw.get("content", [])

    text_parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if text:
            text_parts.append(text)

    return HarmonizedMessage(
        id=raw.get("id"),
        role=Role.normalize(role),
        text="\n".join(text_parts),
        timestamp=parse_timestamp(raw.get("timestamp")),
        provider=Provider.CODEX,
        raw=raw,
    )


_FALLBACK_EXTRACTORS = {
    Provider.CLAUDE_CODE: _fallback_extract_claude_code,
    Provider.CLAUDE_AI: _fallback_extract_claude_ai,
    Provider.CHATGPT: _fallback_extract_chatgpt,
    Provider.GEMINI: _fallback_extract_gemini,
    Provider.CODEX: _fallback_extract_codex,
}


def extract_fallback_message(provider: Provider, raw: dict[str, Any]) -> HarmonizedMessage:
    """Fallback harmonization for malformed raw provider payloads."""
    try:
        extractor = _FALLBACK_EXTRACTORS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc
    return extractor(raw)


__all__ = ["extract_fallback_message"]
