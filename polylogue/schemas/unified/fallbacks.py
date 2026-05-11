"""Fallback semantic extraction for malformed or partial provider payloads."""

from __future__ import annotations

from collections.abc import Callable

from polylogue.archive.message.roles import Role
from polylogue.archive.provider.semantics import (
    extract_chatgpt_text,
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.archive.viewport.viewports import CostInfo, ReasoningTrace, TokenUsage
from polylogue.core.json import JSONDocument, json_document, json_document_list
from polylogue.core.timestamps import parse_timestamp
from polylogue.schemas.unified.models import HarmonizedMessage, _missing_role, extract_token_usage
from polylogue.types import Provider


def _record(value: object) -> JSONDocument:
    return json_document(value)


def _content_blocks(value: object) -> list[JSONDocument] | None:
    blocks = json_document_list(value)
    return blocks or None


def _payload_record(raw: JSONDocument) -> JSONDocument:
    return json_document(dict(raw))


def _object_record(value: JSONDocument) -> dict[str, object]:
    return dict(value)


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _timestamp_candidate(value: object) -> str | int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        return value
    return None


FallbackExtractor = Callable[[JSONDocument], HarmonizedMessage]


def _fallback_extract_claude_code(raw: JSONDocument) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude Code records."""
    payload = _payload_record(raw)
    if payload.get("message") == {}:
        _missing_role()
    msg = _payload_record(_record(payload.get("message")))
    content = _content_blocks(msg.get("content"))

    return HarmonizedMessage(
        id=_string_value(payload.get("uuid")),
        role=Role.normalize(_string_value(msg.get("role")) or _string_value(payload.get("type")) or _missing_role()),
        text=extract_claude_code_text(content),
        timestamp=parse_timestamp(_timestamp_candidate(payload.get("timestamp"))),
        reasoning_traces=extract_reasoning_traces(content, Provider.CLAUDE_CODE),
        tool_calls=extract_tool_calls(content, Provider.CLAUDE_CODE),
        content_blocks=extract_content_blocks(content),
        model=_string_value(msg.get("model")),
        tokens=extract_token_usage(dict(usage) if (usage := _record(msg.get("usage"))) else None),
        cost=CostInfo(total_usd=cost_usd) if (cost_usd := _float_value(payload.get("costUSD"))) is not None else None,
        duration_ms=_int_value(payload.get("durationMs")),
        provider=Provider.CLAUDE_CODE,
        raw=_object_record(payload),
    )


def _fallback_extract_claude_ai(raw: JSONDocument) -> HarmonizedMessage:
    """Fallback extraction for malformed Claude AI records."""
    payload = _payload_record(raw)
    return HarmonizedMessage(
        id=_string_value(payload.get("uuid")),
        role=Role.normalize(_string_value(payload.get("sender")) or _missing_role()),
        text=_string_value(payload.get("text")) or "",
        timestamp=parse_timestamp(_timestamp_candidate(payload.get("created_at"))),
        provider=Provider.CLAUDE_AI,
        raw=_object_record(payload),
    )


def _fallback_extract_chatgpt(raw: JSONDocument) -> HarmonizedMessage:
    """Fallback extraction for malformed ChatGPT records."""
    payload = _payload_record(raw)
    author = _payload_record(_record(payload.get("author")))
    content = _payload_record(_record(payload.get("content")))
    metadata = _payload_record(_record(payload.get("metadata")))

    return HarmonizedMessage(
        id=_string_value(payload.get("id")),
        role=Role.normalize(_string_value(author.get("role")) or _missing_role()),
        text=extract_chatgpt_text(content),
        timestamp=parse_timestamp(_timestamp_candidate(payload.get("create_time"))),
        model=_string_value(metadata.get("model_slug")),
        provider=Provider.CHATGPT,
        raw=_object_record(payload),
    )


def _fallback_extract_gemini(raw: JSONDocument) -> HarmonizedMessage:
    """Fallback extraction for malformed Gemini records."""
    payload = _payload_record(raw)
    is_thinking = payload.get("isThought", False)
    thinking_budget = payload.get("thinkingBudget")
    token_count = payload.get("tokenCount")
    output_tokens = token_count if isinstance(token_count, int) else None

    return HarmonizedMessage(
        id=None,
        role=Role.normalize(_string_value(payload.get("role")) or _missing_role()),
        text=_string_value(payload.get("text")) or "",
        timestamp=None,
        reasoning_traces=[
            ReasoningTrace(
                text=_string_value(payload.get("text")) or "",
                token_count=thinking_budget if isinstance(thinking_budget, int) else None,
                provider=Provider.GEMINI,
                raw=_object_record(payload),
            )
        ]
        if is_thinking
        else [],
        tokens=TokenUsage(output_tokens=output_tokens) if output_tokens is not None else None,
        provider=Provider.GEMINI,
        raw=_object_record(payload),
    )


def _fallback_extract_local_agent(provider: Provider, raw: JSONDocument) -> HarmonizedMessage:
    payload = _payload_record(raw)
    content = _content_blocks(payload.get("content"))
    role = _string_value(payload.get("role")) or _string_value(payload.get("type")) or _missing_role()
    if provider is Provider.GEMINI_CLI and role == "gemini":
        role = "assistant"
    content_block_models = extract_content_blocks(content)
    return HarmonizedMessage(
        id=_string_value(payload.get("id")) or _string_value(payload.get("tool_call_id")),
        role=Role.normalize(role),
        text="\n".join(block.text for block in content_block_models if block.text),
        timestamp=parse_timestamp(_timestamp_candidate(payload.get("timestamp"))),
        reasoning_traces=extract_reasoning_traces(content, provider),
        tool_calls=extract_tool_calls(content, provider),
        content_blocks=content_block_models,
        model=_string_value(payload.get("model")),
        tokens=extract_token_usage(dict(usage) if (usage := _record(payload.get("tokens"))) else None),
        provider=provider,
        raw=_object_record(payload),
    )


def _fallback_extract_antigravity(raw: JSONDocument) -> HarmonizedMessage:
    payload = _payload_record(raw)
    text = _string_value(payload.get("summary")) or ""
    return HarmonizedMessage(
        id=None,
        role=Role.ASSISTANT,
        text=text,
        timestamp=parse_timestamp(_timestamp_candidate(payload.get("updatedAt"))),
        provider=Provider.ANTIGRAVITY,
        raw=_object_record(payload),
    )


def _fallback_extract_codex(raw: JSONDocument) -> HarmonizedMessage:
    """Fallback extraction for malformed Codex records."""
    raw_payload = _payload_record(raw)
    payload = _payload_record(_record(raw_payload.get("payload")))
    if payload:
        role = _string_value(payload.get("role")) or "unknown"
        content = payload.get("content", [])
    else:
        role = _string_value(raw_payload.get("role")) or "unknown"
        content = raw_payload.get("content", [])

    text_parts = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        text = (
            _string_value(block.get("text"))
            or _string_value(block.get("input_text"))
            or _string_value(block.get("output_text"))
        )
        if text:
            text_parts.append(text)

    return HarmonizedMessage(
        id=_string_value(raw_payload.get("id")),
        role=Role.normalize(role),
        text="\n".join(text_parts),
        timestamp=parse_timestamp(_timestamp_candidate(raw_payload.get("timestamp"))),
        provider=Provider.CODEX,
        raw=_object_record(raw_payload),
    )


_FALLBACK_EXTRACTORS: dict[Provider, FallbackExtractor] = {
    Provider.CLAUDE_CODE: _fallback_extract_claude_code,
    Provider.CLAUDE_AI: _fallback_extract_claude_ai,
    Provider.CHATGPT: _fallback_extract_chatgpt,
    Provider.GEMINI: _fallback_extract_gemini,
    Provider.GEMINI_CLI: lambda raw: _fallback_extract_local_agent(Provider.GEMINI_CLI, raw),
    Provider.HERMES: lambda raw: _fallback_extract_local_agent(Provider.HERMES, raw),
    Provider.ANTIGRAVITY: _fallback_extract_antigravity,
    Provider.CODEX: _fallback_extract_codex,
}


def extract_fallback_message(provider: Provider, raw: JSONDocument) -> HarmonizedMessage:
    """Fallback harmonization for malformed raw provider payloads."""
    try:
        extractor = _FALLBACK_EXTRACTORS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {provider}") from exc
    return extractor(raw)


__all__ = ["extract_fallback_message"]
