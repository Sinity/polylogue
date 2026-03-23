"""Provider-meta hydration and parsed-message integration for harmonized messages."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ValidationError

from polylogue.lib.provider_semantics import extract_display_text_from_content_blocks
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import ContentBlock, ContentType, CostInfo, ReasoningTrace, TokenUsage, ToolCall
from polylogue.schemas.unified_adapters import extract_with_adapter
from polylogue.schemas.unified_fallbacks import extract_fallback_message
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role, extract_token_usage
from polylogue.types import Provider


def _coerce_timestamp(value: datetime | str | float | int | None) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return parse_timestamp(value)


def _coerce_reasoning_traces(
    traces: object,
    provider: Provider,
) -> list[ReasoningTrace]:
    if not isinstance(traces, list):
        return []
    coerced: list[ReasoningTrace] = []
    for trace in traces:
        if isinstance(trace, ReasoningTrace):
            coerced.append(trace)
            continue
        if not isinstance(trace, dict):
            continue
        raw_trace = dict(trace)
        raw_trace.setdefault("provider", provider)
        try:
            coerced.append(ReasoningTrace.model_validate(raw_trace))
        except ValidationError:
            continue
    return coerced


def _coerce_tool_calls(
    calls: object,
    provider: Provider,
) -> list[ToolCall]:
    if not isinstance(calls, list):
        return []
    coerced: list[ToolCall] = []
    for call in calls:
        if isinstance(call, ToolCall):
            coerced.append(call)
            continue
        if not isinstance(call, dict):
            continue
        raw_call = dict(call)
        raw_call.setdefault("provider", provider)
        try:
            coerced.append(ToolCall.model_validate(raw_call))
        except ValidationError:
            continue
    return coerced


def _coerce_content_blocks(blocks: object) -> list[ContentBlock]:
    if not isinstance(blocks, list):
        return []
    coerced: list[ContentBlock] = []
    for block in blocks:
        if isinstance(block, ContentBlock):
            coerced.append(block)
            continue
        if not isinstance(block, dict):
            continue
        try:
            coerced.append(ContentBlock.model_validate(block))
        except ValidationError:
            continue
    return coerced


def _extract_generic_tokens(provider_meta: dict[str, Any]) -> TokenUsage | None:
    usage = provider_meta.get("usage")
    if isinstance(usage, dict):
        return extract_token_usage(usage)

    tokens = provider_meta.get("tokens")
    if isinstance(tokens, TokenUsage):
        return tokens
    if isinstance(tokens, dict):
        try:
            return TokenUsage.model_validate(tokens)
        except ValidationError:
            return None

    token_count = provider_meta.get("tokenCount")
    if isinstance(token_count, int):
        return TokenUsage(output_tokens=token_count)
    return None


def _extract_generic_cost(provider_meta: dict[str, Any]) -> CostInfo | None:
    cost = provider_meta.get("cost")
    if isinstance(cost, CostInfo):
        return cost
    if isinstance(cost, dict):
        try:
            return CostInfo.model_validate(cost)
        except ValidationError:
            return None

    cost_usd = provider_meta.get("costUSD")
    if isinstance(cost_usd, (int, float)):
        return CostInfo(total_usd=float(cost_usd))
    return None


def _harmonize_extracted_provider_meta(
    provider: Provider,
    provider_meta: dict[str, Any],
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage:
    content_blocks = _coerce_content_blocks(provider_meta.get("content_blocks"))
    reasoning_traces = _coerce_reasoning_traces(provider_meta.get("reasoning_traces"), provider)
    tool_calls = _coerce_tool_calls(provider_meta.get("tool_calls"), provider)

    if not reasoning_traces and content_blocks:
        reasoning_traces = [
            ReasoningTrace(text=block.text, provider=provider, raw=block.raw)
            for block in content_blocks
            if block.type == ContentType.THINKING and block.text
        ]
    if not tool_calls:
        tool_calls = [
            block.tool_call
            for block in content_blocks
            if block.type == ContentType.TOOL_USE and block.tool_call is not None
        ]

    resolved_role = role
    if not resolved_role:
        for candidate in (
            provider_meta.get("role"),
            provider_meta.get("sender"),
            provider_meta.get("type"),
        ):
            if candidate:
                resolved_role = str(candidate)
                break

    resolved_text = text
    if not isinstance(resolved_text, str):
        if isinstance(provider_meta.get("text"), str):
            resolved_text = str(provider_meta["text"])
        else:
            raw_content_blocks = provider_meta.get("content_blocks")
            resolved_text = extract_display_text_from_content_blocks(
                raw_content_blocks if isinstance(raw_content_blocks, list) else None
            )

    resolved_timestamp = timestamp
    if resolved_timestamp is None:
        for candidate in (
            provider_meta.get("timestamp"),
            provider_meta.get("created_at"),
            provider_meta.get("create_time"),
            provider_meta.get("updated_at"),
        ):
            if candidate is not None:
                resolved_timestamp = candidate
                break

    return HarmonizedMessage(
        id=message_id or provider_meta.get("id") or provider_meta.get("uuid"),
        role=Role.normalize(resolved_role or _missing_role()),
        text=resolved_text or "",
        timestamp=_coerce_timestamp(resolved_timestamp),
        reasoning_traces=reasoning_traces,
        tool_calls=tool_calls,
        content_blocks=content_blocks,
        model=provider_meta.get("model") or provider_meta.get("model_slug"),
        tokens=_extract_generic_tokens(provider_meta),
        cost=_extract_generic_cost(provider_meta),
        duration_ms=provider_meta.get("durationMs") or provider_meta.get("duration_ms"),
        provider=provider,
        raw=provider_meta,
    )


def _overlay_message_context(
    message: HarmonizedMessage,
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage:
    updates: dict[str, Any] = {}

    if message.id is None and message_id is not None:
        updates["id"] = message_id

    if message.role == Role.UNKNOWN and role:
        updates["role"] = Role.normalize(role)

    if not message.text and isinstance(text, str):
        updates["text"] = text

    if message.timestamp is None and timestamp is not None:
        updates["timestamp"] = _coerce_timestamp(timestamp)

    if not updates:
        return message
    return message.model_copy(update=updates)


def _has_extracted_viewports(provider_meta: dict[str, Any]) -> bool:
    return any(
        key in provider_meta
        for key in (
            "content_blocks",
            "reasoning_traces",
            "tool_calls",
            "tokens",
            "cost",
            "costUSD",
            "durationMs",
            "duration_ms",
        )
    )


def extract_from_provider_meta(
    provider: Provider | str,
    provider_meta: dict[str, Any],
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage:
    """Extract HarmonizedMessage from polylogue database format."""
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    raw = provider_meta.get("raw")
    if raw is not None:
        try:
            harmonized = extract_with_adapter(p, raw)
        except (ValidationError, ValueError):
            harmonized = extract_fallback_message(p, raw)
        return _overlay_message_context(
            harmonized,
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    if _has_extracted_viewports(provider_meta):
        return _harmonize_extracted_provider_meta(
            p,
            provider_meta,
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    try:
        harmonized = extract_with_adapter(p, provider_meta)
    except (ValidationError, ValueError, TypeError):
        return _harmonize_extracted_provider_meta(
            p,
            provider_meta,
            message_id=message_id,
            role=role,
            text=text,
            timestamp=timestamp,
        )

    return _overlay_message_context(
        harmonized,
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
    )


def is_message_record(provider: Provider | str, raw: dict[str, Any]) -> bool:
    """Check if a record is an actual message (vs metadata)."""
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    if p == Provider.CLAUDE_CODE:
        record_type = raw.get("type")
        if record_type is None:
            return True
        return record_type in ("user", "assistant", "system")
    return True


def harmonize_parsed_message(
    provider: str,
    provider_meta: dict[str, Any] | None,
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage | None:
    """Convert ParsedMessage.provider_meta to HarmonizedMessage."""
    if not provider_meta:
        return None

    raw = provider_meta.get("raw", provider_meta)
    if not is_message_record(provider, raw):
        return None

    return extract_from_provider_meta(
        provider,
        provider_meta,
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
    )


def bulk_harmonize(
    provider: str,
    parsed_messages: list[Any],
) -> list[HarmonizedMessage]:
    """Bulk convert ParsedMessages to HarmonizedMessages."""
    results = []
    for parsed_message in parsed_messages:
        meta = getattr(parsed_message, "provider_meta", None)
        if meta:
            harmonized = harmonize_parsed_message(
                provider,
                meta,
                message_id=getattr(parsed_message, "provider_message_id", None),
                role=getattr(parsed_message, "role", None),
                text=getattr(parsed_message, "text", None),
                timestamp=getattr(parsed_message, "timestamp", None),
            )
            if harmonized:
                results.append(harmonized)
    return results


__all__ = [
    "_coerce_content_blocks",
    "_coerce_reasoning_traces",
    "_coerce_tool_calls",
    "_extract_generic_cost",
    "_extract_generic_tokens",
    "_harmonize_extracted_provider_meta",
    "_has_extracted_viewports",
    "_overlay_message_context",
    "bulk_harmonize",
    "extract_from_provider_meta",
    "harmonize_parsed_message",
    "is_message_record",
]
