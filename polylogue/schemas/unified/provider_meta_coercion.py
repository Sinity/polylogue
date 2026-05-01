"""Coercion helpers for provider-meta harmonization."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError

from polylogue.archive.viewport.viewports import ContentBlock, CostInfo, ReasoningTrace, TokenUsage, ToolCall
from polylogue.lib.json import JSONDocument, json_document
from polylogue.lib.timestamps import parse_timestamp
from polylogue.schemas.unified.models import extract_token_usage
from polylogue.types import Provider

ProviderMetaPayload = JSONDocument


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
        raw_trace = json_document(trace)
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
        raw_call = json_document(call)
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
            coerced.append(ContentBlock.model_validate(json_document(block)))
        except ValidationError:
            continue
    return coerced


def _extract_generic_tokens(provider_meta: ProviderMetaPayload) -> TokenUsage | None:
    usage = provider_meta.get("usage")
    if isinstance(usage, dict):
        return extract_token_usage(json_document(usage))

    tokens = provider_meta.get("tokens")
    if isinstance(tokens, TokenUsage):
        return tokens
    if isinstance(tokens, dict):
        try:
            return TokenUsage.model_validate(json_document(tokens))
        except ValidationError:
            return None

    token_count = provider_meta.get("tokenCount")
    if isinstance(token_count, int):
        return TokenUsage(output_tokens=token_count)
    return None


def _extract_generic_cost(provider_meta: ProviderMetaPayload) -> CostInfo | None:
    cost = provider_meta.get("cost")
    if isinstance(cost, CostInfo):
        return cost
    if isinstance(cost, dict):
        try:
            return CostInfo.model_validate(json_document(cost))
        except ValidationError:
            return None

    cost_usd = provider_meta.get("costUSD")
    if isinstance(cost_usd, (int, float)):
        return CostInfo(total_usd=float(cost_usd))
    return None


def _has_extracted_viewports(provider_meta: ProviderMetaPayload) -> bool:
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


__all__ = [
    "_coerce_content_blocks",
    "_coerce_reasoning_traces",
    "_coerce_timestamp",
    "_coerce_tool_calls",
    "_extract_generic_cost",
    "_extract_generic_tokens",
    "_has_extracted_viewports",
    "ProviderMetaPayload",
]
