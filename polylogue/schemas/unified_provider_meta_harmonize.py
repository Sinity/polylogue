"""Harmonization helpers for extracted provider-meta payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from polylogue.lib.provider_semantics import extract_display_text_from_content_blocks
from polylogue.lib.roles import Role
from polylogue.lib.viewports import ContentType, ReasoningTrace
from polylogue.schemas.unified_models import HarmonizedMessage, _missing_role
from polylogue.schemas.unified_provider_meta_coercion import (
    _coerce_content_blocks,
    _coerce_reasoning_traces,
    _coerce_timestamp,
    _coerce_tool_calls,
    _extract_generic_cost,
    _extract_generic_tokens,
)
from polylogue.types import Provider


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


__all__ = [
    "_harmonize_extracted_provider_meta",
    "_overlay_message_context",
]
