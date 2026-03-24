"""Support helpers for semantic fact extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.roles import Role
from polylogue.lib.viewports import ReasoningTrace, ToolCall

if TYPE_CHECKING:
    from polylogue.lib.models import Message
    from polylogue.lib.viewports import TokenUsage
    from polylogue.storage.store import MessageRecord


def normalized_role_label(value: object) -> str:
    if isinstance(value, Role):
        return str(value)
    if value:
        return str(Role.normalize(str(value)))
    return "message"


def sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def message_has_text(message: Message | MessageRecord) -> bool:
    return bool((message.text or "").strip())


def message_tool_calls(message: Message) -> tuple[ToolCall, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        calls = getattr(harmonized, "tool_calls", None)
        if calls:
            return tuple(calls)
    return _message_content_block_tool_calls(message)


def message_reasoning_traces(message: Message) -> tuple[ReasoningTrace, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        traces = getattr(harmonized, "reasoning_traces", None)
        if traces:
            return tuple(traces)
    return _message_content_block_reasoning_traces(message)


def message_tokens(message: Message) -> TokenUsage | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    tokens = getattr(harmonized, "tokens", None)
    return tokens if tokens is not None else None


def message_model_name(message: Message) -> str | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    model = getattr(harmonized, "model", None)
    return str(model) if model else None


def _message_content_block_tool_calls(message: Message) -> tuple[ToolCall, ...]:
    from polylogue.lib.action_events import build_tool_calls_from_content_blocks

    return build_tool_calls_from_content_blocks(
        provider=message.provider,
        content_blocks=message.content_blocks,
    )


def _message_content_block_reasoning_traces(message: Message) -> tuple[ReasoningTrace, ...]:
    traces: list[ReasoningTrace] = []
    for block in message.content_blocks:
        if str(block.get("type")) != "thinking":
            continue
        text = block.get("text")
        if not isinstance(text, str) or not text:
            continue
        traces.append(
            ReasoningTrace(
                text=text,
                provider=message.provider,
                raw={
                    "type": block.get("type"),
                    "media_type": block.get("media_type"),
                    "metadata": block.get("metadata"),
                    "semantic_type": block.get("semantic_type"),
                },
            )
        )
    return tuple(traces)


__all__ = [
    "message_has_text",
    "message_model_name",
    "message_reasoning_traces",
    "message_tokens",
    "message_tool_calls",
    "normalized_role_label",
    "sorted_counts",
]
