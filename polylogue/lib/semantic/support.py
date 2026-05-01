"""Support helpers for semantic fact extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

from polylogue.lib.roles import Role
from polylogue.lib.viewport.viewports import ReasoningTrace, TokenUsage, ToolCall
from polylogue.types import Provider

ContentBlockSequence: TypeAlias = Sequence[Mapping[str, object]]


class TextMessageLike(Protocol):
    @property
    def text(self) -> str | None: ...


class HarmonizedMessageLike(Protocol):
    @property
    def tool_calls(self) -> Sequence[ToolCall] | None: ...

    @property
    def reasoning_traces(self) -> Sequence[ReasoningTrace] | None: ...

    @property
    def tokens(self) -> TokenUsage | None: ...

    @property
    def model(self) -> object | None: ...


class SemanticMessageLike(TextMessageLike, Protocol):
    @property
    def provider(self) -> Provider | str | None: ...

    @property
    def harmonized(self) -> HarmonizedMessageLike | None: ...

    @property
    def content_blocks(self) -> ContentBlockSequence: ...


def normalized_role_label(value: object) -> str:
    if isinstance(value, Role):
        return str(value)
    if value:
        return str(Role.normalize(str(value)))
    return "message"


def sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def message_has_text(message: TextMessageLike) -> bool:
    return bool((message.text or "").strip())


def message_tool_calls(message: SemanticMessageLike) -> tuple[ToolCall, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        calls = harmonized.tool_calls
        if calls:
            return tuple(calls)
    return _message_content_block_tool_calls(message)


def message_reasoning_traces(message: SemanticMessageLike) -> tuple[ReasoningTrace, ...]:
    harmonized = message.harmonized
    if harmonized is not None:
        traces = harmonized.reasoning_traces
        if traces:
            return tuple(traces)
    return _message_content_block_reasoning_traces(message)


def message_tokens(message: SemanticMessageLike) -> TokenUsage | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    return harmonized.tokens


def message_model_name(message: SemanticMessageLike) -> str | None:
    harmonized = message.harmonized
    if harmonized is None:
        return None
    model = harmonized.model
    return str(model) if model else None


def _message_content_block_tool_calls(message: SemanticMessageLike) -> tuple[ToolCall, ...]:
    from polylogue.archive.action_event.action_events import build_tool_calls_from_content_blocks

    return build_tool_calls_from_content_blocks(
        provider=message.provider,
        content_blocks=message.content_blocks,
    )


def _message_content_block_reasoning_traces(message: SemanticMessageLike) -> tuple[ReasoningTrace, ...]:
    traces: list[ReasoningTrace] = []
    provider = (
        message.provider
        if isinstance(message.provider, Provider)
        else Provider.from_string(message.provider)
        if isinstance(message.provider, str)
        else None
    )
    for block in message.content_blocks:
        if str(block.get("type")) != "thinking":
            continue
        text = block.get("text")
        if not isinstance(text, str) or not text:
            continue
        traces.append(
            ReasoningTrace(
                text=text,
                provider=provider,
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
    "ContentBlockSequence",
    "HarmonizedMessageLike",
    "message_has_text",
    "message_model_name",
    "message_reasoning_traces",
    "message_tokens",
    "message_tool_calls",
    "normalized_role_label",
    "SemanticMessageLike",
    "sorted_counts",
    "TextMessageLike",
]
