"""Support helpers for semantic fact extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

from polylogue.archive.message.roles import Role
from polylogue.archive.viewport.viewports import ReasoningTrace, TokenUsage, ToolCall
from polylogue.core.enums import Origin

ContentBlockSequence: TypeAlias = Sequence[Mapping[str, object]]


class TextMessageLike(Protocol):
    @property
    def text(self) -> str | None: ...


class SemanticMessageLike(TextMessageLike, Protocol):
    @property
    def origin(self) -> Origin | str | None: ...

    @property
    def blocks(self) -> ContentBlockSequence: ...


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
    """Derive tool calls from hydrated message ``content_blocks`` (#1256).

    Canonical tool-call evidence lives in typed content-block rows.
    Harmonized parser-level extraction now flows through the typed cost
    projection (#803), not through this helper.
    """

    return _message_content_block_tool_calls(message)


def message_reasoning_traces(message: SemanticMessageLike) -> tuple[ReasoningTrace, ...]:
    """Derive reasoning traces from hydrated ``content_blocks`` (#1256)."""

    return _message_content_block_reasoning_traces(message)


def message_tokens(message: SemanticMessageLike) -> TokenUsage | None:
    """Hydrated messages have no provider-meta token usage (#1256).

    Token usage for hydrated reads is sourced through the typed cost
    projection in ``polylogue.archive.semantic.pricing`` and the
    per-message ``input_tokens``/``output_tokens`` columns on the
    ``messages`` row, not through this helper.
    """

    return None


def message_model_name(message: SemanticMessageLike) -> str | None:
    """Hydrated messages have no provider-meta model name (#1256).

    Per-message model identity is sourced through the typed cost
    projection in ``polylogue.archive.semantic.pricing`` and the
    persisted ``model_name`` column, not through this helper.
    """

    return None


def _message_content_block_tool_calls(message: SemanticMessageLike) -> tuple[ToolCall, ...]:
    from polylogue.archive.actions.actions import build_tool_calls_from_content_blocks

    return build_tool_calls_from_content_blocks(
        origin=message.origin,
        content_blocks=message.blocks,
    )


def _message_content_block_reasoning_traces(message: SemanticMessageLike) -> tuple[ReasoningTrace, ...]:
    traces: list[ReasoningTrace] = []
    origin = (
        message.origin
        if isinstance(message.origin, Origin)
        else Origin.from_string(message.origin)
        if isinstance(message.origin, str)
        else None
    )
    for block in message.blocks:
        if str(block.get("type")) != "thinking":
            continue
        text = block.get("text")
        if not isinstance(text, str) or not text:
            continue
        traces.append(
            ReasoningTrace(
                text=text,
                origin=origin,
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
