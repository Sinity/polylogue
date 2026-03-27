"""Canonical-surface semantic proof contracts."""

from __future__ import annotations

from polylogue.rendering.semantic_surface_contract_helpers import (
    declared_loss_contract as _declared_loss,
)
from polylogue.rendering.semantic_surface_contract_helpers import (
    preserve_contract as _preserve,
)

CANONICAL_MARKDOWN_CONTRACTS = (
    _preserve(
        "renderable_messages",
        "canonical markdown must preserve every renderable message section",
        "renderable_messages",
        "message_sections",
    ),
    _preserve(
        "attachment_lines",
        "canonical markdown must preserve attachment presence as attachment lines",
        "attachment_count",
        "attachment_lines",
    ),
    _preserve(
        "timestamp_lines",
        "canonical markdown must preserve timestamps for renderable messages that have them",
        "timestamped_renderable_messages",
        "timestamp_lines",
    ),
    _preserve(
        "role_sections",
        "canonical markdown must preserve renderable message role sections",
        "renderable_role_counts",
        "role_section_counts",
    ),
    _declared_loss(
        "empty_messages",
        "canonical markdown intentionally omits messages with no text and no attachments",
        "empty_messages",
    ),
    _declared_loss(
        "thinking_semantics",
        "canonical markdown preserves display text but not typed thinking markers",
        "thinking_messages",
        "typed_thinking_markers",
    ),
    _declared_loss(
        "tool_semantics",
        "canonical markdown preserves display text but not typed tool markers",
        "tool_messages",
        "typed_tool_markers",
    ),
)


__all__ = ["CANONICAL_MARKDOWN_CONTRACTS"]
