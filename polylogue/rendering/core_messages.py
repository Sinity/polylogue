"""Shared rendered-message payload helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polylogue.rendering.block_models import RenderableBlock
from polylogue.rendering.blocks import (
    has_structured_blocks,
    render_blocks_html,
    render_blocks_plaintext,
)


@dataclass(slots=True)
class RenderedMessage:
    """Canonical rendered message shared by HTML and site surfaces."""

    id: str
    role: str
    text: str
    html_content: str
    timestamp: str | None
    parent_message_id: str | None = None
    branch_index: int = 0
    role_class: str = ""
    branches: list[RenderedMessage] = field(default_factory=list)


def _normalize_timestamp(timestamp: object) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if isinstance(timestamp, int | float):
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (ValueError, OSError):
            return None
    return str(timestamp)


def _normalize_branch_index(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError:
        return 0


def build_rendered_message(
    *,
    message_id: object,
    role: object,
    text: str,
    timestamp: object,
    render_html: Callable[[str], str],
    content_blocks: tuple[RenderableBlock, ...] = (),
    parent_message_id: object = None,
    branch_index: object = 0,
    preview_limit: int | None = None,
) -> RenderedMessage:
    """Build a canonical rendered message shared by HTML/site surfaces."""
    normalized_role = role or "message"
    if hasattr(normalized_role, "value"):
        normalized_role = normalized_role.value
    display_text = render_blocks_plaintext(content_blocks) if content_blocks else text
    if not display_text:
        display_text = text
    html_content = render_blocks_html(content_blocks) if has_structured_blocks(content_blocks) else render_html(text)
    preview_text = display_text[:preview_limit] if preview_limit is not None else display_text
    normalized_parent = str(parent_message_id) if parent_message_id is not None else None
    return RenderedMessage(
        id=str(message_id),
        role=str(normalized_role),
        text=preview_text,
        html_content=html_content,
        timestamp=_normalize_timestamp(timestamp),
        parent_message_id=normalized_parent,
        branch_index=_normalize_branch_index(branch_index),
    )


def build_rendered_message_payload(
    *,
    message_id: object,
    role: object,
    text: str,
    timestamp: object,
    render_html: Callable[[str], str],
    content_blocks: tuple[RenderableBlock, ...] = (),
    parent_message_id: object = None,
    branch_index: object = 0,
    preview_limit: int | None = None,
) -> RenderedMessage:
    """Backward-compatible alias for the canonical rendered message builder."""
    return build_rendered_message(
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
        render_html=render_html,
        content_blocks=content_blocks,
        parent_message_id=parent_message_id,
        branch_index=branch_index,
        preview_limit=preview_limit,
    )


__all__ = [
    "build_rendered_message",
    "build_rendered_message_payload",
    "RenderedMessage",
]
