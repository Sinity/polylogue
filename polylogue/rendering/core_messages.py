"""Shared rendered-message payload helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import overload

from polylogue.rendering.block_models import RenderableBlock
from polylogue.rendering.blocks import (
    has_structured_blocks,
    render_blocks_html,
    render_blocks_plaintext,
)

_ROLE_CLASS_RE = re.compile(r"[^a-z0-9-]")


@dataclass(slots=True)
class RenderedMessage:
    """Canonical rendered message shared by HTML and site surfaces."""

    id: str
    role: str
    text: str
    html_content: str
    timestamp: str | None
    parent_message_id: str | None = None
    # Creation order among siblings -- NOT display state. See is_active_path.
    branch_index: int = 0
    # Provider-reported "this sibling is the currently-accepted one" signal.
    # None means unknown (not selected / no branch concept), never a
    # fabricated "not active" (polylogue-9qq7).
    is_active_path: bool | None = None
    role_class: str = ""
    branches: list[RenderedMessage] = field(default_factory=list)


def normalize_render_timestamp(timestamp: object) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if isinstance(timestamp, int | float):
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (OverflowError, ValueError, OSError):
            return None
    return str(timestamp)


def normalize_render_branch_index(value: object) -> int:
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


def role_css_class(role: str) -> str:
    return "message-" + _ROLE_CLASS_RE.sub("-", role.lower())


def _is_mainline_variant(is_active_path: bool | None, branch_index: int) -> bool:
    """Whether this sibling is the one displayed as the primary/mainline variant.

    ``is_active_path`` (the provider's "currently accepted sibling" signal) is
    authoritative when known. ``branch_index`` is creation order, not display
    state, and is used only as a fallback when ``is_active_path`` is unknown
    (``None``) -- e.g. a provider with no branch concept. Unknown must never
    collapse to "not mainline" (polylogue-9qq7).
    """
    return is_active_path if is_active_path is not None else branch_index == 0


def _mapping_branch_index(message: Mapping[str, object]) -> int:
    return normalize_render_branch_index(message.get("branch_index"))


def _mapping_is_active_path(message: Mapping[str, object]) -> bool | None:
    value = message.get("is_active_path")
    return None if value is None else bool(value)


def _mapping_parent_message_id(message: Mapping[str, object]) -> str | None:
    parent_message_id = message.get("parent_message_id")
    if parent_message_id is None:
        return None
    return str(parent_message_id)


def branch_group_positions(
    metadata: Sequence[tuple[str | None, int, bool | None]],
) -> tuple[list[int], dict[int, list[int]]]:
    """Select the same mainline and alternative ordering without retaining bodies."""
    if not any(branch_index for _, branch_index, _ in metadata):
        return list(range(len(metadata))), {}
    mainline = [
        position
        for position, (parent, branch_index, active) in enumerate(metadata)
        if _is_mainline_variant(active, branch_index) or parent is None
    ]
    by_parent = {metadata[position][0]: position for position in mainline if metadata[position][0] is not None}
    branches: dict[int, list[int]] = {}
    for position, (parent, branch_index, active) in enumerate(metadata):
        if _is_mainline_variant(active, branch_index) or parent is None:
            continue
        sibling = by_parent.get(parent)
        if sibling is None:
            mainline.append(position)
        else:
            branches.setdefault(sibling, []).append(position)
    return mainline, branches


def _attach_mapping_message_branches(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    copied = [dict(message) for message in messages]
    positions, branches = branch_group_positions(
        [
            (_mapping_parent_message_id(message), _mapping_branch_index(message), _mapping_is_active_path(message))
            for message in copied
        ]
    )
    for position, alternatives in branches.items():
        target = copied[position].setdefault("branches", [])
        if isinstance(target, list):
            target.extend(copied[index] for index in alternatives)
    return [copied[position] for position in positions]


@overload
def attach_rendered_message_branches(messages: Sequence[RenderedMessage]) -> list[RenderedMessage]: ...


@overload
def attach_rendered_message_branches(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]: ...


def attach_rendered_message_branches(
    messages: Sequence[RenderedMessage] | Sequence[Mapping[str, object]],
) -> list[RenderedMessage] | list[dict[str, object]]:
    """Attach branch messages to their mainline sibling for typed or mapping payloads."""
    if not messages:
        return []
    if all(isinstance(message, RenderedMessage) for message in messages):
        typed_messages = [message for message in messages if isinstance(message, RenderedMessage)]
        positions, branches = branch_group_positions(
            [(message.parent_message_id, message.branch_index, message.is_active_path) for message in typed_messages]
        )
        for position, alternatives in branches.items():
            typed_messages[position].branches.extend(typed_messages[index] for index in alternatives)
        return [typed_messages[position] for position in positions]

    mapping_messages = [message for message in messages if isinstance(message, Mapping)]
    return _attach_mapping_message_branches(mapping_messages)


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
    is_active_path: bool | None = None,
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
    normalized_role_text = str(normalized_role)
    return RenderedMessage(
        id=str(message_id),
        role=normalized_role_text,
        text=preview_text,
        html_content=html_content,
        timestamp=normalize_render_timestamp(timestamp),
        parent_message_id=normalized_parent,
        branch_index=normalize_render_branch_index(branch_index),
        is_active_path=is_active_path,
        role_class=role_css_class(normalized_role_text),
    )


__all__ = [
    "attach_rendered_message_branches",
    "build_rendered_message",
    "normalize_render_branch_index",
    "normalize_render_timestamp",
    "role_css_class",
    "RenderedMessage",
]
