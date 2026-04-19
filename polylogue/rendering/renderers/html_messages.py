"""Message-tree shaping helpers for HTML rendering."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from polylogue.rendering.block_models import coerce_renderable_blocks
from polylogue.rendering.core import build_rendered_message
from polylogue.rendering.core_messages import RenderedMessage

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.state_views import ConversationRenderProjection

_ROLE_CLASS_RE = re.compile(r"[^a-z0-9-]")
TMessage = TypeVar("TMessage", RenderedMessage, dict[str, object])


def _role_css_class(role: str) -> str:
    return "message-" + _ROLE_CLASS_RE.sub("-", role.lower())


def _branch_index(message: RenderedMessage | dict[str, object]) -> int:
    if isinstance(message, RenderedMessage):
        return message.branch_index
    value = message.get("branch_index")
    return value if isinstance(value, int) else 0


def _parent_message_id(message: RenderedMessage | dict[str, object]) -> str | None:
    if isinstance(message, RenderedMessage):
        return message.parent_message_id
    value = message.get("parent_message_id")
    return value if isinstance(value, str) else None


def _append_branch(message: RenderedMessage | dict[str, object], branch: RenderedMessage | dict[str, object]) -> None:
    if isinstance(message, RenderedMessage):
        if isinstance(branch, RenderedMessage):
            message.branches.append(branch)
        return
    branches = message.setdefault("branches", [])
    if isinstance(branches, list):
        branches.append(branch)


def _attach_branches(messages: list[TMessage]) -> list[TMessage]:
    if not any(_branch_index(message) for message in messages):
        return messages

    mainline = [message for message in messages if not _branch_index(message)]

    mainline_by_parent: dict[str, TMessage] = {}
    for msg in mainline:
        pid = _parent_message_id(msg)
        if pid:
            mainline_by_parent[pid] = msg

    for msg in messages:
        branch_idx = _branch_index(msg)
        parent_id = _parent_message_id(msg)
        if not branch_idx or not parent_id:
            continue

        sibling = mainline_by_parent.get(parent_id)
        if sibling is not None:
            _append_branch(sibling, msg)
        else:
            mainline.append(msg)

    return mainline


def build_projection_html_messages(
    projection: ConversationRenderProjection,
    *,
    render_html: Callable[[str], str],
    preview_limit: int = 120,
) -> list[RenderedMessage]:
    raw_messages: list[RenderedMessage] = []
    for msg in projection.messages:
        text = msg.text or ""
        if not text:
            continue
        payload = build_rendered_message(
            message_id=msg.message_id,
            role=msg.role,
            text=text,
            timestamp=msg.sort_key,
            content_blocks=coerce_renderable_blocks(msg.content_blocks),
            parent_message_id=msg.parent_message_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        payload.role_class = _role_css_class(payload.role)
        raw_messages.append(payload)
    return _attach_branches(raw_messages)


def build_conversation_html_messages(
    conversation: Conversation,
    *,
    render_html: Callable[[str], str],
    preview_limit: int = 120,
) -> list[RenderedMessage]:
    raw_messages: list[RenderedMessage] = []
    for msg in conversation.messages:
        if not msg.text:
            continue
        payload = build_rendered_message(
            message_id=msg.id,
            role=msg.role,
            text=msg.text,
            timestamp=str(msg.timestamp) if msg.timestamp else None,
            content_blocks=coerce_renderable_blocks(getattr(msg, "content_blocks", None)),
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        payload.role_class = _role_css_class(payload.role)
        raw_messages.append(payload)
    return _attach_branches(raw_messages)


__all__ = [
    "_attach_branches",
    "build_conversation_html_messages",
    "build_projection_html_messages",
]
