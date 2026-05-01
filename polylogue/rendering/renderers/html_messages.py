"""Message-tree shaping helpers for HTML rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polylogue.rendering.block_models import coerce_renderable_blocks
from polylogue.rendering.core import build_rendered_message
from polylogue.rendering.core_messages import RenderedMessage, attach_rendered_message_branches

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation
    from polylogue.storage.archive_views import ConversationRenderProjection


def build_projection_html_messages(
    projection: ConversationRenderProjection,
    *,
    render_html: Callable[[str], str],
    preview_limit: int = 120,
) -> list[RenderedMessage]:
    raw_messages: list[RenderedMessage] = []
    for msg in projection.messages:
        content_blocks = coerce_renderable_blocks(msg.content_blocks)
        text = msg.text or ""
        if not text and not content_blocks:
            continue
        payload = build_rendered_message(
            message_id=msg.message_id,
            role=msg.role,
            text=text,
            timestamp=msg.sort_key,
            content_blocks=content_blocks,
            parent_message_id=msg.parent_message_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        raw_messages.append(payload)
    return attach_rendered_message_branches(raw_messages)


def build_conversation_html_messages(
    conversation: Conversation,
    *,
    render_html: Callable[[str], str],
    preview_limit: int = 120,
) -> list[RenderedMessage]:
    raw_messages: list[RenderedMessage] = []
    for msg in conversation.messages:
        content_blocks = coerce_renderable_blocks(getattr(msg, "content_blocks", None))
        text = msg.text or ""
        if not text and not content_blocks:
            continue
        payload = build_rendered_message(
            message_id=msg.id,
            role=msg.role,
            text=text,
            timestamp=str(msg.timestamp) if msg.timestamp else None,
            content_blocks=content_blocks,
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        raw_messages.append(payload)
    return attach_rendered_message_branches(raw_messages)


_attach_branches = attach_rendered_message_branches


__all__ = [
    "_attach_branches",
    "build_conversation_html_messages",
    "build_projection_html_messages",
]
