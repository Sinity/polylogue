"""Message-tree shaping helpers for HTML rendering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from polylogue.rendering.block_models import coerce_renderable_blocks
from polylogue.rendering.core import build_rendered_message
from polylogue.rendering.core_messages import RenderedMessage, attach_rendered_message_branches

if TYPE_CHECKING:
    from polylogue.archive.models import Message, Session


def build_session_html_messages(
    session: Session,
    *,
    render_html: Callable[[str], str],
    preview_limit: int = 120,
) -> list[RenderedMessage]:
    return attach_rendered_message_branches(
        list(iter_html_messages(session.messages, render_html=render_html, preview_limit=preview_limit))
    )


def iter_html_messages(
    messages: Iterable[Message], *, render_html: Callable[[str], str], preview_limit: int = 120
) -> Iterator[RenderedMessage]:
    for msg in messages:
        content_blocks = coerce_renderable_blocks(getattr(msg, "blocks", None))
        text = msg.text or ""
        if not text and not content_blocks:
            continue
        yield build_rendered_message(
            message_id=msg.id,
            role=msg.role,
            text=text,
            timestamp=str(msg.timestamp) if msg.timestamp else None,
            content_blocks=content_blocks,
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            is_active_path=msg.is_active_path,
            render_html=render_html,
            preview_limit=preview_limit,
        )


_attach_branches = attach_rendered_message_branches


__all__ = [
    "_attach_branches",
    "build_session_html_messages",
]
