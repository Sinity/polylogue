"""Message-tree shaping helpers for HTML rendering."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from polylogue.rendering.core import build_rendered_message_payload

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.state_views import ConversationRenderProjection

_ROLE_CLASS_RE = re.compile(r"[^a-z0-9-]")


def _role_css_class(role: str) -> str:
    return "message-" + _ROLE_CLASS_RE.sub("-", role.lower())


def _attach_branches(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    if not any(m.get("branch_index", 0) for m in messages):
        return messages

    mainline: list[dict[str, object]] = []
    for msg in messages:
        if not msg.get("branch_index"):
            mainline.append(msg)

    mainline_by_parent: dict[str, dict[str, object]] = {}
    for msg in mainline:
        pid = msg.get("parent_message_id")
        if pid:
            mainline_by_parent[str(pid)] = msg

    for msg in messages:
        branch_idx = msg.get("branch_index", 0)
        parent_id = msg.get("parent_message_id")
        if not branch_idx or not parent_id:
            continue

        sibling = mainline_by_parent.get(str(parent_id))
        if sibling is not None:
            branches = sibling.setdefault("branches", [])
            assert isinstance(branches, list)
            branches.append(msg)
        else:
            mainline.append(msg)

    return mainline


def build_projection_html_messages(
    projection: ConversationRenderProjection,
    *,
    render_html,
    preview_limit: int = 120,
) -> list[dict[str, object]]:
    raw_messages: list[dict[str, object]] = []
    for msg in projection.messages:
        text = msg.text or ""
        if not text:
            continue
        payload = build_rendered_message_payload(
            message_id=msg.message_id,
            role=msg.role,
            text=text,
            timestamp=msg.sort_key,
            parent_message_id=msg.parent_message_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        payload["role_class"] = _role_css_class(str(payload["role"]))
        raw_messages.append(payload)
    return _attach_branches(raw_messages)


def build_conversation_html_messages(
    conversation: Conversation,
    *,
    render_html,
    preview_limit: int = 120,
) -> list[dict[str, object]]:
    raw_messages: list[dict[str, object]] = []
    for msg in conversation.messages:
        if not msg.text:
            continue
        payload = build_rendered_message_payload(
            message_id=msg.id,
            role=msg.role,
            text=msg.text,
            timestamp=str(msg.timestamp) if msg.timestamp else None,
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            render_html=render_html,
            preview_limit=preview_limit,
        )
        payload["role_class"] = _role_css_class(str(payload["role"]))
        raw_messages.append(payload)
    return _attach_branches(raw_messages)


__all__ = [
    "_attach_branches",
    "build_conversation_html_messages",
    "build_projection_html_messages",
]
