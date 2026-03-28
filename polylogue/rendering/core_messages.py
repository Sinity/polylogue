"""Shared rendered-message payload helpers."""

from __future__ import annotations

from collections.abc import Callable


def build_rendered_message_payload(
    *,
    message_id: object,
    role: object,
    text: str,
    timestamp: object,
    render_html: Callable[[str], str],
    parent_message_id: object = None,
    branch_index: object = 0,
    preview_limit: int | None = None,
) -> dict[str, object]:
    """Build a canonical rendered-message payload shared by HTML/site surfaces."""
    normalized_role = role or "message"
    if hasattr(normalized_role, "value"):
        normalized_role = normalized_role.value
    return {
        "id": message_id,
        "role": str(normalized_role),
        "text": text[:preview_limit] if preview_limit is not None else text,
        "html_content": render_html(text),
        "timestamp": timestamp,
        "parent_message_id": parent_message_id,
        "branch_index": branch_index,
    }


__all__ = ["build_rendered_message_payload"]
