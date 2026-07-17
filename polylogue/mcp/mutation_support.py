"""Private shared primitives for mutation-oriented MCP registrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from polylogue.mcp.server_support import ServerCallbacks


async def resolve_session_or_error(hooks: ServerCallbacks, session_id: str) -> tuple[str | None, str | None]:
    """Resolve a session ID, returning the canonical ID or an error JSON."""
    summary = await hooks.get_polylogue().get_session_summary(session_id)
    if summary is None:
        return None, hooks.error_json("Session not found", code="not_found", session_id=session_id)
    return str(summary.id), None


TItem = TypeVar("TItem")


def page_items(items: Sequence[TItem], *, limit: int, offset: int) -> tuple[tuple[TItem, ...], int, int, int | None]:
    """Slice a list response while retaining deterministic continuation state."""
    total = len(items)
    page_offset = max(0, offset)
    page = tuple(items[page_offset : page_offset + limit])
    next_offset = page_offset + len(page) if page_offset + len(page) < total else None
    return page, total, page_offset, next_offset
