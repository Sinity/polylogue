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
    from polylogue.surfaces.payloads import make_page

    total = len(items)
    page_offset = max(0, offset)
    page = make_page(
        items[page_offset : page_offset + limit],
        matched=total,
        continuation=(str(page_offset + limit) if page_offset + limit < total else None),
    )
    next_offset = int(page.continuation) if page.continuation is not None else None
    return page.items, page.matched or 0, page_offset, next_offset
