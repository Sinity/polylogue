"""Pinned-archive execution for the CLI compact read view."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.archive.hydration import archive_message_to_domain
from polylogue.core.enums import Origin
from polylogue.operations.read_view_chronicle import _chronicle_plan
from polylogue.operations.read_view_selection import select_read_view_summaries
from polylogue.surfaces.compaction import CompactProjectionSpec, compact_sessions

if TYPE_CHECKING:
    from polylogue.archive.message.models import Message
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


_PAGE_SIZE = 256


def _projection_spec(payload: Mapping[str, object]) -> CompactProjectionSpec:
    projection = payload.get("projection")
    raw = projection.get("max_tokens") if isinstance(projection, Mapping) else None
    if raw is None:
        return CompactProjectionSpec()
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        raise ValueError("compact max_tokens must be an integer")
    return CompactProjectionSpec(max_tokens=int(raw))


def _composed_session(
    archive: ArchiveStore, session_id: str, *, origin: Origin
) -> tuple[list[Message], dict[str, object] | None]:
    """Read one composed transcript page by page, plus its prefix-lineage link."""

    messages: list[Message] = []
    link: dict[str, object] | None = None
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        page = archive.read_session_page(session_id, limit=_PAGE_SIZE, offset=offset)
        if total is None:
            total = page.total_message_count if page.total_message_count is not None else len(page.messages)
            if page.lineage_inheritance == "prefix-sharing" and page.parent_session_id:
                link = {
                    "src_session_id": session_id,
                    "resolved_dst_session_id": page.parent_session_id,
                    "branch_point_message_id": page.lineage_branch_point_message_id,
                }
        if not page.messages:
            break
        messages.extend(archive_message_to_domain(message, origin=origin) for message in page.messages)
        offset += len(page.messages)
    return messages, link


def execute_compact_read(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    vector_provider: VectorProvider | None = None,
) -> dict[str, object]:
    """Execute one corpus-compaction projection against the caller's pinned archive."""

    summaries = select_read_view_summaries(
        _chronicle_plan(payload, vector_provider=vector_provider), archive=archive, default_limit=5
    )
    sessions: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    for summary in summaries:
        session_id = str(summary.id)
        messages, link = _composed_session(archive, session_id, origin=Origin.from_string(summary.origin))
        sessions.append({"id": session_id, "messages": messages})
        if link is not None:
            links.append(link)
    pack = compact_sessions(sessions, spec=_projection_spec(payload), session_links=links)
    wire: dict[str, object] = {"view": "compact", "payload": pack.model_dump(mode="json")}
    return wire


__all__ = ["execute_compact_read"]
