"""Pinned-archive execution for the CLI compact read view."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
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
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope


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
    archive: ArchiveStore, session_id: str, *, origin: Origin, checkpoint: Callable[[], None]
) -> dict[str, object]:
    """Supply one composed page at a time with its already observed lineage."""

    checkpoint()
    archive.check_operation_read()
    first = archive.read_session_page(session_id, limit=_PAGE_SIZE, offset=0, include_working_directories=False)
    link: dict[str, object] = {}
    if first.lineage_inheritance == "prefix-sharing" and first.parent_session_id:
        link = {
            "src_session_id": session_id,
            "resolved_dst_session_id": first.parent_session_id,
            "branch_point_message_id": first.lineage_branch_point_message_id,
        }

    def messages(page: ArchiveSessionEnvelope) -> Iterator[Message]:
        total = page.total_message_count
        offset = 0
        while True:
            for message in page.messages:
                checkpoint()
                yield archive_message_to_domain(message, origin=origin)
            offset += len(page.messages)
            if not page.messages or (total is not None and offset >= total):
                return
            checkpoint()
            archive.check_operation_read()
            page = archive.read_session_page(
                session_id, limit=_PAGE_SIZE, offset=offset, include_working_directories=False
            )

    return {"id": session_id, "messages": messages(first), "lineage_link": link}


def execute_compact_read(
    payload: Mapping[str, object],
    *,
    archive: ArchiveStore,
    vector_provider: VectorProvider | None = None,
    checkpoint: Callable[[], None] = lambda: None,
) -> dict[str, object]:
    """Execute one corpus-compaction projection against the caller's pinned archive."""

    summaries = select_read_view_summaries(
        _chronicle_plan(payload, vector_provider=vector_provider), archive=archive, default_limit=5
    )

    def sessions() -> Iterator[dict[str, object]]:
        for summary in summaries:
            checkpoint()
            session_id = str(summary.id)
            yield _composed_session(
                archive, session_id, origin=Origin.from_string(summary.origin), checkpoint=checkpoint
            )

    pack = compact_sessions(sessions(), spec=_projection_spec(payload), checkpoint=checkpoint)
    archive.check_operation_read()
    wire: dict[str, object] = {"view": "compact", "payload": pack.model_dump(mode="json")}
    return wire


__all__ = ["execute_compact_read"]
