"""Terminal unit-query execution over the archive."""

from __future__ import annotations

from polylogue.archive.query.expression import QueryUnitSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    BlockQueryRowPayload,
    MessageQueryRowPayload,
    QueryUnitEnvelope,
    build_query_unit_envelope,
)


def query_unit_rows(
    archive: ArchiveStore,
    source: QueryUnitSource,
    *,
    query: str,
    limit: int,
    offset: int = 0,
) -> QueryUnitEnvelope:
    """Execute an explicit ``messages/actions/blocks where`` source query."""

    fetch_limit = limit + 1
    if source.unit == "message":
        message_rows = archive.query_messages(source.predicate, limit=fetch_limit, offset=offset)
        return build_query_unit_envelope(
            tuple(MessageQueryRowPayload.from_row(row) for row in message_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(message_rows) > limit,
        )
    if source.unit == "action":
        action_rows = archive.query_actions(source.predicate, limit=fetch_limit, offset=offset)
        return build_query_unit_envelope(
            tuple(ActionQueryRowPayload.from_row(row) for row in action_rows[:limit]),
            unit=source.unit,
            query=query,
            limit=limit,
            offset=offset,
            has_next=len(action_rows) > limit,
        )
    block_rows = archive.query_blocks(source.predicate, limit=fetch_limit, offset=offset)
    return build_query_unit_envelope(
        tuple(BlockQueryRowPayload.from_row(row) for row in block_rows[:limit]),
        unit=source.unit,
        query=query,
        limit=limit,
        offset=offset,
        has_next=len(block_rows) > limit,
    )


__all__ = ["query_unit_rows"]
