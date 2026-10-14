"""Independent session metadata paging on the caller's archive frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.archive.query.transaction import (
    QueryContinuation,
    QueryContinuationInvalidError,
    QueryTransactionRequest,
    archive_snapshot_epoch,
    validate_continuation_epoch,
)
from polylogue.operations.read_contracts import SessionMetadataPage

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class SessionMetadataWindow:
    working_directories: tuple[str, ...]
    user_tags: tuple[str, ...]
    page: SessionMetadataPage


def read_session_metadata(
    archive: ArchiveStore,
    session_id: str,
    *,
    limit: int | None,
    continuation: str | None = None,
    include_user_tags: bool = True,
) -> SessionMetadataWindow:
    """Advance metadata independently of messages, refusing a changed Index/User frame."""
    arguments = {"session_id": session_id, "include_user_tags": include_user_tags}
    if continuation is not None:
        decoded = QueryContinuation.decode(continuation)
        tx = decoded.request
        if (
            tx.operation != "session.metadata"
            or tx.projection != "session-metadata-v1"
            or tx.stable_order != "directory-position,tag"
            or tx.arguments != arguments
            or decoded.result_ref != tx.result_ref
            or decoded.cursor is not None
            or (limit is not None and limit > tx.page_size)
        ):
            raise QueryContinuationInvalidError("continuation does not identify this metadata window")
        validate_continuation_epoch(tx, archive=archive)
        if limit is not None:
            from dataclasses import replace

            tx = replace(tx, page_size=limit)
    else:
        relations = (
            ("sessions", "session_working_dirs", "assertions")
            if include_user_tags
            else ("sessions", "session_working_dirs")
        )
        tx = QueryTransactionRequest(
            operation="session.metadata",
            arguments=arguments,
            page_size=limit if limit is not None else 200,
            projection="session-metadata-v1",
            stable_order="directory-position,tag",
            archive_epoch=archive_snapshot_epoch(archive, relations=relations),
        )
    archive.check_operation_read()
    directories, tags, directory_total, tag_total = archive.read_session_metadata_window(
        session_id, limit=tx.page_size, offset=tx.offset, include_user_tags=include_user_tags
    )
    archive.check_operation_read()
    validate_continuation_epoch(tx, archive=archive)
    total = max(directory_total, tag_total or 0)
    if tx.offset > total:
        raise QueryContinuationInvalidError("metadata continuation is outside its relation")
    next_offset = min(tx.offset + tx.page_size, total) if tx.offset + tx.page_size < total else None
    token = QueryContinuation(tx.next(offset=next_offset), tx.result_ref).encode() if next_offset is not None else None
    return SessionMetadataWindow(
        directories,
        tags,
        SessionMetadataPage(
            working_directories_total=directory_total,
            user_tags_total=tag_total,
            limit=tx.page_size,
            offset=tx.offset,
            next_offset=next_offset,
            continuation=token,
            complete=next_offset is None,
        ),
    )
