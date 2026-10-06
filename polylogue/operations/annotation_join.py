"""Compose the single annotation join against one pinned Index/User read."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.annotations.join import AnnotationStructuralJoinError, join_typed_annotations
from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest
from polylogue.archive.hydration import archive_summary_to_domain
from polylogue.archive.query.execution_control import QueryExecutionContext
from polylogue.core.async_bridge import complete_without_suspension
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.ref_resolution import resolve_ref_against_archive
from polylogue.surfaces.outcome import decide_outcome

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.surfaces.payloads import PublicRefResolutionPayload


class _PinnedAnnotationReader:
    def __init__(self, archive: ArchiveStore) -> None:
        self.archive = archive

    @property
    def archive_root(self) -> Path:
        return self.archive.archive_root

    async def resolve_ref(self, ref: str) -> PublicRefResolutionPayload:
        self.archive.check_operation_read()
        return resolve_ref_against_archive(self.archive, ref)

    async def get_session_summary(self, session_id: str) -> object | None:
        self.archive.check_operation_read()
        try:
            summary = self.archive.read_summary(session_id)
        except KeyError:
            return None
        return archive_summary_to_domain(summary)


def execute_annotation_join(
    payload: dict[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None]
) -> dict[str, object]:
    request = AnnotationStructuralJoinRequest.model_validate(payload)
    connection = archive.index_connection
    if connection is None or "user" not in (archive.operation_schema_versions or {}):
        raise AnnotationStructuralJoinError("annotation user tier is not initialized")
    checkpoint()
    result = complete_without_suspension(
        join_typed_annotations(
            _PinnedAnnotationReader(archive),
            request,
            user_conn=connection,
            user_schema="user_tier",
            checkpoint=checkpoint,
        )
    )
    checkpoint()
    gaps = tuple(
        name
        for name, count in (
            ("missing_target", result.missing_target_count),
            ("ambiguous_target", result.ambiguous_target_count),
            ("schema_drift", result.schema_drift_count),
            ("invalid_value", result.invalid_value_count),
        )
        if count
    )
    return {
        "result": result.model_dump(mode="json"),
        "outcome": decide_outcome(matched=result.joined_count, degraded=gaps).to_dict(),
    }


@contextmanager
def open_annotation_join_read(root: Path, context: QueryExecutionContext) -> Iterator[ArchiveStore]:
    """Borrow the canonical optimistic publication pin under the original read control."""
    with open_operation_read(root, execution_context=context, read_timeout=5.0) as pinned:
        yield pinned.archive
