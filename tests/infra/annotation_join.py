"""Original User read operands for product annotation evaluator controls."""

from __future__ import annotations

from contextlib import closing

from polylogue.annotations.join import StructuralJoinArchive, join_typed_annotations
from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest, AnnotationStructuralJoinResult
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


async def join_fixture_annotations(
    reader: StructuralJoinArchive, request: AnnotationStructuralJoinRequest
) -> AnnotationStructuralJoinResult:
    with closing(
        open_readonly_connection(
            reader.archive_root / "user.db", timeout_class="background-read", validate_schema=False
        )
    ) as connection:
        return await join_typed_annotations(
            reader, request, user_conn=connection, user_schema=None, checkpoint=lambda: None
        )
