"""The resident annotation join uses its original pinned User/Index view."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.annotations.join_contracts import AnnotationJoinOperationResult
from polylogue.annotations.schema import AnnotationField, AnnotationSchema
from polylogue.archive.query.execution_control import QueryCancelledError
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.operations.annotation_join import execute_annotation_join
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.ref_resolution import resolve_ref_against_archive
from polylogue.storage.sqlite.archive_tiers.user_annotations import persist_annotation_schema
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from tests.infra.daemon_operations import running_daemon_operations


def _seed_labels(root: Path) -> None:
    schema = AnnotationSchema(
        schema_id="neutral.labels",
        version=1,
        title="Neutral labels",
        fields=(AnnotationField("label", "string"),),
        target_ref_kinds=("session",),
        evidence_policy="optional",
    )
    with sqlite3.connect(root / "user.db") as connection:
        persist_annotation_schema(connection, schema, registered_at_ms=1)
        for number in range(2):
            upsert_assertion(
                connection,
                assertion_id=f"neutral-label-{number}",
                target_ref=f"session:missing-{number}",
                kind=AssertionKind.ANNOTATION,
                key="label",
                value={"_schema": "neutral.labels@v1", "label": "neutral"},
                author_kind="user",
                status=AssertionStatus.ACTIVE,
                now_ms=2,
            )


def _request(*, offset: int = 0) -> dict[str, object]:
    return {"schema_id": "neutral.labels", "schema_version": 1, "statuses": ["active"], "limit": 1, "offset": offset}


def test_resident_join_preserves_paging_and_missing_target_verdict(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_labels) as stack:
        pages = [
            stack.client.operation_to_completion(
                "annotation.join", _request(offset=offset), archive_root=str(stack.archive_root)
            )
            for offset in range(2)
        ]
    for page in pages:
        assert page is not None and page["outcome"] == "completed", page
        result = AnnotationJoinOperationResult.model_validate(page["result"])
        assert result.result.matched_annotation_count == 2
        assert result.result.selected_annotation_count == 1
        assert result.result.missing_target_count == 1
        assert result.result.joined_count == 0
        assert result.outcome.state == "degraded"
        assert result.outcome.reason == "missing_target"
    first, second = pages
    assert first is not None and second is not None
    assert AnnotationJoinOperationResult.model_validate(first["result"]).result.next_offset == 1
    assert AnnotationJoinOperationResult.model_validate(second["result"]).result.next_offset is None


def test_join_reads_the_pinned_user_selection_after_overlay_changes(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_labels) as stack:
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
            assert resolve_ref_against_archive(pinned.archive, "assertion:neutral-label-0").resolved

            def remove_labels() -> None:
                with sqlite3.connect(stack.archive_root / "user.db") as connection:
                    connection.execute("DELETE FROM assertions WHERE assertion_id LIKE 'neutral-label-%'")

            stack.write_bridge.run_sync("fixture.remove-labels", remove_labels)
            result = AnnotationJoinOperationResult.model_validate(
                execute_annotation_join(
                    _request(), archive=pinned.archive, checkpoint=pinned.archive.check_operation_read
                )
            )
            assert result.result.matched_annotation_count == 2
            assert result.result.selected_annotation_count == 1
            assert resolve_ref_against_archive(pinned.archive, "assertion:neutral-label-0").resolved
        with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as current:
            fresh = execute_annotation_join(
                _request(), archive=current.archive, checkpoint=current.archive.check_operation_read
            )
            assert AnnotationJoinOperationResult.model_validate(fresh).result.matched_annotation_count == 0
            assert not resolve_ref_against_archive(current.archive, "assertion:neutral-label-0").resolved


def test_join_preserves_cancellation_and_closes_the_original_reader(tmp_path: Path) -> None:
    cancellation = QueryCancelledError("neutral-cancellation")
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_labels) as stack:
        with pytest.raises(QueryCancelledError) as caught:
            with open_operation_read(stack.archive_root, publication_guard=stack.runtime.publication_guard) as pinned:
                connection = pinned.archive.index_connection

                def cancelled() -> None:
                    raise cancellation

                execute_annotation_join(_request(), archive=pinned.archive, checkpoint=cancelled)
        assert caught.value is cancellation
        assert connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def _seed_assertion_label(root: Path) -> None:
    _seed_labels(root)
    schema = AnnotationSchema(
        schema_id="neutral.assertion-labels",
        version=1,
        title="Neutral assertion labels",
        fields=(AnnotationField("label", "string"),),
        target_ref_kinds=("assertion",),
        evidence_policy="optional",
    )
    with sqlite3.connect(root / "user.db") as connection:
        persist_annotation_schema(connection, schema, registered_at_ms=1)
        upsert_assertion(
            connection,
            assertion_id="neutral-joined-label",
            target_ref="assertion:neutral-label-0",
            kind=AssertionKind.ANNOTATION,
            key="label",
            value={"_schema": "neutral.assertion-labels@v1", "label": "neutral"},
            author_kind="user",
            status=AssertionStatus.ACTIVE,
            now_ms=2,
        )


def test_resident_join_resolves_user_targets_on_the_original_reader(tmp_path: Path) -> None:
    with running_daemon_operations(tmp_path / "archive", seed_archive=_seed_assertion_label) as stack:
        page = stack.client.operation_to_completion(
            "annotation.join",
            {**_request(), "schema_id": "neutral.assertion-labels"},
            archive_root=str(stack.archive_root),
        )
    assert page is not None and page["outcome"] == "completed", page
    joined = AnnotationJoinOperationResult.model_validate(page["result"])
    assert joined.outcome.state == "ok"
    assert joined.result.joined_count == 1
    assert joined.result.rows[0].target_ref == "assertion:neutral-label-0"
    assert joined.result.rows[0].structural["payload_kind"] == "assertion-claim"
