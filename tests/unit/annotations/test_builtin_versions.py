"""Immutable post-floor annotation history and current production admission."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from polylogue.annotations.batch import AnnotationBatch, AnnotationBatchError
from polylogue.annotations.importer import AnnotationBatchImportRequest
from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest
from polylogue.annotations.schema import (
    SEED_ANNOTATION_SCHEMAS,
    AnnotationSchema,
    AnnotationSchemaError,
    AnnotationSchemaRegistry,
    get_annotation_schema,
)
from polylogue.annotations.write import AnnotationValidationError, upsert_annotation_assertion
from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionStatus, Provider
from polylogue.core.refs import ObjectRef
from polylogue.operations.daemon_errors import DaemonOperationRejectedError
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope
from tests.infra.annotation_history import (
    AnnotationHistoryVariant,
    historical_seed_definitions,
    historical_user_rows,
    seed_annotation_history,
)
from tests.infra.annotation_join import join_fixture_annotations
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.daemon_operations import daemon_serving_archive
from tests.infra.live_ingest import write_index_session


@pytest.mark.parametrize("variant", ["pre_5314", "post_5314"])
def test_ordinary_reopen_preserves_both_post_floor_v1_definitions_and_batches(
    tmp_path: Path,
    variant: AnnotationHistoryVariant,
) -> None:
    """Restoring an in-place v1 change makes this ordinary USER open refuse."""
    seed_annotation_history(tmp_path, variant)
    before = historical_user_rows(tmp_path)
    for _ in range(2):
        initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
        with ArchiveStore.open_existing(tmp_path) as archive:
            for item in historical_seed_definitions(variant):
                schema_id = json.loads(item["definition_json"])["schema_id"]
                historical = archive.get_annotation_schema(schema_id, 1)
                assert historical is not None
                assert historical.definition_json == item["definition_json"]
                assert historical.definition_sha256 == item["definition_sha256"]
                assert historical.schema.canonical_definition_json() == item["definition_json"]
                assert hashlib.sha256(historical.definition_json.encode()).hexdigest() == historical.definition_sha256
                current = archive.get_annotation_schema(schema_id)
                assert current is not None and current.schema == get_annotation_schema(schema_id)
                assert current.schema.version == 2
                assert current.definition_sha256 != historical.definition_sha256
                batch = archive.get_annotation_batch(f"historical-{schema_id}")
                assert batch is not None and batch.qualified_schema_id == f"{schema_id}@v1"
                assert archive.list_annotation_batches(
                    schema_id=schema_id, schema_version=1, target_ref=batch.target_ref
                ) == (batch,)
                assert batch.provenance_document()["target_ref"] == batch.target_ref
            assert len(archive.list_annotation_schemas()) == 11
        with ArchiveStore.open_existing(tmp_path, read_only=False) as writer:
            for item in historical_seed_definitions(variant):
                schema_id = json.loads(item["definition_json"])["schema_id"]
                historical = writer.get_annotation_schema(schema_id, 1)
                assert historical is not None
                with pytest.raises(AnnotationSchemaError):
                    writer.save_annotation_schema(replace(historical.schema, title="Conflicting historical definition"))
        assert historical_user_rows(tmp_path) == before


@pytest.mark.asyncio
async def test_retired_historical_target_is_inspectable_but_cannot_be_resolved(tmp_path: Path) -> None:
    """Historical labels are unsupported structural targets, not invalid schema drift."""
    run_off_event_loop(lambda: seed_annotation_history(tmp_path, "pre_5314"))
    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    with ArchiveStore.open_existing(tmp_path) as archive:
        batch = archive.get_annotation_batch("historical-seed.activity")
        assert batch is not None and batch.target_ref == "phase:historical-evidence"
    with sqlite3.connect(tmp_path / "user.db") as conn:
        label = read_assertion_envelope(conn, "label-seed.activity")
        assert label is not None and label.target_ref == batch.target_ref

    class HistoryArchive:
        archive_root = tmp_path

        async def resolve_ref(self, ref: str) -> object:
            raise AssertionError("retired target must not reach live resolution")

        async def get_session_summary(self, session_id: str) -> object:
            raise AssertionError("retired target has no fabricated live summary")

    result = await join_fixture_annotations(
        HistoryArchive(),
        AnnotationStructuralJoinRequest(
            schema_id="seed.activity", schema_version=1, statuses=(AssertionStatus.CANDIDATE,)
        ),
    )
    assert result.qualified_schema_id == "seed.activity@v1"
    assert result.missing_target_count == 1 and result.joined_count == 0
    assert result.invalid_value_count == result.schema_drift_count == 0
    assert [(d.code, d.target_ref) for d in result.diagnostics] == [("missing_target", batch.target_ref)]


@pytest.mark.parametrize(
    "item", historical_seed_definitions("pre_5314"), ids=lambda item: json.loads(item["definition_json"])["schema_id"]
)
def test_history_decoding_does_not_authorize_new_retired_schema_registration(
    tmp_path: Path, item: dict[str, str]
) -> None:
    schema = AnnotationSchema.from_canonical_definition_json(item["definition_json"])
    with pytest.raises(AnnotationSchemaError):
        AnnotationSchemaRegistry().register(schema)
    with ArchiveStore(tmp_path) as archive:
        with pytest.raises(AnnotationSchemaError):
            archive.save_annotation_schema(schema)
        assert archive.get_annotation_schema(schema.schema_id, 1) is None


@pytest.mark.parametrize("schema", SEED_ANNOTATION_SCHEMAS, ids=lambda schema: schema.schema_id)
def test_current_write_stamps_v2_and_occupied_version_drift_still_refuses(
    tmp_path: Path, schema: AnnotationSchema
) -> None:
    with ArchiveStore(tmp_path) as archive:
        assert archive.get_annotation_schema(schema.schema_id, 1) is None
        stored = archive.get_annotation_schema(schema.schema_id)
        assert stored is not None and stored.schema == schema
        with pytest.raises(AnnotationSchemaError):
            archive.save_annotation_schema(replace(schema, title="Conflicting occupied version"))
    with sqlite3.connect(tmp_path / "user.db") as conn:
        target_kind = "message" if schema.schema_id == "seed.goal-event" else "session"
        label = upsert_annotation_assertion(
            conn,
            schema=schema,
            target_ref=f"{target_kind}:synthetic-current",
            value={"abstain": True},
            row_key="current",
            evidence_refs=["session:synthetic-evidence"],
            author_ref="agent:current-labeler",
            now_ms=789,
        )
        assert label.value == {"_schema": f"{schema.schema_id}@v2", "abstain": True}
        for kind in ("phase", "work_event"):
            with pytest.raises(AnnotationValidationError):
                upsert_annotation_assertion(
                    conn,
                    schema=schema,
                    target_ref=f"{kind}:retired",
                    value={"abstain": True},
                    row_key="retired",
                    evidence_refs=["session:synthetic-evidence"],
                    author_ref="agent:current-labeler",
                    now_ms=790,
                )
            with pytest.raises(ValueError):
                ObjectRef.parse(f"{kind}:retired")


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", [None, "pre_5314", "post_5314"])
async def test_actual_facade_daemon_import_records_current_versions_for_all_five_seeds(
    tmp_path: Path,
    variant: AnnotationHistoryVariant | None,
) -> None:
    root = tmp_path / "archive"
    if variant is not None:
        run_off_event_loop(lambda: seed_annotation_history(root, variant))

    def _seed_archive_1() -> Any:
        with ArchiveStore(root) as archive:
            session_id = write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="current-annotation",
                    messages=[
                        ParsedMessage(provider_message_id="m1", role=Role.USER, text="Synthetic annotation evidence")
                    ],
                ),
            )
            return (session_id,)

    (session_id,) = run_off_event_loop(_seed_archive_1)
    with daemon_serving_archive(root):
        async with Polylogue(archive_root=root, db_path=root / "index.db") as api:
            for schema in SEED_ANNOTATION_SCHEMAS:
                target = (
                    f"message:{session_id}:n:m1" if schema.schema_id == "seed.goal-event" else f"session:{session_id}"
                )
                request_input = io.BytesIO(
                    (
                        json.dumps({"row_key": "current", "value": {"abstain": True}, "evidence_refs": [session_id]})
                    ).encode("utf-8")
                )
                request = AnnotationBatchImportRequest(
                    batch_id=f"current-{schema.schema_id}",
                    schema_id=schema.schema_id,
                    schema_version=schema.version,
                    target_ref=target,
                    source_result_ref="result-set:current-evidence",
                    actor_ref="agent:current-labeler",
                    model_ref="agent:current-model",
                    prompt_ref="block:current-prompt:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
                    created_at_ms=789,
                )
                result = await api.import_annotation_batch(request, input=request_input)
                assert result.qualified_schema_id == f"{schema.schema_id}@v2" and result.valid_count == 1
            for kind in ("phase", "work_event"):
                retired_input = io.BytesIO(
                    (
                        json.dumps({"row_key": "retired", "value": {"abstain": True}, "evidence_refs": [session_id]})
                    ).encode("utf-8")
                )
                retired = AnnotationBatchImportRequest(
                    batch_id=f"retired-{kind}",
                    schema_id="seed.activity",
                    schema_version=2,
                    target_ref=f"{kind}:historical-evidence",
                    source_result_ref="result-set:current-evidence",
                    actor_ref="agent:current-labeler",
                    model_ref="agent:current-model",
                    prompt_ref="block:current-prompt:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
                    created_at_ms=790,
                )
                with pytest.raises(DaemonOperationRejectedError):
                    await api.import_annotation_batch(retired, input=retired_input)
    with ArchiveStore.open_existing(root) as reopened:
        assert reopened.get_annotation_batch("retired-phase") is None
        assert reopened.get_annotation_batch("retired-work_event") is None
        for schema in SEED_ANNOTATION_SCHEMAS:
            batch = reopened.get_annotation_batch(f"current-{schema.schema_id}")
            assert batch is not None and batch.schema_version == 2
            with sqlite3.connect(root / "user.db") as conn:
                label = read_assertion_envelope(conn, batch.assertion_refs[0].removeprefix("assertion:"))
                assert label is not None and isinstance(label.value, dict)
                assert label.value["_schema"] == batch.qualified_schema_id


@pytest.mark.parametrize("kind", ["invented", "phase:", "work_event:"])
def test_provenance_decoding_still_refuses_unknown_or_malformed_targets(kind: str) -> None:
    item = historical_seed_definitions("pre_5314")[0]
    document = json.loads(item["definition_json"])
    document["target_ref_kinds"] = [kind]
    with pytest.raises(AnnotationSchemaError):
        AnnotationSchema.from_definition_document(document)
    with pytest.raises(AnnotationBatchError):
        AnnotationBatch(
            batch_id="invalid",
            schema_id="seed.activity",
            schema_version=1,
            target_ref=f"{kind}:" if kind == "invented" else kind,
            source_result_ref="result-set:evidence",
            actor_ref="agent:labeler",
            model_ref="agent:model",
            prompt_ref="block:prompt:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
            total_count=0,
            valid_count=0,
            invalid_count=0,
            abstained_count=0,
            created_at_ms=1,
        )
