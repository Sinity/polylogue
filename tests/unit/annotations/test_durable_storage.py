"""Durable annotation schema and batch provenance contracts (polylogue-rxdo.7.1)."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.annotations.batch import AnnotationBatch, AnnotationBatchError
from polylogue.annotations.schema import (
    DELEGATION_DISCOURSE_SCHEMA,
    AnnotationField,
    AnnotationSchema,
    AnnotationSchemaError,
    AnnotationSchemaRegistry,
)
from polylogue.annotations.write import (
    AnnotationValidationError,
    assertion_id_for_schema_annotation,
    upsert_annotation_assertion,
)
from polylogue.core.enums import AssertionStatus
from polylogue.core.json import JSONDocument
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    judge_assertion_candidate,
    read_assertion_envelope,
)


def _custom_schema(*, title: str = "Durable test schema") -> AnnotationSchema:
    return AnnotationSchema(
        schema_id="test.durable",
        version=1,
        title=title,
        description="Cold-reopen schema fixture.",
        fields=(
            AnnotationField(name="label", value_type="enum", enum_values=("yes", "no")),
            AnnotationField(name="confidence", value_type="number", minimum=0, maximum=1),
            AnnotationField(name="abstain", value_type="boolean", required=False),
        ),
        target_ref_kinds=("delegation",),
        abstain_field="abstain",
        evidence_policy="required",
        status="active",
    )


def _delegation_value() -> dict[str, object]:
    return {
        "directive_mode": "goal_delegation",
        "prohibitions": "explicit",
        "autonomy": "bounded",
        "output_contract": "structured",
        "scope_control": "owned_and_avoid_paths",
        "verification_demand": "focused_tests",
        "checkpoint_escalation": "escalation",
        "relational_frame": "collaborative",
        "rationale_visibility": "explicit",
        "applicable": True,
        "confidence": 0.9,
    }


def _single_row_batch(
    *,
    batch_id: str,
    target_ref: str = "delegation:dispatch-block-1",
    author_ref: str = "agent:labeler",
    row_key: str = "row-1",
    metadata: JSONDocument | None = None,
) -> tuple[AnnotationBatch, str]:
    schema = DELEGATION_DISCOURSE_SCHEMA
    batch_ref = f"annotation-batch:{batch_id}"
    assertion_id = assertion_id_for_schema_annotation(
        schema_qualified_id=schema.qualified_id,
        target_ref=target_ref,
        author_ref=author_ref,
        row_key=row_key,
        batch_ref=batch_ref,
    )
    return (
        AnnotationBatch(
            batch_id=batch_id,
            schema_id=schema.schema_id,
            schema_version=schema.version,
            target_ref=target_ref,
            source_result_ref=f"result-set:{batch_id}",
            actor_ref=author_ref,
            model_ref="agent:gpt-5.6-terra",
            prompt_ref=f"block:{batch_id}:0",
            total_count=1,
            valid_count=1,
            invalid_count=0,
            abstained_count=0,
            assertion_refs=(f"assertion:{assertion_id}",),
            metadata={} if metadata is None else metadata,
            created_at_ms=1_000,
        ),
        assertion_id,
    )


def test_batch_id_must_form_a_round_trippable_public_ref() -> None:
    batch, _ = _single_row_batch(batch_id="round-trip-baseline")

    with pytest.raises(AnnotationBatchError, match="batch_id.*round-trippable annotation-batch ObjectRef"):
        replace(batch, batch_id="run:")


def test_schema_definition_is_durable_canonical_and_fails_closed_on_reuse(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    schema = _custom_schema()

    with ArchiveStore(archive_root) as archive:
        builtin = archive.get_annotation_schema("delegation.discourse", 1)
        assert builtin is not None
        assert builtin.schema == DELEGATION_DISCOURSE_SCHEMA
        assert builtin.definition_json == DELEGATION_DISCOURSE_SCHEMA.canonical_definition_json()
        assert builtin.definition_sha256 == DELEGATION_DISCOURSE_SCHEMA.definition_fingerprint

        persisted = archive.save_annotation_schema(schema, registered_at_ms=123)
        assert persisted.definition_json == schema.canonical_definition_json()
        assert persisted.definition_sha256 == schema.definition_fingerprint
        assert archive.save_annotation_schema(schema, registered_at_ms=999) == persisted
        with pytest.raises(AnnotationSchemaError, match="incompatible durable definition"):
            archive.save_annotation_schema(_custom_schema(title="Drifted"), registered_at_ms=456)

    cold_registry = AnnotationSchemaRegistry()
    with ArchiveStore.open_existing(archive_root) as reopened:
        resolved = reopened.get_annotation_schema("test.durable", 1)
        assert resolved is not None
        assert resolved.schema == schema
        assert resolved.definition_json == schema.canonical_definition_json()
        assert resolved.definition_sha256 == schema.definition_fingerprint
        assert {entry.schema.qualified_id for entry in reopened.list_annotation_schemas()} >= {
            "delegation.discourse@v1",
            "test.durable@v1",
        }
        cold_registry.register(resolved.schema)
        assert cold_registry.require_active(schema) == schema

    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.execute(
            """
            UPDATE annotation_schemas
            SET definition_sha256 = ?
            WHERE schema_id = 'test.durable' AND schema_version = 1
            """,
            ("0" * 64,),
        )
        conn.commit()
    with ArchiveStore.open_existing(archive_root) as reopened:
        with pytest.raises(AnnotationSchemaError, match="fingerprint mismatch"):
            reopened.get_annotation_schema("test.durable", 1)


def test_registered_delegation_schema_exposes_separate_campaign_dimensions() -> None:
    schema = DELEGATION_DISCOURSE_SCHEMA
    assert schema.status == "active"
    assert schema.target_ref_kinds == ("delegation",)
    assert schema.abstain_field == "abstain"
    assert schema.evidence_policy == "required"
    assert {field.name for field in schema.fields} == {
        "directive_mode",
        "prohibitions",
        "autonomy",
        "output_contract",
        "scope_control",
        "verification_demand",
        "checkpoint_escalation",
        "relational_frame",
        "rationale_visibility",
        "applicable",
        "confidence",
        "abstain",
        "rationale",
    }
    assert schema.definition_fingerprint == "7cb761fc365caaf40ca98a96c4d6d809284fa3aa651656f26e7b01e2476f06e9"


def test_two_batches_for_same_schema_target_remain_distinct_and_scope_assertions(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    schema = DELEGATION_DISCOURSE_SCHEMA
    target_ref = "delegation:dispatch-block-1"
    author_ref = "agent:labeler"
    batch_a_ref = "annotation-batch:batch-a"
    batch_b_ref = "annotation-batch:batch-b"
    assertion_a = assertion_id_for_schema_annotation(
        schema_qualified_id=schema.qualified_id,
        target_ref=target_ref,
        author_ref=author_ref,
        row_key="row-1",
        batch_ref=batch_a_ref,
    )
    assertion_b = assertion_id_for_schema_annotation(
        schema_qualified_id=schema.qualified_id,
        target_ref=target_ref,
        author_ref=author_ref,
        row_key="row-1",
        batch_ref=batch_b_ref,
    )
    assert assertion_a != assertion_b

    batch_a = AnnotationBatch(
        batch_id="batch-a",
        schema_id=schema.schema_id,
        schema_version=schema.version,
        target_ref=target_ref,
        source_result_ref="result-set:delegation-sample-a",
        actor_ref=author_ref,
        model_ref="agent:gpt-5.6-terra",
        prompt_ref="block:prompt-a:0",
        total_count=2,
        valid_count=1,
        invalid_count=1,
        abstained_count=0,
        assertion_refs=(f"assertion:{assertion_a}",),
        validation_failures=({"row": 2, "errors": ["missing evidence"]},),
        metadata={"cohort": "a", "temperature": 0},
        created_at_ms=1_000,
    )
    batch_b = AnnotationBatch(
        batch_id="batch-b",
        schema_id=schema.schema_id,
        schema_version=schema.version,
        target_ref=target_ref,
        source_result_ref="result-set:delegation-sample-b",
        actor_ref=author_ref,
        model_ref="agent:gpt-5.6-terra",
        prompt_ref="block:prompt-b:0",
        total_count=1,
        valid_count=1,
        invalid_count=0,
        abstained_count=0,
        assertion_refs=(f"assertion:{assertion_b}",),
        metadata={"cohort": "b", "temperature": 0},
        created_at_ms=2_000,
    )

    with ArchiveStore(archive_root) as archive:
        assert archive.save_annotation_batch(batch_a) == batch_a
        assert archive.save_annotation_batch(batch_b) == batch_b
        assert archive.save_annotation_batch(batch_a) == batch_a
        with pytest.raises(AnnotationBatchError, match="incompatible provenance"):
            archive.save_annotation_batch(replace(batch_a, metadata={"cohort": "drifted"}))
        assert archive.get_annotation_batch("batch-a") == batch_a
        assert archive.list_annotation_batches(
            schema_id=schema.schema_id,
            schema_version=schema.version,
            target_ref=target_ref,
        ) == (batch_b, batch_a)

    with ArchiveStore.open_existing(archive_root) as reopened:
        assert reopened.get_annotation_batch("batch-a") == batch_a
        assert reopened.get_annotation_batch("batch-b") == batch_b
        assert reopened.list_annotation_batches(target_ref=target_ref) == (batch_b, batch_a)

    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    with sqlite3.connect(archive_root / "user.db") as conn:
        row_a = upsert_annotation_assertion(
            conn,
            schema=schema,
            registry=registry,
            target_ref=target_ref,
            value=_delegation_value(),
            row_key="row-1",
            evidence_refs=("session:codex-session:delegation-demo",),
            author_ref=author_ref,
            batch_ref=batch_a_ref,
            now_ms=3_000,
        )
        row_b = upsert_annotation_assertion(
            conn,
            schema=schema,
            registry=registry,
            target_ref=target_ref,
            value=_delegation_value(),
            row_key="row-1",
            evidence_refs=("session:codex-session:delegation-demo",),
            author_ref=author_ref,
            batch_ref=batch_b_ref,
            now_ms=3_001,
        )
        with pytest.raises(AnnotationValidationError, match="does not declare assertion"):
            upsert_annotation_assertion(
                conn,
                schema=schema,
                registry=registry,
                target_ref=target_ref,
                value=_delegation_value(),
                row_key="unlisted-row",
                evidence_refs=("session:codex-session:delegation-demo",),
                author_ref=author_ref,
                batch_ref=batch_a_ref,
                now_ms=3_002,
            )
        with pytest.raises(AnnotationValidationError, match="targets 'delegation:dispatch-block-1'"):
            upsert_annotation_assertion(
                conn,
                schema=schema,
                registry=registry,
                target_ref="delegation:different-target",
                value=_delegation_value(),
                row_key="row-1",
                evidence_refs=("session:codex-session:delegation-demo",),
                author_ref=author_ref,
                batch_ref=batch_a_ref,
                now_ms=3_003,
            )
        conn.commit()
        stored = conn.execute(
            """
            SELECT assertion_id, scope_ref, value_json
            FROM assertions
            WHERE assertion_id IN (?, ?)
            ORDER BY assertion_id
            """,
            (assertion_a, assertion_b),
        ).fetchall()

    assert {row_a.assertion_id, row_b.assertion_id} == {assertion_a, assertion_b}
    assert {str(row[1]) for row in stored} == {batch_a_ref, batch_b_ref}
    assert all('"_batch":"annotation-batch:batch-' in str(row[2]) for row in stored)


def test_batch_requires_registered_schema_and_consistent_counts(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    with pytest.raises(AnnotationBatchError, match="must equal total_count"):
        AnnotationBatch(
            batch_id="bad-counts",
            schema_id="delegation.discourse",
            schema_version=1,
            target_ref="delegation:x",
            source_result_ref="result-set:x",
            actor_ref="agent:x",
            model_ref="agent:model",
            prompt_ref="block:prompt:0",
            total_count=2,
            valid_count=1,
            invalid_count=0,
            abstained_count=0,
            assertion_refs=("assertion:x",),
        )

    unknown = AnnotationBatch(
        batch_id="unknown-schema",
        schema_id="test.unknown",
        schema_version=1,
        target_ref="delegation:x",
        source_result_ref="result-set:x",
        actor_ref="agent:x",
        model_ref="agent:model",
        prompt_ref="block:prompt:0",
        total_count=0,
        valid_count=0,
        invalid_count=0,
        abstained_count=0,
        created_at_ms=1,
    )
    with ArchiveStore(archive_root) as archive:
        with pytest.raises(AnnotationBatchError, match="unknown schema"):
            archive.save_annotation_batch(unknown)


def test_batch_persistence_enforces_schema_target_source_and_canonical_json(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    batch, assertion_id = _single_row_batch(batch_id="boundary-validation")

    with ArchiveStore(archive_root) as archive:
        with pytest.raises(AnnotationBatchError, match="incompatible with schema target_ref_kinds"):
            archive.save_annotation_batch(replace(batch, target_ref="session:not-a-delegation"))

        poisoned_source = replace(batch, batch_id="poisoned-source")
        object.__setattr__(poisoned_source, "source_result_ref", "analysis:not-a-result-set")
        with pytest.raises(AnnotationBatchError, match="source_result_ref must use the 'result-set'"):
            archive.save_annotation_batch(poisoned_source)

    with pytest.raises(AnnotationBatchError, match="unique after normalization"):
        AnnotationBatch(
            batch_id="duplicate-refs",
            schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
            schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
            target_ref=batch.target_ref,
            source_result_ref="result-set:duplicate-refs",
            actor_ref=batch.actor_ref,
            model_ref=batch.model_ref,
            prompt_ref=batch.prompt_ref,
            total_count=2,
            valid_count=2,
            invalid_count=0,
            abstained_count=0,
            assertion_refs=(f"assertion:{assertion_id}", f"assertion:{assertion_id}"),
        )
    with pytest.raises(AnnotationBatchError, match="finite canonical JSON"):
        replace(batch, metadata={"score": math.nan})


@pytest.mark.parametrize(
    ("original_metadata", "retry_metadata", "original_failures", "retry_failures"),
    [
        ({"score": 1}, {"score": True}, (), ()),
        ({"score": 1}, {"score": 1.0}, (), ()),
        ({"score": 0.0}, {"score": -0.0}, (), ()),
        ({}, {}, ({"nested": {"score": 1}},), ({"nested": {"score": True}},)),
    ],
)
def test_batch_exact_retry_compares_canonical_provenance_not_python_equality(
    workspace_env: dict[str, Path],
    original_metadata: JSONDocument,
    retry_metadata: JSONDocument,
    original_failures: tuple[JSONDocument, ...],
    retry_failures: tuple[JSONDocument, ...],
) -> None:
    def batch(metadata: JSONDocument, failures: tuple[JSONDocument, ...]) -> AnnotationBatch:
        return AnnotationBatch(
            batch_id="canonical-retry",
            schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
            schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
            target_ref="delegation:canonical-retry",
            source_result_ref="result-set:canonical-retry",
            actor_ref="agent:labeler",
            model_ref="agent:model",
            prompt_ref="block:canonical-retry:0",
            total_count=len(failures),
            valid_count=0,
            invalid_count=len(failures),
            abstained_count=0,
            validation_failures=failures,
            metadata=metadata,
            created_at_ms=1,
        )

    original = batch(original_metadata, original_failures)
    drifted = batch(retry_metadata, retry_failures)
    assert original == drifted
    assert original.canonical_provenance_bytes() != drifted.canonical_provenance_bytes()

    with ArchiveStore(workspace_env["archive_root"]) as archive:
        archive.save_annotation_batch(original)
        with pytest.raises(AnnotationBatchError, match="incompatible provenance"):
            archive.save_annotation_batch(drifted)


def test_batch_canonical_nfc_retry_returns_existing_durable_provenance(workspace_env: dict[str, Path]) -> None:
    original = AnnotationBatch(
        batch_id="nfc-retry",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref="delegation:nfc-retry",
        source_result_ref="result-set:nfc-retry",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref="block:nfc-retry:0",
        total_count=0,
        valid_count=0,
        invalid_count=0,
        abstained_count=0,
        metadata={"label": "Caf\u00e9"},
        created_at_ms=1,
    )
    retry = replace(original, metadata={"label": "Cafe\u0301"})
    assert retry.metadata == original.metadata
    assert original.canonical_provenance_bytes() == retry.canonical_provenance_bytes()

    with ArchiveStore(workspace_env["archive_root"]) as archive:
        stored = archive.save_annotation_batch(original)
        replay = archive.save_annotation_batch(retry)

    assert replay == stored
    assert replay.metadata == {"label": "Caf\u00e9"}


def test_batch_opaque_refs_preserve_decomposed_bytes_across_retry_and_cold_read(
    workspace_env: dict[str, Path],
) -> None:
    decomposed = "Cafe\u0301"
    composed = "Caf\u00e9"
    original = AnnotationBatch(
        batch_id="opaque-ref-retry",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref=f"delegation:{decomposed}",
        source_result_ref=f"result-set:{decomposed}",
        actor_ref=f"agent:{decomposed}",
        model_ref=f"agent:model-{decomposed}",
        prompt_ref=f"block:{decomposed}:0",
        total_count=1,
        valid_count=1,
        invalid_count=0,
        abstained_count=0,
        assertion_refs=(f"assertion:{decomposed}",),
        created_at_ms=1,
    )
    exact_retry = replace(original)
    composed_target = replace(original, target_ref=f"delegation:{composed}")

    assert original.canonical_provenance_bytes() == exact_retry.canonical_provenance_bytes()
    assert original.canonical_provenance_bytes() != composed_target.canonical_provenance_bytes()

    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        archive.save_annotation_batch(original)

    with ArchiveStore.open_existing(archive_root) as reopened:
        cold = reopened.get_annotation_batch(original.batch_id)
        assert cold is not None
        assert cold.target_ref == f"delegation:{decomposed}"
        assert cold.source_result_ref == f"result-set:{decomposed}"
        assert cold.actor_ref == f"agent:{decomposed}"
        assert cold.model_ref == f"agent:model-{decomposed}"
        assert cold.prompt_ref == f"block:{decomposed}:0"
        assert cold.assertion_refs == (f"assertion:{decomposed}",)
        assert cold.canonical_provenance_bytes() == original.canonical_provenance_bytes()
        replay = reopened.save_annotation_batch(exact_retry)
        assert replay.canonical_provenance_bytes() == original.canonical_provenance_bytes()
        with pytest.raises(AnnotationBatchError, match="incompatible provenance"):
            reopened.save_annotation_batch(composed_target)


@pytest.mark.parametrize(
    ("metadata", "validation_failures"),
    [
        ({"nested": {"\u00e9": 1, "e\u0301": 2}}, ()),
        ({}, ({"nested": {"\u00e9": 1, "e\u0301": 2}},)),
    ],
)
def test_batch_rejects_nfc_normalized_key_collisions_at_every_depth(
    metadata: JSONDocument,
    validation_failures: tuple[JSONDocument, ...],
) -> None:
    with pytest.raises(AnnotationBatchError, match="NFC-normalized JSON keys collide"):
        AnnotationBatch(
            batch_id="nfc-key-collision",
            schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
            schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
            target_ref="delegation:nfc-key-collision",
            source_result_ref="result-set:nfc-key-collision",
            actor_ref="agent:labeler",
            model_ref="agent:model",
            prompt_ref="block:nfc-key-collision:0",
            total_count=len(validation_failures),
            valid_count=0,
            invalid_count=len(validation_failures),
            abstained_count=0,
            validation_failures=validation_failures,
            metadata=metadata,
            created_at_ms=1,
        )


def test_batch_provenance_snapshot_detaches_nested_aliases_through_cold_storage(
    workspace_env: dict[str, Path],
) -> None:
    metadata_leaf: JSONDocument = {"label": "original metadata"}
    failure_leaf: JSONDocument = {"label": "original failure"}
    source_metadata: JSONDocument = {"nested": [metadata_leaf]}
    source_failure: JSONDocument = {"nested": [failure_leaf]}
    batch = AnnotationBatch(
        batch_id="detached-provenance",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref="delegation:detached-provenance",
        source_result_ref="result-set:detached-provenance",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref="block:detached-provenance:0",
        total_count=1,
        valid_count=0,
        invalid_count=1,
        abstained_count=0,
        validation_failures=(source_failure,),
        metadata=source_metadata,
        created_at_ms=1,
    )
    original_document = batch.provenance_document()
    original_bytes = batch.canonical_provenance_bytes()

    metadata_leaf["label"] = "mutated source metadata"
    failure_leaf["label"] = "mutated source failure"
    assert batch.provenance_document() == original_document
    assert batch.canonical_provenance_bytes() == original_bytes

    batch_metadata_nested = batch.metadata["nested"]
    assert isinstance(batch_metadata_nested, list)
    batch_metadata_leaf = batch_metadata_nested[0]
    assert isinstance(batch_metadata_leaf, dict)
    batch_metadata_leaf["label"] = "mutated exposed metadata"
    batch_failure_nested = batch.validation_failures[0]["nested"]
    assert isinstance(batch_failure_nested, list)
    batch_failure_leaf = batch_failure_nested[0]
    assert isinstance(batch_failure_leaf, dict)
    batch_failure_leaf["label"] = "mutated exposed failure"
    assert batch.provenance_document() == original_document
    assert batch.canonical_provenance_bytes() == original_bytes

    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        persisted = archive.save_annotation_batch(batch)
        assert persisted.provenance_document() == original_document

    with ArchiveStore.open_existing(archive_root) as reopened:
        cold = reopened.get_annotation_batch(batch.batch_id)
        assert cold is not None
        assert cold.provenance_document() == original_document
        cold_metadata_nested = cold.metadata["nested"]
        assert isinstance(cold_metadata_nested, list)
        cold_metadata_leaf = cold_metadata_nested[0]
        assert isinstance(cold_metadata_leaf, dict)
        cold_metadata_leaf["label"] = "mutated cold read"
        cold_failure_nested = cold.validation_failures[0]["nested"]
        assert isinstance(cold_failure_nested, list)
        cold_failure_leaf = cold_failure_nested[0]
        assert isinstance(cold_failure_leaf, dict)
        cold_failure_leaf["label"] = "mutated cold failure read"

    with ArchiveStore.open_existing(archive_root) as reopened_again:
        reread = reopened_again.get_annotation_batch(batch.batch_id)
        assert reread is not None
        assert reread.provenance_document() == original_document


def test_batch_readback_rejects_duplicate_refs_and_non_finite_nested_json(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    duplicate_batch, assertion_id = _single_row_batch(batch_id="corrupt-duplicates")
    non_finite_batch = AnnotationBatch(
        batch_id="corrupt-non-finite",
        schema_id=DELEGATION_DISCOURSE_SCHEMA.schema_id,
        schema_version=DELEGATION_DISCOURSE_SCHEMA.version,
        target_ref="delegation:corrupt-non-finite",
        source_result_ref="result-set:corrupt-non-finite",
        actor_ref="agent:labeler",
        model_ref="agent:model",
        prompt_ref="block:corrupt-non-finite:0",
        total_count=0,
        valid_count=0,
        invalid_count=0,
        abstained_count=0,
        created_at_ms=1,
    )
    with ArchiveStore(archive_root) as archive:
        archive.save_annotation_batch(duplicate_batch)
        archive.save_annotation_batch(non_finite_batch)

    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.execute(
            """
            UPDATE annotation_batches
            SET total_count = 2,
                valid_count = 2,
                assertion_refs_json = ?
            WHERE batch_id = 'corrupt-duplicates'
            """,
            (f'["assertion:{assertion_id}","assertion:{assertion_id}"]',),
        )
        conn.execute(
            "UPDATE annotation_batches SET metadata_json = ? WHERE batch_id = 'corrupt-non-finite'",
            ('{"score":1e999}',),
        )
        conn.commit()

    with ArchiveStore.open_existing(archive_root) as reopened:
        with pytest.raises(AnnotationBatchError, match="unique after normalization"):
            reopened.get_annotation_batch("corrupt-duplicates")
        with pytest.raises(AnnotationBatchError, match="finite"):
            reopened.get_annotation_batch("corrupt-non-finite")


def test_batch_scoped_assertion_replay_is_insert_once_before_and_after_judgment(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    batch, assertion_id = _single_row_batch(batch_id="insert-once")
    with ArchiveStore(archive_root) as archive:
        archive.save_annotation_batch(batch)

    registry = AnnotationSchemaRegistry()
    registry.register(DELEGATION_DISCOURSE_SCHEMA)
    baseline_value = _delegation_value()
    baseline_evidence = ("session:codex-session:delegation-demo",)

    with sqlite3.connect(archive_root / "user.db") as conn:

        def write(
            *,
            value: dict[str, object] | None = None,
            evidence_refs: tuple[str, ...] = baseline_evidence,
            body_text: str = "baseline body",
            confidence: float = 0.75,
            now_ms: int,
        ) -> ArchiveAssertionEnvelope:
            return upsert_annotation_assertion(
                conn,
                schema=DELEGATION_DISCOURSE_SCHEMA,
                registry=registry,
                target_ref=batch.target_ref,
                value=baseline_value if value is None else value,
                row_key="row-1",
                evidence_refs=evidence_refs,
                author_ref=batch.actor_ref,
                batch_ref=batch.batch_ref,
                body_text=body_text,
                confidence=confidence,
                now_ms=now_ms,
            )

        inserted = write(now_ms=2_000)
        replayed = write(now_ms=2_001)
        assert replayed == inserted
        assert read_assertion_envelope(conn, assertion_id) == inserted

        drift_calls: tuple[Callable[[int], ArchiveAssertionEnvelope], ...] = (
            lambda now: write(value={**baseline_value, "confidence": 0.8}, now_ms=now),
            lambda now: write(evidence_refs=("session:codex-session:different",), now_ms=now),
            lambda now: write(body_text="changed body", now_ms=now),
            lambda now: write(confidence=0.8, now_ms=now),
        )
        for offset, drift_call in enumerate(drift_calls, start=1):
            with pytest.raises(AnnotationValidationError, match="immutable input drift"):
                drift_call(2_100 + offset)
            assert read_assertion_envelope(conn, assertion_id) == inserted

        judgment = judge_assertion_candidate(
            conn,
            candidate_ref=f"assertion:{assertion_id}",
            decision="reject",
            reason="operator rejection",
            now_ms=3_000,
        )
        judged = judgment.candidate
        assert judged.status is AssertionStatus.REJECTED
        assert write(now_ms=3_001) == judged
        assert read_assertion_envelope(conn, assertion_id) == judged

        for offset, drift_call in enumerate(drift_calls, start=1):
            with pytest.raises(AnnotationValidationError, match="immutable input drift"):
                drift_call(3_100 + offset)
            assert read_assertion_envelope(conn, assertion_id) == judged
