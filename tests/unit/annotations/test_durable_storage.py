"""Durable annotation schema and batch provenance contracts (polylogue-rxdo.7.1)."""

from __future__ import annotations

import sqlite3
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
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


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
