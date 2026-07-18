"""Seed annotation ontologies and governed archive-local bootstrap (polylogue-dve1)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.annotations.importer import AnnotationBatchImportRequest, import_annotation_batch
from polylogue.annotations.join import AnnotationStructuralJoinRequest, join_typed_annotations
from polylogue.annotations.schema import (
    SEED_ACTIVITY_SCHEMA,
    SEED_ANNOTATION_SCHEMAS,
    SEED_GOAL_EVENT_SCHEMA,
    SEED_KNOWLEDGE_ARTIFACT_SCHEMA,
    SEED_OUTCOME_EVIDENCE_SCHEMA,
    SEED_REUSABILITY_SCHEMA,
    AnnotationField,
    AnnotationSchema,
    AnnotationSchemaError,
    list_annotation_schemas,
    validate_annotation_value,
)
from polylogue.annotations.write import (
    OntologyCandidateGovernance,
    OntologyCandidateNomination,
    OntologyViewProposal,
    assertion_id_for_schema_annotation,
    govern_ontology_candidate,
    nominate_ontology_candidate,
)
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.json import require_json_document
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_annotations import (
    persist_annotation_schema,
    read_durable_annotation_schema,
)
from polylogue.storage.sqlite.archive_tiers.user_write import (
    judge_assertion_candidate,
    read_assertion_envelope,
    upsert_assertion,
)


def _draft_topic_schema(*, schema_id: str = "archive.topic", version: int = 1) -> AnnotationSchema:
    return AnnotationSchema(
        schema_id=schema_id,
        version=version,
        title="Archive topic",
        description="Archive-local topic proposed from multiple declared views.",
        fields=(
            AnnotationField(name="topic", value_type="enum", enum_values=("procurement", "operations")),
            AnnotationField(name="confidence", value_type="number", minimum=0, maximum=1),
            AnnotationField(name="abstain", value_type="boolean", required=False),
        ),
        target_ref_kinds=("session",),
        abstain_field="abstain",
        evidence_policy="required",
        status="draft",
    )


def _nomination(schema: AnnotationSchema, *, source_tag_ref: str) -> OntologyCandidateNomination:
    return OntologyCandidateNomination(
        candidate_schema=schema,
        target_ref="workspace:archive-local",
        affinity=0.97,
        confidence=0.81,
        classifier_ref="ranker:archive-topic-v1",
        classifier_definition={"kind": "nearest-centroid", "version": 1},
        version_crosswalk={"from": "archive.topic@v0", "labels": {"buying": "procurement"}},
        frame_ref="result-set:ontology-bootstrap-frame",
        epoch_ref="analysis:archive-epoch-17",
        privacy_policy_ref="assertion:privacy-policy-v1",
        author_ref="agent:ontology-bootstrap",
        view_proposals=(
            OntologyViewProposal(
                view="content",
                label="procurement",
                confidence=0.91,
                evidence_refs=("session:content-exemplar",),
            ),
            OntologyViewProposal(
                view="action_pattern",
                label="operations",
                confidence=0.83,
                evidence_refs=("session:action-exemplar",),
            ),
        ),
        source_tag_refs=(source_tag_ref,),
        residue_refs=("session:unclassified-residue",),
        rare_sample_refs=("session:rare-category",),
        privacy_excluded_refs=("session:privacy-excluded",),
        evidence_refs=("session:bootstrap-evidence",),
    )


def _seed_tag(conn: sqlite3.Connection) -> str:
    tag = upsert_assertion(
        conn,
        assertion_id="assertion-source-tag",
        target_ref="session:tagged-session",
        kind=AssertionKind.TAG,
        key="procurement",
        value={"tag": "procurement", "affinity": 0.97},
        author_ref="user:local",
        author_kind="user",
        status=AssertionStatus.ACTIVE,
        context_policy={"inject": False},
        require_promotion=False,
        now_ms=10,
    )
    conn.commit()
    return f"assertion:{tag.assertion_id}"


def test_seed_catalog_is_registered_and_replayed_without_user_schema_bump(tmp_path: Path) -> None:
    qualified_ids = {schema.qualified_id for schema in list_annotation_schemas()}
    assert {schema.qualified_id for schema in SEED_ANNOTATION_SCHEMAS} <= qualified_ids
    assert len({schema.definition_fingerprint for schema in SEED_ANNOTATION_SCHEMAS}) == 5
    for schema in SEED_ANNOTATION_SCHEMAS:
        assert schema.status == "active"
        assert schema.evidence_policy == "required"
        assert {"confidence", "abstain", "rationale"} <= {field.name for field in schema.fields}

    activity = next(field for field in SEED_ACTIVITY_SCHEMA.fields if field.name == "activity")
    assert activity.enum_values == (
        "debugging",
        "design",
        "implementation",
        "research",
        "writing",
        "ideation",
        "ops",
        "procurement",
    )
    assert SEED_ACTIVITY_SCHEMA.target_ref_kinds == ("session", "phase", "message", "block")

    artifact_type = next(field for field in SEED_KNOWLEDGE_ARTIFACT_SCHEMA.fields if field.name == "artifact_type")
    artifact_authority = next(field for field in SEED_KNOWLEDGE_ARTIFACT_SCHEMA.fields if field.name == "authority")
    assert artifact_type.enum_values == (
        "decision",
        "lesson",
        "preference",
        "fact_candidate",
        "fact_established",
        "commitment",
    )
    assert artifact_authority.enum_values == (
        "agent_candidate",
        "actor_declared",
        "structural",
        "rule",
        "operator_judged",
    )

    reuse_purpose = next(field for field in SEED_REUSABILITY_SCHEMA.fields if field.name == "purpose")
    reuse_authority = next(field for field in SEED_REUSABILITY_SCHEMA.fields if field.name == "authority")
    assert reuse_purpose.enum_values == ("snippet", "recipe", "demo")
    assert reuse_authority.enum_values == ("agent_candidate", "operator_judged")

    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        rows = {
            f"{schema_id}@v{version}"
            for schema_id, version in conn.execute(
                "SELECT schema_id, schema_version FROM annotation_schemas"
            ).fetchall()
        }
        assert {schema.qualified_id for schema in SEED_ANNOTATION_SCHEMAS} <= rows

        activity_v2 = replace(SEED_ACTIVITY_SCHEMA, version=2, title="Activity v2")
        persist_annotation_schema(conn, activity_v2, registered_at_ms=1)
        with pytest.raises(AnnotationSchemaError, match="incompatible durable definition"):
            persist_annotation_schema(
                conn,
                replace(SEED_ACTIVITY_SCHEMA, title="Drifted activity"),
                registered_at_ms=2,
            )
        conn.execute(
            "DELETE FROM annotation_schemas WHERE schema_id = ? AND schema_version = ?",
            (SEED_ACTIVITY_SCHEMA.schema_id, SEED_ACTIVITY_SCHEMA.version),
        )
        conn.commit()

    # Same user_version: data-only bootstrap replays the missing immutable row.
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM annotation_schemas WHERE schema_id = ? AND schema_version = ?",
                (SEED_ACTIVITY_SCHEMA.schema_id, SEED_ACTIVITY_SCHEMA.version),
            ).fetchone()[0]
            == 1
        )
        assert [
            int(row[0])
            for row in conn.execute(
                "SELECT schema_version FROM annotation_schemas WHERE schema_id = ? ORDER BY schema_version",
                (SEED_ACTIVITY_SCHEMA.schema_id,),
            ).fetchall()
        ] == [1, 2]


def test_goal_events_exclude_derived_inactivity_and_remain_distinct_from_outcomes() -> None:
    goal_event = next(field for field in SEED_GOAL_EVENT_SCHEMA.fields if field.name == "event_type")
    assert goal_event.enum_values == (
        "opened",
        "blocked",
        "resumed",
        "declared_resolved",
        "superseded",
        "explicitly_abandoned",
    )
    assert "abandoned" not in goal_event.enum_values
    assert "unresolved_inactive" not in goal_event.enum_values
    errors = validate_annotation_value(
        SEED_GOAL_EVENT_SCHEMA,
        {
            "event_type": "unresolved_inactive",
            "goal_ref": "goal:1",
            "declared_by_ref": "user:local",
            "declaration_authority": "actor_declared",
            "confidence": 0.9,
        },
    )
    assert any("event_type" in error for error in errors)

    assert SEED_GOAL_EVENT_SCHEMA.schema_id != SEED_OUTCOME_EVIDENCE_SCHEMA.schema_id
    outcome_type = next(field for field in SEED_OUTCOME_EVIDENCE_SCHEMA.fields if field.name == "outcome_type")
    authority = next(field for field in SEED_OUTCOME_EVIDENCE_SCHEMA.fields if field.name == "authority")
    temporal_mode = next(field for field in SEED_OUTCOME_EVIDENCE_SCHEMA.fields if field.name == "temporal_mode")
    assert "test_passed" in outcome_type.enum_values
    assert set(goal_event.enum_values).isdisjoint(outcome_type.enum_values)
    assert authority.enum_values == ("structural", "rule", "judged")
    assert temporal_mode.enum_values == ("observed", "historical_backfill")

    shared_identity = {
        "target_ref": "session:historical",
        "author_ref": "agent:backfill",
        "row_key": "event-1",
    }
    assert assertion_id_for_schema_annotation(
        schema_qualified_id=SEED_GOAL_EVENT_SCHEMA.qualified_id,
        **shared_identity,
    ) != assertion_id_for_schema_annotation(
        schema_qualified_id=SEED_OUTCOME_EVIDENCE_SCHEMA.qualified_id,
        **shared_identity,
    )


def test_rejected_nomination_preserves_tag_and_full_bootstrap_evidence(tmp_path: Path) -> None:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        source_tag_ref = _seed_tag(conn)
        nomination = _nomination(_draft_topic_schema(), source_tag_ref=source_tag_ref)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        candidate = nominate_ontology_candidate(conn, nomination, now_ms=20)
        conn.set_trace_callback(None)

        assert next(statement for statement in statements if statement.strip()).upper().startswith("BEGIN IMMEDIATE")
        assert candidate.kind == AssertionKind.ONTOLOGY_CANDIDATE
        assert candidate.status == AssertionStatus.CANDIDATE
        assert candidate.context_policy == {"inject": False, "promotion_required": True}
        assert isinstance(candidate.value, dict)
        assert candidate.value["source_tag_refs"] == [source_tag_ref]
        assert candidate.value["affinity"] == 0.97
        assert candidate.value["confidence"] == 0.81
        assert candidate.value["version_crosswalk"] == {
            "from": "archive.topic@v0",
            "labels": {"buying": "procurement"},
        }
        assert candidate.confidence == 0.81
        assert candidate.value["cross_view_state"] == "disagreement"
        assert candidate.value["residue_refs"] == ["session:unclassified-residue"]
        assert candidate.value["rare_sample_refs"] == ["session:rare-category"]
        assert candidate.value["epoch_ref"] == "analysis:archive-epoch-17"
        assert candidate.value["privacy_excluded_refs"] == ["session:privacy-excluded"]
        assert read_durable_annotation_schema(conn, "archive.topic", 1) is None
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'annotation'").fetchone()[0] == 0

        result = govern_ontology_candidate(
            conn,
            OntologyCandidateGovernance(
                candidate_ref=f"assertion:{candidate.assertion_id}",
                decision="reject",
                actor_ref="user:operator",
                reason="category boundary is not stable",
            ),
            now_ms=30,
        )
        assert result.judgment.candidate.status == AssertionStatus.REJECTED
        assert result.governance_receipt.kind == AssertionKind.ONTOLOGY_GOVERNANCE
        receipt_value = require_json_document(result.governance_receipt.value, context="governance receipt value")
        candidate_value = require_json_document(candidate.value, context="candidate value")
        assert receipt_value["decision"] == "reject"
        assert receipt_value["affinity"] == 0.97
        assert receipt_value["confidence"] == 0.81
        assert receipt_value["version_crosswalk"] == candidate_value["version_crosswalk"]
        assert read_durable_annotation_schema(conn, "archive.topic", 1) is None

        tag = read_assertion_envelope(conn, source_tag_ref.removeprefix("assertion:"))
        assert tag is not None
        assert tag.kind == AssertionKind.TAG
        assert tag.status == AssertionStatus.ACTIVE
        assert tag.value == {"tag": "procurement", "affinity": 0.97}

        # A later autonomous retry cannot resurrect the terminal rejection.
        retried = nominate_ontology_candidate(conn, nomination, now_ms=40)
        assert retried.status == AssertionStatus.REJECTED
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("decision", "draft_schema_id", "active_schema_ids"),
    (
        ("rename", "archive.topic.rename", ("archive.workflow",)),
        (
            "split",
            "archive.topic.split",
            ("archive.topic.procurement", "archive.topic.operations"),
        ),
    ),
)
def test_rename_and_split_register_only_operator_output_schemas(
    tmp_path: Path,
    decision: str,
    draft_schema_id: str,
    active_schema_ids: tuple[str, ...],
) -> None:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        source_tag_ref = _seed_tag(conn)
        draft = _draft_topic_schema(schema_id=draft_schema_id)
        candidate = nominate_ontology_candidate(
            conn,
            _nomination(draft, source_tag_ref=source_tag_ref),
            now_ms=50,
        )
        active_schemas = tuple(
            replace(
                _draft_topic_schema(schema_id=schema_id),
                status="active",
                title=f"Operator-governed {schema_id}",
            )
            for schema_id in active_schema_ids
        )
        request = OntologyCandidateGovernance(
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision=decision,  # type: ignore[arg-type]
            actor_ref="user:operator",
            reason=f"operator chose to {decision} the proposed category",
            active_schemas=active_schemas,
        )
        result = govern_ontology_candidate(conn, request, now_ms=60)

        assert result.judgment.candidate.status == AssertionStatus.SUPERSEDED
        assert result.judgment.resulting_assertion is not None
        resulting_value = require_json_document(
            result.judgment.resulting_assertion.value, context="resulting assertion value"
        )
        assert resulting_value["governance_decision"] == decision
        assert read_durable_annotation_schema(conn, draft.schema_id, draft.version) is None
        assert {durable.schema.qualified_id for durable in result.active_schemas} == {
            schema.qualified_id for schema in active_schemas
        }
        assert all(
            read_durable_annotation_schema(conn, schema.schema_id, schema.version) is not None
            for schema in active_schemas
        )
        receipt_value = require_json_document(result.governance_receipt.value, context="governance receipt value")
        assert receipt_value["decision"] == decision
        assert receipt_value["annotation_batch_required"] is True

        retry = govern_ontology_candidate(conn, request, now_ms=70)
        assert retry.judgment.outcome == "idempotent"
        assert retry.governance_receipt.assertion_id == result.governance_receipt.assertion_id

        tag = read_assertion_envelope(conn, source_tag_ref.removeprefix("assertion:"))
        assert tag is not None
        assert tag.status == AssertionStatus.ACTIVE
    finally:
        conn.close()


def test_governance_rolls_back_judgment_when_schema_registration_conflicts(tmp_path: Path) -> None:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        source_tag_ref = _seed_tag(conn)
        draft = _draft_topic_schema(schema_id="archive.atomic-topic")
        candidate = nominate_ontology_candidate(
            conn,
            _nomination(draft, source_tag_ref=source_tag_ref),
            now_ms=80,
        )
        persist_annotation_schema(
            conn,
            replace(draft, status="active", title="Conflicting durable definition"),
            registered_at_ms=85,
        )
        conn.commit()

        with pytest.raises(AnnotationSchemaError, match="incompatible durable definition"):
            govern_ontology_candidate(
                conn,
                OntologyCandidateGovernance(
                    candidate_ref=f"assertion:{candidate.assertion_id}",
                    decision="accept",
                    actor_ref="user:operator",
                    active_schemas=(replace(draft, status="active"),),
                ),
                now_ms=90,
            )

        unchanged = read_assertion_envelope(conn, candidate.assertion_id)
        assert unchanged is not None
        assert unchanged.status == AssertionStatus.CANDIDATE
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind IN ('judgment', 'ontology_governance')"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


class _ArchiveFacade:
    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root

    async def resolve_ref(self, ref: str) -> SimpleNamespace:
        return SimpleNamespace(
            resolved=True,
            caveats=(),
            object_refs=(ref,),
            payload_kind="session",
            payload={},
        )

    async def get_session_summary(self, session_id: str) -> None:
        return None


async def _resolved(_ref: str) -> bool:
    return True


def test_accept_then_durable_batch_import_requires_label_judgment_for_active_query(tmp_path: Path) -> None:
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        source_tag_ref = _seed_tag(conn)
        draft = _draft_topic_schema()
        candidate = nominate_ontology_candidate(conn, _nomination(draft, source_tag_ref=source_tag_ref), now_ms=100)
        active = replace(draft, status="active")
        governance_request = OntologyCandidateGovernance(
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="accept",
            actor_ref="user:operator",
            reason="cross-view evidence is useful despite a documented boundary disagreement",
            active_schemas=(active,),
        )
        governance = govern_ontology_candidate(conn, governance_request, now_ms=110)
        assert governance.active_schemas[0].schema == active
        governance_receipt_value = require_json_document(
            governance.governance_receipt.value, context="governance receipt value"
        )
        assert governance_receipt_value["annotation_batch_required"] is True

        governance_retry = govern_ontology_candidate(conn, governance_request, now_ms=115)
        assert governance_retry.judgment.outcome == "idempotent"
        assert governance_retry.governance_receipt.assertion_id == governance.governance_receipt.assertion_id

        # An identical autonomous retry returns the terminal accepted lifecycle row.
        retried = nominate_ontology_candidate(conn, _nomination(draft, source_tag_ref=source_tag_ref), now_ms=116)
        assert retried.status == AssertionStatus.ACCEPTED
        assert read_durable_annotation_schema(conn, active.schema_id, active.version) is not None
    finally:
        conn.close()

    facade = _ArchiveFacade(tmp_path)
    active_request = AnnotationStructuralJoinRequest(
        schema_id=active.schema_id,
        schema_version=active.version,
        statuses=(AssertionStatus.ACTIVE,),
    )
    assert asyncio.run(join_typed_annotations(facade, active_request)).joined_count == 0

    batch_request = AnnotationBatchImportRequest(
        jsonl=json.dumps(
            {
                "row_key": "topic-1",
                "value": {"topic": "procurement", "confidence": 0.88},
                "evidence_refs": ["session:topic-session"],
            }
        ),
        batch_id="archive-topic-backfill-1",
        schema_id=active.schema_id,
        schema_version=active.version,
        target_ref="session:topic-session",
        source_result_ref="result-set:ontology-bootstrap-frame",
        actor_ref="agent:ontology-labeler",
        model_ref="agent:model-v1",
        prompt_ref="block:prompt-session:0",
        metadata={"ontology_governance_ref": f"assertion:{governance.governance_receipt.assertion_id}"},
        created_at_ms=120,
    )
    imported = asyncio.run(
        import_annotation_batch(
            facade,  # type: ignore[arg-type]
            batch_request,
            resolve_ref=_resolved,
        )
    )
    assert imported.valid_count == 1
    assertion_ref = next(row.assertion_ref for row in imported.rows if row.status == "imported")
    assert assertion_ref is not None

    # The governed schema exists, but an agent batch is still candidate-only.
    assert asyncio.run(join_typed_annotations(facade, active_request)).joined_count == 0
    candidate_request = AnnotationStructuralJoinRequest(
        schema_id=active.schema_id,
        schema_version=active.version,
        statuses=(AssertionStatus.CANDIDATE,),
    )
    assert asyncio.run(join_typed_annotations(facade, candidate_request)).joined_count == 1

    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        judged = judge_assertion_candidate(
            conn,
            candidate_ref=assertion_ref,
            decision="accept",
            reason="operator accepts the batch label",
            actor_ref="user:operator",
            inject=False,
            now_ms=130,
        )
        conn.commit()
        assert judged.resulting_assertion is not None
        assert judged.resulting_assertion.status == AssertionStatus.ACTIVE
    finally:
        conn.close()

    # Exact agent replay uses the same BEGIN IMMEDIATE path and preserves the operator judgment.
    replayed = asyncio.run(
        import_annotation_batch(
            facade,  # type: ignore[arg-type]
            batch_request,
            resolve_ref=_resolved,
        )
    )
    assert replayed.valid_count == 1
    assert asyncio.run(join_typed_annotations(facade, candidate_request)).joined_count == 0

    active_join = asyncio.run(join_typed_annotations(facade, active_request))
    assert active_join.joined_count == 1
    assert active_join.rows[0].batch_ref == "annotation-batch:archive-topic-backfill-1"
    assert active_join.rows[0].adjudicator_ref == "user:operator"
