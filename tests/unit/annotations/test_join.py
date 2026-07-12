"""Production-route tests for structural annotation joins without fanout."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.annotations.join import AnnotationStructuralJoinRequest, join_typed_annotations
from polylogue.annotations.schema import (
    DELEGATION_DISCOURSE_SCHEMA,
    AnnotationField,
    AnnotationSchema,
    AnnotationSchemaRegistry,
)
from polylogue.annotations.write import upsert_annotation_assertion
from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, AssertionStatus, BlockType, BranchType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_annotations import persist_annotation_schema
from polylogue.storage.sqlite.archive_tiers.user_write import judge_assertion_candidate, upsert_assertion


def _delegation_value(*, mode: str = "imperative") -> dict[str, object]:
    return {
        "directive_mode": mode,
        "prohibitions": "explicit",
        "autonomy": "bounded",
        "output_contract": "structured",
        "scope_control": "owned_paths",
        "verification_demand": "focused_tests",
        "checkpoint_escalation": "escalation",
        "relational_frame": "collaborative",
        "rationale_visibility": "explicit",
        "applicable": True,
        "confidence": 0.9,
        "abstain": False,
    }


def _seed_delegation(archive_root: Path) -> tuple[str, str]:
    with ArchiveStore(archive_root) as archive:
        parent = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="join-parent",
                title="Join parent",
                created_at="2026-07-01T12:00:00Z",
                git_repository_url="https://github.com/Sinity/polylogue",
                messages=[
                    ParsedMessage(
                        provider_message_id="dispatch",
                        role=Role.ASSISTANT,
                        model_name="claude-opus-4-8",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Task",
                                tool_id="join-task",
                                tool_input={"prompt": "review the join", "subagent_type": "general-purpose"},
                            )
                        ],
                    ),
                    ParsedMessage(
                        provider_message_id="result",
                        role=Role.USER,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="join-task",
                                text="done",
                                is_error=False,
                                exit_code=0,
                            )
                        ],
                    ),
                ],
            )
        )
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="join-child",
                title="Join child",
                messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="working")],
                parent_session_provider_id="join-parent",
                branch_type=BranchType.SUBAGENT,
            )
        )
    block_id = f"{parent}:dispatch:0"
    return f"delegation:{block_id}", f"block:{block_id}"


def _seed_ambiguous_delegation(archive_root: Path) -> tuple[str, str]:
    with ArchiveStore(archive_root) as archive:
        parent = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="join-ambiguous-parent",
                title="Ambiguous join parent",
                created_at="2026-07-01T12:00:00Z",
                git_repository_url="https://github.com/Sinity/polylogue",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"dispatch-{suffix}",
                        role=Role.ASSISTANT,
                        model_name="claude-opus-4-8",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Task",
                                tool_id=f"ambiguous-{suffix}",
                                tool_input={},
                            )
                        ],
                    )
                    for suffix in ("a", "b")
                ],
            )
        )
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="join-ambiguous-child",
                title="Ambiguous join child",
                messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="working")],
                parent_session_provider_id="join-ambiguous-parent",
                branch_type=BranchType.SUBAGENT,
            )
        )
    block_id = f"{parent}:dispatch-a:0"
    return f"delegation:{block_id}", f"block:{block_id}"


def _accept(
    conn: sqlite3.Connection,
    *,
    registry: AnnotationSchemaRegistry,
    schema: AnnotationSchema,
    target_ref: str,
    evidence_ref: str,
    row_key: str,
    author_ref: str,
    value: dict[str, object],
    now_ms: int,
) -> None:
    confidence = value.get("confidence", value.get("score", 0.8))
    assert isinstance(confidence, int | float) and not isinstance(confidence, bool)
    candidate = upsert_annotation_assertion(
        conn,
        schema=schema,
        registry=registry,
        target_ref=target_ref,
        value=value,
        row_key=row_key,
        evidence_refs=(evidence_ref,),
        author_ref=author_ref,
        confidence=float(confidence),
        now_ms=now_ms,
    )
    judge_assertion_candidate(
        conn,
        candidate_ref=f"assertion:{candidate.assertion_id}",
        decision="accept",
        now_ms=now_ms + 1,
    )


@pytest.mark.asyncio
async def test_delegation_join_groups_active_labels_and_reports_nonjoins(
    workspace_env: dict[str, Path],
) -> None:
    """Exact target enrichment retains labels and exposes every fanout risk.

    Anti-vacuity: this exercises the real delegation resolver/view, durable
    schema, assertion judgment lifecycle, session repository metadata, and
    product join. Removing exact target resolution or status/schema/value
    checks changes the row/group/error counts below.
    """
    archive_root = workspace_env["archive_root"]
    target_ref, evidence_ref = _seed_delegation(archive_root)
    ambiguous_ref, ambiguous_evidence = _seed_ambiguous_delegation(archive_root)
    registry = AnnotationSchemaRegistry()
    registry.register(DELEGATION_DISCOURSE_SCHEMA)
    with sqlite3.connect(archive_root / "user.db") as conn:
        _accept(
            conn,
            registry=registry,
            schema=DELEGATION_DISCOURSE_SCHEMA,
            target_ref=target_ref,
            evidence_ref=evidence_ref,
            row_key="label-a",
            author_ref="agent:labeler-a",
            value=_delegation_value(mode="imperative"),
            now_ms=1_000,
        )
        _accept(
            conn,
            registry=registry,
            schema=DELEGATION_DISCOURSE_SCHEMA,
            target_ref=target_ref,
            evidence_ref=evidence_ref,
            row_key="label-b",
            author_ref="agent:labeler-b",
            value=_delegation_value(mode="collaborative"),
            now_ms=2_000,
        )
        upsert_annotation_assertion(
            conn,
            schema=DELEGATION_DISCOURSE_SCHEMA,
            registry=registry,
            target_ref=target_ref,
            value=_delegation_value(),
            row_key="candidate-excluded",
            evidence_refs=(evidence_ref,),
            author_ref="agent:pending",
            confidence=0.9,
            now_ms=3_000,
        )
        _accept(
            conn,
            registry=registry,
            schema=DELEGATION_DISCOURSE_SCHEMA,
            target_ref="delegation:missing-target",
            evidence_ref=evidence_ref,
            row_key="missing",
            author_ref="agent:missing",
            value=_delegation_value(),
            now_ms=4_000,
        )
        _accept(
            conn,
            registry=registry,
            schema=DELEGATION_DISCOURSE_SCHEMA,
            target_ref=ambiguous_ref,
            evidence_ref=ambiguous_evidence,
            row_key="ambiguous",
            author_ref="agent:ambiguous",
            value=_delegation_value(),
            now_ms=4_500,
        )
        upsert_assertion(
            conn,
            assertion_id="invalid-active-label",
            target_ref=target_ref,
            kind=AssertionKind.ANNOTATION,
            key="invalid",
            value={"_schema": "delegation.discourse@v1", "confidence": "high"},
            author_kind="user",
            evidence_refs=(evidence_ref,),
            status="active",
            now_ms=5_000,
        )
        upsert_assertion(
            conn,
            assertion_id="drifted-active-label",
            target_ref=target_ref,
            kind=AssertionKind.ANNOTATION,
            key="drift",
            value={"_schema": "delegation.discourse@v2", **_delegation_value()},
            author_kind="user",
            evidence_refs=(evidence_ref,),
            status="active",
            now_ms=6_000,
        )
        conn.commit()

    request = AnnotationStructuralJoinRequest(
        schema_id="delegation.discourse",
        schema_version=1,
        statuses=(AssertionStatus.ACTIVE,),
        target_kind="delegation",
        group_by=("repo", "model", "time"),
    )
    async with Polylogue(archive_root=archive_root) as poly:
        result = await join_typed_annotations(poly, request)
        accepted = await join_typed_annotations(
            poly,
            request.model_copy(update={"statuses": (AssertionStatus.ACCEPTED,)}),
        )

    assert result.selected_annotation_count == 5
    assert result.matched_annotation_count == 5
    assert result.selection_truncated is False
    assert result.joined_count == 3
    assert result.missing_target_count == 1
    assert result.ambiguous_target_count == 1
    assert result.invalid_value_count == 1
    assert result.schema_drift_count == 1
    assert result.multi_label_target_count == 1
    assert result.duplicate_label_count == 1
    assert len({row.assertion_ref for row in result.rows}) == 3
    assert all(len(row.supersedes) == 1 for row in result.rows)
    assert {row.labeler_ref for row in result.rows} == {
        "agent:labeler-a",
        "agent:labeler-b",
        "agent:ambiguous",
    }
    assert {row.adjudicator_ref for row in result.rows} == {"user:local"}
    assert {row.labeler_ref for row in accepted.rows} == {
        "agent:labeler-a",
        "agent:labeler-b",
        "agent:ambiguous",
    }
    assert {row.adjudicator_ref for row in accepted.rows} == {"user:local"}
    assert {row.judgment_decision for row in accepted.rows} == {"accept"}
    assert all(row.judgment_ref is not None for row in accepted.rows)
    assert {row.value["directive_mode"] for row in result.rows} == {"imperative", "collaborative"}
    [group] = result.groups
    assert group.label_count == 3
    assert group.distinct_target_count == 2
    assert group.dimensions == {
        "repo": "https://github.com/Sinity/polylogue",
        "model": "claude-opus-4-8",
        "time": "2026-07-01",
    }
    assert {diagnostic.code for diagnostic in result.diagnostics} == {
        "missing_target",
        "ambiguous_target",
        "schema_drift",
        "invalid_value",
    }


@pytest.mark.asyncio
async def test_join_is_generic_for_session_targets(workspace_env: dict[str, Path]) -> None:
    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="generic-session",
                title="Generic target",
                created_at="2026-07-02T00:00:00Z",
                git_repository_url="https://github.com/Sinity/sinex",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
            )
        )
    schema = AnnotationSchema(
        schema_id="test.session-quality",
        version=1,
        title="Session quality",
        fields=(AnnotationField(name="score", value_type="number", minimum=0, maximum=1),),
        target_ref_kinds=("session", "delegation"),
        evidence_policy="required",
        status="active",
    )
    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    target_ref = f"session:{session_id}"
    with sqlite3.connect(archive_root / "user.db") as conn:
        persist_annotation_schema(conn, schema, registered_at_ms=1)
        _accept(
            conn,
            registry=registry,
            schema=schema,
            target_ref=target_ref,
            evidence_ref=session_id,
            row_key="session-label",
            author_ref="agent:session-labeler",
            value={"score": 0.8},
            now_ms=10,
        )
        _accept(
            conn,
            registry=registry,
            schema=schema,
            target_ref=target_ref,
            evidence_ref=session_id,
            row_key="session-label-2",
            author_ref="agent:session-labeler-2",
            value={"score": 0.7},
            now_ms=20,
        )
        for index in range(3):
            upsert_assertion(
                conn,
                assertion_id=f"irrelevant-schema-{index}",
                target_ref=target_ref,
                kind=AssertionKind.ANNOTATION,
                value={"_schema": "other.schema@v1", "score": 1},
                author_kind="user",
                status="active",
                now_ms=100 + index,
            )
        upsert_assertion(
            conn,
            assertion_id="newer-schema-version",
            target_ref=target_ref,
            kind=AssertionKind.ANNOTATION,
            value={"_schema": "test.session-quality@v2", "score": 1},
            author_kind="user",
            status="active",
            now_ms=200,
        )
        upsert_assertion(
            conn,
            assertion_id="wrong-target-kind",
            target_ref="delegation:other-kind",
            kind=AssertionKind.ANNOTATION,
            value={"_schema": "test.session-quality@v1", "score": 1},
            author_kind="user",
            status="active",
            now_ms=300,
        )
        conn.commit()

    async with Polylogue(archive_root=archive_root) as poly:
        result = await poly.join_typed_annotations(
            schema_id=schema.schema_id,
            schema_version=1,
            statuses=(AssertionStatus.ACTIVE,),
            target_kind="session",
            group_by=("repo", "origin"),
            limit=1,
        )

    assert result.joined_count == 1
    assert result.matched_annotation_count == 2
    assert result.selection_truncated is True
    assert result.next_offset == 1
    assert result.schema_drift_count == 1
    assert result.rows[0].structural["target_kind"] == "session"
    assert result.groups[0].dimensions == {
        "repo": "https://github.com/Sinity/sinex",
        "origin": "codex-session",
    }


def test_join_rejects_active_plus_accepted_lifecycle_union() -> None:
    with pytest.raises(ValueError, match="represent one label lifecycle"):
        AnnotationStructuralJoinRequest(
            schema_id="delegation.discourse",
            schema_version=1,
            statuses=(AssertionStatus.ACCEPTED, AssertionStatus.ACTIVE),
        )


@pytest.mark.parametrize(
    ("decision", "terminal_status"),
    (
        ("accept", AssertionStatus.ACCEPTED),
        ("reject", AssertionStatus.REJECTED),
        ("defer", AssertionStatus.DEFERRED),
        ("supersede", AssertionStatus.SUPERSEDED),
    ),
)
@pytest.mark.asyncio
async def test_terminal_lifecycle_join_retains_labeler_and_judgment_provenance(
    workspace_env: dict[str, Path],
    decision: str,
    terminal_status: AssertionStatus,
) -> None:
    archive_root = workspace_env["archive_root"]
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"lifecycle-{decision}",
                title="Lifecycle target",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="evidence")],
            )
        )
    schema = AnnotationSchema(
        schema_id=f"test.lifecycle-{decision}",
        version=1,
        title="Lifecycle label",
        fields=(AnnotationField(name="score", value_type="number", minimum=0, maximum=1),),
        target_ref_kinds=("session",),
        evidence_policy="required",
        status="active",
    )
    registry = AnnotationSchemaRegistry()
    registry.register(schema)
    target_ref = f"session:{session_id}"
    with sqlite3.connect(archive_root / "user.db") as conn:
        persist_annotation_schema(conn, schema, registered_at_ms=1)
        candidate = upsert_annotation_assertion(
            conn,
            schema=schema,
            registry=registry,
            target_ref=target_ref,
            value={"score": 0.8},
            row_key="lifecycle-row",
            evidence_refs=(session_id,),
            author_ref="agent:original-labeler",
            confidence=0.8,
            now_ms=10,
        )
        candidate_ref = f"assertion:{candidate.assertion_id}"
        judge_assertion_candidate(
            conn,
            candidate_ref=candidate_ref,
            decision=decision,
            reason=f"operator chose {decision}",
            actor_ref="user:adjudicator",
            now_ms=20,
        )
        conn.commit()

    async with Polylogue(archive_root=archive_root) as poly:
        terminal = await join_typed_annotations(
            poly,
            AnnotationStructuralJoinRequest(
                schema_id=schema.schema_id,
                schema_version=1,
                statuses=(terminal_status,),
            ),
        )
        active = (
            await join_typed_annotations(
                poly,
                AnnotationStructuralJoinRequest(
                    schema_id=schema.schema_id,
                    schema_version=1,
                    statuses=(AssertionStatus.ACTIVE,),
                ),
            )
            if decision in {"accept", "supersede"}
            else None
        )

    [terminal_row] = terminal.rows
    assert terminal_row.source_assertion_ref == candidate_ref
    assert terminal_row.labeler_ref == "agent:original-labeler"
    assert terminal_row.adjudicator_ref == "user:adjudicator"
    assert terminal_row.judgment_ref is not None
    assert terminal_row.judgment_decision == decision
    assert terminal_row.judgment_reason == f"operator chose {decision}"
    if active is not None:
        [active_row] = active.rows
        assert active_row.source_assertion_ref == candidate_ref
        assert active_row.labeler_ref == "agent:original-labeler"
        assert active_row.adjudicator_ref == "user:adjudicator"
        assert active_row.judgment_decision == decision


@pytest.mark.asyncio
async def test_registry_drift_reports_truncated_diagnostics_honestly(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    target_ref, evidence_ref = _seed_delegation(archive_root)
    drifted_schema = replace(DELEGATION_DISCOURSE_SCHEMA, title="Drifted delegation discourse")
    definition = drifted_schema.canonical_definition_json()
    with sqlite3.connect(archive_root / "user.db") as conn:
        for index in range(101):
            upsert_assertion(
                conn,
                assertion_id=f"registry-drift-{index}",
                target_ref=target_ref,
                kind=AssertionKind.ANNOTATION,
                value={"_schema": "delegation.discourse@v1", **_delegation_value()},
                author_kind="user",
                evidence_refs=(evidence_ref,),
                status="active",
                now_ms=10_000 + index,
            )
        conn.execute(
            """
            UPDATE annotation_schemas
            SET definition_json = ?, definition_sha256 = ?
            WHERE schema_id = 'delegation.discourse' AND schema_version = 1
            """,
            (definition, hashlib.sha256(definition.encode("utf-8")).hexdigest()),
        )
        conn.commit()

    async with Polylogue(archive_root=archive_root) as poly:
        result = await join_typed_annotations(
            poly,
            AnnotationStructuralJoinRequest(
                schema_id="delegation.discourse",
                schema_version=1,
                statuses=(AssertionStatus.ACTIVE,),
                limit=101,
            ),
        )

    assert result.matched_annotation_count == 101
    assert result.schema_drift_count == 101
    assert len(result.diagnostics) == 100
    assert result.diagnostics_truncated is True
    assert result.joined_count == 0
