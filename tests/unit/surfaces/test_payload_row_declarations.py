"""Declaration parity and row-conversion contracts for surface payloads."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Any

import pytest

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.storage.sqlite.archive_tiers.archive import (
    ArchiveActionQueryRow,
    ArchiveAssertionQueryRow,
    ArchiveBlockQueryRow,
    ArchiveContextSnapshotQueryRow,
    ArchiveDelegationAncestryRow,
    ArchiveDelegationQueryRow,
    ArchiveDelegationSubtreeRow,
    ArchiveFileQueryRow,
    ArchiveMessageQueryRow,
    ArchiveObservedEventQueryRow,
    ArchiveQueryUnitAggregateRow,
    ArchiveRunQueryRow,
)
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    AssertionQueryRowPayload,
    BlockQueryRowPayload,
    ContextSnapshotQueryRowPayload,
    DelegationAncestryNodePayload,
    DelegationAttemptPayload,
    DelegationQueryRowPayload,
    DelegationSubtreeNodePayload,
    FileQueryRowPayload,
    MessageQueryRowPayload,
    ObservedEventQueryRowPayload,
    QueryUnitAggregateRowPayload,
    RunQueryRowPayload,
    SurfacePayloadModel,
)

_GENERIC_ROW_DECLARATIONS: tuple[tuple[type[SurfacePayloadModel], type[Any], frozenset[str], frozenset[str]], ...] = (
    (MessageQueryRowPayload, ArchiveMessageQueryRow, frozenset({"blocks"}), frozenset({"unit"})),
    (ActionQueryRowPayload, ArchiveActionQueryRow, frozenset(), frozenset({"unit"})),
    (BlockQueryRowPayload, ArchiveBlockQueryRow, frozenset(), frozenset({"unit"})),
    (FileQueryRowPayload, ArchiveFileQueryRow, frozenset(), frozenset({"unit"})),
    (AssertionQueryRowPayload, ArchiveAssertionQueryRow, frozenset(), frozenset({"unit"})),
    (DelegationAttemptPayload, ArchiveDelegationQueryRow, frozenset(), frozenset({"unit"})),
    (
        DelegationQueryRowPayload,
        ArchiveDelegationQueryRow,
        frozenset(
            {
                "artifact_text",
                "branch_point_message_id",
                "child_cost_is_estimated",
                "child_cost_usd",
                "child_session_dominant_model_family",
                "child_terminal_state",
                "child_tokens",
                "child_wall_ms",
                "instruction_payload",
                "parent_session_dominant_model",
                "parent_session_dominant_model_family",
                "parent_terminal_state",
            }
        ),
        frozenset(
            {
                "artifact_preview",
                "artifact_sha256",
                "artifact_truncated",
                "delegation_ref",
                "evidence_basis",
                "evidence_refs",
                "instruction_preview",
                "instruction_sha256",
                "instruction_truncated",
                "unit",
            }
        ),
    ),
    (DelegationAncestryNodePayload, ArchiveDelegationAncestryRow, frozenset(), frozenset()),
    (DelegationSubtreeNodePayload, ArchiveDelegationSubtreeRow, frozenset(), frozenset()),
    (
        ObservedEventQueryRowPayload,
        ArchiveObservedEventQueryRow,
        frozenset({"event"}),
        frozenset(
            {
                "delivery_state",
                "event_ref",
                "evidence_refs",
                "kind",
                "object_refs",
                "subject_ref",
                "summary",
                "unit",
            }
        ),
    ),
    (
        ContextSnapshotQueryRowPayload,
        ArchiveContextSnapshotQueryRow,
        frozenset({"snapshot"}),
        frozenset(
            {
                "boundary",
                "evidence_refs",
                "inheritance_mode",
                "metadata",
                "run_ref",
                "segment_refs",
                "snapshot_ref",
                "unit",
            }
        ),
    ),
    (
        RunQueryRowPayload,
        ArchiveRunQueryRow,
        frozenset({"run"}),
        frozenset(
            {
                "agent_ref",
                "confidence",
                "context_snapshot_ref",
                "cwd",
                "evidence_refs",
                "git_branch",
                "harness",
                "lineage_refs",
                "native_parent_session_id",
                "native_session_id",
                "parent_run_ref",
                "provider_origin",
                "role",
                "run_ref",
                "status",
                "transcript_ref",
                "unit",
            }
        ),
    ),
    (QueryUnitAggregateRowPayload, ArchiveQueryUnitAggregateRow, frozenset(), frozenset({"metrics"})),
)


@pytest.mark.parametrize(
    ("payload_model", "row_type", "expected_row_only", "expected_model_only"),
    _GENERIC_ROW_DECLARATIONS,
    ids=[payload_model.__name__ for payload_model, *_ in _GENERIC_ROW_DECLARATIONS],
)
def test_generic_row_declarations_have_explicit_field_parity(
    payload_model: type[SurfacePayloadModel],
    row_type: type[Any],
    expected_row_only: frozenset[str],
    expected_model_only: frozenset[str],
) -> None:
    """A row/model drift must be reviewed instead of hidden by generic copying."""
    row_fields = {field.name for field in fields(row_type)}
    model_fields = set(payload_model.model_fields)

    assert row_fields - model_fields == expected_row_only
    assert model_fields - row_fields == expected_model_only


def test_delegation_node_from_rows_preserves_declared_wire_fields() -> None:
    ancestry_row = ArchiveDelegationAncestryRow(
        session_id="session-leaf",
        depth=2,
        child_session_id="session-mid",
        mapping_state="resolved",
        instruction_tool_use_block_id="block-dispatch",
        link_confidence=0.75,
        link_method="provider-id",
    )
    subtree_row = ArchiveDelegationSubtreeRow(
        session_id="session-mid",
        depth=1,
        parent_session_id="session-root",
        mapping_state="edge_only",
        instruction_tool_use_block_id=None,
        link_confidence=None,
        link_method="topology-edge",
    )

    ancestry = DelegationAncestryNodePayload.from_row(ancestry_row)
    subtree = DelegationSubtreeNodePayload.from_row(subtree_row)

    assert ancestry.model_dump() == {
        "session_id": "session-leaf",
        "depth": 2,
        "child_session_id": "session-mid",
        "mapping_state": "resolved",
        "instruction_tool_use_block_id": "block-dispatch",
        "link_confidence": 0.75,
        "link_method": "provider-id",
    }
    assert subtree.model_dump() == {
        "session_id": "session-mid",
        "depth": 1,
        "parent_session_id": "session-root",
        "mapping_state": "edge_only",
        "instruction_tool_use_block_id": None,
        "link_confidence": None,
        "link_method": "topology-edge",
    }


def test_hybrid_row_converters_preserve_complete_wire_payloads() -> None:
    evidence = EvidenceRef(session_id="session-1")
    delegation_row = ArchiveDelegationQueryRow(
        parent_session_id="session-parent",
        child_session_id="session-child",
        mapping_state="resolved",
        link_confidence=0.75,
        link_method="provider-id",
        inheritance="spawned-fresh",
        branch_point_message_id=None,
        instruction_message_id="message-dispatch",
        instruction_tool_use_block_id="block-dispatch",
        instruction_payload='{"prompt":"review the diff"}',
        dispatch_turn_model="configured-model",
        requested_model="requested-model",
        artifact_block_id="block-result",
        artifact_text="review complete",
        result_is_error=0,
        result_exit_code=0,
        result_status="ok",
        parent_origin="codex-session",
        parent_session_dominant_model="configured-model",
        parent_session_dominant_model_family="configured-family",
        parent_terminal_state="completed",
        child_session_dominant_model="child-model",
        child_session_dominant_model_family="child-family",
        child_cost_usd=0.01,
        child_cost_is_estimated=0,
        child_tokens=100,
        child_wall_ms=5000,
        child_terminal_state="completed",
    )
    event = ObservedEvent(
        event_ref=ObjectRef(kind="observed-event", object_id="event-1"),
        kind="tool_finished",
        run_ref=ObjectRef(kind="run", object_id="run-1"),
        summary="tool finished",
        subject_ref=ObjectRef(kind="message", object_id="message-1"),
        object_refs=(ObjectRef(kind="run", object_id="run-1"),),
        evidence_refs=(evidence,),
    )
    snapshot = ContextSnapshot(
        snapshot_ref=ObjectRef(kind="context-snapshot", object_id="snapshot-1"),
        run_ref=ObjectRef(kind="run", object_id="run-1"),
        boundary="session_start",
        inheritance_mode="snapshot",
        segment_refs=(ObjectRef(kind="message", object_id="message-1"),),
        evidence_refs=(evidence,),
        metadata={"source": "fixture"},
    )
    run = ProjectedRun(
        run_ref=ObjectRef(kind="run", object_id="run-1"),
        native_session_id="native-1",
        native_parent_session_id="native-parent",
        parent_run_ref=ObjectRef(kind="run", object_id="run-parent"),
        agent_ref=ObjectRef(kind="agent", object_id="agent-1"),
        lineage_refs=(ObjectRef(kind="run", object_id="run-parent"),),
        provider_origin="codex-session",
        harness="codex",
        role="subagent",
        cwd="/workspace",
        git_branch="feature/test",
        status="completed",
        confidence="raw",
        transcript_ref=evidence,
        evidence_refs=(evidence,),
        context_snapshot_ref=snapshot.snapshot_ref,
    )

    delegation_payload = DelegationQueryRowPayload.from_row(delegation_row)
    observed_payload = ObservedEventQueryRowPayload.from_row(
        ArchiveObservedEventQueryRow(
            session_id="session-1",
            origin="codex-session",
            title="Observed run",
            event=event,
        )
    )
    context_payload = ContextSnapshotQueryRowPayload.from_row(
        ArchiveContextSnapshotQueryRow(
            session_id="session-1",
            origin="codex-session",
            title="Context run",
            snapshot=snapshot,
        )
    )
    run_payload = RunQueryRowPayload.from_row(
        ArchiveRunQueryRow(
            session_id="session-1",
            origin="codex-session",
            title="Projected run",
            run=run,
        )
    )

    assert delegation_payload.model_dump() == {
        "unit": "delegation",
        "delegation_ref": "delegation:block-dispatch",
        "parent_session_id": "session-parent",
        "child_session_id": "session-child",
        "parent_origin": "codex-session",
        "mapping_state": "resolved",
        "evidence_basis": "action",
        "instruction_message_id": "message-dispatch",
        "instruction_tool_use_block_id": "block-dispatch",
        "instruction_preview": "review the diff",
        "instruction_sha256": hashlib.sha256(b"review the diff").hexdigest(),
        "instruction_truncated": False,
        "artifact_block_id": "block-result",
        "artifact_preview": "review complete",
        "artifact_sha256": hashlib.sha256(b"review complete").hexdigest(),
        "artifact_truncated": False,
        "dispatch_turn_model": "configured-model",
        "requested_model": "requested-model",
        "child_session_dominant_model": "child-model",
        "result_is_error": False,
        "result_exit_code": 0,
        "result_status": "ok",
        "link_confidence": 0.75,
        "link_method": "provider-id",
        "inheritance": "spawned-fresh",
        "evidence_refs": ("block:block-dispatch", "block:block-result"),
    }
    assert observed_payload.model_dump() == {
        "unit": "observed-event",
        "event_ref": "observed-event:event-1",
        "session_id": "session-1",
        "origin": "codex-session",
        "title": "Observed run",
        "kind": "tool_finished",
        "summary": "tool finished",
        "delivery_state": "observed",
        "subject_ref": "message:message-1",
        "object_refs": ("run:run-1",),
        "evidence_refs": ("session-1",),
    }
    assert context_payload.model_dump() == {
        "unit": "context-snapshot",
        "snapshot_ref": "context-snapshot:snapshot-1",
        "session_id": "session-1",
        "origin": "codex-session",
        "title": "Context run",
        "run_ref": "run:run-1",
        "boundary": "session_start",
        "inheritance_mode": "snapshot",
        "segment_refs": ("message:message-1",),
        "evidence_refs": ("session-1",),
        "metadata": {"source": "fixture"},
    }
    assert run_payload.model_dump() == {
        "unit": "run",
        "run_ref": "run:run-1",
        "session_id": "session-1",
        "origin": "codex-session",
        "title": "Projected run",
        "native_session_id": "native-1",
        "native_parent_session_id": "native-parent",
        "parent_run_ref": "run:run-parent",
        "agent_ref": "agent:agent-1",
        "lineage_refs": ("run:run-parent",),
        "provider_origin": "codex-session",
        "harness": "codex",
        "role": "subagent",
        "cwd": "/workspace",
        "git_branch": "feature/test",
        "status": "completed",
        "confidence": "raw",
        "transcript_ref": "session-1",
        "evidence_refs": ("session-1",),
        "context_snapshot_ref": "context-snapshot:snapshot-1",
    }
