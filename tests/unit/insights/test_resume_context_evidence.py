from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.analysis.resume import classify_resume_context_evidence
from polylogue.analysis.work_evidence import WorkEvidenceNode
from polylogue.context.compiler import (
    ContextImage,
    ContextSegment,
    ContextSpec,
    context_snapshot_record_from_image,
)
from polylogue.core.refs import ActorRef, EvidenceRef, ExecutionContextRef, ObjectRef
from polylogue.storage.sqlite.archive_tiers.context_delivery_write import (
    ArchiveContextDeliveryEnvelope,
    write_context_delivery,
)
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL


def _delivery(
    database_path: Path, *, recipient_ref: str, snapshot_suffix: str = "fixture"
) -> ArchiveContextDeliveryEnvelope:
    conn = sqlite3.connect(database_path)
    conn.executescript(USER_DDL)
    image = ContextImage(
        spec=ContextSpec(seed_refs=("session:codex-session:seed",), read_views=()),
        segments=(
            ContextSegment(
                segment_id="read-view:codex-session:seed:messages",
                kind="read_view",
                title="Seed messages",
                markdown="User: continue from this evidence\n",
                evidence_refs=(EvidenceRef(session_id="codex-session:seed", message_id="m1"),),
            ),
        ),
        evidence_refs=(EvidenceRef(session_id="codex-session:seed", message_id="m1"),),
    )
    record = context_snapshot_record_from_image(
        image,
        boundary="resume-context",
        run_ref=f"run:{snapshot_suffix}",
    )
    receipt = write_context_delivery(
        conn,
        image=image,
        record=record,
        recipient_ref=recipient_ref,
        delivered_by_ref="agent:codex-main",
        delivered_at_ms=100,
    )
    conn.close()
    return receipt


_SOURCE_REF = ObjectRef(kind="artifact", object_id="raw:resume-fixture").format()


def _resume_invocation() -> WorkEvidenceNode:
    return WorkEvidenceNode(
        ref=ObjectRef(kind="work-invocation", object_id="mcp-call:resume-brief"),
        kind="invocation",
        label="get_resume_brief",
        evidence_refs=(ObjectRef(kind="artifact", object_id="raw:mcp-call"),),
        corpus_snapshot_ref=ObjectRef(kind="context-snapshot", object_id="fixture:corpus"),
        authority="provider",
        confidence=1.0,
        occurred_at_ms=90,
        actor_ref=ActorRef(kind="agent", identity="codex-main"),
        execution_context_ref=ExecutionContextRef.from_observation({"cwd": "/synthetic/repo"}),
    )


def test_context_delivery_joins_successor_without_becoming_resume_topology(tmp_path: Path) -> None:
    receipt = _delivery(tmp_path / "context-delivery.db", recipient_ref="session:codex-session:successor")
    invocation = _resume_invocation()

    result = classify_resume_context_evidence(
        successor_session_id="codex-session:successor",
        topology_link_type="continuation",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_invocations=(invocation,),
        context_deliveries=(receipt,),
        context_receipts_complete=True,
    )

    assert result.arm == "context_assisted_continuation"
    assert result.topology_link_type == "continuation"
    assert result.context_delivery_refs == (receipt.snapshot_ref,)
    assert result.successor_session_ref == "session:codex-session:successor"
    assert result.context_deliveries[0].delivered_by_ref == "agent:codex-main"
    assert result.context_deliveries[0].delivered_at_ms == 100
    assert result.context_deliveries[0].evidence_refs == receipt.evidence_refs
    assert result.context_invocations == (invocation,)
    assert result.context_invocations[0].actor_ref == invocation.actor_ref
    assert result.context_invocations[0].execution_context_ref == invocation.execution_context_ref


@pytest.mark.parametrize("consumed_preparation_refs", [None, ()])
def test_direct_delivery_outranks_unrelated_preparation(
    tmp_path: Path, consumed_preparation_refs: tuple[str, ...] | None
) -> None:
    receipt = _delivery(tmp_path / "direct-delivery.db", recipient_ref="session:codex-session:successor")
    unrelated_preparation = "context-snapshot:unrelated"

    result = classify_resume_context_evidence(
        successor_session_id="codex-session:successor",
        topology_link_type="continuation",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_deliveries=(receipt,),
        preparation_refs=(unrelated_preparation,),
        consumed_preparation_refs=consumed_preparation_refs,
    )

    assert result.arm == "context_assisted_continuation"
    assert result.context_delivery_refs == (receipt.snapshot_ref,)
    assert result.preparation_refs == (unrelated_preparation,)


def test_same_contract_distinguishes_native_resume_and_bare_continuation() -> None:
    native_resume = classify_resume_context_evidence(
        successor_session_id="codex-session:native-resume",
        topology_link_type="resume",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_receipts_complete=True,
    )
    bare_continuation = classify_resume_context_evidence(
        successor_session_id="codex-session:bare-continuation",
        topology_link_type="continuation",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_receipts_complete=True,
    )

    assert native_resume.arm == "provider_native_resume_without_context"
    assert bare_continuation.arm == "bare_continuation"


def test_prepared_but_unconsumed_context_requires_complete_work_evidence() -> None:
    preparation_ref = "context-snapshot:prepared-fixture"

    complete = classify_resume_context_evidence(
        successor_session_id=None,
        topology_link_type=None,
        preparation_refs=(preparation_ref,),
        consumed_preparation_refs=(),
    )
    incomplete = classify_resume_context_evidence(
        successor_session_id=None,
        topology_link_type=None,
        preparation_refs=(preparation_ref,),
        consumed_preparation_refs=None,
    )

    assert complete.arm == "prepared_unused_context"
    assert complete.preparation_refs == (preparation_ref,)
    assert incomplete.arm == "unavailable"
    assert incomplete.unavailable_reason == "work-evidence consumption projection is incomplete"


def test_unresolved_successor_and_missing_native_source_evidence_stay_unavailable() -> None:
    unresolved = classify_resume_context_evidence(
        successor_session_id=None,
        topology_link_type="continuation",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_receipts_complete=True,
    )
    unsupported_resume = classify_resume_context_evidence(
        successor_session_id="codex-session:unknown-resume",
        topology_link_type="resume",
        context_receipts_complete=True,
    )

    assert unresolved.arm == "unavailable"
    assert unresolved.unavailable_reason == "successor identity is unresolved"
    assert unsupported_resume.arm == "unavailable"
    assert "exact source evidence" in (unsupported_resume.unavailable_reason or "")


def test_tool_names_or_timestamps_cannot_create_a_resume_arm() -> None:
    result = classify_resume_context_evidence(
        successor_session_id="codex-session:unclassified",
        topology_link_type=None,
        context_invocations=(_resume_invocation(),),
        context_receipts_complete=True,
    )

    assert result.arm == "unavailable"
    assert result.topology_link_type is None
    assert result.context_invocations[0].label == "get_resume_brief"


def test_only_direct_successor_receipts_assist_a_continuation(tmp_path: Path) -> None:
    other_recipient = _delivery(
        tmp_path / "other-context-delivery.db",
        recipient_ref="agent:codex-main",
        snapshot_suffix="other",
    )

    result = classify_resume_context_evidence(
        successor_session_id="codex-session:successor",
        topology_link_type="continuation",
        topology_source_evidence_refs=(_SOURCE_REF,),
        context_deliveries=(other_recipient,),
        context_receipts_complete=True,
    )

    assert result.arm == "bare_continuation"
    assert result.context_delivery_refs == ()


@pytest.mark.parametrize("invalid_ref", ["session:seed", "run:r1", "not-a-ref"])
def test_preparation_refs_must_be_context_snapshot_object_refs(invalid_ref: str) -> None:
    with pytest.raises(ValueError, match="preparation refs must use context-snapshot ObjectRefs"):
        classify_resume_context_evidence(
            successor_session_id=None,
            topology_link_type=None,
            preparation_refs=(invalid_ref,),
            consumed_preparation_refs=(),
        )


def test_consumed_preparation_refs_must_also_be_context_snapshots() -> None:
    with pytest.raises(ValueError, match="consumed preparation refs must use context-snapshot ObjectRefs"):
        classify_resume_context_evidence(
            successor_session_id=None,
            topology_link_type=None,
            consumed_preparation_refs=("artifact:raw:unrelated",),
        )
