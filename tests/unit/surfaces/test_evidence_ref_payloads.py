"""Public evidence payload ref validation contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.surfaces.payloads import (
    AssertionClaimPayload,
    AssertionQueryRowPayload,
    ContextSnapshotQueryRowPayload,
    ObservedEventQueryRowPayload,
)


def test_assertion_query_payload_rejects_unparseable_target_ref() -> None:
    with pytest.raises(ValidationError, match="object ref must use"):
        AssertionQueryRowPayload(
            assertion_id="a1",
            target_ref="claim-target",
            kind="note",
            value={"ok": True},
            author_ref="user:local",
            author_kind="user",
            status="active",
            visibility="private",
            evidence_refs=("session:codex-session:s1::m1",),
            staleness={},
            context_policy={"inject": False},
            created_at_ms=1,
            updated_at_ms=1,
        )


def test_assertion_claim_payload_accepts_object_and_evidence_refs() -> None:
    payload = AssertionClaimPayload(
        assertion_id="a1",
        scope_ref="workspace:polylogue",
        target_ref="message:codex-session:s1:m1",
        kind="note",
        body_text="needs review",
        author_ref="user:local",
        evidence_refs=(
            "session:codex-session:s1",
            "codex-session:s1::m1",
        ),
        created_at_ms=1,
        updated_at_ms=1,
    )

    assert payload.scope_ref == "workspace:polylogue"
    assert payload.target_ref == "message:codex-session:s1:m1"
    assert payload.evidence_refs == (
        "session:codex-session:s1",
        "codex-session:s1::m1",
    )


def test_observed_event_payload_rejects_unparseable_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="unsupported public ref"):
        ObservedEventQueryRowPayload(
            event_ref="observed-event:event-1",
            session_id="s1",
            origin="codex-session",
            kind="review",
            summary="review injected",
            delivery_state="observed",
            subject_ref="message:s1:m1",
            object_refs=("tool-call:call-1",),
            evidence_refs=("codex-session:s1::m1::-1",),
        )


def test_context_snapshot_payload_validates_run_segment_and_evidence_refs() -> None:
    payload = ContextSnapshotQueryRowPayload(
        snapshot_ref="context-snapshot:snap-1",
        session_id="s1",
        origin="codex-session",
        run_ref="run:run-1",
        boundary="review_injection",
        inheritance_mode="copied",
        segment_refs=("message:s1:m1", "block:s1:m1:0"),
        evidence_refs=("codex-session:s1::m1::0",),
        metadata={"source": "review"},
    )

    assert payload.snapshot_ref == "context-snapshot:snap-1"
    assert payload.run_ref == "run:run-1"
    assert payload.segment_refs == ("message:s1:m1", "block:s1:m1:0")
    assert payload.evidence_refs == ("codex-session:s1::m1::0",)


def test_context_snapshot_payload_rejects_unparseable_segment_ref() -> None:
    with pytest.raises(ValidationError, match="object ref must use"):
        ContextSnapshotQueryRowPayload(
            snapshot_ref="context-snapshot:snap-1",
            session_id="s1",
            origin="codex-session",
            run_ref="run:run-1",
            boundary="review_injection",
            inheritance_mode="copied",
            segment_refs=("segment-1",),
            evidence_refs=("codex-session:s1",),
            metadata={},
        )
