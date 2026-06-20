"""Successor-context compiler over existing recovery/query evidence.

This module is intentionally thin.  It does not introduce a durable memory
store or a new handoff ontology; it normalizes the already-shipped recovery
transform, work-packet bundle, and report presets into one compiled read-model
shape that CLI/API/MCP surfaces can share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import Field, model_validator

from polylogue.context.assertion_claims import context_claim_text
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.transforms import RecoveryDigest, RecoveryReportPreset, RecoveryWorkPacket
from polylogue.surfaces.payloads import AssertionClaimPayload

RecoveryContextKind = Literal["recovery_digest", "recovery_report", "work_packet"]
ContextPurpose = Literal["continue", "review", "handoff", "debug", "export"]
ContextSegmentKind = Literal["recovery", "read_view", "assertion", "caveat"]
ContextOmissionReason = Literal["budget", "unsupported", "not_found", "policy", "redacted"]

_SUPPORTED_REPORTS: frozenset[str] = frozenset({"continue", "blame", "work-packet"})


class ContextSegment(ArchiveInsightModel):
    """One bounded section of a compiled context image."""

    segment_id: str
    kind: ContextSegmentKind
    title: str
    markdown: str | None = None
    payload_kind: str | None = None
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0
    lossiness: str | None = None


class ContextOmission(ArchiveInsightModel):
    """A requested context input that was omitted deliberately."""

    ref: str | None = None
    query: str | None = None
    view: str | None = None
    reason: ContextOmissionReason
    detail: str


class ContextSpec(ArchiveInsightModel):
    """Declarative request for a compiled context image.

    The spec is pure input. It does not imply delivery, durable recording, or
    automatic context injection.
    """

    purpose: ContextPurpose = "continue"
    seed_query: str | None = None
    seed_refs: tuple[str, ...] = ()
    read_views: tuple[str, ...] = ("recovery",)
    max_tokens: int | None = Field(default=None, ge=1)
    include_assertions: bool = True
    include_candidates: bool = False
    redaction_policy: Literal["default", "raw-opt-in"] = "default"

    @model_validator(mode="after")
    def _requires_seed(self) -> ContextSpec:
        if self.seed_query is None and not self.seed_refs:
            raise ValueError("ContextSpec requires seed_query or seed_refs")
        return self


class ContextImage(ArchiveInsightModel):
    """Compiled, storage-free context payload over archive refs and views."""

    spec: ContextSpec
    segments: tuple[ContextSegment, ...]
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    omitted: tuple[ContextOmission, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0


class ContextSnapshotRecord(ArchiveInsightModel):
    """Evidence record for context that was actually delivered."""

    snapshot_ref: str
    run_ref: str | None = None
    boundary: str
    inheritance_mode: str = "explicit"
    segment_refs: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


class RecoveryContextCompilation(ArchiveInsightModel):
    """Compiled successor-context read model for one recovery view request.

    ``RecoveryDigest`` remains the transform output, ``RecoveryWorkPacket``
    remains the first Bundle(kind=work_packet) DTO, and Markdown remains a
    rendered view.  This wrapper only records which one was requested and
    exposes the refs/caveats needed by caller surfaces without making each
    caller rediscover the report-vs-packet relationship.
    """

    kind: RecoveryContextKind
    view_id: Literal["recovery"] = "recovery"
    session_id: str
    report: RecoveryReportPreset | None = None
    digest: RecoveryDigest | None = None
    work_packet: RecoveryWorkPacket | None = None
    markdown: str | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    object_refs: tuple[ObjectRef, ...] = ()
    caveats: tuple[str, ...] = ()
    assertion_claims: tuple[AssertionClaimPayload, ...] = ()
    context_image: ContextImage | None = None

    @model_validator(mode="after")
    def _validate_requested_shape(self) -> RecoveryContextCompilation:
        if self.kind in {"recovery_digest", "recovery_report"} and self.digest is None:
            raise ValueError(f"{self.kind} compilation requires digest")
        if self.kind == "work_packet" and self.work_packet is None:
            raise ValueError("work_packet compilation requires work_packet")
        if self.kind in {"recovery_report", "work_packet"} and not self.markdown:
            raise ValueError(f"{self.kind} compilation requires markdown")
        if self.context_image is not None and self.context_image.segments == ():
            raise ValueError("context_image requires at least one segment")
        return self


def compile_recovery_context(
    digest: RecoveryDigest,
    *,
    report: RecoveryReportPreset | str | None = None,
    work_packet: RecoveryWorkPacket | None = None,
    assertion_claims: Sequence[AssertionClaimPayload] = (),
) -> RecoveryContextCompilation:
    """Compile a recovery digest into the requested successor-context view.

    The compiler deliberately consumes existing primitives:

    * ``RecoveryDigest`` for deterministic transform output;
    * ``RecoveryReportPreset`` for continue/blame/work-packet rendering;
    * ``RecoveryWorkPacket`` for Bundle(kind=work_packet) handoff material;
    * ``EvidenceRef``/``ObjectRef`` for addressable support and targets.
    * ``AssertionClaimPayload`` for explicit context-injection claims.
    """

    claims = tuple(assertion_claims)

    if report is None:
        if work_packet is not None:
            raise ValueError("work_packet may only be supplied when report='work-packet'")
        compiled = RecoveryContextCompilation(
            kind="recovery_digest",
            session_id=digest.session_id,
            digest=digest,
            markdown=_append_assertion_claims_markdown(digest.resume_markdown, claims),
            evidence_refs=_digest_evidence_refs(digest),
            object_refs=_digest_object_refs(digest),
            caveats=_digest_caveats(digest),
            assertion_claims=claims,
        )
        return _attach_context_image(compiled)

    if report not in _SUPPORTED_REPORTS:
        raise ValueError(f"unsupported recovery report preset: {report}")
    preset = cast(RecoveryReportPreset, report)
    if preset == "work-packet":
        packet = work_packet or digest.work_packet()
        compiled = RecoveryContextCompilation(
            kind="work_packet",
            session_id=digest.session_id,
            report=preset,
            digest=digest,
            work_packet=packet,
            markdown=packet.render_markdown(),
            evidence_refs=packet.evidence_refs,
            object_refs=packet.target_refs,
            caveats=_packet_caveats(packet),
            assertion_claims=claims,
        )
        return _attach_context_image(compiled)

    if work_packet is not None:
        raise ValueError("work_packet may only be supplied when report='work-packet'")
    compiled = RecoveryContextCompilation(
        kind="recovery_report",
        session_id=digest.session_id,
        report=preset,
        digest=digest,
        markdown=_append_assertion_claims_markdown(digest.report_markdown(preset), claims),
        evidence_refs=_digest_evidence_refs(digest),
        object_refs=_digest_object_refs(digest),
        caveats=_digest_caveats(digest),
        assertion_claims=claims,
    )
    return _attach_context_image(compiled)


def context_image_from_recovery(compilation: RecoveryContextCompilation) -> ContextImage:
    """Promote a recovery compilation into the general context-image shape."""

    segment = ContextSegment(
        segment_id=f"recovery:{compilation.session_id}:{compilation.report or compilation.kind}",
        kind="recovery",
        title=_recovery_segment_title(compilation),
        markdown=compilation.markdown,
        payload_kind=compilation.kind,
        object_refs=compilation.object_refs,
        evidence_refs=compilation.evidence_refs,
        assertion_refs=tuple(claim.assertion_id for claim in compilation.assertion_claims),
        caveats=compilation.caveats,
        token_estimate=_estimate_tokens(compilation.markdown),
        lossiness="bounded_recovery_transform",
    )
    spec = ContextSpec(
        purpose="handoff" if compilation.report == "work-packet" else "continue",
        seed_refs=(f"session:{compilation.session_id}",),
        read_views=("work-packet",) if compilation.report == "work-packet" else ("recovery",),
        include_assertions=bool(compilation.assertion_claims),
        include_candidates=False,
    )
    assertion_refs = tuple(claim.assertion_id for claim in compilation.assertion_claims)
    return ContextImage(
        spec=spec,
        segments=(segment,),
        object_refs=compilation.object_refs,
        evidence_refs=compilation.evidence_refs,
        assertion_refs=assertion_refs,
        caveats=compilation.caveats,
        token_estimate=segment.token_estimate,
    )


def compile_messages_context_segment(
    *,
    session_id: str,
    title: str | None,
    messages: Sequence[tuple[str, str]],
    evidence_refs: Sequence[EvidenceRef],
) -> ContextSegment:
    """Compile a normalized message transcript into a context segment."""

    lines = [f"# Messages: {title or session_id}", ""]
    for role, text in messages:
        lines.append(f"{role}: {text}")
        lines.append("")
    markdown = "\n".join(lines).rstrip() + "\n"
    return ContextSegment(
        segment_id=f"read-view:{session_id}:messages",
        kind="read_view",
        title="Messages",
        markdown=markdown,
        payload_kind="messages",
        object_refs=(ObjectRef(kind="session", object_id=session_id),),
        evidence_refs=tuple(evidence_refs),
        token_estimate=_estimate_tokens(markdown),
        lossiness="normalized_message_text",
    )


def context_snapshot_record_from_image(
    image: ContextImage,
    *,
    boundary: str,
    run_ref: str | None = None,
    inheritance_mode: str = "explicit",
) -> ContextSnapshotRecord:
    """Build a storage-free evidence record for delivered context.

    Compilation remains pure. Callers use this helper only at a delivery
    boundary, then persist or emit the returned record through the surface that
    actually performed the handoff.
    """
    if not boundary.strip():
        raise ValueError("ContextSnapshotRecord requires a delivery boundary")
    segment_refs = tuple(segment.segment_id for segment in image.segments)
    metadata: dict[str, str] = {
        "purpose": _metadata_value_to_text(image.spec.purpose),
        "read_views": _metadata_value_to_text(image.spec.read_views),
        "max_tokens": _metadata_value_to_text(image.spec.max_tokens),
        "token_estimate": _metadata_value_to_text(image.token_estimate),
        "include_assertions": _metadata_value_to_text(image.spec.include_assertions),
        "include_candidates": _metadata_value_to_text(image.spec.include_candidates),
        "redaction_policy": _metadata_value_to_text(image.spec.redaction_policy),
        "omitted_count": _metadata_value_to_text(len(image.omitted)),
        "assertion_refs": _metadata_value_to_text(image.assertion_refs),
        "caveats": _metadata_value_to_text(image.caveats),
    }
    fingerprint_payload = {
        "boundary": boundary,
        "run_ref": run_ref,
        "inheritance_mode": inheritance_mode,
        "segment_refs": segment_refs,
        "evidence_refs": tuple(ref.format() for ref in image.evidence_refs),
        "metadata": metadata,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return ContextSnapshotRecord(
        snapshot_ref=f"context-snapshot:{fingerprint}",
        run_ref=run_ref,
        boundary=boundary,
        inheritance_mode=inheritance_mode,
        segment_refs=segment_refs,
        evidence_refs=image.evidence_refs,
        metadata=metadata,
    )


def _metadata_value_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _attach_context_image(compilation: RecoveryContextCompilation) -> RecoveryContextCompilation:
    return compilation.model_copy(update={"context_image": context_image_from_recovery(compilation)})


def _recovery_segment_title(compilation: RecoveryContextCompilation) -> str:
    if compilation.report == "work-packet":
        return "Recovery work packet"
    if compilation.report is not None:
        return f"Recovery {compilation.report} report"
    return "Recovery digest"


def _estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _digest_evidence_refs(digest: RecoveryDigest) -> tuple[EvidenceRef, ...]:
    return tuple(ref.to_evidence_ref() for ref in digest.raw_refs)


def _digest_object_refs(digest: RecoveryDigest) -> tuple[ObjectRef, ...]:
    refs = [ObjectRef(kind="session", object_id=digest.session_id)]
    if digest.git_branch:
        refs.append(ObjectRef(kind="branch", object_id=digest.git_branch))
    return tuple(refs)


def _digest_caveats(digest: RecoveryDigest) -> tuple[str, ...]:
    caveats: list[str] = []
    if digest.run_state is None:
        caveats.append("run_state_missing")
    if not digest.events:
        caveats.append("events_missing")
    if not digest.tool_summaries:
        caveats.append("tool_summaries_missing")
    if not digest.subagent_reports:
        caveats.append("subagent_reports_missing")
    return tuple(caveats)


def _packet_caveats(packet: RecoveryWorkPacket) -> tuple[str, ...]:
    return tuple(entry.label for entry in packet.entries if entry.support in {"caveat", "missing_evidence"})


def _append_assertion_claims_markdown(markdown: str, claims: Sequence[AssertionClaimPayload]) -> str:
    if not claims:
        return markdown
    lines = [markdown.rstrip(), "", "## Assertion Claims"]
    for claim in claims:
        lines.append(f"- {context_claim_text(kind=claim.kind, body_text=claim.body_text, target_ref=claim.target_ref)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ContextImage",
    "ContextOmission",
    "ContextOmissionReason",
    "ContextPurpose",
    "ContextSegment",
    "ContextSegmentKind",
    "ContextSnapshotRecord",
    "ContextSpec",
    "RecoveryContextCompilation",
    "RecoveryContextKind",
    "compile_messages_context_segment",
    "compile_recovery_context",
    "context_image_from_recovery",
    "context_snapshot_record_from_image",
]
