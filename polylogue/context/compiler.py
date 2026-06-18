"""Successor-context compiler over existing recovery/query evidence.

This module is intentionally thin.  It does not introduce a durable memory
store or a new handoff ontology; it normalizes the already-shipped recovery
transform, work-packet bundle, and report presets into one compiled read-model
shape that CLI/API/MCP surfaces can share.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import model_validator

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.transforms import RecoveryDigest, RecoveryReportPreset, RecoveryWorkPacket

RecoveryContextKind = Literal["recovery_digest", "recovery_report", "work_packet"]

_SUPPORTED_REPORTS: frozenset[str] = frozenset({"continue", "blame", "work-packet"})


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

    @model_validator(mode="after")
    def _validate_requested_shape(self) -> RecoveryContextCompilation:
        if self.kind in {"recovery_digest", "recovery_report"} and self.digest is None:
            raise ValueError(f"{self.kind} compilation requires digest")
        if self.kind == "work_packet" and self.work_packet is None:
            raise ValueError("work_packet compilation requires work_packet")
        if self.kind in {"recovery_report", "work_packet"} and not self.markdown:
            raise ValueError(f"{self.kind} compilation requires markdown")
        return self


def compile_recovery_context(
    digest: RecoveryDigest,
    *,
    report: RecoveryReportPreset | str | None = None,
    work_packet: RecoveryWorkPacket | None = None,
) -> RecoveryContextCompilation:
    """Compile a recovery digest into the requested successor-context view.

    The compiler deliberately consumes existing primitives:

    * ``RecoveryDigest`` for deterministic transform output;
    * ``RecoveryReportPreset`` for continue/blame/work-packet rendering;
    * ``RecoveryWorkPacket`` for Bundle(kind=work_packet) handoff material;
    * ``EvidenceRef``/``ObjectRef`` for addressable support and targets.
    """

    if report is None:
        if work_packet is not None:
            raise ValueError("work_packet may only be supplied when report='work-packet'")
        return RecoveryContextCompilation(
            kind="recovery_digest",
            session_id=digest.session_id,
            digest=digest,
            markdown=digest.resume_markdown,
            evidence_refs=_digest_evidence_refs(digest),
            object_refs=_digest_object_refs(digest),
            caveats=_digest_caveats(digest),
        )

    if report not in _SUPPORTED_REPORTS:
        raise ValueError(f"unsupported recovery report preset: {report}")
    preset = cast(RecoveryReportPreset, report)
    if preset == "work-packet":
        packet = work_packet or digest.work_packet()
        return RecoveryContextCompilation(
            kind="work_packet",
            session_id=digest.session_id,
            report=preset,
            digest=digest,
            work_packet=packet,
            markdown=packet.render_markdown(),
            evidence_refs=packet.evidence_refs,
            object_refs=packet.target_refs,
            caveats=_packet_caveats(packet),
        )

    if work_packet is not None:
        raise ValueError("work_packet may only be supplied when report='work-packet'")
    return RecoveryContextCompilation(
        kind="recovery_report",
        session_id=digest.session_id,
        report=preset,
        digest=digest,
        markdown=digest.report_markdown(preset),
        evidence_refs=_digest_evidence_refs(digest),
        object_refs=_digest_object_refs(digest),
        caveats=_digest_caveats(digest),
    )


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


__all__ = [
    "RecoveryContextCompilation",
    "RecoveryContextKind",
    "compile_recovery_context",
]
