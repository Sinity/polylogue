"""Deterministic recovery/digest transforms for coding-agent sessions.

The v0 transform surface is intentionally storage-free: it compiles an
already-hydrated :class:`~polylogue.archive.session.domain_models.Session`
into small typed recovery artifacts while preserving raw message/block refs for
drilldown. Raw archive rows stay the evidence source; these records are
read-model candidates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal, TypeVar

from pydantic import Field, field_validator, model_validator

from polylogue.archive.message.models import Message
from polylogue.archive.session.domain_models import Session
from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.insights.run_projection import RunProjection, build_run_projection

RECOVERY_TRANSFORM_ID = "recovery_digest_v0"
RECOVERY_TRANSFORM_VERSION = 1
T = TypeVar("T")

ForensicClaimKind = Literal[
    "digest",
    "tool_summary",
    "subagent_report",
    "run_state",
    "event",
    "decision_candidate",
]
RecoveryReportPreset = Literal["continue", "blame", "work-packet"]
WorkPacketSection = Literal[
    "execution",
    "events",
    "subagents",
    "run_state",
    "tools",
    "decisions",
    "assertions",
    "evidence_gaps",
]
WorkPacketSupport = Literal["raw_evidence", "assertion", "inference", "caveat", "missing_evidence"]
SubagentChildLinkStatus = Literal["resolved", "unresolved", "repaired", "quarantined"]

_GITHUB_REPO_REF = r"[\w.-]+/[\w.-]+#"
_ISSUE_RE = re.compile(
    rf"(?:issues/|issue\s+(?:{_GITHUB_REPO_REF}|#?)|closed\s+(?:issue\s+)?(?:{_GITHUB_REPO_REF}|#?))"
    r"(?P<number>\d{3,6})",
    re.IGNORECASE,
)
_PR_RE = re.compile(r"(?:pull/|PR\s+#?)(?P<number>\d{3,6})", re.IGNORECASE)
_REVIEW_PR_RE = re.compile(r"\b(?:review|comment|finding)s?\b.*(?:pull/|PR\s+#?)(?P<number>\d{3,6})", re.IGNORECASE)
_CREATED_PR_RE = re.compile(
    r"\b(?:created|opened)\s+(?:pull\s+request|PR)\s+#?(?P<number>\d{3,6})\b",
    re.IGNORECASE,
)
_MERGED_RE = re.compile(r"\bMERGED\s+#(?P<number>\d{3,6})\b", re.IGNORECASE)
_TEST_PASS_RE = re.compile(r"\b(?P<count>\d+)\s+passed\b", re.IGNORECASE)
_TEST_FAIL_RE = re.compile(r"\b(?P<count>\d+)\s+failed\b", re.IGNORECASE)
_CHECK_PASS_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_.() -]+)\s+\.\.\.\s+ok\b")
_CHECK_FAIL_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_.() -]+)\s+\.\.\.\s+(?:FAIL|FAILED|failure)\b", re.IGNORECASE)
_COMMIT_SHA_RE = re.compile(r"\b(?P<sha>[0-9a-f]{7,40})\b", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(decision|decided|choose|chosen):?\s+(?P<text>.+)", re.IGNORECASE)
_STATUS_HEADING_RE = re.compile(r"^\s*(goal|done|in flight|blockers?|next):\s*(?P<text>.+)$", re.IGNORECASE)
_RUNSTATE_SECTION_RE = re.compile(
    r"^\s*(goal|done|in flight|blockers?|next(?: action)?s?):\s*(?P<text>.*)$",
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TransformRawRef(ArchiveInsightModel):
    """Pointer back to the raw session evidence that produced a digest claim."""

    session_id: str
    message_id: str | None = None
    block_index: int | None = None
    ref_kind: Literal["session", "message", "block"] = "message"
    preview: str = ""

    @field_validator("session_id")
    @classmethod
    def _session_id_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("session_id cannot be empty")
        return value

    def to_evidence_ref(self) -> EvidenceRef:
        """Return the shared DTO used to parse/format recovery evidence ids."""

        return EvidenceRef(
            session_id=self.session_id,
            message_id=self.message_id,
            block_index=self.block_index,
        )


class ForensicIndexEntry(ArchiveInsightModel):
    """One raw evidence location and the extracted claims it supports."""

    evidence_id: str
    raw_ref: TransformRawRef
    claim_kinds: tuple[ForensicClaimKind, ...]
    claim_labels: tuple[str, ...]

    @model_validator(mode="after")
    def _requires_claims(self) -> ForensicIndexEntry:
        if not self.claim_kinds:
            raise ValueError("ForensicIndexEntry requires at least one claim kind")
        if not self.claim_labels:
            raise ValueError("ForensicIndexEntry requires at least one claim label")
        return self


class ForensicIndex(ArchiveInsightModel):
    """Deterministic raw-ref index for blame/continue recovery surfaces."""

    session_id: str
    entries: tuple[ForensicIndexEntry, ...] = ()
    claim_count: int = 0


class TransformMetadata(ArchiveInsightModel):
    transform_id: str
    transform_version: int
    input_session_id: str
    source_origin: str
    computed_at: str = Field(default_factory=_utc_now_iso)
    input_message_count: int = 0


class RecoverySizeMetrics(ArchiveInsightModel):
    raw_bytes: int
    normal_read_bytes: int
    resume_bundle_bytes: int
    message_count: int
    tool_summary_count: int
    subagent_report_count: int
    run_state_count: int
    event_count: int
    decision_candidate_count: int

    @property
    def resume_to_raw_ratio(self) -> float:
        if self.raw_bytes <= 0:
            return 0.0
        return self.resume_bundle_bytes / self.raw_bytes


class ToolSummary(ArchiveInsightModel):
    tool_name: str
    tool_id: str | None = None
    command: str | None = None
    handler_kind: Literal["shell", "file_read", "github", "git", "test", "generic"] = "generic"
    status: Literal["ok", "failed", "unknown"] = "unknown"
    line_count: int = 0
    output_preview: str = ""
    pr_refs: tuple[str, ...] = ()
    issue_refs: tuple[str, ...] = ()
    test_evidence: tuple[str, ...] = ()
    file_refs: tuple[str, ...] = ()
    commit_refs: tuple[str, ...] = ()
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_raw_refs(self) -> ToolSummary:
        if not self.raw_refs:
            raise ValueError("ToolSummary requires at least one raw ref")
        return self


class SubagentReport(ArchiveInsightModel):
    subagent_type: str = "unknown"
    tool_id: str | None = None
    task_id: str | None = None
    child_session_id: str | None = None
    child_link_status: SubagentChildLinkStatus | None = None
    child_link_type: str | None = None
    resolved_child_session_id: str | None = None
    prompt: str = ""
    final_report_preview: str = ""
    pr_refs: tuple[str, ...] = ()
    issue_refs: tuple[str, ...] = ()
    test_evidence: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_raw_refs(self) -> SubagentReport:
        if not self.raw_refs:
            raise ValueError("SubagentReport requires at least one raw ref")
        return self


class RunStateSummary(ArchiveInsightModel):
    goal: str | None = None
    done: tuple[str, ...] = ()
    in_flight: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_content_and_raw_refs(self) -> RunStateSummary:
        if not self.raw_refs:
            raise ValueError("RunStateSummary requires at least one raw ref")
        if not any((self.goal, self.done, self.in_flight, self.blockers, self.next_actions)):
            raise ValueError("RunStateSummary requires at least one parsed field")
        return self


class RecoveryEvent(ArchiveInsightModel):
    kind: Literal[
        "pr_opened",
        "pr_merged",
        "review_posted",
        "review_seen_by_tool",
        "review_injected_context",
        "review_acknowledged",
        "review_acted_on",
        "issue_closed",
        "check_passed",
        "check_failed",
        "test_passed",
        "test_failed",
    ]
    summary: str
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_raw_refs(self) -> RecoveryEvent:
        if not self.raw_refs:
            raise ValueError("RecoveryEvent requires at least one raw ref")
        return self


class DecisionCandidate(ArchiveInsightModel):
    kind: Literal["decision", "run_state"]
    text: str
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_raw_refs(self) -> DecisionCandidate:
        if not self.raw_refs:
            raise ValueError("DecisionCandidate requires at least one raw ref")
        return self


class RecoveryWorkPacketEntry(ArchiveInsightModel):
    """One storage-free successor-session packet item with typed evidence refs."""

    section: WorkPacketSection
    label: str
    text: str
    evidence_refs: tuple[EvidenceRef, ...]
    object_refs: tuple[ObjectRef, ...] = ()
    support: WorkPacketSupport = "raw_evidence"
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _requires_evidence_refs(self) -> RecoveryWorkPacketEntry:
        if not self.evidence_refs:
            raise ValueError("RecoveryWorkPacketEntry requires at least one evidence ref")
        return self


class RecoveryWorkPacket(ArchiveInsightModel):
    """Storage-free DTO for the compact continuation bundle."""

    session_id: str
    title: str
    source_origin: str
    message_count: int
    git_branch: str | None = None
    working_directories: tuple[str, ...] = ()
    target_refs: tuple[ObjectRef, ...] = ()
    entries: tuple[RecoveryWorkPacketEntry, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...]

    @model_validator(mode="after")
    def _requires_evidence_refs(self) -> RecoveryWorkPacket:
        if not self.evidence_refs:
            raise ValueError("RecoveryWorkPacket requires at least one evidence ref")
        return self

    def render_markdown(self) -> str:
        """Render the operator-facing continuation bundle."""

        return render_work_packet(self)


class RecoveryDigest(ArchiveInsightModel):
    """Typed v0 output for transform-first successor-session recovery."""

    session_id: str
    title: str | None = None
    git_branch: str | None = None
    working_directories: tuple[str, ...] = ()
    transform: TransformMetadata
    size_metrics: RecoverySizeMetrics
    role_counts: dict[str, int] = Field(default_factory=dict)
    tool_summaries: tuple[ToolSummary, ...] = ()
    subagent_reports: tuple[SubagentReport, ...] = ()
    run_state: RunStateSummary | None = None
    events: tuple[RecoveryEvent, ...] = ()
    decision_candidates: tuple[DecisionCandidate, ...] = ()
    run_projection: RunProjection
    forensic_index: ForensicIndex
    resume_markdown: str
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_session_ref(self) -> RecoveryDigest:
        if not self.raw_refs:
            raise ValueError("RecoveryDigest requires at least one raw ref")
        return self

    def report_markdown(self, preset: RecoveryReportPreset) -> str:
        """Render a deterministic evidence-linked recovery report preset."""

        return render_recovery_report(self, preset=preset)

    def work_packet(self) -> RecoveryWorkPacket:
        """Return the storage-free continuation packet DTO for this digest."""

        return build_recovery_work_packet(self)

    def work_packet_markdown(self) -> str:
        """Render the storage-free continuation packet DTO."""

        return self.work_packet().render_markdown()


class TransformDescriptor(ArchiveInsightModel):
    transform_id: str
    version: int
    input_kind: Literal["session"] = "session"
    output_kind: Literal["recovery_digest"] = "recovery_digest"
    deterministic: bool = True
    uses_llm: bool = False


RECOVERY_TRANSFORM = TransformDescriptor(
    transform_id=RECOVERY_TRANSFORM_ID,
    version=RECOVERY_TRANSFORM_VERSION,
)

TRANSFORM_REGISTRY: dict[str, TransformDescriptor] = {
    RECOVERY_TRANSFORM.transform_id: RECOVERY_TRANSFORM,
}


def compile_recovery_digest(
    session: Session,
    *,
    session_links: Sequence[Mapping[str, object]] = (),
) -> RecoveryDigest:
    """Compile a session into a small deterministic recovery/digest bundle."""

    messages = list(session.messages)
    session_ref = TransformRawRef(
        session_id=str(session.id),
        message_id=None,
        block_index=None,
        ref_kind="session",
        preview=session.display_title,
    )
    tool_summaries = tuple(_extract_tool_summaries(session, messages))
    subagent_reports = _enrich_subagent_reports_with_links(
        tuple(_extract_subagent_reports(session, messages)),
        session_links,
    )
    run_state = _extract_run_state(session, messages)
    events = tuple(_extract_events(session, messages))
    decisions = tuple(_extract_decision_candidates(session, messages))
    run_projection = build_run_projection(
        session_id=str(session.id),
        source_origin=str(session.origin),
        title=session.title,
        git_branch=session.git_branch,
        working_directories=tuple(session.working_directories),
        session_raw_refs=(session_ref,),
        tool_summaries=tool_summaries,
        subagent_reports=subagent_reports,
        recovery_events=events,
    )
    role_counts = dict(Counter(_role_value(message) for message in messages))
    normal_read = _normal_read_text(messages)
    raw_bytes = _session_raw_bytes(session, messages)
    resume_markdown = render_resume_bundle(
        session=session,
        tool_summaries=tool_summaries,
        subagent_reports=subagent_reports,
        run_state=run_state,
        events=events,
        decisions=decisions,
        run_projection=run_projection,
    )
    forensic_index = _build_forensic_index(
        session_id=str(session.id),
        session_ref=session_ref,
        tool_summaries=tool_summaries,
        subagent_reports=subagent_reports,
        run_state=run_state,
        events=events,
        decisions=decisions,
    )
    return RecoveryDigest(
        session_id=str(session.id),
        title=session.title,
        transform=TransformMetadata(
            transform_id=RECOVERY_TRANSFORM_ID,
            transform_version=RECOVERY_TRANSFORM_VERSION,
            input_session_id=str(session.id),
            source_origin=str(session.origin),
            input_message_count=len(messages),
        ),
        size_metrics=RecoverySizeMetrics(
            raw_bytes=raw_bytes,
            normal_read_bytes=len(normal_read.encode("utf-8")),
            resume_bundle_bytes=len(resume_markdown.encode("utf-8")),
            message_count=len(messages),
            tool_summary_count=len(tool_summaries),
            subagent_report_count=len(subagent_reports),
            run_state_count=1 if run_state is not None else 0,
            event_count=len(events),
            decision_candidate_count=len(decisions),
        ),
        role_counts=role_counts,
        git_branch=session.git_branch,
        working_directories=tuple(session.working_directories),
        tool_summaries=tool_summaries,
        subagent_reports=subagent_reports,
        run_state=run_state,
        events=events,
        decision_candidates=decisions,
        run_projection=run_projection,
        forensic_index=forensic_index,
        resume_markdown=resume_markdown,
        raw_refs=(session_ref,),
    )


def render_resume_bundle(
    *,
    session: Session,
    tool_summaries: Sequence[ToolSummary],
    subagent_reports: Sequence[SubagentReport],
    run_state: RunStateSummary | None,
    events: Sequence[RecoveryEvent],
    decisions: Sequence[DecisionCandidate],
    run_projection: RunProjection,
) -> str:
    """Render the small successor-session boot packet for a digest."""

    packet = RecoveryWorkPacket(
        session_id=str(session.id),
        title=session.display_title,
        source_origin=str(session.origin),
        message_count=len(session.messages),
        git_branch=session.git_branch,
        working_directories=tuple(session.working_directories),
        target_refs=_work_packet_target_refs(str(session.id), session.git_branch),
        entries=tuple(
            _work_packet_entries(
                events=events,
                subagent_reports=subagent_reports,
                run_state=run_state,
                tool_summaries=tool_summaries,
                decisions=decisions,
                run_projection=run_projection,
                packet_raw_refs=(
                    TransformRawRef(
                        session_id=str(session.id),
                        message_id=None,
                        block_index=None,
                        ref_kind="session",
                        preview=session.display_title,
                    ),
                ),
            )
        ),
        evidence_refs=(EvidenceRef(session_id=str(session.id)),),
    )
    return packet.render_markdown()


def render_work_packet(packet: RecoveryWorkPacket) -> str:
    """Render a storage-free recovery work packet."""

    lines = [
        f"# Resume: {packet.title}",
        "",
        f"- session_id: {packet.session_id}",
        f"- origin: {packet.source_origin}",
        f"- messages: {packet.message_count}",
    ]
    if packet.git_branch:
        lines.append(f"- branch: {packet.git_branch}")
    if packet.target_refs:
        lines.append(f"- refs: {', '.join(ref.format() for ref in packet.target_refs)}")
    if packet.working_directories:
        lines.append(f"- workdirs: {', '.join(packet.working_directories)}")
    lines.extend(["", "## Execution Projection"])
    execution_entries = _packet_section(packet, "execution")
    for entry in execution_entries[:20]:
        lines.append(_work_packet_line(entry))
        object_refs = _object_refs_line(entry)
        if object_refs:
            lines.append(f"  - refs: {object_refs}")
        detail = _packet_metadata_line(
            entry.metadata,
            keys=("role", "status", "harness", "boundary", "inheritance_mode", "delivery_state"),
        )
        if detail:
            lines.append(f"  - details: {detail}")
    if not execution_entries:
        lines.append("- none projected")
    lines.extend(["", "## Events"])
    event_entries = _packet_section(packet, "events")
    for entry in event_entries[:8]:
        lines.append(_work_packet_line(entry))
        object_refs = _object_refs_line(entry)
        if object_refs:
            lines.append(f"  - refs: {object_refs}")
        detail = _packet_metadata_line(
            entry.metadata,
            keys=("pr_refs", "review_refs", "issue_refs", "commit_refs", "test_evidence"),
        )
        if detail:
            lines.append(f"  - details: {detail}")
    if not event_entries:
        lines.append("- none extracted")
    lines.extend(["", "## Subagents"])
    subagent_entries = _packet_section(packet, "subagents")
    for entry in subagent_entries[:8]:
        prompt = f" — {entry.text}" if entry.text else ""
        lines.append(f"- {_support_marker(entry)} {entry.label}{prompt}")
        refs = _subagent_metadata_line(entry.metadata)
        if refs:
            lines.append(f"  - refs: {refs}")
        report = entry.metadata.get("report", "")
        if report:
            lines.append(f"  - report: {report}")
    if not subagent_entries:
        lines.append("- none extracted")
    lines.extend(["", "## Run State"])
    run_state_entries = _packet_section(packet, "run_state")
    if not run_state_entries:
        lines.append("- none extracted")
    else:
        lines.extend(_work_packet_line(entry) for entry in run_state_entries[:33])
    lines.extend(["", "## Tools"])
    tool_entries = _packet_section(packet, "tools")
    for entry in tool_entries[:8]:
        command = f" — {entry.text}" if entry.text else ""
        handler = entry.metadata.get("handler_kind", "generic")
        status = entry.metadata.get("status", "unknown")
        lines.append(f"- {_support_marker(entry)} {entry.label} [{handler}] ({status}){command}")
        object_refs = _object_refs_line(entry)
        if object_refs:
            lines.append(f"  - refs: {object_refs}")
        detail = _packet_metadata_line(
            entry.metadata,
            keys=("pr_refs", "issue_refs", "file_refs", "commit_refs", "test_evidence"),
        )
        if detail:
            lines.append(f"  - details: {detail}")
    if not tool_entries:
        lines.append("- none extracted")
    lines.extend(["", "## Candidate Decisions / Run State"])
    decision_entries = _packet_section(packet, "decisions")
    lines.extend(_work_packet_line(entry) for entry in decision_entries[:8])
    if not decision_entries:
        lines.append("- none extracted")
    assertion_entries = _packet_section(packet, "assertions")
    if assertion_entries:
        lines.extend(["", "## Assertion Claims"])
        lines.extend(_work_packet_line(entry) for entry in assertion_entries[:12])
    gap_entries = _packet_section(packet, "evidence_gaps")
    if gap_entries:
        lines.extend(["", "## Evidence Gaps"])
        lines.extend(_work_packet_line(entry) for entry in gap_entries[:8])
    lines.extend(["", "## Evidence"])
    lines.append("Every packet row carries evidence refs and a support marker.")
    return "\n".join(lines).strip() + "\n"


def build_recovery_work_packet(digest: RecoveryDigest) -> RecoveryWorkPacket:
    """Build the storage-free continuation packet DTO from a recovery digest."""

    return RecoveryWorkPacket(
        session_id=digest.session_id,
        title=digest.title or digest.session_id,
        source_origin=digest.transform.source_origin,
        message_count=digest.size_metrics.message_count,
        git_branch=digest.git_branch,
        working_directories=digest.working_directories,
        target_refs=_work_packet_target_refs(digest.session_id, digest.git_branch),
        entries=tuple(
            _work_packet_entries(
                events=digest.events,
                subagent_reports=digest.subagent_reports,
                run_state=digest.run_state,
                tool_summaries=digest.tool_summaries,
                decisions=digest.decision_candidates,
                run_projection=digest.run_projection,
                packet_raw_refs=digest.raw_refs,
            )
        ),
        evidence_refs=tuple(ref.to_evidence_ref() for ref in digest.raw_refs),
    )


def _packet_section(packet: RecoveryWorkPacket, section: WorkPacketSection) -> tuple[RecoveryWorkPacketEntry, ...]:
    return tuple(entry for entry in packet.entries if entry.section == section)


def _work_packet_target_refs(session_id: str, git_branch: str | None) -> tuple[ObjectRef, ...]:
    refs = [ObjectRef(kind="session", object_id=session_id)]
    if git_branch:
        refs.append(ObjectRef(kind="branch", object_id=git_branch))
    return tuple(refs)


def _support_marker(entry: RecoveryWorkPacketEntry) -> str:
    return f"[{entry.support.replace('_', '-')}]"


def _work_packet_line(entry: RecoveryWorkPacketEntry) -> str:
    return f"- {_support_marker(entry)} {entry.label}: {entry.text}"


def _object_refs_line(entry: RecoveryWorkPacketEntry) -> str:
    """Render packet object refs in their public string form."""

    return ", ".join(ref.format() for ref in entry.object_refs)


def _packet_metadata_line(metadata: Mapping[str, str], *, keys: Sequence[str]) -> str:
    parts = [f"{key}={metadata[key]}" for key in keys if metadata.get(key)]
    return "; ".join(parts)


def _work_packet_entries(
    *,
    events: Sequence[RecoveryEvent],
    subagent_reports: Sequence[SubagentReport],
    run_state: RunStateSummary | None,
    tool_summaries: Sequence[ToolSummary],
    decisions: Sequence[DecisionCandidate],
    run_projection: RunProjection,
    packet_raw_refs: Sequence[TransformRawRef],
) -> Iterable[RecoveryWorkPacketEntry]:
    packet_evidence_refs = _to_evidence_refs(packet_raw_refs)
    for run in run_projection.runs:
        metadata: dict[str, str] = {"role": run.role, "status": run.status, "harness": run.harness}
        if run.cwd:
            metadata["cwd"] = run.cwd
        if run.git_branch:
            metadata["branch"] = run.git_branch
        yield RecoveryWorkPacketEntry(
            section="execution",
            label="run",
            text=run.title or run.run_ref.object_id,
            metadata=metadata,
            object_refs=tuple(
                ref
                for ref in (
                    run.run_ref,
                    run.parent_run_ref,
                    run.context_snapshot_ref,
                    ObjectRef(kind="branch", object_id=run.git_branch) if run.git_branch else None,
                )
                if ref is not None
            ),
            evidence_refs=run.evidence_refs,
        )
    for snapshot in run_projection.context_snapshots:
        yield RecoveryWorkPacketEntry(
            section="execution",
            label="context_snapshot",
            text=snapshot.boundary,
            metadata={
                "boundary": snapshot.boundary,
                "inheritance_mode": snapshot.inheritance_mode,
                **snapshot.metadata,
            },
            object_refs=(snapshot.snapshot_ref, snapshot.run_ref, *snapshot.segment_refs),
            evidence_refs=snapshot.evidence_refs,
        )
    for observed in run_projection.events:
        yield RecoveryWorkPacketEntry(
            section="execution",
            label=observed.kind,
            text=observed.summary,
            metadata={"delivery_state": observed.delivery_state},
            object_refs=tuple(
                ref
                for ref in (
                    observed.event_ref,
                    observed.run_ref,
                    observed.subject_ref,
                    *observed.object_refs,
                )
                if ref is not None
            ),
            evidence_refs=observed.evidence_refs,
        )
    for event in events:
        yield RecoveryWorkPacketEntry(
            section="events",
            label=event.kind,
            text=event.summary,
            metadata=_event_packet_metadata(event),
            object_refs=_event_object_refs(event),
            evidence_refs=_to_evidence_refs(event.raw_refs),
        )
    for report in subagent_reports:
        metadata = _subagent_report_metadata(report)
        if report.final_report_preview:
            metadata["report"] = report.final_report_preview
        yield RecoveryWorkPacketEntry(
            section="subagents",
            label=report.subagent_type,
            text=report.prompt,
            metadata=metadata,
            evidence_refs=_to_evidence_refs(report.raw_refs),
        )
    if run_state is not None:
        if run_state.goal:
            yield RecoveryWorkPacketEntry(
                section="run_state",
                label="goal",
                text=run_state.goal,
                support="assertion",
                evidence_refs=_to_evidence_refs(run_state.raw_refs),
            )
        for label, items in (
            ("done", run_state.done),
            ("in_flight", run_state.in_flight),
            ("blocker", run_state.blockers),
            ("next", run_state.next_actions),
        ):
            for item in items[:8]:
                yield RecoveryWorkPacketEntry(
                    section="run_state",
                    label=label,
                    text=item,
                    support="caveat" if label == "blocker" else "assertion",
                    evidence_refs=_to_evidence_refs(run_state.raw_refs),
                )
    else:
        yield RecoveryWorkPacketEntry(
            section="evidence_gaps",
            label="run_state",
            text="No structured RunState section was extracted from the session.",
            support="missing_evidence",
            evidence_refs=packet_evidence_refs,
        )
    for tool in tool_summaries:
        yield RecoveryWorkPacketEntry(
            section="tools",
            label=tool.tool_name,
            text=tool.command or "",
            metadata=_tool_packet_metadata(tool),
            object_refs=_tool_object_refs(tool),
            evidence_refs=_to_evidence_refs(tool.raw_refs),
        )
    for decision in decisions:
        yield RecoveryWorkPacketEntry(
            section="decisions",
            label=decision.kind,
            text=decision.text,
            support="inference",
            evidence_refs=_to_evidence_refs(decision.raw_refs),
        )
    if not events:
        yield RecoveryWorkPacketEntry(
            section="evidence_gaps",
            label="events",
            text="No PR, issue, test, check, or merge events were extracted.",
            support="missing_evidence",
            evidence_refs=packet_evidence_refs,
        )
    if not subagent_reports:
        yield RecoveryWorkPacketEntry(
            section="evidence_gaps",
            label="subagents",
            text="No subagent handoff or child-session evidence was extracted.",
            support="missing_evidence",
            evidence_refs=packet_evidence_refs,
        )
    if not tool_summaries:
        yield RecoveryWorkPacketEntry(
            section="evidence_gaps",
            label="tools",
            text="No tool execution summary was extracted.",
            support="missing_evidence",
            evidence_refs=packet_evidence_refs,
        )
    if not decisions:
        yield RecoveryWorkPacketEntry(
            section="evidence_gaps",
            label="decisions",
            text="No candidate decision statement was extracted.",
            support="missing_evidence",
            evidence_refs=packet_evidence_refs,
        )


def _to_evidence_refs(refs: Sequence[TransformRawRef]) -> tuple[EvidenceRef, ...]:
    return tuple(ref.to_evidence_ref() for ref in refs)


def _event_packet_metadata(event: RecoveryEvent) -> dict[str, str]:
    metadata: dict[str, str] = {}
    pr_refs = tuple(_number_refs(_PR_RE, event.summary)) if event.kind.startswith("pr_") else ()
    review_refs = tuple(_number_refs(_PR_RE, event.summary)) if event.kind.startswith("review_") else ()
    issue_refs = tuple(_number_refs(_ISSUE_RE, event.summary)) if event.kind == "issue_closed" else ()
    test_evidence = (event.summary,) if event.kind.startswith(("check_", "test_")) else ()
    if pr_refs:
        metadata["pr_refs"] = ", ".join(pr_refs)
    if review_refs:
        metadata["review_refs"] = ", ".join(review_refs)
    if issue_refs:
        metadata["issue_refs"] = ", ".join(issue_refs)
    if test_evidence:
        metadata["test_evidence"] = " | ".join(test_evidence)
    return metadata


def _event_object_refs(event: RecoveryEvent) -> tuple[ObjectRef, ...]:
    """Return typed refs for addressable event targets extracted from summaries."""

    if event.kind.startswith("pr_"):
        return _github_object_refs("github-pr", _number_refs(_PR_RE, event.summary))
    if event.kind.startswith("review_"):
        return _github_object_refs("github-review", _number_refs(_PR_RE, event.summary))
    if event.kind == "issue_closed":
        return _github_object_refs("github-issue", _number_refs(_ISSUE_RE, event.summary))
    if event.kind.startswith("check_"):
        return _check_run_object_refs(event.summary)
    return ()


def _tool_packet_metadata(tool: ToolSummary) -> dict[str, str]:
    metadata: dict[str, str] = {"handler_kind": tool.handler_kind, "status": tool.status}
    issue_refs = _refs_without_pr_refs(tool.issue_refs, tool.pr_refs)
    if tool.pr_refs:
        metadata["pr_refs"] = ", ".join(tool.pr_refs)
    if issue_refs:
        metadata["issue_refs"] = ", ".join(issue_refs)
    if tool.file_refs:
        metadata["file_refs"] = ", ".join(tool.file_refs)
    if tool.commit_refs:
        metadata["commit_refs"] = ", ".join(tool.commit_refs)
    if tool.test_evidence:
        metadata["test_evidence"] = " | ".join(tool.test_evidence)
    return metadata


def _tool_object_refs(tool: ToolSummary) -> tuple[ObjectRef, ...]:
    """Return typed refs for PR/issue/file evidence extracted from a tool row."""

    issue_refs = _refs_without_pr_refs(tool.issue_refs, tool.pr_refs)
    return (
        *_github_object_refs("github-pr", tool.pr_refs),
        *_github_object_refs("github-issue", issue_refs),
        *(ObjectRef(kind="file", object_id=ref) for ref in tool.file_refs),
        *(ObjectRef(kind="commit", object_id=ref) for ref in tool.commit_refs),
    )


def _github_object_refs(
    kind: Literal["github-issue", "github-pr", "github-review"], refs: Iterable[str]
) -> tuple[ObjectRef, ...]:
    """Format GitHub number refs as opaque public object refs."""

    return tuple(ObjectRef(kind=kind, object_id=ref) for ref in refs)


def _check_run_object_refs(summary: str) -> tuple[ObjectRef, ...]:
    """Format a named check event summary as a public check-run ref."""

    name = summary.removesuffix(" passed").removesuffix(" failed").strip()
    if not name:
        return ()
    return (ObjectRef(kind="check-run", object_id=name),)


def _refs_without_pr_refs(issue_refs: Sequence[str], pr_refs: Sequence[str]) -> tuple[str, ...]:
    pr_ref_set = set(pr_refs)
    return tuple(ref for ref in issue_refs if ref not in pr_ref_set)


def render_recovery_report(digest: RecoveryDigest, *, preset: RecoveryReportPreset) -> str:
    """Render a deterministic recovery report preset from a recovery digest."""

    if preset == "continue":
        return _render_continue_report(digest)
    if preset == "blame":
        return _render_blame_report(digest)
    if preset == "work-packet":
        return digest.work_packet_markdown()
    raise ValueError(f"unsupported recovery report preset: {preset}")


def _render_continue_report(digest: RecoveryDigest) -> str:
    session_refs = digest.raw_refs
    lines = [
        f"# Continue: {digest.title or digest.session_id}{_evidence_suffix(digest, session_refs)}",
        "",
        "## Session",
        f"- session_id: {digest.session_id}{_evidence_suffix(digest, session_refs)}",
        f"- origin: {digest.transform.source_origin}{_evidence_suffix(digest, session_refs)}",
        f"- messages: {digest.size_metrics.message_count}{_evidence_suffix(digest, session_refs)}",
    ]
    lines.extend(["", "## Boot Packet"])
    if digest.run_state is None:
        lines.append(f"- run_state: none extracted{_evidence_suffix(digest, session_refs)}")
    else:
        if digest.run_state.goal:
            lines.append(f"- goal: {digest.run_state.goal}{_evidence_suffix(digest, digest.run_state.raw_refs)}")
        lines.extend(
            f"- done: {item}{_evidence_suffix(digest, digest.run_state.raw_refs)}" for item in digest.run_state.done[:8]
        )
        lines.extend(
            f"- in_flight: {item}{_evidence_suffix(digest, digest.run_state.raw_refs)}"
            for item in digest.run_state.in_flight[:8]
        )
        lines.extend(
            f"- blocker: {item}{_evidence_suffix(digest, digest.run_state.raw_refs)}"
            for item in digest.run_state.blockers[:8]
        )
        lines.extend(
            f"- next: {item}{_evidence_suffix(digest, digest.run_state.raw_refs)}"
            for item in digest.run_state.next_actions[:8]
        )
    lines.extend(["", "## Recent Events"])
    if digest.events:
        lines.extend(
            f"- {event.kind}: {event.summary}{_evidence_suffix(digest, event.raw_refs)}" for event in digest.events[:10]
        )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Subagent Reports"])
    if digest.subagent_reports:
        for report in digest.subagent_reports[:8]:
            prompt = f" — {report.prompt}" if report.prompt else ""
            lines.append(
                f"- {report.subagent_type}{_subagent_link_suffix(report)}{prompt}"
                f"{_evidence_suffix(digest, report.raw_refs)}"
            )
            if report.final_report_preview:
                lines.append(f"  report: {report.final_report_preview}{_evidence_suffix(digest, report.raw_refs)}")
            for caveat in report.caveats[:4]:
                lines.append(f"  caveat: {caveat}{_evidence_suffix(digest, report.raw_refs)}")
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Useful Tools"])
    if digest.tool_summaries:
        for tool in digest.tool_summaries[:10]:
            command = f" — {tool.command}" if tool.command else ""
            lines.append(
                f"- {tool.tool_name} [{tool.handler_kind}] ({tool.status}){command}"
                f"{_evidence_suffix(digest, tool.raw_refs)}"
            )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Candidate Decisions"])
    if digest.decision_candidates:
        lines.extend(
            f"- {candidate.kind}: {candidate.text}{_evidence_suffix(digest, candidate.raw_refs)}"
            for candidate in digest.decision_candidates[:10]
        )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Evidence Index"])
    lines.extend(_render_evidence_index_lines(digest, limit=12))
    return "\n".join(lines).strip() + "\n"


def _render_blame_report(digest: RecoveryDigest) -> str:
    session_refs = digest.raw_refs
    lines = [
        f"# Blame: {digest.title or digest.session_id}{_evidence_suffix(digest, session_refs)}",
        "",
        "## Forensic Summary",
        f"- session_id: {digest.session_id}{_evidence_suffix(digest, session_refs)}",
        f"- transform: {digest.transform.transform_id} v{digest.transform.transform_version}"
        f"{_evidence_suffix(digest, session_refs)}",
        f"- extracted_claims: {digest.forensic_index.claim_count}{_evidence_suffix(digest, session_refs)}",
        f"- evidence_locations: {len(digest.forensic_index.entries)}{_evidence_suffix(digest, session_refs)}",
        "",
        "## Checks And Tests",
    ]
    check_events = [
        event for event in digest.events if event.kind in {"check_passed", "check_failed", "test_passed", "test_failed"}
    ]
    if check_events:
        lines.extend(
            f"- {event.kind}: {event.summary}{_evidence_suffix(digest, event.raw_refs)}" for event in check_events
        )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## GitHub Events"])
    github_events = [event for event in digest.events if event.kind in {"pr_opened", "pr_merged", "issue_closed"}]
    if github_events:
        lines.extend(
            f"- {event.kind}: {event.summary}{_evidence_suffix(digest, event.raw_refs)}" for event in github_events
        )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Tool Envelopes"])
    if digest.tool_summaries:
        for tool in digest.tool_summaries:
            command = f" — {tool.command}" if tool.command else ""
            lines.append(
                f"- {tool.tool_name} [{tool.handler_kind}] status={tool.status} lines={tool.line_count}{command}"
                f"{_evidence_suffix(digest, tool.raw_refs)}"
            )
            if tool.output_preview:
                lines.append(f"  output: {tool.output_preview}{_evidence_suffix(digest, tool.raw_refs)}")
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Subagent Evidence"])
    if digest.subagent_reports:
        for report in digest.subagent_reports:
            lines.append(
                f"- {report.subagent_type}{_subagent_link_suffix(report)}: "
                f"{report.final_report_preview or report.prompt}"
                f"{_evidence_suffix(digest, report.raw_refs)}"
            )
            for caveat in report.caveats:
                lines.append(f"  caveat: {caveat}{_evidence_suffix(digest, report.raw_refs)}")
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Decision Candidates"])
    if digest.decision_candidates:
        lines.extend(
            f"- {candidate.kind}: {candidate.text}{_evidence_suffix(digest, candidate.raw_refs)}"
            for candidate in digest.decision_candidates
        )
    else:
        lines.append(f"- none extracted{_evidence_suffix(digest, session_refs)}")
    lines.extend(["", "## Evidence Timeline"])
    lines.extend(_render_evidence_index_lines(digest, limit=None))
    return "\n".join(lines).strip() + "\n"


def _render_evidence_index_lines(digest: RecoveryDigest, *, limit: int | None) -> list[str]:
    entries = digest.forensic_index.entries if limit is None else digest.forensic_index.entries[:limit]
    if not entries:
        return [f"- none extracted{_evidence_suffix(digest, digest.raw_refs)}"]
    lines = []
    for entry in entries:
        location = _raw_location(entry.raw_ref)
        preview = f" preview={entry.raw_ref.preview!r}" if entry.raw_ref.preview else ""
        lines.append(
            f"- evidence_id: {entry.evidence_id}; raw: {location}; claims: {', '.join(entry.claim_labels)}"
            f"{preview}{_evidence_suffix(digest, (entry.raw_ref,))}"
        )
    if limit is not None and len(digest.forensic_index.entries) > limit:
        lines.append(
            f"- truncated: {len(digest.forensic_index.entries) - limit} more{_evidence_suffix(digest, digest.raw_refs)}"
        )
    return lines


def _raw_location(ref: TransformRawRef) -> str:
    parts: list[str] = [ref.ref_kind]
    if ref.message_id is not None:
        parts.append(f"message={ref.message_id}")
    if ref.block_index is not None:
        parts.append(f"block={ref.block_index}")
    return " ".join(parts)


def _evidence_suffix(digest: RecoveryDigest, refs: Sequence[TransformRawRef]) -> str:
    evidence_ids = _evidence_ids_for_refs(digest, refs)
    if not evidence_ids:
        evidence_ids = tuple(_evidence_id(ref) for ref in refs)
    return f" [evidence: {', '.join(evidence_ids)}]"


def _evidence_ids_for_refs(digest: RecoveryDigest, refs: Sequence[TransformRawRef]) -> tuple[str, ...]:
    evidence_ids_by_key = {_raw_ref_key(entry.raw_ref): entry.evidence_id for entry in digest.forensic_index.entries}
    result: list[str] = []
    for ref in refs:
        evidence_id = evidence_ids_by_key.get(_raw_ref_key(ref), _evidence_id(ref))
        _append_unique(result, evidence_id)
    return tuple(result)


def _build_forensic_index(
    *,
    session_id: str,
    session_ref: TransformRawRef,
    tool_summaries: Sequence[ToolSummary],
    subagent_reports: Sequence[SubagentReport],
    run_state: RunStateSummary | None,
    events: Sequence[RecoveryEvent],
    decisions: Sequence[DecisionCandidate],
) -> ForensicIndex:
    refs_by_key: dict[tuple[str, str | None, int | None, str], TransformRawRef] = {}
    kinds_by_key: dict[tuple[str, str | None, int | None, str], list[ForensicClaimKind]] = {}
    labels_by_key: dict[tuple[str, str | None, int | None, str], list[str]] = {}
    claim_count = 0

    def add(kind: ForensicClaimKind, label: str, refs: Sequence[TransformRawRef]) -> None:
        nonlocal claim_count
        claim_count += 1
        for ref in refs:
            key = _raw_ref_key(ref)
            refs_by_key.setdefault(key, ref)
            _append_unique(kinds_by_key.setdefault(key, []), kind)
            _append_unique(labels_by_key.setdefault(key, []), label)

    add("digest", "digest:session", (session_ref,))
    for tool in tool_summaries:
        label = f"tool:{tool.tool_name}"
        if tool.tool_id:
            label = f"{label}:{tool.tool_id}"
        add("tool_summary", label, tool.raw_refs)
    for index, report in enumerate(subagent_reports):
        label = f"subagent:{report.subagent_type}:{index}"
        add("subagent_report", label, report.raw_refs)
    if run_state is not None:
        add("run_state", "run_state", run_state.raw_refs)
    for index, event in enumerate(events):
        add("event", f"event:{event.kind}:{index}", event.raw_refs)
    for index, decision in enumerate(decisions):
        add("decision_candidate", f"decision:{decision.kind}:{index}", decision.raw_refs)

    entries = tuple(
        ForensicIndexEntry(
            evidence_id=_evidence_id(ref),
            raw_ref=ref,
            claim_kinds=tuple(kinds_by_key[key]),
            claim_labels=tuple(labels_by_key[key]),
        )
        for key, ref in sorted(refs_by_key.items(), key=lambda item: _raw_ref_sort_key(item[1]))
    )
    return ForensicIndex(session_id=session_id, entries=entries, claim_count=claim_count)


def _raw_ref_key(ref: TransformRawRef) -> tuple[str, str | None, int | None, str]:
    return (ref.session_id, ref.message_id, ref.block_index, ref.ref_kind)


def _raw_ref_sort_key(ref: TransformRawRef) -> tuple[str, str, int, str]:
    return (ref.session_id, ref.message_id or "", -1 if ref.block_index is None else ref.block_index, ref.ref_kind)


def _evidence_id(ref: TransformRawRef) -> str:
    return ref.to_evidence_ref().format()


def _append_unique(target: list[T], value: T) -> None:
    if value not in target:
        target.append(value)


def _extract_tool_summaries(session: Session, messages: Sequence[Message]) -> Iterable[ToolSummary]:
    result_by_tool_id: dict[str, tuple[dict[str, object], TransformRawRef]] = {}
    for message in messages:
        for index, block in enumerate(message.blocks):
            if str(block.get("type") or "") != "tool_result":
                continue
            tool_id = _optional_text(block.get("tool_id") or block.get("id"))
            if tool_id:
                result_by_tool_id[tool_id] = (block, _block_ref(session, message, index, block))

    for message in messages:
        for index, block in enumerate(message.blocks):
            if str(block.get("type") or "") != "tool_use":
                continue
            if _is_subagent_tool(block):
                continue
            tool_id = _optional_text(block.get("id") or block.get("tool_id"))
            result_block: dict[str, object] | None = None
            result_ref: TransformRawRef | None = None
            if tool_id and tool_id in result_by_tool_id:
                result_block, result_ref = result_by_tool_id[tool_id]
            refs = [_block_ref(session, message, index, block)]
            if result_ref is not None:
                refs.append(result_ref)
            output_text = _block_text(result_block or {})
            tool_name = _tool_name(block)
            command = _tool_command(block)
            yield ToolSummary(
                tool_name=tool_name,
                tool_id=tool_id,
                command=command,
                handler_kind=_tool_handler_kind(tool_name=tool_name, command=command, output_text=output_text),
                status=_tool_status(output_text),
                line_count=_line_count(output_text),
                output_preview=_preview(output_text),
                pr_refs=tuple(_number_refs(_PR_RE, output_text)),
                issue_refs=tuple(_number_refs(_ISSUE_RE, output_text)),
                test_evidence=tuple(_test_evidence(output_text)),
                file_refs=tuple(_tool_file_refs(tool_name=tool_name, command=command, output_text=output_text)),
                commit_refs=tuple(_tool_commit_refs(command=command, output_text=output_text)),
                raw_refs=tuple(refs),
            )


def _extract_subagent_reports(session: Session, messages: Sequence[Message]) -> Iterable[SubagentReport]:
    result_by_tool_id: dict[str, tuple[dict[str, object], TransformRawRef]] = {}
    for message in messages:
        for index, block in enumerate(message.blocks):
            if str(block.get("type") or "") != "tool_result":
                continue
            tool_id = _optional_text(block.get("tool_id") or block.get("id"))
            if tool_id:
                result_by_tool_id[tool_id] = (block, _block_ref(session, message, index, block))

    for message in messages:
        for index, block in enumerate(message.blocks):
            if str(block.get("type") or "") != "tool_use":
                continue
            if not _is_subagent_tool(block):
                continue
            tool_id = _optional_text(block.get("id") or block.get("tool_id"))
            result_block: dict[str, object] | None = None
            result_ref: TransformRawRef | None = None
            if tool_id and tool_id in result_by_tool_id:
                result_block, result_ref = result_by_tool_id[tool_id]
            refs = [_block_ref(session, message, index, block)]
            if result_ref is not None:
                refs.append(result_ref)
            result_text = _block_text(result_block or {})
            yield SubagentReport(
                subagent_type=_subagent_type(block),
                tool_id=tool_id,
                task_id=_subagent_task_id(block),
                child_session_id=_subagent_child_session_id(block, result_text),
                prompt=_subagent_prompt(block),
                final_report_preview=_preview(result_text, limit=320),
                pr_refs=tuple(_number_refs(_PR_RE, result_text)),
                issue_refs=tuple(_number_refs(_ISSUE_RE, result_text)),
                test_evidence=tuple(_test_evidence(result_text)),
                caveats=tuple(_caveats(result_text)),
                raw_refs=tuple(refs),
            )


def _enrich_subagent_reports_with_links(
    reports: Sequence[SubagentReport],
    session_links: Sequence[Mapping[str, object]],
) -> tuple[SubagentReport, ...]:
    """Attach topology/session-link status to extracted subagent reports."""

    if not session_links:
        return tuple(reports)

    links_by_child_ref: dict[str, Mapping[str, object]] = {}
    for session_link in session_links:
        for key in _session_link_child_keys(session_link):
            links_by_child_ref.setdefault(key, session_link)

    enriched: list[SubagentReport] = []
    for report in reports:
        link: Mapping[str, object] | None = links_by_child_ref.get(report.child_session_id or "")
        if link is None:
            enriched.append(report)
            continue
        status = _optional_text(link.get("status"))
        enriched.append(
            report.model_copy(
                update={
                    "child_link_status": status
                    if status in {"resolved", "unresolved", "repaired", "quarantined"}
                    else None,
                    "child_link_type": _optional_text(link.get("link_type")),
                    "resolved_child_session_id": _optional_text(link.get("resolved_dst_session_id")),
                }
            )
        )
    return tuple(enriched)


def _session_link_child_keys(link: Mapping[str, object]) -> tuple[str, ...]:
    """Return all child-reference spellings that can identify one link row."""

    keys: list[str] = []
    resolved = _optional_text(link.get("resolved_dst_session_id"))
    if resolved:
        keys.append(resolved)
    native = _optional_text(link.get("dst_native_id"))
    if native:
        keys.append(native)
    origin = _optional_text(link.get("dst_origin"))
    if origin and native:
        keys.append(f"{origin}:{native}")
    return tuple(keys)


def _extract_events(session: Session, messages: Sequence[Message]) -> Iterable[RecoveryEvent]:
    seen: set[tuple[str, str]] = set()
    for message in messages:
        message_ref = _message_ref(session, message)
        for text in _message_text_fragments(message):
            for event in _events_from_text(text, message_ref):
                key = (event.kind, event.summary)
                if key in seen:
                    continue
                seen.add(key)
                yield event


def _extract_run_state(session: Session, messages: Sequence[Message]) -> RunStateSummary | None:
    goal: str | None = None
    done: list[str] = []
    in_flight: list[str] = []
    blockers: list[str] = []
    next_actions: list[str] = []
    refs: list[TransformRawRef] = []

    for message in messages:
        parsed = _parse_run_state_text(message.text or "")
        if parsed is None:
            continue
        parsed_goal, parsed_done, parsed_in_flight, parsed_blockers, parsed_next = parsed
        if parsed_goal:
            goal = parsed_goal
        _extend_unique(done, parsed_done)
        _extend_unique(in_flight, parsed_in_flight)
        _extend_unique(blockers, parsed_blockers)
        _extend_unique(next_actions, parsed_next)
        refs.append(_message_ref(session, message))

    if not refs:
        return None
    return RunStateSummary(
        goal=goal,
        done=tuple(done),
        in_flight=tuple(in_flight),
        blockers=tuple(blockers),
        next_actions=tuple(next_actions),
        raw_refs=tuple(refs),
    )


def _parse_run_state_text(
    text: str,
) -> tuple[str | None, list[str], list[str], list[str], list[str]] | None:
    goal: str | None = None
    done: list[str] = []
    in_flight: list[str] = []
    blockers: list[str] = []
    next_actions: list[str] = []
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            current_section = None
            continue
        heading = _RUNSTATE_SECTION_RE.match(line)
        if heading is not None:
            current_section = _run_state_section_key(heading.group(1))
            value = _clean_run_state_item(heading.group("text"))
            if value:
                if current_section == "goal":
                    goal = value
                elif current_section == "done":
                    done.append(value)
                elif current_section == "in_flight":
                    in_flight.append(value)
                elif current_section == "blockers":
                    blockers.append(value)
                elif current_section == "next":
                    next_actions.append(value)
            continue
        if current_section is None:
            continue
        value = _clean_run_state_item(line)
        if not value:
            continue
        if current_section == "goal":
            goal = value if goal is None else f"{goal}; {value}"
        elif current_section == "done":
            done.append(value)
        elif current_section == "in_flight":
            in_flight.append(value)
        elif current_section == "blockers":
            blockers.append(value)
        elif current_section == "next":
            next_actions.append(value)

    if not any((goal, done, in_flight, blockers, next_actions)):
        return None
    return goal, done, in_flight, blockers, next_actions


def _run_state_section_key(value: str) -> Literal["goal", "done", "in_flight", "blockers", "next"]:
    normalized = value.lower().strip()
    if normalized == "goal":
        return "goal"
    if normalized == "done":
        return "done"
    if normalized == "in flight":
        return "in_flight"
    if normalized.startswith("blocker"):
        return "blockers"
    return "next"


def _clean_run_state_item(value: str) -> str:
    return _preview(value.strip().lstrip("-*").strip(), limit=240)


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    seen = set(target)
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        target.append(value)


def _events_from_text(text: str, ref: TransformRawRef) -> Iterable[RecoveryEvent]:
    for match in _MERGED_RE.finditer(text):
        number = match.group("number")
        yield RecoveryEvent(kind="pr_merged", summary=f"PR #{number} merged", raw_refs=(ref,))
    for line in text.splitlines():
        lowered = line.lower()
        created_pr_match = _CREATED_PR_RE.search(line)
        if created_pr_match is not None:
            yield RecoveryEvent(
                kind="pr_opened",
                summary=f"PR #{created_pr_match.group('number')} opened",
                raw_refs=(ref,),
            )
        review_match = _REVIEW_PR_RE.search(line)
        if review_match is not None:
            number = review_match.group("number")
            review_kind: Literal[
                "review_posted",
                "review_seen_by_tool",
                "review_injected_context",
                "review_acknowledged",
                "review_acted_on",
            ] = "review_posted"
            if any(term in lowered for term in ("addressed", "acted on", "fixed", "resolved")):
                review_kind = "review_acted_on"
            elif any(term in lowered for term in ("acknowledged", "classified", "triaged")):
                review_kind = "review_acknowledged"
            elif any(term in lowered for term in ("injected", "context")):
                review_kind = "review_injected_context"
            elif any(term in lowered for term in ("read", "fetched", "inspected", "seen")):
                review_kind = "review_seen_by_tool"
            review_label = {
                "review_posted": "posted",
                "review_seen_by_tool": "seen by tool",
                "review_injected_context": "injected into context",
                "review_acknowledged": "acknowledged",
                "review_acted_on": "acted on",
            }[review_kind]
            yield RecoveryEvent(kind=review_kind, summary=f"Review on PR #{number} {review_label}", raw_refs=(ref,))
            continue
        if "pull/" in lowered or "pr " in lowered:
            pr_match = _PR_RE.search(line)
            if pr_match is not None:
                number = pr_match.group("number")
                kind: Literal["pr_opened", "pr_merged"] = "pr_merged" if "merge" in lowered else "pr_opened"
                yield RecoveryEvent(
                    kind=kind, summary=f"PR #{number} {'merged' if kind == 'pr_merged' else 'opened'}", raw_refs=(ref,)
                )
        if "closed issue" in lowered or "closed #" in lowered:
            issue_match = _ISSUE_RE.search(line)
            if issue_match is not None:
                yield RecoveryEvent(
                    kind="issue_closed",
                    summary=f"Issue #{issue_match.group('number')} closed",
                    raw_refs=(ref,),
                )
        pass_match = _TEST_PASS_RE.search(line)
        if pass_match:
            yield RecoveryEvent(
                kind="test_passed",
                summary=f"{pass_match.group('count')} tests passed",
                raw_refs=(ref,),
            )
        fail_match = _TEST_FAIL_RE.search(line)
        if fail_match:
            yield RecoveryEvent(
                kind="test_failed",
                summary=f"{fail_match.group('count')} tests failed",
                raw_refs=(ref,),
            )
        check_match = _CHECK_PASS_RE.search(line)
        if check_match:
            yield RecoveryEvent(
                kind="check_passed",
                summary=f"{check_match.group('name').strip()} passed",
                raw_refs=(ref,),
            )
        failed_check_match = _CHECK_FAIL_RE.search(line)
        if failed_check_match:
            yield RecoveryEvent(
                kind="check_failed",
                summary=f"{failed_check_match.group('name').strip()} failed",
                raw_refs=(ref,),
            )


def _extract_decision_candidates(session: Session, messages: Sequence[Message]) -> Iterable[DecisionCandidate]:
    seen: set[tuple[str, str]] = set()
    for message in messages:
        ref = _message_ref(session, message)
        for line in (message.text or "").splitlines():
            decision = _DECISION_RE.search(line)
            if decision:
                item = DecisionCandidate(
                    kind="decision", text=_preview(decision.group("text"), limit=240), raw_refs=(ref,)
                )
                key = (item.kind, item.text)
                if key not in seen:
                    seen.add(key)
                    yield item
            status = _STATUS_HEADING_RE.search(line)
            if status:
                text = f"{status.group(1).lower()}: {_preview(status.group('text'), limit=240)}"
                item = DecisionCandidate(kind="run_state", text=text, raw_refs=(ref,))
                key = (item.kind, item.text)
                if key not in seen:
                    seen.add(key)
                    yield item


def _session_raw_bytes(session: Session, messages: Sequence[Message]) -> int:
    payload = {
        "id": str(session.id),
        "origin": str(session.origin),
        "title": session.title,
        "metadata": session.metadata,
        "messages": [
            {
                "id": message.id,
                "role": _role_value(message),
                "text": message.text,
                "blocks": message.blocks,
            }
            for message in messages
        ],
    }
    return len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


def _normal_read_text(messages: Sequence[Message]) -> str:
    return "\n\n".join(f"{_role_value(message)}: {message.text or ''}" for message in messages)


def _message_text_fragments(message: Message) -> Iterable[str]:
    if message.text:
        yield message.text
    for block in message.blocks:
        text = _block_text(block)
        if text:
            yield text


def _block_ref(session: Session, message: Message, block_index: int, block: Mapping[str, object]) -> TransformRawRef:
    return TransformRawRef(
        session_id=str(session.id),
        message_id=str(message.id),
        block_index=block_index,
        ref_kind="block",
        preview=_preview(_block_text(block) or _tool_command(block) or _tool_name(block)),
    )


def _message_ref(session: Session, message: Message) -> TransformRawRef:
    return TransformRawRef(
        session_id=str(session.id),
        message_id=str(message.id),
        ref_kind="message",
        preview=_preview(message.text or ""),
    )


def _role_value(message: Message) -> str:
    role = message.role
    return str(getattr(role, "value", role))


def _tool_name(block: Mapping[str, object]) -> str:
    return _optional_text(block.get("name") or block.get("tool_name") or block.get("tool")) or "unknown"


def _tool_command(block: Mapping[str, object]) -> str | None:
    candidates: list[object] = [block.get("command"), block.get("cmd")]
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        candidates.extend(
            [
                tool_input.get("command"),
                tool_input.get("cmd"),
                tool_input.get("file_path"),
                tool_input.get("path"),
            ]
        )
    for candidate in candidates:
        text = _optional_text(candidate)
        if text:
            return text
    return None


def _is_subagent_tool(block: Mapping[str, object]) -> bool:
    name = _tool_name(block).lower()
    if name == "task":
        return True
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        return any(key in tool_input for key in ("subagent_type", "agent_type"))
    return False


def _subagent_type(block: Mapping[str, object]) -> str:
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        value = _optional_text(
            tool_input.get("subagent_type") or tool_input.get("agent_type") or tool_input.get("description")
        )
        if value:
            return value
    return "unknown"


def _subagent_prompt(block: Mapping[str, object]) -> str:
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        value = _optional_text(tool_input.get("prompt") or tool_input.get("task"))
        if value:
            return _preview(value, limit=240)
    return ""


def _subagent_task_id(block: Mapping[str, object]) -> str | None:
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        return _optional_text(tool_input.get("taskId") or tool_input.get("task_id"))
    return None


def _subagent_child_session_id(block: Mapping[str, object], result_text: str) -> str | None:
    tool_input = block.get("tool_input") or block.get("input")
    if isinstance(tool_input, Mapping):
        for key in ("child_session_id", "subagent_session_id", "agent_session_id", "sessionId", "session_id"):
            value = _optional_text(tool_input.get(key))
            if value:
                return value
    return _subagent_child_session_id_from_text(result_text)


def _subagent_child_session_id_from_text(text: str) -> str | None:
    for line in text.splitlines():
        if "session" not in line.lower():
            continue
        for token in line.replace("=", " ").split():
            cleaned = token.strip("`'\"(),")
            if cleaned.startswith(("claude-code-session:", "codex-session:", "gemini-cli-session:")):
                return cleaned
    return None


def _subagent_report_metadata(report: SubagentReport) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if report.tool_id:
        metadata["tool_id"] = report.tool_id
    if report.task_id:
        metadata["task_id"] = report.task_id
    if report.child_session_id:
        metadata["child_session_id"] = report.child_session_id
    if report.child_link_status:
        metadata["child_link_status"] = report.child_link_status
    if report.child_link_type:
        metadata["child_link_type"] = report.child_link_type
    if report.resolved_child_session_id:
        metadata["resolved_child_session_id"] = report.resolved_child_session_id
    return metadata


def _subagent_metadata_line(metadata: Mapping[str, str]) -> str:
    parts = [
        f"{key}={metadata[key]}"
        for key in (
            "tool_id",
            "task_id",
            "child_session_id",
            "child_link_status",
            "child_link_type",
            "resolved_child_session_id",
        )
        if metadata.get(key)
    ]
    return ", ".join(parts)


def _subagent_link_suffix(report: SubagentReport) -> str:
    refs = _subagent_metadata_line(_subagent_report_metadata(report))
    return f" [{refs}]" if refs else ""


def _number_refs(pattern: re.Pattern[str], text: str) -> Iterable[str]:
    seen: set[str] = set()
    for match in pattern.finditer(text):
        ref = f"#{match.group('number')}"
        if ref in seen:
            continue
        seen.add(ref)
        yield ref


def _test_evidence(text: str) -> Iterable[str]:
    for line in text.splitlines():
        if _TEST_PASS_RE.search(line) or _TEST_FAIL_RE.search(line) or _CHECK_PASS_RE.search(line):
            yield _preview(line, limit=180)


def _caveats(text: str) -> Iterable[str]:
    for line in text.splitlines():
        lowered = line.lower()
        if "caveat" in lowered or "blocker" in lowered or "not included" in lowered:
            yield _preview(line, limit=180)


def _tool_status(output_text: str) -> Literal["ok", "failed", "unknown"]:
    lowered = output_text.lower()
    if not lowered:
        return "unknown"
    if "exit code 0" in lowered or " passed" in lowered or "\nok" in lowered:
        return "ok"
    if "exit code 1" in lowered or "failed" in lowered or "traceback" in lowered:
        return "failed"
    return "unknown"


def _tool_handler_kind(
    *,
    tool_name: str,
    command: str | None,
    output_text: str,
) -> Literal["shell", "file_read", "github", "git", "test", "generic"]:
    name = tool_name.lower()
    command_text = command or ""
    command_lower = command_text.lower()
    if name in {"read", "read_file"}:
        return "file_read"
    if command_lower.startswith("git "):
        return "git"
    if command_lower.startswith("gh "):
        return "github"
    if _looks_like_test_output(command_lower, output_text):
        return "test"
    if name in {"bash", "shell", "exec_command"}:
        return "shell"
    return "generic"


def _looks_like_test_output(command_lower: str, output_text: str) -> bool:
    if any(token in command_lower for token in ("pytest", "devtools test", "devtools verify", "ruff ", "mypy")):
        return True
    return any(
        (_TEST_PASS_RE.search(output_text), _TEST_FAIL_RE.search(output_text), _CHECK_PASS_RE.search(output_text))
    )


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _tool_file_refs(
    *,
    tool_name: str,
    command: str | None,
    output_text: str,
) -> Iterable[str]:
    if tool_name.lower() in {"read", "read_file"} and command:
        yield command
        return
    for value in (command or "", output_text):
        for token in value.split():
            cleaned = token.strip("`'\"(),:")
            if "/" in cleaned and "#" not in cleaned and not cleaned.startswith(("http://", "https://")):
                yield cleaned


def _tool_commit_refs(*, command: str | None, output_text: str) -> Iterable[str]:
    """Return commit hashes from git-shaped tool evidence.

    The detector is intentionally conservative: a bare 40-character hash in
    arbitrary command output is not enough. Work packets cite commit refs only
    when the command itself is git-shaped or the output line names commit-ish
    context, keeping handoff packets small and avoiding random hex soup.
    """

    command_text = command or ""
    should_scan_all_output = _command_invokes_git(command_text)
    seen: set[str] = set()
    for value in (command_text, output_text):
        for line in value.splitlines() or [value]:
            lowered = line.lower()
            if not (should_scan_all_output or any(token in lowered for token in ("commit", "sha", "head"))):
                continue
            for match in _COMMIT_SHA_RE.finditer(line):
                sha = match.group("sha").lower()
                if sha in seen:
                    continue
                seen.add(sha)
                yield sha


def _command_invokes_git(command: str) -> bool:
    return re.search(r"(?:^|[;&|({]\s*)git(?:\s|$)", command, re.IGNORECASE) is not None


def _block_text(block: Mapping[str, object]) -> str:
    for key in ("text", "content", "output", "result"):
        value = block.get(key)
        text = _optional_text(value)
        if text:
            return text
    return ""


def _optional_text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _preview(value: str, *, limit: int = 160) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


__all__ = [
    "RECOVERY_TRANSFORM",
    "RECOVERY_TRANSFORM_ID",
    "RECOVERY_TRANSFORM_VERSION",
    "TRANSFORM_REGISTRY",
    "DecisionCandidate",
    "ForensicIndex",
    "ForensicIndexEntry",
    "RecoveryDigest",
    "RecoveryEvent",
    "RecoveryReportPreset",
    "RecoverySizeMetrics",
    "RunStateSummary",
    "SubagentReport",
    "ToolSummary",
    "TransformDescriptor",
    "TransformMetadata",
    "TransformRawRef",
    "compile_recovery_digest",
    "render_recovery_report",
    "render_resume_bundle",
]
