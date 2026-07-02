"""Storage-free Run / ContextSnapshot / ObservedEvent projection.

The projection is intentionally fed by existing session-digest DTOs. It gives
successor-context consumers a typed execution view without inventing a new
durable table before there is measured query pressure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, model_validator

from polylogue.core.refs import EvidenceRef, ObjectRef
from polylogue.insights.archive_models import ArchiveInsightModel

RunHarness = Literal["claude-code", "codex", "chatgpt", "local", "unknown"]
RunStatus = Literal["completed", "failed", "unknown"]
ContextBoundary = Literal["session_start", "subagent_start", "resume", "unknown"]
ContextInheritanceMode = Literal["clean", "summary", "prefix", "snapshot", "injected", "unknown"]
ObservedEventKind = Literal[
    "session_started",
    "tool_finished",
    "subagent_started",
    "subagent_finished",
    # Structured in-session outcomes from the keystone tool-result fields (#2482).
    # External truths (PR/issue/CI state) are not synthesized from prose.
    "command_succeeded",
    "command_failed",
    "test_passed",
    "test_failed",
]
ObservedDeliveryState = Literal["observed", "unknown"]


class _RawRefLike(Protocol):
    def to_evidence_ref(self) -> EvidenceRef: ...


class _ToolSummaryLike(Protocol):
    @property
    def tool_name(self) -> str: ...

    @property
    def tool_id(self) -> str | None: ...

    @property
    def command(self) -> str | None: ...

    @property
    def handler_kind(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def raw_refs(self) -> Sequence[_RawRefLike]: ...

    @property
    def pr_refs(self) -> Sequence[str]: ...

    @property
    def issue_refs(self) -> Sequence[str]: ...

    @property
    def file_refs(self) -> Sequence[str]: ...

    @property
    def commit_refs(self) -> Sequence[str]: ...


class _SubagentReportLike(Protocol):
    @property
    def subagent_type(self) -> str: ...

    @property
    def tool_id(self) -> str | None: ...

    @property
    def task_id(self) -> str | None: ...

    @property
    def child_session_id(self) -> str | None: ...

    @property
    def resolved_child_session_id(self) -> str | None: ...

    @property
    def prompt(self) -> str: ...

    @property
    def final_report_preview(self) -> str: ...

    @property
    def raw_refs(self) -> Sequence[_RawRefLike]: ...


class _SessionDigestEventLike(Protocol):
    @property
    def kind(self) -> ObservedEventKind: ...

    @property
    def summary(self) -> str: ...

    @property
    def tool_name(self) -> str | None: ...

    @property
    def tool_id(self) -> str | None: ...

    @property
    def command(self) -> str | None: ...

    @property
    def handler_kind(self) -> str | None: ...

    @property
    def status(self) -> str | None: ...

    @property
    def raw_refs(self) -> Sequence[_RawRefLike]: ...


class ProjectedRun(ArchiveInsightModel):
    """One concrete session/subagent execution inferred from raw evidence."""

    run_ref: ObjectRef
    native_session_id: str | None = None
    native_parent_session_id: str | None = None
    parent_run_ref: ObjectRef | None = None
    agent_ref: ObjectRef | None = None
    lineage_refs: tuple[ObjectRef, ...] = ()
    provider_origin: str = "unknown"
    harness: RunHarness = "unknown"
    role: Literal["main", "subagent"] = "main"
    title: str = ""
    cwd: str | None = None
    git_branch: str | None = None
    status: RunStatus = "unknown"
    confidence: Literal["raw", "inferred"] = "inferred"
    transcript_ref: EvidenceRef | None = None
    evidence_refs: tuple[EvidenceRef, ...]
    context_snapshot_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def _requires_evidence(self) -> ProjectedRun:
        if not self.evidence_refs:
            raise ValueError("ProjectedRun requires evidence_refs")
        return self


class ContextSnapshot(ArchiveInsightModel):
    """What a run had visible at one concrete boundary."""

    snapshot_ref: ObjectRef
    run_ref: ObjectRef
    boundary: ContextBoundary
    inheritance_mode: ContextInheritanceMode = "unknown"
    segment_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...]
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _requires_evidence(self) -> ContextSnapshot:
        if not self.evidence_refs:
            raise ValueError("ContextSnapshot requires evidence_refs")
        return self


class ObservedEvent(ArchiveInsightModel):
    """Evidence-backed event attached to a projected run."""

    event_ref: ObjectRef
    kind: ObservedEventKind
    run_ref: ObjectRef
    summary: str
    delivery_state: ObservedDeliveryState = "observed"
    subject_ref: ObjectRef | None = None
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...]
    tool_name: str | None = None
    tool_id: str | None = None
    command: str | None = None
    handler_kind: str | None = None
    status: str | None = None

    @model_validator(mode="after")
    def _requires_evidence(self) -> ObservedEvent:
        if not self.evidence_refs:
            raise ValueError("ObservedEvent requires evidence_refs")
        return self


class RunProjection(ArchiveInsightModel):
    """Typed execution projection over session-digest evidence."""

    session_id: str
    runs: tuple[ProjectedRun, ...]
    context_snapshots: tuple[ContextSnapshot, ...] = ()
    events: tuple[ObservedEvent, ...] = ()

    @model_validator(mode="after")
    def _requires_main_run(self) -> RunProjection:
        if not self.runs:
            raise ValueError("RunProjection requires at least one run")
        return self


def build_run_projection(
    *,
    session_id: str,
    source_origin: str,
    title: str | None,
    git_branch: str | None,
    working_directories: Sequence[str],
    session_raw_refs: Sequence[_RawRefLike],
    tool_summaries: Sequence[_ToolSummaryLike],
    subagent_reports: Sequence[_SubagentReportLike],
    session_digest_events: Sequence[_SessionDigestEventLike],
) -> RunProjection:
    """Project digest evidence into bounded Run/ContextSnapshot/ObservedEvent DTOs."""

    session_evidence = _to_evidence_refs(session_raw_refs)
    main_run_ref = _run_ref(session_id)
    main_snapshot_ref = _context_snapshot_ref(session_id, "session_start")
    harness = _harness_for_origin(source_origin)
    main_run = ProjectedRun(
        run_ref=main_run_ref,
        native_session_id=session_id,
        agent_ref=_agent_ref(harness, "main"),
        lineage_refs=(main_run_ref,),
        provider_origin=source_origin,
        harness=harness,
        role="main",
        title=title or session_id,
        cwd=working_directories[0] if working_directories else None,
        git_branch=git_branch,
        status=_main_run_status(session_digest_events, tool_summaries),
        confidence="raw",
        transcript_ref=session_evidence[0],
        evidence_refs=session_evidence,
        context_snapshot_ref=main_snapshot_ref,
    )
    snapshots: list[ContextSnapshot] = [
        ContextSnapshot(
            snapshot_ref=main_snapshot_ref,
            run_ref=main_run_ref,
            boundary="session_start",
            inheritance_mode="unknown",
            segment_refs=(ObjectRef(kind="session", object_id=session_id),),
            evidence_refs=session_evidence,
            metadata={"source": "archive-session"},
        )
    ]
    runs: list[ProjectedRun] = [main_run]
    observed: list[ObservedEvent] = [
        ObservedEvent(
            event_ref=_event_ref(session_id, "session_started", 0),
            kind="session_started",
            run_ref=main_run_ref,
            summary=f"Session {session_id} started",
            subject_ref=ObjectRef(kind="session", object_id=session_id),
            evidence_refs=session_evidence,
        )
    ]

    for index, tool in enumerate(tool_summaries):
        evidence_refs = _to_evidence_refs(tool.raw_refs)
        observed.append(
            ObservedEvent(
                event_ref=_event_ref(session_id, "tool_finished", index),
                kind="tool_finished",
                run_ref=main_run_ref,
                summary=_tool_event_summary(tool),
                subject_ref=ObjectRef(kind="message", object_id=evidence_refs[0].message_id or session_id),
                object_refs=_tool_object_refs(tool),
                evidence_refs=evidence_refs,
                tool_name=tool.tool_name,
                tool_id=tool.tool_id,
                command=tool.command,
                handler_kind=tool.handler_kind,
                status=tool.status,
            )
        )

    for index, event in enumerate(session_digest_events):
        # Structured outcome events carry no GitHub/issue object refs and always
        # have delivery_state "observed" — their evidence is the tool-call block.
        evidence_refs = _to_evidence_refs(event.raw_refs)
        observed.append(
            ObservedEvent(
                event_ref=_event_ref(session_id, event.kind, index),
                kind=event.kind,
                run_ref=main_run_ref,
                summary=event.summary,
                subject_ref=ObjectRef(kind="message", object_id=evidence_refs[0].message_id or session_id),
                evidence_refs=evidence_refs,
                tool_name=event.tool_name,
                tool_id=event.tool_id,
                command=event.command,
                handler_kind=event.handler_kind,
                status=event.status,
            )
        )

    for index, report in enumerate(subagent_reports):
        child_id = report.resolved_child_session_id or report.child_session_id or report.task_id or f"subagent-{index}"
        child_run_ref = _run_ref(child_id)
        child_snapshot_ref = _context_snapshot_ref(child_id, "subagent_start")
        evidence_refs = _to_evidence_refs(report.raw_refs)
        child_agent_ref = _agent_ref(harness, report.subagent_type)
        report_ref = _subagent_report_ref(session_id, report, index)
        runs.append(
            ProjectedRun(
                run_ref=child_run_ref,
                native_session_id=report.child_session_id,
                native_parent_session_id=session_id,
                parent_run_ref=main_run_ref,
                agent_ref=child_agent_ref,
                lineage_refs=(main_run_ref, child_run_ref),
                provider_origin=source_origin,
                harness=harness,
                role="subagent",
                title=report.prompt or report.subagent_type,
                cwd=working_directories[0] if working_directories else None,
                git_branch=git_branch,
                status="completed" if report.final_report_preview else "unknown",
                confidence="raw" if report.child_session_id else "inferred",
                transcript_ref=evidence_refs[0],
                evidence_refs=evidence_refs,
                context_snapshot_ref=child_snapshot_ref,
            )
        )
        snapshots.append(
            ContextSnapshot(
                snapshot_ref=child_snapshot_ref,
                run_ref=child_run_ref,
                boundary="subagent_start",
                inheritance_mode="summary",
                segment_refs=(main_run_ref, report_ref),
                evidence_refs=evidence_refs,
                metadata={"subagent_type": report.subagent_type},
            )
        )
        observed.append(
            ObservedEvent(
                event_ref=_event_ref(session_id, "subagent_started", index),
                kind="subagent_started",
                run_ref=main_run_ref,
                summary=f"{report.subagent_type} subagent started",
                subject_ref=child_run_ref,
                object_refs=(child_agent_ref, report_ref),
                evidence_refs=evidence_refs,
            )
        )
        if report.final_report_preview:
            observed.append(
                ObservedEvent(
                    event_ref=_event_ref(session_id, "subagent_finished", index),
                    kind="subagent_finished",
                    run_ref=child_run_ref,
                    summary=report.final_report_preview,
                    subject_ref=child_run_ref,
                    object_refs=(child_agent_ref, report_ref),
                    evidence_refs=evidence_refs,
                )
            )

    return RunProjection(
        session_id=session_id,
        runs=tuple(runs),
        context_snapshots=tuple(snapshots),
        events=tuple(observed),
    )


def _to_evidence_refs(raw_refs: Sequence[_RawRefLike]) -> tuple[EvidenceRef, ...]:
    return tuple(ref.to_evidence_ref() for ref in raw_refs)


def _run_ref(run_id: str) -> ObjectRef:
    return ObjectRef(kind="run", object_id=run_id)


def _agent_ref(harness: RunHarness, role_or_type: str) -> ObjectRef:
    return ObjectRef(kind="agent", object_id=f"{harness}/{role_or_type or 'unknown'}")


def _subagent_report_ref(session_id: str, report: _SubagentReportLike, index: int) -> ObjectRef:
    stable_id = report.tool_id or report.task_id or str(index)
    return ObjectRef(kind="subagent-report", object_id=f"{session_id}:{stable_id}")


def _context_snapshot_ref(run_id: str, boundary: str) -> ObjectRef:
    return ObjectRef(kind="context-snapshot", object_id=f"{run_id}:{boundary}")


def _event_ref(session_id: str, kind: str, index: int) -> ObjectRef:
    return ObjectRef(kind="observed-event", object_id=f"{session_id}:{kind}:{index}")


def _harness_for_origin(source_origin: str) -> RunHarness:
    if source_origin == "codex-session":
        return "codex"
    if source_origin == "claude-code-session":
        return "claude-code"
    if source_origin == "chatgpt-export":
        return "chatgpt"
    if source_origin in {"gemini-cli-session", "hermes-session", "antigravity-session"}:
        return "local"
    return "unknown"


def _main_run_status(
    session_digest_events: Sequence[_SessionDigestEventLike],
    tool_summaries: Sequence[_ToolSummaryLike],
) -> RunStatus:
    if any(event.kind in {"command_failed", "test_failed"} for event in session_digest_events):
        return "failed"
    if any(tool.status == "failed" for tool in tool_summaries):
        return "failed"
    if session_digest_events or tool_summaries:
        return "completed"
    return "unknown"


def _tool_event_summary(tool: _ToolSummaryLike) -> str:
    command = f" — {tool.command}" if tool.command else ""
    return f"{tool.tool_name} [{tool.handler_kind}] ({tool.status}){command}"


def _tool_object_refs(tool: _ToolSummaryLike) -> tuple[ObjectRef, ...]:
    tool_ref = _tool_call_ref(tool)
    return (
        *(ref for ref in (tool_ref,) if ref is not None),
        *(ObjectRef(kind="github-pr", object_id=ref) for ref in tool.pr_refs),
        *(ObjectRef(kind="github-issue", object_id=ref) for ref in tool.issue_refs if ref not in set(tool.pr_refs)),
        *(ObjectRef(kind="file", object_id=ref) for ref in tool.file_refs),
        *(ObjectRef(kind="commit", object_id=ref) for ref in tool.commit_refs),
    )


def _tool_call_ref(tool: _ToolSummaryLike) -> ObjectRef | None:
    if not tool.tool_id or not tool.raw_refs:
        return None
    session_id = tool.raw_refs[0].to_evidence_ref().session_id
    return ObjectRef(kind="tool-call", object_id=f"{session_id}:{tool.tool_id}")


__all__ = [
    "ContextSnapshot",
    "ObservedEvent",
    "ProjectedRun",
    "RunProjection",
    "build_run_projection",
]
