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
from typing import Literal

from pydantic import Field, field_validator, model_validator

from polylogue.archive.message.models import Message
from polylogue.archive.session.domain_models import Session
from polylogue.insights.archive_models import ArchiveInsightModel

RECOVERY_TRANSFORM_ID = "recovery_digest_v0"
RECOVERY_TRANSFORM_VERSION = 1

_ISSUE_RE = re.compile(r"(?:issues/|issue\s+|closed\s+|#)(?P<number>\d{3,6})", re.IGNORECASE)
_PR_RE = re.compile(r"(?:pull/|PR\s+|#)(?P<number>\d{3,6})", re.IGNORECASE)
_MERGED_RE = re.compile(r"\bMERGED\s+#(?P<number>\d{3,6})\b", re.IGNORECASE)
_TEST_PASS_RE = re.compile(r"\b(?P<count>\d+)\s+passed\b", re.IGNORECASE)
_TEST_FAIL_RE = re.compile(r"\b(?P<count>\d+)\s+failed\b", re.IGNORECASE)
_CHECK_PASS_RE = re.compile(r"\b(?P<name>[A-Za-z0-9_. -]+)\s+\.\.\.\s+ok\b")
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
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_raw_refs(self) -> ToolSummary:
        if not self.raw_refs:
            raise ValueError("ToolSummary requires at least one raw ref")
        return self


class SubagentReport(ArchiveInsightModel):
    subagent_type: str = "unknown"
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
    kind: Literal["pr_opened", "pr_merged", "issue_closed", "check_passed", "test_passed", "test_failed"]
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


class RecoveryDigest(ArchiveInsightModel):
    """Typed v0 output for transform-first successor-session recovery."""

    session_id: str
    title: str | None = None
    transform: TransformMetadata
    size_metrics: RecoverySizeMetrics
    role_counts: dict[str, int] = Field(default_factory=dict)
    tool_summaries: tuple[ToolSummary, ...] = ()
    subagent_reports: tuple[SubagentReport, ...] = ()
    run_state: RunStateSummary | None = None
    events: tuple[RecoveryEvent, ...] = ()
    decision_candidates: tuple[DecisionCandidate, ...] = ()
    resume_markdown: str
    raw_refs: tuple[TransformRawRef, ...]

    @model_validator(mode="after")
    def _requires_session_ref(self) -> RecoveryDigest:
        if not self.raw_refs:
            raise ValueError("RecoveryDigest requires at least one raw ref")
        return self


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


def compile_recovery_digest(session: Session) -> RecoveryDigest:
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
    subagent_reports = tuple(_extract_subagent_reports(session, messages))
    run_state = _extract_run_state(session, messages)
    events = tuple(_extract_events(session, messages))
    decisions = tuple(_extract_decision_candidates(session, messages))
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
        tool_summaries=tool_summaries,
        subagent_reports=subagent_reports,
        run_state=run_state,
        events=events,
        decision_candidates=decisions,
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
) -> str:
    """Render the small successor-session boot packet for a digest."""

    lines = [
        f"# Resume: {session.display_title}",
        "",
        f"- session_id: {session.id}",
        f"- origin: {session.origin}",
        f"- messages: {len(session.messages)}",
    ]
    if session.git_branch:
        lines.append(f"- branch: {session.git_branch}")
    if session.working_directories:
        lines.append(f"- workdirs: {', '.join(session.working_directories)}")
    lines.extend(["", "## Events"])
    lines.extend(f"- {event.kind}: {event.summary}" for event in events[:8])
    if not events:
        lines.append("- none extracted")
    lines.extend(["", "## Subagents"])
    for report in subagent_reports[:8]:
        prompt = f" — {report.prompt}" if report.prompt else ""
        lines.append(f"- {report.subagent_type}{prompt}")
        if report.final_report_preview:
            lines.append(f"  - report: {report.final_report_preview}")
    if not subagent_reports:
        lines.append("- none extracted")
    lines.extend(["", "## Run State"])
    if run_state is None:
        lines.append("- none extracted")
    else:
        if run_state.goal:
            lines.append(f"- goal: {run_state.goal}")
        lines.extend(f"- done: {item}" for item in run_state.done[:8])
        lines.extend(f"- in_flight: {item}" for item in run_state.in_flight[:8])
        lines.extend(f"- blocker: {item}" for item in run_state.blockers[:8])
        lines.extend(f"- next: {item}" for item in run_state.next_actions[:8])
    lines.extend(["", "## Tools"])
    for tool in tool_summaries[:8]:
        command = f" — {tool.command}" if tool.command else ""
        lines.append(f"- {tool.tool_name} [{tool.handler_kind}] ({tool.status}){command}")
    if not tool_summaries:
        lines.append("- none extracted")
    lines.extend(["", "## Candidate Decisions / Run State"])
    lines.extend(f"- {item.kind}: {item.text}" for item in decisions[:8])
    if not decisions:
        lines.append("- none extracted")
    lines.extend(["", "## Evidence"])
    lines.append("Raw refs are available on every extracted event/tool/decision record.")
    return "\n".join(lines).strip() + "\n"


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
                prompt=_subagent_prompt(block),
                final_report_preview=_preview(result_text, limit=320),
                pr_refs=tuple(_number_refs(_PR_RE, result_text)),
                issue_refs=tuple(_number_refs(_ISSUE_RE, result_text)),
                test_evidence=tuple(_test_evidence(result_text)),
                caveats=tuple(_caveats(result_text)),
                raw_refs=tuple(refs),
            )


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
            if "/" in cleaned and not cleaned.startswith(("http://", "https://")):
                yield cleaned


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
    "RecoveryDigest",
    "RecoveryEvent",
    "RecoverySizeMetrics",
    "RunStateSummary",
    "SubagentReport",
    "ToolSummary",
    "TransformDescriptor",
    "TransformMetadata",
    "TransformRawRef",
    "compile_recovery_digest",
    "render_resume_bundle",
]
