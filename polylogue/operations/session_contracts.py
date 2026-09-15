"""Exact read-operation contracts owned by the session archive product."""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from polylogue.archive.message.roles import Role
from polylogue.core.enums import MaterialOrigin, Origin

Bound = Annotated[int, Field(ge=1, le=1000)]
Offset = Annotated[int, Field(ge=0)]
Text = Annotated[str, Field(min_length=1, max_length=8192)]
Continuation = Annotated[str, Field(min_length=1, max_length=65536)]
RawOrigin = Literal["claude-code-session", "codex-session"]
#: Mirrors ``MessageTypeName`` in the storage query layer. Declared here rather
#: than imported so an operation contract does not reach into storage.
MessageTypeFilter = Literal["message", "summary", "tool_use", "tool_result", "thinking", "context", "protocol"]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionList(Request):
    operation: Literal["sessions.list"] = "sessions.list"
    expression: Text | None = None
    origin: Origin | None = None
    tag: str | None = None
    repo: str | None = None
    since: str | None = None
    until: str | None = None
    sort: Literal["date", "tokens", "messages", "words", "longest"] | None = None
    min_messages: Offset | None = None
    max_messages: Offset | None = None
    min_words: Offset | None = None
    limit: Bound = 50
    offset: Offset = 0
    continuation: Continuation | None = None


class SessionSearch(SessionList):
    operation: Literal["sessions.search"] = "sessions.search"  # type: ignore[assignment]


class SessionRead(Request):
    """One bounded transcript window for an exact session reference.

    The three message filters are part of the request identity, not of the
    window: a continuation minted for ``message_role=("user",)`` cannot resume
    an unfiltered read, because the two do not name the same row sequence.
    They live here rather than only on ``Polylogue.get_messages_paginated`` so
    that routing every surface through the one bound execution route
    (``polylogue/operations/transcript_window.py``) preserves the selection
    vocabulary instead of dropping it (polylogue-ijbwq).
    """

    operation: Literal["sessions.read"] = "sessions.read"
    ref: Text
    message_role: tuple[Role, ...] = ()
    message_type: MessageTypeFilter | None = None
    material_origin: tuple[MaterialOrigin, ...] = ()
    limit: Bound = 50
    offset: Offset = 0
    continuation: Continuation | None = None


class SessionOrchestration(Request):
    operation: Literal["sessions.orchestration"] = "sessions.orchestration"
    ref: Text


class ResumeContext(Request):
    operation: Literal["context.resume"] = "context.resume"
    session_id: str | None = None
    repo_path: str | None = None
    cwd: str | None = None
    recent_files: list[str] = Field(default_factory=list, max_length=100)
    related_limit: Annotated[int, Field(ge=1, le=20)] = 5


class SessionTimeline(Request):
    operation: Literal["sessions.timeline"] = "sessions.timeline"
    origin: Origin | None = None
    since: str | None = None
    until: str | None = None
    expression: Text | None = None
    limit: Bound = 100
    offset: Offset = 0
    continuation: Continuation | None = None


class RawList(Request):
    operation: Literal["sessions.raw.list"] = "sessions.raw.list"
    origin: RawOrigin
    limit: Bound = 100
    continuation: Text | None = None


class RawSearch(Request):
    operation: Literal["sessions.raw.search"] = "sessions.raw.search"
    origin: RawOrigin
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    reference: Text | None = None
    limit: Bound = 100
    scan_bytes: Annotated[int, Field(ge=1, le=8_388_608)] = 8_388_608
    continuation: Text | None = None


class RawRead(Request):
    operation: Literal["sessions.raw.read", "memory.raw.get"] = "sessions.raw.read"
    reference: Text
    offset: Offset = 0
    max_bytes: Annotated[int, Field(ge=4, le=64_000)] = 64_000


class RawTimeline(Request):
    operation: Literal["sessions.raw.timeline"] = "sessions.raw.timeline"
    origins: list[RawOrigin] = Field(
        default_factory=lambda: list[RawOrigin](["claude-code-session", "codex-session"]), min_length=1, max_length=2
    )
    since: str | None = None
    until: str | None = None
    query: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    limit: Bound = 100
    scan_bytes: Annotated[int, Field(ge=1, le=8_388_608)] = 8_388_608
    continuation: Text | None = None


class RawMemorySearch(Request):
    operation: Literal["memory.raw.search"] = "memory.raw.search"
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    origins: list[RawOrigin] = Field(
        default_factory=lambda: list[RawOrigin](["claude-code-session", "codex-session"]), min_length=1, max_length=2
    )
    limit: Bound = 100
    scan_bytes: Annotated[int, Field(ge=1, le=8_388_608)] = 8_388_608
    source_cursors: dict[RawOrigin, str | None] | None = None


SessionOperation = Annotated[
    SessionList
    | SessionSearch
    | SessionRead
    | SessionOrchestration
    | ResumeContext
    | SessionTimeline
    | RawList
    | RawSearch
    | RawRead
    | RawTimeline
    | RawMemorySearch,
    Field(discriminator="operation"),
]
SESSION_OPERATION_ADAPTER: TypeAdapter[SessionOperation] = TypeAdapter(SessionOperation)


class Coverage(BaseModel):
    authority: Literal["indexed-archive", "original-local-session-jsonl"]
    complete: bool
    gaps: list[str] = Field(default_factory=list)
    time_basis: Literal["event-timestamp", "session-file-mtime", "none"] = "none"
    scanned_bytes: int = 0


T = TypeVar("T")


class SessionPage(BaseModel, Generic[T]):
    items: list[T]
    total: int | None
    limit: int
    offset: int
    next_offset: int | None = None
    continuation: str | None = None
    coverage: Coverage
    outcome: Literal["ok", "empty", "degraded"]


class TimelineEvent(BaseModel):
    reference: str
    session_id: str
    origin: str
    kind: Literal["message", "session-event"]
    event_type: str
    timestamp_ms: int
    event_id: str | None = None
    message_id: str | None = None
    text: str


class RawObservation(BaseModel):
    reference: str
    origin: RawOrigin
    indexed_session_id: None = None
    mtime_ns: int
    bytes: int
    line: int | None = None
    offset: int | None = None
    text: str | None = None


class RawSourceCoverage(BaseModel):
    origin: RawOrigin
    availability: Literal["available", "unavailable"]
    reason: str | None = None
    scanned_bytes: int = 0
    truncated: bool = False


class RawPage(BaseModel):
    items: list[RawObservation]
    sources: list[RawSourceCoverage]
    coverage: Coverage
    continuation: str | None = None
    source_cursors: dict[RawOrigin, str | None] | None = None
    outcome: Literal["ok", "empty", "degraded"]


class RawContent(BaseModel):
    reference: str
    origin: RawOrigin
    indexed_session_id: None = None
    mtime_ns: int
    offset: int
    bytes: int
    next_offset: int | None
    content: str
    coverage: Coverage
    outcome: Literal["ok", "empty", "degraded"]


class SessionOperationError(BaseModel):
    outcome: Literal["error"] = "error"
    code: str
    message: str


def session_operation_contracts() -> dict[str, Any]:
    """Generate adapter contracts from the same models the owner validates."""
    from polylogue.analysis.orchestration_evidence import SessionOrchestrationEvidence
    from polylogue.surfaces.payloads import (
        ContextPreamble,
        SessionMessageRowPayload,
        SessionSearchHitPayload,
        SessionSummaryPayload,
    )

    rows: tuple[tuple[type[BaseModel], type[BaseModel], str, str, dict[str, str]], ...] = (
        (SessionList, SessionPage[SessionSummaryPayload], "exhaustive-page", "query", {"projection": "sessions"}),
        (SessionSearch, SessionPage[SessionSearchHitPayload], "ranked-page", "query", {"projection": "sessions"}),
        (SessionRead, SessionPage[SessionMessageRowPayload], "exhaustive-page", "read", {"view": "messages"}),
        (SessionOrchestration, SessionOrchestrationEvidence, "single-object", "get", {"projection": "orchestration"}),
        (ResumeContext, ContextPreamble, "bounded-context", "context", {"intent": "resume"}),
        (SessionTimeline, SessionPage[TimelineEvent], "exhaustive-page", "query", {"projection": "timeline"}),
        (RawList, RawPage, "bounded-source-scan", "query", {}),
        (RawSearch, RawPage, "bounded-source-scan", "query", {}),
        (RawRead, RawContent, "byte-page", "query", {}),
        (RawTimeline, RawPage, "bounded-source-scan", "query", {}),
        (RawMemorySearch, RawPage, "bounded-source-scan", "query", {}),
    )
    import copy

    operations = {}
    for request, result, semantics, tool, selector in rows:
        schema = request.model_json_schema()
        operation_field = schema["properties"]["operation"]
        names = operation_field.get("enum", [operation_field.get("const")])
        for name in names:
            operation_schema = copy.deepcopy(schema)
            operation_schema["properties"]["operation"] = {"type": "string", "const": name, "default": name}
            operations[name] = {
                "request_schema": operation_schema,
                "result_schema": result.model_json_schema(),
                "error_schema": SessionOperationError.model_json_schema(),
                "result_semantics": semantics,
                "authority": "original-local-session-jsonl" if ".raw." in name else "indexed-archive",
                "capability": "read",
                "effects": ["disposable-context-receipt"] if name == "context.resume" else [],
                "mcp": {
                    "tool": "query",
                    "selector": {"projection": "session-operations"},
                    "argument": "session_operation",
                    "operation": name,
                },
                "convenience_surface": {"tool": tool, "selector": selector},
                "owner": "polylogue.operations.session_reads.execute_session_operation",
            }
    return {"schema": "polylogue.session-operations.v1", "operations": operations}
