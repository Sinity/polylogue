"""Shared surface payload models for CLI, MCP, and presentation adapters."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Literal, NotRequired, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.core.json import JSONDocument, JSONValue, require_json_document

if TYPE_CHECKING:
    from collections.abc import Container

    from polylogue.archive.conversation.neighbor_candidates import ConversationNeighborCandidate, NeighborReason
    from polylogue.archive.models import Conversation, ConversationSummary, Message
    from polylogue.archive.query.search_hits import ConversationSearchHit


def serialize_surface_payload(payload: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialize a surface payload model with stable JSON formatting."""
    return payload.model_dump_json(indent=2, exclude_none=exclude_none)


def model_json_document(payload: BaseModel, *, exclude_none: bool = False) -> JSONDocument:
    """Return a model dump constrained to the shared JSON document type."""
    return require_json_document(
        payload.model_dump(mode="json", exclude_none=exclude_none),
        context=f"{payload.__class__.__name__} JSON payload",
    )


class SurfacePayloadModel(BaseModel):
    """Shared base for immutable JSON payload models exposed by surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self, *, exclude_none: bool = False) -> str:
        return serialize_surface_payload(self, exclude_none=exclude_none)


class MachineErrorEnvelope(TypedDict):
    """Serialized machine-error envelope with sparse optional keys."""

    status: Literal["error"]
    code: str
    message: str
    command: NotRequired[list[str]]
    details: NotRequired[JSONDocument]


class MachineSuccessEnvelope(TypedDict):
    """Serialized machine-success envelope."""

    status: Literal["ok"]
    result: JSONDocument


class MachineErrorPayload(SurfacePayloadModel):
    """Structured error payload for machine-readable CLI surfaces."""

    status: Literal["error"] = "error"
    code: str
    message: str
    command: Sequence[str] = ()
    details: Mapping[str, object] = Field(default_factory=dict)

    def to_dict(self) -> MachineErrorEnvelope:
        payload: MachineErrorEnvelope = {
            "status": self.status,
            "code": self.code,
            "message": self.message,
        }
        if self.command:
            payload["command"] = list(self.command)
        if self.details:
            payload["details"] = require_json_document(dict(self.details), context="machine error details")
        return payload

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.to_dict(), indent=2)

    def emit(self, *, exit_code: int = 1) -> None:
        """Write the payload to stdout and exit."""
        sys.stdout.write(self.to_json(exclude_none=True))
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise SystemExit(exit_code)


class MachineSuccessPayload(SurfacePayloadModel):
    """Structured success payload for machine-readable CLI surfaces."""

    status: Literal["ok"] = "ok"
    result: Mapping[str, object] = Field(default_factory=dict)

    def to_dict(self) -> MachineSuccessEnvelope:
        return {
            "status": self.status,
            "result": require_json_document(dict(self.result), context="machine success result"),
        }


def normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if isinstance(role, Enum):
        role = role.value
    return str(role)


_ANCHOR_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


class TargetRefPayload(SurfacePayloadModel):
    """Stable reader target reference for selectable archive objects."""

    target_type: Literal["conversation", "message"]
    target_id: str
    conversation_id: str | None = None
    message_id: str | None = None
    block_index: int | None = None
    identity_key: str | None = None

    @classmethod
    def conversation(cls, conversation_id: object) -> TargetRefPayload:
        target_id = str(conversation_id)
        return cls(
            target_type="conversation",
            target_id=target_id,
            conversation_id=target_id,
            identity_key=f"conversation:{target_id}",
        )

    @classmethod
    def message(cls, *, conversation_id: object, message_id: object) -> TargetRefPayload:
        conversation_target_id = str(conversation_id)
        target_id = str(message_id)
        return cls(
            target_type="message",
            target_id=target_id,
            conversation_id=conversation_target_id,
            message_id=target_id,
            identity_key=f"message:{conversation_target_id}:{target_id}",
        )


class ReaderActionAvailabilityPayload(SurfacePayloadModel):
    """Per-target reader action availability with explicit disabled reason."""

    enabled: bool
    disabled_reason: str | None = None


def reader_anchor(target_type: Literal["conversation", "message"], target_id: object) -> str:
    """Return a deterministic DOM-safe anchor for a reader target."""
    prefix = "conversation" if target_type == "conversation" else "message"
    safe_id = _ANCHOR_SAFE_RE.sub("-", str(target_id)).strip("-")
    return f"{prefix}-{safe_id or 'target'}"


def reader_conversation_actions() -> dict[str, ReaderActionAvailabilityPayload]:
    """Default action contract for conversation-level reader targets."""
    return {
        "open": ReaderActionAvailabilityPayload(enabled=True),
        "copy_link": ReaderActionAvailabilityPayload(enabled=True),
        "annotate": ReaderActionAvailabilityPayload(
            enabled=False,
            disabled_reason="annotations_not_implemented",
        ),
    }


def reader_message_actions() -> dict[str, ReaderActionAvailabilityPayload]:
    """Default action contract for message-level reader targets."""
    return {
        "copy_text": ReaderActionAvailabilityPayload(enabled=True),
        "copy_link": ReaderActionAvailabilityPayload(enabled=True),
        "annotate": ReaderActionAvailabilityPayload(
            enabled=False,
            disabled_reason="annotations_not_implemented",
        ),
    }


class ConversationMessagePayload(SurfacePayloadModel):
    """Machine-readable message payload shared across CLI and MCP surfaces."""

    id: str
    role: str
    text: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_message_actions)
    timestamp: datetime | None = None
    message_type: str = "message"
    content_blocks: list[dict[str, object]] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
    parent_id: str | None = None

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        conversation_id: object | None = None,
    ) -> ConversationMessagePayload:
        raw_message_type = getattr(message, "message_type", None)
        if raw_message_type is None:
            message_type = "message"
        elif hasattr(raw_message_type, "value"):
            message_type = str(raw_message_type.value)
        else:
            message_type = str(raw_message_type)
        target_ref = (
            TargetRefPayload.message(conversation_id=conversation_id, message_id=message.id)
            if conversation_id is not None
            else None
        )
        return cls(
            id=str(message.id),
            role=normalize_role(message.role),
            text=message.text or "",
            target_ref=target_ref,
            anchor=reader_anchor("message", message.id),
            timestamp=message.timestamp,
            message_type=message_type,
            content_blocks=message.content_blocks,
            provider_meta=message.provider_meta,
            parent_id=message.parent_id,
        )


class ConversationSummaryPayload(SurfacePayloadModel):
    """Compact conversation summary payload used by MCP/search surfaces."""

    id: str
    provider: str
    title: str
    message_count: int
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_conversation_actions)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationSummaryPayload:
        conversation_id = str(conversation.id)
        return cls(
            id=conversation_id,
            provider=str(conversation.provider),
            title=conversation.display_title,
            message_count=len(conversation.messages),
            target_ref=TargetRefPayload.conversation(conversation_id),
            anchor=reader_anchor("conversation", conversation_id),
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int | None = None,
    ) -> ConversationSummaryPayload:
        conversation_id = str(summary.id)
        return cls(
            id=conversation_id,
            provider=str(summary.provider),
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            target_ref=TargetRefPayload.conversation(conversation_id),
            anchor=reader_anchor("conversation", conversation_id),
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class ConversationDetailPayload(ConversationSummaryPayload):
    """Full conversation detail payload with serialized messages."""

    messages: tuple[ConversationMessagePayload, ...]

    @classmethod
    def from_conversation(
        cls,
        conversation: Conversation,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> ConversationDetailPayload:
        if content_projection is not None and content_projection.filters_content():
            conversation = conversation.with_content_projection(content_projection)
        summary = ConversationSummaryPayload.from_conversation(conversation)
        return cls(
            **summary.model_dump(),
            messages=tuple(
                ConversationMessagePayload.from_message(msg, conversation_id=conversation.id)
                for msg in conversation.messages
            ),
        )


class ConversationFlagsPayload(SurfacePayloadModel):
    """Boolean flags summarizing conversation content characteristics."""

    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False


class ConversationListRowPayload(SurfacePayloadModel):
    """Conversation row payload used by CLI list, JSON, and YAML surfaces.

    Carries the canonical row shape for the web reader contract (#848).
    """

    id: str
    provider: str
    title: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_conversation_actions)
    date: str | None = None
    messages: int
    tags: tuple[str, ...] = ()
    summary: str | None = None
    words: int | None = None
    repo: str | None = None
    cwd_display: str | None = None
    flags: ConversationFlagsPayload | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationListRowPayload:
        conversation_id = str(conversation.id)
        return cls(
            id=conversation_id,
            provider=str(conversation.provider),
            title=conversation.display_title,
            target_ref=TargetRefPayload.conversation(conversation_id),
            anchor=reader_anchor("conversation", conversation_id),
            date=conversation.display_date.isoformat() if conversation.display_date else None,
            messages=len(conversation.messages),
            tags=tuple(conversation.tags),
            summary=conversation.summary,
            words=sum(message.word_count for message in conversation.messages),
            repo=_extract_repo(conversation.provider_meta),
            cwd_display=_extract_cwd(conversation.provider_meta),
            flags=_build_flags_from_conversation(conversation),
        )

    @classmethod
    def from_summary(
        cls,
        summary: ConversationSummary,
        *,
        message_count: int,
        word_count: int | None = None,
        flags: ConversationFlagsPayload | None = None,
        repo: str | None = None,
        cwd_display: str | None = None,
    ) -> ConversationListRowPayload:
        conversation_id = str(summary.id)
        return cls(
            id=conversation_id,
            provider=str(summary.provider),
            title=summary.display_title,
            target_ref=TargetRefPayload.conversation(conversation_id),
            anchor=reader_anchor("conversation", conversation_id),
            date=summary.display_date.isoformat() if summary.display_date else None,
            messages=message_count,
            tags=tuple(summary.tags),
            summary=summary.summary,
            words=word_count,
            repo=repo or _extract_repo(summary.provider_meta),
            cwd_display=cwd_display or _extract_cwd(summary.provider_meta),
            flags=flags,
        )

    def selected(self, fields: Container[str] | None = None) -> JSONDocument:
        data = model_json_document(self, exclude_none=True)
        if fields is None:
            return data
        return {key: value for key, value in data.items() if key in fields}


class ConversationSearchMatchPayload(SurfacePayloadModel):
    """Evidence explaining why a conversation appeared in search results."""

    rank: int
    retrieval_lane: str
    match_surface: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=dict)
    message_id: str | None = None
    snippet: str | None = None
    score: float | None = None
    matched_terms: tuple[str, ...] = ()
    score_components: dict[str, float] = Field(default_factory=dict)


class ConversationSearchHitPayload(SurfacePayloadModel):
    """Search-hit payload with summary identity and match evidence."""

    conversation: ConversationSummaryPayload
    match: ConversationSearchMatchPayload

    @classmethod
    def from_search_hit(
        cls,
        hit: ConversationSearchHit,
        *,
        message_count: int | None = None,
    ) -> ConversationSearchHitPayload:
        if hit.message_id is not None:
            target_ref = TargetRefPayload.message(conversation_id=hit.conversation_id, message_id=hit.message_id)
            anchor = reader_anchor("message", hit.message_id)
            actions = reader_message_actions()
        else:
            target_ref = TargetRefPayload.conversation(hit.conversation_id)
            anchor = reader_anchor("conversation", hit.conversation_id)
            actions = reader_conversation_actions()
        return cls(
            conversation=ConversationSummaryPayload.from_summary(
                hit.summary,
                message_count=message_count if message_count is not None else hit.summary.message_count,
            ),
            match=ConversationSearchMatchPayload(
                rank=hit.rank,
                retrieval_lane=hit.retrieval_lane,
                match_surface=hit.match_surface,
                target_ref=target_ref,
                anchor=anchor,
                actions=actions,
                message_id=hit.message_id,
                snippet=hit.snippet,
                score=hit.score,
                matched_terms=hit.matched_terms,
                score_components=hit.score_components,
            ),
        )


class ConversationNeighborReasonPayload(SurfacePayloadModel):
    """Evidence explaining one neighboring-candidate reason."""

    kind: str
    detail: str
    evidence: str | None = None
    weight: float

    @classmethod
    def from_reason(cls, reason: NeighborReason) -> ConversationNeighborReasonPayload:
        return cls(
            kind=reason.kind,
            detail=reason.detail,
            evidence=reason.evidence,
            weight=round(reason.weight, 6),
        )


class ConversationNeighborCandidatePayload(SurfacePayloadModel):
    """Machine-readable neighboring-conversation candidate payload."""

    conversation: ConversationSummaryPayload
    rank: int
    score: float
    reasons: tuple[ConversationNeighborReasonPayload, ...]
    source_conversation_id: str | None = None
    query: str | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: ConversationNeighborCandidate,
    ) -> ConversationNeighborCandidatePayload:
        return cls(
            conversation=ConversationSummaryPayload.from_summary(
                candidate.summary,
                message_count=candidate.summary.message_count,
            ),
            rank=candidate.rank,
            score=candidate.score,
            reasons=tuple(ConversationNeighborReasonPayload.from_reason(reason) for reason in candidate.reasons),
            source_conversation_id=candidate.source_conversation_id,
            query=candidate.query,
        )


# ---------------------------------------------------------------------------
# Shared response envelopes
# ---------------------------------------------------------------------------


class QueryErrorPayload(SurfacePayloadModel):
    """Shared error payload for daemon HTTP, MCP, and other surfaces.

    Compatible with daemon HTTP's ``{"ok": False, "error": ..., "detail": ...}``
    shape and MCP's ``MCPErrorPayload``.
    """

    ok: Literal[False] = False
    error: str
    detail: str | None = None
    field: str | None = None


class QueryMissReasonPayload(SurfacePayloadModel):
    """Shared reason entry explaining why a query produced no results."""

    code: str
    severity: str
    summary: str
    detail: str | None = None
    count: int | None = None

    @classmethod
    def from_reason(cls, reason: object) -> QueryMissReasonPayload:
        return cls(
            code=getattr(reason, "code", ""),
            severity=getattr(reason, "severity", ""),
            summary=getattr(reason, "summary", ""),
            detail=getattr(reason, "detail", None),
            count=getattr(reason, "count", None),
        )


class QueryMissDiagnosticsPayload(SurfacePayloadModel):
    """Shared diagnostics payload for zero-result queries."""

    message: str
    filters: tuple[str, ...]
    reasons: tuple[QueryMissReasonPayload, ...]
    archive_conversation_count: int | None = None
    raw_conversation_count: int | None = None

    @classmethod
    def from_diagnostics(cls, diagnostics: object) -> QueryMissDiagnosticsPayload:
        return cls(
            message=getattr(diagnostics, "message", ""),
            filters=tuple(getattr(diagnostics, "filters", ())),
            reasons=tuple(QueryMissReasonPayload.from_reason(reason) for reason in getattr(diagnostics, "reasons", ())),
            archive_conversation_count=getattr(diagnostics, "archive_conversation_count", None),
            raw_conversation_count=getattr(diagnostics, "raw_conversation_count", None),
        )


class ConversationListResponse(SurfacePayloadModel):
    """Shared response envelope for list and search results.

    All read surfaces (daemon HTTP, MCP, CLI JSON output) adapt this shape.
    """

    items: tuple[ConversationListRowPayload, ...]
    total: int
    limit: int
    offset: int
    query_description: list[str] = Field(default_factory=list)
    diagnostics: QueryMissDiagnosticsPayload | None = None


TagMutationOutcome: TypeAlias = Literal["added", "no_op", "removed", "not_present"]
"""Tag idempotency outcome exposed by all mutation surfaces."""


class TagMutationResult(SurfacePayloadModel):
    """Shared mutation-result contract returned by the archive facade.

    Every surface (CLI, MCP, API, daemon) adapts this same result so the
    ``bool → outcome`` mapping is centralized in one place.

    Truthiness: ``added`` and ``removed`` are ``True`` (the tag changed);
    ``no_op`` and ``not_present`` are ``False`` (nothing changed).
    """

    outcome: TagMutationOutcome
    detail: str | None = None
    """Machine-readable detail: ``already_present`` or ``tag_not_present``."""

    def __bool__(self) -> bool:
        """Backward-compatible truthiness for ``if result:`` patterns."""
        return self.outcome in ("added", "removed")


class ConversationDetailResponse(SurfacePayloadModel):
    """Shared response envelope for a single conversation detail."""

    conversation: ConversationDetailPayload


class FacetTimeRange(SurfacePayloadModel):
    """Time range boundary for facet results."""

    min: str | None = None
    max: str | None = None


class FacetsResponse(SurfacePayloadModel):
    """Shared facets response envelope with scope semantics."""

    scoped_to_query: bool = False
    providers: dict[str, int] = Field(default_factory=dict)
    tags: dict[str, int] = Field(default_factory=dict)
    repos: dict[str, int] = Field(default_factory=dict)
    cwd_prefixes: dict[str, int] = Field(default_factory=dict)
    message_types: dict[str, int] = Field(default_factory=dict)
    action_types: dict[str, int] = Field(default_factory=dict)
    has_flags: dict[str, int] = Field(default_factory=dict)
    time_range: FacetTimeRange | None = None
    total_conversations: int = 0
    total_messages: int = 0


class MutationResultPayload(SurfacePayloadModel):
    """Shared result envelope for tag, metadata, and delete mutations.

    Carries idempotent status codes, context fields, and bulk-operation
    counts so that CLI, MCP, API, and daemon surfaces all expose the
    same mutation contract shape.
    """

    status: str
    """``ok``, ``deleted``, ``not_found``, ``unchanged``, or ``partial``."""

    conversation_id: str | None = None
    detail: str | None = None
    """Machine-readable detail: ``already_present``, ``tag_not_present``,
    ``key_not_found``, ``value_unchanged``, ``conversation_not_found``."""

    outcome: str | None = None
    """Tag idempotency outcome: ``added``, ``no_op``, ``removed``, or ``not_present``."""

    affected_count: int | None = None
    skipped_count: int | None = None
    tag: str | None = None
    key: str | None = None
    conversation_count: int | None = None
    tag_count: int | None = None
    applied_count: int | None = None


# ---------------------------------------------------------------------------
# Payload builder helpers
# ---------------------------------------------------------------------------


def _extract_repo(provider_meta: dict[str, object] | None) -> str | None:
    if provider_meta is None:
        return None
    repo = provider_meta.get("repo") or provider_meta.get("repository") or provider_meta.get("git_repo")
    return str(repo) if repo else None


def _extract_cwd(provider_meta: dict[str, object] | None) -> str | None:
    if provider_meta is None:
        return None
    cwd = provider_meta.get("cwd") or provider_meta.get("working_directory") or provider_meta.get("cwd_display")
    return str(cwd) if cwd else None


def _build_flags_from_conversation(conversation: object) -> ConversationFlagsPayload | None:
    has_tool = bool(getattr(conversation, "has_tool_use", None))
    has_thinking = bool(getattr(conversation, "has_thinking", None))
    has_paste = bool(getattr(conversation, "has_paste", None))
    if not has_tool and not has_thinking and not has_paste:
        return None
    return ConversationFlagsPayload(has_tool_use=has_tool, has_thinking=has_thinking, has_paste=has_paste)


__all__ = [
    "ConversationDetailPayload",
    "ConversationDetailResponse",
    "ConversationFlagsPayload",
    "ConversationListResponse",
    "ConversationListRowPayload",
    "ConversationMessagePayload",
    "ConversationNeighborCandidatePayload",
    "ConversationNeighborReasonPayload",
    "ConversationSearchHitPayload",
    "ConversationSearchMatchPayload",
    "ConversationSummaryPayload",
    "FacetTimeRange",
    "FacetsResponse",
    "MachineErrorPayload",
    "MachineErrorEnvelope",
    "MachineSuccessEnvelope",
    "MachineSuccessPayload",
    "MutationResultPayload",
    "QueryErrorPayload",
    "QueryMissDiagnosticsPayload",
    "QueryMissReasonPayload",
    "ReaderActionAvailabilityPayload",
    "SurfacePayloadModel",
    "TagMutationOutcome",
    "TagMutationResult",
    "TargetRefPayload",
    "JSONDocument",
    "JSONValue",
    "model_json_document",
    "normalize_role",
    "reader_anchor",
    "reader_conversation_actions",
    "reader_message_actions",
    "serialize_surface_payload",
]
