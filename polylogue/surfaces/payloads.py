"""Shared surface payload models for CLI, MCP, and presentation adapters."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.core.refs import normalize_object_ref_text, normalize_public_ref_text
from polylogue.surfaces.action_affordances import ActionAffordancePayload

MutationStatus: TypeAlias = Literal[
    "ok",
    "deleted",
    "not_found",
    "unchanged",
    "partial",
    "preview",
    "aborted",
    "failed",
]
MutationOutcome: TypeAlias = Literal[
    "added",
    "updated",
    "no_op",
    "removed",
    "not_present",
    "set",
    "deleted",
    "not_found",
    "cleared",
    "tag_reject",
    "tag_accept",
    "summary_override",
]
MutationOperation: TypeAlias = Literal[
    "add_tag",
    "set_meta",
    "mutate",
    "delete",
    "mark.add",
    "mark.delete",
    "annotation.save",
    "annotation.delete",
    "saved_view.save",
    "saved_view.delete",
    "recall_pack.save",
    "recall_pack.delete",
    "workspace.save",
    "workspace.delete",
    "correction.save",
    "correction.delete",
    "correction.clear",
]

if TYPE_CHECKING:
    from collections.abc import Container

    from polylogue.archive.models import Message, Session, SessionSummary
    from polylogue.archive.query.search_hits import SessionSearchHit
    from polylogue.archive.session.neighbor_candidates import NeighborReason, SessionNeighborCandidate
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveActionQueryRow,
        ArchiveAssertionQueryRow,
        ArchiveBlockQueryRow,
        ArchiveFileQueryRow,
        ArchiveMessageQueryRow,
        ArchiveQueryUnitAggregateRow,
    )
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionEnvelope,
        ArchiveAssertionJudgmentEnvelope,
    )


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


class ImportDetectorEvidencePayload(SurfacePayloadModel):
    """One bounded detector/classifier observation for an import explain row."""

    check: str
    matched: bool
    reason: str | None = None


class ImportProducedRowsPayload(SurfacePayloadModel):
    """Counts of archive rows that parsing a raw artifact would produce."""

    sessions: int = 0
    messages: int = 0
    blocks: int = 0
    actions: int = 0
    raw_records: int = 0
    session_refs: tuple[str, ...] = ()


class ImportSkippedRowPayload(SurfacePayloadModel):
    """Bounded explanation for an artifact or payload skipped during import explain."""

    reason: str
    source_path: str | None = None
    raw_ref: str | None = None


class ImportExplainEntryPayload(SurfacePayloadModel):
    """Explanation for one raw artifact or archive entry inspected by import explain."""

    raw_ref: str | None = None
    source_path: str | None = None
    artifact_kind: str | None = None
    provider_hint: str | None = None
    detected_origin: str | None = None
    detected_provider: str | None = None
    detector: str
    detector_evidence: tuple[ImportDetectorEvidencePayload, ...] = ()
    parser: str | None = None
    parser_version: str | None = None
    parser_mode: str | None = None
    schema_resolution: str | None = None
    produced: ImportProducedRowsPayload = Field(default_factory=ImportProducedRowsPayload)
    skipped: tuple[ImportSkippedRowPayload, ...] = ()
    caveats: tuple[str, ...] = ()
    raw_evidence_refs: tuple[str, ...] = ()
    normalization_warnings: tuple[str, ...] = ()


class ImportExplainPayload(SurfacePayloadModel):
    """Finite machine-readable explanation for `polylogue import --explain`."""

    mode: Literal["import-explain"] = "import-explain"
    source_path: str
    entries: tuple[ImportExplainEntryPayload, ...]
    produced: ImportProducedRowsPayload
    skipped: tuple[ImportSkippedRowPayload, ...] = ()
    caveats: tuple[str, ...] = ()


ProviderCompletenessStatus = Literal["complete", "partial", "missing", "proposed"]
CompletenessItemStatus = Literal["complete", "partial", "missing", "not_applicable"]


class ProviderCompletenessItemPayload(SurfacePayloadModel):
    """Evidence-backed status for one provider-package capability."""

    status: CompletenessItemStatus
    owner_path: str | None = None
    evidence: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


class ProviderPackageCompletenessRowPayload(SurfacePayloadModel):
    """Completeness row for one origin/capture-mode package."""

    package_ref: str
    origin: str
    capture_mode: str
    provider_wire: str | None = None
    maturity: Literal["accepted", "proposed"] = "accepted"
    detector: ProviderCompletenessItemPayload
    raw_model: ProviderCompletenessItemPayload
    parser: ProviderCompletenessItemPayload
    normalizer: ProviderCompletenessItemPayload
    fixtures: ProviderCompletenessItemPayload
    schema_package: ProviderCompletenessItemPayload
    query_units: ProviderCompletenessItemPayload
    read_views: ProviderCompletenessItemPayload
    import_explain: ProviderCompletenessItemPayload
    privacy_caveats: ProviderCompletenessItemPayload
    generated_docs: ProviderCompletenessItemPayload
    debt_rows: ProviderCompletenessItemPayload
    status: ProviderCompletenessStatus
    blockers: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class ProviderPackageCompletenessTotalsPayload(SurfacePayloadModel):
    """Aggregate counts for provider-package completeness rows."""

    total: int = 0
    complete: int = 0
    partial: int = 0
    missing: int = 0
    proposed: int = 0
    accepted_blocked: int = 0


class ProviderPackageCompletenessPayload(SurfacePayloadModel):
    """Finite readiness report for provider/importer package modes."""

    mode: Literal["provider-package-completeness"] = "provider-package-completeness"
    generated_at: str
    rows: tuple[ProviderPackageCompletenessRowPayload, ...]
    totals: ProviderPackageCompletenessTotalsPayload
    caveats: tuple[str, ...] = ()


ArchiveDebtKind = Literal[
    "archive-tier",
    "assertion-candidate",
    "convergence",
    "embedding",
    "fts",
    "provider-usage",
    "raw-materialization",
]
ArchiveDebtSeverity = Literal["info", "warning", "critical"]
ArchiveDebtStatus = Literal["open", "actionable", "blocked"]


class ArchiveDebtActionPayload(SurfacePayloadModel):
    """Operator action that can resolve or inspect one archive debt row."""

    label: str
    command: tuple[str, ...] = ()
    description: str | None = None


class ArchiveDebtRowPayload(SurfacePayloadModel):
    """One actionable archive debt row from an operational readiness provider."""

    debt_ref: str
    kind: ArchiveDebtKind
    stage: str
    subject_ref: str
    severity: ArchiveDebtSeverity
    status: ArchiveDebtStatus = "open"
    owner: str
    summary: str
    details: str | None = None
    source_family: str | None = None
    observed_at: str | None = None
    age_seconds: float | None = None
    evidence_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    actions: tuple[ArchiveDebtActionPayload, ...] = ()


class ArchiveDebtTotalsPayload(SurfacePayloadModel):
    """Aggregate counts for archive debt rows."""

    total: int = 0
    critical: int = 0
    warning: int = 0
    info: int = 0
    actionable: int = 0
    blocked: int = 0


class ArchiveDebtListPayload(SurfacePayloadModel):
    """Unified operational debt report across archive readiness providers."""

    mode: Literal["archive-debt-list"] = "archive-debt-list"
    generated_at: str
    archive_root: str
    rows: tuple[ArchiveDebtRowPayload, ...]
    totals: ArchiveDebtTotalsPayload
    caveats: tuple[str, ...] = ()


def normalize_role(role: object) -> str:
    if not role:
        return "unknown"
    if isinstance(role, Enum):
        role = role.value
    return str(role)


_ANCHOR_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


class TargetRefPayload(SurfacePayloadModel):
    """Stable reader target reference for selectable archive objects."""

    target_type: Literal["session", "message"]
    target_id: str
    session_id: str | None = None
    message_id: str | None = None
    block_index: int | None = None
    identity_key: str | None = None

    @classmethod
    def session(cls, session_id: object) -> TargetRefPayload:
        target_id = str(session_id)
        return cls(
            target_type="session",
            target_id=target_id,
            session_id=target_id,
            identity_key=f"session:{target_id}",
        )

    @classmethod
    def message(cls, *, session_id: object, message_id: object) -> TargetRefPayload:
        session_target_id = str(session_id)
        target_id = str(message_id)
        return cls(
            target_type="message",
            target_id=target_id,
            session_id=session_target_id,
            message_id=target_id,
            identity_key=f"message:{session_target_id}:{target_id}",
        )


#: Closed vocabulary of reader action states. Per #1488: every visible
#: action surfaces one of these explicit states with a disabled-reason
#: when applicable. ``enabled`` is the default; the other states encode
#: distinct UI semantics that should never be conflated.
#:
#: - ``enabled`` — actionable now.
#: - ``disabled`` — known unavailable; ``disabled_reason`` explains why.
#: - ``partial`` — usable in a degraded form (e.g., copy of best-effort
#:   text when paste boundaries are not exact).
#: - ``dangerous`` — actionable but irreversible; UI should require a
#:   confirmation gesture.
#: - ``loading`` — temporarily disabled while the prerequisite resolves.
#: - ``target`` — placeholder for an action that targets a different
#:   object than the one displayed (e.g., "continue elsewhere").
#: - ``unavailable`` — the action cannot be enabled in this archive
#:   shape (e.g., copy_raw on a quarantined raw record).
ReaderActionState: TypeAlias = Literal[
    "enabled",
    "disabled",
    "partial",
    "dangerous",
    "loading",
    "target",
    "unavailable",
]

#: Stable registry of action ids the reader knows about. The canonical
#: set covers the actions named in #1488: copy text/markdown/raw/permalink,
#: copy selected range, typed-only/paste-only copy variants, open raw/
#: source, inspect provenance, mark/annotate, add to context, compare,
#: open stack, continue elsewhere. New actions must be added here so
#: the payload contract stays a closed enum rather than a free-form
#: string field.
READER_ACTION_IDS: tuple[str, ...] = (
    "open",
    "copy_text",
    "copy_markdown",
    "copy_raw",
    "copy_link",
    "copy_permalink",
    "copy_selected_range",
    "copy_typed_only",
    "copy_paste_only",
    "open_raw",
    "open_source",
    "inspect_provenance",
    "mark",
    "annotate",
    "add_to_context",
    "compare",
    "open_stack",
    "continue_elsewhere",
)


class ReaderActionAvailabilityPayload(SurfacePayloadModel):
    """Per-target reader action availability with explicit disabled reason.

    Reader actions surface seven distinct states (see :data:`ReaderActionState`)
    so the UI can render "disabled because X" / "loading" / "dangerous"
    without conflating them under a single boolean (#1488). ``enabled`` is the
    compact actionability bit consumed by current reader clients; ``state`` is
    the richer reason category.
    """

    enabled: bool = True
    state: ReaderActionState = "enabled"
    disabled_reason: str | None = None
    #: Optional opt-in path to surface a repair affordance when the action
    #: is disabled (e.g., a CLI command operators can run to enable it).
    repair_path: str | None = None
    #: Optional opt-in path to surface an inspector affordance when the
    #: action is disabled or partial (e.g., a "see why" link).
    inspect_path: str | None = None


def reader_anchor(target_type: Literal["session", "message"], target_id: object) -> str:
    """Return a deterministic DOM-safe anchor for a reader target."""
    prefix = "session" if target_type == "session" else "message"
    safe_id = _ANCHOR_SAFE_RE.sub("-", str(target_id)).strip("-")
    return f"{prefix}-{safe_id or 'target'}"


def reader_session_actions() -> dict[str, ReaderActionAvailabilityPayload]:
    """Default action contract for session-level reader targets."""
    return {
        "open": ReaderActionAvailabilityPayload(enabled=True),
        "copy_link": ReaderActionAvailabilityPayload(enabled=True),
        "annotate": ReaderActionAvailabilityPayload(enabled=True),
    }


def reader_message_actions() -> dict[str, ReaderActionAvailabilityPayload]:
    """Default action contract for message-level reader targets."""
    return {
        "copy_text": ReaderActionAvailabilityPayload(enabled=True),
        "copy_link": ReaderActionAvailabilityPayload(enabled=True),
        "annotate": ReaderActionAvailabilityPayload(enabled=True),
    }


class SessionMessagePayload(SurfacePayloadModel):
    """Machine-readable message payload shared across CLI and MCP surfaces.

    #1487: this is the canonical ``MessageRenderEnvelope`` for every
    reader entry. Both session-detail and paginated-message
    endpoints emit this exact shape so the UI can render parent/branch
    state, paste/attachment flags, raw/source refs, usage/cost
    metadata, and disabled actions without per-endpoint special cases.

    Every field beyond the original id/role/text/target_ref/anchor/
    actions/timestamp/message_type/blocks/parent_id set is
    additive with sensible default — so existing callers that only
    set the original fields continue to work unchanged.
    """

    id: str
    role: str
    text: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_message_actions)
    timestamp: datetime | None = None
    message_type: str = "message"
    material_origin: str = "unknown"
    content_blocks: list[dict[str, object]] = Field(default_factory=list)
    parent_id: str | None = None
    # #1487 envelope: branch/lineage state.
    branch_index: int = 0
    # #1487 envelope: per-message content flags. Surface what the storage
    # layer already projects (#1201/#1583) so the reader does not have
    # to re-derive them from the rendered text.
    has_paste: bool = False
    has_tool_use: bool = False
    has_thinking: bool = False
    # #1487 envelope: usage/cost metadata. Carries through from parsers
    # that populated token counts (claude-code session JSONL records
    # usage; codex sometimes does). All four default to zero so
    # rendering paths can compute cost without dispatching on presence.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_name: str | None = None
    # #1487 envelope: provider attachment identifiers for the message.
    # Tuple (immutable + ordered) so the reader can render them in
    # parser order and the test contract pins exact equality.
    attachment_refs: tuple[str, ...] = ()
    # #1487 envelope: raw/source refs so the inspector can deep-link
    # to provenance without re-querying. ``raw_id`` is the
    # content-addressed blob id from ``raw_sessions``;
    # ``source_path`` is the on-disk path the session was
    # acquired from. Both default to None for messages whose
    # session has no recorded raw artifact.
    raw_id: str | None = None
    source_path: str | None = None
    # #1655: paste boundary state — exact, projected, whole_message_fallback,
    # hash_only, or None when the message has no paste evidence. Populated
    # from the stored ``messages.paste_boundary_state`` column.
    paste_boundary_state: str | None = None

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        session_id: object | None = None,
        raw_id: str | None = None,
        source_path: str | None = None,
    ) -> SessionMessagePayload:
        raw_message_type = getattr(message, "message_type", None)
        if raw_message_type is None:
            message_type = "message"
        elif hasattr(raw_message_type, "value"):
            message_type = str(raw_message_type.value)
        else:
            message_type = str(raw_message_type)
        raw_material_origin = getattr(message, "material_origin", None)
        if raw_material_origin is None:
            material_origin = "unknown"
        elif hasattr(raw_material_origin, "value"):
            material_origin = str(raw_material_origin.value)
        else:
            material_origin = str(raw_material_origin)
        target_ref = (
            TargetRefPayload.message(session_id=session_id, message_id=message.id) if session_id is not None else None
        )
        attachment_refs = tuple(
            str(getattr(attachment, "id", ""))
            for attachment in getattr(message, "attachments", ())
            if getattr(attachment, "id", None)
        )
        return cls(
            id=str(message.id),
            role=normalize_role(message.role),
            text=message.text or "",
            target_ref=target_ref,
            anchor=reader_anchor("message", message.id),
            timestamp=message.timestamp,
            message_type=message_type,
            material_origin=material_origin,
            content_blocks=message.blocks,
            parent_id=message.parent_id,
            branch_index=int(getattr(message, "branch_index", 0) or 0),
            has_paste=bool(getattr(message, "has_paste", False)),
            has_tool_use=bool(getattr(message, "has_tool_use", False)),
            has_thinking=bool(getattr(message, "has_thinking", False)),
            input_tokens=int(getattr(message, "input_tokens", 0) or 0),
            output_tokens=int(getattr(message, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(message, "cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(message, "cache_write_tokens", 0) or 0),
            model_name=getattr(message, "model_name", None),
            attachment_refs=attachment_refs,
            raw_id=raw_id,
            source_path=source_path,
        )


class SessionMessageRowPayload(SessionMessagePayload):
    """One `read --view messages` machine-output row."""

    session_id: str

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        session_id: object | None = None,
        raw_id: str | None = None,
        source_path: str | None = None,
    ) -> SessionMessageRowPayload:
        if session_id is None:
            msg_id = getattr(message, "id", "<unknown>")
            raise ValueError(f"SessionMessageRowPayload requires session_id for message {msg_id}")
        base = SessionMessagePayload.from_message(
            message,
            session_id=session_id,
            raw_id=raw_id,
            source_path=source_path,
        )
        return cls(session_id=str(session_id), **base.model_dump())


class SessionSummaryPayload(SurfacePayloadModel):
    """Compact session summary payload used by MCP/search surfaces."""

    id: str
    origin: str
    title: str
    message_count: int
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_session_actions)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_session(cls, session: Session) -> SessionSummaryPayload:
        session_id = str(session.id)
        return cls(
            id=session_id,
            origin=session.origin,
            title=session.display_title,
            message_count=len(session.messages),
            target_ref=TargetRefPayload.session(session_id),
            anchor=reader_anchor("session", session_id),
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    @classmethod
    def from_summary(
        cls,
        summary: SessionSummary,
        *,
        message_count: int | None = None,
    ) -> SessionSummaryPayload:
        session_id = str(summary.id)
        return cls(
            id=session_id,
            origin=summary.origin,
            title=summary.display_title,
            message_count=summary.message_count or 0 if message_count is None else message_count,
            target_ref=TargetRefPayload.session(session_id),
            anchor=reader_anchor("session", session_id),
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


class SessionDetailPayload(SessionSummaryPayload):
    """Full session detail payload with serialized messages."""

    messages: tuple[SessionMessagePayload, ...]

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        content_projection: ContentProjectionSpec | None = None,
    ) -> SessionDetailPayload:
        if content_projection is not None and content_projection.filters_content():
            session = session.with_content_projection(content_projection)
        summary = SessionSummaryPayload.from_session(session)
        return cls(
            **summary.model_dump(),
            messages=tuple(SessionMessagePayload.from_message(msg, session_id=session.id) for msg in session.messages),
        )


class SessionFlagsPayload(SurfacePayloadModel):
    """Boolean flags summarizing session content characteristics."""

    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False


class SessionListRowPayload(SurfacePayloadModel):
    """Session row payload used by CLI list, JSON, and YAML surfaces.

    Carries the canonical row shape for the web reader contract (#848).
    """

    id: str
    origin: str
    title: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_session_actions)
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0
    tags: tuple[str, ...] = ()
    summary: str | None = None
    words: int | None = None
    repo: str | None = None
    cwd_display: str | None = None
    flags: SessionFlagsPayload | None = None

    @classmethod
    def from_session(cls, session: Session) -> SessionListRowPayload:
        session_id = str(session.id)
        created_at = session.created_at.isoformat() if session.created_at else None
        updated_at = session.updated_at.isoformat() if session.updated_at else None
        msg_count = len(session.messages)
        return cls(
            id=session_id,
            origin=session.origin,
            title=session.display_title,
            target_ref=TargetRefPayload.session(session_id),
            anchor=reader_anchor("session", session_id),
            created_at=created_at,
            updated_at=updated_at,
            message_count=msg_count,
            tags=tuple(session.tags),
            summary=session.summary,
            words=sum(message.word_count for message in session.messages),
            repo=_session_repo(session),
            cwd_display=_session_cwd(session),
            flags=_build_flags_from_session(session),
        )

    @classmethod
    def from_summary(
        cls,
        summary: SessionSummary,
        *,
        message_count: int,
        word_count: int | None = None,
        flags: SessionFlagsPayload | None = None,
        repo: str | None = None,
        cwd_display: str | None = None,
    ) -> SessionListRowPayload:
        session_id = str(summary.id)
        created_at = summary.created_at.isoformat() if summary.created_at else None
        updated_at = summary.updated_at.isoformat() if summary.updated_at else None
        return cls(
            id=session_id,
            origin=summary.origin,
            title=summary.display_title,
            target_ref=TargetRefPayload.session(session_id),
            anchor=reader_anchor("session", session_id),
            created_at=created_at,
            updated_at=updated_at,
            message_count=message_count,
            tags=tuple(summary.tags),
            summary=summary.summary,
            words=word_count,
            repo=repo or _session_repo(summary),
            cwd_display=cwd_display or _session_cwd(summary),
            flags=flags,
        )

    def selected(self, fields: Container[str] | None = None) -> JSONDocument:
        data = model_json_document(self, exclude_none=True)
        if fields is None:
            return data
        return {key: value for key, value in data.items() if key in fields}


class SessionSearchMatchPayload(SurfacePayloadModel):
    """Evidence explaining why a session appeared in search results.

    ``score_kind`` describes the meaning of ``score`` so consumers don't
    have to guess at ordering or comparability:

    - ``"bm25"`` — SQLite FTS5 raw BM25 (lower magnitude = better match;
      values typically negative; never comparable across queries).
    - ``"rrf"`` — Reciprocal Rank Fusion (higher = stronger consensus;
      ``score_components`` carries per-lane ``*_rank`` and ``*_rrf``
      contributions).
    - ``"vector_distance"`` — semantic distance (lower = closer).
    - ``None`` — no numeric score (e.g. attachment-identity lane).
    """

    rank: int
    retrieval_lane: str
    match_surface: str
    target_ref: TargetRefPayload | None = None
    anchor: str | None = None
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=dict)
    message_id: str | None = None
    snippet: str | None = None
    score: float | None = None
    score_kind: str | None = None
    matched_terms: tuple[str, ...] = ()
    score_components: dict[str, float] = Field(default_factory=dict)
    lane_rank: int | None = Field(default=None, description="Rank within the primary contributing lane, when known.")
    lane_contribution: float | None = Field(
        default=None,
        description="Primary lane contribution to the public score, when known.",
    )
    raw_score: float | None = Field(default=None, description="Backend-native score before public interpretation.")


class SessionSearchHitPayload(SurfacePayloadModel):
    """Search-hit payload with summary identity and match evidence."""

    session: SessionSummaryPayload
    match: SessionSearchMatchPayload

    @classmethod
    def from_search_hit(
        cls,
        hit: SessionSearchHit,
        *,
        message_count: int | None = None,
    ) -> SessionSearchHitPayload:
        if hit.message_id is not None:
            target_ref = TargetRefPayload.message(session_id=hit.session_id, message_id=hit.message_id)
            anchor = reader_anchor("message", hit.message_id)
            actions = reader_message_actions()
        else:
            target_ref = TargetRefPayload.session(hit.session_id)
            anchor = reader_anchor("session", hit.session_id)
            actions = reader_session_actions()
        return cls(
            session=SessionSummaryPayload.from_summary(
                hit.summary,
                message_count=message_count if message_count is not None else hit.summary.message_count,
            ),
            match=SessionSearchMatchPayload(
                rank=hit.rank,
                retrieval_lane=hit.retrieval_lane,
                match_surface=hit.match_surface,
                target_ref=target_ref,
                anchor=anchor,
                actions=actions,
                message_id=hit.message_id,
                snippet=hit.snippet,
                score=hit.score,
                score_kind=hit.score_kind,
                matched_terms=hit.matched_terms,
                score_components=hit.score_components,
                lane_rank=hit.lane_rank,
                lane_contribution=hit.lane_contribution,
                raw_score=hit.raw_score,
            ),
        )


class SessionNeighborReasonPayload(SurfacePayloadModel):
    """Evidence explaining one neighboring-candidate reason."""

    kind: str
    detail: str
    evidence: str | None = None
    weight: float

    @classmethod
    def from_reason(cls, reason: NeighborReason) -> SessionNeighborReasonPayload:
        return cls(
            kind=reason.kind,
            detail=reason.detail,
            evidence=reason.evidence,
            weight=round(reason.weight, 6),
        )


class SessionNeighborCandidatePayload(SurfacePayloadModel):
    """Machine-readable neighboring-session candidate payload."""

    session: SessionSummaryPayload
    rank: int
    score: float
    reasons: tuple[SessionNeighborReasonPayload, ...]
    source_session_id: str | None = None
    query: str | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: SessionNeighborCandidate,
    ) -> SessionNeighborCandidatePayload:
        return cls(
            session=SessionSummaryPayload.from_summary(
                candidate.summary,
                message_count=candidate.summary.message_count,
            ),
            rank=candidate.rank,
            score=candidate.score,
            reasons=tuple(SessionNeighborReasonPayload.from_reason(reason) for reason in candidate.reasons),
            source_session_id=candidate.source_session_id,
            query=candidate.query,
        )


# ---------------------------------------------------------------------------
# Shared response envelopes
# ---------------------------------------------------------------------------


RouteReadinessState: TypeAlias = Literal[
    "ready",
    "loading",
    "empty",
    "no_results",
    "stale",
    "degraded",
    "failed",
    "budget_exceeded",
]


class RouteReadinessPayload(SurfacePayloadModel):
    """Route-visible readiness state for a web reader panel.

    ``total=0`` can mean an empty archive, a scoped no-result query, or a
    degraded backend that cannot compute results. This payload lets browserless
    tests and the web shell distinguish those states without guessing from
    counters alone (#2304).
    """

    state: RouteReadinessState
    route: str
    reason: str | None = None
    component: str | None = None
    generated_at: str | None = None
    stale_available: bool = False


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
    archive_session_count: int | None = None
    raw_session_count: int | None = None

    @classmethod
    def from_diagnostics(cls, diagnostics: object) -> QueryMissDiagnosticsPayload:
        return cls(
            message=getattr(diagnostics, "message", ""),
            filters=tuple(getattr(diagnostics, "filters", ())),
            reasons=tuple(QueryMissReasonPayload.from_reason(reason) for reason in getattr(diagnostics, "reasons", ())),
            archive_session_count=getattr(diagnostics, "archive_session_count", None),
            raw_session_count=getattr(diagnostics, "raw_session_count", None),
        )


class SessionListResponse(SurfacePayloadModel):
    """Shared response envelope for list and search results.

    All read surfaces (daemon HTTP, MCP, CLI JSON output) adapt this shape.
    """

    items: tuple[SessionListRowPayload, ...]
    total: int
    limit: int
    offset: int
    query_description: list[str] = Field(default_factory=list)
    diagnostics: QueryMissDiagnosticsPayload | None = None
    route_state: RouteReadinessPayload | None = None


# ---------------------------------------------------------------------------
# Ranked search envelope (#1266, #873 slice A)
# ---------------------------------------------------------------------------

#: Canonical ranking policy identifier exposed in the search envelope.
#: When the per-hit lane is ``hybrid`` the fused score is RRF (Reciprocal
#: Rank Fusion); for ``dialogue``/``auto`` it is BM25; for ``semantic`` it
#: is vector distance. ``mixed`` is the umbrella policy that selects per
#: retrieval lane.
RANKING_POLICY_MIXED: Literal["mixed-bm25-rrf-vector"] = "mixed-bm25-rrf-vector"

#: Version of the ranking policy implementation. Bump on semantic changes
#: to BM25/RRF/vector ordering; consumers can pin a known good version.
RANKING_POLICY_VERSION: Literal["1"] = "1"


class SearchEnvelope(SurfacePayloadModel):
    """Typed ranked-result envelope shared across CLI, MCP, API, and daemon HTTP.

    This is the canonical search response shape (#1266). The same envelope
    is emitted from every read surface so downstream consumers — shell
    pipelines, LLM tool-use harnesses, web clients — see one stable
    contract regardless of which surface produced the result.

    Fields:

    - ``hits``: ordered tuple of :class:`SessionSearchHitPayload`
      (already evidence-bearing — see #873).
    - ``total``: total number of matching sessions. Every read surface
      (CLI, MCP, daemon HTTP, Python API) computes this via the shared
      ``spec.count()`` so the field carries a concrete count uniformly
      (#1749). ``None`` is reserved for the genuine no-spec case where a
      surface emits a hits list with no backing query spec to count from.
    - ``limit``: applied page size.
    - ``offset``: applied byte/row offset (offset-based pagination is
      "best-effort, not stable" for ranked results; prefer ``next_cursor``).
    - ``next_offset``: convenience pointer for offset-based clients; only
      set when more results are likely available.
    - ``next_cursor``: opaque keyset cursor encoding (rank, session_id)
      for stable rank-first pagination. Pass back unchanged in the
      following request.
    - ``query``: the search query text actually applied (after CLI/MCP/HTTP
      coercion). Empty string when there is no FTS query.
    - ``sort``: applied sort field, e.g. ``"rank"`` (default for ranked
      search), ``"date"``, ``"messages"``. ``None`` when no explicit sort
      was requested and the underlying lane order is preserved.
    - ``retrieval_lane``: the *resolved* per-hit ranking lane — one of
      ``"dialogue"``, ``"actions"``, ``"hybrid"``, ``"semantic"``,
      ``"auto"``. This is distinct from the *request* ``retrieval_lane``
      input parameter, whose closed accepted vocabulary is only
      ``QUERY_RETRIEVAL_LANES`` (``"auto"``/``"dialogue"``/``"actions"``/
      ``"hybrid"``) — ``"semantic"`` is never a request lane (a vector-only
      run is driven by ``similar_text``) but IS reported here as the
      resolved lane for vector hits (#1749). ``auto`` means the surface
      left the lane to the planner.
    - ``ranking_policy``: policy identifier the lane used to order hits.
      Currently always :data:`RANKING_POLICY_MIXED`.
    - ``ranking_policy_version``: numeric version of the ranking policy
      so external consumers can detect ordering changes.
    - ``action_affordances``: query-result actions clients may offer after
      receiving this result set. This is intentionally top-level so it is not
      confused with per-hit reader actions in ``hit.match.actions``.
    - ``diagnostics``: optional zero-result explanation when the query
      produced no hits but filters were applied.
    """

    hits: tuple[SessionSearchHitPayload, ...]
    total: int | None
    limit: int
    offset: int
    next_offset: int | None = None
    next_cursor: str | None = None
    query: str
    sort: str | None = None
    retrieval_lane: str
    ranking_policy: str = RANKING_POLICY_MIXED
    ranking_policy_version: str = RANKING_POLICY_VERSION
    action_affordances: tuple[ActionAffordancePayload, ...] = ()
    diagnostics: QueryMissDiagnosticsPayload | None = None
    route_state: RouteReadinessPayload | None = None


QueryUnitKind: TypeAlias = Literal[
    "message",
    "action",
    "block",
    "assertion",
    "file",
    "run",
    "observed-event",
    "context-snapshot",
]
"""Terminal query source unit exposed by query-unit envelopes."""


class MessageQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for archive messages."""

    unit: Literal["message"] = "message"
    message_id: str
    session_id: str
    origin: str
    title: str | None = None
    role: str
    message_type: str
    material_origin: str = "unknown"
    position: int
    word_count: int
    text: str

    @classmethod
    def from_row(cls, row: ArchiveMessageQueryRow) -> MessageQueryRowPayload:
        return cls(
            message_id=row.message_id,
            session_id=row.session_id,
            origin=row.origin,
            title=row.title,
            role=row.role,
            message_type=row.message_type,
            material_origin=row.material_origin,
            position=row.position,
            word_count=row.word_count,
            text=row.text,
        )


class ActionQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for normalized tool/action evidence."""

    unit: Literal["action"] = "action"
    session_id: str
    message_id: str
    origin: str
    title: str | None = None
    tool_use_block_id: str
    tool_result_block_id: str | None = None
    tool_name: str | None = None
    semantic_type: str | None = None
    tool_command: str | None = None
    tool_path: str | None = None
    output_text: str | None = None

    @classmethod
    def from_row(cls, row: ArchiveActionQueryRow) -> ActionQueryRowPayload:
        return cls(
            session_id=row.session_id,
            message_id=row.message_id,
            origin=row.origin,
            title=row.title,
            tool_use_block_id=row.tool_use_block_id,
            tool_result_block_id=row.tool_result_block_id,
            tool_name=row.tool_name,
            semantic_type=row.semantic_type,
            tool_command=row.tool_command,
            tool_path=row.tool_path,
            output_text=row.output_text,
        )


class BlockQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for archive content blocks."""

    unit: Literal["block"] = "block"
    block_id: str
    message_id: str
    session_id: str
    origin: str
    title: str | None = None
    block_type: str
    position: int
    text: str | None = None
    tool_name: str | None = None
    semantic_type: str | None = None
    tool_command: str | None = None
    tool_path: str | None = None

    @classmethod
    def from_row(cls, row: ArchiveBlockQueryRow) -> BlockQueryRowPayload:
        return cls(
            block_id=row.block_id,
            message_id=row.message_id,
            session_id=row.session_id,
            origin=row.origin,
            title=row.title,
            block_type=row.block_type,
            position=row.position,
            text=row.text,
            tool_name=row.tool_name,
            semantic_type=row.semantic_type,
            tool_command=row.tool_command,
            tool_path=row.tool_path,
        )


class FileQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for affected file-path evidence."""

    unit: Literal["file"] = "file"
    session_id: str
    origin: str
    title: str | None = None
    path: str
    action_count: int
    first_message_id: str | None = None
    first_tool_use_block_id: str | None = None
    last_tool_use_block_id: str | None = None
    first_seen_ms: int | None = None
    last_seen_ms: int | None = None

    @classmethod
    def from_row(cls, row: ArchiveFileQueryRow) -> FileQueryRowPayload:
        return cls(
            session_id=row.session_id,
            origin=row.origin,
            title=row.title,
            path=row.path,
            action_count=row.action_count,
            first_message_id=row.first_message_id,
            first_tool_use_block_id=row.first_tool_use_block_id,
            last_tool_use_block_id=row.last_tool_use_block_id,
            first_seen_ms=row.first_seen_ms,
            last_seen_ms=row.last_seen_ms,
        )


class AssertionQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for user-tier assertions."""

    unit: Literal["assertion"] = "assertion"
    assertion_id: str
    target_ref: str
    scope_ref: str | None = None
    kind: AssertionKind
    key: str | None = None
    body_text: str | None = None
    value: object
    author_ref: str
    author_kind: str
    status: AssertionStatus
    visibility: AssertionVisibility
    evidence_refs: tuple[str, ...]
    staleness: object
    context_policy: object
    created_at_ms: int
    updated_at_ms: int

    @field_validator("target_ref")
    @classmethod
    def _validate_target_ref(cls, value: str) -> str:
        return normalize_object_ref_text(value)

    @field_validator("scope_ref", "author_ref")
    @classmethod
    def _validate_optional_object_ref(cls, value: str | None) -> str | None:
        return normalize_object_ref_text(value) if value is not None else None

    @field_validator("evidence_refs")
    @classmethod
    def _validate_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)

    @classmethod
    def from_row(cls, row: ArchiveAssertionQueryRow) -> AssertionQueryRowPayload:
        return cls(
            assertion_id=row.assertion_id,
            target_ref=row.target_ref,
            scope_ref=row.scope_ref,
            kind=AssertionKind.from_string(row.kind),
            key=row.key,
            body_text=row.body_text,
            value=row.value,
            author_ref=row.author_ref,
            author_kind=row.author_kind,
            status=AssertionStatus.from_string(row.status),
            visibility=AssertionVisibility.from_string(row.visibility),
            evidence_refs=row.evidence_refs,
            staleness=row.staleness,
            context_policy=row.context_policy,
            created_at_ms=row.created_at_ms,
            updated_at_ms=row.updated_at_ms,
        )


class AssertionClaimPayload(SurfacePayloadModel):
    """Shared payload for assertion-backed lifecycle claims."""

    assertion_id: str
    scope_ref: str | None = None
    target_ref: str
    key: str | None = None
    kind: AssertionKind
    value: object | None = None
    body_text: str | None = None
    author_ref: str | None = None
    author_kind: str | None = None
    evidence_refs: tuple[str, ...] = ()
    status: AssertionStatus | None = None
    visibility: AssertionVisibility | None = None
    confidence: float | None = None
    staleness: dict[str, Any] | None = None
    context_policy: dict[str, Any] | None = None
    supersedes: tuple[str, ...] = ()
    created_at_ms: int
    updated_at_ms: int

    @field_validator("target_ref")
    @classmethod
    def _validate_claim_target_ref(cls, value: str) -> str:
        return normalize_object_ref_text(value)

    @field_validator("scope_ref", "author_ref")
    @classmethod
    def _validate_claim_optional_object_ref(cls, value: str | None) -> str | None:
        return normalize_object_ref_text(value) if value is not None else None

    @field_validator("evidence_refs")
    @classmethod
    def _validate_claim_public_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)

    @classmethod
    def from_envelope(cls, envelope: ArchiveAssertionEnvelope) -> AssertionClaimPayload:
        return cls(
            assertion_id=envelope.assertion_id,
            scope_ref=envelope.scope_ref,
            target_ref=envelope.target_ref,
            key=envelope.key,
            kind=AssertionKind.from_string(envelope.kind),
            value=envelope.value,
            body_text=envelope.body_text,
            author_ref=envelope.author_ref,
            author_kind=envelope.author_kind,
            evidence_refs=tuple(envelope.evidence_refs),
            status=AssertionStatus.from_string(envelope.status),
            visibility=AssertionVisibility.from_string(envelope.visibility),
            confidence=envelope.confidence,
            staleness=envelope.staleness,
            context_policy=envelope.context_policy,
            supersedes=tuple(envelope.supersedes),
            created_at_ms=envelope.created_at_ms,
            updated_at_ms=envelope.updated_at_ms,
        )


class AssertionClaimListPayload(SurfacePayloadModel):
    """Shared list envelope for assertion-backed lifecycle claims."""

    items: tuple[AssertionClaimPayload, ...]
    total: int
    limit: int
    statuses: tuple[AssertionStatus, ...] | None = None
    kinds: tuple[AssertionKind, ...] | None = None

    @classmethod
    def from_envelopes(
        cls,
        envelopes: Sequence[ArchiveAssertionEnvelope],
        *,
        limit: int,
        statuses: Sequence[str | AssertionStatus] | None = None,
        kinds: Sequence[str | AssertionKind] | None = None,
    ) -> AssertionClaimListPayload:
        items = tuple(AssertionClaimPayload.from_envelope(envelope) for envelope in envelopes)
        return cls(
            items=items,
            total=len(items),
            limit=limit,
            statuses=None if statuses is None else tuple(AssertionStatus.from_string(status) for status in statuses),
            kinds=None if kinds is None else tuple(AssertionKind.from_string(kind) for kind in kinds),
        )


class AssertionJudgmentPayload(SurfacePayloadModel):
    """Shared payload for explicit candidate-assertion judgments."""

    judgment_id: str
    candidate_ref: str
    decision: Literal["accept", "reject", "defer", "supersede"]
    reason: str | None = None
    actor_ref: str | None = None
    decided_at_ms: int
    resulting_assertion_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @field_validator("candidate_ref", "actor_ref", "resulting_assertion_ref")
    @classmethod
    def _validate_judgment_object_refs(cls, value: str | None) -> str | None:
        return normalize_object_ref_text(value) if value is not None else None

    @field_validator("evidence_refs")
    @classmethod
    def _validate_judgment_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)

    @classmethod
    def from_envelope(cls, envelope: ArchiveAssertionEnvelope) -> AssertionJudgmentPayload:
        value = envelope.value if isinstance(envelope.value, dict) else {}
        decision = str(value.get("decision") or "")
        if decision not in {"accept", "reject", "defer", "supersede"}:
            raise ValueError("judgment assertion has invalid decision")
        resulting_assertion_ref = value.get("resulting_assertion_ref")
        return cls(
            judgment_id=envelope.assertion_id,
            candidate_ref=str(value.get("candidate_ref") or envelope.target_ref),
            decision=cast(Literal["accept", "reject", "defer", "supersede"], decision),
            reason=str(value["reason"]) if value.get("reason") is not None else None,
            actor_ref=envelope.author_ref,
            decided_at_ms=envelope.updated_at_ms,
            resulting_assertion_ref=str(resulting_assertion_ref) if resulting_assertion_ref is not None else None,
            evidence_refs=tuple(envelope.evidence_refs),
        )


class AssertionJudgmentResultPayload(SurfacePayloadModel):
    """Shared result envelope for candidate-assertion judgment workflows."""

    candidate: AssertionClaimPayload
    judgment: AssertionJudgmentPayload
    resulting_assertion: AssertionClaimPayload | None = None

    @classmethod
    def from_envelope(cls, envelope: ArchiveAssertionJudgmentEnvelope) -> AssertionJudgmentResultPayload:
        return cls(
            candidate=AssertionClaimPayload.from_envelope(envelope.candidate),
            judgment=AssertionJudgmentPayload.from_envelope(envelope.judgment),
            resulting_assertion=None
            if envelope.resulting_assertion is None
            else AssertionClaimPayload.from_envelope(envelope.resulting_assertion),
        )


RecoveryReportKind: TypeAlias = Literal["digest", "work-packet"]
RecoveryReportFormat: TypeAlias = Literal["json", "markdown"]


class RecoveryReadPayload(SurfacePayloadModel):
    """Shared recovery/read envelope for daemon web and future API/MCP parity."""

    session_id: str
    report: RecoveryReportKind
    format: RecoveryReportFormat
    digest: dict[str, object] | None = None
    work_packet: dict[str, object] | None = None
    markdown: str | None = None

    @classmethod
    def from_digest(cls, digest: BaseModel) -> RecoveryReadPayload:
        return cls(
            session_id=str(digest.model_dump(mode="python")["session_id"]),
            report="digest",
            format="json",
            digest=cast(dict[str, object], model_json_document(digest, exclude_none=True)),
        )

    @classmethod
    def from_work_packet_json(cls, packet: BaseModel) -> RecoveryReadPayload:
        return cls(
            session_id=str(packet.model_dump(mode="python")["session_id"]),
            report="work-packet",
            format="json",
            work_packet=cast(dict[str, object], model_json_document(packet, exclude_none=True)),
        )

    @classmethod
    def from_work_packet_markdown(cls, *, session_id: object, markdown: str) -> RecoveryReadPayload:
        return cls(session_id=str(session_id), report="work-packet", format="markdown", markdown=markdown)


class RefResolutionActionPayload(SurfacePayloadModel):
    """One affordance a client can offer after resolving a public ref."""

    label: str
    command: str | None = None
    href: str | None = None
    enabled: bool = True


class PublicRefResolutionPayload(SurfacePayloadModel):
    """Bounded shared payload for resolving public object/evidence refs."""

    mode: Literal["ref-resolution"] = "ref-resolution"
    ref: str
    normalized_ref: str | None = None
    kind: str | None = None
    resolved: bool
    payload_kind: str | None = None
    payload: dict[str, Any] | None = None
    title: str | None = None
    summary: str | None = None
    object_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    actions: tuple[RefResolutionActionPayload, ...] = ()

    @field_validator("normalized_ref")
    @classmethod
    def _validate_normalized_ref(cls, value: str | None) -> str | None:
        return normalize_public_ref_text(value) if value is not None else None

    @field_validator("object_refs")
    @classmethod
    def _validate_resolution_object_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_object_ref_text(ref) for ref in value)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_resolution_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class SessionReadViewEnvelope(SurfacePayloadModel):
    """Stable daemon envelope for one executed session read-view."""

    session_id: str
    view: str
    format: str
    target_refs: tuple[str, ...]
    object_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    lossiness: str
    evidence_policy: str
    privacy_policy: str
    actions: dict[str, ReaderActionAvailabilityPayload] = Field(default_factory=reader_session_actions)
    payload: Any = Field(description="Profile-specific JSON payload for the selected read view.")

    @field_validator("target_refs", "object_refs")
    @classmethod
    def _validate_session_read_object_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_object_ref_text(ref) for ref in value)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_session_read_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class ObservedEventQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for runtime-transform observed events."""

    unit: Literal["observed-event"] = "observed-event"
    event_ref: str
    session_id: str
    origin: str
    title: str | None = None
    kind: str
    summary: str
    delivery_state: str
    subject_ref: str | None = None
    object_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    @field_validator("event_ref")
    @classmethod
    def _validate_event_ref(cls, value: str) -> str:
        return normalize_object_ref_text(value)

    @field_validator("subject_ref")
    @classmethod
    def _validate_optional_subject_ref(cls, value: str | None) -> str | None:
        return normalize_object_ref_text(value) if value is not None else None

    @field_validator("object_refs")
    @classmethod
    def _validate_object_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_object_ref_text(ref) for ref in value)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_observed_event_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class ContextSnapshotQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for runtime-transform context snapshots."""

    unit: Literal["context-snapshot"] = "context-snapshot"
    snapshot_ref: str
    session_id: str
    origin: str
    title: str | None = None
    run_ref: str
    boundary: str
    inheritance_mode: str
    segment_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    metadata: dict[str, str]

    @field_validator("snapshot_ref", "run_ref")
    @classmethod
    def _validate_context_object_ref(cls, value: str) -> str:
        return normalize_object_ref_text(value)

    @field_validator("segment_refs")
    @classmethod
    def _validate_segment_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_object_ref_text(ref) for ref in value)

    @field_validator("evidence_refs")
    @classmethod
    def _validate_context_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class RunQueryRowPayload(SurfacePayloadModel):
    """Shared terminal-query row for runtime-transform runs."""

    unit: Literal["run"] = "run"
    run_ref: str
    session_id: str
    origin: str
    title: str | None = None
    native_session_id: str | None = None
    native_parent_session_id: str | None = None
    parent_run_ref: str | None = None
    agent_ref: str | None = None
    lineage_refs: tuple[str, ...]
    provider_origin: str
    harness: str
    role: str
    cwd: str | None = None
    git_branch: str | None = None
    status: str
    confidence: str
    transcript_ref: str | None = None
    evidence_refs: tuple[str, ...]
    context_snapshot_ref: str | None = None

    @field_validator("run_ref", "parent_run_ref", "agent_ref", "context_snapshot_ref")
    @classmethod
    def _validate_optional_run_object_refs(cls, value: str | None) -> str | None:
        return normalize_object_ref_text(value) if value is not None else None

    @field_validator("lineage_refs")
    @classmethod
    def _validate_lineage_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_object_ref_text(ref) for ref in value)

    @field_validator("transcript_ref")
    @classmethod
    def _validate_optional_transcript_ref(cls, value: str | None) -> str | None:
        return normalize_public_ref_text(value) if value is not None else None

    @field_validator("evidence_refs")
    @classmethod
    def _validate_run_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


QueryUnitRowPayload: TypeAlias = (
    MessageQueryRowPayload
    | ActionQueryRowPayload
    | BlockQueryRowPayload
    | AssertionQueryRowPayload
    | FileQueryRowPayload
    | RunQueryRowPayload
    | ObservedEventQueryRowPayload
    | ContextSnapshotQueryRowPayload
)
"""Union of terminal row payloads returned by explicit unit-source queries."""


_QUERY_UNIT_PIPELINE_STAGE_SCHEMA: Any = {
    "items": {
        "additionalProperties": True,
        "properties": {
            "kind": {
                "enum": ["session_scope", "sort", "limit", "offset", "group", "count"],
                "type": "string",
            },
            "predicate": {"type": "object"},
            "sort": {"type": "object"},
            "value": {"type": "integer"},
            "field": {"type": "string"},
            "metric": {"enum": ["count"], "type": "string"},
        },
        "required": ["kind"],
        "type": "object",
    }
}


class QueryUnitAggregateRowPayload(SurfacePayloadModel):
    """Grouped aggregate row over a terminal query-unit result set."""

    unit: QueryUnitKind
    group_by: str | None = None
    group_key: str | None = None
    count: int

    @classmethod
    def from_row(cls, row: ArchiveQueryUnitAggregateRow) -> QueryUnitAggregateRowPayload:
        return cls(
            unit=cast(QueryUnitKind, row.unit),
            group_by=row.group_by,
            group_key=row.group_key,
            count=row.count,
        )


class QueryUnitEnvelope(SurfacePayloadModel):
    """Shared envelope for explicit terminal unit-source query results."""

    mode: Literal["query-unit"] = "query-unit"
    unit: QueryUnitKind
    query: str
    items: tuple[QueryUnitRowPayload, ...]
    pipeline: dict[str, object] | None = None
    """Typed source-to-result pipeline that shaped this terminal-unit page."""
    pipeline_stages: tuple[dict[str, object], ...] = Field(
        default=(),
        json_schema_extra=_QUERY_UNIT_PIPELINE_STAGE_SCHEMA,
    )
    """Ordered terminal pipeline stages that shaped this page, if any."""
    total: int
    """Number of terminal rows returned in this page."""
    limit: int
    offset: int
    next_offset: int | None = None


class QueryUnitAggregateEnvelope(SurfacePayloadModel):
    """Shared envelope for terminal query-unit aggregate results."""

    mode: Literal["query-unit-aggregate"] = "query-unit-aggregate"
    unit: QueryUnitKind
    query: str
    items: tuple[QueryUnitAggregateRowPayload, ...]
    pipeline: dict[str, object] | None = None
    """Typed source-to-result pipeline that shaped this aggregate page."""
    pipeline_stages: tuple[dict[str, object], ...] = Field(
        default=(),
        json_schema_extra=_QUERY_UNIT_PIPELINE_STAGE_SCHEMA,
    )
    """Ordered terminal pipeline stages that shaped this aggregate page."""
    total: int
    """Number of aggregate rows returned in this page."""
    limit: int
    offset: int
    next_offset: int | None = None


QueryUnitResultEnvelope: TypeAlias = QueryUnitEnvelope | QueryUnitAggregateEnvelope


class OtelSpanPayload(SurfacePayloadModel):
    """OTel-like span projection over Polylogue archive evidence.

    This is intentionally an export shape, not archive identity. Span and trace
    ids are stable projection ids; Polylogue refs stay in attributes and links
    so consumers can navigate back to canonical archive evidence.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: str = "INTERNAL"
    attributes: dict[str, object] = Field(default_factory=dict)
    links: tuple[str, ...] = ()
    events: tuple[dict[str, object], ...] = ()

    @field_validator("links")
    @classmethod
    def _validate_otel_span_links(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class OtelLogRecordPayload(SurfacePayloadModel):
    """OTel-like log/event projection over non-span archive evidence."""

    trace_id: str
    observed_at: str | None = None
    body: str
    attributes: dict[str, object] = Field(default_factory=dict)
    links: tuple[str, ...] = ()

    @field_validator("links")
    @classmethod
    def _validate_otel_log_links(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


class OtelProjectionPayload(SurfacePayloadModel):
    """Bounded outbound OTel-style projection for one Polylogue evidence set."""

    mode: Literal["otel-projection"] = "otel-projection"
    source_ref: str
    format: Literal["otlp-json"] = "otlp-json"
    trace_count: int
    span_count: int
    log_count: int
    spans: tuple[OtelSpanPayload, ...] = ()
    logs: tuple[OtelLogRecordPayload, ...] = ()
    refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    @field_validator("source_ref")
    @classmethod
    def _validate_otel_source_ref(cls, value: str) -> str:
        return normalize_public_ref_text(value)

    @field_validator("refs")
    @classmethod
    def _validate_otel_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_public_ref_text(ref) for ref in value)


def build_query_unit_envelope(
    items: Sequence[QueryUnitRowPayload],
    *,
    unit: QueryUnitKind,
    query: str,
    limit: int,
    offset: int,
    has_next: bool,
    pipeline: Mapping[str, object] | None = None,
    pipeline_stages: Sequence[Mapping[str, object]] = (),
) -> QueryUnitEnvelope:
    """Construct the canonical terminal query-unit response envelope."""

    items_tuple = tuple(items)
    return QueryUnitEnvelope(
        unit=unit,
        query=query,
        items=items_tuple,
        pipeline=dict(pipeline) if pipeline is not None else None,
        pipeline_stages=tuple(dict(stage) for stage in pipeline_stages),
        total=len(items_tuple),
        limit=limit,
        offset=offset,
        next_offset=offset + limit if has_next else None,
    )


def build_query_unit_aggregate_envelope(
    items: Sequence[QueryUnitAggregateRowPayload],
    *,
    unit: QueryUnitKind,
    query: str,
    limit: int,
    offset: int,
    has_next: bool,
    pipeline: Mapping[str, object] | None = None,
    pipeline_stages: Sequence[Mapping[str, object]] = (),
) -> QueryUnitAggregateEnvelope:
    """Construct the canonical terminal aggregate response envelope."""

    items_tuple = tuple(items)
    return QueryUnitAggregateEnvelope(
        unit=unit,
        query=query,
        items=items_tuple,
        pipeline=dict(pipeline) if pipeline is not None else None,
        pipeline_stages=tuple(dict(stage) for stage in pipeline_stages),
        total=len(items_tuple),
        limit=limit,
        offset=offset,
        next_offset=offset + limit if has_next else None,
    )


#: Cursor envelope version. Bump when the encoded shape changes in a way
#: that earlier decoders cannot tolerate; consumers that pin a version
#: can detect skew. archive carries ``r`` (anchor rank within the current
#: ranking pass), ``s`` (optional float score, ``None`` for lanes without
#: a numeric score), ``c`` (session_id tie-break), and ``l`` (lane
#: resolved at the time the cursor was minted).
SEARCH_CURSOR_VERSION: Literal[1] = 1


class SearchCursor(BaseModel):
    """Decoded, validated keyset cursor for ranked search pagination.

    The cursor is an opaque base64-encoded JSON envelope as far as
    callers are concerned (``SearchEnvelope.next_cursor`` is a ``str``);
    this type is the typed in-process representation surfaces use after
    :func:`decode_search_cursor` validates the token.

    Fields are intentionally short single-letter keys so the encoded
    form stays compact in URLs and JSON pipes:

    - ``v``: cursor envelope version (always :data:`SEARCH_CURSOR_VERSION`
      today). A token with an unknown version is rejected.
    - ``r``: anchor rank (1-based position within the previous ranking
      pass). The next page starts strictly after this position.
    - ``s``: anchor score, when the lane emits a numeric score (BM25,
      RRF, vector distance). ``None`` for lanes without a score.
    - ``c``: anchor session id — deterministic tie-break when two
      hits share the same score.
    - ``l``: retrieval lane name resolved when the cursor was minted.
      Surfaces SHOULD reject a cursor whose lane does not match the
      current request so paginated state does not silently leak across
      lane changes.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    v: int = Field(default=SEARCH_CURSOR_VERSION)
    r: int
    s: float | None = None
    c: str
    lane: str = Field(validation_alias="l", serialization_alias="l")


class InvalidSearchCursorError(ValueError):
    """Raised when a caller-provided cursor token cannot be decoded.

    Surfaces should translate this into the surface-native error shape
    (CLI usage error, MCP error envelope, daemon 400 response).
    """


def build_search_cursor(hits: Sequence[SessionSearchHitPayload]) -> str | None:
    """Build an opaque keyset cursor token from the last hit of a page.

    Encodes ``(rank, score, session_id, lane)`` of the final hit so
    a follow-up request can resume strictly after the anchor even when
    the underlying archive grows between requests. Returns ``None`` when
    ``hits`` is empty.

    The token is a base64 JSON envelope (see :class:`SearchCursor`).
    Consumers should treat it as opaque and pass it back unchanged.
    """
    if not hits:
        return None
    import base64

    last = hits[-1]
    cursor = SearchCursor(
        v=SEARCH_CURSOR_VERSION,
        r=last.match.rank,
        s=last.match.score,
        c=last.session.id,
        lane=last.match.retrieval_lane,
    )
    payload = cursor.model_dump_json(by_alias=True)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_search_cursor(token: str) -> SearchCursor:
    """Decode an opaque cursor token into a typed :class:`SearchCursor`.

    Raises :class:`InvalidSearchCursorError` when the token is malformed,
    base64-undecodable, JSON-invalid, missing fields, or carries an
    unsupported version.
    """
    import base64

    if not token:
        raise InvalidSearchCursorError("cursor token is empty")
    # urlsafe_b64decode tolerates missing padding when we re-add it.
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise InvalidSearchCursorError(f"cursor token is not valid base64: {exc}") from exc
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidSearchCursorError(f"cursor token payload is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise InvalidSearchCursorError("cursor token payload is not an object")
    try:
        cursor = SearchCursor.model_validate(body)
    except Exception as exc:  # pydantic ValidationError
        raise InvalidSearchCursorError(f"cursor token payload is invalid: {exc}") from exc
    if cursor.v != SEARCH_CURSOR_VERSION:
        raise InvalidSearchCursorError(
            f"cursor version {cursor.v!r} is not supported (expected {SEARCH_CURSOR_VERSION!r})"
        )
    return cursor


def apply_search_cursor(
    hits: Sequence[SessionSearchHitPayload],
    cursor: SearchCursor,
    *,
    retrieval_lane: str | None = None,
) -> tuple[SessionSearchHitPayload, ...]:
    """Drop hits up to and including the cursor anchor.

    Stability rule: a hit survives iff its ``(score, session_id)``
    sorts strictly *after* the cursor anchor under the lane's natural
    ordering, or — when no score is available — its rank strictly
    exceeds the anchor rank.

    When ``retrieval_lane`` is supplied it is compared against the
    cursor's lane field; mismatched lanes raise
    :class:`InvalidSearchCursorError` so a paginated session cannot
    silently switch ranking policy mid-walk.
    """
    if retrieval_lane is not None and not search_cursor_lane_matches_request(cursor.lane, retrieval_lane):
        raise InvalidSearchCursorError(
            f"cursor was minted for retrieval_lane={cursor.lane!r} but this request is {retrieval_lane!r}"
        )
    result: list[SessionSearchHitPayload] = []
    for hit in hits:
        if _cursor_strictly_before(cursor, hit):
            result.append(hit)
    return tuple(result)


def search_cursor_lane_matches_request(cursor_lane: str, requested_lane: str | None) -> bool:
    """Return whether a cursor lane can be reused for the requested lane.

    ``auto`` is a planner request, not a concrete ranking lane. A first page
    requested with ``auto`` may resolve to ``dialogue`` and mint a dialogue
    cursor; the next page should be allowed to pass the same default ``auto``
    request without being rejected as a lane switch. Concrete lane-to-lane
    changes still fail.
    """
    if not cursor_lane:
        return True
    if requested_lane in {None, "", "auto"}:
        return True
    return requested_lane == cursor_lane


def _cursor_strictly_before(cursor: SearchCursor, hit: SessionSearchHitPayload) -> bool:
    """Return True when ``hit`` is strictly after the cursor anchor.

    Comparison uses lane-natural score ordering when both sides have a
    numeric score, otherwise falls back to rank. ``session_id`` is
    the deterministic tie-break.
    """
    anchor_score = cursor.s
    hit_score = hit.match.score
    score_kind = hit.match.score_kind
    # bm25 and vector_distance: lower is better; rrf: higher is better.
    lower_is_better = score_kind in {"bm25", "vector_distance"}
    if anchor_score is not None and hit_score is not None and anchor_score != hit_score:
        return (hit_score > anchor_score) if lower_is_better else (hit_score < anchor_score)
    # Same score (or no score on one side) — use rank, then conv id.
    if hit.match.rank != cursor.r:
        return hit.match.rank > cursor.r
    return hit.session.id > cursor.c


def build_search_envelope(
    hits: Sequence[SessionSearchHitPayload],
    *,
    total: int | None,
    limit: int,
    offset: int,
    query: str,
    retrieval_lane: str,
    sort: str | None = None,
    diagnostics: QueryMissDiagnosticsPayload | None = None,
    action_affordances: Sequence[ActionAffordancePayload] | None = None,
    ranking_policy: str = RANKING_POLICY_MIXED,
    ranking_policy_version: str = RANKING_POLICY_VERSION,
    cursor: SearchCursor | None = None,
) -> SearchEnvelope:
    """Construct a :class:`SearchEnvelope` with the canonical cursor logic.

    This is the one builder all four surfaces should call so the envelope
    field shape — and the cursor/next_offset semantics in particular —
    stay aligned.

    When ``cursor`` is supplied the builder drops every supplied hit up
    to and including the anchor (see :func:`apply_search_cursor`) before
    truncating to ``limit``. This means surfaces can pass the raw
    paginated fetch — typically ``offset = cursor.r`` plus ``limit`` of
    rows — and the envelope handles the keyset trim.
    """
    hits_tuple = tuple(hits)
    if cursor is not None:
        hits_tuple = apply_search_cursor(hits_tuple, cursor, retrieval_lane=retrieval_lane)
    if len(hits_tuple) > limit:
        hits_tuple = hits_tuple[:limit]
    next_offset: int | None = None
    next_cursor: str | None = None
    if hits_tuple and len(hits_tuple) == limit and (total is None or offset + limit < total):
        # More results likely available; expose both pagination handles.
        next_offset = offset + limit
        next_cursor = build_search_cursor(hits_tuple)
    if action_affordances is None:
        from polylogue.operations.action_contracts import query_result_action_affordance_payloads

        action_affordances = query_result_action_affordance_payloads()
    return SearchEnvelope(
        hits=hits_tuple,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
        next_cursor=next_cursor,
        query=query,
        sort=sort,
        retrieval_lane=retrieval_lane,
        ranking_policy=ranking_policy,
        ranking_policy_version=ranking_policy_version,
        action_affordances=tuple(action_affordances),
        diagnostics=diagnostics,
    )


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


MetadataMutationOutcome: TypeAlias = Literal["set", "unchanged", "deleted", "not_found"]
"""Metadata idempotency outcome exposed by all mutation surfaces."""


class MetadataMutationResult(SurfacePayloadModel):
    """Typed result for metadata set/delete mutations.

    ``outcome``:
    - ``set`` — the value changed (insert or update of a different value)
    - ``unchanged`` — the key already held the same value
    - ``deleted`` — the key existed and was removed
    - ``not_found`` — the key did not exist on delete
    """

    outcome: MetadataMutationOutcome
    session_id: str
    key: str
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.outcome in ("set", "deleted")


DeleteSessionOutcome: TypeAlias = Literal["deleted", "not_found"]
"""Session delete idempotency outcome exposed by all surfaces."""


class DeleteSessionResult(SurfacePayloadModel):
    """Typed result for a single session delete."""

    outcome: DeleteSessionOutcome
    session_id: str
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.outcome == "deleted"


class BulkTagMutationResult(SurfacePayloadModel):
    """Typed result for bulk tag mutations.

    Carries the same counts the MCP/CLI/daemon surfaces expose: how many
    sessions had at least one tag applied (``affected_count``) and how
    many were untouched because every tag was already present
    (``skipped_count``).
    """

    outcome: Literal["ok"] = "ok"
    session_count: int
    tag_count: int
    affected_count: int
    skipped_count: int


class SessionDetailResponse(SurfacePayloadModel):
    """Shared response envelope for a single session detail."""

    session: SessionDetailPayload


class SessionMessagesResponsePayload(SurfacePayloadModel):
    """Finite `read --view messages --format json` response."""

    session_id: str
    messages: tuple[SessionMessageRowPayload, ...]
    total: int
    limit: int
    offset: int


class FacetTimeRange(SurfacePayloadModel):
    """Time range boundary for facet results."""

    min: str | None = None
    max: str | None = None


class FacetBucketsPayload(SurfacePayloadModel):
    """Bucket counts for one facet scope (scoped or global).

    Carries the same shape as the top-level facet fields on
    :class:`FacetsResponse` so a reader can render scoped and global
    side by side without having to interpret ``scoped_to_query``.
    See #1269 (slice D of #873).
    """

    origins: dict[str, int] = Field(default_factory=dict)
    tags: dict[str, int] = Field(default_factory=dict)
    repos: dict[str, int] = Field(default_factory=dict)
    role_counts: dict[str, int] = Field(default_factory=dict)
    message_types: dict[str, int] = Field(default_factory=dict)
    material_origins: dict[str, int] = Field(default_factory=dict)
    action_types: dict[str, int] = Field(default_factory=dict)
    has_flags: dict[str, int] = Field(default_factory=dict)
    omitted: dict[str, int] = Field(default_factory=dict)
    total_sessions: int = 0
    total_messages: int = 0


FacetFamilyRouteState = Literal["complete", "deferred", "error"]


class FacetFamilyStatusPayload(SurfacePayloadModel):
    """Route-visible status for one facet family.

    The web workbench consumes this metadata to keep cheap first paint
    truthful when optional archive-wide families are deferred or budgeted.
    Existing bucket fields remain present on :class:`FacetsResponse`; this
    status map says whether an empty bucket means "actually empty" or
    "not materialized in this response" (#2304).
    """

    state: FacetFamilyRouteState
    reason: str | None = None
    error: str | None = None
    stale: bool = False
    stale_age_s: float | None = None
    label: str | None = None
    source: str | None = None
    canonicalization: str | None = None
    expensive: bool = False


class FacetsResponse(SurfacePayloadModel):
    """Shared facets response envelope with scope semantics.

    Top-level fields (``origins``, ``tags`` etc.) carry the *active*
    view — scoped when ``scoped_to_query`` is true, global otherwise.
    Consumers that need both views read the explicit :attr:`scoped` and
    :attr:`global_` payloads.
    """

    scoped_to_query: bool = False
    generated_at: str | None = None
    stale: bool = False
    stale_age_s: float | None = None
    budget_exceeded: bool = False
    complete_families: tuple[str, ...] = ()
    deferred_families: dict[str, str] = Field(default_factory=dict)
    family_errors: dict[str, str] = Field(default_factory=dict)
    family_status: dict[str, FacetFamilyStatusPayload] = Field(default_factory=dict)
    origins: dict[str, int] = Field(default_factory=dict)
    tags: dict[str, int] = Field(default_factory=dict)
    repos: dict[str, int] = Field(default_factory=dict)
    cwd_prefixes: dict[str, int] = Field(default_factory=dict)
    role_counts: dict[str, int] = Field(default_factory=dict)
    message_types: dict[str, int] = Field(default_factory=dict)
    material_origins: dict[str, int] = Field(default_factory=dict)
    action_types: dict[str, int] = Field(default_factory=dict)
    has_flags: dict[str, int] = Field(default_factory=dict)
    omitted_facet_counts: dict[str, int] = Field(default_factory=dict)
    time_range: FacetTimeRange | None = None
    total_sessions: int = 0
    total_messages: int = 0
    # #1269: scoped/global pair plus optional IDF weighting. The global
    # field is named ``global_`` in Python (``global`` is a reserved word)
    # but serializes as ``global`` over JSON via the field alias.
    scoped: FacetBucketsPayload = Field(default_factory=FacetBucketsPayload)
    global_: FacetBucketsPayload = Field(
        default_factory=FacetBucketsPayload,
        alias="global",
        serialization_alias="global",
    )
    idf: dict[str, dict[str, float]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ContextPreambleAssertionGuidance(SurfacePayloadModel):
    """Assertion guidance eligible for explicit context injection."""

    kind: str
    text: str | None = None
    target_ref: str | None = None
    scope_ref: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleGuidance(SurfacePayloadModel):
    """Structured guidance carried by context preamble payloads."""

    assertions: list[ContextPreambleAssertionGuidance] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreamble(SurfacePayloadModel):
    """Typed delivery envelope for SessionStart context injection (#1696).

    Assembled by polylogue MCP tools and delivered via SessionStart hook
    scripts so new agent sessions receive relevant context: prior sessions,
    open issues, git state, lineage, guidance.
    """

    preamble_version: str = "1.0"
    injected_at: str | None = None
    source_tool_calls: dict[str, str] = Field(default_factory=dict)

    session_lineage: ContextPreambleLineage | None = None
    recent_related_sessions: list[ContextPreambleSession] = Field(default_factory=list)
    open_issues: list[ContextPreambleIssue] = Field(default_factory=list)
    project_state: ContextPreambleProjectState | None = None
    blackboard_notes: list[ContextPreambleBlackboardNote] = Field(default_factory=list)
    guidance: str | ContextPreambleGuidance | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleLineage(SurfacePayloadModel):
    """Session lineage for the context preamble."""

    logical_session_root: str | None = None
    parent_session_id: str | None = None
    sibling_session_ids: list[str] = Field(default_factory=list)
    continuation_chain_depth: int = 0

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleSession(SurfacePayloadModel):
    """A recent related session for the context preamble."""

    session_id: str
    title: str | None = None
    date: str | None = None
    terminal_state: str | None = None
    summary: str | None = None
    origin: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleIssue(SurfacePayloadModel):
    """A GitHub issue for the context preamble."""

    number: int
    title: str
    state: str = "open"
    labels: list[str] = Field(default_factory=list)
    active_session: str | None = None
    url: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleProjectState(SurfacePayloadModel):
    """Current project state for the context preamble."""

    repo: str | None = None
    branch: str | None = None
    recent_commits: list[str] = Field(default_factory=list)
    active_worktrees: list[str] = Field(default_factory=list)
    dirty_files: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextPreambleBlackboardNote(SurfacePayloadModel):
    """A blackboard note for the context preamble."""

    key: str
    content: str
    repo: str | None = None
    created_at: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class MutationResultPayload(SurfacePayloadModel):
    """Shared result envelope for user-visible mutation surfaces.

    Carries idempotent status codes, context fields, and bulk-operation
    counts so that CLI, MCP, API, and daemon surfaces all expose the
    same mutation contract shape.
    """

    status: MutationStatus
    """Closed status for user-visible mutation surfaces."""

    session_id: str | None = None
    detail: str | None = None
    """Machine-readable detail: ``already_present``, ``tag_not_present``,
    ``updated``, ``key_not_found``, ``value_unchanged``, ``session_not_found``."""

    outcome: MutationOutcome | None = None
    """Optional idempotency outcome for resource-level mutations."""

    affected_count: int | None = None
    skipped_count: int | None = None
    tag: str | None = None
    key: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    message_id: str | None = None
    mark_type: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    session_count: int | None = None
    tag_count: int | None = None
    applied_count: int | None = None
    operation: MutationOperation | None = None
    """Closed mutation discriminator for surfaces that expose operation names."""
    session_ids: tuple[str, ...] | None = None
    """Session ids enumerated by a CLI bulk operation (e.g. the delete dry-run
    preview lists the sessions that *would* be deleted). ``None`` for
    single-session surfaces, which carry the lone id in ``session_id``."""


# ---------------------------------------------------------------------------
# Payload builder helpers
# ---------------------------------------------------------------------------


def _session_repo(session: object) -> str | None:
    repo = getattr(session, "git_repository_url", None)
    return str(repo) if repo else None


def _session_cwd(session: object) -> str | None:
    directories = getattr(session, "working_directories", ()) or ()
    for directory in directories:
        text = str(directory).strip()
        if text:
            return text
    return None


def _build_flags_from_session(session: object) -> SessionFlagsPayload | None:
    has_tool = bool(getattr(session, "has_tool_use", None))
    has_thinking = bool(getattr(session, "has_thinking", None))
    has_paste = bool(getattr(session, "has_paste", None))
    if not has_tool and not has_thinking and not has_paste:
        return None
    return SessionFlagsPayload(has_tool_use=has_tool, has_thinking=has_thinking, has_paste=has_paste)


METADATA_KEY_MAX_LENGTH = 200
"""Centralized maximum length for user metadata keys."""


class MetadataKeyValidationError(ValueError):
    """Raised when a user metadata key fails the centralized validation rules."""


def validate_metadata_key(key: object) -> str | None:
    """Validate a user metadata key.

    Returns a machine-readable error message string, or ``None`` if the key
    is valid. Centralized so CLI, MCP, daemon, and API surfaces enforce the
    same constraints (issue #862).
    """
    if not isinstance(key, str) or not key or not key.strip():
        return "metadata key must not be empty"
    if len(key) > METADATA_KEY_MAX_LENGTH:
        return f"metadata key exceeds {METADATA_KEY_MAX_LENGTH} characters"
    return None


__all__ = [
    "ActionQueryRowPayload",
    "AssertionClaimListPayload",
    "AssertionClaimPayload",
    "AssertionJudgmentPayload",
    "AssertionJudgmentResultPayload",
    "BlockQueryRowPayload",
    "BulkTagMutationResult",
    "SessionDetailPayload",
    "SessionDetailResponse",
    "SessionFlagsPayload",
    "SessionListResponse",
    "SessionListRowPayload",
    "SessionMessageRowPayload",
    "SessionMessagesResponsePayload",
    "SessionMessagePayload",
    "ContextPreamble",
    "ContextPreambleBlackboardNote",
    "ContextPreambleIssue",
    "ContextPreambleLineage",
    "ContextPreambleProjectState",
    "ContextPreambleSession",
    "SessionNeighborCandidatePayload",
    "SessionNeighborReasonPayload",
    "SessionSearchHitPayload",
    "SessionSearchMatchPayload",
    "SessionReadViewEnvelope",
    "SessionSummaryPayload",
    "FacetBucketsPayload",
    "FacetTimeRange",
    "FacetFamilyStatusPayload",
    "FacetsResponse",
    "ArchiveDebtActionPayload",
    "ArchiveDebtKind",
    "ArchiveDebtListPayload",
    "ArchiveDebtRowPayload",
    "ArchiveDebtSeverity",
    "ArchiveDebtStatus",
    "ArchiveDebtTotalsPayload",
    "ImportDetectorEvidencePayload",
    "ImportExplainEntryPayload",
    "ImportExplainPayload",
    "ImportProducedRowsPayload",
    "ImportSkippedRowPayload",
    "CompletenessItemStatus",
    "ProviderCompletenessItemPayload",
    "ProviderCompletenessStatus",
    "ProviderPackageCompletenessPayload",
    "ProviderPackageCompletenessRowPayload",
    "ProviderPackageCompletenessTotalsPayload",
    "PublicRefResolutionPayload",
    "MachineErrorPayload",
    "MachineErrorEnvelope",
    "MachineSuccessEnvelope",
    "MachineSuccessPayload",
    "METADATA_KEY_MAX_LENGTH",
    "MetadataKeyValidationError",
    "MetadataMutationOutcome",
    "MetadataMutationResult",
    "DeleteSessionOutcome",
    "DeleteSessionResult",
    "MutationResultPayload",
    "QueryErrorPayload",
    "QueryUnitAggregateEnvelope",
    "QueryUnitAggregateRowPayload",
    "QueryUnitEnvelope",
    "FileQueryRowPayload",
    "QueryUnitKind",
    "QueryUnitResultEnvelope",
    "QueryUnitRowPayload",
    "RunQueryRowPayload",
    "QueryMissDiagnosticsPayload",
    "QueryMissReasonPayload",
    "RouteReadinessPayload",
    "RouteReadinessState",
    "ContextSnapshotQueryRowPayload",
    "ObservedEventQueryRowPayload",
    "OtelLogRecordPayload",
    "OtelProjectionPayload",
    "OtelSpanPayload",
    "RecoveryReadPayload",
    "RecoveryReportFormat",
    "RecoveryReportKind",
    "RefResolutionActionPayload",
    "RANKING_POLICY_MIXED",
    "RANKING_POLICY_VERSION",
    "ReaderActionAvailabilityPayload",
    "SEARCH_CURSOR_VERSION",
    "SearchCursor",
    "SearchEnvelope",
    "MessageQueryRowPayload",
    "AssertionQueryRowPayload",
    "InvalidSearchCursorError",
    "SurfacePayloadModel",
    "TagMutationOutcome",
    "TagMutationResult",
    "TargetRefPayload",
    "JSONDocument",
    "JSONValue",
    "apply_search_cursor",
    "build_search_cursor",
    "build_search_envelope",
    "build_query_unit_aggregate_envelope",
    "build_query_unit_envelope",
    "decode_search_cursor",
    "model_json_document",
    "normalize_role",
    "reader_anchor",
    "reader_session_actions",
    "reader_message_actions",
    "search_cursor_lane_matches_request",
    "serialize_surface_payload",
    "validate_metadata_key",
]
