"""Typed archive-scoped daemon operation envelopes.

The envelope deliberately carries readiness and authority with the result so
clients do not need a health/probe request before every operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polylogue.core.enums import OperationStatus
from polylogue.operations.machine_receipts import IngestHistoricalReceipt
from polylogue.operations.read_contracts import (
    QueryAggregateRequest,
    QueryAggregateResult,
    SessionReadRequest,
    SessionReadResult,
    SessionReferenceRequest,
    SessionReferenceResult,
)

DAEMON_OPERATION_PROTOCOL = "polylogue.daemon-operation/v1"
MAX_OPERATION_BODY_BYTES = 64 * 1024
MAX_OPERATION_RESULT_BYTES = 8 * 1024 * 1024


class DaemonAuthority(StrEnum):
    READ = "read"
    WRITE = "write"
    CONTROL = "control"
    LONG_RUNNING = "long-running"


class DaemonFallback(StrEnum):
    DIRECT_READ = "direct-read"
    NEVER = "never"


DAEMON_OPERATION_OUTCOMES = frozenset(
    {
        *(status.value for status in OperationStatus),
        "cancelled",
        "timed-out",
        "disconnected-before-acceptance",
        "disconnected-after-acceptance",
        "indeterminate",
        "restarted",
    }
)


class _OperationPayload(BaseModel):
    """Base for a concrete machine-operation payload type."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class StatusRequest(_OperationPayload):
    include_archive_readiness: bool = False


class QueryRequest(_OperationPayload):
    params: dict[str, object] = Field(default_factory=dict)


class QueryUnitsRequest(QueryRequest):
    pass


class CompletionRequest(_OperationPayload):
    """One shell-completion question.

    ``source`` selects the archive-backed value vocabularies (session ids,
    user tags, repositories, tool names); every other ``kind`` is answered from
    the declared query grammar alone and needs no archive. A completer runs on
    every TAB, so ``limit`` is part of the request rather than a server
    default: the shell wants a short list quickly, not a complete one.
    """

    kind: str = "field"
    incomplete: str = ""
    unit: str | None = None
    field: str | None = None
    source: str | None = None
    limit: int = Field(default=32, ge=1, le=200)


class FacetsRequest(QueryRequest):
    pass


class IngestRequest(_OperationPayload):
    path: str = Field(min_length=1)
    source_path: str | None = None
    idempotency_key: str | None = None


class InsightRebuildRequest(_OperationPayload):
    session_ids: list[str] | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def nonempty_identifiers(self) -> InsightRebuildRequest:
        if self.session_ids is not None and any(not value for value in self.session_ids):
            raise ValueError("session identifiers must be nonempty")
        return self


class DeletePreviewRequest(_OperationPayload):
    session_ids: list[str] = Field(min_length=1, max_length=10_000)


class DeleteAuthorizeRequest(_OperationPayload):
    preview_ref: str | None = Field(default=None, min_length=1)
    preview_refs: list[str] | None = Field(default=None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def exact_reference_shape(self) -> DeleteAuthorizeRequest:
        if (self.preview_ref is None) == (self.preview_refs is None):
            raise ValueError("supply exactly one of preview_ref or preview_refs")
        if self.preview_refs is not None and (
            any(not ref for ref in self.preview_refs) or len(set(self.preview_refs)) != len(self.preview_refs)
        ):
            raise ValueError("preview_refs must be distinct nonempty references")
        return self


class DeleteCancelRequest(DeleteAuthorizeRequest):
    pass


class DeleteExecuteRequest(_OperationPayload):
    authorization_ref: str | None = Field(default=None, min_length=1)
    authorization_refs: list[str] | None = Field(default=None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def exact_reference_shape(self) -> DeleteExecuteRequest:
        if (self.authorization_ref is None) == (self.authorization_refs is None):
            raise ValueError("supply exactly one of authorization_ref or authorization_refs")
        if self.authorization_refs is not None and (
            any(not ref for ref in self.authorization_refs)
            or len(set(self.authorization_refs)) != len(self.authorization_refs)
        ):
            raise ValueError("authorization_refs must be distinct nonempty references")
        return self


class SessionTagRequest(_OperationPayload):
    """Add, or remove, user tags across a matched selection.

    Exactly one direction per request: the add path is the audited chunked
    batch (``BulkTagActuator``), the remove path is a per-target actuator
    cycle, and a request that meant both would have to report two different
    receipt shapes as one. A surface that wants both sends two requests.
    """

    session_ids: list[str] = Field(min_length=1, max_length=10_000)
    tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_direction(self) -> SessionTagRequest:
        if bool(self.tags) == bool(self.remove_tags):
            raise ValueError("supply exactly one of tags or remove_tags")
        if any(not value.strip() for value in (*self.tags, *self.remove_tags)):
            raise ValueError("tags must be nonempty")
        return self


class SessionMetadataRequest(_OperationPayload):
    session_ids: list[str] = Field(min_length=1, max_length=10_000)
    pairs: list[list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_pairs(self) -> SessionMetadataRequest:
        from polylogue.surfaces.payloads import validate_metadata_key

        for pair in self.pairs:
            if len(pair) != 2:
                raise ValueError("metadata pairs must contain exactly a key and value")
            error = validate_metadata_key(pair[0])
            if error is not None:
                raise ValueError(error)
        return self


class SessionMarkRequest(_OperationPayload):
    """Star/pin/archive marks over whole sessions.

    Session-scoped by name and by contract: a message- or block-targeted mark
    needs the async insight-target resolver that still lives on the Python
    facade, so this operation deliberately carries session ids only and the
    handler resolves each one against the index with the durable user-tier
    alias fallback.
    """

    session_ids: list[str] = Field(min_length=1, max_length=10_000)
    add_marks: list[str] = Field(default_factory=list)
    remove_marks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonempty_disjoint_marks(self) -> SessionMarkRequest:
        if not self.add_marks and not self.remove_marks:
            raise ValueError("supply at least one mark to add or remove")
        if any(not value.strip() for value in (*self.add_marks, *self.remove_marks)):
            raise ValueError("mark types must be nonempty")
        if set(self.add_marks) & set(self.remove_marks):
            raise ValueError("a mark cannot be added and removed in one request")
        if any(not value for value in self.session_ids):
            raise ValueError("session identifiers must be nonempty")
        return self


class AnnotationSaveRequest(_OperationPayload):
    """One create-or-update of a session-scoped annotation body."""

    annotation_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1)
    note_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def nonblank_text(self) -> AnnotationSaveRequest:
        if not self.note_text.strip():
            raise ValueError("note_text must not be blank")
        if not self.annotation_id.strip():
            raise ValueError("annotation_id must not be blank")
        return self


class JudgmentRecordRequest(_OperationPayload):
    """Durable judgment writes: one comparative judgment, or a review batch.

    The two families share an operation because they share an authority and a
    tier (``user.db`` assertion rows) and differ only in body. ``judgment_kind``
    is the discriminator; exactly one body field is present.
    """

    judgment_kind: Literal["comparative", "assertion-review"]
    comparative: dict[str, object] | None = None
    author_kind: str = "user"
    reviews: list[dict[str, object]] | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def exact_body_for_kind(self) -> JudgmentRecordRequest:
        if self.judgment_kind == "comparative":
            if self.comparative is None or self.reviews is not None:
                raise ValueError("a comparative judgment carries exactly a comparative body")
        elif self.reviews is None or self.comparative is not None:
            raise ValueError("an assertion review carries exactly a reviews batch")
        elif not self.reviews:
            raise ValueError("an assertion review batch must not be empty")
        if not self.author_kind.strip():
            raise ValueError("author_kind must not be blank")
        return self


class SessionExcisionRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=4096)
    actor: str = Field(min_length=1, max_length=512)
    cascade_lineage: bool = False


class SessionLifecycleRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    mode: Literal["mirror", "primary"]
    reason: str = Field(min_length=1, max_length=4096)
    actor: str = Field(min_length=1, max_length=512)


class IdentityResetRequest(_OperationPayload):
    session_ids: list[str] = Field(min_length=1, max_length=10_000)
    reason: str = Field(min_length=1, max_length=4096)


class RawAuthorityBlockerResolveRequest(_OperationPayload):
    blocker_id: str = Field(min_length=1)
    resolution: str = Field(min_length=1, max_length=4096)
    assertion_id: str | None = None
    judgment_disposition: Literal["retain_canonical_authority"] | None = None


class ResetRequest(_OperationPayload):
    index: bool = False
    database: bool = False
    include_user_db: bool = False
    include_source_db: bool = False
    blob: bool = False
    assets: bool = False
    cache: bool = False
    auth: bool = False
    reset_all: bool = False


class BlobGCRecoverRequest(_OperationPayload):
    generation_id: str = Field(min_length=1)


class DemoAugmentRequest(_OperationPayload):
    with_overlays: bool = False


class OperationStatusRequest(_OperationPayload):
    request_id: str = Field(min_length=1)


class OperationAwaitRequest(OperationStatusRequest):
    after_sequence: int = Field(default=0, ge=0)
    timeout_ms: int = Field(default=30_000, ge=1, le=30_000)


class OperationCancelRequest(OperationStatusRequest):
    pass


#: Sentinel distinguishing "the key is absent" from "the key is an honest null".
_MISSING = object()


class _OperationResult(BaseModel):
    """Base for declared result payloads; envelopes own authority metadata."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


class StatusResult(_OperationResult):
    total_sessions: int = Field(ge=0)
    total_messages: int = Field(ge=0)
    archive_stats: dict[str, object]


class QueryResult(_OperationResult):
    @model_validator(mode="before")
    @classmethod
    def canonical_query_contract(cls, value: object) -> object:
        from polylogue.surfaces.payloads import SearchEnvelope, SessionListResponse

        if not isinstance(value, dict):
            raise ValueError("query result must be an object")
        payload = dict(value)
        # The ``with <units>`` projection is a sibling of the canonical list
        # envelope, not a field of it: the envelope forbids extras and lives
        # inside the derived-schema closure, which a CLI projection must not
        # move.  Validate its shape here and hand the envelope the rest.
        attached = payload.pop("attached_units", None)
        if attached is not None and not (
            isinstance(attached, dict)
            and all(
                isinstance(unit, str)
                and isinstance(by_session, dict)
                and all(isinstance(rows, list) for rows in by_session.values())
                for unit, by_session in attached.items()
            )
        ):
            raise ValueError("attached units must map each unit to per-session row lists")
        # ``total_unit`` is a sibling of both envelopes for the same reason as
        # the projection above: it names what the total counted (top-level
        # sessions, or subagent/branch children under ``--no-root``), and the
        # envelopes forbid extras.  A ranked page owes that label exactly as a
        # list page does — without it a reader has only a default to fall back
        # on, which mislabels every non-default root filter.
        unit = payload.pop("total_unit", None)
        if not isinstance(unit, str) or not unit:
            raise ValueError("session query result requires its total unit")
        # ``next_offset`` is a sibling for the third time and the same reason:
        # both envelopes forbid extras and both live in the derived-schema
        # closure.  Every page owes it, because a client that must see the
        # complete matched set (a mutating verb's cardinality guard) walks it,
        # and its absence read as "this page was the last one" -- which let
        # ``delete --all`` act on the first page only (polylogue-w3s0q).
        next_offset = payload.pop("next_offset", _MISSING)
        if next_offset is _MISSING:
            raise ValueError("session query result requires its next-page offset")
        if not (next_offset is None or (isinstance(next_offset, int) and not isinstance(next_offset, bool))):
            raise ValueError("next_offset must be an integer offset or null")
        if "items" in payload:
            SessionListResponse.model_validate_json(json.dumps(payload), strict=True)
        else:
            SearchEnvelope.model_validate_json(json.dumps(payload), strict=True)
        return value


class QueryUnitsResult(_OperationResult):
    @model_validator(mode="before")
    @classmethod
    def canonical_unit_contract(cls, value: object) -> object:
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope, QueryUnitEnvelope

        if not isinstance(value, dict):
            raise ValueError("query unit result must be an object")
        model = QueryUnitAggregateEnvelope if value.get("mode") == "query-unit-aggregate" else QueryUnitEnvelope
        model.model_validate_json(json.dumps(value), strict=True)
        return value


class CompletionCandidateResult(_OperationPayload):
    value: str
    insert: str
    display: str
    kind: str
    group: str
    description: str
    source: str
    replace_start: int | None
    replace_end: int | None
    stale: bool
    danger: bool
    score: float
    payload_model: str | None
    unsupported_reason: str | None
    preview_command: str | None
    route: dict[str, object] | None


class CompletionCandidatesResult(_OperationPayload):
    kind: str
    incomplete: str
    unit: str | None
    field: str | None
    candidates: list[CompletionCandidateResult]


class CompletionValueResult(_OperationPayload):
    """One archive-backed completion value and the count that explains it."""

    value: str
    help: str | None = None


class CompletionValuesResult(_OperationPayload):
    source: str
    incomplete: str
    values: list[CompletionValueResult]


class CompletionResult(_OperationPayload):
    """Exactly one of the two completion vocabularies.

    A grammar completion reports ``query_completions``; an archive-backed value
    completion reports ``value_completions``. They are separate fields rather
    than one polymorphic list because they answer different questions and a
    consumer must not have to guess which it received.
    """

    query_completions: CompletionCandidatesResult | None = None
    value_completions: CompletionValuesResult | None = None

    @model_validator(mode="after")
    def exactly_one_vocabulary(self) -> CompletionResult:
        """A completion result carries one answer, never none and never both.

        Without this an empty document validates, and "the archive has no
        matching tags" would be indistinguishable from "nothing answered this
        request" -- a completer that silently produces nothing forever.
        """
        if (self.query_completions is None) == (self.value_completions is None):
            raise ValueError("supply exactly one of query_completions or value_completions")
        return self


class FacetsResult(_OperationResult):
    @model_validator(mode="before")
    @classmethod
    def canonical_facets_contract(cls, value: object) -> object:
        from polylogue.surfaces.payloads import FacetsResponse

        FacetsResponse.model_validate_json(json.dumps(value), strict=True)
        return value


class IngestResult(_OperationResult):
    source_generation_id: str = Field(min_length=1)
    outcome: OperationStatus
    sequence: int = Field(ge=0)
    historical_receipt: IngestHistoricalReceipt

    @model_validator(mode="after")
    def binds_terminal_receipt(self) -> IngestResult:
        if self.outcome is not OperationStatus.COMPLETED:
            raise ValueError("ingest terminal result must be completed")
        if self.source_generation_id != self.historical_receipt.source_generation_id:
            raise ValueError("ingest result and historical receipt disagree on source generation")
        if self.sequence != self.historical_receipt.final_sequence:
            raise ValueError("ingest result and historical receipt disagree on terminal sequence")
        return self


class InsightRebuildResult(_OperationPayload):
    profiles: int = Field(ge=0)
    work_events: int = Field(ge=0)
    phases: int = Field(ge=0)
    threads: int = Field(ge=0)
    tag_rollups: int = Field(ge=0)


class MutationResult(_OperationPayload):
    status: Literal["prepared", "authorized", "cancelled"] | None = None
    operation: str | None = None
    preview_ref: str | None = None
    preview_refs: list[str] | None = None
    authorization_ref: str | None = None
    authorization_refs: list[str] | None = None
    session_ids: list[str] | None = None
    session_count: int | None = Field(default=None, ge=0)
    expires_at_ms: int | None = None
    outcome: str | None = None
    sequence: int | None = Field(default=None, ge=0)
    reference: dict[str, object] | None = None
    effect: Literal["committed", "no-effect", "indeterminate"] | None = None
    completed_chunks: int | None = Field(default=None, ge=0)
    affected_count: int | None = Field(default=None, ge=0)
    not_attempted: list[int] | None = None
    parts: list[dict[str, object]] | None = None
    stop_reason: str | None = None
    artifact_refs: list[str] | None = None
    result: dict[str, object] | None = None
    cancellation_requested: bool | None = None
    accepted: bool | None = None
    # polylogue-oil1q: ``machine_request_state`` reports the accepted frozen
    # source generation for a ``source-generation`` artifact (every ingest).
    # Declaring it keeps ``extra="forbid"`` meaningful instead of making a
    # clean, committed ingest fail its own await contract and surface to the
    # client as ``DaemonMutationIndeterminateError``.
    source_generation_id: str | None = None

    @model_validator(mode="after")
    def exact_result_family(self) -> MutationResult:
        if self.status is None:
            if self.outcome not in DAEMON_OPERATION_OUTCOMES or self.sequence is None:
                raise ValueError("mutation lifecycle result requires outcome and durable sequence")
        elif self.status == "prepared":
            if not self.preview_refs or self.preview_ref != self.preview_refs[0] or self.session_ids is None:
                raise ValueError("prepared result requires exact preview references and selection")
            if self.session_count != len(self.session_ids) or self.expires_at_ms is None:
                raise ValueError("prepared result requires selection count and expiry")
        elif self.status == "authorized":
            if not self.authorization_refs or self.authorization_ref != self.authorization_refs[0]:
                raise ValueError("authorized result requires exact authorization references")
        elif not self.preview_refs:
            raise ValueError("cancelled preview result requires exact preview references")
        return self


class AcceptedOperationReference(_OperationPayload):
    """Immutable reference returned after durable admission of long work."""

    archive_identity: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    principal_ref: str = Field(min_length=1)
    fingerprint: str = Field(pattern="^[0-9a-f]{64}$")
    operation_name: str = Field(min_length=1)
    artifact_kind: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)
    accepted_at_ms: int = Field(ge=0)
    part_count: int = Field(ge=1, le=4096)
    accepted_deadline_unix_ms: int | None

    @model_validator(mode="after")
    def operation_part_bound(self) -> AcceptedOperationReference:
        if self.operation_name != "maintenance.insights.rebuild" and self.part_count > 40:
            raise ValueError("operation exceeds its forty-part acceptance bound")
        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AcceptedOperationReference:
        """Project immutable acceptance facts from the mutable lifecycle row."""
        return cls.model_validate({key: record[key] for key in cls.model_fields})


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    """Coherent authority evidence attached to every result."""

    archive_identity: str
    generation: str
    schema_versions: dict[str, int]
    served_by: str
    elapsed_ms: int = 0
    queue_ms: int = 0
    degraded_components: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_identity": self.archive_identity,
            "generation": self.generation,
            "schema_versions": dict(self.schema_versions),
            "served_by": self.served_by,
            "elapsed_ms": self.elapsed_ms,
            "queue_ms": self.queue_ms,
            "degraded_components": list(self.degraded_components),
        }


@dataclass(frozen=True, slots=True)
class DaemonOperationError:
    code: str
    detail: str
    retryable: bool = False
    outcome: OperationStatus = OperationStatus.FAILED

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "retryable": self.retryable,
            "outcome": self.outcome.value,
        }


@dataclass(frozen=True, slots=True)
class DaemonOperationSpec:
    """The machine operation authority shared by every daemon surface.

    This is deliberately metadata only.  Implementations remain in the
    canonical operation/facade layer and adapters only serialize this shape.
    """

    name: str
    authority: DaemonAuthority
    fallback: DaemonFallback
    capability: str = "read"
    deadline_s: float = 2.0
    cancellable: bool = True
    progress: bool = False
    accepted_reference: bool = False
    request_contract: str = "object"
    result_contract: str = "object"
    max_body_bytes: int = MAX_OPERATION_BODY_BYTES
    """Bound on this operation's request body; a selection carries more than a
    parameter map, so the limit belongs to the operation, not the transport."""
    error_contract: str = "polylogue.daemon-error/v1"
    authority_metadata: tuple[str, ...] = (
        "archive",
        "generation",
        "served_by",
        "elapsed_ms",
        "queue_ms",
        "degraded_components",
    )
    additional_capabilities: tuple[str, ...] = ()
    """Capabilities this operation exercises beyond the one it is named by.

    ``capability`` names an operation's primary authority, which is enough while
    an operation does exactly one thing. ``mutation.session.mark`` carries both
    halves of one reversible mark edit, so its primary capability
    (``archive.add_mark``) does not cover the removal its own request model
    accepts. Declaring the remainder here keeps an operation's full authority on
    the operation, rather than in a side table the declaration cannot see.
    """
    request_type: str = ""
    result_type: str = ""
    request_model: type[BaseModel] = _OperationPayload
    result_model: type[BaseModel] = _OperationResult
    idempotent: bool = False
    handler: str = ""
    handler_module: str = "polylogue.operations.daemon_mutations"
    """Module that owns :attr:`handler`.

    Execution resolves the handler on this module, so a spec that names a
    function living elsewhere is a declaration error rather than a dispatch-time
    ``AttributeError`` (polylogue-ms4uy). Read operations dispatch through
    :mod:`polylogue.operations.daemon_reads` and ``operation.*`` control
    operations through the daemon runtime, so neither consults this field.
    """
    cancellation_outcomes: tuple[str, ...] = (
        "cancelled",
        "timed-out",
        "disconnected-before-acceptance",
        "disconnected-after-acceptance",
        "indeterminate",
    )

    def __post_init__(self) -> None:
        if self.direct_allowed and self.authority is not DaemonAuthority.READ:
            raise ValueError("only read operations may permit direct fallback")
        if self.request_model is _OperationPayload or self.result_model is _OperationResult:
            raise ValueError("operation declarations require concrete request and result models")
        if not self.handler:
            object.__setattr__(self, "handler", self.name.replace(".", "_"))
        if self.request_type and self.result_type:
            return
        stem = "".join(part.capitalize() for part in self.name.replace(".", "-").split("-"))
        object.__setattr__(self, "request_type", self.request_type or f"{stem}Request")
        object.__setattr__(self, "result_type", self.result_type or f"{stem}Result")

    @property
    def direct_allowed(self) -> bool:
        return self.fallback is DaemonFallback.DIRECT_READ

    def to_dict(self) -> dict[str, object]:
        """Serialize the declaration for discovery and conformance checks."""
        return {
            "name": self.name,
            "authority": self.authority.value,
            "fallback": self.fallback.value,
            "capability": self.capability,
            "additional_capabilities": list(self.additional_capabilities),
            "deadline_s": self.deadline_s,
            "cancellable": self.cancellable,
            "progress": self.progress,
            "accepted_reference": self.accepted_reference,
            "request_contract": self.request_contract,
            "result_contract": self.result_contract,
            "max_body_bytes": self.max_body_bytes,
            "error_contract": self.error_contract,
            "authority_metadata": list(self.authority_metadata),
            "request_type": self.request_type,
            "result_type": self.result_type,
            "request_model": self.request_model.__name__,
            "result_model": self.result_model.__name__,
            "idempotent": self.idempotent,
            "handler": self.handler,
            "cancellation_outcomes": list(self.cancellation_outcomes),
        }


DAEMON_OPERATION_SPECS: tuple[DaemonOperationSpec, ...] = (
    DaemonOperationSpec(
        "maintenance.insights.rebuild",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.rebuild_insights",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="maintenance.insights.rebuild.request/v1",
        result_contract="maintenance.insights.rebuild.result/v1",
        request_type="InsightRebuildRequest",
        result_type="InsightRebuildResult",
        request_model=InsightRebuildRequest,
        result_model=InsightRebuildResult,
        idempotent=True,
        handler="execute_insights_rebuild_operation",
        handler_module="polylogue.operations.daemon_insights",
    ),
    DaemonOperationSpec(
        "operation.status",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="read",
        request_model=OperationStatusRequest,
        result_model=MutationResult,
        handler="operation_status",
    ),
    DaemonOperationSpec(
        "operation.await",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="read",
        deadline_s=30.0,
        request_model=OperationAwaitRequest,
        result_model=MutationResult,
        handler="operation_await",
    ),
    DaemonOperationSpec(
        "operation.cancel",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="read",
        request_model=OperationCancelRequest,
        result_model=MutationResult,
        handler="operation_cancel",
    ),
    DaemonOperationSpec(
        "cli.query",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="cli.query.result/v1",
        request_type="QueryRequest",
        result_type="QueryResult",
        request_model=QueryRequest,
        result_model=QueryResult,
    ),
    DaemonOperationSpec(
        "query.units",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="query.units.result/v1",
        request_type="QueryUnitsRequest",
        result_type="QueryUnitsResult",
        request_model=QueryUnitsRequest,
        result_model=QueryUnitsResult,
    ),
    DaemonOperationSpec(
        "query.aggregate",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        # Aggregates scan the selection rather than one page of it.
        deadline_s=10.0,
        result_contract="query.aggregate.result/v1",
        request_type="QueryAggregateRequest",
        result_type="QueryAggregateResult",
        request_model=QueryAggregateRequest,
        result_model=QueryAggregateResult,
    ),
    DaemonOperationSpec(
        "session.read",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="session.read.result/v1",
        request_type="SessionReadRequest",
        result_type="SessionReadResult",
        request_model=SessionReadRequest,
        result_model=SessionReadResult,
    ),
    DaemonOperationSpec(
        "session.reference",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        deadline_s=5.0,
        result_contract="session.reference.result/v1",
        request_type="SessionReferenceRequest",
        result_type="SessionReferenceResult",
        request_model=SessionReferenceRequest,
        result_model=SessionReferenceResult,
    ),
    DaemonOperationSpec(
        "status",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="status.result/v1",
        request_type="StatusRequest",
        result_type="StatusResult",
        request_model=StatusRequest,
        result_model=StatusResult,
    ),
    DaemonOperationSpec(
        "completion",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="completion.result/v1",
        request_type="CompletionRequest",
        result_type="CompletionResult",
        request_model=CompletionRequest,
        result_model=CompletionResult,
    ),
    DaemonOperationSpec(
        "facets",
        DaemonAuthority.READ,
        DaemonFallback.DIRECT_READ,
        result_contract="facets.result/v1",
        request_type="FacetsRequest",
        result_type="FacetsResult",
        request_model=FacetsRequest,
        result_model=FacetsResult,
    ),
    DaemonOperationSpec(
        "ingest",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.ingest",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="ingest.request/v1",
        result_contract="ingest.result/v1",
        request_type="IngestRequest",
        result_type="IngestResult",
        request_model=IngestRequest,
        result_model=IngestResult,
        handler="execute_ingest_operation",
        handler_module="polylogue.operations.daemon_ingest",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.preview",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        # A preview carries the exact selection: up to
        # ``DELETE_PREVIEW_MAX_SESSION_IDS`` session ids, not a parameter map.
        max_body_bytes=64 * 1024 * 1024,
        request_contract="mutation.session.delete.preview.request/v1",
        result_contract="mutation.session.delete.preview.result/v1",
        request_type="DeletePreviewRequest",
        result_type="MutationResult",
        request_model=DeletePreviewRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.authorize",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        request_contract="mutation.session.delete.authorize.request/v1",
        result_contract="mutation.session.delete.authorize.result/v1",
        request_type="DeleteAuthorizeRequest",
        result_type="MutationResult",
        request_model=DeleteAuthorizeRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.cancel",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=30.0,
        request_contract="mutation.session.delete.cancel.request/v1",
        result_contract="mutation.session.delete.cancel.result/v1",
        request_type="DeleteCancelRequest",
        result_type="MutationResult",
        request_model=DeleteCancelRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.execute",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="mutation.session.delete.execute.request/v1",
        result_contract="mutation.result/v1",
        request_type="DeleteExecuteRequest",
        result_type="MutationResult",
        request_model=DeleteExecuteRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.tag",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        max_body_bytes=64 * 1024 * 1024,
        capability="archive.bulk_tag_sessions",
        deadline_s=120.0,
        request_contract="mutation.session.tag.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionTagRequest",
        result_type="MutationResult",
        request_model=SessionTagRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.metadata",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        max_body_bytes=64 * 1024 * 1024,
        capability="archive.set_metadata",
        deadline_s=120.0,
        request_contract="mutation.session.metadata.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionMetadataRequest",
        result_type="MutationResult",
        request_model=SessionMetadataRequest,
        result_model=MutationResult,
    ),
    DaemonOperationSpec(
        "mutation.session.mark",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.add_mark",
        additional_capabilities=("archive.remove_mark",),
        deadline_s=120.0,
        request_contract="mutation.session.mark.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionMarkRequest",
        result_type="MutationResult",
        request_model=SessionMarkRequest,
        result_model=MutationResult,
        handler="mutation_session_mark",
    ),
    DaemonOperationSpec(
        "mutation.annotation.save",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_annotation",
        deadline_s=120.0,
        request_contract="mutation.annotation.save.request/v1",
        result_contract="mutation.result/v1",
        request_type="AnnotationSaveRequest",
        result_type="MutationResult",
        request_model=AnnotationSaveRequest,
        result_model=MutationResult,
        handler="mutation_annotation_save",
    ),
    DaemonOperationSpec(
        "mutation.judgment.record",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_judgment",
        deadline_s=120.0,
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.judgment.record.request/v1",
        result_contract="mutation.result/v1",
        request_type="JudgmentRecordRequest",
        result_type="MutationResult",
        request_model=JudgmentRecordRequest,
        result_model=MutationResult,
        handler="mutation_judgment_record",
    ),
    DaemonOperationSpec(
        "mutation.session.excision",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.excise_session",
        deadline_s=300.0,
        progress=True,
        request_contract="mutation.session.excision.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionExcisionRequest",
        result_type="MutationResult",
        request_model=SessionExcisionRequest,
        result_model=MutationResult,
        handler="mutation_session_excision",
    ),
    DaemonOperationSpec(
        "mutation.session.lifecycle-request",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.request_session_lifecycle",
        deadline_s=30.0,
        request_contract="mutation.session.lifecycle-request.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionLifecycleRequest",
        result_type="MutationResult",
        request_model=SessionLifecycleRequest,
        result_model=MutationResult,
        handler="mutation_session_lifecycle_request",
    ),
    DaemonOperationSpec(
        "mutation.identity-reset",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.identity_reset",
        deadline_s=300.0,
        progress=True,
        # IdentityResetRequest declares session_ids up to 10_000 and a 4 KiB
        # reason. At ~60 bytes per JSON-quoted session id that is ~600 KiB, so
        # the default 64 KiB body cap would refuse ~1,100 ids as
        # ``request_too_large`` while the contract still advertised 10_000 --
        # and cli/commands/reset.py submits exactly that payload. The bound the
        # request contract already promises is the honest one to admit.
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.identity-reset.request/v1",
        result_contract="mutation.result/v1",
        request_type="IdentityResetRequest",
        result_type="MutationResult",
        request_model=IdentityResetRequest,
        result_model=MutationResult,
        handler="mutation_identity_reset",
    ),
    DaemonOperationSpec(
        "mutation.raw-authority-blocker.resolve",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.raw_authority.resolve_blocker",
        deadline_s=30.0,
        request_contract="mutation.raw-authority-blocker.resolve.request/v1",
        result_contract="mutation.result/v1",
        request_type="RawAuthorityBlockerResolveRequest",
        result_type="MutationResult",
        request_model=RawAuthorityBlockerResolveRequest,
        result_model=MutationResult,
        handler="mutation_raw_authority_blocker_resolve",
    ),
    DaemonOperationSpec(
        "maintenance.reset",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.reset",
        deadline_s=300.0,
        progress=True,
        request_contract="maintenance.reset.request/v1",
        result_contract="mutation.result/v1",
        request_type="ResetRequest",
        result_type="MutationResult",
        request_model=ResetRequest,
        result_model=MutationResult,
        handler="maintenance_reset",
    ),
    DaemonOperationSpec(
        "maintenance.blob-gc.recover",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.blob_gc.abandon_pending_generation",
        deadline_s=30.0,
        request_contract="maintenance.blob-gc.recover.request/v1",
        result_contract="mutation.result/v1",
        request_type="BlobGCRecoverRequest",
        result_type="MutationResult",
        request_model=BlobGCRecoverRequest,
        result_model=MutationResult,
        handler="maintenance_blob_gc_recover",
    ),
    DaemonOperationSpec(
        "maintenance.demo.augment",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.demo_augment",
        deadline_s=120.0,
        request_contract="maintenance.demo.augment.request/v1",
        result_contract="mutation.result/v1",
        request_type="DemoAugmentRequest",
        result_type="MutationResult",
        request_model=DemoAugmentRequest,
        result_model=MutationResult,
        handler="maintenance_demo_augment",
    ),
)

MAX_DECLARED_OPERATION_BODY_BYTES: int = max(spec.max_body_bytes for spec in DAEMON_OPERATION_SPECS)
"""Transport-level read bound: the operation is only known after parsing, so
the socket read is capped by the largest declared body and the operation's own
``max_body_bytes`` is enforced once the request names it."""

MUTATION_OPERATION_NAMES: frozenset[str] = frozenset(
    spec.name for spec in DAEMON_OPERATION_SPECS if spec.authority is not DaemonAuthority.READ
)
"""Operations a CLI adapter may never execute in its own process."""

DAEMON_PRINCIPAL_CAPABILITIES: frozenset[str] = frozenset(
    capability for spec in DAEMON_OPERATION_SPECS for capability in (spec.capability, *spec.additional_capabilities)
)
"""Every capability the daemon's own operation handlers may need to exercise.

Derived from the declarations, so an operation that gains an authority gains it
here by declaring it -- there is no second list to keep in step. A principal
short one capability does not fail loudly at the call: it is denied at the
target-authority check, which reads as the mutation simply never applying.
"""

if len({spec.name for spec in DAEMON_OPERATION_SPECS}) != len(DAEMON_OPERATION_SPECS):
    raise RuntimeError("daemon operation names must be unique")


def module_dispatched_specs() -> tuple[DaemonOperationSpec, ...]:
    """Specs whose handler is resolved on :attr:`DaemonOperationSpec.handler_module`.

    Read operations are dispatched by name through
    :mod:`polylogue.operations.daemon_reads` and ``operation.*`` control
    operations by the daemon runtime; every other declaration names a handler
    function that must exist on its declared module.
    """
    return tuple(
        spec
        for spec in DAEMON_OPERATION_SPECS
        if spec.authority is not DaemonAuthority.READ and not spec.name.startswith("operation.")
    )


def resolve_operation_handler(spec: DaemonOperationSpec) -> Callable[..., Any]:
    """Resolve one declaration's handler, raising when it is not on its module."""
    import importlib

    module = importlib.import_module(spec.handler_module)
    try:
        handler = getattr(module, spec.handler)
    except AttributeError as exc:
        raise RuntimeError(
            f"operation {spec.name!r} declares handler {spec.handler!r} which does not exist on {spec.handler_module!r}"
        ) from exc
    return cast(Callable[..., Any], handler)


def validate_declared_handlers() -> None:
    """Resolve every module-dispatched handler; raise on the first that is missing.

    Registry-level validation rather than an import-time side effect: the
    handler modules import this one, so resolving them here at import would be
    circular (polylogue-ms4uy).
    """
    for spec in module_dispatched_specs():
        resolve_operation_handler(spec)


def daemon_operation_schema() -> dict[str, dict[str, object]]:
    """Derive wire schemas on discovery without adding work to client imports."""
    return {
        spec.name: {
            **spec.to_dict(),
            "request_schema": spec.request_model.model_json_schema(),
            "result_schema": spec.result_model.model_json_schema(),
        }
        for spec in DAEMON_OPERATION_SPECS
    }


def daemon_operation_spec(name: str) -> DaemonOperationSpec | None:
    return next((spec for spec in DAEMON_OPERATION_SPECS if spec.name == name), None)


class OperationResultContractError(RuntimeError):
    """An executor or peer returned a value outside its declared contract."""


def validate_operation_result(operation: str, result: object) -> None:
    """Validate without coercing or rewriting the product's wire value."""
    spec = daemon_operation_spec(operation)
    if spec is None:
        raise OperationResultContractError(f"undeclared operation: {operation}")
    try:
        spec.result_model.model_validate_json(json.dumps(result, allow_nan=False), strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise OperationResultContractError(f"invalid {operation} result: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DaemonOperationRequest:
    operation: str
    payload: dict[str, object]
    archive_root: str | None = None
    index_schema_version: int | None = None
    daemon_version: str | None = None
    expected_archive_identity: str | None = None
    expected_generation_id: str | None = None
    request_id: str | None = None
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    cancellation_token: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DaemonOperationRequest:
        if not isinstance(raw, Mapping):
            raise ValueError("operation request must be an object")
        allowed = {
            "protocol",
            "operation",
            "payload",
            "archive_root",
            "index_schema_version",
            "daemon_version",
            "expected_archive_identity",
            "expected_generation_id",
            "request_id",
            "deadline_ms",
            "idempotency_key",
            "cancellation_token",
        }
        if set(raw) - allowed:
            raise ValueError("unexpected operation request fields")
        protocol = raw.get("protocol")
        operation = raw.get("operation")
        payload = raw.get("payload", {})
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if protocol != DAEMON_OPERATION_PROTOCOL:
            raise ValueError("unsupported daemon operation protocol")
        spec = daemon_operation_spec(operation)
        if spec is None:
            raise ValueError(f"operation is not declared: {operation}")
        if len(json.dumps(dict(raw), separators=(",", ":"), allow_nan=False).encode()) > spec.max_body_bytes:
            raise ValueError("request_too_large")
        try:
            validated_payload = spec.request_model.model_validate(payload).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"invalid {spec.request_type} payload: {exc}") from exc
        archive_root = raw.get("archive_root")
        schema = raw.get("index_schema_version")
        version = raw.get("daemon_version")
        expected_archive_identity = raw.get("expected_archive_identity")
        expected_generation_id = raw.get("expected_generation_id")
        request_id = raw.get("request_id")
        deadline_ms = raw.get("deadline_ms")
        idempotency_key = raw.get("idempotency_key")
        cancellation_token = raw.get("cancellation_token")
        if archive_root is not None and not isinstance(archive_root, str):
            raise ValueError("archive_root must be a string")
        for name in (
            "archive_root",
            "daemon_version",
            "expected_archive_identity",
            "expected_generation_id",
            "idempotency_key",
            "cancellation_token",
        ):
            value = raw.get(name)
            if isinstance(value, str) and len(value) > 4096:
                raise ValueError(f"{name} exceeds 4096 characters")
        if schema is not None and (not isinstance(schema, int) or isinstance(schema, bool)):
            raise ValueError("index_schema_version must be an integer")
        if version is not None and not isinstance(version, str):
            raise ValueError("daemon_version must be a string")
        if expected_archive_identity is not None and (
            not isinstance(expected_archive_identity, str) or not expected_archive_identity
        ):
            raise ValueError("expected_archive_identity must be a non-empty string")
        if expected_generation_id is not None and (
            not isinstance(expected_generation_id, str) or not expected_generation_id
        ):
            raise ValueError("expected_generation_id must be a non-empty string")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if len(request_id) > 128:
            raise ValueError("request_id exceeds 128 characters")
        if deadline_ms is not None and (
            not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool) or deadline_ms <= 0
        ):
            raise ValueError("deadline_ms must be a positive integer")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key.strip()):
            raise ValueError("idempotency_key must be a non-empty string")
        if cancellation_token is not None and (
            not isinstance(cancellation_token, str) or not cancellation_token.strip()
        ):
            raise ValueError("cancellation_token must be a non-empty string")
        return cls(
            operation.strip(),
            validated_payload,
            archive_root,
            schema,
            version,
            expected_archive_identity,
            expected_generation_id,
            request_id,
            deadline_ms,
            idempotency_key,
            cancellation_token,
        )

    @property
    def fingerprint(self) -> str:
        """Stable exchange identity used for safe duplicate recovery."""
        intent = {
            "operation": self.operation,
            "payload": self.payload,
            "archive_root": self.archive_root,
            "expected_archive_identity": self.expected_archive_identity,
            "expected_generation_id": self.expected_generation_id,
        }
        encoded = json.dumps(intent, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "payload": self.payload,
            "archive_root": self.archive_root,
            "index_schema_version": self.index_schema_version,
            "daemon_version": self.daemon_version,
            "expected_archive_identity": self.expected_archive_identity,
            "expected_generation_id": self.expected_generation_id,
            "request_id": self.request_id,
            "deadline_ms": self.deadline_ms,
        }
        if self.idempotency_key is not None:
            result["idempotency_key"] = self.idempotency_key
        if self.cancellation_token is not None:
            result["cancellation_token"] = self.cancellation_token
        return result


@dataclass(frozen=True, slots=True)
class DaemonOperationEnvelope:
    operation: str
    archive: dict[str, object]
    generation: dict[str, object]
    readiness: dict[str, object]
    authority: dict[str, object]
    progress: dict[str, object]
    outcome: OperationStatus | str = OperationStatus.COMPLETED
    served_by: dict[str, object] | None = None
    timing: dict[str, object] | None = None
    degraded_components: tuple[str, ...] = ()
    schema_versions: dict[str, int] | None = None
    result: object = None
    error: dict[str, object] | None = None
    request_id: str | None = None
    accepted_reference: dict[str, object] | None = None
    authority_snapshot: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": self.operation,
            "archive": self.archive,
            "generation": self.generation,
            "readiness": self.readiness,
            "authority": self.authority,
            "progress": self.progress,
            "outcome": self.outcome.value if isinstance(self.outcome, OperationStatus) else self.outcome,
            "served_by": self.served_by or {},
            "timing": self.timing or {},
            "degraded_components": list(self.degraded_components),
            "schema_versions": self.schema_versions or {},
            "result": self.result,
            "error": self.error,
            "request_id": self.request_id,
            "accepted_reference": self.accepted_reference,
            "authority_snapshot": self.authority_snapshot,
        }


def archive_identity(
    archive_root: Path, *, schema_version: int, daemon_version: str
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build one archive authority snapshot without a separate probe.

    File size and mtime are not generation evidence: a WAL commit can change
    neither.  The operation boundary already has a canonical archive identity
    resolver which pins the active index generation and all tier versions.
    Keep the historical parameters for callers while deriving the returned
    evidence from that authority instead of inventing a second identity.
    """

    from polylogue.operations.authority import authority_for_root

    authority = authority_for_root(archive_root, server_identity="daemon").to_dict()
    tier_schema_versions = authority["tier_schema_versions"]
    if not isinstance(tier_schema_versions, dict):  # pragma: no cover - AuthorityEnvelope is typed
        raise RuntimeError("archive authority omitted tier schema versions")
    index_path = archive_root / "index.db"
    ready = index_path.is_file()
    actual_index_schema = tier_schema_versions.get("index")
    if not isinstance(actual_index_schema, int):  # pragma: no cover - tier registry is typed
        raise RuntimeError("archive authority omitted index schema version")
    archive: dict[str, object] = {
        "root": str(archive_root),
        "daemon_version": daemon_version,
        "index_schema_version": actual_index_schema,
        "archive_identity": authority["archive_epoch"],
        "tier_schema_versions": tier_schema_versions,
    }
    generation: dict[str, object] = {
        "id": authority["generation_id"],
        "index_schema_version": actual_index_schema,
        "tier_schema_versions": tier_schema_versions,
    }
    readiness: dict[str, object] = {
        "state": "ready" if ready else "unavailable",
        "ready": ready,
        "reason": None if ready else "index_missing",
        "degraded_components": authority["degraded"],
    }
    return archive, generation, readiness


__all__ = [
    "DAEMON_OPERATION_PROTOCOL",
    "DAEMON_OPERATION_SPECS",
    "module_dispatched_specs",
    "resolve_operation_handler",
    "validate_declared_handlers",
    "DAEMON_PRINCIPAL_CAPABILITIES",
    "MUTATION_OPERATION_NAMES",
    "MAX_DECLARED_OPERATION_BODY_BYTES",
    "MAX_OPERATION_BODY_BYTES",
    "MAX_OPERATION_RESULT_BYTES",
    "DaemonAuthority",
    "DAEMON_OPERATION_OUTCOMES",
    "DaemonFallback",
    "DaemonOperationSpec",
    "DaemonOperationEnvelope",
    "DaemonOperationRequest",
    "AcceptedOperationReference",
    "AuthoritySnapshot",
    "DaemonOperationError",
    "OperationStatus",
    "archive_identity",
    "daemon_operation_spec",
    "daemon_operation_schema",
]
