"""Typed archive-scoped daemon operation envelopes.

The envelope deliberately carries readiness and authority with the result so
clients do not need a health/probe request before every operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from polylogue.core.enums import OperationStatus
from polylogue.operations.machine_receipts import IngestHistoricalReceipt

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


class DaemonOperationOutcome(StrEnum):
    """Lifecycle outcomes for one machine exchange."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed-out"
    DISCONNECTED_BEFORE_ACCEPTANCE = "disconnected-before-acceptance"
    DISCONNECTED_AFTER_ACCEPTANCE = "disconnected-after-acceptance"
    INDETERMINATE = "indeterminate"
    RESTARTED = "restarted"
    REJECTED = "rejected"


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
    kind: str = "field"
    incomplete: str = ""
    unit: str | None = None
    field: str | None = None


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
    session_ids: list[str] = Field(min_length=1, max_length=10_000)
    tags: list[str] = Field(min_length=1)


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


class OperationStatusRequest(_OperationPayload):
    request_id: str = Field(min_length=1)


class OperationAwaitRequest(OperationStatusRequest):
    after_sequence: int = Field(default=0, ge=0)
    timeout_ms: int = Field(default=30_000, ge=1, le=30_000)


class OperationCancelRequest(OperationStatusRequest):
    pass


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
        if "items" in payload:
            unit = payload.pop("total_unit", None)
            if not isinstance(unit, str) or not unit:
                raise ValueError("session list result requires its total unit")
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


class CompletionResult(_OperationPayload):
    query_completions: CompletionCandidatesResult


class FacetsResult(_OperationResult):
    @model_validator(mode="before")
    @classmethod
    def canonical_facets_contract(cls, value: object) -> object:
        from polylogue.surfaces.payloads import FacetsResponse

        FacetsResponse.model_validate_json(json.dumps(value), strict=True)
        return value


class IngestResult(_OperationResult):
    source_generation_id: str = Field(min_length=1)
    outcome: DaemonOperationOutcome
    sequence: int = Field(ge=0)
    historical_receipt: IngestHistoricalReceipt

    @model_validator(mode="after")
    def binds_terminal_receipt(self) -> IngestResult:
        if self.outcome is not DaemonOperationOutcome.COMPLETED:
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

    @model_validator(mode="after")
    def exact_result_family(self) -> MutationResult:
        if self.status is None:
            if self.outcome not in {item.value for item in DaemonOperationOutcome} or self.sequence is None:
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
    outcome: DaemonOperationOutcome = DaemonOperationOutcome.FAILED

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
    request_type: str = ""
    result_type: str = ""
    request_model: type[BaseModel] = _OperationPayload
    result_model: type[BaseModel] = _OperationResult
    idempotent: bool = False
    handler: str = ""
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
)

MAX_DECLARED_OPERATION_BODY_BYTES: int = max(spec.max_body_bytes for spec in DAEMON_OPERATION_SPECS)
"""Transport-level read bound: the operation is only known after parsing, so
the socket read is capped by the largest declared body and the operation's own
``max_body_bytes`` is enforced once the request names it."""

MUTATION_OPERATION_NAMES: frozenset[str] = frozenset(
    spec.name for spec in DAEMON_OPERATION_SPECS if spec.authority is not DaemonAuthority.READ
)
"""Operations a CLI adapter may never execute in its own process."""

if len({spec.name for spec in DAEMON_OPERATION_SPECS}) != len(DAEMON_OPERATION_SPECS):
    raise RuntimeError("daemon operation names must be unique")


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
    "MUTATION_OPERATION_NAMES",
    "MAX_DECLARED_OPERATION_BODY_BYTES",
    "MAX_OPERATION_BODY_BYTES",
    "MAX_OPERATION_RESULT_BYTES",
    "DaemonAuthority",
    "DaemonOperationOutcome",
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
