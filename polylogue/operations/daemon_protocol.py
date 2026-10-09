"""Typed archive-scoped daemon operation envelopes.

The envelope deliberately carries readiness and authority with the result so
clients do not need a health/probe request before every operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, ValidationError, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaValidator, core_schema

from polylogue.analysis.resume_contracts import ResumeCandidate
from polylogue.core.enums import OperationStatus
from polylogue.operations.machine_receipts import IngestTerminalReceipt
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


class DaemonAuthority(StrEnum):
    READ = "read"
    WRITE = "write"
    CONTROL = "control"
    LONG_RUNNING = "long-running"


class DaemonFallback(StrEnum):
    NEVER = "never"


class DaemonAuthorization(StrEnum):
    """What a request must present before a mutating operation executes."""

    #: The principal's capability is the whole authorization.
    NONE = "none"
    #: The request carries ``confirm=true``; the daemon then prepares and
    #: authorizes its own preview inside the one execution.
    CONFIRMATION = "confirmation"
    #: The request consumes exact daemon-issued preview or authorization
    #: references; target, generation, plan or body drift since issue refuses.
    PREVIEW = "preview"


class DaemonRecovery(StrEnum):
    """How a client settles an operation whose outcome it never received."""

    #: A read: sending the request again is safe.
    RETRY = "retry"
    #: A durable request record exists: ``operation.await`` or resubmitting
    #: the same request id replays the recorded outcome and never re-executes.
    AWAIT_REQUEST = "await-request"
    #: No durable request record: never resend blindly. Daemon startup settles
    #: an interrupted attempt from durable evidence; read the target's state
    #: before issuing a new request.
    RECONCILE = "reconcile"


DAEMON_OPERATION_OUTCOMES = frozenset(
    {
        *(status.value for status in OperationStatus),
        "degraded",
        "cancelled",
        "timed-out",
        "disconnected-before-acceptance",
        "disconnected-after-acceptance",
        "indeterminate",
        "restarted",
    }
)


#: Outcomes after which the daemon may have applied the request's effects
#: although the caller holds no receipt. Every surface raises
#: ``DaemonMutationIndeterminateError``-shaped recovery for these instead of
#: reporting a failure a caller could safely retry.
DAEMON_INDETERMINATE_OUTCOMES = frozenset({"indeterminate", "disconnected-after-acceptance", "restarted"})


class _OperationPayload(BaseModel):
    """Base for a concrete machine-operation payload type."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InsightReadRequest(_OperationPayload):
    """Import-light wire boundary; selected insight owns nested validation."""

    page: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_insight_query(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightListRequest

        InsightListRequest.model_validate(value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightListRequest

        return handler.resolve_ref_schema(handler(InsightListRequest.__pydantic_core_schema__))


class InsightReadResult(_OperationPayload):
    """Validate the exact registry page without loading it on unrelated routes."""

    page: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_insight_page(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightListResult

        _validate_json_result_model(InsightListResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightListResult

        return handler.resolve_ref_schema(handler(InsightListResult.__pydantic_core_schema__))


class InsightReadinessWireRequest(_OperationPayload):
    query: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_selected_readiness(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightReadinessRequest

        InsightReadinessRequest.model_validate(value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightReadinessRequest

        return handler.resolve_ref_schema(handler(InsightReadinessRequest.__pydantic_core_schema__))


class InsightReadinessWireResult(_OperationPayload):
    report: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_readiness_report(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightReadinessResult

        _validate_json_result_model(InsightReadinessResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightReadinessResult

        return handler.resolve_ref_schema(handler(InsightReadinessResult.__pydantic_core_schema__))


class InsightRigorWireRequest(_OperationPayload):
    query: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_selected_rigor(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightRigorRequest

        InsightRigorRequest.model_validate(value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightRigorRequest

        return handler.resolve_ref_schema(handler(InsightRigorRequest.__pydantic_core_schema__))


class InsightRigorWireResult(_OperationPayload):
    report: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_rigor_report(cls, value: object) -> object:
        from polylogue.operations.insight_contracts import InsightRigorResult

        _validate_json_result_model(InsightRigorResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_contracts import InsightRigorResult

        return handler.resolve_ref_schema(handler(InsightRigorResult.__pydantic_core_schema__))


class HermesHealthWireRequest(_OperationPayload):
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.hermes_health_contracts import HermesHealthRequest

        return handler.resolve_ref_schema(handler(HermesHealthRequest.__pydantic_core_schema__))


class HermesHealthWireResult(_OperationPayload):
    report: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_health(cls, value: object) -> object:
        from polylogue.operations.hermes_health_contracts import HermesHealthResult

        _validate_json_result_model(HermesHealthResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.hermes_health_contracts import HermesHealthResult

        return handler.resolve_ref_schema(handler(HermesHealthResult.__pydantic_core_schema__))


class InsightExportWireRequest(_OperationPayload):
    request: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_request(cls, value: object) -> object:
        from polylogue.operations.insight_export_contracts import InsightExportRequest

        _validate_json_result_model(InsightExportRequest, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_export_contracts import InsightExportRequest

        return handler.resolve_ref_schema(handler(InsightExportRequest.__pydantic_core_schema__))


class InsightExportWireResult(_OperationPayload):
    bundle: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_bundle(cls, value: object) -> object:
        from polylogue.operations.insight_export_contracts import InsightExportResult

        _validate_json_result_model(InsightExportResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.insight_export_contracts import InsightExportResult

        return handler.resolve_ref_schema(handler(InsightExportResult.__pydantic_core_schema__))


class AnnotationJoinWireRequest(_OperationPayload):
    schema_id: str
    schema_version: int
    statuses: list[str]
    target_kind: str | None = None
    group_by: list[str] = Field(default_factory=list)
    limit: int = 500
    offset: int = 0

    @model_validator(mode="before")
    @classmethod
    def validate_join_request(cls, value: object) -> object:
        from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest

        _validate_json_result_model(AnnotationStructuralJoinRequest, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.annotations.join_contracts import AnnotationStructuralJoinRequest

        return handler.resolve_ref_schema(handler(AnnotationStructuralJoinRequest.__pydantic_core_schema__))


class AnnotationJoinWireResult(_OperationPayload):
    result: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_join_result(cls, value: object) -> object:
        from polylogue.annotations.join_contracts import AnnotationJoinOperationResult

        _validate_json_result_model(AnnotationJoinOperationResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.annotations.join_contracts import AnnotationJoinOperationResult

        return handler.resolve_ref_schema(handler(AnnotationJoinOperationResult.__pydantic_core_schema__))


class FablePacketWireRequest(_OperationPayload):
    seed: str
    requested_size: int = Field(ge=0)
    schema_id: str = "delegation.discourse"
    schema_version: int = Field(default=1, ge=1)
    exact_template_cap: int = Field(default=1, ge=1)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.fable_packet_contracts import FablePacketRequest

        return handler.resolve_ref_schema(handler(FablePacketRequest.__pydantic_core_schema__))


class FablePacketWireResult(_OperationPayload):
    packet: dict[str, object]
    outcome: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_packet(cls, value: object) -> object:
        from polylogue.operations.fable_packet_contracts import FablePacketResult

        _validate_json_result_model(FablePacketResult, value)
        return value

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from polylogue.operations.fable_packet_contracts import FablePacketResult

        return handler.resolve_ref_schema(handler(FablePacketResult.__pydantic_core_schema__))


class StatusRequest(_OperationPayload):
    include_archive_readiness: bool = False


class QueryRequest(_OperationPayload):
    params: dict[str, object] = Field(default_factory=dict)


class QueryUnitsRequest(QueryRequest):
    pass


class DialogueReadRequest(QueryRequest):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)
    projection: dict[str, object] = Field(default_factory=dict)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=200)
    continuation: str | None = Field(default=None, min_length=1)


class TemporalReadRequest(QueryRequest):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str | None = None
    projection: dict[str, object] = Field(default_factory=dict)


class ChronicleReadRequest(QueryRequest):
    session_id: str | None = None
    projection: dict[str, object] = Field(default_factory=dict)


class CompactReadRequest(QueryRequest):
    session_id: str | None = None
    projection: dict[str, object] = Field(default_factory=dict)


class EffectiveContextReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)
    at_position: int | None = None


class OrchestrationReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)


class LineageReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)
    node_offset: int = Field(default=0, ge=0)
    node_limit: int | None = Field(default=None, ge=1)
    edge_offset: int = Field(default=0, ge=0)
    edge_limit: int | None = Field(default=None, ge=1)


class TopologyReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)
    node_offset: int = Field(default=0, ge=0)
    node_limit: int = Field(default=200, ge=1)
    edge_limit: int = Field(default=500, ge=1)


class NeighborReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str | None = None
    query: str | None = None
    origin: str | None = None
    limit: int = Field(default=10, ge=1)
    window_hours: int = Field(default=24, ge=1)

    @model_validator(mode="after")
    def has_seed(self) -> NeighborReadRequest:
        if not self.session_id and not self.query:
            raise ValueError("supply a session_id or query seed")
        return self


class CorrelationReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str = Field(min_length=1)
    repo_path: str | None = None
    since_hours: int = 2
    confidence_threshold: float = 0.3


class ContextPreambleReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    session_id: str | None = None
    related_limit: int = Field(default=5, ge=1, le=100)
    repo_path: str | None = None
    cwd: str | None = None
    recent_files: list[str] = Field(default_factory=list)
    require_session: bool = True
    boundary: str = "session_start"
    token_budget: int | None = Field(default=None, ge=1)
    source_tool_calls: dict[str, str] = Field(default_factory=dict)
    observed_at: str = Field(min_length=1)
    observed_project_state: dict[str, object] | None = None
    project_failure: str | None = None


class ContextImageReadRequest(_OperationPayload):
    selection_epoch: str | None = Field(default=None, min_length=1)
    seed_session_id: str | None = None
    seed_session_ids: list[str] = Field(default_factory=list)
    project_path: str | None = None
    project_repo: str | None = None
    since: str | None = None
    until: str | None = None
    origin: str | None = None
    query: str | None = None
    observed_at_ms: int = Field(ge=1)
    max_sessions: int = Field(default=5, ge=1, le=20)
    max_tokens: int | None = Field(default=None, ge=1)
    max_messages_per_session: int | None = Field(default=24, ge=1)
    max_chars_per_message: int | None = Field(default=1800, ge=1)
    include_messages: bool = True
    include_assertions: bool = True
    redact_paths: bool = True
    #: Compiled read views for a composed image (``messages``, ``temporal``,
    #: ``chronicle``). ``None`` keeps the context-image lens, whose only view
    #: is the message transcript gated by ``include_messages``. A view the
    #: compiler does not support is reported as an ``unsupported`` omission.
    read_views: list[str] | None = None
    purpose: Literal["handoff", "continue"] = "handoff"


class IdentityResetPreviewRequest(_OperationPayload):
    """Resolve one selector into a frozen, authenticated audited preview."""

    session: str | None = Field(default=None, min_length=1)
    source_path: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def exactly_one_selector(self) -> IdentityResetPreviewRequest:
        if (self.session is None) == (self.source_path is None):
            raise ValueError("identity reset preview needs exactly one selector")
        return self


class IdentityResetTargetsRequest(_OperationPayload):
    preview_request_id: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    page_size: int = Field(default=256, ge=1)


class ExcisionPlanRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    cascade_lineage: bool = False


class AssertionExportRequest(_OperationPayload):
    """Every assertion row by default: an export includes marks, overlays and deleted rows."""

    kinds: list[str] | None = None
    statuses: list[str] | None = None
    limit: int | None = Field(default=None, ge=0)
    offset: int = Field(default=0, ge=0)
    page_size: int = Field(default=256, ge=1)
    selection_ref: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def continued_export_requires_selection(self) -> AssertionExportRequest:
        if self.offset and self.selection_ref is None:
            raise ValueError("continued assertion export requires selection_ref")
        return self


class AssertionExportReleaseRequest(_OperationPayload):
    selection_ref: str = Field(min_length=1)


class ContinuationRouteRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    selection_epoch: str | None = Field(default=None, min_length=1)


class ContinuationContextRequest(ContinuationRouteRequest):
    observed_at_ms: int = Field(ge=1)


class ContinuationCandidatesRequest(_OperationPayload):
    repo_path: str = Field(min_length=1)
    cwd: str | None = None
    recent_files: list[str] = Field(default_factory=list)
    limit: int = 10


class AssertionClaimsListRequest(_OperationPayload):
    kinds: list[str] | None = None
    statuses: list[str] | None = Field(default_factory=lambda: ["active", "candidate"])
    target_ref: str | None = None
    scope_ref: str | None = None
    context_inject: bool | None = None
    limit: int = Field(default=20, ge=1, le=1000)


class UserMarksListRequest(_OperationPayload):
    mark_type: str | None = None
    session_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    message_id: str | None = None


class UserAnnotationsListRequest(_OperationPayload):
    session_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    message_id: str | None = None


class UserOverlayListRequest(_OperationPayload):
    pass


class UserOverlayGetRequest(_OperationPayload):
    id: str = Field(min_length=1)


class UserSettingGetRequest(_OperationPayload):
    setting_key: str = Field(min_length=1)


class UserMarkMutationRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    mark_type: str = Field(min_length=1)
    target_type: str = Field(default="session", min_length=1)
    target_id: str | None = None
    message_id: str | None = None


class UserAnnotationSaveRequest(_OperationPayload):
    annotation_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    note_text: str = Field(min_length=1)
    target_type: str = Field(default="session", min_length=1)
    target_id: str | None = None
    message_id: str | None = None


class UserOverlayDeleteRequest(_OperationPayload):
    id: str = Field(min_length=1)


class UserSavedViewSaveRequest(_OperationPayload):
    view_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    query: dict[str, object]
    watch: bool = False


class UserRecallPackSaveRequest(_OperationPayload):
    pack_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    payload: dict[str, object]

    @model_validator(mode="after")
    def items_are_objects(self) -> UserRecallPackSaveRequest:
        items = self.payload.get("items")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("recall pack payload.items must be a list of objects")
        return self


class UserWorkspaceSaveRequest(_OperationPayload):
    workspace_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mode: Literal["tabs", "stack", "compare", "timeline"] = "tabs"
    open_targets: list[dict[str, object]] = Field(default_factory=list)
    layout: dict[str, object] = Field(default_factory=dict)
    active_target: dict[str, object] = Field(default_factory=dict)


class FacadeMetadataSetRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: object


class FacadeMetadataDeleteRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    key: str = Field(min_length=1)


class FacadeTagAddRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    author_ref: str | None = None
    author_kind: str | None = None


class FacadeTagRemoveRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    tag: str = Field(min_length=1)


class FacadeBulkTagSessionsRequest(_OperationPayload):
    session_ids: list[str] = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    author_ref: str | None = None
    author_kind: str | None = None


class FacadeRecallPackCreateRequest(_OperationPayload):
    pack_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    session_ids_json: str
    payload_json: str


class FacadeRecallPackDeleteRequest(_OperationPayload):
    pack_id: str = Field(min_length=1)


class FacadeSavedViewSaveRequest(_OperationPayload):
    view_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    query_json: str
    watch: bool = False


class FacadeWorkspaceSaveRequest(_OperationPayload):
    workspace_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mode: Literal["tabs", "stack", "compare", "timeline"]
    open_targets_json: str
    layout_json: str
    active_target_json: str = "{}"


class FacadeWorkspaceDeleteRequest(_OperationPayload):
    workspace_id: str = Field(min_length=1)


class FacadeRecordCorrectionRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    payload: dict[str, str]
    note: str | None = None
    author_ref: str | None = None
    author_kind: str | None = None


class FacadeCorrectionRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)


class FacadeClearCorrectionsRequest(_OperationPayload):
    session_id: str = Field(min_length=1)


class FacadeBlackboardPostRequest(_OperationPayload):
    kind: str = Field(min_length=1)
    title: str
    content: str
    scope_repo: str | None = None
    scope_session: str | None = None
    scope_issue: int | None = None
    scope_path: str | None = None
    related_sessions: list[str] = Field(default_factory=list)
    author_ref: str | None = None
    author_kind: str = "user"
    evidence_refs: list[str] = Field(default_factory=list)
    staleness: dict[str, object] | None = None
    context_policy: dict[str, object] | None = None


class FacadeDeleteSessionRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    # The caller's own ``mutation.session.delete.preview`` reference; the
    # handler never prepares a preview on the caller's behalf.
    preview_ref: str = Field(min_length=1)


class FacadeWorkEventRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    summary: str
    payload: dict[str, object] | None = None
    timestamp: str | None = None


class FacadeManualContinuationRequest(_OperationPayload):
    child_session_id: str = Field(min_length=1)
    parent_session_id: str = Field(min_length=1)


class FacadeContextDeliveryRequest(_OperationPayload):
    image: dict[str, object]
    boundary: str = Field(min_length=1)
    recipient_ref: str = Field(min_length=1)
    delivered_by_ref: str = Field(min_length=1)
    run_ref: str | None = None
    inheritance_mode: str = "explicit"


class FacadeCandidateJudgmentRequest(_OperationPayload):
    candidate_ref: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    reason: str | None = None
    actor_ref: str = "user:local"
    inject: bool = False
    replacement_kind: str | None = None
    replacement_body_text: str | None = None
    replacement_value: object | None = None


class ComparativeJudgmentJudgeRequest(_OperationPayload):
    actor_ref: str = Field(min_length=1)
    execution_context_id: str = Field(min_length=1)
    role: str = "judge"


class ComparativeJudgmentWireRequest(_OperationPayload):
    judgment_id: str = Field(min_length=1)
    items: list[str] = Field(min_length=2)
    dimension: str = Field(min_length=1)
    verdict: str | list[str]
    judge: ComparativeJudgmentJudgeRequest
    blinded: bool
    rubric_id: str = Field(min_length=1)
    rubric_version: int = Field(ge=1)
    evidence_refs: list[str] = Field(default_factory=list)
    elicitation_ref: str | None = None
    rationale: str | None = None
    rationale_visible: bool = False
    decided_at_ms: int = Field(default=0, ge=0)


class FacadeComparativeJudgmentRequest(_OperationPayload):
    judgment: ComparativeJudgmentWireRequest
    author_kind: str = "user"


class ContextLedgerRowRequest(_OperationPayload):
    decision: Literal["included", "degraded", "dropped"]
    source: str = Field(min_length=1)
    item_ref: str = Field(min_length=1)
    token_cost: int = Field(ge=0)
    source_local_rank: int = Field(ge=0)
    budget_before: int = Field(ge=0)
    budget_after: int = Field(ge=0)
    disclosure_verdict: str = Field(min_length=1)
    authority_verdict: str = Field(min_length=1)
    authority_reason: str
    policy_refs: list[str] = Field(default_factory=list)
    target_session: str | None = None
    execution_context_ref: str = Field(min_length=1)


class FacadeContextLedgerRequest(_OperationPayload):
    build_ref: str = Field(min_length=1)
    ledger_rows: list[ContextLedgerRowRequest] = Field(default_factory=list)
    observed_at_ms: int = Field(ge=0)


class CompletionRequest(_OperationPayload):
    """One shell-completion question.

    ``source`` selects the archive-backed value vocabularies (session ids,
    user tags, repositories, working directories, tool names); every other ``kind`` is answered from
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


class BackupRequest(_OperationPayload):
    output_dir: str = Field(min_length=1)
    check_only: bool = False
    verify: bool = False
    profile: Literal["full_evidence", "user_overlays", "rebuildable_cache_exclude", "diagnostics_bundle"] = (
        "rebuildable_cache_exclude"
    )


class RestoreVerifiedBackupRequest(_OperationPayload):
    backup_dir: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class SecretScanRequest(_OperationPayload):
    session_id: str | None = None
    scan_all: bool = False
    max_sessions: int | None = Field(default=None, ge=1)
    origin: str | None = None
    status_only: bool = False

    @model_validator(mode="after")
    def one_scan_mode(self) -> SecretScanRequest:
        if sum((self.session_id is not None, self.scan_all, self.status_only)) != 1:
            raise ValueError("select exactly one of session_id, scan_all, or status_only")
        return self


class SchemaQuarantineVerdict(_OperationPayload):
    raw_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SchemaQuarantineRequest(_OperationPayload):
    """Schema-verification verdicts to persist on Source ``raw_sessions`` rows."""

    verdicts: list[SchemaQuarantineVerdict] = Field(min_length=1)


class WorkEvidenceGraphReplaceRequest(_OperationPayload):
    """One whole work-evidence graph, and the stored base it was derived from.

    ``expected_base_digest`` is the digest of the stored graph the caller read
    (``absent`` when none was stored); ``None`` replaces unconditionally.
    """

    graph: dict[str, object]
    expected_base_digest: str | None = Field(default=None, min_length=1)


class IngestRequest(_OperationPayload):
    path: str = Field(min_length=1)
    source_path: str | None = None
    source_name: str | None = None
    idempotency_key: str | None = None


class InsightRebuildRequest(_OperationPayload):
    session_ids: list[str] | None = None

    @model_validator(mode="after")
    def nonempty_identifiers(self) -> InsightRebuildRequest:
        if self.session_ids is not None and any(not value for value in self.session_ids):
            raise ValueError("session identifiers must be nonempty")
        return self


#: Transport bound shared by every delete phase. A selection accepted by the
#: preview yields one preview (then authorization) reference per chunk, so the
#: follow-up phases must accept a body sized for the same selection.
DELETE_SELECTION_MAX_BODY_BYTES = 64 * 1024 * 1024


class MutationSelectionRequest(_OperationPayload):
    """The resident operation owns matching and its explicit cardinality intent."""

    params: dict[str, object] = Field(default_factory=dict)
    mode: Literal["single", "first", "all", "page"] = "single"

    @model_validator(mode="after")
    def preserve_selection_intent(self) -> MutationSelectionRequest:
        from polylogue.operations.query_lowering import cli_query_spec

        spec = cli_query_spec(self.params)
        if self.mode != "page" and (spec.limit is not None or spec.offset or spec.cursor or spec.sample is not None):
            raise ValueError("whole-set mutation cardinality does not accept a query display window")
        if self.mode == "all" and spec.latest:
            raise ValueError("latest and all select different scopes")
        return self


class DeletePreviewRequest(_OperationPayload):
    session_ids: list[str] | None = Field(default=None, min_length=1)
    selection: MutationSelectionRequest | None = None

    @model_validator(mode="after")
    def exact_selection(self) -> DeletePreviewRequest:
        if (self.session_ids is None) == (self.selection is None):
            raise ValueError("supply exactly one explicit selection or resident query selection")
        if self.session_ids is not None:
            if any(not value for value in self.session_ids):
                raise ValueError("session identifiers must be nonempty")
            if len(set(self.session_ids)) != len(self.session_ids):
                raise ValueError("selection_is_not_canonical")
        return self


class DeleteAuthorizeRequest(_OperationPayload):
    preview_ref: str | None = Field(default=None, min_length=1)
    preview_request_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def exact_reference_shape(self) -> DeleteAuthorizeRequest:
        if (self.preview_ref is None) == (self.preview_request_id is None):
            raise ValueError("supply exactly one preview reference or preview operation reference")
        return self


class DeleteCancelRequest(DeleteAuthorizeRequest):
    pass


class DeleteExecuteRequest(_OperationPayload):
    authorization_ref: str | None = Field(default=None, min_length=1)
    authorization_request_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def exact_reference_shape(self) -> DeleteExecuteRequest:
        if (self.authorization_ref is None) == (self.authorization_request_id is None):
            raise ValueError("supply exactly one authorization reference or authorization operation reference")
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
    """One combined User change over explicit API IDs or a resident query scope."""

    session_ids: list[str] | None = Field(default=None, min_length=1)
    selection: MutationSelectionRequest | None = None
    add_marks: list[str] = Field(default_factory=list)
    remove_marks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    pairs: list[list[str]] = Field(default_factory=list)
    note_text: str | None = None

    @model_validator(mode="after")
    def validate_all_intents(self) -> SessionMarkRequest:
        from polylogue.core.user_state_targets import validate_mark_type
        from polylogue.surfaces.payloads import validate_metadata_key

        if (self.session_ids is None) == (self.selection is None):
            raise ValueError("supply exactly one explicit session selection or resident query selection")
        if self.session_ids is not None and any(not value for value in self.session_ids):
            raise ValueError("session identifiers must be nonempty")
        for value in (*self.add_marks, *self.remove_marks):
            validate_mark_type(value)
        if set(self.add_marks) & set(self.remove_marks):
            raise ValueError("a mark cannot be added and removed in one request")
        if any(not value.strip() for value in (*self.tags, *self.remove_tags)):
            raise ValueError("tags must be nonempty")
        if set(self.tags) & set(self.remove_tags):
            raise ValueError("a tag cannot be added and removed in one request")
        for pair in self.pairs:
            if len(pair) != 2:
                raise ValueError("metadata pairs must contain exactly a key and value")
            error = validate_metadata_key(pair[0])
            if error is not None:
                raise ValueError(error)
        if self.note_text is not None and not self.note_text.strip():
            raise ValueError("note text must be nonblank")
        if not (self.add_marks or self.remove_marks or self.tags or self.remove_tags or self.pairs or self.note_text):
            raise ValueError("supply at least one User change")
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


class AssertionCandidateCaptureRequest(_OperationPayload):
    """One terminal assertion captured as a private, non-injected candidate.

    ``cwd`` travels on the wire because ``--ref last`` resolves the caller's
    repository, and the handler runs in the daemon: falling back to the
    executing process's own working directory would silently resolve the
    daemon's cwd instead of the operator's.
    """

    body_text: str = Field(min_length=1)
    #: The canonical :class:`~polylogue.core.enums.AssertionKind` value, not
    #: the surface's capture vocabulary: ``polylogue note --kind claim`` and
    #: MCP's ``kind`` field are adapter spellings that each surface resolves
    #: before it lowers, so the operation contract carries one vocabulary.
    kind: str = Field(min_length=1, max_length=64)
    refs: list[str] = Field(default_factory=list, max_length=64)
    scope_refs: list[str] = Field(default_factory=list, max_length=64)
    # Bounded by the operation's request body size, not by a count.
    evidence_refs: list[str] = Field(default_factory=list)
    cwd: str | None = None
    author_ref: str = Field(default="user:local", min_length=1, max_length=512)
    author_kind: str = Field(default="user", min_length=1, max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=240)
    ttl_seconds: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def nonblank_body_and_refs(self) -> AssertionCandidateCaptureRequest:
        if not self.body_text.strip():
            raise ValueError("note text cannot be empty")
        if any(not value.strip() for value in (*self.refs, *self.scope_refs, *self.evidence_refs)):
            raise ValueError("refs must be nonempty")
        return self


class UserSettingSetRequest(_OperationPayload):
    """Insert-or-update one typed ``user_settings`` row under daemon authority.

    ``value`` is a JSON scalar rather than an open object because
    ``user_settings`` is a closed, typed key registry
    (:mod:`polylogue.storage.sqlite.archive_tiers.user_settings_write`), not a
    junk drawer: a structural value no registered validator could accept is
    refused by the contract rather than after an authorization was issued for
    a write that cannot happen.
    """

    setting_key: str = Field(min_length=1, max_length=256)
    value: str | int | float | bool | None
    author_ref: str = Field(default="user:local", min_length=1, max_length=512)

    @model_validator(mode="after")
    def nonblank_identifiers(self) -> UserSettingSetRequest:
        if not self.setting_key.strip():
            raise ValueError("setting_key must not be blank")
        if not self.author_ref.strip():
            raise ValueError("author_ref must not be blank")
        return self


class AnnotationInputDescriptor(_OperationPayload):
    """Exact acquired input bytes, independent of server scratch coordinates."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class AnnotationBatchImportOperationRequest(_OperationPayload):
    """Annotation batch controls and exact acquired-input identity, framed for the wire.

    Mirrors ``polylogue.annotations.importer.AnnotationBatchImportRequest``
    field-for-field. Declared separately rather than reused directly: this
    layer is the transport contract (what a client may send), while the
    product-layer request is what the actuator validates once the daemon
    process holds it -- the same separation every other mutation operation
    here already keeps. ``polylogue annotations import`` used to build the
    product-layer request and drive ``OperationExecutor`` itself from the CLI
    process, invisible to the mutation-authority layering rule because
    ``polylogue.annotations.importer`` is not one of the four executor
    modules it names (polylogue-gjwto / polylogue-r29bv AC3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    input: AnnotationInputDescriptor
    batch_id: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    target_ref: str = Field(min_length=1)
    source_result_ref: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    prompt_ref: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at_ms: int | None = Field(default=None, ge=0)
    schema_definition_json: str | None = None

    @model_validator(mode="after")
    def nonblank_refs(self) -> AnnotationBatchImportOperationRequest:
        if any(
            not value.strip()
            for value in (
                self.batch_id,
                self.schema_id,
                self.target_ref,
                self.source_result_ref,
                self.actor_ref,
                self.model_ref,
                self.prompt_ref,
            )
        ):
            raise ValueError("annotation batch import identifiers and refs must not be blank")
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
    confirm: bool = False


class SessionLifecycleRequest(_OperationPayload):
    session_id: str = Field(min_length=1)
    mode: Literal["mirror", "primary"]
    reason: str = Field(min_length=1, max_length=4096)
    actor: str = Field(min_length=1, max_length=512)
    confirm: bool = False


class IdentityResetAuthorizeRequest(_OperationPayload):
    preview_request_id: str = Field(min_length=1)
    confirm: bool = False


class IdentityResetRequest(_OperationPayload):
    authorization_request_id: str = Field(min_length=1)


class RawAuthorityBlockerResolveRequest(_OperationPayload):
    blocker_id: str = Field(min_length=1)
    resolution: str = Field(min_length=1, max_length=4096)
    confirm: bool = False


class RawAuthorityFrontierRequest(_OperationPayload):
    """Publish the accepted-frontier census's durable obligations; no parameters."""


class ResetRequest(_OperationPayload):
    index: bool = False
    database: bool = False
    blob: bool = False
    assets: bool = False
    cache: bool = False
    auth: bool = False
    reset_all: bool = False
    confirm: bool = False
    # Omitted means "no preview was asserted"; an explicit list, even an empty
    # one, must equal the resolved targets.
    expected_targets: list[str] | None = Field(default=None, max_length=10_000)


class BlobPublicationsAbandonRequest(_OperationPayload):
    publication_ids: list[str] = Field(min_length=1, max_length=256)
    confirm: bool = False


class DemoAugmentRequest(_OperationPayload):
    with_overlays: bool = False


class EmbeddingBackfillRequest(_OperationPayload):
    """Operator bounds for one daemon-owned embedding catch-up pass."""

    max_sessions: int | None = Field(default=None, ge=1)
    max_messages: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    min_messages: int | None = Field(default=None, ge=1)
    stop_after_seconds: int | None = Field(default=None, ge=1)
    max_errors: int | None = Field(default=None, ge=1)
    rebuild: bool = False


class EmbeddingFailureResolveRequest(_OperationPayload):
    failure_id: str = Field(min_length=1)
    resolution: Literal["acknowledge", "requeue", "supersede"]
    note: str | None = None
    superseded_by: str | None = None


class OperationStatusRequest(_OperationPayload):
    request_id: str = Field(min_length=1)
    parts_offset: int = Field(default=0, ge=0)
    parts_limit: int = Field(default=40, ge=1, le=40)


class OperationAwaitRequest(OperationStatusRequest):
    after_sequence: int = Field(default=0, ge=0)
    after_progress_sequence: int = Field(default=0, ge=0)
    timeout_ms: int = Field(default=30_000, ge=1, le=30_000)


class OperationResultDocument(_OperationPayload):
    request_id: str = Field(min_length=1)
    byte_length: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OperationResultRequest(OperationResultDocument):
    offset: int = Field(default=0, ge=0)


class OperationCancelRequest(OperationStatusRequest):
    pass


#: Sentinel distinguishing "the key is absent" from "the key is an honest null".
_MISSING = object()


class _OperationResult(BaseModel):
    """Base for declared result payloads; envelopes own authority metadata."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


class OperationResultPage(_OperationPayload):
    document: OperationResultDocument
    offset: int = Field(ge=0)
    data_base64: str
    next_offset: int | None = Field(default=None, ge=0)


class UserOverlayListResult(_OperationResult):
    items: list[dict[str, object]]
    total: int = Field(ge=0)


class AssertionClaimsListResult(UserOverlayListResult):
    limit: int = Field(ge=1)
    statuses: list[str] | None = None
    kinds: list[str] | None = None


class UserOverlayGetResult(_OperationResult):
    found: bool
    item: dict[str, object] | None


class UserSettingGetResult(UserOverlayGetResult):
    outcome: dict[str, object]


class AssertionExportReleaseResult(_OperationResult):
    released: bool


class AssertionExportResult(UserOverlayListResult):
    selection_ref: str
    outcome: dict[str, object]
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    snapshot_epoch: str


class IdentityResetTargetsResult(_OperationResult):
    session_ids: list[str]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    outcome: dict[str, object]


class ExcisionPlanResult(_OperationResult):
    found: bool
    #: Set when the session is a prefix-sharing lineage parent and the request
    #: did not cascade: the dependents that excising it alone would orphan.
    lineage_dependent_session_ids: list[str]
    refused: bool
    detail: str | None = None
    plan: dict[str, object] | None = None


class UserSettingListResult(UserOverlayListResult):
    outcome: dict[str, object]


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
        epoch = payload.get("snapshot_epoch")
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("session query result requires its pinned snapshot epoch")
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
            _validate_json_result_model(SessionListResponse, payload)
        else:
            _validate_json_result_model(SearchEnvelope, payload)
        return value


class QueryUnitsResult(_OperationResult):
    @model_validator(mode="before")
    @classmethod
    def canonical_unit_contract(cls, value: object) -> object:
        from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope, QueryUnitEnvelope

        if not isinstance(value, dict):
            raise ValueError("query unit result must be an object")
        model = QueryUnitAggregateEnvelope if value.get("mode") == "query-unit-aggregate" else QueryUnitEnvelope
        _validate_json_result_model(model, value)
        return value


class DialogueReadResult(_OperationResult):
    view: Literal["dialogue"]
    payload: dict[str, object]


class TemporalReadResult(_OperationResult):
    view: Literal["temporal"]
    payload: dict[str, object]


class ChronicleReadResult(_OperationResult):
    view: Literal["chronicle"]
    payload: dict[str, object]


class CompactReadResult(_OperationResult):
    view: Literal["compact"]
    payload: dict[str, object]


class EffectiveContextReadResult(_OperationResult):
    view: Literal["effective_context"]
    payload: dict[str, object]


class OrchestrationReadResult(_OperationResult):
    view: Literal["orchestration"]
    payload: dict[str, object]


class LineageReadResult(_OperationResult):
    view: Literal["lineage"]
    payload: dict[str, object]


class TopologyReadResult(_OperationResult):
    view: Literal["topology"]
    payload: dict[str, object]


class NeighborReadResult(_OperationResult):
    view: Literal["neighbors"]
    payload: dict[str, object]


class CorrelationReadResult(_OperationResult):
    view: Literal["correlation"]
    payload: dict[str, object]


class ContextPreambleReadResult(_OperationResult):
    view: Literal["context"]
    payload: dict[str, object] | None
    ledger: dict[str, object] | None


class ContextImageReadResult(_OperationResult):
    view: Literal["context-image"]
    payload: dict[str, object]


class ContinuationRouteResult(_OperationResult):
    status: Literal["supported", "unsupported"]
    origin: str
    native_session_id: str
    argv: list[str]
    cwd: str | None
    detail: str | None
    open_state: Literal["open", "closed", "unknown"]
    command: str | None


class ContinuationCandidatesResult(_OperationResult):
    candidates: list[ResumeCandidate]
    returned: int = Field(ge=0)
    limit: int

    @model_validator(mode="after")
    def _exact_window(self) -> ContinuationCandidatesResult:
        if self.returned != len(self.candidates):
            raise ValueError("returned must equal the candidate window length")
        return self


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

        _validate_json_result_model(FacetsResponse, value)
        return value


class IngestResult(_OperationResult):
    source_generation_id: str = Field(min_length=1)
    #: ``degraded``: rows committed, derived convergence did not finish; the
    #: receipt's recorded convergence decides which, never the caller.
    outcome: Literal["completed", "degraded"]
    sequence: int = Field(ge=0)
    historical_receipt: IngestTerminalReceipt

    @model_validator(mode="after")
    def binds_terminal_receipt(self) -> IngestResult:
        from polylogue.operations.machine_receipts import ingest_terminal_outcome

        if self.outcome != ingest_terminal_outcome(self.historical_receipt):
            raise ValueError("ingest terminal outcome disagrees with its receipt's recorded convergence")
        if self.source_generation_id != self.historical_receipt.source_generation_id:
            raise ValueError("ingest result and historical receipt disagree on source generation")
        if self.sequence != self.historical_receipt.final_sequence:
            raise ValueError("ingest result and historical receipt disagree on terminal sequence")
        return self


class InsightRebuildResult(_OperationPayload):
    profiles: int = Field(ge=0)
    threads: int = Field(ge=0)
    tag_rollups: int = Field(ge=0)


class MutationResult(_OperationPayload):
    result_document: OperationResultDocument | None = None
    status: Literal["prepared", "authorized", "cancelled"] | None = None
    operation: str | None = None
    preview_ref: str | None = None
    authorization_ref: str | None = None
    session_ids_sample: list[str] | None = None
    session_count: int | None = Field(default=None, ge=0)
    expires_at_ms: int | None = None
    lifetime: Literal["accepted-request"] | None = None
    outcome: str | None = None
    sequence: int | None = Field(default=None, ge=0)
    reference: AcceptedOperationReference | None = None
    source_request_id: str | None = None
    tag_count: int | None = Field(default=None, ge=0)
    applied_count: int | None = Field(default=None, ge=0)
    not_attempted_count: int | None = Field(default=None, ge=0)
    parts_total: int | None = Field(default=None, ge=0)
    parts_offset: int | None = Field(default=None, ge=0)
    next_parts_offset: int | None = Field(default=None, ge=0)
    effect: Literal["committed", "no-effect", "indeterminate"] | None = None
    completed_chunks: int | None = Field(default=None, ge=0)
    affected_count: int | None = Field(default=None, ge=0)
    not_attempted: list[int] | None = None
    parts: list[dict[str, object]] | None = None
    stop_reason: str | None = None
    artifact_refs: list[str] | None = None
    result: dict[str, object] | None = None
    cancellation_requested: bool | None = None
    #: The actual terminal future is still owned because private scratch
    #: transfer failed; this is separate from the operation's own error.
    terminal_custody_error: str | None = None
    accepted: bool | None = None
    progress_sequence: int | None = Field(default=None, ge=0)
    progress_events: list[dict[str, object]] | None = None
    progress_gap: dict[str, int] | None = None
    # polylogue-oil1q: ``machine_request_state`` reports the accepted frozen
    # source generation for a ``source-generation`` artifact (every ingest).
    # Declaring it keeps ``extra="forbid"`` meaningful instead of making a
    # clean, committed ingest fail its own await contract and surface to the
    # client as ``DaemonMutationIndeterminateError``.
    source_generation_id: str | None = None
    #: The typed error of a settled ``degraded`` ingest, carried in the
    #: durable lifecycle state so ``operation.await``/``status``/``cancel``
    #: report what the executing request reported.
    error: dict[str, object] | None = None
    #: The executor's durable handle for the audited attempt
    #: (``mutation-operation:<operation_id>``).  ``OperationExecutor`` already
    #: stamps it onto the :class:`MutationReceipt` it returns, but the
    #: daemon-side envelope used to keep only ``receipt.domain_receipt``, so a
    #: surface that declared a receipt reference could never populate one and
    #: rendered ``null`` for every audited destructive mutation.
    receipt_ref: str | None = None

    @model_validator(mode="after")
    def exact_result_family(self) -> MutationResult:
        if self.status is None:
            if self.outcome not in DAEMON_OPERATION_OUTCOMES or self.sequence is None:
                raise ValueError("mutation lifecycle result requires outcome and durable sequence")
        elif self.status == "prepared":
            from polylogue.operations.mutation_transaction import DELETE_PREVIEW_SAMPLE_IDS

            if self.session_count is None or self.session_ids_sample is None:
                raise ValueError("prepared result requires its measured selection")
            if len(self.session_ids_sample) != min(self.session_count, DELETE_PREVIEW_SAMPLE_IDS):
                raise ValueError("prepared result requires the declared bounded selection sample")
            reset = self.operation == "identity-reset"
            if reset:
                if (
                    self.lifetime != "accepted-request"
                    or self.expires_at_ms is not None
                    or self.reference is None
                    or self.reference.operation_name != "mutation.identity-reset.preview"
                    or self.preview_ref is not None
                ):
                    raise ValueError("reset preview requires its sealed accepted-request custody")
            elif self.lifetime is not None:
                raise ValueError("accepted-request lifetime belongs only to reset previews")
            if self.session_count or reset:
                if self.reference is None or (not reset and self.expires_at_ms is None):
                    raise ValueError("nonempty preview requires its durable operation reference and expiry")
                if self.reference.artifact_kind != "preview-batch":
                    raise ValueError("prepared result does not name sealed preview authority")
        elif self.status == "authorized":
            if self.reference is None or self.reference.artifact_kind != "authorization-batch":
                raise ValueError("authorized result requires its durable operation reference")
        elif self.reference is None or self.reference.artifact_kind != "cancelled-preview-batch":
            raise ValueError("cancelled result requires its durable operation reference")
        return self


class EmbeddingBackfillProgress(_OperationPayload):
    state: Literal["stopped", "complete", "unknown"]
    computed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class EmbeddingBackfillCounts(_OperationPayload):
    done: int | None = Field(default=None, ge=0)
    pending: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)


class EmbeddingBackfillFailure(_OperationPayload):
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=512)


class EmbeddingBackfillResult(_OperationPayload):
    operation: Literal["maintenance.embeddings.backfill"]
    outcome: Literal["completed", "stopped", "cancelled", "failed"]
    sequence: int = Field(ge=1)
    effect: Literal["committed", "no-effect", "indeterminate"]
    affected_count: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    progress: EmbeddingBackfillProgress
    result: EmbeddingBackfillCounts
    error: EmbeddingBackfillFailure | None = None


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
    #: Paged machine batches accept any number of parts (polylogue-zxbbl).
    part_count: int = Field(ge=1)
    accepted_deadline_unix_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="json"))

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
    deadline_s: float | None = None
    """No implicit READ limit; other authorities declare their execution bound."""
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
    authorization: DaemonAuthorization = DaemonAuthorization.NONE
    """What a request must present before this operation executes."""
    durable_request: bool = False
    """The handler records every request under its id before any effect and
    answers a repeat of that id from the record, without an accepted
    reference. Implied by ``accepted_reference``."""

    @property
    def recovery(self) -> DaemonRecovery:
        """How a client settles this operation when its outcome is lost."""
        if self.authority is DaemonAuthority.READ or self.name == "operation.result":
            return DaemonRecovery.RETRY
        if self.accepted_reference or self.durable_request:
            return DaemonRecovery.AWAIT_REQUEST
        return DaemonRecovery.RECONCILE

    def __post_init__(self) -> None:
        if self.request_model is _OperationPayload or self.result_model is _OperationResult:
            raise ValueError("operation declarations require concrete request and result models")
        if self.authority is DaemonAuthority.READ and self.authorization is not DaemonAuthorization.NONE:
            raise ValueError("a read operation carries no authorization binding")
        if self.authority is DaemonAuthority.READ and self.deadline_s is not None:
            raise ValueError("read operations have no implicit execution deadline")
        if (
            self.authority in {DaemonAuthority.WRITE, DaemonAuthority.LONG_RUNNING}
            and self.deadline_s is None
            and not (self.accepted_reference or self.durable_request)
        ):
            raise ValueError("write operations require an execution deadline or durable owner")
        if not self.handler:
            object.__setattr__(self, "handler", self.name.replace(".", "_"))
        if self.request_type and self.result_type:
            return
        stem = "".join(part.capitalize() for part in self.name.replace(".", "-").split("-"))
        object.__setattr__(self, "request_type", self.request_type or f"{stem}Request")
        object.__setattr__(self, "result_type", self.result_type or f"{stem}Result")

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
            "authorization": self.authorization.value,
            "recovery": self.recovery.value,
        }


DAEMON_OPERATION_SPECS: tuple[DaemonOperationSpec, ...] = (
    DaemonOperationSpec(
        "maintenance.insights.rebuild",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.rebuild_insights",
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
        "operation.result",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="read",
        request_model=OperationResultRequest,
        result_model=OperationResultPage,
        handler="operation_result",
    ),
    DaemonOperationSpec(
        "operation.status",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="read",
        request_model=OperationStatusRequest,
        result_model=MutationResult,
        handler="operation_status",
        deadline_s=2.0,
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
        deadline_s=2.0,
    ),
    DaemonOperationSpec(
        "insights.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.list.request/v1",
        result_contract="insights.list.result/v1",
        request_type="InsightReadRequest",
        result_type="InsightReadResult",
        request_model=InsightReadRequest,
        result_model=InsightReadResult,
    ),
    DaemonOperationSpec(
        "insights.readiness",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.readiness.request/v1",
        result_contract="insights.readiness.result/v1",
        request_type="InsightReadinessWireRequest",
        result_type="InsightReadinessWireResult",
        request_model=InsightReadinessWireRequest,
        result_model=InsightReadinessWireResult,
    ),
    DaemonOperationSpec(
        "annotation.join",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="annotation.join.request/v1",
        result_contract="annotation.join.result/v1",
        request_model=AnnotationJoinWireRequest,
        result_model=AnnotationJoinWireResult,
    ),
    DaemonOperationSpec(
        "insights.rigor",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.rigor.request/v1",
        result_contract="insights.rigor.result/v1",
        request_type="InsightRigorWireRequest",
        result_type="InsightRigorWireResult",
        request_model=InsightRigorWireRequest,
        result_model=InsightRigorWireResult,
    ),
    DaemonOperationSpec(
        "insights.export_bundle",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.export_bundle.request/v1",
        result_contract="insights.export_bundle.result/v1",
        request_model=InsightExportWireRequest,
        result_model=InsightExportWireResult,
    ),
    DaemonOperationSpec(
        "insights.fable_packet",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.fable_packet.request/v1",
        result_contract="insights.fable_packet.result/v1",
        request_model=FablePacketWireRequest,
        result_model=FablePacketWireResult,
    ),
    DaemonOperationSpec(
        "insights.hermes_health",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="insights.hermes_health.request/v1",
        result_contract="insights.hermes_health.result/v1",
        request_model=HermesHealthWireRequest,
        result_model=HermesHealthWireResult,
    ),
    DaemonOperationSpec(
        "cli.query",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="cli.query.result/v1",
        request_type="QueryRequest",
        result_type="QueryResult",
        request_model=QueryRequest,
        result_model=QueryResult,
    ),
    DaemonOperationSpec(
        "query.units",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="query.units.result/v1",
        request_type="QueryUnitsRequest",
        result_type="QueryUnitsResult",
        request_model=QueryUnitsRequest,
        result_model=QueryUnitsResult,
    ),
    DaemonOperationSpec(
        "query.aggregate",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        # Aggregates scan the selection rather than one page of it.
        result_contract="query.aggregate.result/v1",
        request_type="QueryAggregateRequest",
        result_type="QueryAggregateResult",
        request_model=QueryAggregateRequest,
        result_model=QueryAggregateResult,
    ),
    DaemonOperationSpec(
        "session.read",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="session.read.result/v1",
        request_type="SessionReadRequest",
        result_type="SessionReadResult",
        request_model=SessionReadRequest,
        result_model=SessionReadResult,
    ),
    DaemonOperationSpec(
        "read.dialogue",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.dialogue.result/v1",
        request_model=DialogueReadRequest,
        result_model=DialogueReadResult,
    ),
    DaemonOperationSpec(
        "read.temporal",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.temporal.result/v1",
        request_model=TemporalReadRequest,
        result_model=TemporalReadResult,
    ),
    DaemonOperationSpec(
        "read.chronicle",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.chronicle.result/v1",
        request_model=ChronicleReadRequest,
        result_model=ChronicleReadResult,
    ),
    DaemonOperationSpec(
        "read.compact",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.compact.result/v1",
        request_model=CompactReadRequest,
        result_model=CompactReadResult,
    ),
    DaemonOperationSpec(
        "read.effective_context",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.effective_context.result/v1",
        request_model=EffectiveContextReadRequest,
        result_model=EffectiveContextReadResult,
    ),
    DaemonOperationSpec(
        "read.orchestration",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.orchestration.result/v1",
        request_model=OrchestrationReadRequest,
        result_model=OrchestrationReadResult,
    ),
    DaemonOperationSpec(
        "read.lineage",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.lineage.result/v1",
        request_model=LineageReadRequest,
        result_model=LineageReadResult,
    ),
    DaemonOperationSpec(
        "read.topology",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.topology.result/v1",
        request_model=TopologyReadRequest,
        result_model=TopologyReadResult,
    ),
    DaemonOperationSpec(
        "read.neighbors",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.neighbors.result/v1",
        request_model=NeighborReadRequest,
        result_model=NeighborReadResult,
    ),
    DaemonOperationSpec(
        "read.correlation",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.correlation.result/v1",
        request_model=CorrelationReadRequest,
        result_model=CorrelationReadResult,
    ),
    DaemonOperationSpec(
        "read.context",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.context.result/v1",
        request_model=ContextPreambleReadRequest,
        result_model=ContextPreambleReadResult,
    ),
    DaemonOperationSpec(
        "read.context-image",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="read.context-image.result/v1",
        request_model=ContextImageReadRequest,
        result_model=ContextImageReadResult,
    ),
    DaemonOperationSpec(
        "continuation.route",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_model=ContinuationRouteRequest,
        result_model=ContinuationRouteResult,
    ),
    DaemonOperationSpec(
        "continuation.context",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_model=ContinuationContextRequest,
        result_model=ContextImageReadResult,
    ),
    DaemonOperationSpec(
        "continuation.candidates",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_model=ContinuationCandidatesRequest,
        result_model=ContinuationCandidatesResult,
    ),
    DaemonOperationSpec(
        "user.assertions.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.assertions.list.request/v1",
        result_contract="user.assertions.list.result/v1",
        request_model=AssertionClaimsListRequest,
        result_model=AssertionClaimsListResult,
    ),
    DaemonOperationSpec(
        "user.settings.get",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.settings.get.request/v1",
        result_contract="user.settings.get.result/v1",
        request_model=UserSettingGetRequest,
        result_model=UserSettingGetResult,
    ),
    DaemonOperationSpec(
        "user.settings.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.settings.list.request/v1",
        result_contract="user.settings.list.result/v1",
        request_model=UserOverlayListRequest,
        result_model=UserSettingListResult,
    ),
    DaemonOperationSpec(
        "user.assertions.export",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.assertions.export.request/v3",
        result_contract="user.assertions.export.result/v3",
        request_model=AssertionExportRequest,
        result_model=AssertionExportResult,
    ),
    DaemonOperationSpec(
        "user.assertions.export.release",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        request_contract="user.assertions.export.release.request/v1",
        result_contract="user.assertions.export.release.result/v1",
        request_model=AssertionExportReleaseRequest,
        result_model=AssertionExportReleaseResult,
    ),
    DaemonOperationSpec(
        "session.identity-reset.targets",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        capability="archive.identity_reset",
        handler="identity_reset_targets",
        request_contract="session.identity-reset.targets.request/v3",
        result_contract="session.identity-reset.targets.result/v2",
        request_model=IdentityResetTargetsRequest,
        result_model=IdentityResetTargetsResult,
    ),
    DaemonOperationSpec(
        "session.excision.plan",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="session.excision.plan.request/v1",
        result_contract="session.excision.plan.result/v1",
        request_model=ExcisionPlanRequest,
        result_model=ExcisionPlanResult,
    ),
    DaemonOperationSpec(
        "user.marks.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.marks.list.request/v1",
        result_contract="user.marks.list.result/v1",
        request_model=UserMarksListRequest,
        result_model=UserOverlayListResult,
    ),
    DaemonOperationSpec(
        "user.annotations.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.annotations.list.request/v1",
        result_contract="user.annotations.list.result/v1",
        request_model=UserAnnotationsListRequest,
        result_model=UserOverlayListResult,
    ),
    DaemonOperationSpec(
        "user.annotations.get",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.annotations.get.request/v1",
        result_contract="user.annotations.get.result/v1",
        request_model=UserOverlayGetRequest,
        result_model=UserOverlayGetResult,
    ),
    DaemonOperationSpec(
        "user.saved_views.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.saved_views.list.request/v1",
        result_contract="user.saved_views.list.result/v1",
        request_model=UserOverlayListRequest,
        result_model=UserOverlayListResult,
    ),
    DaemonOperationSpec(
        "user.saved_views.get",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.saved_views.get.request/v1",
        result_contract="user.saved_views.get.result/v1",
        request_model=UserOverlayGetRequest,
        result_model=UserOverlayGetResult,
    ),
    DaemonOperationSpec(
        "user.recall_packs.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.recall_packs.list.request/v1",
        result_contract="user.recall_packs.list.result/v1",
        request_model=UserOverlayListRequest,
        result_model=UserOverlayListResult,
    ),
    DaemonOperationSpec(
        "user.recall_packs.get",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.recall_packs.get.request/v1",
        result_contract="user.recall_packs.get.result/v1",
        request_model=UserOverlayGetRequest,
        result_model=UserOverlayGetResult,
    ),
    DaemonOperationSpec(
        "user.workspaces.list",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.workspaces.list.request/v1",
        result_contract="user.workspaces.list.result/v1",
        request_model=UserOverlayListRequest,
        result_model=UserOverlayListResult,
    ),
    DaemonOperationSpec(
        "user.workspaces.get",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        request_contract="user.workspaces.get.request/v1",
        result_contract="user.workspaces.get.result/v1",
        request_model=UserOverlayGetRequest,
        result_model=UserOverlayGetResult,
    ),
    DaemonOperationSpec(
        "user.mark.add",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.add_mark",
        deadline_s=120.0,
        request_contract="user.mark.add.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserMarkMutationRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_mark_add",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.mark.remove",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.remove_mark",
        deadline_s=120.0,
        request_contract="user.mark.remove.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserMarkMutationRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_mark_remove",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.annotation.save",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_annotation",
        deadline_s=120.0,
        request_contract="user.annotation.save.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserAnnotationSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_annotation_save",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.annotation.delete",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_annotation",
        deadline_s=120.0,
        request_contract="user.annotation.delete.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserOverlayDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_annotation_delete",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.saved_view.save",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_view",
        deadline_s=120.0,
        request_contract="user.saved_view.save.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserSavedViewSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_saved_view_save",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.saved_view.delete",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_view",
        deadline_s=120.0,
        request_contract="user.saved_view.delete.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserOverlayDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_saved_view_delete",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.recall_pack.save",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.create_recall_pack",
        deadline_s=120.0,
        request_contract="user.recall_pack.save.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserRecallPackSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_recall_pack_save",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.recall_pack.delete",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_recall_pack",
        deadline_s=120.0,
        request_contract="user.recall_pack.delete.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserOverlayDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_recall_pack_delete",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.workspace.save",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_workspace",
        deadline_s=120.0,
        request_contract="user.workspace.save.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserWorkspaceSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_workspace_save",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "user.workspace.delete",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_workspace",
        deadline_s=120.0,
        request_contract="user.workspace.delete.request/v1",
        result_contract="mutation.result/v1",
        request_model=UserOverlayDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="user_workspace_delete",
        handler_module="polylogue.operations.user_overlay_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.set_metadata",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.set_metadata",
        deadline_s=120.0,
        request_contract="mutation.facade.set_metadata.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeMetadataSetRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_set_metadata",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.delete_metadata",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_metadata",
        deadline_s=120.0,
        request_contract="mutation.facade.delete_metadata.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeMetadataDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_delete_metadata",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.bulk_tag_sessions",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.bulk_tag_sessions",
        deadline_s=120.0,
        request_contract="mutation.facade.bulk_tag_sessions.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeBulkTagSessionsRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_bulk_tag_sessions",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.add_tag",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.add_tag",
        deadline_s=120.0,
        request_contract="mutation.facade.add_tag.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeTagAddRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_add_tag",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.remove_tag",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.remove_tag",
        deadline_s=120.0,
        request_contract="mutation.facade.remove_tag.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeTagRemoveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_remove_tag",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.save_view",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_view",
        deadline_s=120.0,
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.facade.save_view.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeSavedViewSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_save_view",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.create_recall_pack",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.create_recall_pack",
        deadline_s=120.0,
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.facade.create_recall_pack.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeRecallPackCreateRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_create_recall_pack",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.delete_recall_pack",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_recall_pack",
        deadline_s=120.0,
        request_contract="mutation.facade.delete_recall_pack.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeRecallPackDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_delete_recall_pack",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.save_workspace",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.save_workspace",
        deadline_s=120.0,
        request_contract="mutation.facade.save_workspace.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeWorkspaceSaveRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_save_workspace",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.delete_workspace",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_workspace",
        deadline_s=120.0,
        request_contract="mutation.facade.delete_workspace.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeWorkspaceDeleteRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_delete_workspace",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.record_correction",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_correction",
        deadline_s=120.0,
        request_contract="mutation.facade.record_correction.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeRecordCorrectionRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_record_correction",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.delete_correction",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_correction",
        deadline_s=120.0,
        request_contract="mutation.facade.delete_correction.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeCorrectionRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_delete_correction",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.clear_corrections",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.clear_corrections",
        deadline_s=120.0,
        request_contract="mutation.facade.clear_corrections.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeClearCorrectionsRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_clear_corrections",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.post_blackboard_note",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.post_blackboard_note",
        deadline_s=120.0,
        request_contract="mutation.facade.post_blackboard_note.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeBlackboardPostRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_post_blackboard_note",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.delete_session",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=120.0,
        request_contract="mutation.facade.delete_session.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeDeleteSessionRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_delete_session",
        handler_module="polylogue.operations.facade_mutations",
    ),
    DaemonOperationSpec(
        "mutation.facade.record_work_event",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_work_event",
        deadline_s=120.0,
        request_contract="mutation.facade.record_work_event.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeWorkEventRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_record_work_event",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "mutation.facade.record_manual_continuation",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_manual_continuation",
        deadline_s=120.0,
        request_contract="mutation.facade.record_manual_continuation.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeManualContinuationRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_record_manual_continuation",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "mutation.facade.record_context_delivery",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_context_delivery",
        deadline_s=120.0,
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.facade.record_context_delivery.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeContextDeliveryRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_record_context_delivery",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "mutation.facade.judge_assertion_candidate",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.judge_assertion_candidate",
        deadline_s=120.0,
        request_contract="mutation.facade.judge_assertion_candidate.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeCandidateJudgmentRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_judge_assertion_candidate",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "mutation.facade.record_comparative_judgment",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.record_comparative_judgment",
        deadline_s=120.0,
        request_contract="mutation.facade.record_comparative_judgment.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeComparativeJudgmentRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_record_comparative_judgment",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "mutation.facade.context_ledger",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.context_injection_ledger",
        deadline_s=30.0,
        max_body_bytes=8 * 1024 * 1024,
        request_contract="mutation.facade.context_ledger.request/v1",
        result_contract="mutation.result/v1",
        request_model=FacadeContextLedgerRequest,
        result_model=MutationResult,
        idempotent=False,
        handler="facade_context_ledger",
        handler_module="polylogue.operations.facade_writers",
    ),
    DaemonOperationSpec(
        "session.reference",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="session.reference.result/v1",
        request_type="SessionReferenceRequest",
        result_type="SessionReferenceResult",
        request_model=SessionReferenceRequest,
        result_model=SessionReferenceResult,
    ),
    DaemonOperationSpec(
        "status",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="status.result/v1",
        request_type="StatusRequest",
        result_type="StatusResult",
        request_model=StatusRequest,
        result_model=StatusResult,
    ),
    DaemonOperationSpec(
        "completion",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
        result_contract="completion.result/v1",
        request_type="CompletionRequest",
        result_type="CompletionResult",
        request_model=CompletionRequest,
        result_model=CompletionResult,
    ),
    DaemonOperationSpec(
        "facets",
        DaemonAuthority.READ,
        DaemonFallback.NEVER,
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
        "maintenance.backup",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.backup",
        deadline_s=300.0,
        cancellable=False,
        request_contract="maintenance.backup.request/v1",
        result_contract="maintenance.backup.result/v1",
        request_model=BackupRequest,
        result_model=MutationResult,
        handler="maintenance_backup",
    ),
    DaemonOperationSpec(
        "maintenance.restore_verified_backup",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.restore_verified_backup",
        deadline_s=300.0,
        cancellable=False,
        request_contract="maintenance.restore_verified_backup.request/v1",
        result_contract="maintenance.restore_verified_backup.result/v1",
        request_model=RestoreVerifiedBackupRequest,
        result_model=MutationResult,
        handler="maintenance_restore_verified_backup",
    ),
    DaemonOperationSpec(
        "maintenance.secret_scan",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.scan_secrets",
        deadline_s=300.0,
        request_contract="maintenance.secret_scan.request/v1",
        result_contract="maintenance.secret_scan.result/v1",
        request_model=SecretScanRequest,
        result_model=MutationResult,
        handler="maintenance_secret_scan",
    ),
    DaemonOperationSpec(
        "maintenance.schema.quarantine",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.schema_quarantine",
        deadline_s=120.0,
        request_contract="maintenance.schema.quarantine.request/v1",
        result_contract="mutation.result/v1",
        request_type="SchemaQuarantineRequest",
        result_type="MutationResult",
        request_model=SchemaQuarantineRequest,
        result_model=MutationResult,
        handler="maintenance_schema_quarantine",
    ),
    DaemonOperationSpec(
        "mutation.work_evidence.graph.replace",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.replace_work_evidence_graph",
        deadline_s=120.0,
        # A whole incident or reconciled graph, not a parameter map.
        max_body_bytes=64 * 1024 * 1024,
        request_contract="mutation.work_evidence.graph.replace.request/v1",
        result_contract="mutation.result/v1",
        request_model=WorkEvidenceGraphReplaceRequest,
        result_model=MutationResult,
        idempotent=True,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.preview",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        # Accepted durably at its first page; the caller follows the durable
        # request to completion (polylogue-zxbbl), within the budget execute has.
        deadline_s=300.0,
        # A preview carries the exact selection -- every session id, split
        # into bounded preview chunks -- not a parameter map.
        max_body_bytes=DELETE_SELECTION_MAX_BODY_BYTES,
        request_contract="mutation.session.delete.preview.request/v1",
        result_contract="mutation.session.delete.preview.result/v1",
        request_type="DeletePreviewRequest",
        result_type="MutationResult",
        request_model=DeletePreviewRequest,
        result_model=MutationResult,
        durable_request=True,
        handler="execute_selected_preview_operation",
    ),
    DaemonOperationSpec(
        "mutation.session.delete.authorize",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        # Accepted durably at its first page; the caller follows the durable
        # request to completion (polylogue-zxbbl), within the budget execute has.
        deadline_s=300.0,
        # Carries one reference per preview chunk of the selection.
        max_body_bytes=DELETE_SELECTION_MAX_BODY_BYTES,
        request_contract="mutation.session.delete.authorize.request/v1",
        result_contract="mutation.session.delete.authorize.result/v1",
        request_type="DeleteAuthorizeRequest",
        result_type="MutationResult",
        request_model=DeleteAuthorizeRequest,
        result_model=MutationResult,
        authorization=DaemonAuthorization.PREVIEW,
        durable_request=True,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.cancel",
        DaemonAuthority.CONTROL,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        # Paged like the preview it releases, within the same budget.
        deadline_s=300.0,
        # Carries one reference per preview chunk of the selection.
        max_body_bytes=DELETE_SELECTION_MAX_BODY_BYTES,
        request_contract="mutation.session.delete.cancel.request/v1",
        result_contract="mutation.session.delete.cancel.result/v1",
        request_type="DeleteCancelRequest",
        result_type="MutationResult",
        request_model=DeleteCancelRequest,
        result_model=MutationResult,
        authorization=DaemonAuthorization.PREVIEW,
        durable_request=True,
    ),
    DaemonOperationSpec(
        "mutation.session.delete.execute",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.delete_session",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        # Carries one reference per preview chunk of the selection.
        max_body_bytes=DELETE_SELECTION_MAX_BODY_BYTES,
        request_contract="mutation.session.delete.execute.request/v1",
        result_contract="mutation.result/v1",
        request_type="DeleteExecuteRequest",
        result_type="MutationResult",
        request_model=DeleteExecuteRequest,
        result_model=MutationResult,
        authorization=DaemonAuthorization.PREVIEW,
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
        durable_request=True,
    ),
    DaemonOperationSpec(
        "mutation.session.mark",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.add_mark",
        additional_capabilities=(
            "archive.remove_mark",
            "archive.add_tag",
            "archive.remove_tag",
            "archive.set_metadata",
            "archive.save_annotation",
        ),
        deadline_s=120.0,
        request_contract="mutation.session.mark.request/v1",
        result_contract="mutation.result/v1",
        request_type="SessionMarkRequest",
        result_type="MutationResult",
        request_model=SessionMarkRequest,
        result_model=MutationResult,
        handler="execute_session_mark_operation",
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
        "mutation.assertion.candidate.capture",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.capture_assertion_candidate",
        deadline_s=120.0,
        # 256 KiB of stdin can expand to six JSON bytes per control character
        # when ensure_ascii escaping is applied by the daemon client.
        max_body_bytes=2 * 1024 * 1024,
        request_contract="mutation.assertion.candidate.capture.request/v1",
        result_contract="mutation.result/v1",
        request_type="AssertionCandidateCaptureRequest",
        result_type="MutationResult",
        request_model=AssertionCandidateCaptureRequest,
        result_model=MutationResult,
        handler="mutation_assertion_candidate_capture",
    ),
    DaemonOperationSpec(
        "mutation.user.setting.set",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.set_setting",
        deadline_s=120.0,
        max_body_bytes=64 * 1024,
        idempotent=True,
        request_contract="mutation.user.setting.set.request/v1",
        result_contract="mutation.result/v1",
        request_type="UserSettingSetRequest",
        result_type="MutationResult",
        request_model=UserSettingSetRequest,
        result_model=MutationResult,
        handler="mutation_user_setting_set",
    ),
    DaemonOperationSpec(
        "mutation.annotation.import_batch",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.import_annotation_batch",
        deadline_s=None,
        max_body_bytes=64 * 1024,
        durable_request=True,
        request_contract="mutation.annotation.import_batch.request/v1",
        result_contract="mutation.result/v1",
        request_type="AnnotationBatchImportOperationRequest",
        result_type="MutationResult",
        request_model=AnnotationBatchImportOperationRequest,
        result_model=MutationResult,
        idempotent=True,
        handler="mutation_annotation_import_batch",
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
        handler="execute_session_excision_operation",
        handler_module="polylogue.operations.daemon_excision",
        authorization=DaemonAuthorization.CONFIRMATION,
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
        authorization=DaemonAuthorization.CONFIRMATION,
    ),
    DaemonOperationSpec(
        "mutation.identity-reset.preview",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.identity_reset",
        accepted_reference=True,
        request_contract="mutation.identity-reset.preview.request/v2",
        result_contract="mutation.result/v1",
        request_model=IdentityResetPreviewRequest,
        result_model=MutationResult,
        handler="execute_selected_preview_operation",
        handler_module="polylogue.operations.daemon_mutations",
    ),
    DaemonOperationSpec(
        "mutation.identity-reset.authorize",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.identity_reset",
        accepted_reference=True,
        request_contract="mutation.identity-reset.authorize.request/v1",
        result_contract="mutation.result/v1",
        request_model=IdentityResetAuthorizeRequest,
        result_model=MutationResult,
        handler="mutation_identity_reset_authorize",
        authorization=DaemonAuthorization.CONFIRMATION,
    ),
    DaemonOperationSpec(
        "mutation.identity-reset",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.identity_reset",
        accepted_reference=True,
        request_contract="mutation.identity-reset.request/v3",
        result_contract="mutation.result/v1",
        request_type="IdentityResetRequest",
        result_type="MutationResult",
        request_model=IdentityResetRequest,
        result_model=MutationResult,
        handler="mutation_identity_reset",
        authorization=DaemonAuthorization.PREVIEW,
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
        handler="execute_raw_authority_blocker_resolve_operation",
        authorization=DaemonAuthorization.CONFIRMATION,
    ),
    DaemonOperationSpec(
        # The census publishes ``raw_authority_blockers`` rows into source.db,
        # so it is a write of the durable tier, not an inspection a CLI may run
        # beside the daemon (polylogue-5vps8 AC1).
        "maintenance.raw-authority-frontier",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.raw_authority.inspect_frontier",
        deadline_s=300.0,
        request_contract="maintenance.raw-authority-frontier.request/v1",
        result_contract="mutation.result/v1",
        request_type="RawAuthorityFrontierRequest",
        result_type="MutationResult",
        request_model=RawAuthorityFrontierRequest,
        result_model=MutationResult,
        handler="execute_raw_authority_frontier_operation",
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
        authorization=DaemonAuthorization.CONFIRMATION,
    ),
    DaemonOperationSpec(
        "maintenance.blob-publications.abandon",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.blob_publication.abandon_receipts",
        deadline_s=30.0,
        request_contract="maintenance.blob-publications.abandon.request/v1",
        result_contract="mutation.result/v1",
        request_type="BlobPublicationsAbandonRequest",
        result_type="MutationResult",
        request_model=BlobPublicationsAbandonRequest,
        result_model=MutationResult,
        handler="maintenance_blob_publications_abandon",
        authorization=DaemonAuthorization.CONFIRMATION,
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
    DaemonOperationSpec(
        "maintenance.embeddings.backfill",
        DaemonAuthority.LONG_RUNNING,
        DaemonFallback.NEVER,
        capability="archive.embeddings.backfill",
        deadline_s=300.0,
        progress=True,
        accepted_reference=True,
        request_contract="maintenance.embeddings.backfill.request/v1",
        result_contract="maintenance.embeddings.backfill.result/v1",
        request_type="EmbeddingBackfillRequest",
        result_type="EmbeddingBackfillResult",
        request_model=EmbeddingBackfillRequest,
        result_model=EmbeddingBackfillResult,
        idempotent=True,
        handler="execute_embedding_backfill_operation",
        handler_module="polylogue.daemon.embedding_owner",
    ),
    DaemonOperationSpec(
        "maintenance.embeddings.failure.resolve",
        DaemonAuthority.WRITE,
        DaemonFallback.NEVER,
        capability="archive.embed",
        deadline_s=60.0,
        request_contract="maintenance.embeddings.failure.resolve.request/v1",
        result_contract="mutation.result/v1",
        request_type="EmbeddingFailureResolveRequest",
        result_type="MutationResult",
        request_model=EmbeddingFailureResolveRequest,
        result_model=MutationResult,
        handler="maintenance_embedding_failure_resolve",
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


def _check_result_wire(value: object, *, native_result: bool) -> None:
    """Walk the existing value without encoding strings or duplicating rows."""
    stack = [iter((value,))]
    active: set[int] = set()
    owners: list[int] = []
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            if owners:
                active.remove(owners.pop())
            continue
        if item is None or isinstance(item, str | bool | int):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("result contains a nonfinite JSON number")
            continue
        if isinstance(item, dict):
            for key in item:
                if isinstance(key, str):
                    continue
                if not native_result or not (key is None or isinstance(key, bool | int | float)):
                    raise ValueError("result object keys must be JSON strings")
                if isinstance(key, float) and not math.isfinite(key):
                    raise ValueError("result contains a nonfinite JSON object key")
            children = iter(item.values())
        elif isinstance(item, list) or (native_result and isinstance(item, tuple)):
            children = iter(item)
        else:
            raise TypeError(f"result contains a non-JSON value: {type(item).__name__}")
        identity = id(item)
        if identity in active:
            raise ValueError("result contains a circular JSON value")
        active.add(identity)
        owners.append(identity)
        stack.append(children)


def _json_result_object(value: object) -> object:
    """Normalize native encoder-supported keys only when a mapping needs it."""
    if not isinstance(value, dict) or all(isinstance(key, str) for key in value):
        return value

    def wire_key(key: object) -> str:
        if isinstance(key, str):
            return key
        if key is None:
            return "null"
        if key is True:
            return "true"
        if key is False:
            return "false"
        if isinstance(key, int | float):
            return str(key)
        raise TypeError("result object key is not JSON encodable")

    return {wire_key(key): item for key, item in value.items()}


@cache
def _json_result_validator(model: type[BaseModel]) -> SchemaValidator:
    """Compile this protocol's observed JSON tuple/enum/datetime vocabulary.

    Scalars remain strict. JSON arrays can fill tuple fields; string enums and
    ISO datetimes use their declared JSON forms. This does not relax wire
    admission or add a general schema conversion facility.
    """

    def adapt(value: Any) -> Any:
        if isinstance(value, list):
            return [adapt(child) for child in value]
        if not isinstance(value, dict):
            return value
        result = {key: adapt(child) for key, child in value.items()}
        kind = result.get("type")
        if kind in {"str", "int", "float", "bool", "dict", "enum", "datetime"}:
            result["strict"] = True
        if kind == "model":
            result["config"] = {**result.get("config", {}), "strict": True}
        if kind == "dataclass":
            # JSON objects fill the declared dataclass fields. Its closed
            # config and nested strict scalar schemas still govern admission.
            result["strict"] = False
        if kind in {"tuple", "list"}:
            # The wire walker admits only arrays (native server tuples are
            # arrays at delivery). Item schemas retain strict scalar checks.
            result["strict"] = False
        # Named definitions belong to the wrapping validator, so reused refs
        # keep resolving to the same adapted strict schema.
        elif kind == "dict":
            return core_schema.no_info_before_validator_function(
                _json_result_object, result, ref=result.pop("ref", None)
            )
        elif kind == "enum":
            enum_type = result["cls"]

            def enum_value(item: object) -> object:
                if not isinstance(item, str):
                    raise ValueError("JSON enum value must be a string")
                return enum_type(item)

            return core_schema.no_info_before_validator_function(enum_value, result, ref=result.pop("ref", None))
        elif kind == "datetime":
            datetime_validator = SchemaValidator(core_schema.datetime_schema(strict=False))

            def datetime_value(item: object) -> object:
                if not isinstance(item, str):
                    raise ValueError("JSON datetime value must be a string")
                return datetime_validator.validate_python(item)

            return core_schema.no_info_before_validator_function(datetime_value, result, ref=result.pop("ref", None))
        return result

    # pydantic_core's rebuild option prevents reuse of the model's original
    # Python validator, which would discard the JSON-specific adaptations.
    model.model_rebuild()
    return SchemaValidator(adapt(model.__pydantic_core_schema__), _use_prebuilt=False)


def _validate_json_result_model(model: type[BaseModel], result: object) -> None:
    _json_result_validator(cast(Hashable, model)).validate_python(_json_result_object(result))


def validate_operation_result(operation: str, result: object, *, native_result: bool = True) -> None:
    """Validate native server or decoded client JSON without whole-result encoding."""
    spec = daemon_operation_spec(operation)
    if spec is None:
        raise OperationResultContractError(f"undeclared operation: {operation}")
    try:
        _check_result_wire(result, native_result=native_result)
        _validate_json_result_model(spec.result_model, result)
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
        if spec.request_model is AnnotationBatchImportOperationRequest:
            # Validate the finite JSON control without allocating another whole
            # envelope or imposing a batch outcome ceiling on its metadata.
            for _fragment in json.JSONEncoder(separators=(",", ":"), allow_nan=False).iterencode(dict(raw)):
                pass
        elif len(json.dumps(dict(raw), separators=(",", ":"), allow_nan=False).encode()) > spec.max_body_bytes:
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
    def fingerprint_intent(self) -> dict[str, object]:
        """The complete semantic intent used by exchange identity."""
        return {
            "operation": self.operation,
            "payload": self.payload,
            "archive_root": self.archive_root,
            "expected_archive_identity": self.expected_archive_identity,
            "expected_generation_id": self.expected_generation_id,
        }

    @property
    def fingerprint(self) -> str:
        """Stable exchange identity used for safe duplicate recovery."""
        encoded = json.dumps(self.fingerprint_intent, sort_keys=True, separators=(",", ":")).encode()
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
