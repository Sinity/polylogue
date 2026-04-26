"""Non-durable planning and run-result views layered above archive storage."""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from typing import ClassVar, TypedDict, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.storage.cursor_state import CursorFailurePayload, CursorStatePayload
from polylogue.types import PlanStage

_PayloadModelT = TypeVar("_PayloadModelT", bound=BaseModel)


class PlanCountsPayload(TypedDict, total=False):
    scan: int
    store_raw: int
    validate: int
    parse: int
    materialize: int
    render: int
    index: int


class PlanDetailsPayload(TypedDict, total=False):
    new_raw: int
    existing_raw: int
    duplicate_raw: int
    backlog_validate: int
    backlog_parse: int
    preview_invalid: int
    preview_skipped_no_schema: int


class RunCountsPayload(TypedDict, total=False):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int
    acquired: int
    skipped: int
    acquire_errors: int
    validated: int
    validation_invalid: int
    validation_drift: int
    validation_skipped_no_schema: int
    validation_errors: int
    materialized: int
    rendered: int
    render_failures: int
    parse_failures: int
    schemas_generated: int
    schemas_failed: int
    new_conversations: int
    changed_conversations: int


class DriftBucketPayload(TypedDict):
    conversations: int
    messages: int
    attachments: int


class RunDriftPayload(TypedDict):
    new: DriftBucketPayload
    removed: DriftBucketPayload
    changed: DriftBucketPayload


class RenderFailurePayload(TypedDict):
    conversation_id: str
    error: str


def _coerce_model(value: object, model_type: type[_PayloadModelT]) -> _PayloadModelT:
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value or {})


def _coerce_stage_sequence_payload(value: object) -> list[PlanStage]:
    if value is None:
        return []
    values = value if isinstance(value, list) else list(value) if isinstance(value, tuple) else [value]
    return [PlanStage.from_string(str(item)) for item in values]


def _coerce_cursor_state_payload(value: object) -> dict[str, CursorStatePayload]:
    if not isinstance(value, Mapping):
        return {}

    payload: dict[str, CursorStatePayload] = {}
    for name, cursor in value.items():
        if not isinstance(name, str) or not isinstance(cursor, Mapping):
            continue
        cursor_payload: CursorStatePayload = {}
        int_fields = ("file_count", "error_count", "failed_count")
        float_fields = ("latest_mtime",)
        str_fields = (
            "latest_file_name",
            "latest_path",
            "latest_file_id",
            "latest_error",
            "latest_error_file",
        )
        for field_name in int_fields:
            field_value = cursor.get(field_name)
            if isinstance(field_value, int) and not isinstance(field_value, bool):
                if field_name == "file_count":
                    cursor_payload["file_count"] = field_value
                elif field_name == "error_count":
                    cursor_payload["error_count"] = field_value
                elif field_name == "failed_count":
                    cursor_payload["failed_count"] = field_value
        for field_name in float_fields:
            field_value = cursor.get(field_name)
            if (
                field_name == "latest_mtime"
                and isinstance(field_value, (int, float))
                and not isinstance(field_value, bool)
            ):
                cursor_payload["latest_mtime"] = float(field_value)
        for field_name in str_fields:
            field_value = cursor.get(field_name)
            if isinstance(field_value, str):
                if field_name == "latest_file_name":
                    cursor_payload["latest_file_name"] = field_value
                elif field_name == "latest_path":
                    cursor_payload["latest_path"] = field_value
                elif field_name == "latest_file_id":
                    cursor_payload["latest_file_id"] = field_value
                elif field_name == "latest_error":
                    cursor_payload["latest_error"] = field_value
                elif field_name == "latest_error_file":
                    cursor_payload["latest_error_file"] = field_value
        failures = cursor.get("failed_files")
        if isinstance(failures, list):
            typed_failures: list[CursorFailurePayload] = []
            for item in failures:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("path")
                error = item.get("error")
                if isinstance(path, str) and isinstance(error, str):
                    typed_failures.append({"path": path, "error": error})
            if typed_failures:
                cursor_payload["failed_files"] = typed_failures
        payload[name] = cursor_payload
    return payload


class _IntPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    _exclude_none_payload: ClassVar[bool] = True

    def _payload_dict(self) -> dict[str, int]:
        payload: dict[str, int] = {}
        for field_name, field_info in self.__class__.model_fields.items():
            key = field_info.alias or field_name
            value = getattr(self, field_name)
            if value is None and self._exclude_none_payload:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                payload[key] = value
        return payload

    def to_dict(self) -> dict[str, int]:
        return self._payload_dict()

    def int_value(self, field_name: str) -> int:
        value = getattr(self, field_name)
        return value if isinstance(value, int) else 0

    def __getitem__(self, key: str) -> int:
        payload = self.to_dict()
        if key not in payload:
            raise KeyError(key)
        return payload[key]

    def get(self, key: str, default: int | None = None) -> int | None:
        return self.to_dict().get(key, default)

    def items(self) -> ItemsView[str, int]:
        return self.to_dict().items()

    def keys(self) -> KeysView[str]:
        return self.to_dict().keys()

    def values(self) -> ValuesView[int]:
        return self.to_dict().values()

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return super().__eq__(other)


class PlanCounts(_IntPayloadModel):
    scan: int | None = None
    store_raw: int | None = None
    validate_count: int | None = Field(default=None, alias="validate")
    parse: int | None = None
    materialize: int | None = None
    render: int | None = None
    index: int | None = None

    def to_payload(self) -> PlanCountsPayload:
        payload: PlanCountsPayload = {}
        if self.scan is not None:
            payload["scan"] = self.scan
        if self.store_raw is not None:
            payload["store_raw"] = self.store_raw
        if self.validate_count is not None:
            payload["validate"] = self.validate_count
        if self.parse is not None:
            payload["parse"] = self.parse
        if self.materialize is not None:
            payload["materialize"] = self.materialize
        if self.render is not None:
            payload["render"] = self.render
        if self.index is not None:
            payload["index"] = self.index
        return payload


class PlanDetails(_IntPayloadModel):
    new_raw: int | None = None
    existing_raw: int | None = None
    duplicate_raw: int | None = None
    backlog_validate: int | None = None
    backlog_parse: int | None = None
    preview_invalid: int | None = None
    preview_skipped_no_schema: int | None = None

    def to_payload(self) -> PlanDetailsPayload:
        payload: PlanDetailsPayload = {}
        if self.new_raw is not None:
            payload["new_raw"] = self.new_raw
        if self.existing_raw is not None:
            payload["existing_raw"] = self.existing_raw
        if self.duplicate_raw is not None:
            payload["duplicate_raw"] = self.duplicate_raw
        if self.backlog_validate is not None:
            payload["backlog_validate"] = self.backlog_validate
        if self.backlog_parse is not None:
            payload["backlog_parse"] = self.backlog_parse
        if self.preview_invalid is not None:
            payload["preview_invalid"] = self.preview_invalid
        if self.preview_skipped_no_schema is not None:
            payload["preview_skipped_no_schema"] = self.preview_skipped_no_schema
        return payload


class RunCounts(_IntPayloadModel):
    conversations: int | None = None
    messages: int | None = None
    attachments: int | None = None
    skipped_conversations: int | None = None
    skipped_messages: int | None = None
    skipped_attachments: int | None = None
    acquired: int | None = None
    skipped: int | None = None
    acquire_errors: int | None = None
    validated: int | None = None
    validation_invalid: int | None = None
    validation_drift: int | None = None
    validation_skipped_no_schema: int | None = None
    validation_errors: int | None = None
    materialized: int | None = None
    rendered: int | None = None
    render_failures: int | None = None
    parse_failures: int | None = None
    schemas_generated: int | None = None
    schemas_failed: int | None = None
    new_conversations: int | None = None
    changed_conversations: int | None = None

    def to_payload(self) -> RunCountsPayload:
        payload: RunCountsPayload = {}
        if self.conversations is not None:
            payload["conversations"] = self.conversations
        if self.messages is not None:
            payload["messages"] = self.messages
        if self.attachments is not None:
            payload["attachments"] = self.attachments
        if self.skipped_conversations is not None:
            payload["skipped_conversations"] = self.skipped_conversations
        if self.skipped_messages is not None:
            payload["skipped_messages"] = self.skipped_messages
        if self.skipped_attachments is not None:
            payload["skipped_attachments"] = self.skipped_attachments
        if self.acquired is not None:
            payload["acquired"] = self.acquired
        if self.skipped is not None:
            payload["skipped"] = self.skipped
        if self.acquire_errors is not None:
            payload["acquire_errors"] = self.acquire_errors
        if self.validated is not None:
            payload["validated"] = self.validated
        if self.validation_invalid is not None:
            payload["validation_invalid"] = self.validation_invalid
        if self.validation_drift is not None:
            payload["validation_drift"] = self.validation_drift
        if self.validation_skipped_no_schema is not None:
            payload["validation_skipped_no_schema"] = self.validation_skipped_no_schema
        if self.validation_errors is not None:
            payload["validation_errors"] = self.validation_errors
        if self.materialized is not None:
            payload["materialized"] = self.materialized
        if self.rendered is not None:
            payload["rendered"] = self.rendered
        if self.render_failures is not None:
            payload["render_failures"] = self.render_failures
        if self.parse_failures is not None:
            payload["parse_failures"] = self.parse_failures
        if self.schemas_generated is not None:
            payload["schemas_generated"] = self.schemas_generated
        if self.schemas_failed is not None:
            payload["schemas_failed"] = self.schemas_failed
        if self.new_conversations is not None:
            payload["new_conversations"] = self.new_conversations
        if self.changed_conversations is not None:
            payload["changed_conversations"] = self.changed_conversations
        return payload


class DriftBucket(_IntPayloadModel):
    _exclude_none_payload: ClassVar[bool] = False
    conversations: int = 0
    messages: int = 0
    attachments: int = 0

    def to_payload(self) -> DriftBucketPayload:
        return {
            "conversations": self.conversations,
            "messages": self.messages,
            "attachments": self.attachments,
        }


class RunDrift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new: DriftBucket = Field(default_factory=DriftBucket)
    removed: DriftBucket = Field(default_factory=DriftBucket)
    changed: DriftBucket = Field(default_factory=DriftBucket)

    def to_dict(self) -> RunDriftPayload:
        return self.to_payload()

    def to_payload(self) -> RunDriftPayload:
        return {
            "new": self.new.to_payload(),
            "removed": self.removed.to_payload(),
            "changed": self.changed.to_payload(),
        }

    def __getitem__(self, key: str) -> DriftBucket:
        payload = self._bucket_mapping()
        if key not in payload:
            raise KeyError(key)
        return payload[key]

    def get(self, key: str, default: DriftBucket | None = None) -> DriftBucket | None:
        return self._bucket_mapping().get(key, default)

    def items(self) -> ItemsView[str, DriftBucket]:
        return self._bucket_mapping().items()

    def keys(self) -> KeysView[str]:
        return self._bucket_mapping().keys()

    def values(self) -> ValuesView[DriftBucket]:
        return self._bucket_mapping().values()

    def __contains__(self, key: object) -> bool:
        return key in self._bucket_mapping()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return super().__eq__(other)

    def _bucket_mapping(self) -> dict[str, DriftBucket]:
        return {"new": self.new, "removed": self.removed, "changed": self.changed}


class PlanResult(BaseModel):
    timestamp: int
    stage: PlanStage = PlanStage.ALL
    stage_sequence: list[PlanStage] = Field(default_factory=list)
    counts: PlanCounts = Field(default_factory=PlanCounts)
    details: PlanDetails = Field(default_factory=PlanDetails)
    sources: list[str]
    cursors: dict[str, CursorStatePayload] = Field(default_factory=dict)

    @field_validator("stage", mode="before")
    @classmethod
    def coerce_stage(cls, v: object) -> PlanStage:
        return PlanStage.from_string(str(v))

    @field_validator("stage_sequence", mode="before")
    @classmethod
    def coerce_stage_sequence(cls, v: object) -> list[PlanStage]:
        return _coerce_stage_sequence_payload(v)

    @field_validator("counts", mode="before")
    @classmethod
    def coerce_plan_counts(cls, v: object) -> PlanCounts:
        return _coerce_model(v, PlanCounts)

    @field_validator("details", mode="before")
    @classmethod
    def coerce_plan_details(cls, v: object) -> PlanDetails:
        return _coerce_model(v, PlanDetails)

    @field_validator("cursors", mode="before")
    @classmethod
    def coerce_cursors(cls, v: object) -> dict[str, CursorStatePayload]:
        return _coerce_cursor_state_payload(v)


class RunResult(BaseModel):
    run_id: str
    counts: RunCounts = Field(default_factory=RunCounts)
    drift: RunDrift = Field(default_factory=RunDrift)
    indexed: bool
    index_error: str | None
    duration_ms: int
    render_failures: list[RenderFailurePayload] = Field(default_factory=list)
    run_path: str | None = None

    @field_validator("counts", mode="before")
    @classmethod
    def coerce_run_counts(cls, v: object) -> RunCounts:
        return _coerce_model(v, RunCounts)

    @field_validator("drift", mode="before")
    @classmethod
    def coerce_run_drift(cls, v: object) -> RunDrift:
        return _coerce_model(v, RunDrift)

    @field_validator("render_failures", mode="before")
    @classmethod
    def coerce_render_failures(cls, v: object) -> list[RenderFailurePayload]:
        if not isinstance(v, list):
            return []
        failures: list[RenderFailurePayload] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            conversation_id = item.get("conversation_id")
            error = item.get("error")
            if isinstance(conversation_id, str) and isinstance(error, str):
                failures.append({"conversation_id": conversation_id, "error": error})
        return failures


__all__ = [
    "DriftBucket",
    "DriftBucketPayload",
    "PlanCounts",
    "PlanCountsPayload",
    "PlanDetails",
    "PlanDetailsPayload",
    "PlanResult",
    "RenderFailurePayload",
    "RunCounts",
    "RunCountsPayload",
    "RunDrift",
    "RunDriftPayload",
    "RunResult",
]
