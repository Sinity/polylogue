"""Non-durable planning and run-result views layered above archive storage."""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.types import PlanStage


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


class _SparseIntMap(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_dict(self) -> dict[str, int]:
        return cast(dict[str, int], self.model_dump(by_alias=True, exclude_none=True))

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


class _DenseIntMap(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_dict(self) -> dict[str, int]:
        return cast(dict[str, int], self.model_dump(by_alias=True))

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


class PlanCounts(_SparseIntMap):
    scan: int | None = None
    store_raw: int | None = None
    validate_count: int | None = Field(default=None, alias="validate")
    parse: int | None = None
    materialize: int | None = None
    render: int | None = None
    index: int | None = None


class PlanDetails(_SparseIntMap):
    new_raw: int | None = None
    existing_raw: int | None = None
    duplicate_raw: int | None = None
    backlog_validate: int | None = None
    backlog_parse: int | None = None
    preview_invalid: int | None = None
    preview_skipped_no_schema: int | None = None


class RunCounts(_SparseIntMap):
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


class DriftBucket(_DenseIntMap):
    conversations: int = 0
    messages: int = 0
    attachments: int = 0


class RunDrift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new: DriftBucket = Field(default_factory=DriftBucket)
    removed: DriftBucket = Field(default_factory=DriftBucket)
    changed: DriftBucket = Field(default_factory=DriftBucket)

    def to_dict(self) -> RunDriftPayload:
        return {
            "new": cast(DriftBucketPayload, self.new.to_dict()),
            "removed": cast(DriftBucketPayload, self.removed.to_dict()),
            "changed": cast(DriftBucketPayload, self.changed.to_dict()),
        }

    def __getitem__(self, key: str) -> DriftBucket:
        if key not in {"new", "removed", "changed"}:
            raise KeyError(key)
        return cast(DriftBucket, getattr(self, key))

    def get(self, key: str, default: DriftBucket | None = None) -> DriftBucket | None:
        if key in {"new", "removed", "changed"}:
            return cast(DriftBucket, getattr(self, key))
        return default

    def items(self) -> ItemsView[str, DriftBucket]:
        return {"new": self.new, "removed": self.removed, "changed": self.changed}.items()

    def keys(self) -> KeysView[str]:
        return {"new": self.new, "removed": self.removed, "changed": self.changed}.keys()

    def values(self) -> ValuesView[DriftBucket]:
        return {"new": self.new, "removed": self.removed, "changed": self.changed}.values()

    def __contains__(self, key: object) -> bool:
        return key in {"new", "removed", "changed"}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return super().__eq__(other)


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
        if v is None:
            return []
        values = v if isinstance(v, list) else list(v) if isinstance(v, tuple) else [v]
        return [PlanStage.from_string(str(item)) for item in values]

    @field_validator("counts", mode="before")
    @classmethod
    def coerce_plan_counts(cls, v: object) -> PlanCounts:
        if isinstance(v, PlanCounts):
            return v
        return PlanCounts.model_validate(v or {})

    @field_validator("details", mode="before")
    @classmethod
    def coerce_plan_details(cls, v: object) -> PlanDetails:
        if isinstance(v, PlanDetails):
            return v
        return PlanDetails.model_validate(v or {})

    @field_validator("cursors", mode="before")
    @classmethod
    def coerce_cursors(cls, v: object) -> dict[str, CursorStatePayload]:
        if not isinstance(v, dict):
            return {}
        payload: dict[str, CursorStatePayload] = {}
        for name, cursor in v.items():
            if isinstance(name, str) and isinstance(cursor, dict):
                payload[name] = cast(CursorStatePayload, cursor)
        return payload


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
        if isinstance(v, RunCounts):
            return v
        return RunCounts.model_validate(v or {})

    @field_validator("drift", mode="before")
    @classmethod
    def coerce_run_drift(cls, v: object) -> RunDrift:
        if isinstance(v, RunDrift):
            return v
        return RunDrift.model_validate(v or {})

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
