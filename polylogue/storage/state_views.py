"""Non-durable storage and pipeline state views.

These models are not direct table rows. They represent aggregate views,
pipeline/operator artifacts, or transient lookup state layered on top of the
durable storage records in ``store.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import (
    ArtifactSupportStatus,
    PlanStage,
    Provider,
    ValidationMode,
    ValidationStatus,
)


class ArtifactCohortSummary(BaseModel):
    """Aggregate summary for one observed artifact cohort."""

    provider_name: str
    payload_provider: Provider | None = None
    artifact_kind: str
    support_status: ArtifactSupportStatus
    cohort_id: str | None = None
    observation_count: int = 0
    unique_raw_ids: int = 0
    first_observed_at: str | None = None
    last_observed_at: str | None = None
    bundle_scope_count: int = 0
    sample_source_paths: list[str] = Field(default_factory=list)
    resolved_package_version: str | None = None
    resolved_element_kind: str | None = None
    resolution_reason: str | None = None
    link_group_count: int = 0
    linked_sidecar_count: int = 0

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_cohort_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("support_status", mode="before")
    @classmethod
    def coerce_cohort_support_status(cls, v: object) -> ArtifactSupportStatus:
        return ArtifactSupportStatus.from_string(str(v))


class PlanResult(BaseModel):
    timestamp: int
    stage: PlanStage = PlanStage.ALL
    stage_sequence: list[PlanStage] = Field(default_factory=list)
    counts: dict[str, int]
    details: dict[str, int] = Field(default_factory=dict)
    sources: list[str]
    cursors: dict[str, dict[str, Any]]

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


class RawConversationState(BaseModel):
    raw_id: str
    source_name: str | None = None
    source_path: str | None = None
    parsed_at: str | None = None
    parse_error: str | None = None
    payload_provider: Provider | None = None
    validation_status: ValidationStatus | None = None
    validation_provider: Provider | None = None

    @field_validator("payload_provider", "validation_provider", mode="before")
    @classmethod
    def coerce_optional_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_status", mode="before")
    @classmethod
    def coerce_state_validation_status(cls, v: object) -> ValidationStatus | None:
        if v is None:
            return None
        return ValidationStatus.from_string(str(v))


class _RawStateUnset:
    """Sentinel for update fields that should remain unchanged."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _RawStateUnset()


@dataclass(frozen=True)
class RawConversationStateUpdate:
    """Typed raw-state mutation payload used by update_raw_state calls."""

    parsed_at: str | None | _RawStateUnset = UNSET
    parse_error: str | None | _RawStateUnset = UNSET
    payload_provider: Provider | str | None | _RawStateUnset = UNSET
    validation_status: ValidationStatus | str | None | _RawStateUnset = UNSET
    validation_error: str | None | _RawStateUnset = UNSET
    validation_drift_count: int | None | _RawStateUnset = UNSET
    validation_provider: Provider | str | None | _RawStateUnset = UNSET
    validation_mode: ValidationMode | str | None | _RawStateUnset = UNSET

    @property
    def has_values(self) -> bool:
        """Return whether any field is explicitly provided."""
        return any(
            value is not UNSET
            for value in (
                self.parsed_at,
                self.parse_error,
                self.payload_provider,
                self.validation_status,
                self.validation_error,
                self.validation_drift_count,
                self.validation_provider,
                self.validation_mode,
            )
        )


class RunResult(BaseModel):
    run_id: str
    counts: dict[str, int]
    drift: dict[str, dict[str, int]]
    indexed: bool
    index_error: str | None
    duration_ms: int
    render_failures: list[dict[str, str]] = Field(default_factory=list)
    run_path: str | None = None


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str


@dataclass(frozen=True)
class ConversationRenderProjection:
    """Repository-owned render projection preserving raw attachment layout."""

    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


__all__ = [
    "ArtifactCohortSummary",
    "ConversationRenderProjection",
    "ExistingConversation",
    "PlanResult",
    "RawConversationState",
    "RawConversationStateUpdate",
    "RunResult",
]
