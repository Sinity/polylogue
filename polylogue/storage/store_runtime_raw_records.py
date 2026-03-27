"""Raw-ingest and artifact inspection record models."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from polylogue.types import ArtifactSupportStatus, Provider, ValidationMode, ValidationStatus


class RawConversationRecord(BaseModel):
    raw_id: str
    provider_name: str
    payload_provider: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    raw_content: bytes
    acquired_at: str
    file_mtime: str | None = None
    parsed_at: str | None = None
    parse_error: str | None = None
    validated_at: str | None = None
    validation_status: ValidationStatus | None = None
    validation_error: str | None = None
    validation_drift_count: int | None = None
    validation_provider: Provider | None = None
    validation_mode: ValidationMode | None = None

    @field_validator("raw_id", "provider_name", "source_path")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("raw_content")
    @classmethod
    def non_empty_bytes(cls, v: bytes) -> bytes:
        if not v:
            raise ValueError("raw_content cannot be empty")
        return v

    @field_validator("validation_status", mode="before")
    @classmethod
    def coerce_validation_status(cls, v: object) -> ValidationStatus | None:
        if v is None:
            return None
        return ValidationStatus.from_string(str(v))

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_provider", mode="before")
    @classmethod
    def coerce_validation_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_mode", mode="before")
    @classmethod
    def coerce_validation_mode(cls, v: object) -> ValidationMode | None:
        if v is None:
            return None
        return ValidationMode.from_string(str(v))


class ArtifactObservationRecord(BaseModel):
    observation_id: str
    raw_id: str
    provider_name: str
    payload_provider: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    wire_format: str | None = None
    artifact_kind: str
    classification_reason: str
    parse_as_conversation: bool
    schema_eligible: bool
    support_status: ArtifactSupportStatus
    malformed_jsonl_lines: int = 0
    decode_error: str | None = None
    bundle_scope: str | None = None
    cohort_id: str | None = None
    resolved_package_version: str | None = None
    resolved_element_kind: str | None = None
    resolution_reason: str | None = None
    link_group_key: str | None = None
    sidecar_agent_type: str | None = None
    first_observed_at: str
    last_observed_at: str

    @field_validator(
        "observation_id",
        "raw_id",
        "provider_name",
        "source_path",
        "artifact_kind",
        "classification_reason",
        "first_observed_at",
        "last_observed_at",
    )
    @classmethod
    def observation_non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("payload_provider", mode="before")
    @classmethod
    def coerce_observation_payload_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("support_status", mode="before")
    @classmethod
    def coerce_support_status(cls, v: object) -> ArtifactSupportStatus:
        return ArtifactSupportStatus.from_string(str(v))
