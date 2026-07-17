"""Raw-ingest and artifact inspection record models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from polylogue.archive.revision_authority import RawRevisionEnvelope
from polylogue.core.enums import ArtifactSupportStatus, Provider, ValidationMode, ValidationStatus


class RawSessionRecord(BaseModel):
    raw_id: str
    blob_hash: str | None = None
    blob_publication_receipt_id: str | None = Field(default=None, exclude=True)
    payload_provider: Provider | None = None
    capture_mode: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    blob_size: int
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
    detection_warnings: str | None = None
    revision: RawRevisionEnvelope | None = None
    captured_source_revision: str | None = Field(default=None, exclude=True)
    requires_complete_record_boundary: bool = Field(default=False, exclude=True)

    @field_validator("raw_id", "blob_hash", "blob_publication_receipt_id", "source_name", "source_path")
    @classmethod
    def non_empty_string(cls, v: str | None) -> str | None:
        # Runs after type validation: required ``str`` fields (raw_id, source_path)
        # reject None at the type layer, so None only reaches here from the
        # nullable ``source_name`` (legacy/unknown source, legitimately None).
        # Reject only an explicitly-empty string.
        if v is not None and not v.strip():
            raise ValueError("Field cannot be empty")
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

    @field_validator("capture_mode", mode="before")
    @classmethod
    def coerce_capture_mode(cls, v: object) -> Provider | None:
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
    payload_provider: Provider | None = None
    source_name: str | None = None
    source_path: str
    source_index: int | None = None
    file_mtime: str | None = None
    wire_format: str | None = None
    artifact_kind: str
    classification_reason: str
    parse_as_session: bool
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
        "source_name",
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
