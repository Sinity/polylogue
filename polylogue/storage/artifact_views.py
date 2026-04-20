"""Artifact cohort summaries layered on top of durable archive observations."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from polylogue.types import ArtifactSupportStatus, Provider


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


__all__ = ["ArtifactCohortSummary"]
