"""Closed request and result contracts for structural annotation joins."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from polylogue.core.enums import AssertionStatus
from polylogue.surfaces.outcome import OutcomeEnvelope

AnnotationGroupDimension = Literal["repo", "model", "time", "origin"]
AnnotationJoinDiagnosticCode = Literal["missing_target", "ambiguous_target", "schema_drift", "invalid_value"]
_MAX_JOIN_LIMIT = 1_000


class AnnotationStructuralJoinRequest(BaseModel):
    """Explicit schema/status selection for one bounded annotation join."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1, max_length=256)
    schema_version: int = Field(ge=1)
    statuses: tuple[AssertionStatus, ...] = Field(min_length=1)
    target_kind: str | None = Field(default=None, min_length=1, max_length=64)
    group_by: tuple[AnnotationGroupDimension, ...] = ()
    limit: int = Field(default=500, ge=1, le=_MAX_JOIN_LIMIT)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_lifecycle_selection(self) -> AnnotationStructuralJoinRequest:
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("annotation join statuses must be unique")
        if AssertionStatus.ACCEPTED in self.statuses and AssertionStatus.ACTIVE in self.statuses:
            raise ValueError("accepted and active cannot be joined together because they represent one label lifecycle")
        return self


class AnnotationJoinDiagnostic(BaseModel):
    """One bounded reason a selected annotation did not join cleanly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: AnnotationJoinDiagnosticCode
    assertion_ref: str
    target_ref: str
    detail: str = Field(max_length=512)


class AnnotationStructuralJoinRow(BaseModel):
    """One label joined to one exact target without collapsing label identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assertion_ref: str
    batch_ref: str | None
    schema_id: str
    schema_version: int
    status: AssertionStatus
    labeler_ref: str | None
    adjudicator_ref: str | None
    source_assertion_ref: str
    judgment_ref: str | None
    judgment_decision: str | None
    judgment_reason: str | None
    supersedes: tuple[str, ...]
    target_ref: str
    value: dict[str, Any]
    evidence_refs: tuple[str, ...]
    structural: dict[str, Any]


class AnnotationStructuralGroup(BaseModel):
    """Deterministic aggregate over successfully joined label rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: dict[str, Any]
    label_count: int = Field(ge=1)
    distinct_target_count: int = Field(ge=1)


class AnnotationStructuralJoinResult(BaseModel):
    """Rows, aggregates, and explicit non-join/fanout accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualified_schema_id: str
    requested_statuses: tuple[AssertionStatus, ...]
    selected_annotation_count: int = Field(ge=0)
    matched_annotation_count: int = Field(ge=0)
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    selection_truncated: bool
    joined_count: int = Field(ge=0)
    missing_target_count: int = Field(ge=0)
    ambiguous_target_count: int = Field(ge=0)
    schema_drift_count: int = Field(ge=0)
    invalid_value_count: int = Field(ge=0)
    multi_label_target_count: int = Field(ge=0)
    duplicate_label_count: int = Field(ge=0)
    diagnostics_truncated: bool
    diagnostics: tuple[AnnotationJoinDiagnostic, ...]
    rows: tuple[AnnotationStructuralJoinRow, ...]
    groups: tuple[AnnotationStructuralGroup, ...]


class AnnotationJoinOperationResult(BaseModel):
    """The unchanged join report and its explicit selection verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: AnnotationStructuralJoinResult
    outcome: OutcomeEnvelope
