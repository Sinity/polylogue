"""Timeline-oriented derived insight storage models."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from polylogue.insights.archive_models import (
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.run_projection import ContextSnapshot, ObservedEvent, ProjectedRun
from polylogue.storage.runtime.store_constants import (
    SESSION_INFERENCE_FAMILY,
    SESSION_INFERENCE_VERSION,
    SESSION_INSIGHT_MATERIALIZER_VERSION,
)
from polylogue.types import SessionId


class SessionWorkEventRecord(BaseModel):
    event_id: str
    session_id: SessionId
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    input_row_count: int = 0
    source_name: str
    event_index: int
    heuristic_label: str
    confidence: float
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    canonical_session_date: str | None = None
    summary: str
    file_paths: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    evidence_payload: WorkEventEvidencePayload
    inference_payload: WorkEventInferencePayload
    search_text: str
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "event_id",
        "session_id",
        "materialized_at",
        "source_name",
        "heuristic_label",
        "summary",
        "search_text",
        "inference_family",
    )
    @classmethod
    def work_event_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


class SessionPhaseRecord(BaseModel):
    phase_id: str
    session_id: SessionId
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    source_sort_key: float | None = None
    input_high_water_mark: str | None = None
    input_high_water_mark_source: str | None = None
    input_row_count: int = 0
    source_name: str
    phase_index: int
    kind: str
    start_index: int
    end_index: int
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int = 0
    canonical_session_date: str | None = None
    confidence: float = 0.0
    evidence_reasons: tuple[str, ...] = ()
    tool_counts: dict[str, int]
    word_count: int = 0
    evidence_payload: SessionPhaseEvidencePayload
    inference_payload: SessionPhaseInferencePayload
    search_text: str
    inference_version: int = SESSION_INFERENCE_VERSION
    inference_family: str = SESSION_INFERENCE_FAMILY

    @field_validator(
        "phase_id",
        "session_id",
        "materialized_at",
        "source_name",
        "kind",
        "search_text",
        "inference_family",
    )
    @classmethod
    def session_phase_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


class SessionRunRecord(BaseModel):
    """Materialized ``session_runs`` row wrapping one :class:`ProjectedRun`.

    The run model is the single source of truth: typed columns (run_ref,
    harness, role, status, ...) are derived from it on write, and ``payload_json``
    stores ``run.model_dump(mode="json")`` for lossless hydration on read. This
    guarantees stored rows round-trip back to the exact projection output.
    """

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    run: ProjectedRun
    search_text: str


class SessionObservedEventRecord(BaseModel):
    """Materialized ``session_observed_events`` row wrapping one :class:`ObservedEvent`."""

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    event: ObservedEvent
    search_text: str


class SessionContextSnapshotRecord(BaseModel):
    """Materialized ``session_context_snapshots`` row wrapping one :class:`ContextSnapshot`."""

    session_id: SessionId
    position: int
    materializer_version: int = SESSION_INSIGHT_MATERIALIZER_VERSION
    materialized_at: str
    source_updated_at: str | None = None
    snapshot: ContextSnapshot
    search_text: str


__all__ = [
    "SessionContextSnapshotRecord",
    "SessionObservedEventRecord",
    "SessionPhaseRecord",
    "SessionRunRecord",
    "SessionWorkEventRecord",
]
