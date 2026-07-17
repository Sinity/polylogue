"""Typed session-profile semantic models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from polylogue.archive.phase.extraction import SessionPhase
from polylogue.archive.semantic.facts import SessionSemanticFacts
from polylogue.archive.session.attribution import SessionAttribution
from polylogue.archive.session.documents import (
    SessionPhaseDocument,
    SessionProfileDocument,
)
from polylogue.archive.session.extraction import WorkEvent
from polylogue.archive.session.repo_identity import normalize_repo_names, normalize_repo_paths
from polylogue.core.payload_coercion import (
    coerce_float,
    coerce_int,
    int_pair,
    mapping_or_empty,
    mapping_sequence,
    optional_date,
    optional_datetime,
    optional_string,
    string_int_mapping,
    string_sequence,
)

SessionPhasePayload = SessionPhaseDocument
SessionProfilePayload = SessionProfileDocument


def _scalar_mapping(value: object) -> dict[str, int | float | str | None]:
    return {
        str(key): item
        for key, item in mapping_or_empty(value).items()
        if isinstance(item, int | float | str) or item is None
    }


def _phase_to_dict(phase: SessionPhase) -> SessionPhasePayload:
    start_time = phase.start_time.isoformat() if phase.start_time else None
    end_time = phase.end_time.isoformat() if phase.end_time else None
    canonical_session_date = phase.canonical_session_date.isoformat() if phase.canonical_session_date else None
    return {
        "start_time": start_time,
        "end_time": end_time,
        "canonical_session_date": canonical_session_date,
        "timing_provenance": _range_timing_provenance(start_time, end_time),
        "date_provenance": _date_provenance(canonical_session_date, start_time, end_time),
        "message_range": list(phase.message_range),
        "duration_ms": phase.duration_ms,
        "phase_idle_threshold_ms": phase.phase_idle_threshold_ms,
        "tool_counts": phase.tool_counts,
        "word_count": phase.word_count,
        "confidence": phase.confidence,
        "evidence": list(phase.evidence),
    }


def _phase_from_mapping(payload: SessionPhasePayload | Mapping[str, object]) -> SessionPhase:
    return SessionPhase(
        start_time=optional_datetime(payload.get("start_time")),
        end_time=optional_datetime(payload.get("end_time")),
        canonical_session_date=optional_date(payload.get("canonical_session_date")),
        message_range=int_pair(payload.get("message_range")),
        duration_ms=coerce_int(payload.get("duration_ms"), 0),
        phase_idle_threshold_ms=coerce_int(payload.get("phase_idle_threshold_ms"), 300_000),
        tool_counts=string_int_mapping(payload.get("tool_counts")),
        word_count=coerce_int(payload.get("word_count"), 0),
        confidence=coerce_float(payload.get("confidence"), 0.0),
        evidence=string_sequence(payload.get("evidence")),
    )


def _range_timing_provenance(start_time: str | None, end_time: str | None) -> str:
    if start_time is not None and end_time is not None:
        return "timestamped_range"
    if start_time is not None:
        return "start_timestamp_only"
    if end_time is not None:
        return "end_timestamp_only"
    return "untimestamped"


def _date_provenance(canonical_session_date: str | None, start_time: str | None, end_time: str | None) -> str:
    if canonical_session_date is None:
        return "none"
    if start_time is not None or end_time is not None:
        return "event_timestamp"
    return "date_only"


@dataclass(frozen=True)
class SessionProfile:
    """Complete semantic profile of a session session."""

    session_id: str
    origin: str
    title: str | None
    created_at: datetime | None
    updated_at: datetime | None
    message_count: int
    substantive_count: int
    tool_use_count: int
    thinking_count: int
    attachment_count: int
    word_count: int
    total_cost_usd: float
    total_duration_ms: int
    tool_categories: dict[str, int]
    repo_paths: tuple[str, ...]
    cwd_paths: tuple[str, ...]
    branch_names: tuple[str, ...]
    file_paths_touched: tuple[str, ...]
    languages_detected: tuple[str, ...]
    repo_names: tuple[str, ...]
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]
    inferred_topic: str | None = None
    inferred_topic_source: str = "absent"
    first_message_at: datetime | None = None
    last_message_at: datetime | None = None
    timestamp_source: str = "provider_supplied"
    timestamped_message_count: int = 0
    untimestamped_message_count: int = 0
    timestamp_coverage: str = "none"
    canonical_session_date: date | None = None
    engaged_duration_ms: int = 0
    tool_active_duration_ms: int = 0
    wall_duration_ms: int = 0
    workflow_shape: str = "unknown"
    workflow_shape_confidence: float = 0.0
    workflow_shape_features: dict[str, int | float | str] = field(default_factory=dict)
    terminal_state: str = "unknown"
    terminal_state_confidence: float = 0.0
    terminal_state_evidence: dict[str, int | float | str | None] = field(default_factory=dict)
    cost_is_estimated: bool = False
    logical_session_id: str | None = None
    thread_id: str | None = None
    continuation_depth: int = 0
    compaction_count: int = 0
    tags: tuple[str, ...] = ()
    auto_tags: tuple[str, ...] = ()
    is_continuation: bool = False
    parent_id: str | None = None

    # ------------------------------------------------------------------
    # Derived temporal timing fields  (issue #804)
    # ------------------------------------------------------------------
    thinking_duration_ms: int = 0
    output_duration_ms: int = 0
    tool_duration_ms: int = 0
    latency_percentiles_ms: dict[str, int] = field(default_factory=dict)
    tool_calls_per_minute: float = 0.0
    timing_provenance: str = "sort_key_estimated"

    # Cost / token attribution (added by #942/#943 cost forecasting work).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_credit_cost: float = 0.0
    cost_provenance: str = "unknown"
    per_model_cost_json: str = "{}"
    # 1vpm.1: the SESSION-WIDE dominant model by assistant output-token
    # share, and its canonical vendor family (anthropic/openai/deepseek/...)
    # via archive.semantic.pricing.canonical_model_family. polylogue-4c27:
    # this is a session-level aggregate fallback, explicitly excluded from
    # the `delegations` view's per-turn dispatch/requested/child-observed
    # model identity -- see that view's DDL comment.
    primary_model_name: str | None = None
    primary_model_family: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.latency_percentiles_ms, dict):
            object.__setattr__(self, "latency_percentiles_ms", dict(self.latency_percentiles_ms or {}))

    def to_dict(self) -> SessionProfilePayload:
        return {
            "session_id": self.session_id,
            "origin": self.origin,
            "title": self.title,
            "inferred_topic": self.inferred_topic,
            "inferred_topic_source": self.inferred_topic_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "substantive_count": self.substantive_count,
            "tool_use_count": self.tool_use_count,
            "thinking_count": self.thinking_count,
            "attachment_count": self.attachment_count,
            "word_count": self.word_count,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "tool_categories": self.tool_categories,
            "repo_paths": list(self.repo_paths),
            "cwd_paths": list(self.cwd_paths),
            "branch_names": list(self.branch_names),
            "file_paths_touched": list(self.file_paths_touched),
            "languages_detected": list(self.languages_detected),
            "repo_names": list(self.repo_names),
            "work_events": [event.to_dict() for event in self.work_events],
            "phases": [_phase_to_dict(phase) for phase in self.phases],
            "first_message_at": self.first_message_at.isoformat() if self.first_message_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "timestamp_source": self.timestamp_source,
            "timestamped_message_count": self.timestamped_message_count,
            "untimestamped_message_count": self.untimestamped_message_count,
            "timestamp_coverage": self.timestamp_coverage,
            "canonical_session_date": self.canonical_session_date.isoformat() if self.canonical_session_date else None,
            "engaged_duration_ms": self.engaged_duration_ms,
            "engaged_minutes": round(self.engaged_duration_ms / 60_000.0, 4),
            "tool_active_duration_ms": self.tool_active_duration_ms,
            "tool_active_minutes": round(self.tool_active_duration_ms / 60_000.0, 4),
            "wall_duration_ms": self.wall_duration_ms,
            "workflow_shape": self.workflow_shape,
            "workflow_shape_confidence": self.workflow_shape_confidence,
            "workflow_shape_features": dict(self.workflow_shape_features),
            "terminal_state": self.terminal_state,
            "terminal_state_confidence": self.terminal_state_confidence,
            "terminal_state_evidence": dict(self.terminal_state_evidence),
            "cost_is_estimated": self.cost_is_estimated,
            "logical_session_id": self.logical_session_id,
            "compaction_count": self.compaction_count,
            "thread_id": self.thread_id,
            "continuation_depth": self.continuation_depth,
            "tags": list(self.tags),
            "auto_tags": list(self.auto_tags),
            "is_continuation": self.is_continuation,
            "parent_id": self.parent_id,
            "thinking_duration_ms": self.thinking_duration_ms,
            "output_duration_ms": self.output_duration_ms,
            "tool_duration_ms": self.tool_duration_ms,
            "latency_percentiles_ms": dict(self.latency_percentiles_ms),
            "tool_calls_per_minute": self.tool_calls_per_minute,
            "timing_provenance": self.timing_provenance,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_write_tokens": self.total_cache_write_tokens,
            "total_credit_cost": self.total_credit_cost,
            "cost_provenance": self.cost_provenance,
            "per_model_cost_json": self.per_model_cost_json,
            "primary_model_name": self.primary_model_name,
            "primary_model_family": self.primary_model_family,
        }

    @classmethod
    def from_dict(cls, payload: SessionProfilePayload | Mapping[str, object]) -> SessionProfile:
        repo_paths = normalize_repo_paths(string_sequence(payload.get("repo_paths")))
        explicit_repo_names = tuple(
            sorted({item.strip() for item in string_sequence(payload.get("repo_names")) if item.strip()})
        )
        repo_names = explicit_repo_names or normalize_repo_names(repo_paths=repo_paths)
        return cls(
            session_id=str(payload["session_id"]),
            origin=str(payload["origin"]),
            title=optional_string(payload.get("title")),
            inferred_topic=optional_string(payload.get("inferred_topic")),
            inferred_topic_source=optional_string(payload.get("inferred_topic_source")) or "absent",
            created_at=optional_datetime(payload.get("created_at")),
            updated_at=optional_datetime(payload.get("updated_at")),
            message_count=coerce_int(payload.get("message_count"), 0),
            substantive_count=coerce_int(payload.get("substantive_count"), 0),
            tool_use_count=coerce_int(payload.get("tool_use_count"), 0),
            thinking_count=coerce_int(payload.get("thinking_count"), 0),
            attachment_count=coerce_int(payload.get("attachment_count"), 0),
            word_count=coerce_int(payload.get("word_count"), 0),
            total_cost_usd=coerce_float(payload.get("total_cost_usd"), 0.0),
            total_duration_ms=coerce_int(payload.get("total_duration_ms"), 0),
            tool_categories=string_int_mapping(payload.get("tool_categories")),
            repo_paths=repo_paths,
            cwd_paths=string_sequence(payload.get("cwd_paths")),
            branch_names=string_sequence(payload.get("branch_names")),
            file_paths_touched=string_sequence(payload.get("file_paths_touched")),
            languages_detected=string_sequence(payload.get("languages_detected")),
            repo_names=repo_names,
            work_events=tuple(WorkEvent.from_dict(item) for item in mapping_sequence(payload.get("work_events"))),
            phases=tuple(_phase_from_mapping(item) for item in mapping_sequence(payload.get("phases"))),
            first_message_at=optional_datetime(payload.get("first_message_at")),
            last_message_at=optional_datetime(payload.get("last_message_at")),
            timestamp_source=optional_string(payload.get("timestamp_source")) or "provider_supplied",
            timestamped_message_count=coerce_int(payload.get("timestamped_message_count"), 0),
            untimestamped_message_count=coerce_int(payload.get("untimestamped_message_count"), 0),
            timestamp_coverage=optional_string(payload.get("timestamp_coverage")) or "none",
            canonical_session_date=optional_date(payload.get("canonical_session_date")),
            engaged_duration_ms=coerce_int(payload.get("engaged_duration_ms"), 0),
            tool_active_duration_ms=coerce_int(payload.get("tool_active_duration_ms"), 0),
            wall_duration_ms=coerce_int(payload.get("wall_duration_ms"), 0),
            workflow_shape=optional_string(payload.get("workflow_shape")) or "unknown",
            workflow_shape_confidence=coerce_float(payload.get("workflow_shape_confidence"), 0.0),
            workflow_shape_features={
                key: value
                for key, value in _scalar_mapping(payload.get("workflow_shape_features")).items()
                if value is not None
            },
            terminal_state=optional_string(payload.get("terminal_state")) or "unknown",
            terminal_state_confidence=coerce_float(payload.get("terminal_state_confidence"), 0.0),
            terminal_state_evidence=_scalar_mapping(payload.get("terminal_state_evidence")),
            cost_is_estimated=bool(payload.get("cost_is_estimated", False)),
            logical_session_id=optional_string(payload.get("logical_session_id")),
            compaction_count=coerce_int(payload.get("compaction_count"), 0),
            thread_id=optional_string(payload.get("thread_id")),
            continuation_depth=coerce_int(payload.get("continuation_depth"), 0),
            tags=string_sequence(payload.get("tags")),
            auto_tags=string_sequence(payload.get("auto_tags")),
            is_continuation=bool(payload.get("is_continuation", False)),
            parent_id=optional_string(payload.get("parent_id")),
            thinking_duration_ms=coerce_int(payload.get("thinking_duration_ms"), 0),
            output_duration_ms=coerce_int(payload.get("output_duration_ms"), 0),
            tool_duration_ms=coerce_int(payload.get("tool_duration_ms"), 0),
            latency_percentiles_ms=(
                {str(k): coerce_int(v, 0) for k, v in (payload.get("latency_percentiles_ms") or {}).items()}  # type: ignore[attr-defined]
                if isinstance(payload.get("latency_percentiles_ms"), dict)
                else {}
            ),
            tool_calls_per_minute=coerce_float(payload.get("tool_calls_per_minute"), 0.0),
            timing_provenance=optional_string(payload.get("timing_provenance")) or "sort_key_estimated",
            # Cost/token attribution — present in to_dict() and the record
            # payload but previously dropped here, so a round-tripped or
            # batch-hydrated profile reported zero tokens (and lost credit/
            # provenance/per-model cost) even when the columns were populated.
            total_input_tokens=coerce_int(payload.get("total_input_tokens"), 0),
            total_output_tokens=coerce_int(payload.get("total_output_tokens"), 0),
            total_cache_read_tokens=coerce_int(payload.get("total_cache_read_tokens"), 0),
            total_cache_write_tokens=coerce_int(payload.get("total_cache_write_tokens"), 0),
            total_credit_cost=coerce_float(payload.get("total_credit_cost"), 0.0),
            cost_provenance=optional_string(payload.get("cost_provenance")) or "unknown",
            per_model_cost_json=optional_string(payload.get("per_model_cost_json")) or "{}",
            primary_model_name=optional_string(payload.get("primary_model_name")),
            primary_model_family=optional_string(payload.get("primary_model_family")),
        )


@dataclass(frozen=True)
class SessionAnalysis:
    facts: SessionSemanticFacts
    attribution: SessionAttribution
    work_events: tuple[WorkEvent, ...]
    phases: tuple[SessionPhase, ...]


__all__ = ["SessionAnalysis", "SessionProfile"]
