"""Session insight repair assessment helpers."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.storage.insights.session.runtime import SessionInsightReadyFlag, SessionInsightStatusSnapshot

_SESSION_INSIGHT_READY_FLAGS: tuple[SessionInsightReadyFlag, ...] = (
    "profile_rows_ready",
    "latency_profile_rows_ready",
    "work_event_inference_rows_ready",
    "work_event_inference_fts_ready",
    "phase_inference_rows_ready",
    "run_rows_ready",
    "observed_event_rows_ready",
    "context_snapshot_rows_ready",
    "threads_ready",
    "threads_fts_ready",
    "tag_rollups_ready",
)


@dataclass(slots=True, frozen=True)
class SessionInsightRepairAssessment:
    row_debt: int
    fts_debt: int

    @property
    def pending(self) -> int:
        return self.row_debt + self.fts_debt


def assess_session_insight_repairs(status: SessionInsightStatusSnapshot) -> SessionInsightRepairAssessment:
    return SessionInsightRepairAssessment(
        row_debt=_session_insight_row_repair_count(status),
        fts_debt=_session_insight_fts_repair_count(status),
    )


def session_insight_status_ready(status: SessionInsightStatusSnapshot) -> bool:
    return all(status.ready_flag(flag) for flag in _SESSION_INSIGHT_READY_FLAGS)


def session_insight_fts_ready(status: SessionInsightStatusSnapshot) -> bool:
    return status.ready_flag("work_event_inference_fts_ready") and status.ready_flag("threads_fts_ready")


def _positive_count(value: int) -> int:
    return max(0, value)


def _fts_repair_count(*, source_rows: int, indexed_rows: int, duplicates: int) -> int:
    return _positive_count(source_rows - indexed_rows) + _positive_count(duplicates)


def _session_insight_row_repair_count(status: SessionInsightStatusSnapshot) -> int:
    return (
        status.missing_profile_row_count
        + status.missing_session_profile_materialization_count
        + status.stale_profile_row_count
        + status.orphan_profile_row_count
        + status.missing_latency_profile_row_count
        + status.missing_latency_materialization_count
        + status.stale_latency_profile_row_count
        + status.orphan_latency_profile_row_count
        + status.missing_work_event_materialization_count
        + status.stale_work_event_inference_count
        + status.orphan_work_event_inference_count
        + status.missing_phase_materialization_count
        + status.stale_phase_inference_count
        + status.orphan_phase_inference_count
        + status.missing_run_materialization_count
        + status.missing_observed_event_materialization_count
        + status.missing_context_snapshot_materialization_count
        + status.missing_thread_materialization_count
        + status.stale_thread_count
        + status.orphan_thread_count
        + status.stale_tag_rollup_count
        + status.stale_day_summary_count
    )


def _session_insight_fts_repair_count(status: SessionInsightStatusSnapshot) -> int:
    return sum(
        (
            _fts_repair_count(
                source_rows=status.work_event_inference_count,
                indexed_rows=status.work_event_inference_fts_count,
                duplicates=status.work_event_inference_fts_duplicate_count,
            ),
            _fts_repair_count(
                source_rows=status.thread_count,
                indexed_rows=status.thread_fts_count,
                duplicates=status.thread_fts_duplicate_count,
            ),
        )
    )


__all__ = [
    "SessionInsightRepairAssessment",
    "assess_session_insight_repairs",
    "session_insight_fts_ready",
    "session_insight_status_ready",
]
