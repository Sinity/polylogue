"""Derived-model statuses for session products and archive read models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TypeAlias

from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage.action_events.artifacts import ActionEventArtifactState
from polylogue.storage.runtime import ACTION_EVENT_MATERIALIZER_VERSION, SESSION_PRODUCT_MATERIALIZER_VERSION

MetricValue: TypeAlias = int | bool
Metrics: TypeAlias = Mapping[str, MetricValue]


def pending_rows(source_rows: int, materialized_rows: int) -> int:
    return max(0, source_rows - materialized_rows)


def pending_docs(source_docs: int, materialized_docs: int) -> int:
    return max(0, source_docs - materialized_docs)


def _metric_int(metrics: Metrics, key: str) -> int:
    return int(metrics[key])


def _metric_bool(metrics: Metrics, key: str) -> bool:
    return bool(metrics[key])


def _metric_bool_default(metrics: Metrics, key: str, *, default: bool) -> bool:
    return bool(metrics.get(key, default))


def _ready_detail(*, ready: bool, ready_detail: str, pending_detail: str) -> str:
    return ready_detail if ready else pending_detail


def _matches_version(metrics: Metrics, *debt_keys: str) -> bool:
    return all(_metric_int(metrics, key) == 0 for key in debt_keys)


def _fts_status(
    metrics: Metrics,
    *,
    name: str,
    label: str,
    ready_key: str,
    source_rows_key: str,
    materialized_rows_key: str,
    duplicate_key: str,
) -> DerivedModelStatus:
    ready = _metric_bool(metrics, ready_key)
    source_rows = _metric_int(metrics, source_rows_key)
    materialized_rows = _metric_int(metrics, materialized_rows_key)
    duplicate_rows = _metric_int(metrics, duplicate_key)
    return DerivedModelStatus(
        name=name,
        ready=ready,
        detail=_ready_detail(
            ready=ready,
            ready_detail=f"{label} ready ({materialized_rows:,}/{source_rows:,} rows)",
            pending_detail=f"{label} pending ({materialized_rows:,}/{source_rows:,} rows, duplicates {duplicate_rows:,})",
        ),
        source_rows=source_rows,
        materialized_rows=materialized_rows,
        pending_rows=pending_rows(source_rows, materialized_rows),
        stale_rows=duplicate_rows,
    )


# ---------------------------------------------------------------------------
# Action/search statuses
# ---------------------------------------------------------------------------


def _message_fts_status(metrics: Metrics) -> DerivedModelStatus:
    ready = _metric_bool(metrics, "message_fts_ready")
    exact_counts = _metric_bool_default(metrics, "message_fts_exact_counts", default=True)
    source_rows = _metric_int(metrics, "message_source_rows")
    materialized_rows = _metric_int(metrics, "message_fts_rows")
    if ready:
        detail = (
            f"Messages FTS ready ({materialized_rows:,}/{source_rows:,} rows)"
            if exact_counts
            else "Messages FTS present"
        )
    else:
        detail = (
            f"Messages FTS pending ({materialized_rows:,}/{source_rows:,} rows)"
            if exact_counts
            else "Messages FTS missing or empty; use --deep to verify full coverage"
        )
    return DerivedModelStatus(
        name="messages_fts",
        ready=ready,
        detail=detail,
        source_rows=source_rows,
        materialized_rows=materialized_rows,
        pending_rows=pending_rows(source_rows, materialized_rows),
    )


def build_action_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    state = ActionEventArtifactState.from_metrics(metrics)
    action_rows_status = replace(state.row_status(), materializer_version=ACTION_EVENT_MATERIALIZER_VERSION)
    return {
        "messages_fts": _message_fts_status(metrics),
        "action_events": action_rows_status,
        "action_events_fts": state.fts_status(),
    }


# ---------------------------------------------------------------------------
# Profile/status statuses
# ---------------------------------------------------------------------------


def build_profile_fts_status(
    metrics: Metrics,
    *,
    key_prefix: str,
    name: str,
    label: str,
) -> DerivedModelStatus:
    return _fts_status(
        metrics,
        name=name,
        label=label,
        ready_key=f"{key_prefix}_ready",
        source_rows_key="profile_rows",
        materialized_rows_key=f"{key_prefix}_rows",
        duplicate_key=f"{key_prefix}_duplicates",
    )


def _profile_rows_status(metrics: Metrics) -> DerivedModelStatus:
    ready = _metric_bool(metrics, "profile_rows_ready")
    profile_rows = _metric_int(metrics, "profile_rows")
    total_conversations = _metric_int(metrics, "total_conversations")
    stale_rows = _metric_int(metrics, "stale_profile_rows")
    orphan_rows = _metric_int(metrics, "orphan_profile_rows")
    return DerivedModelStatus(
        name="session_profile_rows",
        ready=ready,
        detail=_ready_detail(
            ready=ready,
            ready_detail=f"Session-profile rows ready ({profile_rows:,}/{total_conversations:,} conversations)",
            pending_detail=f"Session-profile rows pending ({profile_rows:,}/{total_conversations:,} conversations)",
        ),
        source_documents=total_conversations,
        materialized_documents=profile_rows,
        pending_documents=_metric_int(metrics, "missing_profile_rows"),
        stale_rows=stale_rows,
        orphan_rows=orphan_rows,
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        matches_version=stale_rows == 0 and orphan_rows == 0,
    )


def build_profile_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    return {
        "session_profile_rows": _profile_rows_status(metrics),
        "session_profile_merged_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_merged_fts",
            name="session_profile_merged_fts",
            label="Session-profile merged FTS",
        ),
        "session_profile_evidence_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_evidence_fts",
            name="session_profile_evidence_fts",
            label="Session-profile evidence FTS",
        ),
        "session_profile_inference_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_inference_fts",
            name="session_profile_inference_fts",
            label="Session-profile inference FTS",
        ),
        "session_profile_enrichment_fts": build_profile_fts_status(
            metrics,
            key_prefix="profile_enrichment_fts",
            name="session_profile_enrichment_fts",
            label="Session-profile enrichment FTS",
        ),
    }


# ---------------------------------------------------------------------------
# Timeline/work-product statuses
# ---------------------------------------------------------------------------


def _session_timeline_status(
    metrics: Metrics,
    *,
    name: str,
    label: str,
    ready_key: str,
    rows_key: str,
    expected_rows_key: str,
    stale_key: str,
    orphan_key: str,
) -> DerivedModelStatus:
    ready = _metric_bool(metrics, ready_key)
    rows = _metric_int(metrics, rows_key)
    expected_rows = _metric_int(metrics, expected_rows_key)
    profile_rows = _metric_int(metrics, "profile_rows")
    return DerivedModelStatus(
        name=name,
        ready=ready,
        detail=_ready_detail(
            ready=ready,
            ready_detail=f"{label} ready ({rows:,}/{expected_rows:,} rows)",
            pending_detail=f"{label} pending ({rows:,}/{expected_rows:,} rows)",
        ),
        source_documents=profile_rows,
        materialized_documents=profile_rows if profile_rows else 0,
        source_rows=expected_rows,
        materialized_rows=rows,
        pending_rows=pending_rows(expected_rows, rows),
        stale_rows=_metric_int(metrics, stale_key),
        orphan_rows=_metric_int(metrics, orphan_key),
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        matches_version=_matches_version(metrics, stale_key, orphan_key),
    )


def _work_threads_status(metrics: Metrics) -> DerivedModelStatus:
    ready = _metric_bool(metrics, "threads_ready")
    rows = _metric_int(metrics, "work_thread_rows")
    roots = _metric_int(metrics, "total_thread_roots")
    return DerivedModelStatus(
        name="work_threads",
        ready=ready,
        detail=_ready_detail(
            ready=ready,
            ready_detail=f"Work threads ready ({rows:,}/{roots:,} roots)",
            pending_detail=f"Work threads pending ({rows:,}/{roots:,} roots)",
        ),
        source_documents=roots,
        materialized_documents=rows,
        pending_documents=pending_docs(roots, rows),
        stale_rows=_metric_int(metrics, "stale_thread_rows"),
        orphan_rows=_metric_int(metrics, "orphan_thread_rows"),
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        matches_version=_matches_version(metrics, "stale_thread_rows", "orphan_thread_rows"),
    )


def build_timeline_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    return {
        "session_work_event_inference": _session_timeline_status(
            metrics,
            name="session_work_event_inference",
            label="Session work-event inference",
            ready_key="work_event_rows_ready",
            rows_key="work_event_rows",
            expected_rows_key="expected_work_event_rows",
            stale_key="stale_work_event_rows",
            orphan_key="orphan_work_event_rows",
        ),
        "session_work_event_inference_fts": _fts_status(
            metrics,
            name="session_work_event_inference_fts",
            label="Session work-event inference FTS",
            ready_key="work_event_fts_ready",
            source_rows_key="work_event_rows",
            materialized_rows_key="work_event_fts_rows",
            duplicate_key="work_event_fts_duplicates",
        ),
        "session_phase_inference": _session_timeline_status(
            metrics,
            name="session_phase_inference",
            label="Session phase inference",
            ready_key="phase_rows_ready",
            rows_key="phase_rows",
            expected_rows_key="expected_phase_rows",
            stale_key="stale_phase_rows",
            orphan_key="orphan_phase_rows",
        ),
        "work_threads": _work_threads_status(metrics),
        "work_threads_fts": _fts_status(
            metrics,
            name="work_threads_fts",
            label="Work-thread FTS",
            ready_key="thread_fts_ready",
            source_rows_key="work_thread_rows",
            materialized_rows_key="work_thread_fts_rows",
            duplicate_key="work_thread_fts_duplicates",
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate statuses
# ---------------------------------------------------------------------------


def _aggregate_rows_status(
    metrics: Metrics,
    *,
    name: str,
    label: str,
    ready_key: str,
    rows_key: str,
    expected_rows_key: str,
    stale_key: str,
) -> DerivedModelStatus:
    ready = _metric_bool(metrics, ready_key)
    rows = _metric_int(metrics, rows_key)
    expected_rows = _metric_int(metrics, expected_rows_key)
    return DerivedModelStatus(
        name=name,
        ready=ready,
        detail=_ready_detail(
            ready=ready,
            ready_detail=f"{label} ready ({rows:,}/{expected_rows:,} rows)",
            pending_detail=f"{label} pending ({rows:,}/{expected_rows:,} rows)",
        ),
        source_rows=expected_rows,
        materialized_rows=rows,
        pending_rows=pending_rows(expected_rows, rows),
        stale_rows=_metric_int(metrics, stale_key),
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        matches_version=_matches_version(metrics, stale_key),
    )


def _week_summary_status(metrics: Metrics) -> DerivedModelStatus:
    return DerivedModelStatus(
        name="week_session_summaries",
        ready=_metric_bool(metrics, "week_summaries_ready"),
        detail=(
            "Week session summaries ready (derived from day-session summaries)"
            if _metric_bool(metrics, "week_summaries_ready")
            else "Week session summaries pending (day-session summaries not ready)"
        ),
        source_rows=_metric_int(metrics, "expected_day_summary_rows"),
        materialized_rows=_metric_int(metrics, "day_summary_rows"),
        pending_rows=pending_rows(
            _metric_int(metrics, "expected_day_summary_rows"),
            _metric_int(metrics, "day_summary_rows"),
        ),
        stale_rows=_metric_int(metrics, "stale_day_summary_rows"),
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        matches_version=_matches_version(metrics, "stale_day_summary_rows"),
    )


def build_aggregate_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    return {
        "session_tag_rollups": _aggregate_rows_status(
            metrics,
            name="session_tag_rollups",
            label="Session tag rollups",
            ready_key="tag_rollups_ready",
            rows_key="tag_rollup_rows",
            expected_rows_key="expected_tag_rollup_rows",
            stale_key="stale_tag_rollup_rows",
        ),
        "day_session_summaries": _aggregate_rows_status(
            metrics,
            name="day_session_summaries",
            label="Day session summaries",
            ready_key="day_summaries_ready",
            rows_key="day_summary_rows",
            expected_rows_key="expected_day_summary_rows",
            stale_key="stale_day_summary_rows",
        ),
        "week_session_summaries": _week_summary_status(metrics),
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def build_archive_product_statuses(metrics: Metrics) -> dict[str, DerivedModelStatus]:
    return {
        **build_action_statuses(metrics),
        **build_profile_statuses(metrics),
        **build_timeline_statuses(metrics),
        **build_aggregate_statuses(metrics),
    }


__all__ = ["build_archive_product_statuses"]
