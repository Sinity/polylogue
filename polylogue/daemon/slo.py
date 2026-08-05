"""Steady-state ingest SLO projection over existing daemon evidence.

The daemon event ledger and cursor-lag projection remain the sources of truth.
This module adds only a bounded reducer and optional numeric samples. It never
recounts source files or creates a second backlog ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.core.enums import SloSampleLabel
from polylogue.core.stats import percentile
from polylogue.daemon.cursor_lag_status import CursorLagSummary
from polylogue.daemon.events import query_daemon_events
from polylogue.operations.slo import (
    ArchiveSloSample,
    list_slo_samples,
    record_slo_sample,
)


class SloVerdict(StrEnum):
    UNKNOWN = "unknown"
    ACTIVE = "active"
    HEALTHY = "healthy"
    IDLE = "idle"
    DRAINING = "draining"
    STALLED = "stalled"
    SUPPRESSED = "suppressed"


class SloReduction(BaseModel):
    """Reducer output for one label, honest about cold-start fields."""

    label: str
    sample_count: int = 0
    level: float | None = None
    p50: float | None = None
    p95: float | None = None
    slope_per_s: float | None = None
    eta_s: float | None = None
    burn_rate: float | None = None
    confident: bool = False


class BacklogWindow(BaseModel):
    """One bounded catch-up window derived from a catch-up cycle end event."""

    operation_id: str | None = None
    observed_at_ms: int | None = None
    phase: str = "end"
    in_progress: bool = False
    duration_s: float = 0.0
    backlog_start: int = 0
    backlog_end: int = 0
    offered_work: int = 0
    drained_work: int = 0
    drain_rate_per_s: float = 0.0
    offered_rate_per_s: float = 0.0
    verdict: SloVerdict = SloVerdict.UNKNOWN
    reason: str = "no catch-up cycle evidence"
    bulk_import_suppressed: bool = False


class IngestSloStatus(BaseModel):
    """Status/readiness projection for the live ingest SLO."""

    available: bool = False
    verdict: SloVerdict = SloVerdict.UNKNOWN
    reason: str = "cold start: no catch-up cycle evidence"
    backlog: int = 0
    offered_work: int = 0
    drained_work: int = 0
    drain_rate_per_s: float = 0.0
    offered_rate_per_s: float = 0.0
    cursor_lag_stuck_file_count: int = 0
    cursor_lag_degraded_file_count: int = 0
    bulk_import_suppressed: bool = False
    latest_window: BacklogWindow | None = None
    reductions: dict[str, SloReduction] = Field(default_factory=dict)


_BULK_IMPORT_START_KINDS = frozenset({"bulk_import_started", "bulk_import_opened"})
_BULK_IMPORT_END_KINDS = frozenset({"bulk_import_completed", "bulk_import_closed"})


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key, 0)
    return max(0, int(value)) if isinstance(value, int | float | str) and not isinstance(value, bool) else 0


def _float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key, 0.0)
    return max(0.0, float(value)) if isinstance(value, int | float | str) and not isinstance(value, bool) else 0.0


def _event_observed_at_ms(event: Mapping[str, object]) -> int | None:
    """Decode the daemon event reader's canonical ISO ``ts`` field."""
    value = event.get("ts")
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return int(timestamp.timestamp() * 1000)


def _event_order_key(index: int, event: Mapping[str, object]) -> tuple[int, int, int]:
    """Order event projections by their persisted timestamp, then event id."""
    observed_at_ms = _event_observed_at_ms(event)
    return (observed_at_ms if observed_at_ms is not None else -1, _int(event, "id"), index)


def _events_newest_first(events: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        event for _index, event in sorted(enumerate(events), key=lambda item: _event_order_key(*item), reverse=True)
    ]


def bulk_import_is_open(events: Sequence[Mapping[str, object]]) -> bool:
    """Return whether the newest explicit bulk-import marker is still open."""
    for event in _events_newest_first(events):
        kind = event.get("kind")
        if not isinstance(kind, str):
            continue
        if kind in _BULK_IMPORT_START_KINDS:
            return True
        if kind in _BULK_IMPORT_END_KINDS:
            return False
    return False


def backlog_window_from_event(
    event: Mapping[str, object],
    *,
    bulk_import_suppressed: bool = False,
) -> BacklogWindow | None:
    """Project one existing ``catch_up_cycle`` end event into a verdict."""
    if event.get("kind") != "catch_up_cycle":
        return None
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return None
    phase = payload.get("phase")
    if phase not in {"start", "end"}:
        return None
    duration_s = _float(payload, "duration_ms") / 1000.0
    backlog_start = _int(payload, "backlog_start")
    backlog_end = _int(payload, "backlog_end")
    offered_work = _int(payload, "discovered")
    drained_work = min(
        backlog_start, _int(payload, "ingested") + _int(payload, "quarantine_count") + _int(payload, "skipped")
    )
    drain_rate = drained_work / duration_s if duration_s > 0 else 0.0
    offered_rate = offered_work / duration_s if duration_s > 0 else 0.0
    if bulk_import_suppressed:
        verdict = SloVerdict.SUPPRESSED
        reason = "bulk-import marker is open"
    elif phase == "start":
        verdict = SloVerdict.ACTIVE
        reason = "catch-up cycle is in progress"
    elif backlog_end <= 0:
        verdict = SloVerdict.HEALTHY
        reason = "backlog drained"
    elif offered_work <= 0:
        verdict = SloVerdict.IDLE
        reason = "backlog exists but no work was offered in this window"
    elif drained_work <= 0:
        verdict = SloVerdict.STALLED
        reason = "work was offered but no backlog unit drained"
    else:
        verdict = SloVerdict.DRAINING
        reason = "offered work is draining the backlog"
    operation_id = event.get("operation_id")
    return BacklogWindow(
        operation_id=operation_id if isinstance(operation_id, str) else None,
        observed_at_ms=_event_observed_at_ms(event),
        phase=phase,
        in_progress=phase == "start",
        duration_s=round(duration_s, 6),
        backlog_start=backlog_start,
        backlog_end=backlog_end,
        offered_work=offered_work,
        drained_work=drained_work,
        drain_rate_per_s=round(drain_rate, 6),
        offered_rate_per_s=round(offered_rate, 6),
        verdict=verdict,
        reason=reason,
        bulk_import_suppressed=bulk_import_suppressed,
    )


def project_backlog_windows(
    events: Sequence[Mapping[str, object]],
    *,
    bulk_import_suppressed: bool | None = None,
) -> tuple[BacklogWindow, ...]:
    """Return bounded windows from the existing event stream, newest first."""
    suppressed = bulk_import_is_open(events) if bulk_import_suppressed is None else bulk_import_suppressed
    completed: list[tuple[tuple[int, int, int], BacklogWindow]] = []
    active: dict[str, tuple[tuple[int, int, int], BacklogWindow]] = {}
    for index, event in sorted(enumerate(events), key=lambda item: _event_order_key(*item)):
        window = backlog_window_from_event(event, bulk_import_suppressed=suppressed)
        if window is None:
            continue
        key = window.operation_id or f"event:{_int(event, 'id')}:{index}"
        order_key = _event_order_key(index, event)
        if window.in_progress:
            active[key] = (order_key, window)
        else:
            active.pop(key, None)
            completed.append((order_key, window))
    windows = completed + list(active.values())
    return tuple(window for _key, window in sorted(windows, reverse=True)[:20])


def reduce_slo_samples(
    samples: Sequence[ArchiveSloSample],
    *,
    target_rate: float | None = None,
) -> SloReduction:
    """Compute level, quantiles, slope, ETA, and optional burn rate."""
    if not samples:
        return SloReduction(label="unknown")
    ordered = sorted(samples, key=lambda sample: sample.observed_at_ms)
    values = [float(sample.value) for sample in ordered]
    label = ordered[-1].label
    first, latest = ordered[0], ordered[-1]
    delta_s = (latest.observed_at_ms - first.observed_at_ms) / 1000.0
    slope = (latest.value - first.value) / delta_s if len(ordered) > 1 and delta_s > 0 else None
    eta_s = None
    if label == SloSampleLabel.BACKLOG.value and slope is not None and slope < 0 and latest.value > 0:
        eta_s = latest.value / -slope
    burn_rate = None
    if target_rate is not None and target_rate > 0:
        burn_rate = latest.value / target_rate
    return SloReduction(
        label=label,
        sample_count=len(ordered),
        level=round(latest.value, 6),
        p50=round(percentile(values, 0.50), 6),
        p95=round(percentile(values, 0.95), 6),
        slope_per_s=round(slope, 6) if slope is not None else None,
        eta_s=round(eta_s, 3) if eta_s is not None and math.isfinite(eta_s) else None,
        burn_rate=round(burn_rate, 6) if burn_rate is not None else None,
        confident=len(ordered) >= 2,
    )


def record_backlog_window_sample(ops_db: Path, window: BacklogWindow) -> tuple[str, ...]:
    """Persist numeric samples from one derived window in disposable ops state."""
    if window.observed_at_ms is None:
        return ()
    observed_at_ms = window.observed_at_ms
    values = (
        (SloSampleLabel.BACKLOG, float(window.backlog_end)),
        (SloSampleLabel.OFFERED_WORK, float(window.offered_work)),
        (SloSampleLabel.DRAIN_RATE, float(window.drain_rate_per_s)),
    )
    return tuple(
        record_slo_sample(
            ops_db,
            label=label,
            value=value,
            observed_at_ms=observed_at_ms,
            window_start_ms=observed_at_ms - round(window.duration_s * 1000),
            window_end_ms=observed_at_ms,
            metadata={"verdict": window.verdict.value},
        )
        for label, value in (
            (*values, (SloSampleLabel.INGEST_LATENCY, float(window.duration_s)))
            if not window.bulk_import_suppressed
            else values
        )
    )


def load_slo_reductions(ops_db: Path, *, since_ms: int | None = None) -> dict[str, SloReduction]:
    """Read and reduce the optional sample history; cold start stays explicit."""
    grouped: dict[str, list[ArchiveSloSample]] = {}
    for sample in list_slo_samples(ops_db, since_ms=since_ms, limit=2000):
        grouped.setdefault(sample.label, []).append(sample)
    return {label: reduce_slo_samples(samples) for label, samples in sorted(grouped.items())}


def slo_status_info(
    *,
    cursor_lag: CursorLagSummary,
    events: Sequence[Mapping[str, object]] | None = None,
) -> IngestSloStatus:
    """Build status from existing event and cursor-lag sources only."""
    resolved_events = list(events) if events is not None else list(query_daemon_events(limit=50))
    suppressed = bulk_import_is_open(resolved_events)
    windows = project_backlog_windows(resolved_events, bulk_import_suppressed=suppressed)
    latest = windows[0] if windows else None
    if latest is None:
        return IngestSloStatus(
            cursor_lag_stuck_file_count=cursor_lag.stuck_file_count,
            cursor_lag_degraded_file_count=cursor_lag.degraded_file_count,
        )
    return IngestSloStatus(
        available=True,
        verdict=latest.verdict,
        reason=latest.reason,
        backlog=latest.backlog_end,
        offered_work=latest.offered_work,
        drained_work=latest.drained_work,
        drain_rate_per_s=latest.drain_rate_per_s,
        offered_rate_per_s=latest.offered_rate_per_s,
        cursor_lag_stuck_file_count=cursor_lag.stuck_file_count,
        cursor_lag_degraded_file_count=cursor_lag.degraded_file_count,
        bulk_import_suppressed=latest.bulk_import_suppressed,
        latest_window=latest,
    )


__all__ = [
    "BacklogWindow",
    "IngestSloStatus",
    "SloReduction",
    "SloVerdict",
    "backlog_window_from_event",
    "bulk_import_is_open",
    "load_slo_reductions",
    "project_backlog_windows",
    "record_backlog_window_sample",
    "reduce_slo_samples",
    "slo_status_info",
]
