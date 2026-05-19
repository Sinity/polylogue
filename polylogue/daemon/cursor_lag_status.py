"""Status projection for per-source-family cursor lag (#1232).

The daemon's ``live_cursor`` table records, for each source file, how far the
ingest pipeline has advanced (``byte_offset``) versus how much data the cursor
last observed (``byte_size``), the wall-clock timestamp of the most recent
cursor update (``updated_at``), and any current backoff/quarantine state.

A cursor is **stuck** when there is known unprocessed work (``byte_offset <
byte_size`` or ``failure_count > 0``) and the cursor has not advanced for
longer than the configured threshold. A cursor is **idle** when the offset has
caught up to the observed size and no failure is pending — there is simply
nothing new to ingest. The two are operationally very different: a stuck
``claude-code-session`` represents an actual ingest lag the operator likely
wants paged on, whereas an idle ``chatgpt-export`` that has not been touched
all week is the normal steady state.

This module is the pure read-side projection. It is consumed by:

- :mod:`polylogue.daemon.cursor_lag_alert` for SLO evaluation and alert
  emission.
- :mod:`polylogue.daemon.status` to surface per-family lag on the
  ``polylogue status`` output and the ``/health`` envelope.

The projection only reads ``live_cursor`` columns that are durably populated
by :mod:`polylogue.sources.live.cursor`; it never stats source files on disk,
so it stays cheap and side-effect free.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class CursorLagFamilySummary(BaseModel):
    """Per-source-family cursor-lag rollup."""

    family: str
    tracked_file_count: int = 0
    stuck_file_count: int = 0
    idle_file_count: int = 0
    max_lag_s: float = 0.0
    """Maximum ``now - updated_at`` across stuck files in this family (0 if none)."""


class CursorLagItem(BaseModel):
    """One cursor that the projection classified as stuck."""

    family: str
    source_path: str
    byte_offset: int = 0
    byte_size: int = 0
    failure_count: int = 0
    updated_at: str
    lag_s: float = 0.0


class CursorLagSummary(BaseModel):
    """Aggregated cursor-lag projection."""

    tracked_file_count: int = 0
    stuck_file_count: int = 0
    idle_file_count: int = 0
    max_lag_s: float = 0.0
    family_summaries: list[CursorLagFamilySummary] = Field(default_factory=list)
    stuck: list[CursorLagItem] = Field(default_factory=list)


_STUCK_SAMPLE_LIMIT = 10
"""Bound the per-summary list of stuck items so the projection stays small."""


def cursor_lag_summary_info(dbf: Path, *, now: datetime | None = None) -> CursorLagSummary:
    """Return per-family cursor-lag rollups read from ``live_cursor``."""
    if not dbf.exists():
        return CursorLagSummary()
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_cursor'"
            ).fetchone()
            if has_table is None:
                return CursorLagSummary()
            rows = conn.execute(
                """
                SELECT source_path, byte_size, byte_offset, failure_count,
                       excluded, updated_at
                FROM live_cursor
                """
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return CursorLagSummary()

    return _project_rows(rows, now=now or datetime.now(UTC))


def _project_rows(
    rows: list[sqlite3.Row | tuple[object, ...]],
    *,
    now: datetime,
) -> CursorLagSummary:
    """Pure helper: bucket cursor rows into a typed summary.

    Split out so tests can drive the projection without going through the
    SQLite read path.
    """
    # Late import to avoid a hard cycle: convergence_debt_alert imports
    # from polylogue.config which transitively pulls daemon modules.
    from polylogue.daemon.convergence_debt_alert import source_family_for_path

    per_family: dict[str, _FamilyAccumulator] = {}
    stuck_items: list[CursorLagItem] = []
    tracked_total = 0
    stuck_total = 0
    idle_total = 0
    max_lag_total = 0.0

    for row in rows:
        source_path = _required_str(row[0])
        byte_size = _row_int(row[1])
        byte_offset = _row_int(row[2])
        failure_count = _row_int(row[3])
        excluded = bool(row[4]) if row[4] is not None else False
        updated_at = _required_str(row[5])
        tracked_total += 1
        family = source_family_for_path(source_path)
        acc = per_family.setdefault(family, _FamilyAccumulator(family=family))
        acc.tracked += 1
        if excluded:
            # Quarantined cursors are not "stuck" in the SLO sense — they
            # have been explicitly removed from the ingest loop and live
            # under the raw-failures alert surface instead.
            acc.idle += 1
            idle_total += 1
            continue
        is_stuck = (byte_offset < byte_size) or failure_count > 0
        lag_s = _age_seconds(updated_at, now=now)
        if is_stuck:
            acc.stuck += 1
            stuck_total += 1
            if lag_s > acc.max_lag_s:
                acc.max_lag_s = lag_s
            if lag_s > max_lag_total:
                max_lag_total = lag_s
            stuck_items.append(
                CursorLagItem(
                    family=family,
                    source_path=source_path,
                    byte_offset=byte_offset,
                    byte_size=byte_size,
                    failure_count=failure_count,
                    updated_at=updated_at,
                    lag_s=round(lag_s, 3),
                )
            )
        else:
            acc.idle += 1
            idle_total += 1

    family_summaries = [
        CursorLagFamilySummary(
            family=acc.family,
            tracked_file_count=acc.tracked,
            stuck_file_count=acc.stuck,
            idle_file_count=acc.idle,
            max_lag_s=round(acc.max_lag_s, 3),
        )
        for acc in sorted(per_family.values(), key=lambda a: (-a.max_lag_s, -a.stuck, a.family))
    ]
    # Surface the worst offenders first; bound list size for status payloads.
    stuck_items.sort(key=lambda item: item.lag_s, reverse=True)
    stuck_items = stuck_items[:_STUCK_SAMPLE_LIMIT]

    return CursorLagSummary(
        tracked_file_count=tracked_total,
        stuck_file_count=stuck_total,
        idle_file_count=idle_total,
        max_lag_s=round(max_lag_total, 3),
        family_summaries=family_summaries,
        stuck=stuck_items,
    )


class _FamilyAccumulator:
    __slots__ = ("family", "tracked", "stuck", "idle", "max_lag_s")

    def __init__(self, *, family: str) -> None:
        self.family = family
        self.tracked = 0
        self.stuck = 0
        self.idle = 0
        self.max_lag_s = 0.0


def _required_str(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _row_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _age_seconds(iso_value: str, *, now: datetime) -> float:
    try:
        observed = datetime.fromisoformat(iso_value)
    except ValueError:
        return 0.0
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return max(0.0, (now - observed.astimezone(UTC)).total_seconds())


__all__ = [
    "CursorLagFamilySummary",
    "CursorLagItem",
    "CursorLagSummary",
    "cursor_lag_summary_info",
]
