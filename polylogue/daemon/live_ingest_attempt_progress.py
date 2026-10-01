"""Slow-vs-stuck progress classification for live-ingest attempts (#1246).

The daemon's existing live-ingest attempt status carries a binary ``stale``
flag — true when a ``running`` attempt has not updated ``updated_at`` for
longer than ``STUCK_AFTER_S``. That single bit conflates two distinct
operator-visible conditions:

* **stuck** — the attempt has stopped reporting progress at all and is
  almost certainly wedged or crashed. Operators want to intervene
  (inspect the cgroup, kill the worker, replay the source).
* **slow but progressing** — the attempt is reporting fresh
  ``updated_at`` ticks (so the worker is alive) but it is taking
  meaningfully longer than typical attempts. Operators do not need to
  intervene; the daemon is making forward progress on a large/expensive
  payload.

This module exposes a typed ``AttemptProgress`` classification plus the
substrate-layer helpers that compute it. ``polylogue/daemon/status.py``
adapts the heuristic into ``LiveIngestAttemptState`` (per-attempt) and
``LiveIngestAttemptSummary`` (running-count rollups) so CLI, JSON, and
MCP status surfaces converge on the same vocabulary.

The slow threshold is derived from completed attempts' ``total_time_s``
on the same archive — there is no machine-specific budget baked in. When
the archive has fewer than ``SLOW_MIN_SAMPLES`` completed attempts the
heuristic returns ``None`` for the threshold and never classifies a
running attempt as ``slow``; it remains either ``healthy`` or ``stuck``.
"""

from __future__ import annotations

import sqlite3
from typing import Literal

from polylogue.core.stats import percentile

# Public typed vocabulary -----------------------------------------------------

ProgressClassification = Literal["healthy", "slow", "stuck"]
"""Outcome of slow-vs-stuck classification for a live-ingest attempt."""

# Tunables --------------------------------------------------------------------

STUCK_AFTER_S: float = 180.0
"""A running attempt with no ``updated_at`` progress for at least this many
seconds is classified as ``stuck``."""

SLOW_MIN_SAMPLES: int = 5
"""Minimum number of completed attempts required before the p95 slow
threshold is considered representative. Below this, no attempt is
classified ``slow``."""

SLOW_P95_QUANTILE: float = 0.95
"""Quantile of completed-attempt durations used as the slow cutoff."""


def classify_attempt_progress(
    *,
    status: str,
    updated_age_s: float | None,
    total_time_s: float,
    slow_threshold_s: float | None,
    stuck_after_s: float = STUCK_AFTER_S,
) -> ProgressClassification:
    """Return the slow/stuck classification for one live-ingest attempt.

    Only ``status == "running"`` attempts can be ``slow`` or ``stuck``.
    Completed/failed/cancelled attempts are always ``healthy`` (the
    classification is meaningless once the attempt has terminated).

    A missing ``updated_age_s`` is treated as "we cannot tell whether
    progress is happening", and we leave the classification at
    ``healthy`` rather than guess; the existing typed null-row handling
    already drops such attempts out of the stale rollup.
    """

    if status != "running":
        return "healthy"
    if updated_age_s is not None and updated_age_s >= stuck_after_s:
        return "stuck"
    if slow_threshold_s is not None and slow_threshold_s > 0.0 and total_time_s >= slow_threshold_s:
        return "slow"
    return "healthy"


def completed_total_time_samples(conn: sqlite3.Connection) -> list[float]:
    """Return per-attempt ``total_time_s`` for completed attempts.

    The substrate query reads only completed attempts (``status =
    'completed'``) with a strictly positive ``total_time_s``. We
    deliberately ignore failed/cancelled rows because their durations
    reflect early-abort paths, not the cost of a successful attempt; a
    p95 cutoff biased downward by aborts would over-classify successful
    running attempts as ``slow``.

    Returns an empty list when the table is missing or holds no
    eligible rows.
    """

    rows = conn.execute(
        "SELECT started_at_ms, finished_at_ms FROM ingest_attempts "
        "WHERE status = 'completed' AND finished_at_ms IS NOT NULL"
    ).fetchall()
    return [(int(row[1]) - int(row[0])) / 1000.0 for row in rows if int(row[1]) > int(row[0])]


def compute_slow_threshold_s(
    conn: sqlite3.Connection,
    *,
    min_samples: int = SLOW_MIN_SAMPLES,
    quantile: float = SLOW_P95_QUANTILE,
) -> float | None:
    """Return the p95 of completed attempt durations, or ``None``.

    Returns ``None`` when there are fewer than ``min_samples`` completed
    attempts so the caller knows the slow heuristic is disabled for this
    archive.
    """

    samples = completed_total_time_samples(conn)
    if len(samples) < min_samples:
        return None
    samples.sort()
    return percentile(samples, quantile)


__all__ = [
    "ProgressClassification",
    "STUCK_AFTER_S",
    "SLOW_MIN_SAMPLES",
    "SLOW_P95_QUANTILE",
    "classify_attempt_progress",
    "completed_total_time_samples",
    "compute_slow_threshold_s",
]
