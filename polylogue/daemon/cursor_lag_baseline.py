"""Per-source-family cursor-lag rolling baseline (#1349, ambitious-expansion of #1232).

The static SLO ladder in :mod:`polylogue.daemon.cursor_lag_alert` thresholds
on absolute lag seconds. That alone forces operators with mixed archives to
pick a single global default that is either too loud for long-tail sources
(``chatgpt-export``: hours of normal lag) or too quiet for high-traffic ones
(``claude-code-session``: tens of seconds of normal lag). The auto-calibration
layer adds a second, softer signal — *"this family's lag is now N× its
rolling baseline"* — that fires alerts at a lower threshold when a family's
normal lag is tight, and stays silent when a family's normal lag is naturally
high even if absolute lag would cross the global default.

This module owns the sample-history substrate:

- The ``live_cursor_lag_sample`` table (created by
  :func:`ensure_lag_sample_table` and by :class:`~polylogue.sources.live.cursor.CursorStore`
  on daemon startup) records one row per family per health-loop tick when
  the family has at least one stuck cursor. Each sample carries
  ``max_lag_s``, ``stuck_file_count``, ``p50_lag_s``, ``p95_lag_s`` for that
  moment.
- :func:`record_cursor_lag_sample` is called by the periodic health loop
  after the static check evaluates. It is a pure substrate write — it does
  not touch the alert path.
- :func:`gc_cursor_lag_samples` is called by the same periodic tick to bound
  the table. The retention window defaults to ``max(retention_days,
  baseline_window_days * 2)`` so a long-paused daemon does not blow its own
  baseline away on first restart.
- :func:`load_family_baseline` reads the rolling window for one family and
  returns a typed :class:`FamilyBaseline` snapshot. The percentile is
  computed via sorted-SELECT + index math — no external dep, microseconds
  at the ~20K-row scale this table is bounded to.

The samples table is daemon-runtime state, not part of the canonical
``SCHEMA_VERSION`` lifecycle (same pattern as ``live_cursor`` and
``live_convergence_debt``; see ``polylogue/sources/live/cursor.py``). It is
created with ``CREATE TABLE IF NOT EXISTS`` and is missing on a fresh
archive without consequence — :func:`load_family_baseline` returns an
unconfident baseline that the anomaly check refuses to alert on.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polylogue.daemon.cursor_lag_status import CursorLagItem, CursorLagSummary
from polylogue.sources.live._lag_sample_ddl import _LAG_SAMPLE_DDL, _LAG_SAMPLE_INDEX_DDL
from polylogue.storage.sqlite.connection_profile import open_connection, open_readonly_connection


def ensure_lag_sample_table(conn: sqlite3.Connection) -> None:
    """Create the ``live_cursor_lag_sample`` table + index if missing.

    Idempotent. Called from :class:`~polylogue.sources.live.cursor.CursorStore`
    on daemon startup so the table is present before the first health loop
    tick, and from the write/read helpers here so tests and tools can also
    materialize it on demand.
    """
    conn.execute(_LAG_SAMPLE_DDL)
    conn.execute(_LAG_SAMPLE_INDEX_DDL)


@dataclass(frozen=True, slots=True)
class FamilyBaseline:
    """Rolling-window lag statistics for one source family.

    ``confident=False`` means the rolling window has fewer than the
    configured minimum number of samples. The anomaly check refuses to
    raise alerts off an unconfident baseline — the post-restart warm-up
    period is intentionally silent rather than wrong.
    """

    family: str
    sample_count: int
    rolling_median_lag_s: float
    rolling_p95_lag_s: float
    window_started_at: str
    confident: bool


def record_cursor_lag_sample(
    dbf: Path,
    summary: CursorLagSummary,
    *,
    now: datetime | None = None,
) -> int:
    """Persist one ``live_cursor_lag_sample`` row per family with stuck files.

    Pure substrate write — does not touch the alert path. Returns the number
    of rows inserted (= number of families with at least one stuck cursor).
    Families with zero stuck files do not get a sample row: only "interesting"
    moments accrue history so the baseline reflects the family's stuck-lag
    distribution, not its long-quiet baseline of zero.

    Designed to be called by the periodic health loop after the static
    cursor-lag check evaluates, so the sample reflects the same projection
    the operator just saw.
    """
    if summary.stuck_file_count == 0:
        return 0
    observed_at = (now or datetime.now(UTC)).isoformat()
    per_family_lags = _bucket_stuck_lags_by_family(summary.stuck)
    rows: list[tuple[str, str, float, int, float, float]] = []
    for family_summary in summary.family_summaries:
        if family_summary.stuck_file_count == 0:
            continue
        lags = per_family_lags.get(family_summary.family, [])
        if not lags:
            # The projection's ``stuck`` list is bounded; if the family's
            # stuck items did not make the top-10, fall back to the
            # per-family max from the summary. p50/p95 collapse to the same
            # value but the row still records the moment for baseline math.
            p50 = p95 = family_summary.max_lag_s
        else:
            p50 = _percentile(lags, 0.5)
            p95 = _percentile(lags, 0.95)
        rows.append(
            (
                family_summary.family,
                observed_at,
                family_summary.max_lag_s,
                family_summary.stuck_file_count,
                round(p50, 3),
                round(p95, 3),
            )
        )
    if not rows:
        return 0
    dbf.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(open_connection(dbf, timeout=0.1)) as conn:
            ensure_lag_sample_table(conn)
            conn.executemany(
                """
                INSERT INTO live_cursor_lag_sample (
                    family, observed_at, max_lag_s, stuck_file_count, p50_lag_s, p95_lag_s
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
    except sqlite3.OperationalError as exc:
        if _database_is_locked(exc):
            return 0
        raise
    return len(rows)


def gc_cursor_lag_samples(
    dbf: Path,
    *,
    retention_days: int,
    now: datetime | None = None,
) -> int:
    """Delete samples older than ``retention_days``. Returns rows removed.

    Bounded — at 60s health-loop period × 10 families × 14-day default
    retention the table caps near 200K rows. The GC runs in the same tick
    that writes a new sample so amortized cost is one DELETE per tick.
    """
    if retention_days <= 0:
        return 0
    if not dbf.exists():
        return 0
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=retention_days)).isoformat()
    try:
        with closing(open_connection(dbf, timeout=0.1)) as conn:
            ensure_lag_sample_table(conn)
            cur = conn.execute(
                "DELETE FROM live_cursor_lag_sample WHERE observed_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount or 0
    except sqlite3.OperationalError as exc:
        if _database_is_locked(exc):
            return 0
        raise


def load_family_baseline(
    dbf: Path,
    family: str,
    *,
    window_days: int,
    min_samples: int,
    now: datetime | None = None,
) -> FamilyBaseline:
    """Compute the rolling-window baseline for one family.

    Reads ``live_cursor_lag_sample`` rows for ``family`` within the trailing
    ``window_days``, sorts ``max_lag_s`` in-memory (small window, microseconds),
    and returns ``(median, p95, count, confident)``. ``confident`` is true iff
    ``sample_count >= min_samples`` — the anomaly check refuses to alert off
    an unconfident baseline so a fresh archive or a recently-restarted daemon
    does not produce false positives during warm-up.
    """
    moment = now or datetime.now(UTC)
    window_start = moment - timedelta(days=max(1, window_days))
    if not dbf.exists():
        return FamilyBaseline(
            family=family,
            sample_count=0,
            rolling_median_lag_s=0.0,
            rolling_p95_lag_s=0.0,
            window_started_at=window_start.isoformat(),
            confident=False,
        )
    try:
        conn = open_readonly_connection(dbf)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'live_cursor_lag_sample'"
            ).fetchone()
            if has_table is None:
                return FamilyBaseline(
                    family=family,
                    sample_count=0,
                    rolling_median_lag_s=0.0,
                    rolling_p95_lag_s=0.0,
                    window_started_at=window_start.isoformat(),
                    confident=False,
                )
            rows = conn.execute(
                """
                SELECT max_lag_s
                FROM live_cursor_lag_sample
                WHERE family = ? AND observed_at >= ?
                """,
                (family, window_start.isoformat()),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return FamilyBaseline(
            family=family,
            sample_count=0,
            rolling_median_lag_s=0.0,
            rolling_p95_lag_s=0.0,
            window_started_at=window_start.isoformat(),
            confident=False,
        )

    lags = sorted(float(row[0]) for row in rows)
    count = len(lags)
    return FamilyBaseline(
        family=family,
        sample_count=count,
        rolling_median_lag_s=round(_percentile(lags, 0.5), 3) if lags else 0.0,
        rolling_p95_lag_s=round(_percentile(lags, 0.95), 3) if lags else 0.0,
        window_started_at=window_start.isoformat(),
        confident=count >= max(1, min_samples),
    )


def load_family_baselines(
    dbf: Path,
    families: list[str],
    *,
    window_days: int,
    min_samples: int,
    now: datetime | None = None,
) -> dict[str, FamilyBaseline]:
    """Batched read of :func:`load_family_baseline` for many families."""
    return {
        family: load_family_baseline(dbf, family, window_days=window_days, min_samples=min_samples, now=now)
        for family in families
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bucket_stuck_lags_by_family(stuck: list[CursorLagItem]) -> dict[str, list[float]]:
    """Bucket the ``stuck`` projection items by family for sample p50/p95."""
    out: dict[str, list[float]] = {}
    for item in stuck:
        out.setdefault(item.family, []).append(float(item.lag_s))
    for lags in out.values():
        lags.sort()
    return out


def _database_is_locked(exc: sqlite3.OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile over a pre-sorted list.

    Returns 0.0 for an empty list. ``q`` is in ``[0, 1]``. Matches the
    "linear" interpolation used by numpy.percentile so a baseline computed
    here can be compared against externally-collected metrics without an
    impedance mismatch.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


__all__ = [
    "FamilyBaseline",
    "ensure_lag_sample_table",
    "gc_cursor_lag_samples",
    "load_family_baseline",
    "load_family_baselines",
    "record_cursor_lag_sample",
]
