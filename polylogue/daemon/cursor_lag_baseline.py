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

- The archive ops ``cursor_lag_samples`` table records one row per family
  per health-loop tick when the family has at least one stuck cursor. Each
  sample carries the family, representative source path, max lag,
  ``stuck_file_count``, ``p50_lag_ms``, and ``p95_lag_ms`` for that moment.
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

The samples table is daemon-runtime state. In archive it lives in the
disposable ops tier, outside the canonical archive and user tiers. It is
missing on a fresh archive without consequence — :func:`load_family_baseline`
returns an unconfident baseline that the anomaly check refuses to alert on.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polylogue.daemon.cursor_lag_status import CursorLagItem, CursorLagSummary
from polylogue.sources.live._lag_sample_ddl import _LAG_SAMPLE_DDL, _LAG_SAMPLE_INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    record_cursor_lag_sample as record_archive_cursor_lag_sample,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
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
    """Persist one cursor-lag sample row per family with stuck files.

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
    resolved_now = now or datetime.now(UTC)
    observed_at_ms = _epoch_ms(resolved_now)
    per_family_lags = _bucket_stuck_lags_by_family(summary.stuck)
    if not any(family_summary.stuck_file_count > 0 for family_summary in summary.family_summaries):
        return 0
    return _record_archive_cursor_lag_samples(dbf.with_name("ops.db"), summary, per_family_lags, observed_at_ms)


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
    return _gc_archive_cursor_lag_samples(dbf.with_name("ops.db"), retention_days=retention_days, now=now)


def load_family_baseline(
    dbf: Path,
    family: str,
    *,
    window_days: int,
    min_samples: int,
    now: datetime | None = None,
) -> FamilyBaseline:
    """Compute the rolling-window baseline for one family.

    Reads archive ops ``cursor_lag_samples`` rows for ``family`` within the
    trailing ``window_days`` and returns ``(median, p95, count, confident)``.
    ``confident`` is true iff ``sample_count >= min_samples`` — the anomaly
    check refuses to alert off an unconfident baseline so a fresh archive or a
    recently-restarted daemon does not produce false positives during warm-up.
    """
    moment = now or datetime.now(UTC)
    window_start = moment - timedelta(days=max(1, window_days))
    baseline = _load_archive_family_baseline(
        dbf.with_name("ops.db"),
        family,
        window_start=window_start,
        min_samples=min_samples,
    )
    if baseline is not None:
        return baseline
    return FamilyBaseline(
        family=family,
        sample_count=0,
        rolling_median_lag_s=0.0,
        rolling_p95_lag_s=0.0,
        window_started_at=window_start.isoformat(),
        confident=False,
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


def _bucket_stuck_lags_by_family(stuck: list[CursorLagItem]) -> dict[str, list[CursorLagItem]]:
    """Bucket the ``stuck`` projection items by family for sample p50/p95."""
    out: dict[str, list[CursorLagItem]] = {}
    for item in stuck:
        out.setdefault(item.family, []).append(item)
    for items in out.values():
        items.sort(key=lambda item: item.lag_s)
    return out


def _record_archive_cursor_lag_samples(
    ops_db: Path,
    summary: CursorLagSummary,
    per_family_items: dict[str, list[CursorLagItem]],
    observed_at_ms: int,
) -> int:
    rows = []
    for family_summary in summary.family_summaries:
        if family_summary.stuck_file_count == 0:
            continue
        family_items = per_family_items.get(family_summary.family, [])
        lags = [item.lag_s for item in family_items]
        if lags:
            p50 = _percentile(lags, 0.5)
            p95 = _percentile(lags, 0.95)
            representative = max(family_items, key=lambda item: item.lag_s).source_path
        else:
            p50 = p95 = family_summary.max_lag_s
            representative = None
        rows.append(
            (family_summary.family, representative, family_summary.max_lag_s, family_summary.stuck_file_count, p50, p95)
        )
    if not rows:
        return 0
    ops_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(open_connection(ops_db, timeout=0.1)) as conn:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            for family, source_path, max_lag_s, stuck_file_count, p50_s, p95_s in rows:
                record_archive_cursor_lag_sample(
                    conn,
                    family=family,
                    source_path=source_path,
                    lag_ms=_seconds_to_ms(max_lag_s),
                    stuck_file_count=stuck_file_count,
                    p50_lag_ms=_seconds_to_ms(p50_s),
                    p95_lag_ms=_seconds_to_ms(p95_s),
                    severity="warning",
                    sampled_at_ms=observed_at_ms,
                )
    except sqlite3.OperationalError as exc:
        if _database_is_locked(exc):
            return 0
        raise
    return len(rows)


def _gc_archive_cursor_lag_samples(
    ops_db: Path,
    *,
    retention_days: int,
    now: datetime | None,
) -> int:
    if not ops_db.exists():
        return 0
    cutoff_ms = _epoch_ms((now or datetime.now(UTC)) - timedelta(days=retention_days))
    try:
        with closing(open_connection(ops_db, timeout=0.1)) as conn:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            cur = conn.execute("DELETE FROM cursor_lag_samples WHERE sampled_at_ms < ?", (cutoff_ms,))
            conn.commit()
            return cur.rowcount or 0
    except sqlite3.OperationalError as exc:
        if _database_is_locked(exc):
            return 0
        raise


def _load_archive_family_baseline(
    ops_db: Path,
    family: str,
    *,
    window_start: datetime,
    min_samples: int,
) -> FamilyBaseline | None:
    if not ops_db.exists():
        return None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cursor_lag_samples'"
            ).fetchone()
            if has_table is None:
                return None
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(cursor_lag_samples)")}
            if "family" not in columns:
                return None
            rows = conn.execute(
                """
                SELECT lag_ms
                FROM cursor_lag_samples
                WHERE family = ? AND sampled_at_ms >= ?
                """,
                (family, _epoch_ms(window_start)),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    lags = sorted(float(row[0]) / 1000.0 for row in rows)
    count = len(lags)
    return FamilyBaseline(
        family=family,
        sample_count=count,
        rolling_median_lag_s=round(_percentile(lags, 0.5), 3) if lags else 0.0,
        rolling_p95_lag_s=round(_percentile(lags, 0.95), 3) if lags else 0.0,
        window_started_at=window_start.isoformat(),
        confident=count >= max(1, min_samples),
    )


def _seconds_to_ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _epoch_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


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
