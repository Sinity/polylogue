"""trajectory-report: a self-contained HTML velocity/trajectory report.

Answers the question neither sibling answers: **what does this project's
momentum look like over calendar time** -- is it accelerating or plateauing,
what does a typical week and day look like, where has effort concentrated
week by week, and does bead churn actually track code churn.

`workspace backlog-calibration` fits duration/discovery *models* (lead-time
percentiles, survival, close-reason classes); `workspace beads-state-report`
is a point-in-time *census* of backlog shape and graph health. This report is
the missing third view: a real time-axis account of the whole repository's
development rhythm, built from three independent evidence streams:

  1. the full git commit history (author timestamps + per-commit file lists),
  2. PR merge events, recovered offline from squash-merge ``(#N)`` subjects
     on the first-parent line of the default branch -- no network, no `gh`,
  3. the bead corpus (``.beads/issues.jsonl`` created_at/closed_at).

Everything mechanical is computed here; authored judgement is confined to the
``PROSE`` mapping at the top of the module, exactly as in beads-state-report,
so a reader can audit which sentences are interpretation and which text is
derived from data.

Usage: devtools workspace trajectory-report [--out PATH] [--beads FILE]
                                            [--fresh] [--json]
``--fresh`` runs ``bd export`` to a temporary file instead of reading the
committed ``.beads/issues.jsonl`` (bd mutations do not immediately re-export).
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root

# --------------------------------------------------------------------------
# AUTHORED PROSE. Hand-written judgement only; every number is interpolated
# from computed facts at render time.
# --------------------------------------------------------------------------
PROSE: dict[str, str] = {
    "lead": """
      This is the development trajectory of the polylogue repository itself
      &mdash; not the archived conversations it stores &mdash; read from three
      independent evidence streams: {commits_total} commits over
      {span_days} days, {prs_total} squash-merged PRs recovered from
      first-parent commit subjects, and {beads_total} beads. The headline is
      that this project is built in <b>campaigns, not a steady drip</b>: the
      busiest {top_decile_pct} of active days carries
      <b>{top_decile_share}</b> of all commits, and the merge cadence within a
      campaign day reaches {peak_prs} PRs in a single day
      ({peak_prs_day}). Read the <a href="#momentum">momentum</a> section for
      whether the current pace is rising or falling, and
      <a href="#rhythm">rhythm</a> for what a typical day actually looks like.
    """,
    "trajectory": """
      Weekly volume across all three streams on one real time axis. The
      pre-{active_start} tail is thin by construction &mdash; the repository's
      early history is sparse &mdash; so the story starts where the lines
      lift off. Two things are visible here that no aggregate number shows:
      the week-to-week volatility (adjacent weeks differ by multiples, not
      percentages), and that bead lines exist only from {bead_first_day}
      onward &mdash; the tracker was adopted mid-project, so bead volume
      before that date is a structural zero, not a measurement of calm.
    """,
    "momentum": """
      Rolling windows instead of a single average, because a single average
      over a bursty series is a lie of smoothness. The 7-day line is the
      texture; the 28-day line is the trend. Comparing the last
      {momentum_window}d of merged PRs ({recent_prs}) against the
      {momentum_window}d before that ({prior_prs}) gives a ratio of
      <b>{momentum_ratio}&times;</b> &mdash; read mechanically as
      <b>{momentum_verdict}</b>. Treat the verdict as a 56-day statement
      only: in a campaign-driven project, one operator decision to run or
      not run a fanout week moves this ratio more than any underlying
      capacity change.
    """,
    "rhythm": """
      The hour&times;weekday heatmap is computed from author timestamps in
      their recorded local offset. The striking fact is how <b>flat</b> the
      clock is: the busiest hour holds {peak_hour_n} commits against a
      mean of {mean_hour_n} per hour-slot &mdash; a round-the-clock profile
      that is the signature of agent-driven development rather than a
      human workday. Day character makes the same point on the day scale:
      of {active_days} active days, {campaign_days} are campaign days
      (&gt;{campaign_threshold} PRs merged) and they account for
      {campaign_share} of all PR merges.
    """,
    "areas": """
      Where the effort landed, week by week, as file-touches per area (a
      churn proxy, not an effort measure &mdash; see
      <a href="#uncertainties">uncertainties</a>). Each row is one area on
      the shared time axis, so vertical alignment is meaningful: a spike that
      appears in <code>tests/</code> and a <code>polylogue/</code> package in
      the same week is one campaign touching both. Sorted by total volume;
      single-hue by design &mdash; identity is carried by the row label, so
      the chart needs no legend and no categorical palette.
    """,
    "coupling": """
      Each point is one day of the bead era (excluding the
      {bead_import_day} bulk import): merged PRs against beads closed.
      Pearson r = <b>{coupling_r}</b> over {coupling_n} days.
      The reading is not causal either way &mdash; campaign days
      simultaneously merge PRs and close the beads those PRs satisfy, so a
      strong correlation mostly confirms that the tracker is kept honest
      <em>during</em> work rather than backfilled later.
    """,
    "uncertainties": """
      Where these numbers are proxies or censored, stated rather than
      smoothed over.
    """,
}

UNCERTAINTIES: tuple[tuple[str, str], ...] = (
    (
        "Commit count under-counts implementation effort",
        "master is squash-merge only: one commit per PR regardless of how many "
        "review-waypoint commits the branch carried. PR merges and commits are "
        "therefore nearly the same series on the first-parent line; neither "
        "measures hours of work.",
    ),
    (
        "Early history is sparse, not calm",
        "the repository's pre-2026-03 history is thin (single-digit commits "
        "per month). Trend statements in this report are made over the active "
        "era only; the early tail is drawn but not interpreted.",
    ),
    (
        "Bead series starts mid-project",
        "the tracker was adopted with a bulk import (the single busiest "
        "created_at day dominates the series); created/closed series before "
        "first bead activity are structural zeros, and the import day is "
        "excluded from coupling and rate statistics.",
    ),
    (
        "File-touches are a churn proxy",
        "the area view counts files changed per commit per area. A mechanical "
        "sweep touching 40 files outweighs a hard 1-file fix. It locates "
        "attention, not difficulty.",
    ),
    (
        "Author time, recorded offset",
        "hour-of-day uses each commit's author timestamp in its recorded "
        "local offset. Cloud-lane commits may carry UTC offsets, flattening "
        "the apparent clock further.",
    ),
    (
        "PR recovery is textual",
        "PR merge events are recovered from the trailing (#N) convention on "
        "first-parent subjects. Direct pushes without the suffix (rare, "
        "policy-violating) would be invisible as PRs while still counted as "
        "commits.",
    ),
)


# --------------------------------------------------------------------------
# Data collection (side-effecting; everything below build_facts is pure)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Commit:
    ts: dt.datetime  # author time, original offset preserved
    subject: str
    files: tuple[str, ...]


_PR_SUFFIX = re.compile(r"\(#(\d+)\)$")
_COMMIT_HEADER = "\x01"


def collect_commits(root: Path) -> list[Commit]:
    """Full history: author timestamp + subject + changed files per commit."""
    out = subprocess.run(
        ["git", "-C", str(root), "log", f"--format={_COMMIT_HEADER}%aI|%s", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits: list[Commit] = []
    for chunk in out.split(_COMMIT_HEADER):
        if not chunk.strip():
            continue
        head, _, body = chunk.partition("\n")
        ts_raw, _, subject = head.partition("|")
        files = tuple(line for line in body.splitlines() if line.strip())
        commits.append(Commit(dt.datetime.fromisoformat(ts_raw), subject, files))
    commits.sort(key=lambda c: c.ts)
    return commits


def collect_first_parent_subjects(root: Path) -> list[tuple[dt.datetime, str]]:
    """First-parent line of HEAD: where squash-merge PR subjects live."""
    out = subprocess.run(
        ["git", "-C", str(root), "log", "--first-parent", "--format=%aI|%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows: list[tuple[dt.datetime, str]] = []
    for line in out.splitlines():
        ts_raw, _, subject = line.partition("|")
        if ts_raw:
            rows.append((dt.datetime.fromisoformat(ts_raw), subject))
    rows.sort(key=lambda r: r[0])
    return rows


def load_beads(path: Path) -> list[dict[str, Any]]:
    beads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("_type", "issue") == "issue":
            beads.append(record)
    return beads


# --------------------------------------------------------------------------
# Pure computation
# --------------------------------------------------------------------------
def pr_merges(first_parent: Sequence[tuple[dt.datetime, str]]) -> list[tuple[dt.datetime, int]]:
    """(timestamp, pr_number) for every first-parent subject ending in (#N)."""
    merges: list[tuple[dt.datetime, int]] = []
    for ts, subject in first_parent:
        match = _PR_SUFFIX.search(subject.strip())
        if match:
            merges.append((ts, int(match.group(1))))
    return merges


def week_key(day: dt.date) -> str:
    """ISO Monday of the week containing `day`."""
    return (day - dt.timedelta(days=day.weekday())).isoformat()


def area_of(path: str) -> str:
    """Map a repo file path to a coarse effort area."""
    parts = path.split("/")
    top = parts[0]
    if top == "polylogue":
        return f"polylogue/{parts[1]}" if len(parts) > 2 else "polylogue/(root)"
    if top == ".beads":
        return "beads-sync"
    if top in {"docs", "tests", "devtools", "browser-extension", "schemas", "nix"}:
        return top
    return top if len(parts) > 1 else "(repo root)"


def rolling_mean(days: Sequence[str], counts: dict[str, int], window: int) -> list[float]:
    """Trailing-window mean per day, aligned with `days` (contiguous, sorted)."""
    values = [counts.get(day, 0) for day in days]
    out: list[float] = []
    acc = 0
    for i, value in enumerate(values):
        acc += value
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def gini(values: Sequence[int]) -> float:
    """Gini coefficient of a non-negative distribution (0 = even, 1 = one day)."""
    ordered = sorted(values)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    n = len(ordered)
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def top_decile_share(values: Sequence[int]) -> float:
    """Share of total volume carried by the busiest 10% of entries."""
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    k = max(1, math.ceil(len(ordered) / 10))
    return sum(ordered[:k]) / total


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def classify_days(merge_counts: dict[str, int], all_days: Sequence[str]) -> tuple[dict[str, str], int]:
    """Class per day from the data's own distribution of nonzero merge days.

    quiet: 0 merges; organic: <= median of nonzero days; heavy: <= p90;
    campaign: > p90. Returns (classes, campaign_threshold).
    """
    nonzero = sorted(v for v in (merge_counts.get(d, 0) for d in all_days) if v > 0)
    if not nonzero:
        return (dict.fromkeys(all_days, "quiet"), 0)
    median = nonzero[len(nonzero) // 2]
    p90 = nonzero[int(0.9 * (len(nonzero) - 1))]
    classes: dict[str, str] = {}
    for day in all_days:
        v = merge_counts.get(day, 0)
        if v == 0:
            classes[day] = "quiet"
        elif v <= median:
            classes[day] = "organic"
        elif v <= p90:
            classes[day] = "heavy"
        else:
            classes[day] = "campaign"
    return classes, p90


def _parse_bead_ts(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def contiguous_days(first: dt.date, last: dt.date) -> list[str]:
    return [(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


@dataclass
class Facts:
    """Everything the report renders, computed once from the three streams."""

    commits_total: int = 0
    prs_total: int = 0
    beads_total: int = 0
    span_days: int = 0
    first_day: str = ""
    last_day: str = ""
    active_start: str = ""
    days: list[str] = field(default_factory=list)  # contiguous, full span
    commit_daily: dict[str, int] = field(default_factory=dict)
    merge_daily: dict[str, int] = field(default_factory=dict)
    bead_created_daily: dict[str, int] = field(default_factory=dict)
    bead_closed_daily: dict[str, int] = field(default_factory=dict)
    weeks: list[str] = field(default_factory=list)
    commit_weekly: dict[str, int] = field(default_factory=dict)
    merge_weekly: dict[str, int] = field(default_factory=dict)
    bead_created_weekly: dict[str, int] = field(default_factory=dict)
    bead_closed_weekly: dict[str, int] = field(default_factory=dict)
    area_weekly: dict[str, dict[str, int]] = field(default_factory=dict)  # area -> week -> touches
    rolling7: list[float] = field(default_factory=list)  # aligned with active_days
    rolling28: list[float] = field(default_factory=list)
    active_days_axis: list[str] = field(default_factory=list)
    open_beads_by_day: list[tuple[str, int]] = field(default_factory=list)
    hour_weekday: list[list[int]] = field(default_factory=list)  # [7][24]
    day_classes: dict[str, str] = field(default_factory=dict)
    campaign_threshold: int = 0
    momentum_window: int = 28
    recent_prs: int = 0
    prior_prs: int = 0
    momentum_ratio: float | None = None
    momentum_verdict: str = "insufficient data"
    gini_commits: float = 0.0
    top_decile_share: float = 0.0
    peak_prs: int = 0
    peak_prs_day: str = ""
    bead_first_day: str = ""
    bead_import_day: str = ""
    coupling_r: float | None = None
    coupling_n: int = 0
    coupling_points: list[tuple[int, int]] = field(default_factory=list)


_ACTIVE_ERA_MIN_COMMITS_PER_WEEK = 5


def build_facts(
    commits: Sequence[Commit],
    first_parent: Sequence[tuple[dt.datetime, str]],
    beads: Sequence[dict[str, Any]],
    now: dt.datetime,
) -> Facts:
    facts = Facts()
    if not commits:
        return facts
    facts.commits_total = len(commits)
    facts.beads_total = len(beads)

    merges = pr_merges(first_parent)
    facts.prs_total = len(merges)

    first = commits[0].ts.date()
    last = max(commits[-1].ts.date(), now.date())
    facts.first_day = first.isoformat()
    facts.last_day = last.isoformat()
    facts.span_days = (last - first).days + 1
    facts.days = contiguous_days(first, last)

    for commit in commits:
        day = commit.ts.date().isoformat()
        facts.commit_daily[day] = facts.commit_daily.get(day, 0) + 1
        week = week_key(commit.ts.date())
        facts.commit_weekly[week] = facts.commit_weekly.get(week, 0) + 1
        for path in commit.files:
            area = area_of(path)
            facts.area_weekly.setdefault(area, {})
            facts.area_weekly[area][week] = facts.area_weekly[area].get(week, 0) + 1
    for ts, _number in merges:
        day = ts.date().isoformat()
        facts.merge_daily[day] = facts.merge_daily.get(day, 0) + 1
        week = week_key(ts.date())
        facts.merge_weekly[week] = facts.merge_weekly.get(week, 0) + 1

    bead_created: Counter[str] = Counter()
    bead_closed: Counter[str] = Counter()
    for bead in beads:
        created = _parse_bead_ts(bead.get("created_at"))
        closed = _parse_bead_ts(bead.get("closed_at"))
        if created:
            bead_created[created.date().isoformat()] += 1
        if closed:
            bead_closed[closed.date().isoformat()] += 1
    facts.bead_created_daily = dict(bead_created)
    facts.bead_closed_daily = dict(bead_closed)
    for day, n in bead_created.items():
        week = week_key(dt.date.fromisoformat(day))
        facts.bead_created_weekly[week] = facts.bead_created_weekly.get(week, 0) + n
    for day, n in bead_closed.items():
        week = week_key(dt.date.fromisoformat(day))
        facts.bead_closed_weekly[week] = facts.bead_closed_weekly.get(week, 0) + n
    if bead_created:
        facts.bead_first_day = min(bead_created)
        facts.bead_import_day = max(bead_created, key=lambda d: bead_created[d])

    # Full week axis, Monday keys, contiguous.
    first_week = dt.date.fromisoformat(week_key(first))
    last_week = dt.date.fromisoformat(week_key(last))
    facts.weeks = [
        (first_week + dt.timedelta(weeks=i)).isoformat() for i in range((last_week - first_week).days // 7 + 1)
    ]

    # Active era: first week reaching the threshold, for rolling/momentum axes.
    active_week = next(
        (w for w in facts.weeks if facts.commit_weekly.get(w, 0) >= _ACTIVE_ERA_MIN_COMMITS_PER_WEEK),
        facts.weeks[0],
    )
    facts.active_start = active_week
    facts.active_days_axis = [d for d in facts.days if d >= active_week]
    facts.rolling7 = rolling_mean(facts.active_days_axis, facts.commit_daily, 7)
    facts.rolling28 = rolling_mean(facts.active_days_axis, facts.commit_daily, 28)

    # Open-bead reconstruction over the bead era.
    if bead_created:
        bead_days = [d for d in facts.days if d >= facts.bead_first_day]
        open_n = 0
        series: list[tuple[str, int]] = []
        for day in bead_days:
            open_n += bead_created.get(day, 0) - bead_closed.get(day, 0)
            series.append((day, open_n))
        facts.open_beads_by_day = series

    # Hour x weekday from author-local timestamps.
    facts.hour_weekday = [[0] * 24 for _ in range(7)]
    for commit in commits:
        facts.hour_weekday[commit.ts.weekday()][commit.ts.hour] += 1

    # Day character + burstiness over the active era.
    facts.day_classes, facts.campaign_threshold = classify_days(facts.merge_daily, facts.active_days_axis)
    active_commit_counts = [facts.commit_daily.get(d, 0) for d in facts.active_days_axis]
    facts.gini_commits = gini(active_commit_counts)
    facts.top_decile_share = top_decile_share(active_commit_counts)
    if facts.merge_daily:
        facts.peak_prs_day = max(facts.merge_daily, key=lambda d: facts.merge_daily[d])
        facts.peak_prs = facts.merge_daily[facts.peak_prs_day]

    # Momentum: last 28 full days vs the 28 before.
    window = facts.momentum_window
    if facts.span_days >= 2 * window:
        recent_start = last - dt.timedelta(days=window - 1)
        prior_start = last - dt.timedelta(days=2 * window - 1)
        facts.recent_prs = sum(n for day, n in facts.merge_daily.items() if dt.date.fromisoformat(day) >= recent_start)
        facts.prior_prs = sum(
            n for day, n in facts.merge_daily.items() if prior_start <= dt.date.fromisoformat(day) < recent_start
        )
        if facts.prior_prs > 0:
            facts.momentum_ratio = facts.recent_prs / facts.prior_prs
            if facts.momentum_ratio > 1.15:
                facts.momentum_verdict = "accelerating"
            elif facts.momentum_ratio < 0.85:
                facts.momentum_verdict = "decelerating"
            else:
                facts.momentum_verdict = "steady"

    # Coupling: daily PR merges vs bead closures over the bead era, import day excluded.
    if facts.bead_first_day:
        coupling_days = [d for d in facts.days if d >= facts.bead_first_day and d != facts.bead_import_day]
        xs = [float(facts.merge_daily.get(d, 0)) for d in coupling_days]
        ys = [float(facts.bead_closed_daily.get(d, 0)) for d in coupling_days]
        facts.coupling_n = len(coupling_days)
        facts.coupling_r = pearson(xs, ys)
        facts.coupling_points = [
            (facts.merge_daily.get(d, 0), facts.bead_closed_daily.get(d, 0)) for d in coupling_days
        ]
    return facts


def facts_json(facts: Facts) -> dict[str, Any]:
    return {
        "commits_total": facts.commits_total,
        "prs_total": facts.prs_total,
        "beads_total": facts.beads_total,
        "span_days": facts.span_days,
        "first_day": facts.first_day,
        "last_day": facts.last_day,
        "active_start": facts.active_start,
        "weeks": [
            {
                "week": week,
                "commits": facts.commit_weekly.get(week, 0),
                "pr_merges": facts.merge_weekly.get(week, 0),
                "beads_created": facts.bead_created_weekly.get(week, 0),
                "beads_closed": facts.bead_closed_weekly.get(week, 0),
            }
            for week in facts.weeks
        ],
        "momentum": {
            "window_days": facts.momentum_window,
            "recent_prs": facts.recent_prs,
            "prior_prs": facts.prior_prs,
            "ratio": facts.momentum_ratio,
            "verdict": facts.momentum_verdict,
        },
        "burstiness": {
            "gini_commits_active_era": round(facts.gini_commits, 3),
            "top_decile_day_share": round(facts.top_decile_share, 3),
            "peak_prs_day": facts.peak_prs_day,
            "peak_prs": facts.peak_prs,
            "campaign_threshold_prs": facts.campaign_threshold,
            "day_class_counts": dict(Counter(facts.day_classes.values())),
        },
        "areas_weekly": facts.area_weekly,
        "bead_era": {
            "first_day": facts.bead_first_day,
            "import_day": facts.bead_import_day,
            "open_beads_last": facts.open_beads_by_day[-1][1] if facts.open_beads_by_day else None,
        },
        "coupling": {
            "pearson_r": (round(facts.coupling_r, 3) if facts.coupling_r is not None else None),
            "n_days": facts.coupling_n,
        },
        "hour_weekday": facts.hour_weekday,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


# Reference categorical palette (dataviz skill, validated adjacent-pairs in
# both modes): slot1 blue, slot2 orange, slot3 aqua, slot4 yellow. The hex
# values live in the CSS variables (--s1..--s4, --seq0..--seq6); this constant
# only sizes the sequential ramp used by the heatmap binning.
SEQ_STEPS = 7


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return out_lo
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def multi_line_svg(
    axis: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    aria: str,
    height: float = 240.0,
    y_label: str = "",
) -> str:
    """Multi-series line chart over a shared categorical time axis.

    `series` is (name, values-aligned-with-axis, css-class). Pure SVG,
    viewBox only; per-point hover via oversized hit circles with <title>.
    """
    if not axis:
        return ""
    width = 960.0
    pad_l, pad_r, pad_t, pad_b = 44.0, 10.0, 12.0, 26.0
    peak = max((max(values) for _, values, _ in series if values), default=1.0) or 1.0
    n = len(axis)

    def x_of(i: int) -> float:
        return _scale(float(i), 0.0, float(max(n - 1, 1)), pad_l, width - pad_r)

    def y_of(v: float) -> float:
        return _scale(v, 0.0, peak, height - pad_b, pad_t)

    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{esc(aria)}" class="linechart">']
    # Gridlines: 4 horizontal.
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = y_of(peak * frac)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{pad_l - 6:.0f}" y="{y + 3:.1f}" text-anchor="end">{peak * frac:.0f}</text>'
        )
    # X ticks: ~8 evenly spaced.
    step = max(1, n // 8)
    for i in range(0, n, step):
        x = x_of(i)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 8:.0f}" text-anchor="middle">{esc(axis[i][5:])}</text>'
        )
    # Direct end labels with vertical collision resolution (>=14px apart).
    labels = sorted(
        ((y_of(values[-1]), name, cls) for name, values, cls in series if values),
        key=lambda item: item[0],
    )
    placed: list[tuple[float, str, str]] = []
    for y, name, cls in labels:
        if placed and y - placed[-1][0] < 14.0:
            y = placed[-1][0] + 14.0
        placed.append((y, name, cls))
    for name, values, cls in series:
        pts = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline class="ln {cls}" points="{pts}" fill="none"/>')
        for i, v in enumerate(values):
            if v <= 0:
                continue
            parts.append(
                f'<circle class="hit {cls}" cx="{x_of(i):.1f}" cy="{y_of(v):.1f}" r="7">'
                f"<title>{esc(axis[i])} &middot; {esc(name)}: {v:g}</title></circle>"
            )
    for y, name, cls in placed:
        parts.append(
            f'<text class="lbl {cls}" x="{x_of(n - 1) - 4:.1f}" y="{y - 5:.1f}" text-anchor="end">{esc(name)}</text>'
        )
    if y_label:
        parts.append(f'<text class="tick" x="{pad_l}" y="{pad_t}" text-anchor="start">{esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(matrix: Sequence[Sequence[int]], *, aria: str) -> str:
    """7x24 hour-by-weekday heatmap, sequential blue ramp."""
    weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    cell, gap = 34.0, 2.0
    pad_l, pad_t = 42.0, 20.0
    width = pad_l + 24 * (cell + gap)
    height = pad_t + 7 * (cell * 0.62 + gap)
    peak = max((v for row in matrix for v in row), default=1) or 1
    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{esc(aria)}" class="heatmap">']
    for h in range(0, 24, 3):
        parts.append(
            f'<text class="tick" x="{pad_l + h * (cell + gap) + cell / 2:.1f}" y="12" text-anchor="middle">{h:02d}</text>'
        )
    for wd, row in enumerate(matrix):
        y = pad_t + wd * (cell * 0.62 + gap)
        parts.append(
            f'<text class="tick" x="{pad_l - 8:.0f}" y="{y + cell * 0.42:.1f}" text-anchor="end">{weekdays[wd]}</text>'
        )
        for h, value in enumerate(row):
            step_i = 0 if peak == 0 else min(SEQ_STEPS - 1, int(round((value / peak) * (SEQ_STEPS - 1))))
            parts.append(
                f'<rect class="cell s{step_i}" x="{pad_l + h * (cell + gap):.1f}" y="{y:.1f}" '
                f'width="{cell:.0f}" height="{cell * 0.62:.0f}" rx="3">'
                f"<title>{weekdays[wd]} {h:02d}:00 &middot; {value} commits</title></rect>"
            )
    parts.append("</svg>")
    return "".join(parts)


def sparkline_svg(axis: Sequence[str], counts: dict[str, int], peak: int, *, aria: str) -> str:
    """Single-series area sparkline on the shared weekly axis."""
    if not axis:
        return ""
    width, height = 560.0, 36.0
    n = len(axis)

    def x_of(i: int) -> float:
        return _scale(float(i), 0.0, float(max(n - 1, 1)), 2.0, width - 2.0)

    def y_of(v: int) -> float:
        return _scale(float(v), 0.0, float(peak or 1), height - 3.0, 3.0)

    pts = " ".join(f"{x_of(i):.1f},{y_of(counts.get(week, 0)):.1f}" for i, week in enumerate(axis))
    base = f"{x_of(n - 1):.1f},{height - 3:.1f} {x_of(0):.1f},{height - 3:.1f}"
    hits = "".join(
        f'<circle class="hit" cx="{x_of(i):.1f}" cy="{y_of(counts.get(week, 0)):.1f}" r="6">'
        f"<title>week of {esc(week)}: {counts.get(week, 0)} file-touches</title></circle>"
        for i, week in enumerate(axis)
        if counts.get(week, 0)
    )
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{esc(aria)}" '
        f'class="spark" preserveAspectRatio="none">'
        f'<polygon class="area" points="{pts} {base}"/><polyline class="ln s1" points="{pts}" fill="none"/>{hits}</svg>'
    )


def scatter_svg(points: Sequence[tuple[int, int]], *, aria: str) -> str:
    """PRs-merged (x) vs beads-closed (y), one point per day."""
    if not points:
        return ""
    width, height = 420.0, 300.0
    pad = 34.0
    max_x = max((p[0] for p in points), default=1) or 1
    max_y = max((p[1] for p in points), default=1) or 1
    parts = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{esc(aria)}" class="scatter">']
    parts.append(f'<line class="grid" x1="{pad}" y1="{height - pad}" x2="{width - 8}" y2="{height - pad}"/>')
    parts.append(f'<line class="grid" x1="{pad}" y1="{height - pad}" x2="{pad}" y2="8"/>')
    parts.append(f'<text class="tick" x="{width - 10}" y="{height - pad + 16}" text-anchor="end">PRs merged/day</text>')
    parts.append(f'<text class="tick" x="{pad - 4}" y="14" text-anchor="start">beads closed/day</text>')
    parts.append(f'<text class="tick" x="{pad - 6}" y="{height - pad + 4}" text-anchor="end">0</text>')
    parts.append(f'<text class="tick" x="{width - 10}" y="{height - pad + 4}" text-anchor="end">{max_x}</text>')
    parts.append(f'<text class="tick" x="{pad - 6}" y="26" text-anchor="end">{max_y}</text>')
    counts = Counter(points)
    for (x, y), n in counts.items():
        cx = _scale(float(x), 0.0, float(max_x), pad, width - 12.0)
        cy = _scale(float(y), 0.0, float(max_y), height - pad, 12.0)
        r = 3.5 + 1.5 * math.sqrt(n - 1)
        parts.append(
            f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}">'
            f"<title>{x} PRs, {y} beads closed &middot; {n} day(s)</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def day_strip_html(facts: Facts, days_back: int = 91) -> str:
    """Calendar strip of the most recent days, colored by day class."""
    axis = facts.active_days_axis[-days_back:]
    cells = []
    for day in axis:
        cls = facts.day_classes.get(day, "quiet")
        merges = facts.merge_daily.get(day, 0)
        commits = facts.commit_daily.get(day, 0)
        cells.append(
            f'<span class="dcell dc-{cls}" title="{esc(day)}: {merges} PRs merged, {commits} commits ({cls})"></span>'
        )
    return f'<div class="daystrip">{"".join(cells)}</div>'


CSS = """
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --code-bg:#eef1f4;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --seq0:#cde2fb; --seq1:#9ec5f4; --seq2:#6da7ec; --seq3:#3987e5;
  --seq4:#256abf; --seq5:#184f95; --seq6:#0d366b;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --code-bg:#232a32;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --seq0:#1e2733; --seq1:#0d366b; --seq2:#104281; --seq3:#184f95;
  --seq4:#256abf; --seq5:#3987e5; --seq6:#86b6ef;
}}
:root[data-theme="light"]{color-scheme:light;
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --code-bg:#eef1f4;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --seq0:#cde2fb; --seq1:#9ec5f4; --seq2:#6da7ec; --seq3:#3987e5;
  --seq4:#256abf; --seq5:#184f95; --seq6:#0d366b;
}
:root[data-theme="dark"]{color-scheme:dark;
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --code-bg:#232a32;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --seq0:#1e2733; --seq1:#0d366b; --seq2:#104281; --seq3:#184f95;
  --seq4:#256abf; --seq5:#3987e5; --seq6:#86b6ef;
}
:root{--fs:17px}
*{box-sizing:border-box}
html{font-size:var(--fs)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:1rem/1.62 system-ui,-apple-system,"Segoe UI",sans-serif}
header.page{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);padding:.7rem 1.2rem;
  display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem}
header.page h1{font-size:1.3rem;margin:0}
header.page .spacer{flex:1}
.chip{display:inline-block;padding:.12rem .6rem;border:1px solid var(--line);
  border-radius:99px;font-size:.85rem;color:var(--muted);background:var(--bg)}
button.theme,button.fs{border:1px solid var(--line);background:var(--bg);color:var(--ink);
  border-radius:.4rem;padding:.2rem .6rem;cursor:pointer;font-size:.85rem}
.layout{max-width:76rem;margin:0 auto;padding:1.2rem}
section{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;
  padding:1.1rem 1.3rem;margin-bottom:1.1rem}
h2{font-size:1.28rem;margin:.1rem 0 .7rem;line-height:1.3}
h3{font-size:1.07rem;margin:1.1rem 0 .45rem;line-height:1.35}
p{margin:.5rem 0;max-width:76ch}
a{color:var(--accent)}
code{background:var(--code-bg);border-radius:.3rem;padding:.08rem .35rem;
  font:.9em ui-monospace,SFMono-Regular,Menlo,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));
  gap:.7rem;margin:.4rem 0 .9rem}
.tile{border:1px solid var(--line);border-radius:.55rem;padding:.6rem .85rem;background:var(--bg)}
.tile .n{font-size:1.6rem;font-weight:650;display:block;line-height:1.2}
.tile .l{font-size:.85rem;color:var(--muted)}
.badge{display:inline-block;font-size:.8rem;font-weight:600;letter-spacing:.02em;
  padding:.1rem .55rem;border-radius:99px;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)} .warn{color:var(--warn);background:var(--warn-bg)}
.bad{color:var(--bad);background:var(--bad-bg)} .info{color:var(--info);background:var(--info-bg)}
.ev{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;
  padding:0 .35rem;border-radius:.25rem;vertical-align:.1em;text-transform:uppercase}
.ev-measured{color:var(--ok);background:var(--ok-bg)}
.ev-derived{color:var(--info);background:var(--info-bg)}
.ev-inferred{color:var(--warn);background:var(--warn-bg)}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{border-bottom:1px solid var(--line);text-align:left;padding:.42rem .6rem;vertical-align:middle}
th{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:.88rem;text-align:center;padding:1rem}
svg{max-width:100%;height:auto;display:block}
.chartwrap{overflow-x:auto;margin:.6rem 0}
.linechart .grid,.scatter .grid{stroke:var(--line);stroke-width:1}
.linechart .tick,.heatmap .tick,.scatter .tick{fill:var(--muted);font-size:11px}
.ln{stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ln.s1{stroke:var(--s1)} .ln.s2{stroke:var(--s2)} .ln.s3{stroke:var(--s3)} .ln.s4{stroke:var(--s4)}
.lbl{font-size:11px;font-weight:600}
.lbl.s1{fill:var(--s1)} .lbl.s2{fill:var(--s2)} .lbl.s3{fill:var(--s3)} .lbl.s4{fill:var(--s4)}
.hit{fill:transparent;stroke:none;cursor:crosshair}
.hit:hover{fill:currentColor;fill-opacity:.28}
.heatmap .cell.s0{fill:var(--seq0)} .heatmap .cell.s1{fill:var(--seq1)}
.heatmap .cell.s2{fill:var(--seq2)} .heatmap .cell.s3{fill:var(--seq3)}
.heatmap .cell.s4{fill:var(--seq4)} .heatmap .cell.s5{fill:var(--seq5)}
.heatmap .cell.s6{fill:var(--seq6)}
.spark .area{fill:var(--s1);fill-opacity:.16}
.scatter{max-width:28rem}
.scatter .pt{fill:var(--s1);fill-opacity:.55}
.scatter .pt:hover{fill-opacity:1}
.daystrip{display:flex;flex-wrap:wrap;gap:3px;margin:.6rem 0}
.dcell{width:14px;height:14px;border-radius:3px;background:var(--code-bg)}
.dc-quiet{background:var(--code-bg)}
.dc-organic{background:var(--seq1)}
.dc-heavy{background:var(--seq3)}
.dc-campaign{background:var(--seq5)}
.legend{display:flex;flex-wrap:wrap;gap:1rem;font-size:.85rem;color:var(--muted);margin:.3rem 0}
.legend .sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:.3rem}
.qa summary{cursor:pointer;font-size:.82rem;color:var(--muted)}
details.qa{border:none;background:none;padding:0;margin:.2rem 0}
details.qa pre{background:var(--code-bg);padding:.5rem .7rem;border-radius:.4rem;
  overflow-x:auto;font-size:.82rem;margin:.3rem 0}
"""

JS = """
(function(){
  var root=document.documentElement;
  var saved=localStorage.getItem('traj-theme');
  if(saved)root.setAttribute('data-theme',saved);
  var btn=document.getElementById('themebtn');
  if(btn)btn.addEventListener('click',function(){
    var cur=root.getAttribute('data-theme')||
      (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
    var next=cur==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    localStorage.setItem('traj-theme',next);
  });
  var fs=parseFloat(localStorage.getItem('traj-fs'));
  if(fs)root.style.setProperty('--fs',fs+'px');
  function bump(d){
    var cur=parseFloat(getComputedStyle(root).getPropertyValue('--fs'))||17;
    var next=Math.min(22,Math.max(14,cur+d));
    root.style.setProperty('--fs',next+'px');
    localStorage.setItem('traj-fs',next);
  }
  var minus=document.getElementById('fsminus'),plus=document.getElementById('fsplus');
  if(minus)minus.addEventListener('click',function(){bump(-1)});
  if(plus)plus.addEventListener('click',function(){bump(1)});
})();
"""


def _prose(key: str, values: dict[str, Any]) -> str:
    return f"<p>{PROSE[key].format(**values)}</p>"


def qa_block(label: str, command: str) -> str:
    """Query-attribution: the exact reproducible invocation behind a figure."""
    return f'<details class="qa"><summary>reproduce: {esc(label)}</summary><pre>{esc(command)}</pre></details>'


def render(facts: Facts, beads_source: Path, generated: dt.datetime) -> str:
    values: dict[str, Any] = {
        "commits_total": f"{facts.commits_total:,}",
        "prs_total": f"{facts.prs_total:,}",
        "beads_total": f"{facts.beads_total:,}",
        "span_days": facts.span_days,
        "active_start": facts.active_start,
        "bead_first_day": facts.bead_first_day or "n/a",
        "bead_import_day": facts.bead_import_day or "n/a",
        "top_decile_pct": "10%",
        "top_decile_share": f"{facts.top_decile_share:.0%}",
        "peak_prs": facts.peak_prs,
        "peak_prs_day": facts.peak_prs_day or "n/a",
        "momentum_window": facts.momentum_window,
        "recent_prs": facts.recent_prs,
        "prior_prs": facts.prior_prs,
        "momentum_ratio": (f"{facts.momentum_ratio:.2f}" if facts.momentum_ratio is not None else "n/a"),
        "momentum_verdict": facts.momentum_verdict,
        "coupling_r": (f"{facts.coupling_r:.2f}" if facts.coupling_r is not None else "n/a"),
        "coupling_n": facts.coupling_n,
        "campaign_threshold": facts.campaign_threshold,
        "peak_hour_n": max((v for row in facts.hour_weekday for v in row), default=0),
        "mean_hour_n": (
            f"{sum(v for row in facts.hour_weekday for v in row) / 168:.0f}" if facts.hour_weekday else "0"
        ),
        "active_days": len(facts.active_days_axis),
        "campaign_days": sum(1 for c in facts.day_classes.values() if c == "campaign"),
        "campaign_share": "",
    }
    campaign_prs = sum(facts.merge_daily.get(d, 0) for d, c in facts.day_classes.items() if c == "campaign")
    values["campaign_share"] = f"{campaign_prs / facts.prs_total:.0%}" if facts.prs_total else "n/a"

    # --- charts ---
    active_weeks = [w for w in facts.weeks if w >= facts.active_start]
    trajectory = multi_line_svg(
        active_weeks,
        [
            ("commits", [float(facts.commit_weekly.get(w, 0)) for w in active_weeks], "s1"),
            ("PRs merged", [float(facts.merge_weekly.get(w, 0)) for w in active_weeks], "s2"),
            ("beads created", [float(facts.bead_created_weekly.get(w, 0)) for w in active_weeks], "s3"),
            ("beads closed", [float(facts.bead_closed_weekly.get(w, 0)) for w in active_weeks], "s4"),
        ],
        aria="weekly commits, PR merges, beads created and closed",
        y_label="per week",
    )
    momentum = multi_line_svg(
        facts.active_days_axis,
        [
            ("7d avg commits/day", facts.rolling7, "s1"),
            ("28d avg commits/day", facts.rolling28, "s2"),
        ],
        aria="rolling 7-day and 28-day mean commits per day",
        height=200.0,
    )
    open_beads = ""
    if facts.open_beads_by_day:
        axis = [d for d, _ in facts.open_beads_by_day]
        open_beads = multi_line_svg(
            axis,
            [("open beads", [float(n) for _, n in facts.open_beads_by_day], "s3")],
            aria="open bead count reconstructed per day",
            height=170.0,
        )
    heat = heatmap_svg(facts.hour_weekday, aria="commits by hour of day and weekday")

    top_areas = sorted(
        ((area, sum(weeks.values())) for area, weeks in facts.area_weekly.items()),
        key=lambda item: -item[1],
    )
    top_areas = [(a, t) for a, t in top_areas if a not in {"beads-sync"}][:12]
    area_peak = max((facts.area_weekly[a].get(w, 0) for a, _ in top_areas for w in active_weeks), default=1)
    area_rows = "".join(
        f"<tr><td><code>{esc(area)}</code></td>"
        f'<td class="num">{total:,}</td>'
        f"<td>{sparkline_svg(active_weeks, facts.area_weekly[area], area_peak, aria=f'weekly file-touches in {area}')}</td>"
        f"<td><code>{esc(max(active_weeks, key=lambda w: facts.area_weekly[area].get(w, 0)))}</code></td></tr>"
        for area, total in top_areas
    )
    scatter = scatter_svg(facts.coupling_points, aria="daily PR merges vs beads closed")

    day_counts = Counter(facts.day_classes.values())
    tiles = "".join(
        f'<div class="tile"><span class="n">{esc(n)}</span><span class="l">{esc(label)} '
        f'<span class="ev {ev}">{ev[3:]}</span></span></div>'
        for n, label, ev in (
            (f"{facts.commits_total:,}", "commits, full history", "ev-measured"),
            (f"{facts.prs_total:,}", "PRs merged (squash subjects)", "ev-measured"),
            (f"{facts.beads_total:,}", "beads in corpus", "ev-measured"),
            (f"{facts.top_decile_share:.0%}", "of commits on busiest 10% of days", "ev-derived"),
            (
                f"{facts.momentum_ratio:.2f}x" if facts.momentum_ratio is not None else "n/a",
                f"PR pace, last {facts.momentum_window}d vs prior",
                "ev-derived",
            ),
            (f"{facts.gini_commits:.2f}", "Gini of daily commit volume", "ev-derived"),
        )
    )

    verdict_badge = {"accelerating": "ok", "steady": "info", "decelerating": "warn"}.get(facts.momentum_verdict, "warn")
    uncertainty_rows = "".join(
        f"<tr><td><b>{esc(title)}</b></td><td>{esc(body)}</td></tr>" for title, body in UNCERTAINTIES
    )
    streams_qa = qa_block(
        "all three streams",
        "git log --format='%aI|%s' --name-only\n"
        "git log --first-parent --format='%aI|%s'   # (#N) suffix = squash-merged PR\n"
        "python -m devtools workspace trajectory-report --json",
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>polylogue &mdash; trajectory report</title>
<style>{CSS}</style></head>
<body>
<header class="page"><h1>Trajectory: velocity over calendar time</h1>
<span class="chip">generated {esc(generated.strftime("%Y-%m-%d %H:%M UTC"))}</span>
<span class="chip">{esc(facts.first_day)} &rarr; {esc(facts.last_day)}</span>
<span class="badge {verdict_badge}">{esc(facts.momentum_verdict)}</span>
<span class="spacer"></span>
<button class="fs" id="fsminus">A&minus;</button>
<button class="fs" id="fsplus">A+</button>
<button class="theme" id="themebtn">theme</button></header>
<div class="layout"><main>

<section id="lead"><h2>The shape of the work</h2>
<div class="tiles">{tiles}</div>
{_prose("lead", values)}
{streams_qa}
</section>

<section id="trajectory"><h2>Trajectory &mdash; weekly volume, all streams</h2>
{_prose("trajectory", values)}
<div class="chartwrap">{trajectory}</div>
<div class="legend">
<span><span class="sw" style="background:var(--s1)"></span>commits</span>
<span><span class="sw" style="background:var(--s2)"></span>PRs merged</span>
<span><span class="sw" style="background:var(--s3)"></span>beads created</span>
<span><span class="sw" style="background:var(--s4)"></span>beads closed</span>
</div>
{qa_block("weekly buckets", "python -m devtools workspace trajectory-report --json | jq '.weeks'")}
</section>

<section id="momentum"><h2>Momentum &mdash; rolling rates</h2>
{_prose("momentum", values)}
<div class="chartwrap">{momentum}</div>
<h3>Open beads over time <span class="ev ev-derived">derived</span></h3>
<p>Reconstructed by cumulating created&minus;closed per day from the corpus
timestamps; the level is exact for the current export, and the curve's slope
is the discovery-vs-drain balance backlog-calibration reports as a single
ratio.</p>
<div class="chartwrap">{open_beads}</div>
{qa_block("momentum window", "python -m devtools workspace trajectory-report --json | jq '.momentum, .bead_era'")}
</section>

<section id="rhythm"><h2>Rhythm &mdash; the shape of a day</h2>
{_prose("rhythm", values)}
<div class="chartwrap">{heat}</div>
<h3>Day character, last 13 weeks</h3>
<p>Class thresholds are data-derived from the distribution of nonzero
merge days: quiet = 0 PRs, organic &le; median, heavy &le; p90, campaign
&gt; p90 (&gt;{facts.campaign_threshold} PRs/day).
Counts over the active era: {day_counts.get("quiet", 0)} quiet &middot;
{day_counts.get("organic", 0)} organic &middot; {day_counts.get("heavy", 0)} heavy &middot;
{day_counts.get("campaign", 0)} campaign.</p>
{day_strip_html(facts)}
<div class="legend">
<span><span class="sw dc-quiet"></span>quiet</span>
<span><span class="sw dc-organic"></span>organic</span>
<span><span class="sw dc-heavy"></span>heavy</span>
<span><span class="sw dc-campaign"></span>campaign</span>
</div>
{qa_block("day classes", "python -m devtools workspace trajectory-report --json | jq '.burstiness'")}
</section>

<section id="areas"><h2>Where effort concentrated</h2>
{_prose("areas", values)}
<table><thead><tr><th>area</th><th>file-touches</th><th>weekly profile (shared axis, shared scale)</th>
<th>peak week</th></tr></thead>
<tbody>{area_rows}</tbody></table>
{qa_block("area attribution", "python -m devtools workspace trajectory-report --json | jq '.areas_weekly'")}
</section>

<section id="coupling"><h2>Does bead churn track code churn?</h2>
{_prose("coupling", values)}
<div class="chartwrap">{scatter}</div>
{qa_block("coupling", "python -m devtools workspace trajectory-report --json | jq '.coupling'")}
</section>

<section id="uncertainties"><h2>Honest uncertainties</h2>
{_prose("uncertainties", values)}
<table><tbody>{uncertainty_rows}</tbody></table>
</section>

</main></div>
<footer>generated by <code>devtools workspace trajectory-report</code> &middot;
beads from <code>{esc(beads_source.name)}</code> &middot; git history from the working checkout</footer>
<script>{JS}</script>
</body></html>"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    root = _get_root()
    parser = argparse.ArgumentParser(
        prog="devtools workspace trajectory-report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default=None, help="write HTML here (default: .local/trajectory-report.html)")
    parser.add_argument("--beads", default=None, help="bead JSONL export (default: .beads/issues.jsonl)")
    parser.add_argument("--fresh", action="store_true", help="bd export to a temp file instead of the committed jsonl")
    parser.add_argument("--json", action="store_true", dest="json_out", help="print computed facts as JSON")
    args = parser.parse_args(argv)

    beads: list[dict[str, Any]]
    beads_source: Path
    if args.fresh:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", prefix="trajectory-") as handle:
            result = subprocess.run(["bd", "export", "-o", handle.name], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"bd export failed: {result.stderr.strip() or result.stdout.strip()}", file=sys.stderr)
                return 1
            beads_source = Path(handle.name)
            beads = load_beads(beads_source)
        beads_source = Path("<bd export --fresh>")
    else:
        beads_source = Path(args.beads) if args.beads else root / ".beads/issues.jsonl"
        if not beads_source.exists():
            print(f"no such file: {beads_source}", file=sys.stderr)
            return 1
        beads = load_beads(beads_source)

    commits = collect_commits(root)
    first_parent = collect_first_parent_subjects(root)
    now = dt.datetime.now(dt.UTC)
    facts = build_facts(commits, first_parent, beads, now)

    if args.json_out:
        print(json.dumps(facts_json(facts), indent=2, sort_keys=True))
        return 0

    out = Path(args.out) if args.out else root / ".local/trajectory-report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(facts, beads_source, now), encoding="utf-8")
    print(
        f"wrote {out} ({out.stat().st_size:,} bytes) from "
        f"{facts.commits_total:,} commits, {facts.prs_total:,} PRs, {facts.beads_total:,} beads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
