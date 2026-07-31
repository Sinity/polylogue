"""backlog-calibration: measured duration/discovery models over the bead corpus.

Fits the numbers a backlog-execution plan needs from the actual bead history
(and optionally the merged-PR history) instead of guessing them:

  - closed-bead lead-time percentiles, overall and split by priority, type,
    epic-child vs standalone, and dependency degree;
  - a right-censoring-honest survival view (fraction of a >=14d-old cohort
    closed within 1/3/7/14 days) -- closed-only medians are survivorship-
    biased and this section is the corrective;
  - close-reason classification (worked vs already-satisfied / obsolete /
    duplicate / misframed) with median age-at-closure per class -- the
    "verify before dispatching" economy;
  - discovery-vs-drain dynamics: created and closed per day, the
    created-per-close ratio, and net backlog growth;
  - optionally, PR open->merge latency by changed-file bucket from a
    `gh pr list --json` dump.

Calibrated against 2026-07 history for the backlog-execution design
(/realm/inbox/polylogue-audits-2026-07-31/backlog-execution-design.html);
registered as a devtools command so the model can be re-fitted as the corpus
grows instead of going stale like the guesses it replaced.

Usage:
    # Fresh export (bd export -o <tmpfile> under the hood):
    devtools workspace backlog-calibration

    # From an existing export / in tests:
    devtools workspace backlog-calibration --input beads.jsonl

    # Include PR merge-latency calibration:
    #   gh pr list --state merged --limit 4000 \
    #     --json number,createdAt,mergedAt,additions,deletions,changedFiles > prs.json
    devtools workspace backlog-calibration --prs prs.json

    # Machine-readable:
    devtools workspace backlog-calibration --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from re import IGNORECASE
from re import compile as re_compile
from typing import Any

BeadDict = dict[str, Any]

_DAY_SECONDS = 86400.0
_PERCENTILES = (10, 25, 50, 75, 90, 95)
_SURVIVAL_WINDOWS_DAYS = (1, 3, 7, 14)
_COHORT_MIN_AGE_DAYS = 14.0

# Close-reason classes, checked in order; first match wins. "worked" is the
# fall-through. Regexes over free text are advisory classification, not truth.
_CLOSE_REASON_CLASSES: tuple[tuple[str, Any], ...] = (
    ("duplicate", re_compile(r"\bdup(?:e|licate)?\b", IGNORECASE)),
    (
        "already-satisfied",
        re_compile(r"\balready\b|\bsatisfied by\b|\bfixed by\b|\bfixed in\b", IGNORECASE),
    ),
    (
        "obsolete",
        re_compile(r"\bobsolete\b|\bsuperseded\b|\bno longer\b|\bstale\b|\bmoot\b", IGNORECASE),
    ),
    (
        "misframed",
        re_compile(r"\bmisframed\b|\binvalid\b|\bnot a bug\b|\bworking as\b", IGNORECASE),
    ),
)
NO_IMPLEMENTATION_CLASSES = frozenset(c for c, _ in _CLOSE_REASON_CLASSES)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("empty distribution")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    if lower >= len(sorted_values) - 1:
        return sorted_values[-1]
    frac = rank - lower
    return sorted_values[lower] + (sorted_values[lower + 1] - sorted_values[lower]) * frac


def summarize_days(values: list[float]) -> dict[str, Any]:
    """Percentile summary of a list of durations expressed in days."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    summary: dict[str, Any] = {"n": len(ordered)}
    for pct in _PERCENTILES:
        summary[f"p{pct}_days"] = round(_percentile(ordered, pct), 3)
    summary["max_days"] = round(ordered[-1], 3)
    return summary


def classify_close_reason(reason: str | None) -> str:
    """Classify a free-text close reason; 'worked' is the fall-through."""
    text = reason or ""
    for name, pattern in _CLOSE_REASON_CLASSES:
        if pattern.search(text):
            return name
    return "worked"


def _lead_days(bead: BeadDict) -> float | None:
    created = _parse_ts(bead.get("created_at"))
    closed = _parse_ts(bead.get("closed_at"))
    if created is None or closed is None or bead.get("status") != "closed":
        return None
    return (closed - created).total_seconds() / _DAY_SECONDS


def _epic_child_ids(beads: list[BeadDict]) -> frozenset[str]:
    children: set[str] = set()
    for bead in beads:
        for dep in bead.get("dependencies") or []:
            if isinstance(dep, dict) and dep.get("type") == "parent-child":
                children.add(str(bead.get("id")))
    return frozenset(children)


def _split_summaries(closed: list[tuple[BeadDict, float]], key: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for bead, lead in closed:
        groups[str(key(bead))].append(lead)
    return {name: summarize_days(values) for name, values in sorted(groups.items())}


def _survival(beads: list[BeadDict], as_of: datetime) -> dict[str, Any]:
    cohort = [
        bead
        for bead in beads
        if (created := _parse_ts(bead.get("created_at"))) is not None
        and (as_of - created).total_seconds() / _DAY_SECONDS >= _COHORT_MIN_AGE_DAYS
    ]

    def _fractions(members: list[BeadDict]) -> dict[str, Any]:
        row: dict[str, Any] = {"n": len(members)}
        for window in _SURVIVAL_WINDOWS_DAYS:
            closed_in = sum(1 for bead in members if (lead := _lead_days(bead)) is not None and lead <= window)
            row[f"closed_within_{window}d_pct"] = round(100.0 * closed_in / len(members), 1) if members else None
        return row

    by_priority = {
        f"P{priority}": _fractions([b for b in cohort if b.get("priority") == priority]) for priority in range(5)
    }
    return {
        "cohort_min_age_days": _COHORT_MIN_AGE_DAYS,
        "overall": _fractions(cohort),
        "by_priority": by_priority,
    }


def _discovery(beads: list[BeadDict]) -> dict[str, Any]:
    created_per_day: Counter[str] = Counter()
    closed_per_day: Counter[str] = Counter()
    for bead in beads:
        created = _parse_ts(bead.get("created_at"))
        closed = _parse_ts(bead.get("closed_at"))
        if created is not None:
            created_per_day[created.date().isoformat()] += 1
        if closed is not None:
            closed_per_day[closed.date().isoformat()] += 1
    days = sorted(set(created_per_day) | set(closed_per_day))
    if not days:
        return {"days": [], "note": "no dated beads"}
    # The first day of a corpus is typically a bulk import, not discovery.
    import_day = days[0]
    series: list[dict[str, Any]] = [
        {
            "day": day,
            "created": created_per_day.get(day, 0),
            "closed": closed_per_day.get(day, 0),
            "net": created_per_day.get(day, 0) - closed_per_day.get(day, 0),
        }
        for day in days
    ]
    post = [row for row in series if row["day"] != import_day]
    created_total = sum(int(row["created"]) for row in post)
    closed_total = sum(int(row["closed"]) for row in post)
    nets: list[float] = sorted(float(row["net"]) for row in post)
    return {
        "import_day_excluded": import_day,
        "days": series,
        "post_import_created": created_total,
        "post_import_closed": closed_total,
        "created_per_close": (round(created_total / closed_total, 2) if closed_total else None),
        "median_net_per_day": (round(_percentile(nets, 50), 1) if nets else None),
        "net_negative_days": sum(1 for net in nets if net < 0),
        "post_import_day_count": len(post),
    }


def _close_reasons(closed: list[tuple[BeadDict, float]]) -> dict[str, Any]:
    with_reason = [(bead, lead) for bead, lead in closed if bead.get("close_reason")]
    classes: dict[str, list[float]] = defaultdict(list)
    for bead, lead in with_reason:
        classes[classify_close_reason(bead.get("close_reason"))].append(lead)
    no_impl = sum(len(v) for name, v in classes.items() if name in NO_IMPLEMENTATION_CLASSES)
    return {
        "closed_with_reason": len(with_reason),
        "no_implementation_pct": (round(100.0 * no_impl / len(with_reason), 1) if with_reason else None),
        "classes": {name: summarize_days(values) for name, values in sorted(classes.items())},
    }


_PR_FILE_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (1, 3, "1-2"),
    (3, 6, "3-5"),
    (6, 11, "6-10"),
    (11, 31, "11-30"),
    (31, 10**9, "31+"),
)


def _pr_latency(prs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[tuple[int, float]] = []
    for pr in prs:
        created = _parse_ts(pr.get("createdAt"))
        merged = _parse_ts(pr.get("mergedAt"))
        files = pr.get("changedFiles")
        if created is None or merged is None or not isinstance(files, int):
            continue
        rows.append((files, (merged - created).total_seconds() / 3600.0))
    buckets = {
        label: summarize_days([hours / 24.0 for files, hours in rows if lo <= files < hi])
        for lo, hi, label in _PR_FILE_BUCKETS
    }
    return {
        "n": len(rows),
        "overall_latency_hours": {
            k: (round(v * 24.0, 3) if isinstance(v, float) else v)
            for k, v in summarize_days([hours / 24.0 for _, hours in rows]).items()
        },
        "by_changed_files_days": buckets,
        "note": (
            "open->merge latency measures the merge train, not implementation: "
            "PRs in this repo open after the work is done"
        ),
    }


def build_report(
    beads: list[BeadDict],
    *,
    as_of: datetime,
    prs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    closed = [(bead, lead) for bead in beads if (lead := _lead_days(bead)) is not None]
    epic_children = _epic_child_ids(beads)
    corpus_age_days = None
    created_times = [t for bead in beads if (t := _parse_ts(bead.get("created_at")))]
    if created_times:
        corpus_age_days = round((as_of - min(created_times)).total_seconds() / _DAY_SECONDS, 1)
    report: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "population": {
            "total": len(beads),
            "by_status": dict(Counter(str(b.get("status")) for b in beads)),
            "corpus_age_days": corpus_age_days,
            "censoring_note": (
                "closed-only lead times are right-censored by corpus age and "
                "survivorship-biased by still-open beads; read the survival "
                "section for population-honest fractions"
            ),
        },
        "closed_lead_days": {
            "overall": summarize_days([lead for _, lead in closed]),
            "by_priority": _split_summaries(closed, lambda b: f"P{b.get('priority')}"),
            "by_type": _split_summaries(closed, lambda b: b.get("issue_type")),
            "by_epic_membership": _split_summaries(
                closed,
                lambda b: "epic-child" if str(b.get("id")) in epic_children else "standalone",
            ),
            "by_dependency_degree": _split_summaries(
                closed,
                lambda b: min(int(b.get("dependency_count") or 0), 2),
            ),
        },
        "survival": _survival(beads, as_of),
        "close_reasons": _close_reasons(closed),
        "discovery": _discovery(beads),
    }
    if prs is not None:
        report["pr_merge_latency"] = _pr_latency(prs)
    return report


def _load_beads_jsonl(path: Path) -> list[BeadDict]:
    beads: list[BeadDict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: not valid JSON ({exc})") from exc
        if isinstance(record, dict):
            beads.append(record)
    return beads


def _export_beads() -> list[BeadDict]:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", prefix="backlog-calib-") as handle:
        result = subprocess.run(
            ["bd", "export", "-o", handle.name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"bd export failed: {result.stderr.strip() or result.stdout.strip()}")
        return _load_beads_jsonl(Path(handle.name))


def _fmt_days(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 24:.1f}h" if value < 1 else f"{value:.1f}d"


def _render_summary_line(name: str, summary: dict[str, Any]) -> str:
    if summary.get("n", 0) == 0:
        return f"  {name:<16} n=0"
    return (
        f"  {name:<16} n={summary['n']:<5} p50={_fmt_days(summary.get('p50_days')):<7} "
        f"p90={_fmt_days(summary.get('p90_days')):<7} max={_fmt_days(summary.get('max_days'))}"
    )


def _render_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    population = report["population"]
    lines.append(
        f"backlog-calibration as of {report['as_of']} -- "
        f"{population['total']} beads ({population['by_status']}), "
        f"corpus age {population['corpus_age_days']}d"
    )
    lines.append(f"NOTE: {population['censoring_note']}")
    lead = report["closed_lead_days"]
    lines.append("\nclosed lead time (days):")
    lines.append(_render_summary_line("overall", lead["overall"]))
    for section in ("by_priority", "by_type", "by_epic_membership", "by_dependency_degree"):
        lines.append(f" {section}:")
        for name, summary in lead[section].items():
            lines.append(_render_summary_line(name, summary))
    survival = report["survival"]
    lines.append(
        f"\nsurvival (cohort created >={survival['cohort_min_age_days']:.0f}d ago, "
        f"n={survival['overall']['n']}): fraction closed within window"
    )
    for name, row in [("overall", survival["overall"]), *survival["by_priority"].items()]:
        if row["n"] == 0:
            continue
        windows = "  ".join(f"{window}d={row[f'closed_within_{window}d_pct']}%" for window in _SURVIVAL_WINDOWS_DAYS)
        lines.append(f"  {name:<8} n={row['n']:<5} {windows}")
    reasons = report["close_reasons"]
    lines.append(
        f"\nclose reasons (n={reasons['closed_with_reason']} with reason; "
        f"{reasons['no_implementation_pct']}% needed no implementation):"
    )
    for name, summary in reasons["classes"].items():
        lines.append(_render_summary_line(name, summary))
    discovery = report["discovery"]
    lines.append(
        f"\ndiscovery vs drain (excluding import day {discovery.get('import_day_excluded')}): "
        f"created={discovery.get('post_import_created')} "
        f"closed={discovery.get('post_import_closed')} "
        f"created-per-close={discovery.get('created_per_close')} "
        f"median-net/day={discovery.get('median_net_per_day')} "
        f"net-negative-days={discovery.get('net_negative_days')}"
        f"/{discovery.get('post_import_day_count')}"
    )
    if "pr_merge_latency" in report:
        pr = report["pr_merge_latency"]
        lines.append(f"\nPR open->merge latency (n={pr['n']}): {pr['note']}")
        for label, summary in pr["by_changed_files_days"].items():
            lines.append(_render_summary_line(f"{label} files", summary))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools workspace backlog-calibration",
        description="Measured duration/discovery models over the bead corpus.",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Bead JSONL export (bd export -o FILE); default runs bd export itself",
    )
    parser.add_argument(
        "--prs",
        metavar="FILE",
        help=(
            "Optional gh dump: gh pr list --state merged --limit 4000 "
            "--json number,createdAt,mergedAt,additions,deletions,changedFiles"
        ),
    )
    parser.add_argument("--json", action="store_true", dest="json_out", help="JSON output")
    args = parser.parse_args(argv)

    beads = _load_beads_jsonl(Path(args.input)) if args.input else _export_beads()
    prs: list[dict[str, Any]] | None = None
    if args.prs:
        loaded = json.loads(Path(args.prs).read_text())
        if not isinstance(loaded, list):
            raise SystemExit(f"{args.prs}: expected a JSON array of PRs")
        prs = loaded

    report = build_report(beads, as_of=datetime.now(UTC), prs=prs)
    if args.json_out:
        print(json.dumps(report, indent=2))
    else:
        print(_render_human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
