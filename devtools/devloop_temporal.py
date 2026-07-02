"""Compose devloop-local event sources into a TemporalEvidenceWindow."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from polylogue.surfaces.temporal_evidence import TemporalEvidenceEvent, build_temporal_evidence_window

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPERATING_LOG = ROOT / ".agent/scratch/current/2026-06-30-devloop/OPERATING-LOG.md"
DEFAULT_EVENT_LOG = ROOT / ".agent/scratch/current/2026-06-30-devloop/EVENTS.jsonl"
WARSAW = ZoneInfo("Europe/Warsaw")
LOG_HEADING_RE = re.compile(
    r"^## (?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}(?::\d{2})?) CEST [—-] (?P<title>.+)$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace temporal-devloop",
        description="Compose git and operating-log events into the shared temporal evidence window.",
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="Git repository to read commits from.")
    parser.add_argument("--log", type=Path, default=DEFAULT_OPERATING_LOG, help="Conductor OPERATING-LOG.md path.")
    parser.add_argument(
        "--event-log", type=Path, default=DEFAULT_EVENT_LOG, help="Structured devloop EVENTS.jsonl path."
    )
    parser.add_argument("--since", default=None, help="Inclusive ISO datetime lower bound.")
    parser.add_argument("--until", default=None, help="Inclusive ISO datetime upper bound.")
    parser.add_argument("--max-commits", type=int, default=100, help="Maximum git commits to include.")
    parser.add_argument("--max-log-events", type=int, default=200, help="Maximum operating-log headings to include.")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report to this path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout. Accepted for devtools parity.")
    return parser


def _parse_bound(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=WARSAW)
    return parsed


def _parse_log_datetime(date_text: str, time_text: str) -> datetime:
    if time_text.count(":") == 1:
        time_text = f"{time_text}:00"
    return datetime.fromisoformat(f"{date_text}T{time_text}").replace(tzinfo=WARSAW)


def operating_log_events(path: Path, *, limit: int) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    """Parse timestamped operating-log headings into temporal events."""

    if not path.exists():
        return [], ("operating_log_missing",)
    events: list[TemporalEvidenceEvent] = []
    skipped = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("## "):
            continue
        match = LOG_HEADING_RE.match(line)
        if match is None:
            skipped += 1
            continue
        if len(events) >= limit:
            break
        occurred_at = _parse_log_datetime(match.group("date"), match.group("time"))
        title = match.group("title").strip()
        phase = None
        if title.startswith("focus: ") and " -> " in title:
            phase = title.rsplit(" -> ", maxsplit=1)[-1].strip()
        events.append(
            TemporalEvidenceEvent(
                event_id=f"operating-log:{line_no}",
                occurred_at=occurred_at,
                family="devloop-log",
                kind="focus" if phase else "checkpoint",
                label=title,
                source_ref=f"{path}:{line_no}",
                evidence_refs=(f"file:{path}:{line_no}",),
                phase=phase,
            )
        )
    caveats: list[str] = []
    if skipped:
        caveats.append("operating_log_unparsed_headings")
    if len(events) >= limit:
        caveats.append("operating_log_events_capped")
    return events, tuple(caveats)


def structured_devloop_events(path: Path, *, limit: int) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    """Read structured devloop JSONL records into temporal events."""

    if not path.exists():
        return [], ("devloop_event_log_missing",)
    events: list[TemporalEvidenceEvent] = []
    malformed = 0
    unsupported = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if len(events) >= limit:
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if record.get("schema_version") != 1:
            unsupported += 1
            continue
        occurred_raw = record.get("occurred_at")
        title_raw = record.get("title")
        if not isinstance(occurred_raw, str) or not isinstance(title_raw, str):
            malformed += 1
            continue
        occurred_at: datetime | None
        if occurred_raw.endswith(" CEST"):
            occurred_text = occurred_raw.removesuffix(" CEST")
            try:
                occurred_at = datetime.strptime(occurred_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=WARSAW)
            except ValueError:
                malformed += 1
                continue
        else:
            try:
                occurred_at = _parse_bound(occurred_raw)
            except ValueError:
                malformed += 1
                continue
        if occurred_at is None:
            malformed += 1
            continue
        title = title_raw.strip()
        phase = None
        if title.startswith("focus: ") and " -> " in title:
            phase = title.rsplit(" -> ", maxsplit=1)[-1].strip()
        events.append(
            TemporalEvidenceEvent(
                event_id=f"devloop-event:{line_no}",
                occurred_at=occurred_at,
                family="devloop-log",
                kind="focus" if phase else "checkpoint",
                label=title,
                source_ref=f"{path}:{line_no}",
                evidence_refs=(f"file:{path}:{line_no}",),
                phase=phase,
            )
        )
    caveats: list[str] = []
    if malformed:
        caveats.append("devloop_event_log_malformed_rows")
    if unsupported:
        caveats.append("devloop_event_log_unsupported_rows")
    if len(events) >= limit:
        caveats.append("devloop_event_log_events_capped")
    return events, tuple(caveats)


def git_commit_events(
    repo: Path, *, limit: int, since: datetime | None
) -> tuple[list[TemporalEvidenceEvent], tuple[str, ...]]:
    """Read local git commit events as temporal events."""

    command = [
        "git",
        "log",
        f"--max-count={max(limit, 0)}",
        "--date=iso-strict",
        "--format=%cI%x09%h%x09%s",
    ]
    if since is not None:
        command.insert(2, f"--since={since.isoformat()}")
    try:
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return [], ("git_log_failed",)
    if completed.returncode != 0:
        return [], ("git_log_failed",)
    events: list[TemporalEvidenceEvent] = []
    for line in completed.stdout.splitlines():
        parts = line.split("\t", maxsplit=2)
        if len(parts) != 3:
            continue
        occurred_at = datetime.fromisoformat(parts[0])
        short_sha = parts[1]
        subject = parts[2]
        events.append(
            TemporalEvidenceEvent(
                event_id=f"git-commit:{short_sha}",
                occurred_at=occurred_at,
                family="git",
                kind="commit",
                label=subject,
                source_ref=f"commit:{short_sha}",
                evidence_refs=(f"commit:{short_sha}",),
            )
        )
    caveats = ("git_commits_capped",) if len(events) >= limit else ()
    return events, caveats


def build_report(args: argparse.Namespace) -> dict[str, object]:
    since = _parse_bound(args.since)
    until = _parse_bound(args.until)
    log_source = "structured_jsonl"
    log_events, log_caveats = structured_devloop_events(args.event_log, limit=args.max_log_events)
    if not log_events:
        log_source = "markdown_fallback"
        log_events, log_caveats = operating_log_events(args.log, limit=args.max_log_events)
    commit_events, commit_caveats = git_commit_events(args.repo, limit=args.max_commits, since=since)
    window = build_temporal_evidence_window(
        [*log_events, *commit_events],
        since=since,
        until=until,
        caveats=(*log_caveats, *commit_caveats),
    )
    return {
        "report_version": 1,
        "command": "devtools workspace temporal-devloop",
        "repo": str(args.repo),
        "operating_log": str(args.log),
        "event_log": str(args.event_log),
        "devloop_event_source": log_source,
        "source_counts": {
            "devloop_log_events": len(log_events),
            "git_commit_events": len(commit_events),
        },
        "temporal_window": window.model_dump(mode="json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
