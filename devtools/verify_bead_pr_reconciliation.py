"""Surface beads whose referenced PR merged but the bead is still open.

Background
----------

2026-07-14: a Workflow-driven fanout campaign (`.agent/scratch/wave2-campaign.js`,
55 beads across 7 clusters) explicitly barred worker agents from closing beads
("Do NOT close beads yourself... orchestrator runs merge-train after review"),
and the promised merge-train/reconciliation phase never ran. Every PR the
campaign produced was independently reviewed and merged -- the review loop
itself worked -- but zero beads were closed afterward. Recovering an accurate
picture took a multi-hour manual archaeology pass: reading every merged PR
body, cross-referencing bead IDs, checking each bead's own acceptance
criteria against what the diff actually did, then closing what was genuinely
satisfied (`bd close`) and leaving honest notes on what wasn't.

This check turns the *detection* half of that archaeology into a fast,
repeatable command. It does not auto-close anything -- deciding whether a
merged PR actually satisfies a bead's AC needs judgment (this session found
real cases of a PR claiming a sweep was complete when it wasn't; see
polylogue-a7xr.9's notes). It only surfaces candidates: a still-open bead
mentioned in a merged PR's body is drift worth a human or agent look, not
proof of anything.

Heuristic: scan `git log <base>..HEAD --grep='(#N)'` (squash-merge commits)
since `--since`, extract every `polylogue-<id>` token from each commit's PR
body (via `gh pr view`), and report the ones whose bead is currently open.
Over-reports by design (a PR's own "deferred" section also mentions bead
IDs) -- the point is recall, not precision; a false positive costs one `bd
show` to dismiss, a false negative costs the multi-hour archaeology this
check exists to avoid repeating.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field

from devtools import repo_root as _get_root

_BEAD_ID_RE = re.compile(r"\bpolylogue-[a-z0-9]+(?:\.[0-9]+)*\b")
_PR_SUBJECT_RE = re.compile(r"\(#(\d+)\)\s*$")


@dataclass(frozen=True, slots=True)
class DriftCandidate:
    bead_id: str
    pr_number: int
    pr_title: str
    merged_at: str
    bead_status: str


@dataclass(frozen=True, slots=True)
class _MergedPr:
    number: int
    title: str
    merged_at: str
    body: str = field(repr=False)


def _merged_prs_since(*, since: str, limit: int) -> list[_MergedPr]:
    raw = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "merged",
            "--search",
            f"merged:>={since}",
            "--limit",
            str(limit),
            "--json",
            "number,title,mergedAt,body",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    payload = json.loads(raw)
    return [
        _MergedPr(number=item["number"], title=item["title"], merged_at=item["mergedAt"], body=item.get("body") or "")
        for item in payload
    ]


def _bead_statuses(bead_ids: set[str]) -> dict[str, str]:
    if not bead_ids:
        return {}
    raw = subprocess.run(
        ["bd", "list", "--limit", "0", "--json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    decoder = json.JSONDecoder()
    issues, _ = decoder.raw_decode(raw.lstrip())
    statuses = {item["id"]: item.get("status", "unknown") for item in issues}
    raw_closed = subprocess.run(
        ["bd", "list", "--status", "closed", "--limit", "0", "--json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    closed_issues, _ = decoder.raw_decode(raw_closed.lstrip())
    statuses.update({item["id"]: item.get("status", "closed") for item in closed_issues})
    return statuses


def collect_findings(prs: list[_MergedPr]) -> list[DriftCandidate]:
    per_pr_beads: dict[int, set[str]] = {}
    all_ids: set[str] = set()
    for pr in prs:
        ids = {m.group(0) for m in _BEAD_ID_RE.finditer(pr.body)}
        if ids:
            per_pr_beads[pr.number] = ids
            all_ids.update(ids)

    statuses = _bead_statuses(all_ids)
    by_number = {pr.number: pr for pr in prs}

    findings: list[DriftCandidate] = []
    for number, ids in per_pr_beads.items():
        pr = by_number[number]
        for bead_id in sorted(ids):
            status = statuses.get(bead_id, "unknown")
            if status not in ("closed",):
                findings.append(
                    DriftCandidate(
                        bead_id=bead_id,
                        pr_number=number,
                        pr_title=pr.title,
                        merged_at=pr.merged_at,
                        bead_status=status,
                    )
                )
    findings.sort(key=lambda f: (f.bead_id, f.pr_number))
    return findings


def _format_report(findings: list[DriftCandidate], *, prs_scanned: int) -> str:
    if not findings:
        return f"bead/PR reconciliation: no drift candidates across {prs_scanned} merged PR(s) scanned."
    lines = [f"{len(findings)} drift candidate(s) across {prs_scanned} merged PR(s) scanned:"]
    for f in findings:
        lines.append(f'  {f.bead_id} ({f.bead_status}) <- PR #{f.pr_number} "{f.pr_title}" (merged {f.merged_at})')
    lines.append(
        "\nEach is a candidate, not a verdict -- check the bead's AC against what the PR actually did "
        "(bd show <id> --json; gh pr view <N> --json body) before closing."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--since",
        default=None,
        help="only scan PRs merged on/after this date (YYYY-MM-DD); default: 14 days ago",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="max merged PRs to scan (default: 200)",
    )
    args = parser.parse_args(argv)

    _get_root()  # validates we're inside the repo; gh/bd resolve the workspace themselves

    since = args.since
    if since is None:
        import datetime

        since = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=14)).strftime("%Y-%m-%d")

    prs = _merged_prs_since(since=since, limit=args.limit)
    findings = collect_findings(prs)

    if args.json:
        payload = {
            "ok": not findings,
            "prs_scanned": len(prs),
            "since": since,
            "findings": [
                {
                    "bead_id": f.bead_id,
                    "pr_number": f.pr_number,
                    "pr_title": f.pr_title,
                    "merged_at": f.merged_at,
                    "bead_status": f.bead_status,
                }
                for f in findings
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(findings, prs_scanned=len(prs)))

    return 0  # advisory report, never fails a gate


if __name__ == "__main__":
    sys.exit(main())
