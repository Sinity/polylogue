"""merge-gate: make "is this PR actually safe to merge" a structural check, not a memory habit.

Two real incidents from a single ~28-PR merge-train session (2026-08-01) motivate
this:

  1. PR #3502 was squash-merged with 0 review comments showing at check time;
     CodeRabbit posted 3 real findings 30-60s later. The fix (an explicit
     "poll comments a few times before merging" habit) worked for the rest of
     that session, but it lived entirely in the coordinator's own discipline.
  2. PR #3517 nearly merged carrying a 43-test regression that no CI check and
     no review comment ever flagged -- per-PR CI deliberately skips the heavy
     test suite (see CLAUDE.md), so nothing but a coordinator choosing, from
     memory, to run the broader local suite before merging would have caught
     it. It was caught, that time.

A coordinator merging dozens of PRs across a few hours cannot reliably repeat
either habit purely from memory every single time. This command turns both
into something that fails closed instead of silently not happening:

  - ``record``: run (or accept the exit code of) a local verification command
    against a PR branch's current HEAD commit, and persist a receipt keyed to
    that exact sha under ``.cache/verify/merge-gate/pr-<N>.json``.
  - ``check``: BLOCK unless a receipt exists for the PR's *current* head sha
    (not a stale one from an earlier push), was recorded within a freshness
    window, and had exit code 0 -- AND no review comment's ``created_at`` is
    newer than that head commit's ``committedDate``. Late-arriving comments
    are printed explicitly rather than requiring a human to eyeball two
    timestamps; a push after the last recorded receipt is a hard block, not
    an advisory.

This does not replace judgment about *what* a late comment means -- it makes
the presence of an unverified late signal impossible to merge past silently.

Usage:
    devtools workspace merge-gate record 3517 --command "devtools verify --quick"
    devtools workspace merge-gate check 3517
    devtools workspace merge-gate check 3517 --json --max-age-s 7200
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_RECEIPT_DIR = Path(".cache/verify/merge-gate")
_DEFAULT_MAX_AGE_S = 3600


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    return json.loads(result.stdout)


@dataclass
class GateVerdict:
    pr: int
    ok: bool
    reasons: list[str] = field(default_factory=list)
    head_sha: str = ""
    receipt: dict[str, Any] | None = None
    late_comments: list[dict[str, Any]] = field(default_factory=list)


def _receipt_path(pr: int) -> Path:
    return _RECEIPT_DIR / f"pr-{pr}.json"


def cmd_record(pr: int, command: str) -> int:
    info = _gh_json(["pr", "view", str(pr), "--json", "headRefOid,headRefName"])
    head_sha = info["headRefOid"]

    argv = shlex.split(command)
    started = time.time()
    result = subprocess.run(argv, capture_output=True, text=True)
    duration_s = round(time.time() - started, 2)

    receipt = {
        "pr": pr,
        "head_sha": head_sha,
        "branch": info["headRefName"],
        "command": command,
        "exit_code": result.returncode,
        "duration_s": duration_s,
        "recorded_at": time.time(),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    _RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    _receipt_path(pr).write_text(json.dumps(receipt, indent=2))

    print(f"recorded receipt for PR #{pr} @ {head_sha[:8]}: exit={result.returncode} ({duration_s}s)")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


def cmd_check(pr: int, *, max_age_s: int, as_json: bool) -> int:
    verdict = GateVerdict(pr=pr, ok=True)

    try:
        info = _gh_json(
            [
                "pr",
                "view",
                str(pr),
                "--json",
                "headRefOid,mergeStateStatus,state,commits",
            ]
        )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        verdict.ok = False
        verdict.reasons.append(f"gh pr view failed: {exc}")
        _emit(verdict, as_json)
        return 1

    if info.get("state") != "OPEN":
        verdict.ok = False
        verdict.reasons.append(f"PR state is {info.get('state')!r}, not OPEN")
        _emit(verdict, as_json)
        return 1

    head_sha = info["headRefOid"]
    verdict.head_sha = head_sha

    mss = info.get("mergeStateStatus", "")
    if mss not in {"CLEAN", "UNSTABLE", "UNKNOWN"}:
        verdict.ok = False
        verdict.reasons.append(f"mergeStateStatus is {mss!r} (expected CLEAN/UNSTABLE/UNKNOWN)")

    commits = info.get("commits") or []
    head_commit = commits[-1] if commits else None
    head_committed_at = head_commit.get("committedDate") if head_commit else None

    receipt_path = _receipt_path(pr)
    if not receipt_path.exists():
        verdict.ok = False
        verdict.reasons.append(
            f"no local verification receipt found at {receipt_path} -- run "
            f'`devtools workspace merge-gate record {pr} --command "..."` against the current head first'
        )
    else:
        receipt = json.loads(receipt_path.read_text())
        verdict.receipt = receipt
        if receipt.get("head_sha") != head_sha:
            verdict.ok = False
            verdict.reasons.append(
                f"receipt is for sha {receipt.get('head_sha', '')[:8]} but PR head is now {head_sha[:8]} "
                "-- a new commit landed since the receipt was recorded; re-record before merging"
            )
        else:
            age_s = time.time() - receipt.get("recorded_at", 0)
            if age_s > max_age_s:
                verdict.ok = False
                verdict.reasons.append(f"receipt is {int(age_s)}s old (max {max_age_s}s) -- re-record")
            if receipt.get("exit_code", 1) != 0:
                verdict.ok = False
                verdict.reasons.append(f"receipt exit_code is {receipt.get('exit_code')}, not 0")

    try:
        review_comments = _gh_json(["api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/comments"])
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        verdict.ok = False
        verdict.reasons.append(f"could not fetch review comments: {exc}")
        review_comments = []

    if head_committed_at:
        for comment in review_comments:
            created_at = comment.get("created_at", "")
            if created_at > head_committed_at:
                verdict.late_comments.append(
                    {
                        "path": comment.get("path"),
                        "line": comment.get("line"),
                        "created_at": created_at,
                        "body_head": (comment.get("body") or "")[:200],
                    }
                )
        if verdict.late_comments:
            verdict.ok = False
            verdict.reasons.append(
                f"{len(verdict.late_comments)} review comment(s) posted after the head commit "
                f"({head_committed_at}) -- read and triage before merging"
            )
    elif review_comments:
        verdict.reasons.append("could not determine head commit timestamp; late-comment check skipped")

    _emit(verdict, as_json)
    return 0 if verdict.ok else 1


def _emit(verdict: GateVerdict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(verdict), indent=2))
        return
    print(f"PR #{verdict.pr} @ {verdict.head_sha[:8] if verdict.head_sha else '?'}: {'OK' if verdict.ok else 'BLOCK'}")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    for late in verdict.late_comments:
        print(f"    late comment [{late['path']}:{late['line']}] {late['created_at']}: {late['body_head']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    record_p = sub.add_parser("record", help="Run local verification against a PR's current head and persist a receipt")
    record_p.add_argument("pr", type=int)
    record_p.add_argument(
        "--command", required=True, help="Local verification command to run, e.g. 'devtools verify --quick'"
    )

    check_p = sub.add_parser("check", help="Decide whether a PR is safe to merge right now")
    check_p.add_argument("pr", type=int)
    check_p.add_argument("--max-age-s", type=int, default=_DEFAULT_MAX_AGE_S)
    check_p.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)

    if args.action == "record":
        return cmd_record(args.pr, args.command)
    return cmd_check(args.pr, max_age_s=args.max_age_s, as_json=args.as_json)


if __name__ == "__main__":
    sys.exit(main())
