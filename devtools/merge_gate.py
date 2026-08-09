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

  - ``record``: run a local verification command against a PR branch's
    current HEAD commit, and persist a receipt keyed to that exact sha under
    ``.cache/verify/merge-gate/pr-<N>.json``. Refuses to record unless the
    CURRENT git checkout is clean and its ``HEAD`` matches the PR's fetched
    ``headRefOid`` -- otherwise the receipt would attest to code that was
    never actually tested (review-caught gap: recording from an unrelated
    checkout, e.g. master or a stale worktree, previously produced a receipt
    that ``check`` would accept).
  - ``check``: polls PR review comments across a real grace window (default
    3 rounds x 20s, covering the 30-60s late-arrival window from incident 1)
    before deciding. BLOCKs unless a receipt exists for the PR's *current*
    head sha (not a stale one from an earlier push), was recorded within a
    freshness window, had exit code 0, and its command actually looks like it
    ran tests (a bare ``--quick`` profile is flagged, not silently accepted --
    review-caught gap: the documented example used exactly the profile that
    would have missed the PR #3517 regression). No review comment's
    ``created_at`` may be newer than the head commit's ``committedDate``
    unless it has been explicitly acknowledged via ``ack`` for this exact
    head sha (review-caught gap: without ``ack``, a stale-forever comparison
    made even a reviewed false positive permanently unmergeable without an
    empty commit).

This does not replace judgment about *what* a late comment means -- ``ack``
still requires a human/agent to have actually read it and decided it's not
actionable. It makes the presence of an unverified late signal impossible to
merge past silently, and impossible to permanently paper over without an
explicit, current-head-scoped decision.

``check --post-status`` closes the remaining gap (2026-08-03,
polylogue-1cbeh): the verdict above is a purely local CLI result, so nothing
in GitHub itself -- branch protection, `gh pr merge --auto` -- has anything
to wait on, and a coordinator ends up manually cycling
checkout->test->check->merge one PR at a time. `--post-status` posts the
verdict as a commit status (`context=merge-gate`) on the PR's current head
sha via `gh api repos/{owner}/{repo}/statuses/{sha}`: OK -> `success`, BLOCK
-> `failure` with the block reason(s) in the description. Branch protection
can then require this status like any other check, and `gh pr merge --auto`
has something real to gate on. This does not itself change branch
protection settings (a one-time operator action) or switch lanes to
`--auto` merging -- see polylogue-1cbeh follow-ups.

Usage:
    devtools workspace merge-gate record 3517 --command "devtools verify"
    devtools workspace merge-gate check 3517
    devtools workspace merge-gate check 3517 --json --max-age-s 7200 --poll-rounds 1
    devtools workspace merge-gate check 3517 --post-status
    devtools workspace merge-gate ack 3517 <comment-id> --reason "false positive, already fixed upstream"
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

from devtools import pr_scope
from devtools.testmon_state import TerminalAuthorization, VerificationScope

_RECEIPT_DIR = Path(".cache/verify/merge-gate")
_DEFAULT_MAX_AGE_S = 3600
_DEFAULT_POLL_ROUNDS = 3
_DEFAULT_POLL_INTERVAL_S = 20
# Heuristic: profiles that explicitly skip tests (see CLAUDE.md -- `devtools
# verify --quick` is format+lint+mypy+render, no pytest). Not exhaustive; a
# command containing neither this nor an obvious test-runner name still gets
# flagged as an advisory, since the whole point is not trusting a plausible-
# looking command string without comment.
_TEST_SKIPPING_MARKERS: tuple[str, ...] = ("verify --quick", "verify --lab")
_LOOKS_LIKE_TESTS_MARKERS: tuple[str, ...] = ("test", "pytest", "verify --all", "devtools verify")


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    return json.loads(result.stdout)


def _gh_json_paginated(args: list[str]) -> list[Any]:
    """Like ``_gh_json`` but follows pagination -- the GitHub REST list
    endpoints cap at 30-100 items per page, and a PR with more review comments
    than that would otherwise silently hide later (possibly late-arriving)
    ones from the late-comment check. ``--slurp`` wraps every page's own JSON
    array into one outer array, which this then flattens."""
    result = subprocess.run(["gh", *args, "--paginate", "--slurp"], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} --paginate failed")
    pages = json.loads(result.stdout)
    items: list[Any] = []
    for page in pages:
        if isinstance(page, list):
            items.extend(page)
        else:
            items.append(page)
    return items


# GitHub commit-status `description` is truncated to 140 chars by the API
# itself; trim ourselves so the reported description matches what actually
# gets stored rather than being silently cut mid-word server-side.
_STATUS_DESCRIPTION_MAX = 140


def _status_description(verdict: GateVerdict) -> str:
    if verdict.ok:
        return "merge-gate OK: verification receipt fresh, no unacked late comments"
    joined = "; ".join(verdict.reasons) or "merge-gate BLOCK"
    if len(joined) > _STATUS_DESCRIPTION_MAX:
        joined = joined[: _STATUS_DESCRIPTION_MAX - 1] + "…"
    return joined


def _post_commit_status(head_sha: str, *, state: str, description: str) -> dict[str, Any]:
    """Post a GitHub commit status for ``head_sha`` under ``context=merge-gate``.

    Returns a small result dict (never raises) so a status-posting failure --
    rate limit, auth, network -- surfaces as an advisory in the verdict rather
    than crashing a check that otherwise completed successfully."""
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/statuses/{head_sha}",
            "-f",
            f"state={state}",
            "-f",
            "context=merge-gate",
            "-f",
            f"description={description}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {
            "posted": False,
            "state": state,
            "description": description,
            "error": result.stderr.strip()[:300] or "gh api statuses call failed",
        }
    return {"posted": True, "state": state, "description": description}


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Return the parsed object, or None when the file is absent, unreadable,
    truncated, or not a JSON object (a hand edit or an interrupted write must
    not raise a traceback out of a merge-safety check)."""
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _git_head_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_is_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=15)
    return result.returncode == 0 and not result.stdout.strip()


@dataclass
class GateVerdict:
    pr: int
    ok: bool
    reasons: list[str] = field(default_factory=list)
    head_sha: str = ""
    receipt: dict[str, Any] | None = None
    late_comments: list[dict[str, Any]] = field(default_factory=list)
    status_post: dict[str, Any] | None = None
    pr_scope: dict[str, Any] | None = None


def _receipt_path(pr: int) -> Path:
    return _RECEIPT_DIR / f"pr-{pr}.json"


def _ack_path(pr: int) -> Path:
    return _RECEIPT_DIR / f"pr-{pr}-acks.json"


def _command_skips_tests(command: str) -> bool:
    lowered = command.lower()
    if any(marker in lowered for marker in _TEST_SKIPPING_MARKERS):
        return True
    return not any(marker in lowered for marker in _LOOKS_LIKE_TESTS_MARKERS)


def _release_baseline_permission(stdout: str) -> bool | None:
    """Read the structured verify decision when the command emitted one."""
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("release_baseline_allowed")
    return value if isinstance(value, bool) else None


def _verification_scope(stdout: str) -> str | None:
    """Read the typed verification scope from a structured verify receipt."""
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("verification_scope")
    return value if value in {scope.value for scope in VerificationScope} else None


def _terminal_authorization(stdout: str) -> str | None:
    """Read the typed terminal authorization from a structured receipt."""
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("terminal_authorization")
    return value if value in {authorization.value for authorization in TerminalAuthorization} else None


def cmd_record(pr: int, command: str) -> int:
    info = _gh_json(["pr", "view", str(pr), "--json", "headRefOid,headRefName,body,isDraft"])
    head_sha = info["headRefOid"]

    local_head = _git_head_sha()
    if local_head != head_sha:
        print(
            f"REFUSING to record: current checkout HEAD ({local_head[:8] if local_head else '?'}) does not "
            f"match PR #{pr}'s head ({head_sha[:8]}). Check out the PR's exact commit (in an isolated worktree "
            "if other work is in progress elsewhere) before recording -- a receipt attesting to the wrong "
            "checkout is worse than no receipt.",
            file=sys.stderr,
        )
        return 2
    if not _git_is_clean():
        print(
            "REFUSING to record: current checkout has uncommitted changes. The receipt must attest to "
            "exactly the PR's committed content, not a locally-modified tree.",
            file=sys.stderr,
        )
        return 2

    scope = pr_scope.validate_pr_body(
        info.get("body") or "",
        head_sha=head_sha,
        is_draft=bool(info.get("isDraft")),
    )
    if not scope.ok:
        print(f"REFUSING to record: PR #{pr} has an invalid structured pr-scope carrier:", file=sys.stderr)
        for reason in scope.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2

    argv = shlex.split(command)
    if not argv:
        print("REFUSING to record: --command is empty after shell splitting.", file=sys.stderr)
        return 2
    started = time.time()
    try:
        result = subprocess.run(argv, capture_output=True, text=True)
    except OSError as exc:
        print(f"REFUSING to record: could not run {command!r}: {exc}", file=sys.stderr)
        return 2
    duration_s = round(time.time() - started, 2)

    receipt = {
        "pr": pr,
        "head_sha": head_sha,
        "pr_scope_digest": scope.scope_digest,
        "pr_scope_beads_digest": scope.beads_digest,
        "pr_scope_assigned_beads": scope.assigned_beads,
        "branch": info["headRefName"],
        "command": command,
        "skips_tests": _command_skips_tests(command),
        "verification_scope": _verification_scope(result.stdout),
        "release_baseline_allowed": _release_baseline_permission(result.stdout),
        "terminal_authorization": _terminal_authorization(result.stdout),
        "exit_code": result.returncode,
        "duration_s": duration_s,
        "recorded_at": time.time(),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    _RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    _receipt_path(pr).write_text(json.dumps(receipt, indent=2))

    print(f"recorded receipt for PR #{pr} @ {head_sha[:8]}: exit={result.returncode} ({duration_s}s)")
    if receipt["skips_tests"]:
        print(
            f"  advisory: command {command!r} does not look like it ran tests -- `check` will flag this",
            file=sys.stderr,
        )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


def cmd_ack(pr: int, comment_id: int, *, reason: str) -> int:
    info = _gh_json(["pr", "view", str(pr), "--json", "headRefOid"])
    head_sha = info["headRefOid"]

    ack_path = _ack_path(pr)
    if ack_path.exists():
        acks = _read_json_object(ack_path)
        if acks is None:
            print(
                f"REFUSING to ack: {ack_path} exists but is unreadable/corrupt -- fix or remove it by hand "
                "first, rather than silently losing prior acknowledgements.",
                file=sys.stderr,
            )
            return 2
    else:
        acks = {}
    acks[str(comment_id)] = {"head_sha": head_sha, "reason": reason, "acked_at": time.time()}
    _RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ack_path.write_text(json.dumps(acks, indent=2))
    print(f"acknowledged comment {comment_id} on PR #{pr} @ {head_sha[:8]}: {reason}")
    return 0


def _fetch_review_comments(pr: int) -> list[dict[str, Any]] | None:
    """Combine every top-level review signal the late-comment check should
    see: inline diff comments, issue-level PR comments, and review bodies
    (a review's own summary text is a separate object from its line
    comments -- see GitHub's REST API docs). All three are normalized to a
    common shape (id, created_at, path, line, body) and empty-bodied entries
    (e.g. an APPROVE review with no summary text) are dropped -- they carry
    no signal for triage."""
    normalized: list[dict[str, Any]] = []
    try:
        inline = _gh_json_paginated(["api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/comments"])
        for item in inline:
            normalized.append(
                {
                    "id": item.get("id"),
                    "created_at": item.get("created_at", ""),
                    "path": item.get("path"),
                    "line": item.get("line"),
                    "body": item.get("body") or "",
                }
            )
        issue_comments = _gh_json_paginated(["api", f"repos/{{owner}}/{{repo}}/issues/{pr}/comments"])
        for item in issue_comments:
            normalized.append(
                {
                    "id": item.get("id"),
                    "created_at": item.get("created_at", ""),
                    "path": None,
                    "line": None,
                    "body": item.get("body") or "",
                }
            )
        reviews = _gh_json_paginated(["api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/reviews"])
        for item in reviews:
            normalized.append(
                {
                    "id": item.get("id"),
                    "created_at": item.get("submitted_at", ""),
                    "path": None,
                    "line": None,
                    "body": item.get("body") or "",
                }
            )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return None
    return [comment for comment in normalized if comment["body"].strip()]


def _poll_stable_comments(pr: int, *, rounds: int, interval_s: int) -> list[dict[str, Any]] | None:
    """Poll review comments repeatedly so a comment posted 30-60s after CI
    goes green (the PR #3502 incident) is observed rather than missed by a
    single snapshot taken too early."""
    last: list[dict[str, Any]] | None = None
    for round_index in range(max(1, rounds)):
        comments = _fetch_review_comments(pr)
        if comments is None:
            return None
        last = comments
        if round_index < rounds - 1:
            time.sleep(interval_s)
    return last


def cmd_check(
    pr: int,
    *,
    max_age_s: int,
    poll_rounds: int,
    poll_interval_s: int,
    as_json: bool,
    post_status: bool = False,
) -> int:
    verdict = GateVerdict(pr=pr, ok=True)

    try:
        info = _gh_json(
            [
                "pr",
                "view",
                str(pr),
                "--json",
                "headRefOid,mergeStateStatus,state,commits,body,isDraft",
            ]
        )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        verdict.ok = False
        verdict.reasons.append(f"gh pr view failed: {exc}")
        # No head sha resolved yet -- nothing to post a status against.
        _emit(verdict, as_json)
        return 1

    head_sha = info["headRefOid"]
    verdict.head_sha = head_sha

    scope = pr_scope.validate_pr_body(
        info.get("body") or "",
        head_sha=head_sha,
        is_draft=bool(info.get("isDraft")),
    )
    verdict.pr_scope = asdict(scope)
    if not scope.ok:
        verdict.ok = False
        verdict.reasons.extend(f"pr-scope: {reason}" for reason in scope.reasons)

    if info.get("state") != "OPEN":
        verdict.ok = False
        verdict.reasons.append(f"PR state is {info.get('state')!r}, not OPEN")
        if post_status:
            verdict.status_post = _post_commit_status(
                head_sha, state="failure", description=_status_description(verdict)
            )
        _emit(verdict, as_json)
        return 1

    mss = info.get("mergeStateStatus", "")
    if mss not in {"CLEAN", "UNSTABLE", "UNKNOWN"}:
        verdict.ok = False
        verdict.reasons.append(f"mergeStateStatus is {mss!r} (expected CLEAN/UNSTABLE/UNKNOWN)")

    commits = info.get("commits") or []
    head_commit = next((commit for commit in commits if commit.get("oid") == head_sha), None)
    head_committed_at = head_commit.get("committedDate") if head_commit else None

    receipt_path = _receipt_path(pr)
    receipt = _read_json_object(receipt_path)
    if receipt is None:
        verdict.ok = False
        verdict.reasons.append(
            f"no local verification receipt found (or it is unreadable) at {receipt_path} -- run "
            f'`devtools workspace merge-gate record {pr} --command "..."` against the current head first'
        )
    else:
        verdict.receipt = receipt
        if receipt.get("head_sha") != head_sha:
            verdict.ok = False
            verdict.reasons.append(
                f"receipt is for sha {receipt.get('head_sha', '')[:8]} but PR head is now {head_sha[:8]} "
                "-- a new commit landed since the receipt was recorded; re-record before merging"
            )
        else:
            if receipt.get("pr_scope_digest") != scope.scope_digest:
                verdict.ok = False
                verdict.reasons.append(
                    "receipt pr_scope_digest does not match the current carrier -- re-record after scope changes"
                )
            if receipt.get("pr_scope_beads_digest") != scope.beads_digest:
                verdict.ok = False
                verdict.reasons.append(
                    "receipt pr_scope_beads_digest does not match the current canonical Bead records -- re-record"
                )
            if receipt.get("pr_scope_assigned_beads") != scope.assigned_beads:
                verdict.ok = False
                verdict.reasons.append(
                    "receipt pr_scope_assigned_beads does not match the current carrier -- re-record"
                )
            age_s = time.time() - receipt.get("recorded_at", 0)
            if age_s > max_age_s:
                verdict.ok = False
                verdict.reasons.append(f"receipt is {int(age_s)}s old (max {max_age_s}s) -- re-record")
            if receipt.get("exit_code", 1) != 0:
                verdict.ok = False
                verdict.reasons.append(f"receipt exit_code is {receipt.get('exit_code')}, not 0")
            verification_scope = receipt.get("verification_scope")
            if verification_scope not in {scope.value for scope in VerificationScope}:
                verdict.ok = False
                verdict.reasons.append(
                    "verification receipt lacks a valid typed verification_scope; command text cannot grant authority"
                )
            release_allowed = receipt.get("release_baseline_allowed")
            if not isinstance(release_allowed, bool):
                verdict.ok = False
                verdict.reasons.append("verification receipt lacks typed release_baseline_allowed permission")
            if verification_scope == VerificationScope.RELEASE_BASELINE.value and release_allowed is not True:
                verdict.ok = False
                verdict.reasons.append(
                    "release-baseline verification receipt does not grant release_baseline_allowed=true"
                )
            if receipt.get("skips_tests"):
                verdict.reasons.append(
                    f"advisory: receipt command {receipt.get('command')!r} does not look like it ran tests "
                    "-- confirm this PR genuinely needs no test coverage before merging"
                )

    review_comments = _poll_stable_comments(pr, rounds=poll_rounds, interval_s=poll_interval_s)
    if review_comments is None:
        verdict.ok = False
        verdict.reasons.append("could not fetch review comments after polling")
        review_comments = []

    acks = _read_json_object(_ack_path(pr)) or {}

    if head_committed_at:
        for comment in review_comments:
            created_at = comment.get("created_at", "")
            if created_at <= head_committed_at:
                continue
            comment_id = comment.get("id")
            ack = acks.get(str(comment_id))
            if ack is not None and ack.get("head_sha") == head_sha:
                continue  # explicitly triaged for this exact head sha
            verdict.late_comments.append(
                {
                    "id": comment_id,
                    "path": comment.get("path"),
                    "line": comment.get("line"),
                    "created_at": created_at,
                    "body_head": (comment.get("body") or "")[:200],
                }
            )
        if verdict.late_comments:
            verdict.ok = False
            verdict.reasons.append(
                f"{len(verdict.late_comments)} unacknowledged review comment(s) posted after the head commit "
                f"({head_committed_at}) -- read and `ack` (if not actionable) or fix before merging"
            )
    else:
        verdict.ok = False
        verdict.reasons.append(
            "could not determine the head commit timestamp, so the late-comment check cannot run -- "
            "refusing to report OK"
        )

    if post_status:
        verdict.status_post = _post_commit_status(
            head_sha,
            state="success" if verdict.ok else "failure",
            description=_status_description(verdict),
        )

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
        print(
            f"    late comment id={late['id']} [{late['path']}:{late['line']}] {late['created_at']}: {late['body_head']}"
        )
    if verdict.status_post is not None:
        if verdict.status_post.get("posted"):
            print(f"  posted commit status: state={verdict.status_post['state']} context=merge-gate")
        else:
            print(
                f"  FAILED to post commit status (state={verdict.status_post.get('state')}): "
                f"{verdict.status_post.get('error')}",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    record_p = sub.add_parser("record", help="Run local verification against a PR's current head and persist a receipt")
    record_p.add_argument("pr", type=int)
    record_p.add_argument("--command", required=True, help="Local verification command to run, e.g. 'devtools verify'")

    check_p = sub.add_parser("check", help="Decide whether a PR is safe to merge right now")
    check_p.add_argument("pr", type=int)
    check_p.add_argument("--max-age-s", type=int, default=_DEFAULT_MAX_AGE_S)
    check_p.add_argument("--poll-rounds", type=int, default=_DEFAULT_POLL_ROUNDS)
    check_p.add_argument("--poll-interval-s", type=int, default=_DEFAULT_POLL_INTERVAL_S)
    check_p.add_argument("--json", action="store_true", dest="as_json")
    check_p.add_argument(
        "--post-status",
        action="store_true",
        dest="post_status",
        help=(
            "Post the verdict as a GitHub commit status (context=merge-gate) on the PR's current "
            "head sha, so branch protection / `gh pr merge --auto` has something to gate on"
        ),
    )

    ack_p = sub.add_parser("ack", help="Acknowledge a specific review comment as triaged for the current head sha")
    ack_p.add_argument("pr", type=int)
    ack_p.add_argument("comment_id", type=int)
    ack_p.add_argument("--reason", required=True, help="Why this comment does not block merging")

    args = parser.parse_args(argv)

    if args.action == "record":
        return cmd_record(args.pr, args.command)
    if args.action == "ack":
        return cmd_ack(args.pr, args.comment_id, reason=args.reason)
    return cmd_check(
        args.pr,
        max_age_s=args.max_age_s,
        poll_rounds=args.poll_rounds,
        poll_interval_s=args.poll_interval_s,
        as_json=args.as_json,
        post_status=args.post_status,
    )


if __name__ == "__main__":
    sys.exit(main())
