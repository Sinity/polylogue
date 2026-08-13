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
  - ``check``: polls GitHub's structured review-thread state across a real
    grace window (default 3 rounds x 20s, covering the 30-60s late-arrival
    window from incident 1)
    before deciding. BLOCKs unless a receipt exists for the PR's *current*
    head sha (not a stale one from an earlier push), was recorded within a
    freshness window, had exit code 0, and emitted a typed verification scope
    and release-baseline decision (command wording grants no authority). Every
    review thread must be
    resolved in GitHub and ``reviewDecision`` must not be
    ``CHANGES_REQUESTED``. Review requests, review summaries,
    acknowledgements, and ordinary PR conversation are not findings and do
    not need a second local acknowledgement registry.

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
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from devtools import pr_scope
from devtools.testmon_state import TerminalAuthorization, VerificationScope
from devtools.verify_runs import VERIFICATION_INVOCATION_ID_ENV as VERIFICATION_INVOCATION_ID_ENV
from devtools.verify_runs import VERIFICATION_RECEIPT_PATH_ENV as VERIFICATION_RECEIPT_PATH_ENV

_RECEIPT_DIR = Path(".cache/verify/merge-gate")
_DEFAULT_MAX_AGE_S = 3600
_DEFAULT_POLL_ROUNDS = 3
_DEFAULT_POLL_INTERVAL_S = 20


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    return json.loads(result.stdout)


# GitHub commit-status `description` is truncated to 140 chars by the API
# itself; trim ourselves so the reported description matches what actually
# gets stored rather than being silently cut mid-word server-side.
_STATUS_DESCRIPTION_MAX = 140


def _status_description(verdict: GateVerdict) -> str:
    if verdict.ok:
        return "merge-gate OK: verification fresh, no unresolved review threads"
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


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve(strict=False)
    return Path.cwd().resolve(strict=False)


@dataclass
class GateVerdict:
    pr: int
    ok: bool
    reasons: list[str] = field(default_factory=list)
    head_sha: str = ""
    receipt: dict[str, Any] | None = None
    unresolved_review_threads: list[dict[str, Any]] = field(default_factory=list)
    status_post: dict[str, Any] | None = None
    pr_scope: dict[str, Any] | None = None


def _receipt_path(pr: int) -> Path:
    return _repository_root() / _RECEIPT_DIR / f"pr-{pr}.json"


def _invocation_receipt(
    *,
    path: Path,
    invocation_id: str,
    head_sha: str,
    command_exit: int,
    checkout_root: Path,
) -> dict[str, Any] | None:
    """Load the exact run artifact bound to the launched verifier process."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("invocation_id") != invocation_id
        or payload.get("git_head") != head_sha
        or payload.get("exit_code") != command_exit
    ):
        return None
    recorded_root = payload.get("checkout_root")
    if not isinstance(recorded_root, str) or Path(recorded_root).resolve(strict=False) != checkout_root.resolve(
        strict=False
    ):
        return None
    if _verification_scope(payload) is None or _release_baseline_permission(payload) is None:
        return None
    terminal_authorization = payload.get("terminal_authorization")
    if terminal_authorization is not None and terminal_authorization not in {
        authorization.value for authorization in TerminalAuthorization
    }:
        return None
    return payload


def _release_baseline_permission(payload: Mapping[str, Any] | None) -> bool | None:
    """Read the typed release decision from an invocation-bound receipt."""
    if payload is None:
        return None
    value = payload.get("release_baseline_allowed")
    return value if isinstance(value, bool) else None


def _verification_scope(payload: Mapping[str, Any] | None) -> str | None:
    """Read the typed verification scope from an invocation-bound receipt."""
    if payload is None:
        return None
    value = payload.get("verification_scope")
    return value if value in {scope.value for scope in VerificationScope} else None


def _terminal_authorization(payload: Mapping[str, Any] | None) -> str | None:
    """Read terminal authorization from an invocation-bound receipt."""
    if payload is None:
        return None
    value = payload.get("terminal_authorization")
    return value if value in {authorization.value for authorization in TerminalAuthorization} else None


def _base_sha(info: dict[str, Any]) -> str | None:
    """Read the PR base commit SHA when GitHub reported one."""
    value = info.get("baseRefOid")
    return value if isinstance(value, str) else None


def _scope_verdict(
    pr: int,
    info: dict[str, Any],
    *,
    head_sha: str,
    checkout_root: Path,
) -> pr_scope.ScopeVerdict:
    """Use the same carrier or typed bot exception for record and check."""
    author = info.get("author")
    files = info.get("files")
    author_login = author.get("login") if isinstance(author, dict) else None
    author_type = author.get("type") if isinstance(author, dict) else None
    author_is_bot = author.get("is_bot") if isinstance(author, dict) else None
    changed_files: tuple[str, ...] = ()
    if isinstance(files, list):
        changed_files = tuple(
            path for item in files if isinstance(item, dict) and isinstance((path := item.get("path")), str)
        )
    if not bool(info.get("isDraft")) and pr_scope.automated_dependency_scope_allowed(
        author_login=author_login if isinstance(author_login, str) else None,
        author_type=author_type if isinstance(author_type, str) else None,
        author_is_bot=author_is_bot if isinstance(author_is_bot, bool) else None,
        changed_files=changed_files,
    ):
        return pr_scope.ScopeVerdict(ok=True)
    return pr_scope.validate_pr_body(
        info.get("body") or "",
        head_sha=head_sha,
        is_draft=bool(info.get("isDraft")),
        beads_path=checkout_root / ".beads" / "issues.jsonl",
        base_sha=_base_sha(info),
    )


def cmd_record(pr: int, command: str) -> int:
    info = _gh_json(["pr", "view", str(pr), "--json", "headRefOid,headRefName,baseRefOid,body,isDraft,author,files"])
    head_sha = info["headRefOid"]
    checkout_root = _repository_root()

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

    scope = _scope_verdict(pr, info, head_sha=head_sha, checkout_root=checkout_root)
    if not scope.ok:
        print(f"REFUSING to record: PR #{pr} has an invalid structured pr-scope carrier:", file=sys.stderr)
        for reason in scope.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2

    argv = shlex.split(command)
    if not argv:
        print("REFUSING to record: --command is empty after shell splitting.", file=sys.stderr)
        return 2
    invocation_id = uuid.uuid4().hex
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="polylogue-merge-gate-") as temp_dir:
        receipt_path = Path(temp_dir) / "run.json"
        env = dict(os.environ)
        env[VERIFICATION_INVOCATION_ID_ENV] = invocation_id
        env[VERIFICATION_RECEIPT_PATH_ENV] = str(receipt_path)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, cwd=checkout_root, env=env)
        except OSError as exc:
            print(f"REFUSING to record: could not run {command!r}: {exc}", file=sys.stderr)
            return 2
        verification_receipt = _invocation_receipt(
            path=receipt_path,
            invocation_id=invocation_id,
            head_sha=head_sha,
            command_exit=result.returncode,
            checkout_root=checkout_root,
        )
    duration_s = round(time.time() - started, 2)

    receipt = {
        "pr": pr,
        "head_sha": head_sha,
        "pr_scope_digest": scope.scope_digest,
        "pr_scope_beads_digest": scope.beads_digest,
        "pr_scope_assigned_beads": scope.assigned_beads,
        "pr_scope_mutated_beads": scope.mutated_beads,
        "pr_scope_attestation_digest": pr_scope.attestation_payload(
            scope,
            head_sha=head_sha,
            base_sha=_base_sha(info),
        )["attestation_digest"],
        "branch": info["headRefName"],
        "command": command,
        "verification_scope": _verification_scope(verification_receipt),
        "release_baseline_allowed": _release_baseline_permission(verification_receipt),
        "terminal_authorization": _terminal_authorization(verification_receipt),
        "exit_code": result.returncode,
        "duration_s": duration_s,
        "recorded_at": time.time(),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    receipt_path = _receipt_path(pr)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"recorded receipt for PR #{pr} @ {head_sha[:8]}: exit={result.returncode} ({duration_s}s)")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


_REVIEW_STATE_QUERY = """
query($owner:String!,$repo:String!,$number:Int!,$endCursor:String) {
  repository(owner:$owner,name:$repo) {
    pullRequest(number:$number) {
      reviewDecision
      reviewThreads(first:100,after:$endCursor) {
        nodes {
          id
          isResolved
          isOutdated
          comments(first:100) {
            nodes { databaseId createdAt path line body }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def _fetch_review_state(pr: int) -> dict[str, Any] | None:
    """Return GitHub's typed review decision and every unresolved thread.

    Conversation bodies are deliberately not classified. A finding is a
    review thread; its disposition is GitHub's ``isResolved`` field. This
    avoids treating review requests, bot summaries, and repair replies as new
    findings merely because they are prose posted after a commit.
    """
    result = subprocess.run(
        [
            "gh",
            "api",
            "graphql",
            "--paginate",
            "--slurp",
            "-F",
            "owner={owner}",
            "-F",
            "repo={repo}",
            "-F",
            f"number={pr}",
            "-f",
            f"query={_REVIEW_STATE_QUERY}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return None
    try:
        pages = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(pages, list) or not pages:
        return None

    decision: str | None = None
    unresolved: dict[str, dict[str, Any]] = {}
    for page in pages:
        try:
            pull_request = page["data"]["repository"]["pullRequest"]
            page_decision = pull_request.get("reviewDecision")
            threads = pull_request["reviewThreads"]["nodes"]
        except (KeyError, TypeError):
            return None
        if isinstance(page_decision, str):
            decision = page_decision
        if not isinstance(threads, list):
            return None
        for thread in threads:
            if not isinstance(thread, dict) or thread.get("isResolved") is not False:
                continue
            thread_id = thread.get("id")
            comments = thread.get("comments", {}).get("nodes", [])
            if not isinstance(thread_id, str) or not isinstance(comments, list):
                return None
            last = comments[-1] if comments and isinstance(comments[-1], dict) else {}
            unresolved[thread_id] = {
                "thread_id": thread_id,
                "is_outdated": bool(thread.get("isOutdated")),
                "comment_id": last.get("databaseId"),
                "path": last.get("path"),
                "line": last.get("line"),
                "created_at": last.get("createdAt"),
                "body_head": str(last.get("body") or "")[:200],
            }
    return {"review_decision": decision, "unresolved_threads": list(unresolved.values())}


def _poll_stable_review_state(pr: int, *, rounds: int, interval_s: int) -> dict[str, Any] | None:
    """Poll typed review state so findings arriving after CI are observed."""
    last: dict[str, Any] | None = None
    for round_index in range(max(1, rounds)):
        review_state = _fetch_review_state(pr)
        if review_state is None:
            return None
        last = review_state
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
                "headRefOid,baseRefOid,mergeStateStatus,state,body,isDraft,author,files",
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

    local_head = _git_head_sha()
    if local_head != head_sha:
        verdict.ok = False
        verdict.reasons.append(
            f"current checkout HEAD ({local_head[:8] if local_head else '?'}) does not match PR #{pr}'s "
            f"head ({head_sha[:8]}); check out the exact PR commit before checking"
        )
    if not _git_is_clean():
        verdict.ok = False
        verdict.reasons.append(
            "current checkout has uncommitted changes; merge-gate check requires committed PR content"
        )

    scope = _scope_verdict(pr, info, head_sha=head_sha, checkout_root=_repository_root())
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
            if receipt.get("pr_scope_mutated_beads") != scope.mutated_beads:
                verdict.ok = False
                verdict.reasons.append(
                    "receipt pr_scope_mutated_beads does not match the complete current Bead mutation scope -- re-record"
                )
            expected_attestation_digest = pr_scope.attestation_payload(
                scope,
                head_sha=head_sha,
                base_sha=_base_sha(info),
            )["attestation_digest"]
            if receipt.get("pr_scope_attestation_digest") != expected_attestation_digest:
                verdict.ok = False
                verdict.reasons.append(
                    "receipt pr_scope_attestation_digest does not match the current head-bound scope attestation -- re-record"
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

    review_state = _poll_stable_review_state(pr, rounds=poll_rounds, interval_s=poll_interval_s)
    if review_state is None:
        verdict.ok = False
        verdict.reasons.append("could not fetch structured review-thread state after polling")
    else:
        verdict.unresolved_review_threads = review_state["unresolved_threads"]
        if verdict.unresolved_review_threads:
            verdict.ok = False
            verdict.reasons.append(
                f"{len(verdict.unresolved_review_threads)} unresolved GitHub review thread(s) -- "
                "fix or explicitly resolve each thread before merging"
            )
        if review_state["review_decision"] == "CHANGES_REQUESTED":
            verdict.ok = False
            verdict.reasons.append("GitHub reviewDecision is CHANGES_REQUESTED")

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
    for thread in verdict.unresolved_review_threads:
        print(
            f"    unresolved thread {thread['thread_id']} [{thread['path']}:{thread['line']}] "
            f"{thread['created_at']}: {thread['body_head']}"
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

    args = parser.parse_args(argv)

    if args.action == "record":
        return cmd_record(args.pr, args.command)
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
