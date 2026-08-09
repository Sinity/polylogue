"""merge-boundary: the merge-gate/broad-verify safety net, wired into the
actual place PRs get merged, instead of a rule a coordinator has to remember.

``devtools workspace merge-gate record/check`` (see ``merge_gate.py``) and the
one-full-suite-verify-per-merge-train rule from CLAUDE.md are both real,
already-built fixes for real 2026-08-01 incidents -- but both fire only if a
coordinator remembers to invoke them at the right moment. The fanout-
operations report's incident-ledger cross-check (polylogue-ct3r2 /
polylogue-t6iga, duplicate filings of the same finding) found the exact
pattern: "everything that became a command stuck; everything that stayed a
rule someone must remember has already recurred at least once." This module
is the command.

There is no GitHub Actions merge-boundary hook available here (CI is
CircleCI-only, GHA is intentionally dark -- see project memory). The merge
boundary in practice is a human/agent coordinator invoking
``gh pr merge --squash``. So the enforcement point is a wrapper around that
exact call:

    devtools workspace merge <PR>

which:

  1. Refuses unless the PR is OPEN.
  2. Requires the versioned ``pr-scope`` carrier on the current non-draft PR,
     including its current Bead-record digest and whole-Bead dispositions.
  3. If no fresh ``merge-gate`` receipt exists for the PR's *current* head
     sha, records one automatically (running ``--command``, default
     ``devtools verify``) instead of just failing and telling the caller to
     go run a separate command first.
  4. Runs ``merge-gate check`` (late-review-comment grace-window poll +
     receipt freshness/exit-code checks). Refuses to merge on any BLOCK.
  5. Applies title hygiene: strips a doubled ``(#N) (#N)`` suffix (the
     2026-07-12/13 incident where the squash-merge subject carried the PR
     number twice because a manual ``gh pr edit --title`` step was skipped)
     and ensures exactly one trailing ``(#N)``.
  6. Runs the actual ``gh pr merge --squash``.
  7. Appends a merge-train ledger entry (``.cache/verify/merge-gate/merge-train-ledger.json``)
     and, unless ``--with-verify`` was given, prints a reminder that the
     ledger's terminal step -- one full-suite ``devtools verify --all`` (or
     narrower agreed selection) since the last one -- has not yet been
     recorded for this train.

``devtools workspace merge train-status`` inspects the ledger and reports
(exit 1 if so) any PRs merged since the last recorded full-suite verify --
this is the structural stand-in for "a merge-train run records the full-
suite verify as its terminal ledger step": the train is not clean until this
reports OK.

``devtools workspace merge record-full-verify --command "devtools verify --all"``
runs that command now and records it as the train's terminal step, clearing
every pending PR in the ledger.

Usage:
    devtools workspace merge 3517
    devtools workspace merge 3517 --command "devtools test tests/unit/foo.py"
    devtools workspace merge 3517 --dry-run
    devtools workspace merge 3517 --with-verify --verify-command "devtools verify --all"
    devtools workspace merge train-status
    devtools workspace merge record-full-verify --command "devtools verify --all"
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from devtools import merge_gate
from devtools.testmon_state import VerificationScope

_LEDGER_PATH = Path(".cache/verify/merge-gate/merge-train-ledger.json")
_LEDGER_PENDING_PATH = _LEDGER_PATH.with_name(f"{_LEDGER_PATH.name}.pending")
_LEDGER_LOCK_PATH = _LEDGER_PATH.with_name(f"{_LEDGER_PATH.name}.lock")


class LedgerStateError(RuntimeError):
    """The merge-train ledger cannot safely authorize a result."""


# Matches a squash-merge subject that already carries the PR number and picked
# up a second, duplicate one -- e.g. "fix: thing (#3517) (#3517)". Only the
# exact-duplicate case is collapsed; a title that already ends with exactly
# one "(#N)" for the right N is left untouched.
_DOUBLED_PR_SUFFIX_RE = re.compile(r"\s*\(#(\d+)\)\s*\(#\1\)\s*$")


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    return json.loads(result.stdout)


def clean_merge_title(title: str, pr: int) -> str:
    """Strip a doubled ``(#N) (#N)`` suffix and ensure exactly one trailing
    ``(#N)`` for this PR number. Idempotent: a title already shaped correctly
    passes through unchanged."""
    collapsed = _DOUBLED_PR_SUFFIX_RE.sub(f" (#{pr})", title).strip()
    if not re.search(rf"\(#{pr}\)\s*$", collapsed):
        collapsed = f"{collapsed} (#{pr})"
    return collapsed


@contextlib.contextmanager
def _ledger_lock() -> Iterator[None]:
    """Serialize every merge-train ledger read-modify-write transaction."""
    _LEDGER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_LOCK_PATH.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_nonempty_string(entry: Mapping[str, Any], key: str, *, message: str) -> None:
    if not isinstance(entry.get(key), str) or not entry[key]:
        raise LedgerStateError(message)


def _validate_ledger(data: object) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("merges"), list):
        raise LedgerStateError("merge-train ledger is malformed")
    for entry in data["merges"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("pr"), int)
            or isinstance(entry.get("pr"), bool)
            or entry["pr"] <= 0
            or not isinstance(entry.get("head_sha"), str)
            or not entry["head_sha"]
            or not isinstance(entry.get("title"), str)
            or not entry["title"]
            or not _is_real_number(entry.get("merged_at"))
            or (
                "merge_sequence" in entry
                and (
                    not isinstance(entry.get("merge_sequence"), int)
                    or isinstance(entry.get("merge_sequence"), bool)
                    or entry["merge_sequence"] <= 0
                )
            )
        ):
            raise LedgerStateError("merge-train ledger contains a malformed merge entry")
    intents = data.get("merge_intents")
    if not isinstance(intents, list):
        raise LedgerStateError("merge-train ledger contains malformed merge intents")
    for intent in intents:
        if (
            not isinstance(intent, dict)
            or not isinstance(intent.get("pr"), int)
            or isinstance(intent.get("pr"), bool)
            or intent["pr"] <= 0
            or not isinstance(intent.get("head_sha"), str)
            or not intent["head_sha"]
            or not isinstance(intent.get("title"), str)
            or not intent["title"]
            or not _is_real_number(intent.get("intent_at"))
        ):
            raise LedgerStateError("merge-train ledger contains a malformed merge intent")
    receipt = data.get("last_full_verify")
    if receipt is None:
        return data
    if not isinstance(receipt, dict):
        raise LedgerStateError("merge-train ledger contains a malformed terminal receipt")
    for key in (
        "command",
        "verification_started_at",
        "at",
        "duration_s",
        "exit_code",
        "accepted",
        "merge_sequence",
        "verification_scope",
        "release_baseline_allowed",
    ):
        if key not in receipt:
            raise LedgerStateError(f"merge-train terminal receipt is missing {key!r}")
    _require_nonempty_string(receipt, "command", message="merge-train terminal receipt has no command")
    if (
        not _is_real_number(receipt.get("verification_started_at"))
        or not _is_real_number(receipt.get("at"))
        or not _is_real_number(receipt.get("duration_s"))
        or not isinstance(receipt.get("exit_code"), int)
        or isinstance(receipt.get("exit_code"), bool)
        or not isinstance(receipt.get("accepted"), bool)
        or not isinstance(receipt.get("merge_sequence"), int)
        or isinstance(receipt.get("merge_sequence"), bool)
        or receipt["merge_sequence"] < 0
    ):
        raise LedgerStateError("merge-train terminal receipt has malformed status fields")
    scope = receipt.get("verification_scope")
    if scope is not None and scope not in {item.value for item in VerificationScope}:
        raise LedgerStateError("merge-train terminal receipt has an invalid verification scope")
    permission = receipt.get("release_baseline_allowed")
    if permission is not None and not isinstance(permission, bool):
        raise LedgerStateError("merge-train terminal receipt has malformed release permission")
    for key in ("terminal_authorization", "verified_head_sha", "target_sha", "merged_master_sha"):
        value = receipt.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise LedgerStateError(f"merge-train terminal receipt has malformed {key!r}")
    return data


def _read_ledger_unlocked() -> dict[str, Any]:
    if _LEDGER_PENDING_PATH.exists():
        raise LedgerStateError("merge-train ledger has an unfinished durable write")
    if not _LEDGER_PATH.exists():
        return {"merges": [], "merge_intents": [], "last_full_verify": None}
    try:
        data = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerStateError("merge-train ledger is unreadable or truncated") from exc
    if isinstance(data, dict):
        data.setdefault("merges", [])
        data.setdefault("merge_intents", [])
        data.setdefault("last_full_verify", None)
    return _validate_ledger(data)


def _read_ledger() -> dict[str, Any]:
    with _ledger_lock():
        return _read_ledger_unlocked()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable_temp(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _durable_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _write_ledger_unlocked(ledger: dict[str, Any]) -> None:
    ledger.setdefault("merges", [])
    ledger.setdefault("merge_intents", [])
    ledger.setdefault("last_full_verify", None)
    _validate_ledger(ledger)
    parent = _LEDGER_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(ledger, indent=2) + "\n"
    pending_tmp = parent / f".{_LEDGER_PENDING_PATH.name}.{os.getpid()}.tmp"
    ledger_tmp = parent / f".{_LEDGER_PATH.name}.{os.getpid()}.tmp"
    pending_tmp.unlink(missing_ok=True)
    ledger_tmp.unlink(missing_ok=True)
    try:
        _write_durable_temp(pending_tmp, serialized)
        _durable_replace(pending_tmp, _LEDGER_PENDING_PATH)
        _fsync_directory(parent)
        _write_durable_temp(ledger_tmp, serialized)
        _durable_replace(ledger_tmp, _LEDGER_PATH)
        _fsync_directory(parent)
        _LEDGER_PENDING_PATH.unlink(missing_ok=True)
        _fsync_directory(parent)
    except OSError as exc:
        raise LedgerStateError(f"merge-train ledger durable write failed: {exc}") from exc
    finally:
        pending_tmp.unlink(missing_ok=True)
        ledger_tmp.unlink(missing_ok=True)


def _write_ledger(ledger: dict[str, Any]) -> None:
    with _ledger_lock():
        _write_ledger_unlocked(ledger)


def _merge_sequence(ledger: Mapping[str, Any]) -> int:
    sequence = 0
    for index, entry in enumerate(ledger.get("merges", []), start=1):
        raw = entry.get("merge_sequence") if isinstance(entry, Mapping) else None
        sequence = max(sequence, raw if isinstance(raw, int) and not isinstance(raw, bool) else index)
    return sequence


def _append_merge_entry(pr: int, head_sha: str, title: str) -> None:
    with _ledger_lock():
        ledger = _read_ledger_unlocked()
        _append_merge_entry_unlocked(ledger, pr, head_sha, title)
        _write_ledger_unlocked(ledger)


def _append_merge_entry_unlocked(ledger: dict[str, Any], pr: int, head_sha: str, title: str) -> None:
    merge_sequence = _merge_sequence(ledger) + 1
    ledger["merges"].append(
        {
            "pr": pr,
            "head_sha": head_sha,
            "title": title,
            "merged_at": time.time(),
            "merge_sequence": merge_sequence,
        }
    )


def _record_merge_intent(pr: int, head_sha: str, title: str) -> None:
    with _ledger_lock():
        ledger = _read_ledger_unlocked()
        if not any(intent.get("pr") == pr and intent.get("head_sha") == head_sha for intent in ledger["merge_intents"]):
            ledger["merge_intents"].append({"pr": pr, "head_sha": head_sha, "title": title, "intent_at": time.time()})
            _write_ledger_unlocked(ledger)


def _complete_merge_intent(pr: int, head_sha: str) -> None:
    with _ledger_lock():
        ledger = _read_ledger_unlocked()
        intents = ledger["merge_intents"]
        matching = [intent for intent in intents if intent.get("pr") == pr and intent.get("head_sha") == head_sha]
        if not matching:
            return
        intent = matching[0]
        if not any(entry.get("pr") == pr and entry.get("head_sha") == head_sha for entry in ledger["merges"]):
            _append_merge_entry_unlocked(ledger, pr, head_sha, str(intent["title"]))
        ledger["merge_intents"] = [item for item in intents if item is not intent]
        _write_ledger_unlocked(ledger)


def _reconcile_merge_intents() -> None:
    """Resolve durable pre-merge intents against GitHub after a restart."""
    ledger = _read_ledger()
    for intent in list(ledger["merge_intents"]):
        try:
            info = _gh_json(["pr", "view", str(intent["pr"]), "--json", "state,mergeCommit"])
        except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
            raise LedgerStateError(f"could not reconcile merge intent for PR #{intent['pr']}: {exc}") from exc
        merge_commit = info.get("mergeCommit")
        if info.get("state") != "MERGED" or not isinstance(merge_commit, dict) or not merge_commit.get("oid"):
            raise LedgerStateError(f"unresolved durable merge intent for PR #{intent['pr']}")
        _complete_merge_intent(int(intent["pr"]), str(intent["head_sha"]))
        if any(
            item.get("pr") == intent["pr"] and item.get("head_sha") == intent["head_sha"]
            for item in _read_ledger()["merge_intents"]
        ):
            raise LedgerStateError(f"merge intent for PR #{intent['pr']} was not durably reconciled")


def _pending_prs_since_last_full_verify(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    last_verify = ledger.get("last_full_verify") or {}
    scope = last_verify.get("verification_scope")
    last_verify_at = (
        last_verify.get("verification_started_at", last_verify.get("at", 0.0))
        if last_verify.get("accepted") is True
        and last_verify.get("exit_code") == 0
        and last_verify.get("release_baseline_allowed") is True
        and scope == VerificationScope.RELEASE_BASELINE.value
        else 0.0
    )
    snapshot_sequence = last_verify.get("merge_sequence")
    return [
        entry
        for entry in ledger.get("merges", [])
        if entry.get("merged_at", 0.0) > last_verify_at
        or (
            isinstance(snapshot_sequence, int)
            and not isinstance(snapshot_sequence, bool)
            and isinstance(entry.get("merge_sequence"), int)
            and entry["merge_sequence"] > snapshot_sequence
        )
    ]


def _receipt_is_fresh_for_head(pr: int, head_sha: str, max_age_s: int) -> bool:
    receipt = merge_gate._read_json_object(merge_gate._receipt_path(pr))
    if receipt is None:
        return False
    if receipt.get("head_sha") != head_sha:
        return False
    age_s = time.time() - float(receipt.get("recorded_at", 0))
    return bool(age_s <= max_age_s)


def _fetched_merged_default_branch_sha(pr: int) -> str | None:
    """Fetch the default branch and return its post-merge commit only."""
    try:
        branch = _default_branch_name()
        if branch is None:
            return None
        merged = _gh_json(["pr", "view", str(pr), "--json", "state,mergeCommit"])
        merge_commit = merged.get("mergeCommit")
        merge_sha = merge_commit.get("oid") if isinstance(merge_commit, dict) else None
        if merged.get("state") != "MERGED" or not isinstance(merge_sha, str) or not merge_sha:
            return None
        fetch = subprocess.run(
            ["git", "fetch", "origin", branch],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if fetch.returncode != 0:
            return None
        target = subprocess.run(
            ["git", "rev-parse", "FETCH_HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if target.returncode != 0:
            return None
        target_sha = target.stdout.strip()
        if not target_sha:
            return None
        included = subprocess.run(
            ["git", "merge-base", "--is-ancestor", merge_sha, target_sha],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return target_sha if included.returncode == 0 else None
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return None


def _default_branch_name() -> str | None:
    repo = _gh_json(["repo", "view", "--json", "defaultBranchRef"])
    default_ref = repo.get("defaultBranchRef")
    branch = default_ref.get("name") if isinstance(default_ref, dict) else None
    return branch if isinstance(branch, str) and branch else None


def _fetched_current_default_branch_sha() -> str | None:
    """Fetch and return the exact current default-branch commit."""
    try:
        branch = _default_branch_name()
        if branch is None:
            return None
        fetch = subprocess.run(["git", "fetch", "origin", branch], capture_output=True, text=True, timeout=120)
        if fetch.returncode != 0:
            return None
        target = subprocess.run(["git", "rev-parse", "FETCH_HEAD"], capture_output=True, text=True, timeout=15)
        if target.returncode != 0 or not target.stdout.strip():
            return None
        return target.stdout.strip()
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return None


def _terminal_verify_snapshot() -> tuple[dict[str, Any], float, int]:
    with _ledger_lock():
        ledger = _read_ledger_unlocked()
        started_at = time.time()
        return ledger, started_at, _merge_sequence(ledger)


def _remove_detached_worktree(repo_root: Path, worktree: Path) -> bool:
    removal = subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )
    if removal.returncode != 0:
        print(
            f"REFUSING terminal verify: failed to unregister detached worktree {worktree}: "
            f"{removal.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return False
    return True


def _run_post_merge_terminal_verify(
    command: str,
    target_sha: str,
    *,
    ledger_snapshot: tuple[dict[str, Any], float, int] | None = None,
) -> int:
    """Run terminal verification in a detached worktree at the fetched target."""
    repo_root = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="polylogue-merge-terminal-") as raw_worktree:
        worktree = Path(raw_worktree)
        add = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), target_sha],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
        if add.returncode != 0:
            print(f"REFUSING terminal verify: could not materialize fetched target {target_sha[:8]}", file=sys.stderr)
            _remove_detached_worktree(repo_root, worktree)
            return 1
        result = cmd_record_full_verify(
            command,
            target_sha=target_sha,
            cwd=worktree,
            execution_root=worktree,
            ledger_snapshot=ledger_snapshot,
        )
        return result if _remove_detached_worktree(repo_root, worktree) else 1


def cmd_merge(
    pr: int,
    *,
    command: str,
    max_age_s: int,
    poll_rounds: int,
    poll_interval_s: int,
    dry_run: bool,
    with_verify: bool,
    verify_command: str,
) -> int:
    try:
        info = _gh_json(["pr", "view", str(pr), "--json", "headRefOid,title,state"])
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        print(f"REFUSING to merge PR #{pr}: gh pr view failed: {exc}", file=sys.stderr)
        return 1

    if info.get("state") != "OPEN":
        print(f"REFUSING to merge PR #{pr}: state is {info.get('state')!r}, not OPEN", file=sys.stderr)
        return 1

    head_sha = info["headRefOid"]

    if not _receipt_is_fresh_for_head(pr, head_sha, max_age_s):
        print(
            f"no fresh merge-gate receipt for PR #{pr} @ {head_sha[:8]} -- recording one now via {command!r}",
            file=sys.stderr,
        )
        record_exit = merge_gate.cmd_record(pr, command)
        if record_exit != 0:
            print(f"REFUSING to merge PR #{pr}: recording verification failed (exit {record_exit})", file=sys.stderr)
            return record_exit

    check_exit = merge_gate.cmd_check(
        pr,
        max_age_s=max_age_s,
        poll_rounds=poll_rounds,
        poll_interval_s=poll_interval_s,
        as_json=False,
    )
    if check_exit != 0:
        print(f"REFUSING to merge PR #{pr}: merge-gate check BLOCKed (see reasons above)", file=sys.stderr)
        return check_exit

    clean_title = clean_merge_title(info.get("title", ""), pr)

    if dry_run:
        print(f"PR #{pr} @ {head_sha[:8]}: merge-gate OK -- dry-run, not merging (title would be {clean_title!r})")
        return 0

    try:
        _record_merge_intent(pr, head_sha, clean_title)
    except LedgerStateError as exc:
        print(f"REFUSING to merge PR #{pr}: could not durably record merge intent: {exc}", file=sys.stderr)
        return 1

    merge_result = subprocess.run(
        [
            "gh",
            "pr",
            "merge",
            str(pr),
            "--squash",
            "--match-head-commit",
            head_sha,
            "--subject",
            clean_title,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if merge_result.returncode != 0:
        print(f"gh pr merge failed: {merge_result.stderr.strip()[:500]}", file=sys.stderr)
        return merge_result.returncode

    print(f"merged PR #{pr} @ {head_sha[:8]}: {clean_title!r}")
    try:
        _complete_merge_intent(pr, head_sha)
    except LedgerStateError as exc:
        print(f"REFUSING to continue: merge-train ledger is not durably writable: {exc}", file=sys.stderr)
        return 1

    if with_verify:
        try:
            ledger_snapshot = _terminal_verify_snapshot()
        except LedgerStateError as exc:
            print(f"REFUSING terminal verify: {exc}", file=sys.stderr)
            return 1
        target_sha = _fetched_merged_default_branch_sha(pr)
        if target_sha is None:
            print(
                "REFUSING terminal verify: the fetched default branch does not prove this squash merge is included",
                file=sys.stderr,
            )
            return 1
        print(f"running post-merge broad verify (merge-train terminal step): {verify_command!r}")
        return _run_post_merge_terminal_verify(verify_command, target_sha, ledger_snapshot=ledger_snapshot)

    print(
        "REMINDER: this merge-train's terminal ledger step (one full-suite verify since the last "
        "merge) is not yet recorded -- run `devtools workspace merge record-full-verify "
        '--command "devtools verify --all"` before declaring the train done, or check '
        "`devtools workspace merge train-status`."
    )
    return 0


def cmd_train_status(as_json: bool) -> int:
    try:
        _reconcile_merge_intents()
        ledger = _read_ledger()
    except LedgerStateError as exc:
        print(f"merge-train REFUSING clean status: {exc}", file=sys.stderr)
        return 1
    pending = _pending_prs_since_last_full_verify(ledger)
    ok = not pending

    if as_json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "last_full_verify": ledger.get("last_full_verify"),
                    "pending_prs": pending,
                },
                indent=2,
            )
        )
        return 0 if ok else 1

    if ok:
        print("merge-train OK: no PRs merged since the last recorded full-suite verify")
        return 0

    print(f"merge-train INCOMPLETE: {len(pending)} PR(s) merged since the last full-suite verify:")
    for entry in pending:
        print(f"  PR #{entry['pr']} @ {entry['head_sha'][:8]}: {entry['title']}")
    print(
        'Run `devtools workspace merge record-full-verify --command "devtools verify --all"` '
        "(or the narrower agreed selection) before declaring this merge-train session done -- "
        "per-PR CI skips the heavy suite, so nothing else will catch a master-red class only "
        "visible on the merged whole."
    )
    return 1


def cmd_record_full_verify(
    command: str,
    *,
    target_sha: str | None = None,
    cwd: Path | None = None,
    execution_root: Path | None = None,
    ledger_snapshot: tuple[dict[str, Any], float, int] | None = None,
) -> int:
    argv = shlex.split(command)
    if not argv:
        print("REFUSING: --command is empty after splitting", file=sys.stderr)
        return 2
    if target_sha is None:
        print("REFUSING: terminal verification has no fetched merged-master target", file=sys.stderr)
        return 1
    try:
        snapshot = ledger_snapshot or _terminal_verify_snapshot()
    except LedgerStateError as exc:
        print(f"REFUSING to run terminal verification: {exc}", file=sys.stderr)
        return 1
    _snapshot_ledger, verification_started_at, merge_sequence = snapshot
    if execution_root is not None:
        argv = ["direnv", "exec", str(execution_root), *argv]
    started = verification_started_at
    try:
        result = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    except OSError as exc:
        print(f"REFUSING: could not run {command!r}: {exc}", file=sys.stderr)
        return 2
    duration_s = round(time.time() - started, 2)
    release_allowed = merge_gate._release_baseline_permission(result.stdout)
    verification_scope = merge_gate._verification_scope(result.stdout)
    terminal_authorization = merge_gate._terminal_authorization(result.stdout)
    try:
        structured = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        structured = None
    verified_head = structured.get("git_head") if isinstance(structured, dict) else None
    accepted = (
        result.returncode == 0
        and release_allowed is True
        and verification_scope == VerificationScope.RELEASE_BASELINE.value
        and verified_head == target_sha
    )

    try:
        with _ledger_lock():
            ledger = _read_ledger_unlocked()
            ledger["last_full_verify"] = {
                "command": command,
                "exit_code": result.returncode,
                "duration_s": duration_s,
                "at": verification_started_at,
                "verification_started_at": verification_started_at,
                "verification_scope": verification_scope,
                "release_baseline_allowed": release_allowed,
                "terminal_authorization": terminal_authorization,
                "verified_head_sha": verified_head,
                "target_sha": target_sha,
                "merged_master_sha": target_sha,
                "merge_sequence": merge_sequence,
                "accepted": accepted,
            }
            _write_ledger_unlocked(ledger)
    except LedgerStateError as exc:
        print(f"REFUSING to record terminal verification: {exc}", file=sys.stderr)
        return 1

    print(
        f"recorded merge-train terminal verify: {command!r} exit={result.returncode} "
        f"release_baseline_allowed={release_allowed!r} accepted={accepted} ({duration_s}s)"
    )
    if not accepted and result.returncode == 0:
        print(
            "POST-MERGE BROAD VERIFY DID NOT GRANT typed release-baseline authority for the selected target; "
            "train-status remains incomplete.",
            file=sys.stderr,
        )
        if verified_head != target_sha:
            print(
                f"terminal verify reported git_head={verified_head!r}, expected fetched target {target_sha}",
                file=sys.stderr,
            )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        print(
            "POST-MERGE BROAD VERIFY FAILED -- this is the master-red drift-latch class (an "
            "unrelated change breaking something only visible on the merged whole); investigate "
            "before merging further PRs in this train.",
            file=sys.stderr,
        )
    return result.returncode if result.returncode != 0 else (0 if accepted else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    merge_p = sub.add_parser("merge", help="Merge-gate check (auto-recording if needed) then squash-merge a PR")
    merge_p.add_argument("pr", type=int)
    merge_p.add_argument(
        "--command", default="devtools verify", help="Verification command to auto-record if no fresh receipt exists"
    )
    merge_p.add_argument("--max-age-s", type=int, default=merge_gate._DEFAULT_MAX_AGE_S)
    merge_p.add_argument("--poll-rounds", type=int, default=merge_gate._DEFAULT_POLL_ROUNDS)
    merge_p.add_argument("--poll-interval-s", type=int, default=merge_gate._DEFAULT_POLL_INTERVAL_S)
    merge_p.add_argument("--dry-run", action="store_true", help="Run every check but do not actually merge")
    merge_p.add_argument(
        "--with-verify",
        action="store_true",
        help="Immediately run and record the merge-train's terminal full-suite verify after merging",
    )
    merge_p.add_argument("--verify-command", default="devtools verify --all", help="Command for --with-verify")

    status_p = sub.add_parser(
        "train-status", help="Report whether the merge-train's terminal full-suite verify is recorded"
    )
    status_p.add_argument("--json", action="store_true", dest="as_json")

    record_p = sub.add_parser(
        "record-full-verify", help="Run and record the merge-train's terminal full-suite verify now"
    )
    record_p.add_argument("--command", default="devtools verify --all")

    args = parser.parse_args(argv)

    if args.action == "merge":
        return cmd_merge(
            args.pr,
            command=args.command,
            max_age_s=args.max_age_s,
            poll_rounds=args.poll_rounds,
            poll_interval_s=args.poll_interval_s,
            dry_run=args.dry_run,
            with_verify=args.with_verify,
            verify_command=args.verify_command,
        )
    if args.action == "train-status":
        return cmd_train_status(args.as_json)
    try:
        ledger_snapshot = _terminal_verify_snapshot()
    except LedgerStateError as exc:
        print(f"REFUSING terminal verify: {exc}", file=sys.stderr)
        return 1
    target_sha = _fetched_current_default_branch_sha()
    if target_sha is None:
        print("REFUSING terminal verify: could not fetch the current default branch", file=sys.stderr)
        return 1
    return _run_post_merge_terminal_verify(args.command, target_sha, ledger_snapshot=ledger_snapshot)


if __name__ == "__main__":
    sys.exit(main())
