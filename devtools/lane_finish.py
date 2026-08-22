"""Package a completed lane for assimilation and release its worktree lock.

The command is the final action in an implementation lane. It refuses dirty,
detached, main-checkout, and master worktrees; records the exact commits and
changed paths a coordinator must assimilate; then unlocks only the invoking
worktree so normal worktree garbage collection can remove it after integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from devtools.verify_worktree import _git, inspect_worktree


@dataclass
class LaneHandoff:
    worktree: str
    branch: str | None = None
    head: str | None = None
    base: str | None = None
    commits: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    status: str = "blocked"
    unlocked: bool = False
    error: str | None = None


def _resolve_base(worktree: Path, requested: str | None) -> str | None:
    candidates = [requested] if requested else ["origin/master", "master"]
    for candidate in candidates:
        if candidate is None:
            continue
        probe = _git(worktree, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")
        if probe.returncode == 0:
            return candidate
    return None


def _is_locked(worktree: Path) -> bool:
    listing = _git(worktree, "worktree", "list", "--porcelain")
    if listing.returncode != 0:
        return False
    target = str(worktree.resolve())
    for block in listing.stdout.split("\n\n"):
        lines = block.splitlines()
        if f"worktree {target}" in lines:
            return any(line == "locked" or line.startswith("locked ") for line in lines)
    return False


def finish_lane(worktree: Path, *, base: str | None = None) -> LaneHandoff:
    worktree = worktree.expanduser().resolve()
    report = LaneHandoff(worktree=str(worktree))
    inspection = inspect_worktree(worktree)
    hard_failures = inspection.hard_failures()
    if hard_failures:
        report.error = "; ".join(check.detail for check in hard_failures)
        return report
    report.branch = inspection.branch
    if report.branch in {None, "master", "refs/heads/master"}:
        report.error = "lane-finish requires a named non-master branch"
        return report

    status = _git(worktree, "status", "--porcelain")
    if status.returncode != 0:
        report.error = f"could not inspect lane status: {status.stderr.strip()}"
        return report
    if status.stdout.strip():
        report.error = "lane has uncommitted changes; commit every logical chunk before lane-finish"
        return report

    head = _git(worktree, "rev-parse", "HEAD")
    if head.returncode != 0:
        report.error = f"could not resolve lane HEAD: {head.stderr.strip()}"
        return report
    report.head = head.stdout.strip()
    report.base = _resolve_base(worktree, base)
    if report.base is None:
        report.error = f"base ref {base!r} does not resolve" if base else "neither origin/master nor master resolves"
        return report

    commits = _git(worktree, "rev-list", "--reverse", f"{report.base}..HEAD")
    paths = _git(worktree, "diff", "--name-only", f"{report.base}...HEAD")
    if commits.returncode != 0 or paths.returncode != 0:
        detail = commits.stderr.strip() or paths.stderr.strip()
        report.error = f"could not package lane delta: {detail}"
        return report
    report.commits = [line for line in commits.stdout.splitlines() if line]
    report.changed_paths = [line for line in paths.stdout.splitlines() if line]
    report.status = "ready-for-assimilation" if report.commits else "no-changes"

    if _is_locked(worktree):
        unlock = _git(worktree, "worktree", "unlock", str(worktree))
        if unlock.returncode != 0:
            report.status = "blocked"
            report.error = f"handoff packaged but worktree unlock failed: {unlock.stderr.strip()}"
            return report
        report.unlocked = True
    return report
