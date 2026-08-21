"""Integrate ordered lane commits into an explicit linked worktree.

This is deliberately a small Git boundary: it validates a clean linked target,
derives only unambiguous ancestry ranges from source refs, and delegates the
actual application to ordinary ``git cherry-pick``. A conflict is left in place
for the operator to resolve; this command never runs ``cherry-pick --abort`` or
attempts a resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from devtools.verify_worktree import _git, inspect_worktree

_ACTIVE_MARKERS = ("CHERRY_PICK_HEAD", "MERGE_HEAD", "REVERT_HEAD", "REBASE_HEAD")
_GIT_SELECTION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")


@contextmanager
def _sanitized_git_environment() -> Iterator[None]:
    """Bind every subprocess Git invocation to its ``-C`` worktree metadata."""
    saved = {name: os.environ.get(name) for name in _GIT_SELECTION_VARS}
    for name in _GIT_SELECTION_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class IntegrationReport:
    target: str
    branch: str | None = None
    source_refs: list[str] = field(default_factory=list)
    explicit_commits: list[str] = field(default_factory=list)
    planned_commits: list[str] = field(default_factory=list)
    applied_commits: list[str] = field(default_factory=list)
    status: str = "blocked"
    conflict: bool = False
    conflict_head: str | None = None
    empty_pick: bool = False
    active_operation: str | None = None
    error: str | None = None


def _fail(report: IntegrationReport, detail: str) -> IntegrationReport:
    report.status = "blocked"
    report.error = detail
    return report


def _commit_sha(target: Path, ref: str) -> str | None:
    result = _git(target, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode:
        return None
    value = result.stdout.strip()
    return value or None


def _source_commits(target: Path, source_ref: str) -> tuple[list[str], str | None]:
    source = _commit_sha(target, source_ref)
    if source is None:
        return [], f"source ref {source_ref!r} does not resolve to a commit"
    target_head = _commit_sha(target, "HEAD")
    if target_head is None:
        return [], "target HEAD does not resolve to a commit"
    merge_base = _git(target, "merge-base", target_head, source)
    if merge_base.returncode or merge_base.stdout.strip() != target_head:
        return [], (
            f"ancestry for source ref {source_ref!r} is ambiguous; target HEAD is not an ancestor; "
            "provide explicit --commit SHA values"
        )
    commits = _git(target, "rev-list", "--reverse", f"{target_head}..{source}")
    if commits.returncode:
        return [], f"could not derive commit range for source ref {source_ref!r}: {commits.stderr.strip()}"
    return [line for line in commits.stdout.splitlines() if line], None


def _git_path(target: Path, name: str) -> Path:
    raw = Path(_git(target, "rev-parse", "--git-path", name).stdout.strip())
    return raw if raw.is_absolute() else target / raw


def _active_operation(target: Path) -> str | None:
    """Return the first active Git operation marker, if any."""
    for marker in _ACTIVE_MARKERS:
        if _git_path(target, marker).exists():
            return marker
    if _git_path(target, "sequencer").is_dir():
        return "sequencer"
    for directory in ("rebase-apply", "rebase-merge"):
        if _git_path(target, directory).is_dir():
            return directory
    return None


def _validate_target(target: Path, report: IntegrationReport) -> tuple[Path | None, str | None]:
    inspection = inspect_worktree(target)
    hard_failures = inspection.hard_failures()
    if hard_failures:
        return None, "; ".join(check.detail for check in hard_failures)

    show_top = _git(target, "rev-parse", "--show-toplevel")
    if show_top.returncode:
        return None, f"could not determine linked worktree top-level: {show_top.stderr.strip()}"
    canonical = Path(show_top.stdout.strip()).resolve()
    if canonical != target:
        return None, f"--target must equal the linked worktree top-level ({canonical}), not a nested path"

    branch = inspection.branch
    report.branch = branch
    if branch in {None, "master", "refs/heads/master"}:
        return None, "target must be on a named non-master branch"
    active = _active_operation(canonical)
    if active is not None:
        return None, f"active Git operation ({active}) must be completed before integration"
    status = _git(canonical, "status", "--porcelain")
    if status.returncode:
        return None, f"could not inspect target status: {status.stderr.strip()}"
    if status.stdout.strip():
        return None, "target worktree must be clean before integration"
    return canonical, None


def _append_unique(planned: list[str], seen: set[str], commits: Sequence[str]) -> None:
    for commit in commits:
        if commit not in seen:
            seen.add(commit)
            planned.append(commit)


def _operation_state_after_timeout(target: Path) -> str | None:
    """Inspect operation state after a Git timeout without classifying it as conflict."""
    try:
        return _active_operation(target)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _reconcile_cherry_pick_timeout(
    report: IntegrationReport, target: Path, commit: str, pre_head: str | None, timeout: float | None
) -> IntegrationReport:
    """Reconcile a timeout without claiming the pick is safe to retry."""
    report.active_operation = _operation_state_after_timeout(target)
    post_head = _commit_sha(target, "HEAD")
    report.conflict_head = _commit_sha(target, "CHERRY_PICK_HEAD")
    status = _git(target, "status", "--porcelain")
    unmerged = any(
        status_line[:2] in {"UU", "AA", "DD", "AU", "UA", "DU"} for status_line in status.stdout.splitlines()
    )
    if report.conflict_head and unmerged:
        report.conflict = True
        report.status = "indeterminate"
        report.error = f"git cherry-pick timed out after {timeout}s; content conflict remains; do not retry"
    elif post_head is not None and pre_head is not None and post_head != pre_head and report.active_operation is None:
        report.applied_commits.append(commit)
        report.status = "applied"
        report.error = f"git cherry-pick timed out after {timeout}s; target HEAD advanced and was reconciled as applied"
    else:
        report.status = "indeterminate"
        report.error = f"git cherry-pick timed out after {timeout}s; outcome is indeterminate; do not retry"
    return report


def _integrate(
    target: Path,
    source_refs: Sequence[str] = (),
    explicit_commits: Sequence[str] = (),
) -> IntegrationReport:
    """Validate and apply an ordered integration plan to ``target``."""
    target = target.expanduser().resolve()
    report = IntegrationReport(
        target=str(target),
        source_refs=list(source_refs),
        explicit_commits=list(explicit_commits),
    )
    try:
        canonical, target_error = _validate_target(target, report)
        if target_error:
            return _fail(report, target_error)
        assert canonical is not None
        target = canonical
        report.target = str(target)
        if not source_refs and not explicit_commits:
            return _fail(report, "provide at least one source ref or explicit commit SHA")

        planned: list[str] = []
        seen: set[str] = set()
        for source_ref in source_refs:
            commits, error = _source_commits(target, source_ref)
            if error:
                return _fail(report, error)
            _append_unique(planned, seen, commits)
        for commit in explicit_commits:
            resolved = _commit_sha(target, commit)
            if resolved is None:
                return _fail(report, f"explicit commit {commit!r} does not resolve to a commit")
            _append_unique(planned, seen, [resolved])
        report.planned_commits = planned

        for commit in planned:
            pre_head = _commit_sha(target, "HEAD")
            try:
                result = _git(target, "cherry-pick", commit)
            except subprocess.TimeoutExpired as exc:
                return _reconcile_cherry_pick_timeout(report, target, commit, pre_head, exc.timeout)
            if result.returncode:
                conflict_head = _git(target, "rev-parse", "--verify", "CHERRY_PICK_HEAD").stdout.strip()
                report.conflict_head = conflict_head or None
                status = _git(target, "status", "--porcelain").stdout
                report.empty_pick = bool(conflict_head) and not status.strip()
                report.conflict = bool(conflict_head) and not report.empty_pick
                if report.empty_pick:
                    report.status = "empty"
                    report.error = "cherry-pick produced no changes; stopped without skipping or aborting"
                else:
                    report.status = "conflict" if report.conflict else "error"
                    report.error = (result.stderr or result.stdout).strip() or "git cherry-pick failed"
                return report
            report.applied_commits.append(commit)

        report.status = "applied"
        return report
    except subprocess.TimeoutExpired as exc:
        report.status = "indeterminate"
        report.active_operation = _operation_state_after_timeout(target)
        report.error = f"git command timed out after {exc.timeout}s; outcome is indeterminate; do not retry"
        return report


def integrate(
    target: Path,
    source_refs: Sequence[str] = (),
    explicit_commits: Sequence[str] = (),
) -> IntegrationReport:
    """Run integration with repository-selection environment sanitized."""
    with _sanitized_git_environment():
        return _integrate(target, source_refs, explicit_commits)


def _render_text(report: IntegrationReport) -> str:
    lines = [f"target: {report.target}"]
    if report.branch:
        lines.append(f"branch: {report.branch}")
    lines.append(f"status: {report.status}")
    lines.append(f"planned commits: {len(report.planned_commits)}")
    lines.append(f"applied commits: {len(report.applied_commits)}")
    if report.applied_commits:
        lines.append("applied SHAs:")
        lines.extend(f"  {sha}" for sha in report.applied_commits)
    if report.conflict:
        lines.append(f"conflict: CHERRY_PICK_HEAD={report.conflict_head}")
        lines.append("conflict state left in target worktree; no automatic resolution or abort performed")
    if report.empty_pick:
        lines.append("empty pick: cherry-pick produced no changes; state left in target worktree")
    if report.active_operation:
        lines.append(f"active operation after timeout: {report.active_operation}")
    if report.error:
        lines.append(f"error: {report.error}")
    return "\n".join(lines)


def _report_for_mixed_sources(target: Path, positional: Sequence[str], options: Sequence[str]) -> IntegrationReport:
    report = IntegrationReport(target=str(target.expanduser().resolve()), source_refs=[*positional, *options])
    return _fail(report, "cannot mix positional source refs with --source options; choose one form")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, type=Path, help="Explicit linked integration worktree")
    parser.add_argument("source_refs", nargs="*", help="Ordered source refs whose unambiguous ranges should be applied")
    parser.add_argument(
        "--source", dest="source_options", action="append", default=[], help="Ordered source ref (repeatable)"
    )
    parser.add_argument(
        "--commit", dest="commits", action="append", default=[], help="Explicit commit SHA (repeatable)"
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit a JSON report")
    args = parser.parse_args(argv)
    if args.source_refs and args.source_options:
        report = _report_for_mixed_sources(args.target, args.source_refs, args.source_options)
    else:
        refs = args.source_refs or args.source_options
        report = integrate(args.target, refs, args.commits)
    payload = asdict(report)
    payload["ok"] = report.status == "applied"
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(report))
    return 0 if report.status == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
