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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from devtools.verify_worktree import _git, inspect_worktree


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


def _validate_target(target: Path, report: IntegrationReport) -> str | None:
    inspection = inspect_worktree(target)
    hard_failures = inspection.hard_failures()
    if hard_failures:
        return "; ".join(check.detail for check in hard_failures)
    branch = inspection.branch
    report.branch = branch
    if branch in {None, "master", "refs/heads/master"}:
        return "target must be on a named non-master branch"
    status = _git(target, "status", "--porcelain")
    if status.returncode:
        return f"could not inspect target status: {status.stderr.strip()}"
    if status.stdout.strip():
        return "target worktree must be clean before integration"
    return None


def integrate(
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
    target_error = _validate_target(target, report)
    if target_error:
        return _fail(report, target_error)
    if not source_refs and not explicit_commits:
        return _fail(report, "provide at least one source ref or explicit commit SHA")

    planned: list[str] = []
    for source_ref in source_refs:
        commits, error = _source_commits(target, source_ref)
        if error:
            return _fail(report, error)
        planned.extend(commits)
    for commit in explicit_commits:
        resolved = _commit_sha(target, commit)
        if resolved is None:
            return _fail(report, f"explicit commit {commit!r} does not resolve to a commit")
        planned.append(resolved)
    report.planned_commits = planned

    for commit in planned:
        result = _git(target, "cherry-pick", commit)
        if result.returncode:
            conflict_head = _git(target, "rev-parse", "--verify", "CHERRY_PICK_HEAD").stdout.strip()
            report.conflict = bool(conflict_head)
            report.conflict_head = conflict_head or None
            report.status = "conflict" if report.conflict else "error"
            report.error = (result.stderr or result.stdout).strip() or "git cherry-pick failed"
            return report
        report.applied_commits.append(commit)

    report.status = "applied"
    return report


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
    if report.error:
        lines.append(f"error: {report.error}")
    return "\n".join(lines)


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
    refs = [*args.source_refs, *args.source_options]
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
