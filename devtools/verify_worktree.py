"""verify-worktree: confirm an agent lane's claimed worktree is real and isolated.

On 2026-08-01 an `Agent` dispatch with `isolation: "worktree"` silently
produced NO worktree at all: the lane ran directly in the coordinator's main
checkout and left ~1700 lines of half-finished schema-regeneration output
sitting uncommitted in the live tree. The coordinator caught it only by
noticing a file's mtime was 42 seconds old. Nothing in the dispatch flow
verified that the isolation actually happened before the lane's diff was
trusted.

This command is that missing verification: run it against a just-spawned
lane's claimed working directory BEFORE trusting anything the lane reports.

Hard checks (exit 1 on any failure):

  - the path exists and is inside a git repository;
  - it is a LINKED worktree (its ``git-dir`` differs from the shared
    ``git-common-dir``) -- a path that resolves to the main checkout is
    exactly the isolation-escape failure mode this exists to catch;
  - its toplevel is not the main checkout's toplevel;
  - with ``--expect-branch``, the checked-out branch matches.

Advisory checks (reported, exit 0 unless ``--strict``):

  - uncommitted changes count (lanes must commit every chunk -- worktree
    auto-cleanup destroys uncommitted work);
  - beads staleness: if the worktree's ``.beads/issues.jsonl`` has an older
    max ``updated_at`` than the main checkout's, ANY ``bd`` invocation (or
    git-checkout hook) inside that worktree can reimport the frozen file and
    time-machine live bead state (polylogue-2ara, live incident 2026-08-01:
    5+ coordinator writes silently reverted, twice).

Usage:
    devtools workspace verify-worktree /path/to/worktree
    devtools workspace verify-worktree /path/to/worktree --expect-branch feature/x/y
    devtools workspace verify-worktree /path/to/worktree --json --strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    severity: str  # "hard" | "advisory"
    detail: str


@dataclass
class WorktreeReport:
    path: str
    checks: list[CheckResult] = field(default_factory=list)
    branch: str | None = None
    head: str | None = None
    main_checkout: str | None = None

    def add(self, name: str, ok: bool, severity: str, detail: str) -> None:
        self.checks.append(CheckResult(name=name, ok=ok, severity=severity, detail=detail))

    def hard_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "hard" and not c.ok]

    def advisory_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "advisory" and not c.ok]


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _max_updated_at(jsonl_path: Path) -> str | None:
    """Max ``updated_at`` across issue rows; None when unreadable/empty.

    ISO-8601 timestamps with a uniform offset compare correctly as strings;
    rows without the field are skipped. This is a freshness heuristic, not a
    merge engine -- devtools/bd_reimport_guard.py owns the real per-row logic.
    """
    try:
        raw = jsonl_path.read_text(errors="replace")
    except OSError:
        return None
    latest: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        updated = record.get("updated_at")
        if isinstance(updated, str) and (latest is None or updated > latest):
            latest = updated
    return latest


def inspect_worktree(path: Path, expect_branch: str | None = None) -> WorktreeReport:
    report = WorktreeReport(path=str(path))

    if not path.is_dir():
        report.add("path_exists", False, "hard", f"{path} is not an existing directory")
        return report
    report.add("path_exists", True, "hard", "directory exists")

    rev = _git(path, "rev-parse", "--absolute-git-dir", "--git-common-dir", "--show-toplevel")
    if rev.returncode != 0:
        report.add("inside_git_repo", False, "hard", f"not a git repository: {rev.stderr.strip()[:200]}")
        return report
    lines = rev.stdout.splitlines()
    if len(lines) < 3:
        report.add("inside_git_repo", False, "hard", f"unexpected rev-parse output: {rev.stdout!r}")
        return report
    git_dir = Path(lines[0]).resolve()
    raw_common = Path(lines[1])
    common_dir = raw_common.resolve() if raw_common.is_absolute() else (path / raw_common).resolve()
    toplevel = Path(lines[2]).resolve()
    report.add("inside_git_repo", True, "hard", f"toplevel {toplevel}")

    main_checkout = common_dir.parent
    report.main_checkout = str(main_checkout)

    if git_dir == common_dir:
        report.add(
            "linked_worktree",
            False,
            "hard",
            f"{path} is a MAIN checkout, not a linked worktree -- worktree isolation "
            "did not happen; do not trust this lane's diff against the live tree",
        )
    else:
        report.add("linked_worktree", True, "hard", f"linked worktree of {main_checkout}")

    if toplevel == main_checkout:
        report.add(
            "distinct_toplevel",
            False,
            "hard",
            f"worktree toplevel equals the main checkout toplevel ({main_checkout})",
        )
    else:
        report.add("distinct_toplevel", True, "hard", f"toplevel {toplevel} != main checkout {main_checkout}")

    branch_proc = _git(path, "branch", "--show-current")
    branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else None
    report.branch = branch or None
    if expect_branch is not None:
        if branch == expect_branch:
            report.add("branch_matches", True, "hard", f"on expected branch {expect_branch}")
        else:
            report.add(
                "branch_matches",
                False,
                "hard",
                f"expected branch {expect_branch!r}, found {branch!r}",
            )

    head_proc = _git(path, "log", "-1", "--format=%h %s")
    if head_proc.returncode == 0 and head_proc.stdout.strip():
        report.head = head_proc.stdout.strip()

    status_proc = _git(path, "status", "--porcelain")
    if status_proc.returncode == 0:
        dirty = [line for line in status_proc.stdout.splitlines() if line.strip()]
        report.add(
            "committed_state",
            not dirty,
            "advisory",
            f"{len(dirty)} uncommitted change(s) -- worktree auto-cleanup destroys uncommitted work"
            if dirty
            else "working tree clean",
        )

    wt_jsonl = toplevel / ".beads" / "issues.jsonl"
    main_jsonl = main_checkout / ".beads" / "issues.jsonl"
    if wt_jsonl.is_file() and main_jsonl.is_file() and wt_jsonl != main_jsonl:
        wt_latest = _max_updated_at(wt_jsonl)
        main_latest = _max_updated_at(main_jsonl)
        if wt_latest is not None and main_latest is not None and wt_latest < main_latest:
            report.add(
                "beads_freshness",
                False,
                "advisory",
                f"worktree .beads/issues.jsonl is STALE (max updated_at {wt_latest} < main {main_latest}); "
                "any bd invocation or checkout hook in this worktree can reimport it and revert "
                "live bead state (polylogue-2ara)",
            )
        else:
            report.add(
                "beads_freshness",
                True,
                "advisory",
                f"worktree jsonl max updated_at {wt_latest} >= main {main_latest}",
            )

    return report


def _render_text(report: WorktreeReport) -> str:
    lines = [f"worktree: {report.path}"]
    if report.branch:
        lines.append(f"branch: {report.branch}")
    if report.head:
        lines.append(f"head: {report.head}")
    if report.main_checkout:
        lines.append(f"main checkout: {report.main_checkout}")
    for check in report.checks:
        marker = "ok" if check.ok else ("FAIL" if check.severity == "hard" else "WARN")
        lines.append(f"  [{marker}] {check.name}: {check.detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="The lane's claimed worktree directory")
    parser.add_argument("--expect-branch", metavar="BRANCH", help="Fail unless this branch is checked out")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit a JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Advisory findings (dirty tree, stale beads jsonl) also fail the check",
    )
    args = parser.parse_args(argv)

    report = inspect_worktree(Path(args.path), expect_branch=args.expect_branch)

    failed = bool(report.hard_failures()) or (args.strict and bool(report.advisory_failures()))
    if args.as_json:
        payload = asdict(report)
        payload["ok"] = not failed
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(report))
        print(f"verdict: {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
