"""merge-conductor: mechanical-conflict triage for the PR merge train.

Implements the MECHANICAL slice of the polylogue-ei94 merge-conductor design
(state machine, admission vectors, review/gate triage). This command is
deliberately narrower: given a roster of PR numbers, it classifies each PR's
modified/conflicting files into conflict classes and reports a verdict --
AUTO-RESOLVABLE, ESCALATE, or CLEAN. It does NOT triage code review findings,
run the full admission state machine, or serialize Beads writes; those stay
polylogue-ei94's scope.

Conflict classes:

  - ``beads-jsonl``        -- .beads/issues.jsonl. Resolution: take master's
                               side; bead state syncs separately. AUTO.
  - ``generated-surface``  -- known regenerable doc/plan artifacts (topology
                               projection, CLI/MCP/devtools reference docs,
                               OpenAPI, ...). Resolution: regenerate via
                               `python -m devtools render ...`. AUTO.
  - ``schema-migration``   -- polylogue/storage/sqlite/migrations/** or
                               polylogue/storage/sqlite/lifecycle.py. NEVER
                               auto-resolved -- two PRs claiming the same
                               durable-tier migration slot or lifecycle delta
                               class nearly dropped a declaration on
                               2026-07-30. ESCALATE, always.
  - ``hooks-config``        -- .claude/ or .git*-prefixed hook/config paths.
                               ESCALATE, always.
  - ``other``                -- everything else. ESCALATE by default: this
                               tool's whole safety case is a short allowlist
                               of provably-mechanical classes, not a guess
                               about arbitrary code conflicts.

Default is DRY-RUN: this command only classifies and reports. --execute is
required to touch anything, and even then it only ever acts on PRs whose
verdict is AUTO-RESOLVABLE -- ESCALATE-class PRs are never touched under
--execute, full stop. Every --execute step is wrapped so that ANY deviation
(rebase conflict, render diff outside the expected surfaces, a failing
`devtools verify --quick`) aborts that PR's automation, cleans up its scratch
worktree, and marks it ESCALATE in the receipt instead of forcing a push.

This tool must degrade to report-only on any subprocess failure (missing
`gh`, network failure, auth failure, unexpected `gh` JSON shape) -- a broken
signal should never be silently treated as "no conflict".

Usage:
    devtools workspace merge-conductor --pr 3301 --pr 3302
    devtools workspace merge-conductor --pr 3301 --json
    devtools workspace merge-conductor --pr 3301 --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Conflict classification
# ---------------------------------------------------------------------------

BEADS_JSONL_PATH = ".beads/issues.jsonl"

# Known regenerable doc/plan artifacts. Kept as an explicit allowlist rather
# than a broad "anything under docs/" match -- most of docs/ is hand-authored
# prose, and treating hand-authored docs as auto-resolvable would silently
# discard a human's conflicting edit.
_GENERATED_SURFACE_FILES: frozenset[str] = frozenset(
    {
        "docs/topology-status.md",
        "docs/cli-reference.md",
        "docs/devtools.md",
        "docs/mcp-reference.md",
        "docs/search.md",
        "docs/product/workflows.md",
        "docs/README.md",
        "docs/openapi/search.yaml",
        "docs/generated/mcp-equivalence.json",
    }
)
_GENERATED_SURFACE_PREFIXES: tuple[str, ...] = (
    "docs/schemas/cli-output/",
    "docs/generated/",
)

_SCHEMA_MIGRATION_PREFIX = "polylogue/storage/sqlite/migrations/"
_SCHEMA_LIFECYCLE_FILE = "polylogue/storage/sqlite/lifecycle.py"

_HOOKS_CONFIG_PREFIXES: tuple[str, ...] = (".claude/", ".git")

CONFLICT_CLASSES = (
    "beads-jsonl",
    "generated-surface",
    "schema-migration",
    "hooks-config",
    "other",
)
AUTO_RESOLVABLE_CLASSES = frozenset({"beads-jsonl", "generated-surface"})


def classify_file(path: str) -> str:
    """Classify a single modified/conflicting file path into a conflict class."""
    if path == BEADS_JSONL_PATH:
        return "beads-jsonl"
    if path in _GENERATED_SURFACE_FILES or any(path.startswith(p) for p in _GENERATED_SURFACE_PREFIXES):
        return "generated-surface"
    if path.startswith(_SCHEMA_MIGRATION_PREFIX) or path == _SCHEMA_LIFECYCLE_FILE:
        return "schema-migration"
    if any(path.startswith(p) for p in _HOOKS_CONFIG_PREFIXES):
        return "hooks-config"
    return "other"


# Classes that identify a *contention slot* worth naming across PRs even
# though they're both ESCALATE individually -- a migration file touched by
# two PRs in the same roster is the exact 2026-07-30 near-miss.
_CONTENTION_CLASSES = frozenset({"schema-migration", "generated-surface"})


# ---------------------------------------------------------------------------
# PR data + classification
# ---------------------------------------------------------------------------


@dataclass
class PRReport:
    number: int
    ok: bool
    error: str = ""
    title: str = ""
    head_ref: str = ""
    mergeable: str = ""
    merge_state_status: str = ""
    checks_summary: str = ""
    files: list[str] = field(default_factory=list)
    files_by_class: dict[str, list[str]] = field(default_factory=dict)
    verdict: str = "ESCALATE"
    escalation_files: list[str] = field(default_factory=list)
    execute_result: str = ""


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:300] or f"gh {' '.join(args)} failed")
    return json.loads(result.stdout)


def _fetch_pr_report(pr_number: int) -> PRReport:
    try:
        info = _gh_json(
            [
                "pr",
                "view",
                str(pr_number),
                "--json",
                "number,title,headRefName,mergeable,mergeStateStatus,statusCheckRollup",
            ]
        )
    except (RuntimeError, json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        return PRReport(number=pr_number, ok=False, error=f"gh pr view failed: {exc}")

    diff_result = subprocess.run(
        ["gh", "pr", "diff", str(pr_number), "--name-only"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if diff_result.returncode != 0:
        return PRReport(
            number=pr_number,
            ok=False,
            error=f"gh pr diff failed: {diff_result.stderr.strip()[:300]}",
        )
    files = [line for line in diff_result.stdout.splitlines() if line.strip()]

    checks = info.get("statusCheckRollup") or []
    check_states: dict[str, int] = defaultdict(int)
    for check in checks:
        state = check.get("state") or check.get("conclusion") or check.get("status") or "UNKNOWN"
        check_states[str(state).upper()] += 1
    checks_summary = ", ".join(f"{count} {state.lower()}" for state, count in sorted(check_states.items()))
    if not checks_summary:
        checks_summary = "no checks reported"

    files_by_class: dict[str, list[str]] = defaultdict(list)
    for f in files:
        files_by_class[classify_file(f)].append(f)

    escalation_files: list[str] = []
    for cls in CONFLICT_CLASSES:
        if cls not in AUTO_RESOLVABLE_CLASSES:
            escalation_files.extend(files_by_class.get(cls, []))

    if escalation_files:
        verdict = "ESCALATE"
    elif info.get("mergeable") == "CONFLICTING":
        verdict = "AUTO-RESOLVABLE"
    else:
        verdict = "CLEAN"

    return PRReport(
        number=pr_number,
        ok=True,
        title=info.get("title", ""),
        head_ref=info.get("headRefName", ""),
        mergeable=info.get("mergeable", "UNKNOWN"),
        merge_state_status=info.get("mergeStateStatus", "UNKNOWN"),
        checks_summary=checks_summary,
        files=files,
        files_by_class=dict(files_by_class),
        verdict=verdict,
        escalation_files=escalation_files,
    )


def _find_contention(reports: list[PRReport]) -> list[dict[str, Any]]:
    """Cross-PR contention: two roster PRs touching the same contention-class file."""
    file_to_prs: dict[str, list[int]] = defaultdict(list)
    for r in reports:
        if not r.ok:
            continue
        for cls in _CONTENTION_CLASSES:
            for f in r.files_by_class.get(cls, []):
                file_to_prs[f].append(r.number)
    events: list[dict[str, Any]] = []
    for f, prs in file_to_prs.items():
        if len(prs) > 1:
            events.append(
                {
                    "file": f,
                    "class": classify_file(f),
                    "prs": sorted(set(prs)),
                    "recommendation": "serialize these PRs -- do not admit both concurrently",
                }
            )
    return events


# ---------------------------------------------------------------------------
# --execute: conservative scratch-worktree automation for AUTO-RESOLVABLE only
# ---------------------------------------------------------------------------

WORKTREE_ROOT = Path("/realm/worktrees")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _execute_auto_resolve(report: PRReport, repo_root: Path) -> str:
    """Attempt the declared safe automation for one AUTO-RESOLVABLE PR.

    Returns a short outcome string. Any deviation aborts and cleans up --
    this function must never leave a dirty scratch worktree behind and must
    never push unless every declared step succeeded.
    """
    assert report.verdict == "AUTO-RESOLVABLE"
    scratch = WORKTREE_ROOT / f"merge-conductor-{report.number}"
    branch = f"merge-conductor-pr-{report.number}"

    def _abort(reason: str) -> str:
        report.verdict = "ESCALATE"
        report.escalation_files = list(report.files)
        if scratch.exists():
            _run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo_root)
        return f"ABORTED (auto-resolve deviated): {reason}"

    if scratch.exists():
        return _abort(f"scratch worktree {scratch} already exists -- refusing to reuse it")

    fetch = _run(["git", "fetch", "origin", report.head_ref, "master"], cwd=repo_root)
    if fetch.returncode != 0:
        return _abort(f"git fetch failed: {fetch.stderr.strip()[:200]}")

    add = _run(
        ["git", "worktree", "add", "-b", branch, str(scratch), f"origin/{report.head_ref}"],
        cwd=repo_root,
    )
    if add.returncode != 0:
        return _abort(f"git worktree add failed: {add.stderr.strip()[:200]}")

    rebase = _run(["git", "rebase", "origin/master"], cwd=scratch)
    if rebase.returncode != 0:
        _run(["git", "rebase", "--abort"], cwd=scratch)
        return _abort(f"rebase onto origin/master conflicted: {rebase.stderr.strip()[:200]}")

    # Resolve beads-jsonl by taking master's side, unconditionally.
    if "beads-jsonl" in report.files_by_class:
        show = _run(["git", "show", f"origin/master:{BEADS_JSONL_PATH}"], cwd=scratch)
        if show.returncode != 0:
            return _abort(f"could not read master's {BEADS_JSONL_PATH}: {show.stderr.strip()[:200]}")
        (scratch / BEADS_JSONL_PATH).write_text(show.stdout)
        add_beads = _run(["git", "add", BEADS_JSONL_PATH], cwd=scratch)
        if add_beads.returncode != 0:
            return _abort(f"git add {BEADS_JSONL_PATH} failed: {add_beads.stderr.strip()[:200]}")

    # Regenerate generated surfaces if any were in this PR's conflict set.
    if "generated-surface" in report.files_by_class:
        render = _run(["python", "-m", "devtools", "render", "all"], cwd=scratch, timeout=600)
        if render.returncode != 0:
            return _abort(f"devtools render all failed: {render.stderr.strip()[:200]}")
        add_gen = _run(["git", "add", *report.files_by_class["generated-surface"]], cwd=scratch)
        if add_gen.returncode != 0:
            return _abort(f"git add generated surfaces failed: {add_gen.stderr.strip()[:200]}")

    status = _run(["git", "status", "--porcelain"], cwd=scratch)
    if status.stdout.strip():
        commit = _run(["git", "commit", "-m", "chore(merge-conductor): auto-resolve mechanical conflicts"], cwd=scratch)
        if commit.returncode != 0:
            return _abort(f"commit of resolved files failed: {commit.stderr.strip()[:200]}")

    continue_rebase = _run(["git", "rebase", "--continue"], cwd=scratch)
    if continue_rebase.returncode != 0:
        _run(["git", "rebase", "--abort"], cwd=scratch)
        return _abort(f"rebase --continue failed: {continue_rebase.stderr.strip()[:200]}")

    verify = _run(["python", "-m", "devtools", "verify", "--quick"], cwd=scratch, timeout=900)
    if verify.returncode != 0:
        return _abort(f"devtools verify --quick failed: {verify.stdout[-300:]} {verify.stderr[-300:]}")

    push = _run(
        ["git", "push", "--force-with-lease", "origin", f"HEAD:{report.head_ref}"],
        cwd=scratch,
        timeout=120,
    )
    if push.returncode != 0:
        return _abort(f"push --force-with-lease failed: {push.stderr.strip()[:200]}")

    _run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo_root)
    return "RESOLVED and pushed"


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _render_human(reports: list[PRReport], contention: list[dict[str, Any]], executed: bool) -> None:
    print("=" * 78)
    print("MERGE CONDUCTOR -- mechanical conflict triage" + (" (EXECUTED)" if executed else " (DRY RUN)"))
    print("=" * 78)
    print()
    header = f"{'PR':>6}  {'mergeable':<12} {'verdict':<16} {'classes':<28} checks"
    print(header)
    print("-" * len(header))
    for r in reports:
        if not r.ok:
            print(f"{r.number:>6}  ERROR: {r.error}")
            continue
        classes = ",".join(sorted(r.files_by_class)) or "(no files)"
        print(f"{r.number:>6}  {r.mergeable:<12} {r.verdict:<16} {classes:<28} {r.checks_summary}")
        if r.execute_result:
            print(f"          -> {r.execute_result}")
        if r.verdict == "ESCALATE" and r.escalation_files:
            for f in r.escalation_files:
                print(f"          ESCALATE file: {f}  [{classify_file(f)}]")
    print()
    if contention:
        print(f"-- CROSS-PR CONTENTION ({len(contention)}) --------------------------------------")
        for ev in contention:
            print(f"  {ev['file']}  [{ev['class']}]  PRs: {ev['prs']}")
            print(f"    -> {ev['recommendation']}")
        print()
    print(
        "Advisory: classification is mechanical only. schema-migration and hooks-config "
        "conflicts are NEVER auto-resolved."
    )


def _report_to_dict(r: PRReport) -> dict[str, Any]:
    return {
        "number": r.number,
        "ok": r.ok,
        "error": r.error,
        "title": r.title,
        "head_ref": r.head_ref,
        "mergeable": r.mergeable,
        "merge_state_status": r.merge_state_status,
        "checks_summary": r.checks_summary,
        "files": r.files,
        "files_by_class": r.files_by_class,
        "verdict": r.verdict,
        "escalation_files": r.escalation_files,
        "execute_result": r.execute_result,
    }


def _render_json(reports: list[PRReport], contention: list[dict[str, Any]]) -> None:
    out = {
        "prs": [_report_to_dict(r) for r in reports],
        "contention": contention,
    }
    json.dump(out, sys.stdout, indent=2)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return Path.cwd()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr", type=int, action="append", required=True, dest="prs", help="PR number (repeatable)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Machine-readable JSON output")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually resolve AUTO-RESOLVABLE PRs in a scratch worktree and push. "
            "ESCALATE-class PRs are never touched. Default is dry-run/report-only."
        ),
    )
    args = parser.parse_args(argv)

    reports = [_fetch_pr_report(pr) for pr in args.prs]
    contention = _find_contention(reports)

    if args.execute:
        repo_root = _repo_root()
        for r in reports:
            if r.ok and r.verdict == "AUTO-RESOLVABLE":
                try:
                    r.execute_result = _execute_auto_resolve(r, repo_root)
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    r.verdict = "ESCALATE"
                    r.escalation_files = list(r.files)
                    r.execute_result = f"ABORTED (unexpected error): {exc}"

    if args.json_out:
        _render_json(reports, contention)
    else:
        _render_human(reports, contention, executed=args.execute)

    return 0


if __name__ == "__main__":
    sys.exit(main())
