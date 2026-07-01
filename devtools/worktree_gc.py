"""Safe worktree garbage collection for agent and feature worktrees (#1222).

Parses ``git worktree list --porcelain``, classifies candidates, checks
dirty state, and applies removals only for safe entries.  Never removes
dirty worktrees or the main worktree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class WorktreeEntry:
    path: Path
    head: str
    branch: str | None = None
    bare: bool = False
    locked: bool = False
    detached: bool = False
    prunable: bool = False


@dataclass(frozen=True, slots=True)
class GcCandidate:
    entry: WorktreeEntry
    reason: str
    safe: bool
    blocked_reason: str | None = None
    action: str | None = None
    evidence: dict[str, object] | None = None


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    """Run a git command and return stripped stdout.  Raises on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    result.check_returncode()
    return result.stdout.strip()


def _run_git_nullable(args: list[str], *, cwd: Path | None = None) -> str | None:
    """Run a git command; return stripped stdout or None on failure."""
    try:
        return _run_git(args, cwd=cwd)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def parse_worktree_list(porcelain: str) -> list[WorktreeEntry]:
    """Parse ``git worktree list --porcelain`` output."""
    entries: list[WorktreeEntry] = []
    current: dict[str, Any] = {}
    for line in porcelain.splitlines():
        line = line.strip()
        if not line:
            if current:
                entries.append(_build_entry(current))
                current = {}
            continue
        if " " in line:
            key, _, value = line.partition(" ")
            current[key] = value
        else:
            current[line] = True
    if current:
        entries.append(_build_entry(current))
    return entries


def _build_entry(raw: dict[str, Any]) -> WorktreeEntry:
    return WorktreeEntry(
        path=Path(raw["worktree"]),
        head=raw.get("HEAD", ""),
        branch=raw.get("branch"),
        bare="bare" in raw,
        locked="locked" in raw,
        detached="detached" in raw,
        prunable="prunable" in raw,
    )


def _merged_branches(repo_root: Path, target: str = "master") -> set[str]:
    """Return set of branch refs fully merged into *target*."""
    out = _run_git(["branch", "--merged", target, "--format=%(refname)"], cwd=repo_root)
    if not out:
        return set()
    return {line.strip() for line in out.splitlines()}


def _existing_branches(repo_root: Path) -> set[str]:
    """Return set of all local branch refs."""
    out = _run_git(["branch", "--format=%(refname)"], cwd=repo_root)
    if not out:
        return set()
    return {line.strip() for line in out.splitlines()}


def _ref_exists(repo_root: Path, ref: str) -> bool:
    return _run_git_nullable(["rev-parse", "--verify", "--quiet", ref], cwd=repo_root) is not None


def _resolve_target(repo_root: Path, target: str | None) -> str:
    if target:
        return target
    if _ref_exists(repo_root, "refs/remotes/origin/master"):
        return "origin/master"
    return "master"


def _branch_short_name(ref: str) -> str:
    return ref.removeprefix("refs/heads/")


def _branch_patch_equivalence(repo_root: Path, target: str, branch_ref: str) -> dict[str, object] | None:
    """Return patch-equivalence evidence for *branch_ref* against *target*.

    ``git branch --merged`` only recognizes ancestry merges. Polylogue's normal
    integration path squash-merges PRs, so a branch can be fully represented on
    ``master`` while still appearing unmerged by ancestry. ``git cherry`` compares
    patch IDs and marks equivalent commits with ``-`` and unique commits with
    ``+``; only the all-``-`` case is safe to remove automatically.
    """
    branch = _branch_short_name(branch_ref)
    out = _run_git_nullable(["cherry", target, branch], cwd=repo_root)
    if out is None:
        return None
    rows = [line for line in out.splitlines() if line]
    equivalent = sum(1 for line in rows if line.startswith("-"))
    unique = sum(1 for line in rows if line.startswith("+"))
    unknown = len(rows) - equivalent - unique
    return {
        "patch_equivalent": bool(rows) and unique == 0 and unknown == 0,
        "equivalent_commits": equivalent,
        "unique_commits": unique,
        "unknown_commits": unknown,
    }


def check_dirty(worktree_path: Path) -> bool:
    """Return True if the worktree has uncommitted changes."""
    if not worktree_path.exists():
        return False
    out = _run_git_nullable(["status", "--porcelain"], cwd=worktree_path)
    return bool(out)


def classify_candidates(
    entries: list[WorktreeEntry],
    *,
    repo_root: Path,
    merged: set[str],
    existing: set[str],
    patch_evidence: dict[str, dict[str, object]] | None = None,
) -> list[GcCandidate]:
    """Classify each worktree entry as a candidate or blocked."""
    candidates: list[GcCandidate] = []
    for entry in entries:
        if entry.bare:
            continue
        if entry.path == repo_root:
            continue
        candidates.append(
            _classify_one(
                entry,
                merged=merged,
                existing=existing,
                patch_evidence=patch_evidence or {},
            )
        )
    return candidates


def _classify_one(
    entry: WorktreeEntry,
    *,
    merged: set[str],
    existing: set[str],
    patch_evidence: dict[str, dict[str, object]],
) -> GcCandidate:
    if entry.prunable:
        dirty = check_dirty(entry.path)
        if dirty:
            return GcCandidate(
                entry=entry,
                reason="prunable",
                safe=False,
                blocked_reason="dirty",
            )
        return GcCandidate(
            entry=entry,
            reason="prunable",
            safe=False,
            action="prune",
            blocked_reason="requires-prune",
        )

    if entry.branch is None:
        if entry.detached:
            dirty = check_dirty(entry.path)
            if dirty:
                return GcCandidate(
                    entry=entry,
                    reason="detached",
                    safe=False,
                    blocked_reason="dirty",
                )
            return GcCandidate(
                entry=entry,
                reason="detached",
                safe=False,
                action="remove-force",
                blocked_reason="requires-force",
            )
        return GcCandidate(
            entry=entry,
            reason="unknown",
            safe=False,
            blocked_reason="no-branch-ref",
        )

    if entry.branch not in existing:
        dirty = check_dirty(entry.path)
        if dirty:
            return GcCandidate(
                entry=entry,
                reason="branch-deleted",
                safe=False,
                blocked_reason="dirty",
            )
        return GcCandidate(
            entry=entry,
            reason="branch-deleted",
            safe=True,
            action="remove",
        )

    if entry.branch in merged:
        dirty = check_dirty(entry.path)
        if dirty:
            return GcCandidate(
                entry=entry,
                reason="merged",
                safe=False,
                blocked_reason="dirty",
            )
        if entry.locked:
            return GcCandidate(
                entry=entry,
                reason="merged",
                safe=False,
                blocked_reason="locked",
            )
        return GcCandidate(
            entry=entry,
            reason="merged",
            safe=True,
            action="remove",
        )

    evidence = patch_evidence.get(entry.branch)
    if evidence and evidence.get("patch_equivalent") is True:
        dirty = check_dirty(entry.path)
        if dirty:
            return GcCandidate(
                entry=entry,
                reason="squash-equivalent",
                safe=False,
                blocked_reason="dirty",
                evidence=evidence,
            )
        if entry.locked:
            return GcCandidate(
                entry=entry,
                reason="squash-equivalent",
                safe=False,
                blocked_reason="locked",
                evidence=evidence,
            )
        return GcCandidate(
            entry=entry,
            reason="squash-equivalent",
            safe=True,
            action="remove",
            evidence=evidence,
        )

    return GcCandidate(
        entry=entry,
        reason="unmerged",
        safe=False,
        blocked_reason="branch-not-merged",
        evidence=evidence,
    )


def collect_candidates(repo_root: Path, *, target: str | None = None) -> tuple[list[GcCandidate], list[WorktreeEntry]]:
    """Run git commands and return classified candidates plus all entries."""
    target_ref = _resolve_target(repo_root, target)
    porcelain = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root)
    entries = parse_worktree_list(porcelain)
    merged = _merged_branches(repo_root, target=target_ref)
    existing = _existing_branches(repo_root)
    patch_evidence = {
        ref: evidence
        for ref in existing
        if ref not in merged
        if (evidence := _branch_patch_equivalence(repo_root, target_ref, ref)) is not None
    }
    candidates = classify_candidates(
        entries,
        repo_root=repo_root,
        merged=merged,
        existing=existing,
        patch_evidence=patch_evidence,
    )
    return candidates, entries


def apply_removals(candidates: list[GcCandidate], *, repo_root: Path, force: bool = False) -> list[dict[str, object]]:
    """Remove safe candidates.  Returns per-entry result dicts."""
    results: list[dict[str, object]] = []
    removed = 0

    for c in candidates:
        if c.reason == "prunable":
            results.append({"path": str(c.entry.path), "removed": False, "reason": "prunable-skipped"})
            continue

        if c.safe and c.action == "remove":
            ok = _run_git_nullable(["worktree", "remove", str(c.entry.path)], cwd=repo_root) is not None
            results.append({"path": str(c.entry.path), "removed": ok, "reason": c.reason})
            if ok:
                removed += 1
            continue

        if force and c.action == "remove-force" and not check_dirty(c.entry.path):
            ok = _run_git_nullable(["worktree", "remove", "--force", str(c.entry.path)], cwd=repo_root) is not None
            results.append({"path": str(c.entry.path), "removed": ok, "reason": c.reason})
            if ok:
                removed += 1
            continue

        results.append(
            {
                "path": str(c.entry.path),
                "removed": False,
                "reason": c.reason,
                "blocked": c.blocked_reason,
            }
        )

    _run_git_nullable(["worktree", "prune"], cwd=repo_root)
    results.append({"prune": True, "removed_count": removed})

    return results


def _build_payload(
    candidates: list[GcCandidate],
    apply_results: list[dict[str, object]] | None = None,
    target: str | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for c in candidates:
        entry: dict[str, object] = {
            "path": str(c.entry.path),
            "branch": c.entry.branch or (f"HEAD {c.entry.head[:8]}" if c.entry.head else "unknown"),
            "head": c.entry.head,
            "reason": c.reason,
            "action": c.action,
            "safe": c.safe,
            "blocked_reason": c.blocked_reason,
            "locked": c.entry.locked,
        }
        if c.evidence is not None:
            entry["evidence"] = c.evidence
        entries.append(entry)

    payload: dict[str, object] = {
        "worktrees": entries,
        "safe_count": sum(1 for c in candidates if c.safe),
        "blocked_count": sum(1 for c in candidates if not c.safe),
        "total_count": len(candidates),
    }
    if target is not None:
        payload["target"] = target
    if apply_results is not None:
        payload["results"] = apply_results
    return payload


def _print_human(payload: dict[str, object]) -> None:
    entries = cast(list[dict[str, object]], payload["worktrees"])
    if not entries:
        print("No linked worktrees found.")
        return

    print(f"{'PATH':<50} {'BRANCH':<40} {'REASON':<20} {'SAFE':<6} {'BLOCKED'}")
    print("-" * 140)
    for e in entries:
        path = str(e["path"])
        branch = str(e["branch"])
        reason = str(e["reason"])
        safe = "yes" if e["safe"] else "no"
        blocked = str(e.get("blocked_reason") or "")
        if len(path) > 49:
            path = "..." + path[-46:]
        if len(branch) > 39:
            branch = "..." + branch[-36:]
        print(f"{path:<50} {branch:<40} {reason:<20} {safe:<6} {blocked}")

    print(f"\n{payload['safe_count']} safe, {payload['blocked_count']} blocked, {payload['total_count']} total")

    results = cast(list[dict[str, object]] | None, payload.get("results"))
    if results:
        removed = sum(1 for r in results if r.get("removed"))
        pruned = any(r.get("prune") for r in results)
        if removed:
            print(f"\nRemoved {removed} worktree(s).")
        if pruned:
            print("Pruned stale worktree metadata.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe worktree garbage collection.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply removals (default: dry-run only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow removal of clean detached/broken worktrees.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target branch for merge check (default: origin/master when available, else master).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(_run_git(["rev-parse", "--show-toplevel"]))
    target_ref = _resolve_target(repo_root, args.target)
    candidates, _entries = collect_candidates(repo_root, target=target_ref)

    apply_results = None
    if args.apply:
        apply_results = apply_removals(candidates, repo_root=repo_root, force=args.force)

    payload = _build_payload(candidates, apply_results=apply_results, target=target_ref)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
