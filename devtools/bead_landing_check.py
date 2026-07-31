"""bead-landing-check: verify whether a bead's cited implementation evidence
already landed on master, before dispatching a full investigation cycle.

Motivating incident (2026-07-30/31): five separate open beads (o4j2, hiu,
0jf4, pbuh, cijx.4) each cost a dispatched agent a full investigation cycle
before discovering the described work was already done on master. The
technique that worked, twice: take the commit(s) a bead cites as evidence,
cherry-pick them onto current master in a throwaway worktree, and check for
an EMPTY DIFF. An empty diff proves the change already exists regardless of
squash-merge id rewriting -- it is cheaper and more reliable than
`git log --is-ancestor` (fails under squash merges) or grepping for issue ids
(false-positives on prose mentions).

This tool automates that technique plus a cheap gh-based PR merge-state check,
scaled to run over the whole open backlog without one investigation cycle per
bead:

  - extracts cited commit hashes / PR numbers / file paths from a bead's
    title, description, design, acceptance_criteria, notes, close_reason,
    and comment text
  - resolves each cited commit against a SINGLE reused throwaway worktree
    (not one worktree per bead) and classifies it: unknown-revision,
    already-on-master, empty-diff, non-empty-diff, does-not-apply
  - resolves each cited PR number via `gh pr view --json state,mergedAt,...`,
    cached to disk so a re-run is nearly free
  - emits a per-bead verdict: LIKELY-STALE / LIKELY-LIVE / UNDETERMINED, with
    the evidence and a confidence tier (a commit empty-diff is "strong"
    evidence; a merged-PR citation alone is "weak" -- it proves the PR
    landed, not that this bead's full scope is subsumed)

Honesty rule: a bead with no cited commits/PRs cannot be verified by this
tool and MUST be reported UNDETERMINED, never guessed either way. This tool
never auto-closes anything -- it only reports; a human or a follow-up bead
decides.

CORRECTION (2026-07-31, five independent human reviewers checked all 190
LIKELY-STALE verdicts from the first sweep against the live archive): "a
cited commit exists on master" is not "the bead is done" -- roughly 6 of
every 114 checked were genuinely safe to close. The dominant failure mode is
exactly the defect class this repo has spent the night finding elsewhere:
code SHIPPED with no live production consumer (an unreferenced function, an
unwired EventBus, a contract test still `xfail`). A second failure mode is
epic/parent beads whose own commit landed while most of their dotted-id
children or `parent-child` dependents are still open. A third is beads whose
own latest note already says "remaining" / "deferred" / "not attempted" in
plain language, contradicted only by an earlier landed-prerequisite citation.

Three corrective checks now run before a verdict is allowed to reach
LIKELY-STALE:

  1. LIVE-CONSUMER CHECK: a landed commit only counts as strong evidence if
     at least one top-level symbol it added has a `git grep` hit outside the
     commit's own touched files and outside tests/. A landed commit with no
     provable consumer (or no extractable symbols at all -- non-Python
     changes) downgrades the bead to UNDETERMINED instead of LIKELY-STALE.
  2. OPEN-DEPENDENTS CHECK: a bead with unresolved `parent-child` dependents
     (open or in_progress) is never LIKELY-STALE regardless of its own
     evidence -- an epic landing a foundational commit does not mean its
     children are done.
  3. SUPPRESSION-PHRASE CHECK: if the bead's own text contains an explicit
     remaining/deferred/not-implemented/no-consumer phrase, that overrides
     any commit/PR evidence and forces UNDETERMINED.

None of these upgrade a verdict -- they only ever downgrade toward
UNDETERMINED, in keeping with the rule that a wrong LIKELY-STALE is worse
than an honest "not verifiable".

Usage:
    # Sweep every open + in_progress bead (the default):
    devtools workspace bead-landing-check

    # Check specific beads:
    devtools workspace bead-landing-check polylogue-o4j2 polylogue-hiu

    # Machine-readable JSON, only the beads it can actually vouch for:
    devtools workspace bead-landing-check --json --stale-only

    # Cheap pass with no network calls, reusing whatever is cached:
    devtools workspace bead-landing-check --offline

    # Skip the (slower) cherry-pick path, PR checks only:
    devtools workspace bead-landing-check --no-commits

Safety: the throwaway worktree defaults to
`.cache/bead-landing-check/worktree` (gitignored, disposable) and is REUSED
across beads and across runs -- never created per-bead. `--remove-worktree`
tears it down explicitly at the end of a run: it verifies no live process has
its cwd inside the worktree before removing it, and never passes `--force` to
`git worktree remove`. By default the worktree is left in place for the next
run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BeadDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

# Bare "#N" only -- this repo's own PR/commit-subject convention cites PRs
# this way (e.g. "(#3390)"), so it is the dominant valid signal. A
# `github.com/.../pull/N` URL pattern was deliberately dropped: bead text
# frequently *quotes* sample payloads containing unrelated PR-linked URLs
# (e.g. a `pr-link` record's sample JSON), and matching those produced
# false PR citations unconnected to the bead's own resolution.
_PR_PATTERN = re.compile(r"(?:^|[^\w/])#(\d{2,6})\b")
_COMMIT_PATTERN = re.compile(r"\b[0-9a-f]{7,40}\b")
# Matches immediately before a hash that reads "master @", "master (",
# "master@", or bare "master " -- this repo's snapshot-anchor idiom.
_SNAPSHOT_ANCHOR_PATTERN = re.compile(r"master\s*[@(]?\s*$", re.IGNORECASE)
_FILE_PATTERN = re.compile(
    r"(?:polylogue|tests|devtools|docs|storage|pipeline|daemon|cli|mcp"
    r"|browser[_-]extension|browser_capture|coordination|archive|insights"
    r"|context|core|hooks|maintenance|schemas)/[\w./\-]+\.(?:py|ts|js|yaml|yml|md|json|sql)",
    re.IGNORECASE,
)

# A line added by a diff that defines a top-level-ish function/class -- used
# by the live-consumer check to find what a landed commit actually shipped.
_ADDED_SYMBOL_PATTERN = re.compile(r"^\+\s*(?:async\s+)?(?:def|class)\s+(\w+)")

# Phrases indicating a bead's own text already says the work is unfinished --
# a strong suppression signal against a stale verdict built on commit/PR
# citation alone. Case-insensitive; matched anywhere in the bead's combined
# text, not only its most recent note (notes are an unstructured blob here).
_SUPPRESSION_PATTERN = re.compile(
    r"\b(?:"
    r"not attempted|no code written|not implemented|not yet implemented|"
    r"not wired|zero callers|no callers|no live (?:producer|consumer)|"
    r"remains? open|still open|still not|deferred(?: to)?|explicitly deferred|"
    r"not yet wired|xfail|"
    r"is NOT satisfied|not satisfied|not attempted"
    r")\b",
    re.IGNORECASE,
)

_CACHE_DIR_NAME = ".cache/bead-landing-check"
_DEFAULT_WORKTREE_SUBDIR = "worktree"
_DEFAULT_CACHE_TTL_DAYS = 7
_DEFAULT_REPO_SLUG = "Sinity/polylogue"
_DEFAULT_STATUSES = frozenset({"open", "in_progress"})
_MAX_FILE_CHECKS_PER_BEAD = 6


@dataclass
class Evidence:
    pr_numbers: list[int] = field(default_factory=list)
    commit_candidates: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)


def _bead_text(bead: BeadDict) -> str:
    parts = [
        bead.get("title") or "",
        bead.get("description") or "",
        bead.get("design") or "",
        bead.get("acceptance_criteria") or "",
        bead.get("notes") or "",
        bead.get("close_reason") or "",
    ]
    for comment in bead.get("comments") or []:
        parts.append(comment.get("text") or "")
    return "\n".join(parts)


def extract_evidence(bead: BeadDict) -> Evidence:
    text = _bead_text(bead)

    prs: set[int] = set()
    for m in _PR_PATTERN.finditer(text):
        prs.add(int(m.group(1)))

    commits: set[str] = set()
    for m in _COMMIT_PATTERN.finditer(text):
        token = m.group(0)
        # Require a hex token to mix digits and letters -- a pure-digit run is
        # a plain number (record count, port, etc.), not a commit-ish hash.
        if not (any(c.isdigit() for c in token) and any(c.isalpha() for c in token)):
            continue
        if _SNAPSHOT_ANCHOR_PATTERN.search(text[max(0, m.start() - 24) : m.start()]):
            # "Generated from master @ <hash>" / "against current master
            # (<hash>)" is this repo's prework-packet/verification-pass
            # snapshot-anchor idiom (185+ occurrences in the live backlog):
            # it records what master looked like when a note was written,
            # not that the cited hash is this bead's own resolving commit.
            # Confirmed false positive: polylogue-lkrc cites its own
            # verification-pass master snapshot this way in a note that
            # explicitly says the named gaps are STILL open.
            continue
        commits.add(token)

    files = list(dict.fromkeys(_FILE_PATTERN.findall(text)))

    return Evidence(pr_numbers=sorted(prs), commit_candidates=sorted(commits), file_paths=files)


def find_suppression_signal(bead: BeadDict) -> str | None:
    """Return a short excerpt if the bead's own text already says the work is
    unfinished (fix 3: weight the bead's own words above commit existence).
    """
    text = _bead_text(bead)
    m = _SUPPRESSION_PATTERN.search(text)
    if not m:
        return None
    start = max(0, m.start() - 40)
    end = min(len(text), m.end() + 40)
    return " ".join(text[start:end].split())


def build_open_parent_child_dependents_index(all_beads: list[BeadDict]) -> dict[str, list[dict[str, str]]]:
    """Map ``parent_bead_id -> [{"id": ..., "status": ...}, ...]`` for every
    still-open/in_progress bead whose ``dependencies`` list records a
    ``parent-child`` edge onto that parent (fix 2: an epic's own commit
    landing does not mean its children are done).
    """
    index: dict[str, list[dict[str, str]]] = {}
    for bead in all_beads:
        if bead.get("status") not in ("open", "in_progress"):
            continue
        for dep in bead.get("dependencies") or []:
            if dep.get("type") != "parent-child":
                continue
            parent_id = dep.get("depends_on_id")
            if not parent_id:
                continue
            index.setdefault(parent_id, []).append({"id": bead["id"], "status": bead["status"]})
    return index


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------


_MAX_CONSUMER_SYMBOLS_PER_COMMIT = 4


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float | None = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _is_test_path(path: str) -> bool:
    if not path:
        return True  # empty/unparseable -- treat conservatively as not a production consumer
    parts = path.split("/")
    return "tests" in parts or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")


class CommitChecker:
    """Resolves cited commits against a single reused throwaway worktree."""

    def __init__(self, repo_root: Path, worktree_dir: Path, base_ref: str = "origin/master") -> None:
        if worktree_dir.resolve() == repo_root.resolve():
            raise ValueError("throwaway worktree must not be the repo root")
        self.repo_root = repo_root
        self.worktree_dir = worktree_dir
        self.base_ref = base_ref
        self._ensured = False
        self._fetched = False
        self._consumer_cache: dict[str, bool | None] = {}

    def _ensure_fetch(self) -> None:
        if not self._fetched:
            _run(["git", "fetch", "origin", "--quiet"], cwd=self.repo_root, timeout=60)
            self._fetched = True

    def _is_real_commit(self, commit: str) -> bool:
        return _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=self.repo_root).returncode == 0

    def _is_ancestor(self, commit: str) -> bool:
        self._ensure_fetch()
        return _run(["git", "merge-base", "--is-ancestor", commit, self.base_ref], cwd=self.repo_root).returncode == 0

    def _ensure_worktree(self) -> None:
        if self._ensured:
            return
        self._ensure_fetch()
        git_marker = self.worktree_dir / ".git"
        if git_marker.exists():
            pass  # ours already, reuse
        elif self.worktree_dir.exists():
            if any(self.worktree_dir.iterdir()):
                raise RuntimeError(
                    f"{self.worktree_dir} exists and is not a git worktree -- refusing to reuse or clobber it"
                )
            self.worktree_dir.rmdir()
            self._create_worktree()
        else:
            self._create_worktree()
        self._ensured = True

    def _create_worktree(self) -> None:
        self.worktree_dir.parent.mkdir(parents=True, exist_ok=True)
        result = _run(
            ["git", "worktree", "add", "--detach", str(self.worktree_dir), self.base_ref],
            cwd=self.repo_root,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to create throwaway worktree at {self.worktree_dir}: {result.stderr.strip()}")

    def _reset_worktree(self) -> None:
        _run(["git", "cherry-pick", "--abort"], cwd=self.worktree_dir)
        _run(["git", "reset", "--hard", self.base_ref], cwd=self.worktree_dir)
        _run(["git", "clean", "-fdq"], cwd=self.worktree_dir)

    def _commit_touched_files(self, commit: str) -> list[str]:
        result = _run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=self.repo_root)
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _content_equivalent_to_base(self, commit: str) -> bool | None:
        """Fallback for a conflicted cherry-pick: compare each touched file's
        content at *commit* against ``self.base_ref`` directly.

        A squash-merge can rewrite surrounding history enough that a raw
        cherry-pick of the *original* pre-squash commit conflicts, even
        though the change is fully present on master under a different
        commit id. If every file the commit touched already has identical
        content on ``self.base_ref``, that is direct proof of landing --
        stronger than the conflict result alone would suggest.

        Returns None when there is nothing to compare (inconclusive).
        """
        files = self._commit_touched_files(commit)
        if not files:
            return None
        for path in files:
            base = _run(["git", "show", f"{self.base_ref}:{path}"], cwd=self.repo_root)
            theirs = _run(["git", "show", f"{commit}:{path}"], cwd=self.repo_root)
            if (base.returncode == 0) != (theirs.returncode == 0):
                return False
            if base.returncode == 0 and base.stdout != theirs.stdout:
                return False
        return True

    def check(self, commit: str) -> str:
        """Classify *commit* relative to ``self.base_ref``.

        Returns one of: unknown-revision, already-on-master, empty-diff,
        content-equivalent, non-empty-diff, does-not-apply.
        """
        if not self._is_real_commit(commit):
            return "unknown-revision"
        if self._is_ancestor(commit):
            return "already-on-master"

        self._ensure_worktree()
        self._reset_worktree()
        result = _run(["git", "cherry-pick", "--no-commit", "--allow-empty", commit], cwd=self.worktree_dir)
        if result.returncode != 0:
            self._reset_worktree()
            if self._content_equivalent_to_base(commit):
                return "content-equivalent"
            return "does-not-apply"
        diff = _run(["git", "diff", "--cached", "--stat"], cwd=self.worktree_dir)
        empty = not diff.stdout.strip()
        self._reset_worktree()
        return "empty-diff" if empty else "non-empty-diff"

    def added_symbols(self, commit: str) -> list[str]:
        """Top-level function/class names *commit* added, per its diff.

        Best-effort text extraction, not an AST diff -- good enough to seed
        `git grep` lookups for the live-consumer check.
        """
        result = _run(["git", "show", "--unified=0", "--no-color", commit], cwd=self.repo_root, timeout=30)
        if result.returncode != 0:
            return []
        symbols: list[str] = []
        for line in result.stdout.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            m = _ADDED_SYMBOL_PATTERN.match(line)
            if not m:
                continue
            name = m.group(1)
            if name.startswith("test_") or name in ("__init__", "__post_init__", "__repr__", "__str__"):
                continue
            if name not in symbols:
                symbols.append(name)
        return symbols[:_MAX_CONSUMER_SYMBOLS_PER_COMMIT]

    def has_live_consumer(self, commit: str, symbols: list[str]) -> bool | None:
        """Return True if any *symbols* has a reference outside the commit's
        own touched files and outside tests/, False if none do, None if there
        was nothing checkable (e.g. a non-Python change with no symbols).
        """
        if not symbols:
            return None
        touched = set(self._commit_touched_files(commit))
        for symbol in symbols:
            result = _run(
                ["git", "grep", "--no-color", "-w", "-l", symbol, self.base_ref, "--", "*.py"],
                cwd=self.repo_root,
                timeout=15,
            )
            if result.returncode not in (0, 1):
                continue  # grep error on this symbol -- skip, don't crash the sweep
            for line in result.stdout.splitlines():
                _, _, path = line.partition(":")
                if path in touched or _is_test_path(path):
                    continue
                return True
        return False

    def consumer_check(self, commit: str) -> bool | None:
        """Combined added_symbols + has_live_consumer, cached per commit."""
        if commit in self._consumer_cache:
            return self._consumer_cache[commit]
        result = self.has_live_consumer(commit, self.added_symbols(commit))
        self._consumer_cache[commit] = result
        return result


def check_file_paths(repo_root: Path, paths: list[str], base_ref: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths[:_MAX_FILE_CHECKS_PER_BEAD]:
        exists = _run(["git", "cat-file", "-e", f"{base_ref}:{p}"], cwd=repo_root).returncode == 0
        entry: dict[str, Any] = {"path": p, "exists_on_master": exists}
        if exists:
            log = _run(["git", "log", "-1", "--format=%H|%cI", base_ref, "--", p], cwd=repo_root)
            if log.stdout.strip():
                sha, _, date = log.stdout.strip().partition("|")
                entry["last_commit"] = sha
                entry["last_commit_date"] = date
        out.append(entry)
    return out


def _worktree_has_live_process(path: Path) -> bool:
    """Conservative check: any process whose cwd resolves inside *path*."""
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return True  # can't verify -- block rather than risk it
    target = str(path.resolve())
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            resolved = str((entry / "cwd").resolve())
        except OSError:
            continue
        if resolved == target or resolved.startswith(target + "/"):
            return True
    return False


def remove_worktree(repo_root: Path, worktree_dir: Path) -> str:
    """Remove *worktree_dir* iff it's ours, clean, and has no live occupant.

    Never passes --force to `git worktree remove`.
    """
    if not worktree_dir.exists():
        return "not-present"
    if _worktree_has_live_process(worktree_dir):
        return "blocked-live-process"
    dirty = _run(["git", "status", "--porcelain"], cwd=worktree_dir).stdout.strip()
    if dirty:
        return "blocked-dirty"
    result = _run(["git", "worktree", "remove", str(worktree_dir)], cwd=repo_root)
    if result.returncode != 0:
        return f"failed: {result.stderr.strip()[:200]}"
    _run(["git", "worktree", "prune"], cwd=repo_root)
    return "removed"


# ---------------------------------------------------------------------------
# PR checking (gh, disk-cached)
# ---------------------------------------------------------------------------


@dataclass
class PrResult:
    number: int
    found: bool
    state: str | None = None
    merged_at: str | None = None
    merge_commit: str | None = None
    title: str | None = None
    error: str | None = None


class PrChecker:
    def __init__(
        self,
        repo_slug: str,
        cache_path: Path,
        ttl_days: int,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> None:
        self.repo_slug = repo_slug
        self.cache_path = cache_path
        self.ttl_seconds = ttl_days * 86400
        self.refresh = refresh
        self.offline = offline
        self._cache: dict[str, Any] = self._load_cache()
        self._dirty = False

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                return dict(json.loads(self.cache_path.read_text()))
            except json.JSONDecodeError:
                return {}
        return {}

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        self._dirty = False

    def _from_cache(self, number: int, cached: dict[str, Any]) -> PrResult:
        return PrResult(
            number=number,
            found=bool(cached.get("found")),
            state=cached.get("state"),
            merged_at=cached.get("merged_at"),
            merge_commit=cached.get("merge_commit"),
            title=cached.get("title"),
            error=cached.get("error"),
        )

    def check(self, number: int) -> PrResult:
        key = str(number)
        cached = self._cache.get(key)
        now = time.time()
        # A MERGED result is permanent -- always safe to reuse. Anything else
        # (open, not-found, error) is only reused within the TTL.
        if (
            cached is not None
            and not self.refresh
            and (cached.get("state") == "MERGED" or (now - cached.get("checked_at", 0)) < self.ttl_seconds)
        ):
            return self._from_cache(number, cached)

        if self.offline:
            if cached is not None:
                return self._from_cache(number, cached)
            return PrResult(number=number, found=False, error="offline: not cached")

        result = self._fetch(number)
        self._cache[key] = {
            "checked_at": now,
            "found": result.found,
            "state": result.state,
            "merged_at": result.merged_at,
            "merge_commit": result.merge_commit,
            "title": result.title,
            "error": result.error,
        }
        self._dirty = True
        return result

    def _fetch(self, number: int) -> PrResult:
        result = _run(
            ["gh", "pr", "view", str(number), "--repo", self.repo_slug, "--json", "state,mergedAt,mergeCommit,title"],
            timeout=20,
        )
        if result.returncode != 0:
            return PrResult(number=number, found=False, error=result.stderr.strip()[:200])
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return PrResult(number=number, found=False, error="gh returned non-JSON output")
        merge_commit = (data.get("mergeCommit") or {}).get("oid")
        return PrResult(
            number=number,
            found=True,
            state=data.get("state"),
            merged_at=data.get("mergedAt"),
            merge_commit=merge_commit,
            title=data.get("title"),
        )


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

_LANDED_COMMIT_STATES = frozenset({"empty-diff", "already-on-master", "content-equivalent"})


@dataclass
class BeadVerdict:
    bead_id: str
    status: str
    priority: int
    title: str
    verdict: str
    confidence: str
    reasons: list[str]
    evidence: dict[str, Any]


def verdict_for_bead(
    bead: BeadDict,
    evidence: Evidence,
    commit_results: dict[str, str],
    pr_results: dict[int, PrResult],
    *,
    commit_consumer: dict[str, bool | None] | None = None,
    open_dependents: list[dict[str, str]] | None = None,
    suppression_signal: str | None = None,
) -> BeadVerdict:
    reasons: list[str] = []
    commit_consumer = commit_consumer or {}
    open_dependents = open_dependents or []

    landed_all = sorted(c for c, s in commit_results.items() if s in _LANDED_COMMIT_STATES)
    landed_verified = sorted(c for c in landed_all if commit_consumer.get(c) is True)
    landed_unconsumed = sorted(c for c in landed_all if commit_consumer.get(c) is False)
    landed_unknown_consumer = sorted(c for c in landed_all if c not in landed_verified and c not in landed_unconsumed)
    live = sorted(c for c, s in commit_results.items() if s == "non-empty-diff")
    unresolved = sorted(c for c, s in commit_results.items() if s == "unknown-revision")
    inapplicable = sorted(c for c, s in commit_results.items() if s == "does-not-apply")

    merged_prs = sorted(n for n, r in pr_results.items() if r.state == "MERGED")
    open_prs = sorted(n for n, r in pr_results.items() if r.found and r.state != "MERGED")
    missing_prs = sorted(n for n, r in pr_results.items() if not r.found)

    if landed_verified:
        verdict, confidence = "LIKELY-STALE", "strong"
        detail = ", ".join(f"{c} ({commit_results[c]})" for c in landed_verified)
        reasons.append(f"cited commit(s) already on master AND reachable from a live caller: {detail}")
        if live:
            reasons.append(
                f"NOTE: commit(s) {', '.join(live)} still produce a non-empty diff -- "
                "possibly a partial landing, verify remaining scope before closing"
            )
    elif live:
        verdict, confidence = "LIKELY-LIVE", "strong"
        reasons.append(
            f"cited commit(s) cherry-pick with a real (non-empty) diff, not yet on master: {', '.join(live)}"
        )
    elif landed_unconsumed or landed_unknown_consumer:
        # Landed on master, but the live-consumer check (fix 1, 2026-07-31)
        # could not prove the shipped code is reachable from a production
        # caller. Measured against 114 human-verified verdicts, "cited commit
        # exists on master" alone was right about 6 times in 114 -- this is
        # exactly the "shipped, nothing reads it" defect class the check
        # exists to catch, so it does NOT count as LIKELY-STALE on its own.
        verdict, confidence = "UNDETERMINED", "none"
        if landed_unconsumed:
            reasons.append(
                "commit(s) already on master but NO live production consumer found outside tests -- "
                f"shipped without being wired in, treat as still open: {', '.join(landed_unconsumed)}"
            )
        if landed_unknown_consumer:
            reasons.append(
                "commit(s) already on master but consumer-reachability could not be checked (non-Python "
                f"change or no top-level symbols added) -- not enough to confirm wiring: {', '.join(landed_unknown_consumer)}"
            )
    elif merged_prs and not open_prs:
        verdict, confidence = "LIKELY-STALE", "weak"
        reasons.append(
            f"cited PR(s) merged: {', '.join(f'#{n}' for n in merged_prs)} -- "
            "weak evidence: proves the PR landed, not that this bead's full scope is subsumed; verify AC by AC"
        )
    elif open_prs:
        verdict, confidence = "LIKELY-LIVE", "weak"
        reasons.append(f"cited PR(s) not yet merged: {', '.join(f'#{n}' for n in open_prs)}")
    else:
        verdict, confidence = "UNDETERMINED", "none"
        if unresolved:
            reasons.append(
                f"{len(unresolved)} cited hex token(s) are not real commits in this repo "
                f"(likely a false-positive extraction, e.g. a worktree/session id): {', '.join(unresolved)}"
            )
        if inapplicable:
            reasons.append(
                f"commit(s) could not be cherry-picked cleanly onto master (conflict) -- ambiguous, needs human review: "
                f"{', '.join(inapplicable)}"
            )
        if missing_prs:
            reasons.append(
                f"PR number(s) not found via gh (deleted, mistyped, or actually an issue number, not a PR): "
                f"{', '.join(f'#{n}' for n in missing_prs)}"
            )
        if not reasons:
            reasons.append("no cited commit hash or PR number found in bead text -- not verifiable by this tool")

    # Cross-cutting downgrades (fix 2, fix 3, 2026-07-31): applied after the
    # evidence-shape branch above. Both only ever pull a verdict DOWN toward
    # UNDETERMINED -- neither can upgrade LIKELY-LIVE or UNDETERMINED to STALE.
    if verdict == "LIKELY-STALE" and open_dependents:
        verdict, confidence = "UNDETERMINED", "none"
        examples = ", ".join(d["id"] for d in open_dependents[:5])
        more = f" (+{len(open_dependents) - 5} more)" if len(open_dependents) > 5 else ""
        reasons.append(
            f"{len(open_dependents)} open parent-child dependent bead(s) still unresolved -- an epic/parent "
            f"commit landing does not mean its children are done: {examples}{more}"
        )

    if verdict == "LIKELY-STALE" and suppression_signal:
        verdict, confidence = "UNDETERMINED", "none"
        reasons.append(
            "bead's own text contains an explicit remaining/deferred/not-done phrase that contradicts a stale "
            f'verdict built on commit/PR citation alone: "...{suppression_signal}..."'
        )

    if verdict == "LIKELY-STALE":
        reasons.append(
            "CAVEAT: this proves the cited commit(s)/PR(s) exist on master AND (for commit evidence) that the "
            "shipped code has a live caller -- it is still not proof every acceptance criterion is satisfied. "
            "Read the bead's AC list before closing."
        )

    evidence_payload: dict[str, Any] = {
        "pr_numbers": evidence.pr_numbers,
        "commit_candidates": evidence.commit_candidates,
        "file_paths": evidence.file_paths,
        "commit_checks": dict(commit_results),
        "commit_consumer_checks": dict(commit_consumer),
        "open_parent_child_dependents": open_dependents,
        "suppression_signal": suppression_signal,
        "pr_checks": {
            str(n): {
                "found": r.found,
                "state": r.state,
                "merged_at": r.merged_at,
                "merge_commit": r.merge_commit,
                "title": r.title,
                "error": r.error,
            }
            for n, r in pr_results.items()
        },
    }

    return BeadVerdict(
        bead_id=bead["id"],
        status=bead.get("status", "?"),
        priority=bead.get("priority", 4),
        title=bead.get("title", ""),
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        evidence=evidence_payload,
    )


# ---------------------------------------------------------------------------
# Loading + rendering
# ---------------------------------------------------------------------------


def load_beads_from_jsonl(path: Path) -> list[BeadDict]:
    beads: list[BeadDict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            beads.append(json.loads(line))
    return beads


def _short(s: str, n: int = 62) -> str:
    return s[:n] + "..." if len(s) > n else s


def _render_human(results: list[BeadVerdict], stats: dict[str, Any]) -> None:
    print("=" * 100)
    print("BEAD LANDING CHECK")
    print("=" * 100)
    print()
    for r in results:
        print(f"{r.bead_id:24s} P{r.priority} [{r.status:11s}] {r.verdict:15s} ({r.confidence:6s})  {_short(r.title)}")
        for reason in r.reasons:
            print(f"    - {reason}")
    print()
    counts = Counter(r.verdict for r in results)
    print(
        f"Checked {stats['beads_checked']} beads in {stats['elapsed_seconds']}s "
        f"({stats['commits_checked']} commit checks, {stats.get('consumer_checks', 0)} consumer checks, "
        f"{stats['prs_checked']} PR checks)"
    )
    print("Verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    removal_note = (
        f" (removed: {stats['worktree_removed']})" if stats.get("worktree_removed") else " (persisted for reuse)"
    )
    print(f"Throwaway worktree: {stats['worktree_dir']}{removal_note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="devtools workspace bead-landing-check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "beads",
        nargs="*",
        metavar="BEAD_ID",
        help="Specific bead id(s) to check (default: all open + in_progress beads)",
    )
    parser.add_argument(
        "--beads-file",
        type=Path,
        default=None,
        help="Path to .beads/issues.jsonl (default: <repo>/.beads/issues.jsonl)",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Bead status to include when no bead ids are given (repeatable; default: open, in_progress)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap the number of beads processed after filtering (0 = no cap)"
    )
    parser.add_argument("--no-commits", action="store_true", help="Skip cherry-pick commit verification")
    parser.add_argument("--no-prs", action="store_true", help="Skip gh PR merge-state checks")
    parser.add_argument("--offline", action="store_true", help="Reuse only cached gh results; never call gh")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached PR results and re-fetch every one")
    parser.add_argument(
        "--repo-slug",
        default=_DEFAULT_REPO_SLUG,
        help=f"GitHub repo slug for gh PR checks (default: {_DEFAULT_REPO_SLUG})",
    )
    parser.add_argument(
        "--worktree-dir",
        type=Path,
        default=None,
        help="Throwaway worktree path (default: <repo>/.cache/bead-landing-check/worktree, reused across runs)",
    )
    parser.add_argument(
        "--remove-worktree",
        action="store_true",
        help="Remove the throwaway worktree after this run (checks for a live occupant first; never --force)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--stale-only", action="store_true", help="Only report LIKELY-STALE verdicts")
    parser.add_argument(
        "--only-evidenced",
        action="store_true",
        help="Skip beads that cite no commit/PR evidence at all (pure UNDETERMINED noise)",
    )
    args = parser.parse_args(argv)

    repo_root_result = _run(["git", "rev-parse", "--show-toplevel"])
    if repo_root_result.returncode != 0:
        print("ERROR: not inside a git repository", file=sys.stderr)
        return 1
    repo_root = Path(repo_root_result.stdout.strip())

    beads_file = args.beads_file or (repo_root / ".beads" / "issues.jsonl")
    if not beads_file.exists():
        print(f"ERROR: beads file not found: {beads_file}", file=sys.stderr)
        return 1

    all_beads = load_beads_from_jsonl(beads_file)
    by_id = {b["id"]: b for b in all_beads}

    if args.beads:
        missing = [b for b in args.beads if b not in by_id]
        if missing:
            print(f"ERROR: unknown bead id(s): {', '.join(missing)}", file=sys.stderr)
            return 1
        selected = [by_id[b] for b in args.beads]
    else:
        statuses = set(args.status) if args.status else set(_DEFAULT_STATUSES)
        selected = [b for b in all_beads if b.get("status") in statuses]
        selected.sort(key=lambda b: (b.get("priority", 4), b["id"]))
        if args.limit:
            selected = selected[: args.limit]

    cache_dir = repo_root / _CACHE_DIR_NAME
    worktree_dir = args.worktree_dir or (cache_dir / _DEFAULT_WORKTREE_SUBDIR)

    commit_checker = None if args.no_commits else CommitChecker(repo_root, worktree_dir)
    pr_checker = (
        None
        if args.no_prs
        else PrChecker(
            args.repo_slug,
            cache_dir / "pr-cache.json",
            _DEFAULT_CACHE_TTL_DAYS,
            refresh=args.refresh_cache,
            offline=args.offline,
        )
    )

    dependents_index = build_open_parent_child_dependents_index(all_beads)

    t_start = time.time()
    results: list[BeadVerdict] = []
    n_commits_checked = 0
    n_prs_checked = 0
    n_consumer_checks = 0

    for bead in selected:
        evidence = extract_evidence(bead)

        commit_results: dict[str, str] = {}
        commit_consumer: dict[str, bool | None] = {}
        if commit_checker is not None:
            for commit in evidence.commit_candidates:
                status = commit_checker.check(commit)
                commit_results[commit] = status
                n_commits_checked += 1
                if status in _LANDED_COMMIT_STATES:
                    commit_consumer[commit] = commit_checker.consumer_check(commit)
                    n_consumer_checks += 1

        pr_results: dict[int, PrResult] = {}
        if pr_checker is not None:
            for number in evidence.pr_numbers:
                pr_results[number] = pr_checker.check(number)
                n_prs_checked += 1

        results.append(
            verdict_for_bead(
                bead,
                evidence,
                commit_results,
                pr_results,
                commit_consumer=commit_consumer,
                open_dependents=dependents_index.get(bead["id"], []),
                suppression_signal=find_suppression_signal(bead),
            )
        )

    if pr_checker is not None:
        pr_checker.flush()

    elapsed = time.time() - t_start

    removal_status = None
    if args.remove_worktree:
        removal_status = remove_worktree(repo_root, worktree_dir)

    if args.stale_only:
        results = [r for r in results if r.verdict == "LIKELY-STALE"]
    if args.only_evidenced:
        results = [r for r in results if r.evidence["pr_numbers"] or r.evidence["commit_candidates"]]

    stats = {
        "beads_checked": len(selected),
        "commits_checked": n_commits_checked,
        "consumer_checks": n_consumer_checks,
        "prs_checked": n_prs_checked,
        "elapsed_seconds": round(elapsed, 2),
        "worktree_dir": str(worktree_dir),
        "worktree_removed": removal_status,
    }

    if args.json:
        counts = Counter(r.verdict for r in results)
        payload = {
            "stats": stats,
            "verdict_counts": dict(counts),
            "results": [
                {
                    "bead_id": r.bead_id,
                    "status": r.status,
                    "priority": r.priority,
                    "title": r.title,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "reasons": r.reasons,
                    "evidence": r.evidence,
                }
                for r in results
            ],
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        print()
    else:
        _render_human(results, stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
