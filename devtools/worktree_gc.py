"""Safe worktree garbage collection for agent and feature worktrees (#1222).

Parses ``git worktree list --porcelain``, classifies candidates, checks
dirty state, and applies removals only for safe entries.  Never removes
dirty worktrees or the main worktree.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast


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
    # Snapshot proof captured during collection and revalidated immediately
    # before removal.  This is separate from diagnostic patch evidence.
    proof: dict[str, object] | None = None


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
    """Parse Git's porcelain records without substring-based matching."""
    entries: list[WorktreeEntry] = []
    record: list[str] = []
    for line in porcelain.splitlines():
        if line == "":
            if record:
                entries.append(_build_entry(record))
                record = []
            continue
        record.append(line)
    if record:
        entries.append(_build_entry(record))
    return entries


def _build_entry(record: list[str]) -> WorktreeEntry:
    values: dict[str, str] = {}
    flags: set[str] = set()
    allowed_values = {"worktree", "HEAD", "branch"}
    allowed_flags = {"bare", "detached", "locked", "prunable"}
    for line in record:
        key, separator, value = line.partition(" ")
        if not key:
            raise ValueError("malformed worktree porcelain record")
        if key in allowed_flags and not separator:
            if key in flags or key in values:
                raise ValueError(f"duplicate worktree flag: {key}")
            flags.add(key)
            continue
        if key in {"locked", "prunable"} and separator:
            if key in flags or key in values:
                raise ValueError(f"duplicate worktree field: {key}")
            values[key] = value
            continue
        if key not in allowed_values or not separator or not value:
            raise ValueError(f"unknown or malformed worktree field: {line!r}")
        if key in values:
            raise ValueError(f"duplicate worktree field: {key}")
        values[key] = value

    if "worktree" not in values:
        raise ValueError("worktree porcelain record has no path")
    if "detached" in flags and "branch" in values:
        raise ValueError("detached worktree record has a branch")
    return WorktreeEntry(
        path=Path(values["worktree"]),
        head=values.get("HEAD", ""),
        branch=values.get("branch"),
        bare="bare" in flags,
        locked="locked" in flags or "locked" in values,
        detached="detached" in flags,
        prunable="prunable" in flags or "prunable" in values,
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


def _lane_handoffs(repo_root: Path) -> dict[str, dict[str, object]]:
    """Load automatic SubagentStop handoffs from the shared Git directory."""
    raw_common = _run_git_nullable(["rev-parse", "--git-common-dir"], cwd=repo_root)
    if not raw_common:
        return {}
    common = Path(raw_common)
    if not common.is_absolute():
        common = (repo_root / common).resolve()
    directory = common / "polylogue" / "lane-handoffs"
    handoffs: dict[str, dict[str, object]] = {}
    try:
        paths = list(directory.glob("worktree-agent-*.json"))
    except OSError:
        return handoffs
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        branch = payload.get("branch") if isinstance(payload, dict) else None
        if isinstance(branch, str) and branch.startswith("worktree-agent-"):
            handoffs[f"refs/heads/{branch}"] = payload
    return handoffs


def _ref_exists(repo_root: Path, ref: str) -> bool:
    return _run_git_nullable(["rev-parse", "--verify", "--quiet", ref], cwd=repo_root) is not None


def _resolve_target(repo_root: Path, target: str | None) -> str:
    """Resolve the merge target without silently leaving the invoking checkout.

    An explicit target remains authoritative for operators who intentionally
    classify against a base branch.  The safe default is the checkout's
    current branch; detached checkouts use ``HEAD``.  In particular, never
    substitute a remote base here: an active integration branch may already
    represent lanes that have not reached ``origin/master``.
    """
    if target:
        return target
    current_branch = _run_git_nullable(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo_root)
    return current_branch or "HEAD"


def _branch_short_name(ref: str) -> str:
    return ref.removeprefix("refs/heads/")


def _branch_patch_equivalence(repo_root: Path, target: str, branch_ref: str) -> dict[str, object] | None:
    """Return patch-equivalence evidence for *branch_ref* against *target*.

    ``git branch --merged`` only recognizes ancestry merges. Polylogue's normal
    integration path squash-merges PRs, so a branch can be fully represented on
    the resolved target while still appearing unmerged by ancestry. ``git
    cherry`` compares per-commit patch IDs and is retained as diagnostic
    evidence, but multi-commit squash merges are proven by virtually merging the
    branch into the target and checking whether the target tree would change.
    """
    branch = _branch_short_name(branch_ref)
    out = _run_git_nullable(["cherry", target, branch], cwd=repo_root)
    if out is None:
        return None
    rows = [line for line in out.splitlines() if line]
    equivalent = sum(1 for line in rows if line.startswith("-"))
    unique = sum(1 for line in rows if line.startswith("+"))
    unknown = len(rows) - equivalent - unique
    tree_equivalent = _branch_tree_equivalent(repo_root, target, branch)
    return {
        "patch_equivalent": tree_equivalent or (bool(rows) and unique == 0 and unknown == 0),
        "tree_equivalent": tree_equivalent,
        "equivalent_commits": equivalent,
        "unique_commits": unique,
        "unknown_commits": unknown,
    }


def _branch_tree_equivalent(repo_root: Path, target: str, branch: str) -> bool:
    """Return True when merging *branch* into *target* would not change target."""
    target_tree = _run_git_nullable(["rev-parse", f"{target}^{{tree}}"], cwd=repo_root)
    merged_tree = _run_git_nullable(["merge-tree", "--write-tree", target, branch], cwd=repo_root)
    return bool(target_tree and merged_tree and target_tree == merged_tree)


@dataclass(frozen=True, slots=True)
class WorktreeResidue:
    """Git-visible state that decides whether a worktree may be collected."""

    dirty: bool
    disposable_paths: tuple[str, ...] = ()


_DISPOSABLE_ROOTS = frozenset(
    {
        ".direnv",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
    }
)
_DISPOSABLE_ROOT_FILES = frozenset({".dmypy.json"})


def _is_disposable_ignored_path(relative: str) -> bool:
    """Recognize only repository-owned generated caches and environments.

    The broad ignored roots ``.cache`` and ``.local`` intentionally are not
    disposable.  Only mypy's configured subdirectory under ``.cache`` is.
    This keeps operator files beside generated state visible to the GC gate.
    """
    path = PurePosixPath(relative)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return False
    if parts[0] in _DISPOSABLE_ROOTS or relative in _DISPOSABLE_ROOT_FILES:
        return True
    if len(parts) >= 2 and parts[:2] == (".cache", "mypy"):
        return True
    if "__pycache__" in parts:
        return True
    return path.suffix in {".pyc", ".pyo"}


def inspect_residue(worktree_path: Path) -> WorktreeResidue:
    """Classify tracked, untracked, and ignored filesystem residue.

    Tracked/untracked state and every unknown ignored path block collection.
    Known generated state is returned explicitly so the quarantine remover can
    delete only the exact files whose identities and contents were inspected.
    Git failures are treated as dirty rather than weakening preservation.
    """
    if not worktree_path.exists():
        return WorktreeResidue(dirty=False)
    ordinary = _run_git_nullable(
        ["status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=worktree_path,
    )
    ignored = _run_git_nullable(
        ["ls-files", "-z", "--others", "--ignored", "--exclude-standard"],
        cwd=worktree_path,
    )
    if ordinary is None or ignored is None or ordinary:
        return WorktreeResidue(dirty=True)
    paths = tuple(path for path in ignored.split("\0") if path)
    if any(not _is_disposable_ignored_path(path) for path in paths):
        return WorktreeResidue(dirty=True)
    return WorktreeResidue(dirty=False, disposable_paths=paths)


def check_dirty(worktree_path: Path) -> bool:
    """Return whether non-disposable filesystem state blocks collection."""
    return inspect_residue(worktree_path).dirty


def normalize_repo_root(repo_root: Path) -> Path:
    """Return the canonical main-worktree identity used by helper APIs."""
    return repo_root.expanduser().resolve()


def classify_candidates(
    entries: list[WorktreeEntry],
    *,
    repo_root: Path,
    merged: set[str],
    existing: set[str],
    patch_evidence: dict[str, dict[str, object]] | None = None,
    main_worktree: Path | None = None,
) -> list[GcCandidate]:
    """Classify each worktree while excluding caller and main checkout."""
    excluded = {normalize_repo_root(repo_root)}
    if main_worktree is not None:
        excluded.add(normalize_repo_root(main_worktree))
    candidates: list[GcCandidate] = []
    for entry in entries:
        if entry.bare:
            continue
        if normalize_repo_root(entry.path) in excluded:
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

    if evidence and evidence.get("handoff_head_matches") is True:
        handoff_status = evidence.get("handoff_status")
        if handoff_status == "ready-for-assimilation":
            return GcCandidate(
                entry=entry,
                reason="ready-for-assimilation",
                safe=False,
                blocked_reason="awaiting-assimilation",
                evidence=evidence,
            )
        if handoff_status == "blocked":
            return GcCandidate(
                entry=entry,
                reason="blocked-handoff",
                safe=False,
                blocked_reason="lane-completion-blocked",
                evidence=evidence,
            )

    if entry.branch.startswith("refs/heads/worktree-agent-") and not entry.locked:
        return GcCandidate(
            entry=entry,
            reason="untracked-completion",
            safe=False,
            blocked_reason="missing-completion-handoff",
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
    repo_root = normalize_repo_root(repo_root)
    target_ref = _resolve_target(repo_root, target)
    porcelain = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root)
    entries = parse_worktree_list(porcelain)
    merged = _merged_branches(repo_root, target=target_ref)
    existing = _existing_branches(repo_root)
    worktree_branches = {entry.branch for entry in entries if entry.branch is not None}
    patch_evidence = {
        ref: evidence
        for ref in existing
        if ref not in merged
        if ref in worktree_branches
        if (evidence := _branch_patch_equivalence(repo_root, target_ref, ref)) is not None
    }
    heads_by_branch = {entry.branch: entry.head for entry in entries if entry.branch is not None}
    for ref, handoff in _lane_handoffs(repo_root).items():
        if ref not in worktree_branches:
            continue
        evidence = patch_evidence.setdefault(ref, {})
        evidence["handoff_status"] = handoff.get("status")
        evidence["handoff_head"] = handoff.get("head")
        evidence["handoff_head_matches"] = handoff.get("head") == heads_by_branch.get(ref)
        commits = handoff.get("commits")
        paths = handoff.get("changed_paths")
        evidence["handoff_commits"] = commits if isinstance(commits, list) else []
        evidence["handoff_changed_path_count"] = len(paths) if isinstance(paths, list) else 0
        if isinstance(handoff.get("error"), str):
            evidence["handoff_error"] = handoff["error"]
    candidates = classify_candidates(
        entries,
        repo_root=repo_root,
        merged=merged,
        existing=existing,
        patch_evidence=patch_evidence,
        main_worktree=entries[0].path if entries else None,
    )
    target_head = _run_git_nullable(["rev-parse", target_ref], cwd=repo_root)
    proven: list[GcCandidate] = []
    for candidate in candidates:
        proof: dict[str, object] = {
            "candidate_head": candidate.entry.head,
            "target": target_ref,
            "target_head": target_head,
            "state": candidate.reason,
        }
        proven.append(replace(candidate, proof=proof))
    return proven, entries


def _owns_exact_worktree(repo_root: Path, entry: WorktreeEntry) -> bool:
    """Prove the current porcelain record owns exactly this path and ref."""
    try:
        records = parse_worktree_list(_run_git(["worktree", "list", "--porcelain"], cwd=repo_root))
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    matches = [
        current
        for current in records
        if current.path.resolve() == entry.path.resolve()
        and current.branch == entry.branch
        and current.detached == entry.detached
        and current.bare == entry.bare
    ]
    return len(matches) == 1


def _recheck_removal_proof(candidate: GcCandidate, *, repo_root: Path) -> str | None:
    """Return a block reason when the collection snapshot is no longer true."""
    proof = candidate.proof
    if proof is None or not proof.get("candidate_head") or not proof.get("target"):
        return "stale-proof"
    try:
        records = parse_worktree_list(_run_git(["worktree", "list", "--porcelain"], cwd=repo_root))
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "ownership-changed"
    expected = candidate.entry
    identity_matches = [
        entry
        for entry in records
        if entry.path.resolve() == expected.path.resolve()
        and entry.branch == expected.branch
        and entry.detached == expected.detached
        and entry.bare == expected.bare
    ]
    if len(identity_matches) != 1:
        return "ownership-changed"
    current = identity_matches[0]
    if current.head != expected.head or current.locked != expected.locked or current.prunable != expected.prunable:
        return "stale-proof"

    target = str(proof["target"])
    target_head = _run_git_nullable(["rev-parse", target], cwd=repo_root)
    if target_head is None or target_head != proof.get("target_head"):
        return "stale-proof"

    if candidate.entry.branch is None:
        return None
    if candidate.reason == "merged":
        if candidate.entry.branch not in _merged_branches(repo_root, target=target):
            return "stale-proof"
    elif candidate.reason == "squash-equivalent":
        evidence = _branch_patch_equivalence(repo_root, target, candidate.entry.branch)
        if evidence is None or evidence.get("patch_equivalent") is not True:
            return "stale-proof"
    elif candidate.reason == "branch-deleted" and candidate.entry.branch in _existing_branches(repo_root):
        return "stale-proof"
    return None


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _RENAMEAT2.restype = ctypes.c_int


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename only when target is absent (Linux renameat2)."""
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 unavailable")
    parent_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        result = _RENAMEAT2(
            parent_fd,
            os.fsencode(source.name),
            parent_fd,
            os.fsencode(target.name),
            1,
        )
    finally:
        os.close(parent_fd)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), target)


def _restore_quarantine(quarantine: Path, path: Path) -> str | None:
    """Restore quarantine, preserving both sides when the target was replaced."""
    if path.exists() or path.is_symlink():
        return "restore-collision-preserved-quarantine-and-replacement"
    try:
        _rename_noreplace(quarantine, path)
    except FileExistsError:
        return "restore-collision-preserved-quarantine-and-replacement"
    except OSError:
        if path.exists() or path.is_symlink():
            return "restore-collision-preserved-quarantine-and-replacement"
        return "restore-failed-preserved-quarantine"
    return None


_Snapshot = tuple[int, int, int, int, str]


def _content_digest(path: Path) -> str:
    """Hash a regular file through an O_NOFOLLOW descriptor."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(fd)


def _path_digest(path: Path, mode: int) -> str:
    """Hash a regular file or the lexical target of a symbolic link."""
    if stat.S_ISREG(mode):
        return _content_digest(path)
    if stat.S_ISLNK(mode):
        return hashlib.sha256(b"symlink\0" + os.fsencode(os.readlink(path))).hexdigest()
    raise OSError(errno.EINVAL, "unsupported worktree item", path)


def _snapshot_worktree_files(path: Path, disposable_paths: tuple[str, ...]) -> dict[str, _Snapshot]:
    """Capture tracked and approved generated files before quarantine."""
    names = set(_run_git(["ls-files", "-z"], cwd=path).split("\0"))
    names.update(disposable_paths)
    snapshot: dict[str, _Snapshot] = {}
    for name in names:
        if not name:
            continue
        try:
            item = os.lstat(path / name)
            digest = _path_digest(path / name, item.st_mode)
        except OSError:
            continue
        snapshot[name] = (item.st_dev, item.st_ino, item.st_mode, item.st_size, digest)
    try:
        item = os.lstat(path / ".git")
        digest = _content_digest(path / ".git")
    except OSError:
        pass
    else:
        snapshot[".git"] = (item.st_dev, item.st_ino, item.st_mode, item.st_size, digest)
    return snapshot


def _remove_snapshot_files(root: Path, snapshot: dict[str, _Snapshot]) -> bool:
    """Remove only unchanged files from a quarantined tree, preserving races."""
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        parents: set[str] = set()
        for relative, expected in sorted(snapshot.items(), key=lambda item: item[0].count("/"), reverse=True):
            parts = relative.split("/")
            parent_fd = root_fd
            opened: list[int] = []
            try:
                for component in parts[:-1]:
                    parent_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    opened.append(parent_fd)
                try:
                    item = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                except OSError:
                    continue
                try:
                    digest = _path_digest(Path(root, *parts), item.st_mode)
                except OSError:
                    continue
                actual = (item.st_dev, item.st_ino, item.st_mode, item.st_size, digest)
                if actual == expected:
                    os.unlink(parts[-1], dir_fd=parent_fd)
                    parents.update("/".join(parts[:index]) for index in range(1, len(parts)))
            except OSError:
                continue
            finally:
                for fd in reversed(opened):
                    os.close(fd)
        for relative in sorted(parents, key=lambda value: value.count("/"), reverse=True):
            with contextlib.suppress(OSError):
                os.rmdir(relative, dir_fd=root_fd)
        return True
    finally:
        os.close(root_fd)


def _remove_admin_tree(fd: int) -> None:
    """Remove a Git worktree admin directory using only retained dirfds."""
    for name in os.listdir(fd):
        try:
            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        except OSError:
            os.unlink(name, dir_fd=fd)
            continue
        try:
            _remove_admin_tree(child_fd)
        finally:
            os.close(child_fd)
        os.rmdir(name, dir_fd=fd)


def _read_gitlink(path: Path) -> str:
    """Read a worktree gitlink without following a replacement symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        return os.read(fd, 4096).decode("utf-8").strip()
    finally:
        os.close(fd)


def _lexical_abs(path: str | Path, *, base: Path | None = None) -> str:
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raw = os.path.join(os.fspath(base or Path.cwd()), raw)
    return os.path.normpath(os.path.abspath(raw))


def _unregister_worktree_admin(repo_root: Path, entry: WorktreeEntry, quarantine: Path) -> str | None:
    """Unregister metadata using lexical checks, then retained no-follow fds."""
    try:
        # Recheck the live porcelain record immediately before mutating admin
        # metadata.  In particular, a newly locked worktree is never removed.
        current = parse_worktree_list(_run_git(["worktree", "list", "--porcelain"], cwd=repo_root))
        expected_path = _lexical_abs(entry.path)
        if not any(
            _lexical_abs(item.path) == expected_path and item.branch == entry.branch and not item.locked
            for item in current
        ):
            return "worktree-locked-or-ownership-changed-preserved-quarantine"

        prefix, admin_text = _read_gitlink(quarantine / ".git").split(" ", 1)
        if prefix != "gitdir:" or not admin_text:
            return "invalid-worktree-gitlink-preserved-quarantine"
        common_text = _run_git(["rev-parse", "--git-common-dir"], cwd=repo_root)
        common = Path(_lexical_abs(common_text, base=repo_root))
        worktrees = Path(_lexical_abs(common / "worktrees"))
        admin = _lexical_abs(admin_text, base=quarantine)
        worktrees_text = os.fspath(worktrees)
        prefix_text = worktrees_text + os.sep
        if not admin.startswith(prefix_text):
            return "invalid-worktree-admin-path-preserved-quarantine"
        relative_name = admin[len(prefix_text) :]
        if not relative_name or os.sep in relative_name or relative_name in {".", ".."}:
            return "invalid-worktree-admin-path-preserved-quarantine"
        expected_gitdir = _lexical_abs(entry.path / ".git")
        parent_fd = os.open(worktrees, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            admin_fd = os.open(relative_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                try:
                    locked_fd = os.open("locked", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=admin_fd)
                except FileNotFoundError:
                    pass
                else:
                    os.close(locked_fd)
                    return "worktree-locked-or-ownership-changed-preserved-quarantine"
                admin_gitdir = os.open("gitdir", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=admin_fd)
                try:
                    recorded_text = os.read(admin_gitdir, 4096).decode("utf-8").strip()
                finally:
                    os.close(admin_gitdir)
                if _lexical_abs(recorded_text, base=worktrees) != expected_gitdir:
                    return "worktree-admin-ownership-changed-preserved-quarantine"
                _remove_admin_tree(admin_fd)
                admin_stat = os.fstat(admin_fd)
            finally:
                os.close(admin_fd)
            current_admin = os.stat(relative_name, dir_fd=parent_fd, follow_symlinks=False)
            if (current_admin.st_dev, current_admin.st_ino) != (admin_stat.st_dev, admin_stat.st_ino):
                return "worktree-admin-replaced-preserved-quarantine"
            os.rmdir(relative_name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "worktree-admin-unregister-failed-preserved-quarantine"
    return None


def _quarantine_name(path: Path) -> str:
    return f".{path.name}.polylogue-gc-{os.getpid()}-{uuid.uuid4().hex}"


def _rename_into_quarantine(path: Path) -> tuple[int, Path]:
    """Rename through a retained, no-symlink parent descriptor."""
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        item = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(item.st_mode):
            raise NotADirectoryError(path)
        name = _quarantine_name(path)
        os.rename(path.name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        return parent_fd, path.parent / name
    except BaseException:
        os.close(parent_fd)
        raise


def _quarantine_and_remove(entry: WorktreeEntry, *, repo_root: Path) -> tuple[bool, str | None]:
    """Unregister a clean worktree without ever deleting its live directory.

    The atomic rename closes the final inspection/removal gap: after the
    directory is moved out of its Git-registered path, any file created in it
    remains in the quarantine directory.  Git removes only the missing
    worktree registration; ``rmdir`` then removes the quarantine only when it
    is still empty, so a late ignored/untracked file is preserved.
    """
    path = entry.path
    if not path.exists() or not path.is_dir():
        return False, "missing-worktree"
    residue = inspect_residue(path)
    if residue.dirty:
        return False, "dirty"
    try:
        snapshot = _snapshot_worktree_files(path, residue.disposable_paths)
    except (OSError, subprocess.CalledProcessError):
        return False, "snapshot-failed"
    try:
        parent_fd, quarantine = _rename_into_quarantine(path)
    except OSError:
        return False, "quarantine-failed"

    try:
        if check_dirty(quarantine):
            detail = _restore_quarantine(quarantine, path)
            return False, detail or "dirty"
        if not (quarantine / ".git").exists() or (quarantine / ".git").is_symlink():
            detail = _restore_quarantine(quarantine, path)
            return False, detail or "missing-worktree-gitlink-preserved-quarantine"

        detail = _unregister_worktree_admin(repo_root, entry, quarantine)
        if detail is not None:
            restored = _restore_quarantine(quarantine, path)
            return False, restored or detail

        # Remove only tracked files whose device/inode/metadata/content still
        # match the pre-quarantine snapshot.
        if not _remove_snapshot_files(quarantine, snapshot):
            return True, "preserved-quarantine-data"
        try:
            quarantine.rmdir()
        except OSError:
            return True, "preserved-quarantine-data"
        return True, None
    finally:
        os.close(parent_fd)


def _delete_completed_agent_branch(candidate: GcCandidate, *, repo_root: Path) -> bool:
    """Delete a generated lane branch after its worktree is removed."""
    branch_ref = candidate.entry.branch
    if candidate.reason not in {"merged", "squash-equivalent"} or branch_ref is None:
        return False
    branch = _branch_short_name(branch_ref)
    if not branch.startswith("worktree-agent-"):
        return False
    return _run_git_nullable(["branch", "-D", branch], cwd=repo_root) is not None


def apply_removals(candidates: list[GcCandidate], *, repo_root: Path, force: bool = False) -> list[dict[str, object]]:
    """Remove safe candidates and completed generated lane branches."""
    repo_root = normalize_repo_root(repo_root)
    results: list[dict[str, object]] = []
    removed = 0

    for c in candidates:
        if c.reason == "prunable":
            results.append({"path": str(c.entry.path), "removed": False, "reason": "prunable-skipped"})
            continue

        if c.safe and c.action == "remove":
            blocked = _recheck_removal_proof(c, repo_root=repo_root)
            if blocked is not None:
                results.append(
                    {
                        "path": str(c.entry.path),
                        "removed": False,
                        "reason": c.reason,
                        "blocked": blocked,
                    }
                )
                continue
            # Re-read filesystem state after all proof checks and immediately
            # before removal.  Collection-time cleanliness is not sufficient.
            if check_dirty(c.entry.path):
                results.append(
                    {
                        "path": str(c.entry.path),
                        "removed": False,
                        "reason": c.reason,
                        "blocked": "dirty",
                    }
                )
                continue
            ok, detail = _quarantine_and_remove(c.entry, repo_root=repo_root)
            branch_deleted = ok and _delete_completed_agent_branch(c, repo_root=repo_root)
            result: dict[str, object] = {
                "path": str(c.entry.path),
                "removed": ok,
                "reason": c.reason,
                "branch_deleted": branch_deleted,
            }
            if detail is not None:
                result["detail"] = detail
            results.append(result)
            if ok:
                removed += 1
            continue

        if force and c.action == "remove-force":
            blocked = _recheck_removal_proof(c, repo_root=repo_root)
            if blocked is not None:
                results.append(
                    {
                        "path": str(c.entry.path),
                        "removed": False,
                        "reason": c.reason,
                        "blocked": blocked,
                    }
                )
                continue
            if check_dirty(c.entry.path):
                results.append(
                    {
                        "path": str(c.entry.path),
                        "removed": False,
                        "reason": c.reason,
                        "blocked": "dirty",
                    }
                )
                continue
            ok, detail = _quarantine_and_remove(c.entry, repo_root=repo_root)
            branch_deleted = ok and _delete_completed_agent_branch(c, repo_root=repo_root)
            result = {
                "path": str(c.entry.path),
                "removed": ok,
                "reason": c.reason,
                "branch_deleted": branch_deleted,
            }
            if detail is not None:
                result["detail"] = detail
            results.append(result)
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
        help="Explicit target for merge check (default: current branch, or HEAD when detached).",
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
