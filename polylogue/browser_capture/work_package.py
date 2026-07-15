"""Deterministic targeted project packs for authenticated Sol Pro workers."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import orjson

from polylogue.browser_capture.sol_pro_prompt import build_sol_pro_prompt

WORK_PACKAGE_MAX_BYTES = 16 * 1024 * 1024
WORK_PACKAGE_MAX_FILES = 2_000
WORK_PACKAGE_MAX_ENTRY_BYTES = 16 * 1024 * 1024
WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_SKIP_PARTS = frozenset({".git", ".cache", ".direnv", ".venv", "__pycache__", "node_modules"})


@dataclass(frozen=True)
class SolProWorkPackage:
    """One immutable context-pack attachment and its integrity receipt."""

    name: str
    content: bytes
    sha256: str
    entry_count: int
    uncompressed_bytes: int


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        command = " ".join(("git", *args))
        detail = result.stdout.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{command} failed: {detail or f'exit {result.returncode}'}")
    return result.stdout


def _json_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"


def _safe_relative_path(repo_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else repo_root / value
    if candidate.is_symlink():
        raise ValueError(f"work-package path must not be a symlink: {value}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"work-package path escapes repository: {value}") from exc
    if any(part in _SKIP_PARTS for part in relative.parts):
        raise ValueError(f"work-package path is excluded: {value}")
    return relative


def _selected_files(repo_root: Path, paths: Sequence[Path]) -> list[Path]:
    selected: set[Path] = set()
    for value in paths:
        relative = _safe_relative_path(repo_root, value)
        resolved = repo_root / relative
        if resolved.is_file():
            selected.add(relative)
            continue
        for candidate in resolved.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            candidate_relative = candidate.relative_to(repo_root)
            if any(part in _SKIP_PARTS for part in candidate_relative.parts):
                continue
            selected.add(candidate_relative)
    return sorted(selected, key=lambda path: path.as_posix())


def _all_worktree_files(repo_root: Path) -> list[Path]:
    raw = _git(repo_root, "ls-files", "-co", "--exclude-standard", "-z")
    paths: list[Path] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = Path(item.decode("utf-8", errors="surrogateescape"))
        if any(part in _SKIP_PARTS for part in relative.parts):
            continue
        candidate = repo_root / relative
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(relative)
    return sorted(set(paths), key=lambda path: path.as_posix())


def _read_beads(repo_root: Path, bead_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not bead_ids:
        return []
    issue_path = repo_root / ".beads" / "issues.jsonl"
    requested = set(bead_ids)
    found: dict[str, dict[str, Any]] = {}
    for line in issue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_id = record.get("id") if isinstance(record, dict) else None
        if isinstance(record_id, str) and record_id in requested:
            found[record_id] = record
    missing = requested - found.keys()
    if missing:
        raise ValueError(f"unknown Beads ids: {', '.join(sorted(missing))}")
    return [found[bead_id] for bead_id in bead_ids]


def _add_entry(entries: dict[str, tuple[bytes, str]], path: str, content: bytes, purpose: str) -> None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path in entries:
        raise ValueError(f"invalid or duplicate work-package path: {path}")
    if len(entries) >= WORK_PACKAGE_MAX_FILES:
        raise ValueError(f"work package exceeds {WORK_PACKAGE_MAX_FILES} files")
    if len(content) > WORK_PACKAGE_MAX_ENTRY_BYTES:
        raise ValueError(f"work-package entry exceeds {WORK_PACKAGE_MAX_ENTRY_BYTES} bytes: {path}")
    if sum(len(value) for value, _purpose in entries.values()) + len(content) > WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"work package exceeds {WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES} uncompressed bytes")
    entries[path] = (content, purpose)


def _preflight_paths(paths: Sequence[Path]) -> None:
    """Reject oversized path sets before any selected source is buffered."""
    if len(paths) > WORK_PACKAGE_MAX_FILES:
        raise ValueError(f"work package exceeds {WORK_PACKAGE_MAX_FILES} files")
    total = 0
    for path in paths:
        size = path.stat().st_size
        if size > WORK_PACKAGE_MAX_ENTRY_BYTES:
            raise ValueError(f"work-package entry exceeds {WORK_PACKAGE_MAX_ENTRY_BYTES} bytes: {path}")
        total += size
        if total > WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"work package exceeds {WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES} uncompressed bytes")


def _render_tar(entries: dict[str, tuple[bytes, str]]) -> bytes:
    manifest = {
        "schema": "polylogue-sol-pro-context-pack-v1",
        "files": [
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "purpose": purpose,
            }
            for path, (content, purpose) in sorted(entries.items())
        ],
    }
    material = {"MANIFEST.json": (_json_bytes(manifest), "Integrity manifest"), **entries}
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path, (content, _purpose) in sorted(material.items()):
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def build_sol_pro_work_package(
    *,
    repo_root: Path,
    job_title: str,
    scope_prompt: str,
    bead_ids: Sequence[str] = (),
    source_paths: Sequence[Path] = (),
    verification_paths: Sequence[Path] = (),
    full_worktree_fallback: bool = False,
) -> SolProWorkPackage:
    """Build a deterministic, manifest-checked project snapshot.

    The default is deliberately targeted. ``full_worktree_fallback`` is an
    explicit escape hatch and never silently expands a requested source set.
    """
    root = repo_root.resolve(strict=True)
    if not (root / ".git").exists():
        # Linked worktrees use a .git file, so existence rather than is_dir is intentional.
        raise ValueError(f"not a git repository root: {root}")
    entries: dict[str, tuple[bytes, str]] = {}
    full_prompt = build_sol_pro_prompt(job_title, scope_prompt)
    _add_entry(entries, "PROMPT.md", full_prompt.encode(), "Exact submitted worker prompt")
    _add_entry(entries, "MISSION.txt", job_title.strip().encode() + b"\n", "Human-readable job title")
    _add_entry(entries, "SCOPE.md", scope_prompt.strip().encode() + b"\n", "Job-specific scope")

    for instruction_name in ("CLAUDE.md", "AGENTS.md"):
        path = root / instruction_name
        if path.exists():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"repository instruction escapes project root: {instruction_name}") from exc
            _preflight_paths([resolved])
            _add_entry(
                entries,
                f"INSTRUCTIONS/{instruction_name}",
                resolved.read_bytes(),
                "Repository agent instructions",
            )

    bead_records = _read_beads(root, bead_ids)
    if bead_records:
        _add_entry(entries, "BEADS/selected.json", _json_bytes(bead_records), "Full selected Beads records")
        _add_entry(
            entries,
            "BEADS/selected-ids.txt",
            ("\n".join(bead_ids) + "\n").encode(),
            "Ordered work-item scope",
        )

    selected = _all_worktree_files(root) if full_worktree_fallback else _selected_files(root, source_paths)
    verification = _selected_files(root, verification_paths)
    _preflight_paths([*(root / relative for relative in selected), *(root / relative for relative in verification)])
    diff_pathspec = [path.as_posix() for path in selected]
    unstaged_diff = _git(root, "diff", "--binary", "--no-ext-diff", "--", *diff_pathspec) if diff_pathspec else b""
    staged_diff = (
        _git(root, "diff", "--cached", "--binary", "--no-ext-diff", "--", *diff_pathspec) if diff_pathspec else b""
    )
    git_entries = {
        "GIT/HEAD.txt": (_git(root, "rev-parse", "HEAD"), "Base revision"),
        "GIT/branch.txt": (_git(root, "branch", "--show-current"), "Source branch"),
        "GIT/status.txt": (_git(root, "status", "--short", "--branch"), "Working-tree state"),
        "GIT/log.txt": (_git(root, "log", "--oneline", "--decorate", "-20"), "Recent revision context"),
        "GIT/working-tree.patch": (unstaged_diff, "Unstaged patch for selected source footprint"),
        "GIT/staged.patch": (staged_diff, "Staged patch for selected source footprint"),
    }
    for entry_path, (content, purpose) in git_entries.items():
        _add_entry(entries, entry_path, content, purpose)

    for relative in selected:
        _add_entry(
            entries,
            f"REPO/{relative.as_posix()}",
            (root / relative).read_bytes(),
            "Selected repository source" if not full_worktree_fallback else "Explicit full-worktree fallback",
        )
    for relative in verification:
        _add_entry(
            entries, f"VERIFICATION/{relative.as_posix()}", (root / relative).read_bytes(), "Verification receipt"
        )

    uncompressed_bytes = sum(len(content) for content, _purpose in entries.values())
    report = {
        "selected_beads": list(bead_ids),
        "selected_source_files": len(selected),
        "full_worktree_fallback": full_worktree_fallback,
        "entry_count_excluding_manifest": len(entries),
        "uncompressed_bytes_excluding_manifest": uncompressed_bytes,
    }
    _add_entry(entries, "PACK-REPORT.json", _json_bytes(report), "Pack size and selection receipt")
    content = _render_tar(entries)
    if len(content) > WORK_PACKAGE_MAX_BYTES:
        raise ValueError(
            f"work package is {len(content)} bytes; targeted attachment limit is {WORK_PACKAGE_MAX_BYTES} bytes"
        )
    digest = hashlib.sha256(content).hexdigest()
    return SolProWorkPackage(
        name=f"polylogue-sol-pro-context-{digest[:16]}.tar.gz",
        content=content,
        sha256=digest,
        entry_count=len(entries) + 1,
        uncompressed_bytes=uncompressed_bytes,
    )


__all__ = [
    "SolProWorkPackage",
    "WORK_PACKAGE_MAX_BYTES",
    "WORK_PACKAGE_MAX_ENTRY_BYTES",
    "WORK_PACKAGE_MAX_FILES",
    "WORK_PACKAGE_MAX_UNCOMPRESSED_BYTES",
    "build_sol_pro_work_package",
]
