"""Repository-root and repo-name normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

_PATH_DELIMITERS = {"#", "`", '"', "\\", ":", "(", ")", "\n", "\r", "\t", " ", "<", ">", ",", ";", "'"}
_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
_PLAIN_REPO_NAME_RE = re.compile(r"^[a-z0-9_.-]+$")


def _extract_local_path_candidate(value: str) -> str | None:
    raw = value.strip()
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = parsed.path
    if not raw.startswith(("/", "~/")):
        return None
    chars: list[str] = []
    for char in raw:
        if char in _PATH_DELIMITERS:
            break
        chars.append(char)
    candidate = "".join(chars).strip()
    return candidate or None


def _iter_repo_root_candidates(path: Path) -> tuple[Path, ...]:
    expanded = path.expanduser()
    current = expanded if not expanded.suffix else expanded.parent
    if current == Path("."):
        return ()
    return (current, *current.parents)


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _is_non_work_repo_root(path: Path) -> bool:
    parts = path.parts
    if path.name == "projects" and (
        ".claude" in parts or (".config" in parts and "claude" in parts)
    ):
        return True
    if path.name in {"projects", "sessions"} and (
        ".codex" in parts or (".config" in parts and "codex" in parts)
    ):
        return True
    return path.name == "blob-repository" and ".local" in parts and "state" in parts


def _find_git_root(path: Path) -> Path | None:
    for candidate in _iter_repo_root_candidates(path):
        if candidate.name == ".git" and _path_exists(candidate):
            repo_root = candidate.parent
            if not _is_non_work_repo_root(repo_root):
                return repo_root
        if _path_exists(candidate / ".git") and not _is_non_work_repo_root(candidate):
            return candidate
    return None


def _repo_name_from_slug(value: str) -> str | None:
    slug = value.strip().lstrip("/").rstrip("/")
    if not slug:
        return None
    name = slug.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or None


def _repo_name_from_remote(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("git@") and ":" in raw:
        return _repo_name_from_slug(raw.split(":", 1)[1])
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "ssh", "git"}:
        return _repo_name_from_slug(parsed.path)
    if _REPO_SLUG_RE.fullmatch(raw):
        return _repo_name_from_slug(raw)
    return None


@lru_cache(maxsize=4096)
def normalize_repo_path(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path_candidate = _extract_local_path_candidate(raw)
    if path_candidate is None:
        return None
    git_root = _find_git_root(Path(path_candidate).expanduser().resolve(strict=False))
    return str(git_root) if git_root is not None else None


@lru_cache(maxsize=4096)
def normalize_repo_name(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    repo_path = normalize_repo_path(raw)
    if repo_path is not None:
        name = Path(repo_path).name.strip()
        return name or None
    return _repo_name_from_remote(raw)


def normalize_repo_names(
    values: Iterable[object] = (),
    *,
    repo_paths: Iterable[object] = (),
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if raw and _PLAIN_REPO_NAME_RE.fullmatch(raw):
            normalized.add(raw)
            continue
        repo_name = normalize_repo_name(value)
        if repo_name is not None:
            normalized.add(repo_name)
    for repo_path in repo_paths:
        repo_root = normalize_repo_path(repo_path)
        if repo_root is None:
            continue
        repo_name = Path(repo_root).name
        if repo_name:
            normalized.add(repo_name)
    return tuple(sorted(normalized))


def normalize_repo_paths(values: Iterable[object]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        repo = normalize_repo_path(value)
        if repo is not None:
            normalized.add(repo)
    return tuple(sorted(normalized))


__all__ = [
    "normalize_repo_name",
    "normalize_repo_names",
    "normalize_repo_path",
    "normalize_repo_paths",
]
