"""Repository-root and repo-name normalization helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
_PLAIN_REPO_NAME_RE = re.compile(r"^[a-z0-9_.-]+$")
_NON_PROBING_ABSOLUTE_PREFIXES = (
    PurePosixPath("/mnt"),
    PurePosixPath("/media"),
    PurePosixPath("/run/media"),
    PurePosixPath("/neo-outer-realm"),
)


def _literal_local_path(value: str) -> str | None:
    """Interpret a declared filesystem path, never a token from prose."""
    if value.startswith("file://"):
        value = urlparse(value).path
    return value if value.startswith(("/", "~/")) else None


def _lexical_expanduser(value: str) -> str:
    if value == "~":
        return str(Path.home())
    if value.startswith("~/"):
        return f"{Path.home()}{value[1:]}"
    return value


def _is_non_probing_absolute_path(value: str) -> bool:
    expanded = _lexical_expanduser(value)
    pure = PurePosixPath(expanded)
    return pure.is_absolute() and any(
        pure == prefix or prefix in pure.parents for prefix in _NON_PROBING_ABSOLUTE_PREFIXES
    )


def _iter_repo_root_candidates(path: Path) -> tuple[Path, ...]:
    expanded = path.expanduser()
    current = expanded
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
    if path.name == "projects" and (".claude" in parts or (".config" in parts and "claude" in parts)):
        return True
    if path.name in {"projects", "sessions"} and (".codex" in parts or (".config" in parts and "codex" in parts)):
        return True
    return path.name == "blob-repository" and ".local" in parts and "state" in parts


def _git_ceiling_directories() -> tuple[str, ...]:
    """Resolved ``GIT_CEILING_DIRECTORIES`` entries.

    Git's semantics: a colon-separated list of absolute directories the
    upward repository walk must stop before examining. Empty and relative
    entries are ignored; entries are resolved because the walk's candidates
    are resolved.
    """
    raw = os.environ.get("GIT_CEILING_DIRECTORIES", "")
    ceilings: list[str] = []
    for entry in raw.split(":"):
        if not entry or not entry.startswith("/"):
            continue
        ceilings.append(str(Path(entry).expanduser().resolve(strict=False)))
    return tuple(sorted(set(ceilings)))


def _find_git_root(path: Path, ceilings: tuple[str, ...] = ()) -> Path | None:
    candidates = tuple(_iter_repo_root_candidates(path))
    starting_candidate = candidates[0] if candidates else None
    for candidate in candidates:
        if str(candidate) in ceilings and candidate != starting_candidate:
            return None
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
        parts = raw.split("/")
        if any(part.startswith(".") for part in parts):
            return None
        return _repo_name_from_slug(raw)
    return None


def normalize_repo_path(value: object) -> str | None:
    """Discover the current Git root of a complete structured local path.

    Paths remain literal, including whitespace and punctuation. Filesystem
    observations are fresh on every call: a negative or enclosing-root result
    cannot survive repository creation or deletion in the same process.
    """
    raw = str(value or "")
    path_candidate = _literal_local_path(raw)
    if path_candidate is None or _is_non_probing_absolute_path(path_candidate):
        return None
    git_root = _find_git_root(Path(path_candidate).expanduser().resolve(strict=False), _git_ceiling_directories())
    return str(git_root) if git_root is not None else None


def normalize_repo_name(value: object) -> str | None:
    raw = str(value or "")
    if not raw:
        return None
    repo_path = normalize_repo_path(raw)
    if repo_path is not None:
        return Path(repo_path).name or None
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


def repo_relative_path(path: str, root_path: str) -> str:
    """Strip ``root_path`` from ``path`` so it is comparable across checkouts.

    polylogue-cijx.4 decision 2: paths are repo-relative. The same file
    edited from two worktree checkouts of one repository is otherwise two
    different absolute paths and no cross-session "which files did I touch"
    question can be answered. This is a pure read-time projection -- it does
    not mutate any stored ``tool_path``/``root_path`` value.

    Returns ``path`` unchanged when ``root_path`` is empty or is not a
    prefix of ``path`` (e.g. the checkout root could not be resolved for
    this session, or the path is outside any known checkout).
    """
    candidate = path.strip()
    root = root_path.strip()
    if not candidate or not root:
        return candidate
    normalized_root = root.rstrip("/")
    if candidate == normalized_root:
        return ""
    prefix = f"{normalized_root}/"
    if candidate.startswith(prefix):
        return candidate[len(prefix) :]
    return candidate


__all__ = [
    "normalize_repo_name",
    "normalize_repo_names",
    "normalize_repo_path",
    "normalize_repo_paths",
    "repo_relative_path",
]
