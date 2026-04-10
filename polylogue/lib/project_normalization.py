"""Canonical workspace project and repo normalization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

_WORKSPACE_ROOT = Path("/realm/project")
_WORKSPACE_MARKER = "/realm/project/"
_PREFIX_DELIMITERS = {"#", "`", '"', "\\", "/", ":", "(", ")", "\n", "\r", "\t", " "}


@lru_cache(maxsize=1)
def workspace_project_names() -> tuple[str, ...]:
    if not _WORKSPACE_ROOT.exists():
        return ()
    return tuple(
        sorted(entry.name for entry in _WORKSPACE_ROOT.iterdir() if entry.is_dir() and not entry.name.startswith("."))
    )


@lru_cache(maxsize=1)
def _workspace_project_name_set() -> frozenset[str]:
    return frozenset(workspace_project_names())


def _repo_path_project_token(value: str) -> str | None:
    if _WORKSPACE_MARKER not in value:
        return None
    tail = value.split(_WORKSPACE_MARKER, 1)[1]
    if not tail:
        return None
    token_chars: list[str] = []
    for char in tail:
        if char in {"/", "\n", "\r", "\t", " "}:
            break
        token_chars.append(char)
    token = "".join(token_chars).strip()
    return token or None


@lru_cache(maxsize=4096)
def normalize_project_name(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    project_names = _workspace_project_name_set()
    if not project_names:
        return None

    repo_token = _repo_path_project_token(raw)
    if repo_token is not None:
        raw = repo_token

    if raw in project_names:
        return raw

    for project in workspace_project_names():
        if raw.startswith(project) and len(raw) > len(project) and raw[len(project)] in _PREFIX_DELIMITERS:
            return project
    return None


@lru_cache(maxsize=4096)
def normalize_repo_path(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    project = normalize_project_name(raw)
    if project is None:
        return None
    return str(_WORKSPACE_ROOT / project)


def normalize_project_names(
    values: Iterable[object] = (),
    *,
    repo_paths: Iterable[object] = (),
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        project = normalize_project_name(value)
        if project is not None:
            normalized.add(project)
    for repo_path in repo_paths:
        repo = normalize_repo_path(repo_path)
        if repo is None:
            continue
        project = Path(repo).name
        if project:
            normalized.add(project)
    return tuple(sorted(normalized))


def normalize_repo_paths(values: Iterable[object]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in values:
        repo = normalize_repo_path(value)
        if repo is not None:
            normalized.add(repo)
    return tuple(sorted(normalized))


def normalize_project_breakdown(values: Mapping[str, int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key, value in values.items():
        project = normalize_project_name(key)
        if project is None:
            continue
        counts[project] += int(value or 0)
    return dict(counts)


__all__ = [
    "normalize_project_breakdown",
    "normalize_project_name",
    "normalize_project_names",
    "normalize_repo_path",
    "normalize_repo_paths",
    "workspace_project_names",
]
