"""Source path discovery and cursor-aware walk setup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.config import Source
from polylogue.types import Provider

from . import cursor as _cursor
from .assembly import SidecarData, get_assembly_spec

_SUPPORTED_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson", ".zip"})
_SUPPORTED_DOUBLE_EXTENSIONS = frozenset({".jsonl.txt"})
_SKIP_DIRS = frozenset({"analysis", "__pycache__", ".git", "node_modules"})


def _has_supported_extension(path: Path) -> bool:
    name_lower = path.name.lower()
    for ext in _SUPPORTED_DOUBLE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def _walk_source_paths(base: Path) -> list[Path]:
    paths: list[Path] = []
    for root, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for filename in files:
            file_path = Path(root) / filename
            if _has_supported_extension(file_path):
                paths.append(file_path)
    return sorted(paths)


def _resolve_source_paths(source: Source) -> list[Path]:
    if not source.path:
        return []
    base = source.path.expanduser()
    if base.is_dir():
        return _walk_source_paths(base)
    if base.is_file():
        return [base]
    return []


@dataclass
class _SourceWalkSetup:
    paths: list[Path]
    paths_to_process: list[tuple[Path, str | None]]
    skipped_mtime: int
    sidecar_data: SidecarData = field(default_factory=dict)


def _setup_source_walk(
    source: Source,
    *,
    cursor_state: dict[str, object] | None,
    include_mtime: bool,
    known_mtimes: dict[str, str] | None,
    discover_sidecars: bool,
) -> _SourceWalkSetup | None:
    paths = _resolve_source_paths(source)
    _cursor._initialize_cursor_state(cursor_state, paths)
    if not paths:
        return None
    paths_to_process, skipped_mtime = _cursor._select_paths_for_processing(
        paths,
        include_file_mtime=include_mtime,
        known_mtimes=known_mtimes,
    )
    sidecar_data: SidecarData = {}
    if discover_sidecars:
        provider = Provider.from_string(source.name)
        spec = get_assembly_spec(provider)
        if spec is not None:
            sidecar_data = spec.discover_sidecars(paths)
    return _SourceWalkSetup(
        paths=paths,
        paths_to_process=paths_to_process,
        skipped_mtime=skipped_mtime,
        sidecar_data=sidecar_data,
    )


__all__ = [
    "_SourceWalkSetup",
    "_SUPPORTED_DOUBLE_EXTENSIONS",
    "_SUPPORTED_EXTENSIONS",
    "_SKIP_DIRS",
    "_has_supported_extension",
    "_resolve_source_paths",
    "_setup_source_walk",
    "_walk_source_paths",
]
