"""Source path discovery and cursor-aware walk setup."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from . import cursor as _cursor
from .assembly import SidecarData, get_assembly_spec
from .origin_specs import SourceClass, recognize_source_class

_SUPPORTED_EXTENSIONS = frozenset({".json", ".jsonl", ".ndjson", ".zip"})
_SUPPORTED_DOUBLE_EXTENSIONS = frozenset({".jsonl.txt"})
_HERMES_SQLITE_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3"})
_SKIP_DIRS = frozenset({"analysis", "__pycache__", ".git", "node_modules"})


@dataclass(frozen=True, slots=True)
class SourceRootCensus:
    """Read-only accounting for every candidate visible to a source walk."""

    provider: Provider
    root: Path
    candidate_count: int
    disposition_counts: dict[SourceClass, int]
    unexplained_candidates: tuple[Path, ...]
    inspection_bytes: int
    inspection_seconds: float

    @property
    def accounted_count(self) -> int:
        return sum(self.disposition_counts.values())

    @property
    def is_complete(self) -> bool:
        return self.accounted_count == self.candidate_count and not self.unexplained_candidates


def census_source_root(root: Path, *, provider: Provider) -> SourceRootCensus:
    """Classify the current candidate denominator without parsing or writing.

    The walk and recognizer are the production discovery and admission
    authorities.  This function only records their result, so a candidate
    cannot disappear from the denominator merely because admission refuses it.
    """
    started = time.perf_counter()
    candidates = _walk_source_paths(root, provider=provider)
    counts: dict[SourceClass, int] = {"session": 0, "non_session": 0, "unsupported": 0}
    unexplained: list[Path] = []
    inspection_bytes = 0
    for path in candidates:
        try:
            inspection_bytes += path.stat().st_size
            recognition = recognize_source_class(provider, path)
        except (OSError, ValueError):
            recognition = None
        if recognition is None:
            unexplained.append(path)
        else:
            counts[recognition.source_class] += 1
    return SourceRootCensus(
        provider=provider,
        root=root,
        candidate_count=len(candidates),
        disposition_counts=counts,
        unexplained_candidates=tuple(unexplained),
        inspection_bytes=inspection_bytes,
        inspection_seconds=time.perf_counter() - started,
    )


def _empty_sidecar_data() -> SidecarData:
    return {}


def _has_supported_extension(path: Path) -> bool:
    name_lower = path.name.lower()
    for ext in _SUPPORTED_DOUBLE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def _is_supported_source_path(path: Path, *, provider: Provider) -> bool:
    if _has_supported_extension(path):
        return True
    if (
        provider is Provider.ANTIGRAVITY
        and "brain" in {part.lower() for part in path.parts[:-1]}
        and path.suffix.lower() == ".md"
    ):
        return True
    # A broad Hermes root must enumerate every SQLite candidate so the
    # OriginSpec recognizer can publish a typed unsupported/non-session
    # observation.  Structural inspection belongs to admission, not the walk;
    # otherwise unrelated databases disappear from the source denominator.
    return provider is Provider.HERMES and path.suffix.lower() in _HERMES_SQLITE_EXTENSIONS


def _walk_source_paths(base: Path, *, provider: Provider = Provider.UNKNOWN) -> list[Path]:
    paths: list[Path] = []
    for root, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for filename in files:
            file_path = Path(root) / filename
            if _is_supported_source_path(file_path, provider=provider):
                paths.append(file_path)
    return sorted(paths)


def _resolve_source_paths(source: Source) -> list[Path]:
    if not source.path:
        return []
    base = source.path.expanduser()
    if base.is_dir():
        return _walk_source_paths(base, provider=Provider.from_string(source.name))
    if base.is_file():
        return [base]
    return []


@dataclass
class _SourceWalkSetup:
    paths: list[Path]
    paths_to_process: list[tuple[Path, str | None]]
    skipped_mtime: int
    sidecar_data: SidecarData = field(default_factory=_empty_sidecar_data)


def _setup_source_walk(
    source: Source,
    *,
    cursor_state: CursorStatePayload | None,
    include_mtime: bool,
    known_mtimes: dict[str, str] | None,
    known_cursors: dict[str, dict[str, object]] | None = None,
    discover_sidecars: bool,
    blob_store: BlobStore | None = None,
) -> _SourceWalkSetup | None:
    paths = _resolve_source_paths(source)
    _cursor._initialize_cursor_state(cursor_state, paths)
    if not paths:
        return None
    paths_to_process, skipped_mtime = _cursor._select_paths_for_processing(
        paths,
        include_file_mtime=include_mtime,
        known_mtimes=known_mtimes,
        known_cursors=known_cursors,
    )
    sidecar_data = _empty_sidecar_data()
    if discover_sidecars:
        provider = Provider.from_string(source.name)
        spec = get_assembly_spec(provider)
        if spec is not None:
            sidecar_data = spec.discover_sidecars(paths, blob_store=blob_store)
    return _SourceWalkSetup(
        paths=paths,
        paths_to_process=paths_to_process,
        skipped_mtime=skipped_mtime,
        sidecar_data=sidecar_data,
    )


__all__ = [
    "SourceRootCensus",
    "_SourceWalkSetup",
    "_SUPPORTED_DOUBLE_EXTENSIONS",
    "_SUPPORTED_EXTENSIONS",
    "_SKIP_DIRS",
    "_has_supported_extension",
    "_is_supported_source_path",
    "_resolve_source_paths",
    "_setup_source_walk",
    "_walk_source_paths",
    "census_source_root",
]
