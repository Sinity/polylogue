"""Source path discovery and cursor-aware walk setup."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from . import cursor as _cursor
from .assembly import SidecarData, get_assembly_spec
from .origin_specs import SourceClass, artifact_rule_for_path, recognize_source_class
from .source_root_admission import SourceRootRefusedError, containing_archive_root, refuse_non_capture_source_root

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
    candidate_bytes: int
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
    ``candidate_bytes`` sums candidate entry sizes; it is not read-I/O
    telemetry, since recognition reads only a bounded prefix of each file.
    """
    started = time.perf_counter()
    if provider is Provider.ANTIGRAVITY:
        from polylogue.sources.parsers.antigravity import AntigravitySourceRole, census_source

        source_census = census_source(root)
        antigravity_counts = source_census.counts
        disposition_counts: dict[SourceClass, int] = {
            "session": antigravity_counts[AntigravitySourceRole.CONVERSATION_PROTOBUF]
            + antigravity_counts[AntigravitySourceRole.EXPORT_DOCUMENT],
            "non_session": antigravity_counts[AntigravitySourceRole.BRAIN_DOCUMENT]
            + antigravity_counts[AntigravitySourceRole.METADATA_SIDECAR],
            "unsupported": antigravity_counts[AntigravitySourceRole.UNKNOWN],
        }
        return SourceRootCensus(
            provider=provider,
            root=root,
            candidate_count=len(source_census.items),
            disposition_counts=disposition_counts,
            unexplained_candidates=source_census.unexplained_items,
            candidate_bytes=sum(item.size_bytes for item in source_census.items),
            inspection_seconds=time.perf_counter() - started,
        )
    unexplained: list[Path] = []

    def record_walk_error(error: OSError) -> None:
        unexplained.append(Path(error.filename) if error.filename is not None else root)

    # The same file-or-directory resolution as ``_resolve_source_paths``: a
    # directly configured file is the one candidate. Under a directory the
    # census takes every supported-name entry before the regular-file
    # admission filter, so a refused link or FIFO stays in the denominator.
    walked = root.is_dir()
    if walked:
        candidates = [
            path
            for path in _iter_source_entries(root, onerror=record_walk_error)
            if _is_supported_source_path(path, provider=provider)
        ]
    else:
        candidates = [root] if root.is_file() else []
    counts: dict[SourceClass, int] = {"session": 0, "non_session": 0, "unsupported": 0}
    candidate_bytes = 0
    for path in candidates:
        try:
            observed = os.stat(path, follow_symlinks=not walked)
            candidate_bytes += observed.st_size
            if not stat.S_ISREG(observed.st_mode):
                counts["unsupported"] += 1
                continue
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
        candidate_bytes=candidate_bytes,
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
        and path.suffix.lower() == ".pb"
        and artifact_rule_for_path(provider, str(path)) is not None
    ):
        return True
    if provider is Provider.ANTIGRAVITY and path.suffix.lower() in _HERMES_SQLITE_EXTENSIONS:
        # The trajectory store has no stable basename.  Enumerate SQLite
        # candidates, then let the schema recognizer distinguish it from
        # unrelated databases.
        return True
    # Declared artifact paths can use formats that have no reliable suffix,
    # such as Claude Code tool-result sidecars. Let the owning declaration
    # admit those paths while keeping ordinary source discovery suffix-bound.
    if artifact_rule_for_path(provider, str(path)) is not None:
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


def _walk_source_paths(
    base: Path, *, provider: Provider = Provider.UNKNOWN, destination: Path | None = None
) -> list[Path]:
    paths: list[Path] = []
    for file_path in _iter_source_entries(base, destination=destination):
        if not _is_supported_source_path(file_path, provider=provider):
            continue
        # Admission refuses symlinks, FIFOs, sockets, and other non-regular
        # entries; the census counts them as unsupported.  lstat is
        # deliberate: following a symlink here would make production admission
        # disagree with the census denominator.  A candidate that cannot be
        # inspected stays in the walk, so its per-file read records the
        # failure on the cursor instead of the scan reporting it as absent.
        try:
            mode = os.stat(file_path, follow_symlinks=False).st_mode
        except OSError:
            paths.append(file_path)
            continue
        if stat.S_ISREG(mode):
            paths.append(file_path)
    return sorted(paths)


def _iter_source_entries(
    base: Path, *, onerror: Callable[[OSError], None] | None = None, destination: Path | None = None
) -> list[Path]:
    """Enumerate source files under the canonical traversal policy.

    Both admission and census use this helper so skipped directories and
    follow-link behavior cannot drift between the two routes. Nested foreign
    archives are excluded before their files are offered; acquisition may
    retain its own destination archive through the existing ownership law.
    """
    entries: list[Path] = []
    # Directory links are followed, so a linked export tree is acquired. A
    # link to a directory already entered on this walk -- an ancestor, or a
    # tree a sorted-earlier link already reached -- is recorded as a
    # non-regular entry and not entered again; without that, a cycle never
    # terminates. Real directories are always entered.
    visited: set[tuple[int, int]] = set()
    try:
        base_stat = os.stat(base)
    except OSError:
        pass
    else:
        visited.add((base_stat.st_dev, base_stat.st_ino))
    for root, dirs, files in os.walk(base, followlinks=True, onerror=onerror):
        if Path(root) != base and containing_archive_root(Path(root)) == Path(root):
            try:
                refuse_non_capture_source_root(Path(root), destination=destination)
            except SourceRootRefusedError:
                dirs[:] = []
                continue
        descend: list[str] = []
        for directory in sorted(dirs):
            if directory in _SKIP_DIRS:
                continue
            path = Path(root) / directory
            try:
                directory_stat = os.stat(path)
            except OSError:
                descend.append(directory)
                continue
            identity = (directory_stat.st_dev, directory_stat.st_ino)
            if identity in visited and path.is_symlink():
                entries.append(path)
                continue
            visited.add(identity)
            descend.append(directory)
        dirs[:] = descend
        entries.extend(Path(root) / filename for filename in files)
    return sorted(entries)


def _resolve_source_paths(source: Source, *, destination: Path | None = None) -> list[Path]:
    if not source.path:
        return []
    base = source.path.expanduser()
    if base.is_dir():
        return _walk_source_paths(base, provider=Provider.from_string(source.name), destination=destination)
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
    paths = _resolve_source_paths(source, destination=blob_store.root.parent if blob_store is not None else None)
    _cursor._initialize_cursor_state(cursor_state, paths)
    if not paths:
        return None
    paths_to_process, skipped_mtime = _cursor._select_paths_for_processing(
        paths,
        include_file_mtime=include_mtime,
        known_mtimes=known_mtimes,
        known_cursors=known_cursors,
        source_name=source.name,
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
    "_iter_source_entries",
    "_resolve_source_paths",
    "_setup_source_walk",
    "_walk_source_paths",
    "census_source_root",
]
