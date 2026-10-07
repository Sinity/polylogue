"""Source path discovery and cursor-aware walk setup.

Every directory source is enumerated by its declared layout
(:func:`polylogue.sources.source_layout.source_layout_for`) through the same
ordered walk the daemon's discovery uses, so a one-shot or census route
cannot reach material the watcher would never admit.
"""

from __future__ import annotations

import os
import stat
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
from .source_layout import SourceLayout, source_layout_for
from .source_root_admission import SourceRootRefusedError, containing_archive_root, refuse_non_capture_source_root


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

    # The same file-or-directory resolution as ``_resolve_source_paths``: a
    # directly configured file is the one candidate. Under a directory the
    # census takes every file the declared layout places -- the admitted
    # files and the links discovery refuses -- so a refused link stays in the
    # denominator.
    walked = root.is_dir()
    candidates: list[Path] = [root] if root.is_file() else []
    if walked:
        candidates = layout_source_candidates(provider.value, root)
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


def _outside_foreign_archive(path: Path, *, base: Path, destination: Path | None) -> bool:
    """Whether ``path`` lies outside every nested archive root the walk must refuse.

    A layout may reach an archive root nested under the source (an export
    drop holding a copied archive); its files are offered only when the
    ownership law admits that root as this acquisition's own destination.
    """

    nested = containing_archive_root(path.parent)
    if nested is None or nested == containing_archive_root(base):
        return True
    try:
        refuse_non_capture_source_root(nested, destination=destination)
    except SourceRootRefusedError:
        return False
    return True


def layout_source_paths(
    name: str, base: Path, *, destination: Path | None = None, layout: SourceLayout | None = None
) -> list[Path]:
    """Every regular file ``base``'s layout admits, in discovery order.

    ``layout`` defaults to the one ``name`` declares.
    """

    from polylogue.sources.live.discovery import _source_path_steps
    from polylogue.sources.live.watcher import WatchSource

    source = WatchSource(name=name, root=base, layout=layout if layout is not None else source_layout_for(name))
    return [
        path
        for path in _source_path_steps(source, (source,), after=None)
        if path is not None and _outside_foreign_archive(path, base=base, destination=destination)
    ]


def layout_source_candidates(name: str, base: Path) -> list[Path]:
    """Files the layout places, admitted or not (links, non-regular files), for a census denominator."""

    from polylogue.sources.live.discovery import _source_path_steps
    from polylogue.sources.live.watcher import WatchSource

    source = WatchSource(name=name, root=base)
    candidates: list[Path] = []

    def record(path: Path, disposition: str, reason: str) -> None:
        placed = disposition in {"accepted", "alias"} or reason == "non_regular_file"
        if placed or (disposition == "fault" and source.accepts(path)):
            candidates.append(path)

    for _ in _source_path_steps(source, (source,), after=None, on_disposition=record):
        pass
    return sorted(path for path in candidates if _outside_foreign_archive(path, base=base, destination=None))


def _resolve_source_paths(source: Source, *, destination: Path | None = None) -> list[Path]:
    if not source.path:
        return []
    base = source.path.expanduser()
    if base.is_dir():
        return layout_source_paths(source.name, base, destination=destination)
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
    "_resolve_source_paths",
    "_setup_source_walk",
    "census_source_root",
    "layout_source_candidates",
    "layout_source_paths",
]
