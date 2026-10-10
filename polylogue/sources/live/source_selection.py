"""Deterministic ownership for overlapping live-source roots."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, Protocol, TypeVar


class RootedSource(Protocol):
    @property
    def root(self) -> Path: ...


SourceT = TypeVar("SourceT", bound=RootedSource)


def _accepts(source: SourceT, path: Path) -> bool:
    """Whether ``source`` admits ``path`` under its own artifact contract.

    A source that declares no acceptance contract admits everything under its
    root, which preserves the plain-depth behaviour for callers whose sources
    are bare roots.
    """
    accepts = getattr(source, "accepts", None)
    if not callable(accepts):
        return True
    try:
        return bool(accepts(path))
    except (OSError, ValueError):
        return False


def _select_owner(path: Path, rooted: tuple[tuple[SourceT, Path], ...]) -> SourceT | None:
    """Return the most-specific configured source owning ``path``.

    Depth alone is the wrong ownership rule once roots overlap. A generic
    additional root nested inside a typed source's root is deeper, so depth
    hands it every file underneath -- including the typed artifacts only the
    typed source admits (Antigravity's ``.pb`` conversations are not in the
    generic additional-root suffix set), which silently drops them from
    ingest. Ownership is therefore resolved among the sources that actually
    accept the path, and falls back to plain depth only when none do, so a
    path no source admits still resolves exactly as before.
    """

    declared = Path(os.path.abspath(path.expanduser()))
    lexical = [(False, len(root.parts), source, declared) for source, root in rooted if declared.is_relative_to(root)]
    explicit: list[tuple[bool, int, SourceT, Path]] = []
    physical: list[tuple[bool, int, SourceT, Path]] = []
    # Physical-root aliases cannot outrank a declared namespace. Resolve only
    # when an explicit file can override it, or no lexical owner exists.
    if not lexical or any(getattr(source, "exact_paths", None) for source, _root in rooted):
        try:
            resolved: Path | None = declared.resolve()
        except OSError:
            resolved = None
        if resolved is not None:
            for source, declared_root in rooted:
                exact_paths = getattr(source, "exact_paths", None)
                if exact_paths is not None and resolved in exact_paths:
                    explicit.append((True, len(declared_root.parts), source, resolved))
                    # An exact match replaces this source's lexical entry.
                    lexical = [match for match in lexical if match[2] is not source]
            if not lexical:
                for source, declared_root in rooted:
                    if any(match[2] is source for match in explicit) or declared.is_relative_to(declared_root):
                        continue
                    try:
                        source_root = declared_root.resolve()
                        if resolved.is_relative_to(source_root):
                            physical.append((False, len(source_root.parts), source, resolved))
                    except (OSError, ValueError):
                        continue
    # A declared namespace owns its subtree even when that subtree is an
    # accepted directory alias. Explicit files retain their stronger claim.
    matches = explicit + (lexical if lexical else physical)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][2]
    accepting = [match for match in matches if _accepts(match[2], match[3])]
    preferred = accepting or matches
    return max(preferred, key=lambda match: (match[0], match[1]))[2]


class SourceSelection(Generic[SourceT]):
    """Reuse lexical root preparation within one configured source walk.

    Physical aliases and exact-file claims remain live observations. Relative
    roots are reanchored when the working directory changes, and changed
    source roots invalidate this owner's lexical preparation.
    """

    def __init__(self, sources: Iterable[SourceT]) -> None:
        self._sources = tuple(sources)
        self._source_roots: tuple[Path, ...] | None = None
        self._cwd: str | None = None
        self._rooted: tuple[tuple[SourceT, Path], ...] = ()

    def owner_for_path(self, path: Path) -> SourceT | None:
        roots = tuple(source.root for source in self._sources)
        cwd = os.getcwd() if any(not root.is_absolute() for root in roots) else None
        if roots != self._source_roots or cwd != self._cwd:
            self._rooted = tuple(
                (source, Path(os.path.abspath(root.expanduser())))
                for source, root in zip(self._sources, roots, strict=True)
            )
            self._source_roots, self._cwd = roots, cwd
        return _select_owner(path, self._rooted)


def deepest_source_for_path(path: Path, sources: Iterable[SourceT]) -> SourceT | None:
    """Resolve one path with the same ownership engine used by source walks."""
    return SourceSelection(sources).owner_for_path(path)


__all__ = ["SourceSelection", "deepest_source_for_path"]
