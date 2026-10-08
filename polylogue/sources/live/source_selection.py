"""Deterministic ownership for overlapping live-source roots."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar


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


def deepest_source_for_path(path: Path, sources: Iterable[SourceT]) -> SourceT | None:
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
    try:
        resolved: Path | None = declared.resolve()
    except OSError:
        resolved = None
    lexical: list[tuple[bool, int, SourceT, Path]] = []
    explicit: list[tuple[bool, int, SourceT, Path]] = []
    physical: list[tuple[bool, int, SourceT, Path]] = []
    for source in sources:
        declared_root = Path(os.path.abspath(source.root.expanduser()))
        exact_paths = getattr(source, "exact_paths", None)
        if exact_paths is not None and resolved is not None and resolved in exact_paths:
            explicit.append((True, len(declared_root.parts), source, resolved))
            continue
        if declared.is_relative_to(declared_root):
            lexical.append((False, len(declared_root.parts), source, declared))
            continue
        if resolved is not None:
            try:
                source_root = declared_root.resolve()
                if resolved.is_relative_to(source_root):
                    physical.append((False, len(source_root.parts), source, resolved))
            except (OSError, ValueError):
                continue
    # A declared namespace owns its subtree even when that subtree is an
    # accepted directory alias. Physical root aliases select only when no
    # declared directory contains the offered path; explicit files retain
    # their stronger physical-file declaration in either case.
    matches = explicit + (lexical if lexical else physical)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][2]
    accepting = [match for match in matches if _accepts(match[2], match[3])]
    preferred = accepting or matches
    return max(preferred, key=lambda match: (match[0], match[1]))[2]


__all__ = ["deepest_source_for_path"]
