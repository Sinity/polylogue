"""Deterministic ownership for overlapping live-source roots."""

from __future__ import annotations

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


def _owning_root_depth(forms: tuple[Path, ...], roots: tuple[Path, ...]) -> int | None:
    """Return the depth of the first root owning any form of a path."""
    for root in roots:
        for form in forms:
            if form.is_relative_to(root):
                return len(root.parts)
    return None


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

    A file reached through a directory symlink placed under a watch root keeps
    that root in its own path while resolving outside every root, so ownership
    admits the path as given as well as its resolved form.
    """

    try:
        resolved = path.resolve()
    except OSError:
        return None
    forms = (resolved,) if path == resolved else (resolved, path)
    matches: list[tuple[int, SourceT]] = []
    for source in sources:
        try:
            source_root = source.root.resolve()
        except OSError:
            continue
        roots = (source_root,) if source.root == source_root else (source_root, source.root)
        depth = _owning_root_depth(forms, roots)
        if depth is not None:
            matches.append((depth, source))
    if not matches:
        return None
    if len(matches) == 1:
        # Unambiguous ownership needs no acceptance check, which keeps the
        # declared-artifact lookup out of the common single-root path.
        return matches[0][1]
    accepting = [match for match in matches if _accepts(match[1], resolved)]
    preferred = accepting or matches
    return max(preferred, key=lambda match: match[0])[1]


__all__ = ["deepest_source_for_path"]
