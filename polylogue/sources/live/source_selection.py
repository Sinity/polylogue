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

    try:
        resolved = path.resolve()
    except OSError:
        return None
    matches: list[tuple[bool, int, SourceT]] = []
    for source in sources:
        try:
            source_root = source.root.resolve()
            if resolved.is_relative_to(source_root):
                exact_paths = getattr(source, "exact_paths", None)
                exact_match = exact_paths is not None and resolved in exact_paths
                matches.append((exact_match, len(source_root.parts), source))
        except (OSError, ValueError):
            continue
    if not matches:
        return None
    if len(matches) == 1:
        # Unambiguous ownership needs no acceptance check, which keeps the
        # declared-artifact lookup out of the common single-root path.
        return matches[0][2]
    accepting = [match for match in matches if _accepts(match[2], resolved)]
    preferred = accepting or matches
    # An explicit file declaration is stronger than a directory root even
    # when both roots have the same depth. It expresses ownership of this
    # exact artifact, while the directory remains a broad discovery root.
    return max(preferred, key=lambda match: (match[0], match[1]))[2]


__all__ = ["deepest_source_for_path"]
