"""Deterministic ownership for overlapping live-source roots."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypeVar


class RootedSource(Protocol):
    @property
    def root(self) -> Path: ...


SourceT = TypeVar("SourceT", bound=RootedSource)


def deepest_source_for_path(path: Path, sources: Iterable[SourceT]) -> SourceT | None:
    """Return the most-specific configured source owning ``path``."""

    try:
        resolved = path.resolve()
    except OSError:
        return None
    matches: list[tuple[int, SourceT]] = []
    for source in sources:
        try:
            source_root = source.root.resolve()
            if resolved.is_relative_to(source_root):
                matches.append((len(source_root.parts), source))
        except (OSError, ValueError):
            continue
    return max(matches, key=lambda match: match[0])[1] if matches else None


__all__ = ["deepest_source_for_path"]
