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


def _path_forms(path: Path) -> list[Path]:
    """The path spellings ownership may be decided on, lexical form first.

    A source root whose export tree is mounted through a symlink owns the
    files the discovery walk reached *through* that link. Deciding ownership
    on ``resolve()`` alone disowned exactly those files -- the resolved path
    lies outside the configured root -- so the walk discovered them and then
    dropped every one. The lexical form keeps them owned; the resolved form
    is retained so a ``..`` segment still cannot escape a root.
    """

    forms: list[Path] = []
    for candidate in (Path(os.path.abspath(path)), _resolved_or_none(path)):
        if candidate is not None and candidate not in forms:
            forms.append(candidate)
    return forms


def _resolved_or_none(path: Path) -> Path | None:
    try:
        return path.resolve()
    except OSError:
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
    """

    forms = _path_forms(path)
    if not forms:
        return None
    owned_form = forms[0]
    matches: list[tuple[int, SourceT]] = []
    for source in sources:
        try:
            root_forms = _path_forms(source.root)
            matched_root = next(
                (root_form for root_form in root_forms for form in forms if form.is_relative_to(root_form)),
                None,
            )
            if matched_root is not None:
                matches.append((len(matched_root.parts), source))
        except (OSError, ValueError):
            continue
    if not matches:
        return None
    if len(matches) == 1:
        # Unambiguous ownership needs no acceptance check, which keeps the
        # declared-artifact lookup out of the common single-root path.
        return matches[0][1]
    accepting = [match for match in matches if _accepts(match[1], owned_form)]
    preferred = accepting or matches
    return max(preferred, key=lambda match: match[0])[1]


__all__ = ["deepest_source_for_path"]
