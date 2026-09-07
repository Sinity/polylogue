"""Set-aware identity abbreviation for listing surfaces.

A listing column that shortens an identifier must stay injective over the
identifiers rendered beside it: two distinct sessions may never produce the
same cell. Hierarchical ids (``<origin>:<parent-uuid>:agent-<hex>``) carry
their distinguishing entropy in the tail, so an abbreviation is a function of
the whole rendered frame, not of one identifier.

Callers build one :class:`IdentityFrame` per rendered result set and ask it
for each cell. The frame's ``tail`` is the context a caller pins when
successive pages must keep showing the same cell for the same identifier; an
identifier the frame has never seen renders in full.

Machine output always carries the full identifier -- abbreviation exists only
for width-bounded human columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Marks a rendered identity as shortened. One character so the elision costs
#: less width than it buys.
ELLIPSIS = "…"

#: Leading characters kept for orientation. Sized so the ``<origin>:`` segment
#: of every declared origin survives whole; a longer head is cut raw.
HEAD_BUDGET = 20

#: Trailing characters kept even when one would already distinguish the frame.
#: A one-character tail is unique and unreadable; this is the readable floor.
MINIMUM_TAIL = 8

#: Characters that end an identifier segment. A head cut here reads as a whole
#: segment rather than a word fragment.
SEGMENT_SEPARATORS = ":/"

#: Extra characters the tail may claim to start at a segment boundary. A
#: hierarchical id spends its entropy in the final segment, so ``:agent-9c2f``
#: is worth more than the same width of fragment.
TAIL_SNAP = 8

__all__ = [
    "ELLIPSIS",
    "HEAD_BUDGET",
    "MINIMUM_TAIL",
    "IdentityFrame",
    "identity_frame",
]


def _context_head(identifier: str) -> str:
    """Return the orientation prefix kept ahead of the elision."""
    window = identifier[: HEAD_BUDGET + 1]
    cut = max(window.rfind(separator) for separator in SEGMENT_SEPARATORS)
    if 0 <= cut < HEAD_BUDGET:
        return identifier[: cut + 1]
    return identifier[:HEAD_BUDGET]


def _render(identifier: str, tail: int) -> str:
    head = _context_head(identifier)
    start = len(identifier) - tail
    if start > 0:
        window = identifier[:start]
        cut = max(window.rfind(separator) for separator in SEGMENT_SEPARATORS)
        if cut >= 0 and start - cut <= TAIL_SNAP:
            start = cut
    if start <= len(head) + len(ELLIPSIS):
        return identifier
    return f"{head}{ELLIPSIS}{identifier[start:]}"


@dataclass(frozen=True, slots=True)
class IdentityFrame:
    """One rendered frame's identifier displays, injective by construction."""

    displays: Mapping[str, str]
    #: Trailing characters this frame kept. Pass it back as ``tail`` to render
    #: a continuation of the same frame identically.
    tail: int
    #: Width of the widest display, so a fixed-width column stays aligned
    #: without cutting any cell.
    column_width: int

    def display(self, identifier: object) -> str:
        """Return the frame's display for ``identifier``.

        An identifier outside the frame has no cohort to be distinguished
        against, so it renders in full rather than being shortened blind.
        """
        text = str(identifier)
        return self.displays.get(text, text)


def identity_frame(identifiers: Iterable[object], *, tail: int | None = None) -> IdentityFrame:
    """Build the shortest injective display for one frame of identifiers.

    The result depends on the *set* of identifiers, never on their order, so
    re-sorting a result set cannot change a single cell. ``tail`` pins the
    frame context instead of deriving it, for a continuation that must render
    an unchanged frame the same way.
    """
    unique = sorted({str(identifier) for identifier in identifiers if str(identifier)})
    if not unique:
        return IdentityFrame(displays=MappingProxyType({}), tail=tail or MINIMUM_TAIL, column_width=0)

    ceiling = max(len(identifier) for identifier in unique)
    if tail is not None:
        chosen = max(MINIMUM_TAIL, tail)
    else:
        # At ``ceiling`` every identifier renders whole, so the ladder always
        # terminates -- the worst frame is one that shows full identities.
        chosen = ceiling
        for candidate in range(MINIMUM_TAIL, ceiling + 1):
            rendered = {identifier: _render(identifier, candidate) for identifier in unique}
            if len(set(rendered.values())) == len(unique):
                chosen = candidate
                break

    displays = {identifier: _render(identifier, chosen) for identifier in unique}
    return IdentityFrame(
        displays=MappingProxyType(displays),
        tail=chosen,
        column_width=max(len(display) for display in displays.values()),
    )
