"""Named evidence for filesystem faults met while discovering source files.

A directory that cannot be read is not an empty directory. Every discovery
route records the fault as a :class:`WalkFault` and then either refuses --
raising :class:`WalkRefusedError` -- or hands a *counted* collection of faults to
its caller. What no route may do is return a short result: a caller cannot
distinguish "the subtree is empty" from "the subtree could not be read", and
on a from-scratch rebuild there is no prior row count that would reveal the
difference afterwards.

This is the recorder side of the ``onerror`` hook that
:func:`polylogue.sources.source_walk._iter_source_entries` already accepts and
that the antigravity source census already uses; it exists so the daemon's
discovery routes record the same evidence rather than inventing their own.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["WalkFault", "WalkFaultRecorder", "WalkRefusedError"]


@dataclass(frozen=True, slots=True)
class WalkFault:
    """One path that discovery could not read, with why it could not."""

    path: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.detail}"


@dataclass(slots=True)
class WalkFaultRecorder:
    """Collects walk faults so a caller can report a counted degradation."""

    faults: list[WalkFault] = field(default_factory=list)

    def record(self, path: Path | str, detail: str) -> None:
        self.faults.append(WalkFault(Path(path), detail))

    def on_walk_error(self, error: OSError) -> None:
        """``os.walk(..., onerror=...)`` hook.

        Without this hook ``os.walk`` swallows the error and simply omits the
        subtree, which is the silent loss this module exists to prevent.
        """
        path = Path(error.filename) if error.filename else Path()
        self.record(path, f"directory could not be read: {error}")

    def collected(self) -> tuple[WalkFault, ...]:
        return tuple(self.faults)

    def paths(self) -> tuple[str, ...]:
        return tuple(str(fault.path) for fault in self.faults)

    def __len__(self) -> int:
        return len(self.faults)

    def __bool__(self) -> bool:
        return bool(self.faults)

    def __iter__(self) -> Iterator[WalkFault]:
        return iter(self.faults)


class WalkRefusedError(Exception):
    """Discovery refused because part of its scope could not be read.

    Deliberately *not* an ``OSError`` subclass: the routes this replaces were
    each an ``except OSError`` that dropped the evidence, and several callers
    still catch ``OSError`` broadly. A distinct type cannot be re-swallowed by
    one of those handlers by accident.
    """

    def __init__(self, summary: str, faults: Iterable[WalkFault]) -> None:
        self.faults: tuple[WalkFault, ...] = tuple(faults)
        detail = "; ".join(str(fault) for fault in self.faults)
        super().__init__(f"{summary}: {detail}" if detail else summary)
