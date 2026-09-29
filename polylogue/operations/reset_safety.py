"""Which filesystem reset targets are archive tier files, and when they may go.

The resident daemon holds every tier database open for its lifetime: the
watcher cursor store, the status registry, readers and the write coordinator
keep their own connections. Unlinking a tier and its ``-wal``/``-shm``
sidecars under them leaves those handles on deleted inodes while new
connections create fresh, empty files. A reset that names a derived tier is
therefore staged by the live request and applied by the next daemon start,
before anything opens a tier (:func:`archive_tiers_closed`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

#: The tiers a reset may delete. Bootstrap recreates both from nothing. A
#: durable tier (source, user, audit) is never recreated in an established
#: archive -- bootstrap refuses a format marker that names a missing one -- and
#: embeddings.db holds purchased vectors no route replays.
RESETTABLE_TIERS = (ArchiveTier.INDEX, ArchiveTier.OPS)

_CLOSED_ARCHIVE: ContextVar[Path | None] = ContextVar("polylogue_reset_closed_archive", default=None)


class UnresettableArchiveTierError(ValueError):
    """A reset target is, or holds, a tier no reset may delete."""

    code = "reset_unresettable_archive_tier"

    def __init__(self, targets: tuple[str, ...]) -> None:
        self.targets = targets
        super().__init__(
            "refusing to reset "
            + ", ".join(targets)
            + ": a reset deletes only index.db and ops.db; durable tiers, embeddings.db and directories "
            "holding tier databases are never deleted; no files were deleted"
        )


class LiveArchiveTierResetError(ValueError):
    """A live APPLY was asked to unlink tier files the daemon may hold open."""

    code = "reset_live_archive_tier"

    def __init__(self, targets: tuple[str, ...]) -> None:
        self.targets = targets
        super().__init__(
            "refusing to delete archive tier files outside the daemon's startup seam: "
            + ", ".join(targets)
            + "; no files were deleted"
        )


@dataclass(frozen=True, slots=True)
class ResetTargetClasses:
    """The reset targets that name tier files, by what may happen to them."""

    #: Targets that are exactly a resettable tier database or one of its
    #: sidecars. They are deleted only at the daemon's startup seam.
    derived_tier_files: tuple[tuple[str, Path], ...]
    #: Targets that are, or contain, a tier no reset deletes.
    unresettable: tuple[tuple[str, Path], ...]

    @property
    def unresettable_names(self) -> tuple[str, ...]:
        return tuple(name for name, _path in self.unresettable)

    @property
    def derived_tier_names(self) -> tuple[str, ...]:
        return tuple(name for name, _path in self.derived_tier_files)


def _tier_files(database: Path) -> tuple[Path, ...]:
    return tuple(
        database.with_name(f"{database.name}{suffix}").resolve(strict=False)
        for suffix in ("", *SQLITE_SIDECAR_SUFFIXES)
    )


def classify_reset_targets(
    archive_root: Path, targets: Iterable[tuple[str, Path]], *, served_index_path: Path
) -> ResetTargetClasses:
    """Classify each target against every tier database and sidecar of ``archive_root``.

    ``served_index_path`` is the index generation readers resolve, which a
    pointer-managed archive keeps outside ``index.db``; it counts as an index
    tier file wherever it lives.
    """
    resettable: set[Path] = set()
    every: set[Path] = set()
    for tier in ArchiveTier:
        files = _tier_files(archive_root / f"{tier.value}.db")
        every.update(files)
        if tier in RESETTABLE_TIERS:
            resettable.update(files)
    served = _tier_files(served_index_path)
    every.update(served)
    resettable.update(served)
    derived: list[tuple[str, Path]] = []
    unresettable: list[tuple[str, Path]] = []
    for name, path in targets:
        resolved = path.resolve(strict=False)
        if resolved in resettable:
            derived.append((name, path))
        elif any(tier_file.is_relative_to(resolved) for tier_file in every):
            unresettable.append((name, path))
    return ResetTargetClasses(derived_tier_files=tuple(derived), unresettable=tuple(unresettable))


def sqlite_primary(path: Path) -> Path | None:
    """The database a sidecar path belongs to, or ``None`` for a primary."""
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix))
    return None


@contextmanager
def archive_tiers_closed(archive_root: Path) -> Iterator[None]:
    """Declare that no connection to ``archive_root``'s tiers is open.

    Only the daemon's startup seam enters this: it holds exclusive archive
    ownership and runs before any tier is opened, so deleting a tier file
    cannot strand a handle on an unlinked inode.
    """
    token = _CLOSED_ARCHIVE.set(archive_root.resolve(strict=False))
    try:
        yield
    finally:
        _CLOSED_ARCHIVE.reset(token)


def archive_tiers_are_closed(archive_root: Path) -> bool:
    return _CLOSED_ARCHIVE.get() == archive_root.resolve(strict=False)


__all__ = [
    "RESETTABLE_TIERS",
    "SQLITE_SIDECAR_SUFFIXES",
    "LiveArchiveTierResetError",
    "ResetTargetClasses",
    "UnresettableArchiveTierError",
    "archive_tiers_are_closed",
    "archive_tiers_closed",
    "classify_reset_targets",
    "sqlite_primary",
]
