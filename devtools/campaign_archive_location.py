"""Owned :class:`ArchiveLocation` plumbing for devtools synthetic benchmark/scale campaigns.

Background
----------

Synthetic/performance campaigns (``devtools bench synthetic``, the FTS
rebuild / incremental-index / session-insight / daemon-live-convergence
benchmark runners) generate a synthetic archive under an output directory
and then measure/mutate it. The historical bug (polylogue-ovme, the
"phantom benchmark.db" regression): callers invented a root-shaped or
filename-shaped sentinel path -- typically ``archive_dir / "benchmark.db"``
-- and passed that literal ``Path`` around. Some consumers
(``SQLiteBackend``) canonicalize any non-``index.db`` filename to
``<parent>/index.db`` before opening it; others
(``polylogue.storage.sqlite.connection.open_connection`` /
``connection_context``) open the literal path handed to them with no such
canonicalization. Handing the *same* sentinel path to both kinds of
consumer across one campaign run therefore silently produces two SQLite
files: the real generated archive at ``index.db``, and an empty phantom
``benchmark.db`` that some campaign runners quietly read/write instead.

This module gives campaigns one stable, ownership-proven handle
(:class:`CampaignArchiveLocation`) instead of a bare sentinel path.
Construction resolves the archive's real :class:`ArchiveLocation` topology
and acquires exclusive maintenance/campaign ownership over it via
:class:`~polylogue.storage.archive_identity.OwnedArchiveLocation` --
failing closed, before any SQLite file is touched, when the target
directory is already owned by another live process. Every reopen for the
lifetime of one campaign run must fetch the active index path through
:attr:`CampaignArchiveLocation.active_index_path`, which re-resolves the
location and reasserts ownership on every call so a stale or foreign
generation is caught immediately rather than served silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from polylogue.storage.archive_identity import (
    ArchiveLocation,
    OwnedArchiveLocation,
    assert_owns_archive_location,
)


@dataclass
class CampaignArchiveLocation:
    """Ownership-proven :class:`ArchiveLocation` held stable across one campaign run.

    Do not construct directly; use :meth:`acquire`. Every consumer that
    needs to open a connection against the generated archive must call
    :attr:`active_index_path` rather than deriving or caching a filename of
    its own -- that is precisely the shape of the phantom-``benchmark.db``
    regression this type exists to prevent.
    """

    owned: OwnedArchiveLocation

    @classmethod
    def acquire(cls, archive_dir: Path, *, owner_id: str | None = None) -> CampaignArchiveLocation:
        """Resolve ``archive_dir`` and claim exclusive campaign ownership over it.

        Raises :class:`~polylogue.storage.archive_identity.ArchiveOwnershipError`
        before any SQLite tier file is opened when ``archive_dir`` is already
        owned by another live campaign/maintenance process.
        """
        location = ArchiveLocation.resolve(archive_dir)
        owned = OwnedArchiveLocation.acquire(location, owner_id=owner_id)
        return cls(owned=owned)

    @property
    def configured_root(self) -> Path:
        return self.owned.location.configured_root

    @property
    def active_index_path(self) -> Path:
        """The archive's current active ``index.db`` path.

        Re-resolves :class:`ArchiveLocation` fresh on every call (rather
        than caching the value observed at :meth:`acquire` time) and
        reasserts ownership via
        :func:`~polylogue.storage.archive_identity.assert_owns_archive_location`.
        A concurrent generation swap (e.g. a promotion that rotates the
        active-index pointer mid-campaign) is therefore surfaced as a fail-
        fast :class:`~polylogue.storage.archive_identity.ArchiveOwnershipError`
        instead of silently serving a stale or foreign path.
        """
        current = ArchiveLocation.resolve(self.configured_root)
        assert_owns_archive_location(self.owned, current)
        return current.active_index_path

    def release(self) -> None:
        self.owned.release()

    def __enter__(self) -> CampaignArchiveLocation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()


__all__ = ["CampaignArchiveLocation"]
