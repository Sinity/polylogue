"""Configure Sinex publication against the archive's durable source tier."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.storage.archive_identity import ArchiveLocation

if TYPE_CHECKING:
    from polylogue.daemon.derivation import PublicationBarrier
    from polylogue.sinex.models import PublicationMode
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import SinexTransport


def publication_service_for_archive(
    archive_root: Path, *, mode: PublicationMode, transport: SinexTransport
) -> PublicationService:
    """Bind publication to configured source authority, never an index generation."""
    from polylogue.sinex.service import PublicationService

    return PublicationService(
        source_db_path=ArchiveLocation.resolve(archive_root).configured_tier("source").configured_path,
        mode=mode,
        transport=transport,
    )


def configured_derivation_barrier(archive_root: Path) -> PublicationBarrier | None:
    """The primary-publication barrier derivation owners honor, when configured.

    The staged routes read it through the Sinex stage's
    ``blocks_following_stages``; derivation owners run their own convergers
    without that stage, so composition hands them the same read directly.
    Outside primary mode nothing is held.
    """
    from polylogue.sinex.models import PublicationMode
    from polylogue.sinex.service import primary_blocking_object_ids

    if PublicationMode.from_string(load_polylogue_config().sinex_mode) is not PublicationMode.PRIMARY:
        return None
    source_db = ArchiveLocation.resolve(archive_root).configured_tier("source").configured_path
    return partial(primary_blocking_object_ids, source_db)
