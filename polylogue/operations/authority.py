"""Resolve the archive facts that attribute one read result.

The operation boundary owns this resolution. It knows the archive root the
caller is actually bound to and, for a reader, the index generation that reader
opened -- neither of which a serialization adapter can recover after the fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from polylogue.config import Config, active_archive_root
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.surfaces.authority import AuthorityEnvelope, build_authority_envelope

if TYPE_CHECKING:
    pass


class _PinnedReader(Protocol):
    """The reader attributes carrying a pinned generation."""

    archive_root: Path
    index_db_path: Path


def _tier_schema_versions() -> dict[str, int]:
    # Bootstrap imports the complete tier graph; import it only when an
    # authority snapshot is built.
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    return {tier.value: int(version) for tier, version in ARCHIVE_VERSION_BY_TIER.items()}


def _envelope_for_identity(
    identity: ArchiveIdentity,
    *,
    server_identity: Literal["daemon", "direct"],
    started_at: float | None,
    run_id: str | None,
    degraded: tuple[str, ...],
) -> AuthorityEnvelope:
    return build_authority_envelope(
        archive_epoch=identity.authority_identity_digest,
        generation_id=identity.active_generation,
        tier_schema_versions=_tier_schema_versions(),
        server_identity=server_identity,
        started_at=started_at,
        run_id=run_id,
        degraded=degraded,
    )


def authority_for_root(
    archive_root: Path,
    *,
    server_identity: Literal["daemon", "direct"],
    started_at: float | None = None,
    run_id: str | None = None,
    degraded: tuple[str, ...] = (),
) -> AuthorityEnvelope:
    """Attribute a result to whichever generation is active under ``archive_root``."""

    identity = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(Path(archive_root)))
    return _envelope_for_identity(
        identity,
        server_identity=server_identity,
        started_at=started_at,
        run_id=run_id,
        degraded=degraded,
    )


def authority_for_config(
    config: Config,
    *,
    server_identity: Literal["daemon", "direct"],
    started_at: float | None = None,
    run_id: str | None = None,
    degraded: tuple[str, ...] = (),
) -> AuthorityEnvelope:
    """Attribute a result to the archive housing the database the config opens.

    ``Config.archive_root`` and ``Config.db_path`` are independently settable,
    so an explicit ``db_path`` under a different root must win here exactly as
    it does for the read itself.
    """

    return authority_for_root(
        active_archive_root(config),
        server_identity=server_identity,
        started_at=started_at,
        run_id=run_id,
        degraded=degraded,
    )


def authority_for_reader(
    reader: _PinnedReader,
    *,
    server_identity: Literal["daemon", "direct"],
    started_at: float | None = None,
    run_id: str | None = None,
    degraded: tuple[str, ...] = (),
) -> AuthorityEnvelope:
    """Attribute a result to the index generation the reader actually opened.

    A reader holds its index open for the life of the read. Re-resolving the
    active pointer here would attribute the rows to a generation the writer
    published after those rows were read.
    """

    identity = ArchiveIdentity.resolve_pinned_index(Path(reader.archive_root), Path(reader.index_db_path))
    return _envelope_for_identity(
        identity,
        server_identity=server_identity,
        started_at=started_at,
        run_id=run_id,
        degraded=degraded,
    )


__all__ = ["authority_for_config", "authority_for_reader", "authority_for_root"]
