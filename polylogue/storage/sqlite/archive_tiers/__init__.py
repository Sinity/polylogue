"""Executable target DDL for the archive."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL, EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL, OPS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL, SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL, USER_SCHEMA_VERSION

ARCHIVE_DDL_BY_TIER: Mapping[ArchiveTier, str] = {
    ArchiveTier.SOURCE: SOURCE_DDL,
    ArchiveTier.INDEX: INDEX_DDL,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_DDL,
    ArchiveTier.USER: USER_DDL,
    ArchiveTier.OPS: OPS_DDL,
}

ARCHIVE_VERSION_BY_TIER: Mapping[ArchiveTier, int] = {
    ArchiveTier.SOURCE: SOURCE_SCHEMA_VERSION,
    ArchiveTier.INDEX: INDEX_SCHEMA_VERSION,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_SCHEMA_VERSION,
    ArchiveTier.USER: USER_SCHEMA_VERSION,
    ArchiveTier.OPS: OPS_SCHEMA_VERSION,
}


def archive_ddl_for_tier(tier: ArchiveTier) -> str:
    """Return the fresh-create DDL script for one archive durability tier."""
    return ARCHIVE_DDL_BY_TIER[tier]


__all__ = [
    "ARCHIVE_DDL_BY_TIER",
    "ARCHIVE_VERSION_BY_TIER",
    "archive_ddl_for_tier",
]
