"""Executable target DDL for the archive."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL, EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL, OPS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.schema_disposition import (
    assert_complete_audit_disposition,
    audit_column_dispositions,
)
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_attachments import SourceAttachment
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

# This format lineage begins with all six tiers at version one. Future durable
# changes advance the relevant tier through a numbered migration.
ARCHIVE_FORMAT_FLOOR_VERSION = 1
ARCHIVE_BASELINE_VERSION_BY_TIER: Mapping[ArchiveTier, int] = dict.fromkeys(ArchiveTier, 1)
SOURCE_TIER_VERSION = 1
USER_TIER_VERSION = 1
AUDIT_TIER_VERSION = 1

AUDIT_COLUMN_DISPOSITIONS = audit_column_dispositions()
assert_complete_audit_disposition(AUDIT_COLUMN_DISPOSITIONS)

ARCHIVE_BASELINE_DDL_BY_TIER: Mapping[ArchiveTier, str] = {
    ArchiveTier.SOURCE: SOURCE_DDL,
    ArchiveTier.INDEX: INDEX_DDL,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_DDL,
    ArchiveTier.USER: USER_DDL,
    ArchiveTier.OPS: OPS_DDL,
    ArchiveTier.AUDIT: AUDIT_DDL,
}


# Every tier's current schema is its baseline until a numbered durable
# migration lands; that migration then extends this mapping.
ARCHIVE_DDL_BY_TIER: Mapping[ArchiveTier, str] = dict(ARCHIVE_BASELINE_DDL_BY_TIER)

ARCHIVE_VERSION_BY_TIER: Mapping[ArchiveTier, int] = {
    ArchiveTier.SOURCE: SOURCE_TIER_VERSION,
    ArchiveTier.INDEX: INDEX_SCHEMA_VERSION,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_SCHEMA_VERSION,
    ArchiveTier.USER: USER_TIER_VERSION,
    ArchiveTier.OPS: OPS_SCHEMA_VERSION,
    ArchiveTier.AUDIT: AUDIT_TIER_VERSION,
}

# The six-tier disposition is not computed here: its canonical embeddings
# tier needs sqlite-vec, which this package must not require to import.
# ``schema_dispositions()`` computes it on demand.


def archive_ddl_for_tier(tier: ArchiveTier) -> str:
    """Return the current schema, including numbered durable additions."""
    return ARCHIVE_DDL_BY_TIER[tier]


__all__ = [
    "AUDIT_COLUMN_DISPOSITIONS",
    "ARCHIVE_DDL_BY_TIER",
    "ARCHIVE_BASELINE_DDL_BY_TIER",
    "ARCHIVE_BASELINE_VERSION_BY_TIER",
    "ARCHIVE_FORMAT_FLOOR_VERSION",
    "ARCHIVE_VERSION_BY_TIER",
    "AUDIT_TIER_VERSION",
    "archive_ddl_for_tier",
    "SOURCE_TIER_VERSION",
    "USER_TIER_VERSION",
    "SourceAttachment",
]
