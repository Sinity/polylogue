"""Executable target DDL for the archive."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL, EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL, OPS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.schema_disposition import (
    assert_complete_audit_disposition,
    assert_complete_schema_dispositions,
    audit_column_dispositions,
    schema_dispositions,
)
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.source_attachments import SourceAttachment
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

# This format lineage begins with all six tiers at version one. Future durable
# changes advance the relevant tier through a numbered migration.
ARCHIVE_FORMAT_FLOOR_VERSION = 1
SOURCE_TIER_VERSION = 1
USER_TIER_VERSION = 1
AUDIT_TIER_VERSION = 1

AUDIT_COLUMN_DISPOSITIONS = audit_column_dispositions()
assert_complete_audit_disposition(AUDIT_COLUMN_DISPOSITIONS)

ARCHIVE_DDL_BY_TIER: Mapping[ArchiveTier, str] = {
    ArchiveTier.SOURCE: SOURCE_DDL,
    ArchiveTier.INDEX: INDEX_DDL,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_DDL,
    ArchiveTier.USER: USER_DDL,
    ArchiveTier.OPS: OPS_DDL,
    ArchiveTier.AUDIT: AUDIT_DDL,
}

ARCHIVE_VERSION_BY_TIER: Mapping[ArchiveTier, int] = {
    ArchiveTier.SOURCE: SOURCE_TIER_VERSION,
    ArchiveTier.INDEX: INDEX_SCHEMA_VERSION,
    ArchiveTier.EMBEDDINGS: EMBEDDINGS_SCHEMA_VERSION,
    ArchiveTier.USER: USER_TIER_VERSION,
    ArchiveTier.OPS: OPS_SCHEMA_VERSION,
    ArchiveTier.AUDIT: AUDIT_TIER_VERSION,
}

SCHEMA_DISPOSITIONS = schema_dispositions()
assert_complete_schema_dispositions(SCHEMA_DISPOSITIONS)


def archive_ddl_for_tier(tier: ArchiveTier) -> str:
    """Return the fresh-create DDL script for one archive durability tier."""
    return ARCHIVE_DDL_BY_TIER[tier]


__all__ = [
    "AUDIT_COLUMN_DISPOSITIONS",
    "ARCHIVE_DDL_BY_TIER",
    "ARCHIVE_FORMAT_FLOOR_VERSION",
    "ARCHIVE_VERSION_BY_TIER",
    "AUDIT_TIER_VERSION",
    "archive_ddl_for_tier",
    "SCHEMA_DISPOSITIONS",
    "SOURCE_TIER_VERSION",
    "USER_TIER_VERSION",
    "SourceAttachment",
]
