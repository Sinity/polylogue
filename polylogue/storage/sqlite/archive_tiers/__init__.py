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

# Durable ``user_version`` begins again at one for the format lineage
# introduced with this archive floor. The six-tier archive marker carries
# every tier's expected version; derived tiers continue to use their separate
# schema identities. A version alone is deliberately insufficient to admit a
# durable file: bootstrap also requires the archive format marker and verifies
# a floor durable schema.
#
# ``ARCHIVE_VERSION_BY_TIER`` below is the single declaration of what each tier
# stamps and what its readers compare against. The durable tier modules
# deliberately declare no ``*_SCHEMA_VERSION`` of their own: two numbers for one
# tier is a split nobody can read, and the pre-floor constants that survived
# this reset went on being consumed as an authority threshold long after
# nothing stamped them. ``tests/unit/storage/test_durable_tier_version_authority.py``
# holds that line.
ARCHIVE_FORMAT_FLOOR_VERSION = 1

#: The source tier's current durable target. It sits above
#: ``ARCHIVE_FORMAT_FLOOR_VERSION`` because slot 002 -- the first numbered
#: source train of this lineage -- registered ``excision_policy_projections``
#: in canonical DDL after an ordinary writer had been creating it at write
#: time (polylogue-j264r AC3). An archive born before that slot carries the
#: table only if a policy-bearing generation happened to be published into it,
#: so the shape has to be reached by migration, not by fresh DDL alone.
SOURCE_TIER_VERSION = 4

#: The user tier's current durable target. Slot 002 -- the first numbered user
#: train of this lineage -- rebuilt ``assertions`` with ``status TEXT NOT NULL
#: DEFAULT 'active'`` (polylogue-lbk1 AC1/AC2). The column was nullable, no
#: production writer could ever produce that NULL, and sixteen read sites in
#: eight modules carried a ``COALESCE(status, ...)`` to resolve it. Fresh DDL
#: alone would leave every existing user.db nullable, and user.db is never
#: rebuilt from source evidence, so the shape has to be reached by
#: copy-forward.
USER_TIER_VERSION = 2

# Audit slot 002 adds the operation lookup index declared in the fresh DDL.
# Keep this in the single tier authority map: durable readers and bootstrap
# must agree on the version an audit migration train produces.
AUDIT_TIER_VERSION = 2

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
