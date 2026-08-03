"""Typed archive-tier primitives."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ArchiveTier(StrEnum):
    """Durability tiers in the split archive file set."""

    SOURCE = "source"
    INDEX = "index"
    EMBEDDINGS = "embeddings"
    USER = "user"
    OPS = "ops"


# polylogue-u6tl: these two live here (a dependency-free leaf module) rather
# than in archive.py, where they conceptually belong, so that index.py's DDL
# can import them for `literal_check` without a cycle -- archive.py
# transitively imports insights.archive -> storage.repair ->
# storage.raw_authority -> storage.sqlite.migration_runner ->
# storage.sqlite.archive_tiers (this package), so `index.py` (loaded by that
# package's __init__) importing anything from archive.py at module scope
# deadlocks partial initialization. archive.py re-exports both names for
# existing call sites.
DelegationMappingState = Literal["resolved", "unresolved", "edge_only", "quarantined"]
DelegationResultStatus = Literal["ok", "error", "unknown"]

# polylogue-h57ic: raw_session_memberships.revision_authority is a genuinely
# narrower domain than polylogue.archive.revision_authority.RawRevisionAuthority
# (asserted/byte_proven/quarantined) used by every other revision_authority /
# previous_revision_authority column. A membership row's authority is always
# an OUTPUT of classify_historical_full_revisions/_classify_deduped_nodes
# (storage/sqlite/archive_tiers/revision_governance.py's membership-decision
# writeback), which only ever emits BYTE_PROVEN or QUARANTINED -- ASSERTED is
# an externally-captured claim about a raw's own revision identity and never
# describes a derived membership-classification verdict. Kept as an explicit,
# separately-named literal (this module, not the 3-value enum) so the
# narrowing is visible in code instead of only in the CHECK clause text.
ProvenRevisionAuthority = Literal["byte_proven", "quarantined"]


__all__ = [
    "ArchiveTier",
    "DelegationMappingState",
    "DelegationResultStatus",
    "ProvenRevisionAuthority",
]
