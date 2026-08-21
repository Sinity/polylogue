"""Typed archive-tier primitives."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from polylogue.core.enums import DelegationMappingState, DelegationResultStatus


class ArchiveTier(StrEnum):
    """Durability tiers in the split archive file set."""

    SOURCE = "source"
    INDEX = "index"
    EMBEDDINGS = "embeddings"
    USER = "user"
    OPS = "ops"
    AUDIT = "audit"


# Re-export the core read vocabulary where index DDL needs its ``Literal``
# arguments. Keeping this compatibility import avoids an archive-tier cycle
# while giving storage and public surfaces one declaration owner.

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
