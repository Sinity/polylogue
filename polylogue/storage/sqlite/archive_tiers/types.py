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

# polylogue-3szyi: the two closed vocabularies of the embeddings tier.
# `embedding_failures.lifecycle_state` and `embedding_derivation_state.
# attempt_state` used to be hand-written `CHECK(col IN (...))` value lists in
# archive_tiers/embeddings.py with no generator tie. `EmbeddingFailureState`
# already existed as a `Literal` in archive_tiers/embedding_write.py and is
# the declaration owner; `EmbeddingAttemptState` is new because the attempt
# vocabulary had no Python owner at all -- its four values were bare string
# literals at a dozen call sites in the same module. Both live here rather
# than in `embedding_write.py` so the DDL declaration
# (archive_tiers_specs.py) and the writer can share one owner without an
# archive-tier import cycle, the same reason the delegation literals above do.
EmbeddingFailureState = Literal["retryable", "terminal", "acknowledged", "superseded", "resolved"]
EmbeddingAttemptState = Literal["pending", "succeeded", "failed_retryable", "failed_terminal"]

# polylogue-3szyi: these three index-tier values are closed, derived facts.
# Their writers and readers agree on a small vocabulary, so the generated
# index DDL must take its CHECK membership from this shared declaration rather
# than maintain a second string list at the storage boundary.  Because index
# DDL contributes to the derived-schema identity, changing any of these
# Literals requires index-tier reconvergence even when the emitted values do
# not change.
ProviderUsageEventType = Literal["token_count", "message_usage"]
SourceMessageResolution = Literal["resolved", "session", "ambiguous", "unresolved"]
AttachmentNativeIdKind = Literal["attachment", "file", "drive", "url"]

# One owner for the revision frontier vocabulary.  Both the nullable
# application receipt and the non-null revision-head row persist the same
# concept; keeping the Literal here prevents those two DDL declarations from
# acquiring independent value lists.
RevisionFrontierKind = Literal["byte", "semantic"]


__all__ = [
    "ArchiveTier",
    "AttachmentNativeIdKind",
    "DelegationMappingState",
    "DelegationResultStatus",
    "EmbeddingAttemptState",
    "EmbeddingFailureState",
    "ProviderUsageEventType",
    "ProvenRevisionAuthority",
    "RevisionFrontierKind",
    "SourceMessageResolution",
]
