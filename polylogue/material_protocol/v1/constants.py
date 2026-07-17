"""Frozen constants for material protocol v1.

These values are part of the wire contract. Bumping any of them changes the
bytes a conforming encoder produces, so they must only change alongside a new
protocol/semantics version and a fixture regeneration in lockstep with the
Sinex counterpart (sinex-4j2.1).
"""

from __future__ import annotations

#: Public protocol identifier embedded in every revision manifest.
PROTOCOL_VERSION = "polylogue.material-protocol/v1"

#: Version of the Polylogue-owned domain semantics (record shapes, id
#: formulas, ordering rules) this encoder implements. Independent of
#: PROTOCOL_VERSION so additive semantics changes that keep the same framing
#: don't require a new protocol version.
#: v2: head/transcript split — mutable summary records (session, lineage,
#: usage) moved into a per-revision head segment; transcript segments carry
#: only immutable records and are the sole byte-reuse surface for appends.
SEMANTICS_VERSION = 2

#: Version of the canonicalization algorithm (NFC + sorted-key JSON framing).
CANONICALIZER_VERSION = 1

#: Media type recorded for NDJSON segment content descriptors.
SEGMENT_MEDIA_TYPE = "application/x-ndjson; charset=utf-8"

#: Default bound on records per segment. Segments are immutable once sealed;
#: a session growing past this bound seals the current segment and opens a
#: new one rather than rewriting prior bytes.
DEFAULT_MAX_RECORDS_PER_SEGMENT = 500

#: Filename template for sealed transcript segments, ordered by index.
SEGMENT_FILENAME_TEMPLATE = "seg-{index:05d}.ndjson"

#: Reserved segment index for the per-revision head segment. The head carries
#: the revision-mutable summary records (session, lineage, usage) and is
#: re-encoded fresh on EVERY revision — it is never byte-reused, so mutable
#: facts can live there without violating the transcript segments'
#: byte-for-byte reuse contract.
HEAD_SEGMENT_INDEX = -1

#: Filename for the per-revision head segment.
HEAD_FILENAME = "head.ndjson"

#: Filename for the revision manifest document.
MANIFEST_FILENAME = "manifest.json"

__all__ = [
    "CANONICALIZER_VERSION",
    "DEFAULT_MAX_RECORDS_PER_SEGMENT",
    "HEAD_FILENAME",
    "HEAD_SEGMENT_INDEX",
    "MANIFEST_FILENAME",
    "PROTOCOL_VERSION",
    "SEGMENT_FILENAME_TEMPLATE",
    "SEGMENT_MEDIA_TYPE",
    "SEMANTICS_VERSION",
]
