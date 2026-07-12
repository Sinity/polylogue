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
SEMANTICS_VERSION = 1

#: Version of the canonicalization algorithm (NFC + sorted-key JSON framing).
CANONICALIZER_VERSION = 1

#: Media type recorded for NDJSON segment content descriptors.
SEGMENT_MEDIA_TYPE = "application/x-ndjson; charset=utf-8"

#: Default bound on records per segment. Segments are immutable once sealed;
#: a session growing past this bound seals the current segment and opens a
#: new one rather than rewriting prior bytes.
DEFAULT_MAX_RECORDS_PER_SEGMENT = 500

#: Filename template for sealed segments, ordered by index.
SEGMENT_FILENAME_TEMPLATE = "seg-{index:05d}.ndjson"

#: Filename for the revision manifest document.
MANIFEST_FILENAME = "manifest.json"

__all__ = [
    "CANONICALIZER_VERSION",
    "DEFAULT_MAX_RECORDS_PER_SEGMENT",
    "MANIFEST_FILENAME",
    "PROTOCOL_VERSION",
    "SEGMENT_FILENAME_TEMPLATE",
    "SEGMENT_MEDIA_TYPE",
    "SEMANTICS_VERSION",
]
