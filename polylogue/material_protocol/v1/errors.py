"""Typed compatibility-failure exceptions for material protocol v1.

Every failure mode named in polylogue-303r.1's acceptance criteria (missing
segment/record, changed byte/count/digest, reordered record, shifted anchor,
unknown Origin vocabulary version) raises one of these instead of returning a
truthy/falsy verification flag, so callers can distinguish *why* a revision
was rejected and fail closed by default (any uncaught subclass is a rejection).
"""

from __future__ import annotations


class MaterialProtocolError(Exception):
    """Base class for all material-protocol-v1 compatibility failures."""


class UnknownOriginVocabularyError(MaterialProtocolError):
    """The manifest declares an Origin vocabulary version/digest we don't recognize."""


class SegmentMissingError(MaterialProtocolError):
    """A segment the manifest requires was not supplied to the reader."""


class DigestMismatchError(MaterialProtocolError):
    """A recomputed digest (segment or revision content) doesn't match the manifest."""


class RecordCountMismatchError(MaterialProtocolError):
    """Observed record counts (total or per-kind) don't match the manifest's declared counts."""


class SequenceOrderError(MaterialProtocolError):
    """Records were not observed in strictly increasing global sequence order."""


class AnchorMismatchError(MaterialProtocolError):
    """A record's actual location/hash doesn't match its manifest-declared anchor."""


class AnchorNotFoundError(MaterialProtocolError):
    """A requested record_id has no anchor entry in the manifest."""


class NotAnAppendError(MaterialProtocolError):
    """A claimed append revision does not reproduce the prior revision's record prefix."""


__all__ = [
    "AnchorMismatchError",
    "AnchorNotFoundError",
    "DigestMismatchError",
    "MaterialProtocolError",
    "NotAnAppendError",
    "RecordCountMismatchError",
    "SegmentMissingError",
    "SequenceOrderError",
    "UnknownOriginVocabularyError",
]
