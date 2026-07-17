"""Material protocol v1: public API surface.

Encode a ``SessionMaterial`` to bounded immutable NDJSON segments plus a
``RevisionManifest``, decode it back with no archive DB involved, and verify
compatibility (byte/count/digest/order/anchor/vocabulary) before trusting
untrusted bytes. See ``docs/material-protocol-v1.md`` for the wire-format
reference and worked example.
"""

from __future__ import annotations

from polylogue.material_protocol.v1.constants import (
    CANONICALIZER_VERSION,
    DEFAULT_MAX_RECORDS_PER_SEGMENT,
    PROTOCOL_VERSION,
    SEMANTICS_VERSION,
)
from polylogue.material_protocol.v1.decode import DecodedMessage, DecodedSession, decode_session_revision
from polylogue.material_protocol.v1.encode import EncodedRevision, encode_appended_revision, encode_session_revision
from polylogue.material_protocol.v1.errors import (
    AnchorMismatchError,
    AnchorNotFoundError,
    DigestMismatchError,
    MaterialProtocolError,
    NotAnAppendError,
    RecordCountMismatchError,
    SegmentMissingError,
    SemanticClosureError,
    SequenceOrderError,
    UnknownOriginVocabularyError,
)
from polylogue.material_protocol.v1.input_model import (
    AttachmentInput,
    BlockInput,
    FidelityGapInput,
    LineageInput,
    MessageInput,
    SessionEventInput,
    SessionMaterial,
    UsageInput,
)
from polylogue.material_protocol.v1.io import read_manifest, read_revision, read_segments, write_revision
from polylogue.material_protocol.v1.manifest import (
    AnchorEntry,
    ContentDigest,
    FidelityGap,
    RevisionManifest,
    SegmentDescriptor,
)
from polylogue.material_protocol.v1.verify import resolve_anchor, verify_revision

__all__ = [
    "AnchorEntry",
    "AnchorMismatchError",
    "AnchorNotFoundError",
    "AttachmentInput",
    "BlockInput",
    "CANONICALIZER_VERSION",
    "ContentDigest",
    "DecodedMessage",
    "DecodedSession",
    "DEFAULT_MAX_RECORDS_PER_SEGMENT",
    "DigestMismatchError",
    "EncodedRevision",
    "FidelityGap",
    "FidelityGapInput",
    "LineageInput",
    "MaterialProtocolError",
    "MessageInput",
    "NotAnAppendError",
    "PROTOCOL_VERSION",
    "RecordCountMismatchError",
    "RevisionManifest",
    "SEMANTICS_VERSION",
    "SegmentDescriptor",
    "SegmentMissingError",
    "SemanticClosureError",
    "SequenceOrderError",
    "SessionEventInput",
    "SessionMaterial",
    "UnknownOriginVocabularyError",
    "UsageInput",
    "decode_session_revision",
    "encode_appended_revision",
    "encode_session_revision",
    "read_manifest",
    "read_revision",
    "read_segments",
    "resolve_anchor",
    "verify_revision",
    "write_revision",
]
