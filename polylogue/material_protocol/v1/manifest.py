"""Revision manifest shape: the single JSON document that makes a set of
sealed NDJSON segments a complete, verifiable session revision.

The manifest is Polylogue-owned domain evidence, not a transport transaction
coordinator (see polylogue-303r.1 design notes) -- it declares what a
complete revision *is* (segment digests/sizes, expected record counts,
per-record anchors, Origin vocabulary pin, fidelity gaps); publishing it is
303r.2's concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.constants import (
    CANONICALIZER_VERSION,
    HEAD_FILENAME,
    HEAD_SEGMENT_INDEX,
    PROTOCOL_VERSION,
    SEGMENT_FILENAME_TEMPLATE,
    SEGMENT_MEDIA_TYPE,
    SEMANTICS_VERSION,
)
from polylogue.material_protocol.v1.errors import (
    MaterialManifestError,
    MaterialProtocolError,
    UnsupportedSemanticsVersionError,
)


def _text(raw: JSONValue) -> str:
    if not isinstance(raw, str):
        raise MaterialManifestError("manifest text declaration must be a string")
    return raw


def _integer(raw: JSONValue) -> int:
    if type(raw) is not int:
        raise MaterialManifestError("manifest integer declaration must be an exact integer")
    return raw


def _object(raw: JSONValue) -> dict[str, JSONValue]:
    if not isinstance(raw, dict):
        raise MaterialManifestError("manifest object declaration must be an object")
    return raw


def _array(raw: JSONValue) -> list[JSONValue]:
    if not isinstance(raw, list):
        raise MaterialManifestError("manifest array declaration must be an array")
    return raw


def _declared_semantics_version(raw: JSONValue) -> int:
    """Read a declared semantics version without letting coercion widen it.

    ``int(3.9)`` truncates to a version this reader does implement, so a
    non-integral declaration would be accepted as its floor. Only an exact
    integer is a version; ``bool`` is not one despite subclassing ``int``.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise UnsupportedSemanticsVersionError(
            f"semantics_version must be an integer, got {type(raw).__name__} {raw!r}"
        )
    return raw


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    index: int
    filename: str
    sha256: str
    size_bytes: int
    record_count: int
    first_seq: int
    last_seq: int

    def require_valid(self) -> None:
        if any(
            type(value) is not int
            for value in (self.index, self.size_bytes, self.record_count, self.first_seq, self.last_seq)
        ):
            raise MaterialManifestError("segment coordinates must be exact integers")
        expected_filename = (
            HEAD_FILENAME if self.index == HEAD_SEGMENT_INDEX else SEGMENT_FILENAME_TEMPLATE.format(index=self.index)
        )
        if (
            self.index < HEAD_SEGMENT_INDEX
            or self.filename != expected_filename
            or self.size_bytes < 0
            or self.record_count < 0
            or self.first_seq < 0
            or self.last_seq != self.first_seq + self.record_count - 1
        ):
            raise MaterialManifestError("invalid material segment descriptor")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "index": self.index,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> SegmentDescriptor:
        return SegmentDescriptor(
            index=_integer(payload["index"]),
            filename=_text(payload["filename"]),
            sha256=_text(payload["sha256"]),
            size_bytes=_integer(payload["size_bytes"]),
            record_count=_integer(payload["record_count"]),
            first_seq=_integer(payload["first_seq"]),
            last_seq=_integer(payload["last_seq"]),
        )


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """Multi-digest content descriptor. None of these fields is the domain object id."""

    polylogue_sha256: str
    canonicalizer_version: int
    size_bytes: int
    media_type: str = SEGMENT_MEDIA_TYPE
    sinex_cas_digest: str | None = None
    provider_digest: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "polylogue_sha256": self.polylogue_sha256,
            "canonicalizer_version": self.canonicalizer_version,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "sinex_cas_digest": self.sinex_cas_digest,
            "provider_digest": self.provider_digest,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> ContentDigest:
        return ContentDigest(
            polylogue_sha256=_text(payload["polylogue_sha256"]),
            canonicalizer_version=_integer(payload["canonicalizer_version"]),
            size_bytes=_integer(payload["size_bytes"]),
            media_type=_text(payload.get("media_type", SEGMENT_MEDIA_TYPE)),
            sinex_cas_digest=(
                _text(payload["sinex_cas_digest"]) if payload.get("sinex_cas_digest") is not None else None
            ),
            provider_digest=(_text(payload["provider_digest"]) if payload.get("provider_digest") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class AnchorEntry:
    """Where exactly one record lives, and what its line must hash to."""

    segment_index: int
    line_index: int
    seq: int
    kind: str
    sha256: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "segment_index": self.segment_index,
            "line_index": self.line_index,
            "seq": self.seq,
            "kind": self.kind,
            "sha256": self.sha256,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> AnchorEntry:
        return AnchorEntry(
            segment_index=_integer(payload["segment_index"]),
            line_index=_integer(payload["line_index"]),
            seq=_integer(payload["seq"]),
            kind=_text(payload["kind"]),
            sha256=_text(payload["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FidelityGap:
    scope: str
    record_id: str
    gap_kind: str
    detail: str = ""

    def to_dict(self) -> dict[str, JSONValue]:
        return {"scope": self.scope, "record_id": self.record_id, "gap_kind": self.gap_kind, "detail": self.detail}

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> FidelityGap:
        return FidelityGap(
            scope=_text(payload["scope"]),
            record_id=_text(payload["record_id"]),
            gap_kind=_text(payload["gap_kind"]),
            detail=_text(payload.get("detail", "")),
        )


@dataclass(frozen=True, slots=True)
class RevisionManifest:
    protocol_version: str
    semantics_version: int
    origin_vocabulary_version: int
    origin_vocabulary_digest: str
    session_id: str
    origin: str
    native_id: str
    revision_id: str
    superseded_revision_id: str | None
    content_digest: ContentDigest
    head_segment: SegmentDescriptor
    segments: tuple[SegmentDescriptor, ...]
    expected_record_counts: dict[str, int]
    anchors: dict[str, AnchorEntry]
    sequence_rule: str
    completeness: str
    fidelity_gaps: tuple[FidelityGap, ...] = field(default_factory=tuple)
    revision_created_at: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "protocol_version": self.protocol_version,
            "semantics_version": self.semantics_version,
            "origin_vocabulary_version": self.origin_vocabulary_version,
            "origin_vocabulary_digest": self.origin_vocabulary_digest,
            "session_id": self.session_id,
            "origin": self.origin,
            "native_id": self.native_id,
            "revision_id": self.revision_id,
            "superseded_revision_id": self.superseded_revision_id,
            "content_digest": self.content_digest.to_dict(),
            "head_segment": self.head_segment.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "expected_record_counts": dict(self.expected_record_counts),
            "anchors": {record_id: anchor.to_dict() for record_id, anchor in self.anchors.items()},
            "sequence_rule": self.sequence_rule,
            "completeness": self.completeness,
            "fidelity_gaps": [gap.to_dict() for gap in self.fidelity_gaps],
            "revision_created_at": self.revision_created_at,
        }

    @staticmethod
    def from_dict(payload: dict[str, JSONValue]) -> RevisionManifest:
        try:
            segments_payload = _array(payload["segments"])
            anchors_payload = _object(payload["anchors"])
            fidelity_payload = _array(payload.get("fidelity_gaps", []))
            expected_counts_payload = _object(payload["expected_record_counts"])
            return RevisionManifest(
                protocol_version=_text(payload["protocol_version"]),
                semantics_version=_declared_semantics_version(payload["semantics_version"]),
                origin_vocabulary_version=_integer(payload["origin_vocabulary_version"]),
                origin_vocabulary_digest=_text(payload["origin_vocabulary_digest"]),
                session_id=_text(payload["session_id"]),
                origin=_text(payload["origin"]),
                native_id=_text(payload["native_id"]),
                revision_id=_text(payload["revision_id"]),
                superseded_revision_id=(
                    _text(payload["superseded_revision_id"])
                    if payload.get("superseded_revision_id") is not None
                    else None
                ),
                content_digest=ContentDigest.from_dict(_object(payload["content_digest"])),
                head_segment=SegmentDescriptor.from_dict(_object(payload["head_segment"])),
                segments=tuple(SegmentDescriptor.from_dict(_object(item)) for item in segments_payload),
                expected_record_counts={_text(k): _integer(v) for k, v in expected_counts_payload.items()},
                anchors={_text(k): AnchorEntry.from_dict(_object(v)) for k, v in anchors_payload.items()},
                sequence_rule=_text(payload["sequence_rule"]),
                completeness=_text(payload["completeness"]),
                fidelity_gaps=tuple(FidelityGap.from_dict(_object(item)) for item in fidelity_payload),
                revision_created_at=(
                    _text(payload["revision_created_at"]) if payload.get("revision_created_at") is not None else None
                ),
            )
        except MaterialProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterialManifestError("invalid revision manifest declaration") from exc


def new_manifest_scaffold() -> tuple[str, int]:
    """Return (protocol_version, semantics_version) constants for manifest construction."""
    return PROTOCOL_VERSION, SEMANTICS_VERSION


def require_current_semantics(manifest: RevisionManifest) -> None:
    """Reject revisions whose bytes use semantics not implemented by this reader."""
    if type(manifest.semantics_version) is not int or manifest.semantics_version != SEMANTICS_VERSION:
        raise UnsupportedSemanticsVersionError(
            f"unsupported material semantics version {manifest.semantics_version}; "
            f"current version is {SEMANTICS_VERSION}"
        )
    if manifest.protocol_version != PROTOCOL_VERSION:
        raise MaterialManifestError(f"unsupported material protocol {manifest.protocol_version!r}")
    if (
        type(manifest.content_digest.canonicalizer_version) is not int
        or manifest.content_digest.canonicalizer_version != CANONICALIZER_VERSION
        or manifest.content_digest.media_type != SEGMENT_MEDIA_TYPE
    ):
        raise MaterialManifestError("unsupported material canonicalizer or media type")


__all__ = [
    "AnchorEntry",
    "ContentDigest",
    "FidelityGap",
    "RevisionManifest",
    "require_current_semantics",
    "SegmentDescriptor",
    "new_manifest_scaffold",
]
