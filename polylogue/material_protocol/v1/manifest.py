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
from polylogue.material_protocol.v1.constants import PROTOCOL_VERSION, SEGMENT_MEDIA_TYPE, SEMANTICS_VERSION


@dataclass(frozen=True, slots=True)
class SegmentDescriptor:
    index: int
    filename: str
    sha256: str
    size_bytes: int
    record_count: int
    first_seq: int
    last_seq: int

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
            index=int(payload["index"]),  # type: ignore[arg-type]
            filename=str(payload["filename"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),  # type: ignore[arg-type]
            record_count=int(payload["record_count"]),  # type: ignore[arg-type]
            first_seq=int(payload["first_seq"]),  # type: ignore[arg-type]
            last_seq=int(payload["last_seq"]),  # type: ignore[arg-type]
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
            polylogue_sha256=str(payload["polylogue_sha256"]),
            canonicalizer_version=int(payload["canonicalizer_version"]),  # type: ignore[arg-type]
            size_bytes=int(payload["size_bytes"]),  # type: ignore[arg-type]
            media_type=str(payload.get("media_type", SEGMENT_MEDIA_TYPE)),
            sinex_cas_digest=(
                str(payload["sinex_cas_digest"]) if payload.get("sinex_cas_digest") is not None else None
            ),
            provider_digest=(str(payload["provider_digest"]) if payload.get("provider_digest") is not None else None),
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
            segment_index=int(payload["segment_index"]),  # type: ignore[arg-type]
            line_index=int(payload["line_index"]),  # type: ignore[arg-type]
            seq=int(payload["seq"]),  # type: ignore[arg-type]
            kind=str(payload["kind"]),
            sha256=str(payload["sha256"]),
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
            scope=str(payload["scope"]),
            record_id=str(payload["record_id"]),
            gap_kind=str(payload["gap_kind"]),
            detail=str(payload.get("detail", "")),
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
        segments_payload = payload["segments"]
        assert isinstance(segments_payload, list)
        anchors_payload = payload["anchors"]
        assert isinstance(anchors_payload, dict)
        fidelity_payload = payload.get("fidelity_gaps", [])
        assert isinstance(fidelity_payload, list)
        expected_counts_payload = payload["expected_record_counts"]
        assert isinstance(expected_counts_payload, dict)
        return RevisionManifest(
            protocol_version=str(payload["protocol_version"]),
            semantics_version=int(payload["semantics_version"]),  # type: ignore[arg-type]
            origin_vocabulary_version=int(payload["origin_vocabulary_version"]),  # type: ignore[arg-type]
            origin_vocabulary_digest=str(payload["origin_vocabulary_digest"]),
            session_id=str(payload["session_id"]),
            origin=str(payload["origin"]),
            native_id=str(payload["native_id"]),
            revision_id=str(payload["revision_id"]),
            superseded_revision_id=(
                str(payload["superseded_revision_id"]) if payload.get("superseded_revision_id") is not None else None
            ),
            content_digest=ContentDigest.from_dict(payload["content_digest"]),  # type: ignore[arg-type]
            segments=tuple(SegmentDescriptor.from_dict(item) for item in segments_payload),  # type: ignore[arg-type]
            expected_record_counts={str(k): int(v) for k, v in expected_counts_payload.items()},  # type: ignore[arg-type]
            anchors={str(k): AnchorEntry.from_dict(v) for k, v in anchors_payload.items()},  # type: ignore[arg-type]
            sequence_rule=str(payload["sequence_rule"]),
            completeness=str(payload["completeness"]),
            fidelity_gaps=tuple(FidelityGap.from_dict(item) for item in fidelity_payload),  # type: ignore[arg-type]
            revision_created_at=(
                str(payload["revision_created_at"]) if payload.get("revision_created_at") is not None else None
            ),
        )


def new_manifest_scaffold() -> tuple[str, int]:
    """Return (protocol_version, semantics_version) constants for manifest construction."""
    return PROTOCOL_VERSION, SEMANTICS_VERSION


__all__ = [
    "AnchorEntry",
    "ContentDigest",
    "FidelityGap",
    "RevisionManifest",
    "SegmentDescriptor",
    "new_manifest_scaffold",
]
