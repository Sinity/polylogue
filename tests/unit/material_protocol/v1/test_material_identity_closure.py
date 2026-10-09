"""Verified material revisions reconstruct exact accepted identities and owners."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import pytest

from polylogue.core.enums import Origin, Role
from polylogue.material_protocol.v1 import (
    MaterialProtocolError,
    MessageInput,
    RevisionManifest,
    SessionMaterial,
    decode_session_revision,
    encode_session_revision,
    verify_revision,
)
from polylogue.material_protocol.v1.decode import iter_records
from polylogue.material_protocol.v1.encode import _content_digest, _pack_segments
from tests.unit.material_protocol.v1.fixture import build_small_session_material


def test_decomposed_native_identifiers_preserve_exact_archive_identity() -> None:
    material = SessionMaterial(
        origin=Origin.CLAUDE_CODE_SESSION,
        native_id="cafe\u0301",
        messages=(MessageInput(native_id="e\u0301", position=0, role=Role.USER, text="e\u0301"),),
    )
    encoded = encode_session_revision(material, revision_created_at="2026-01-01T00:00:00Z")
    verify_revision(encoded.manifest, encoded.segments)
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert decoded.session["session_id"] == material.session_id
    assert decoded.messages[0].message_id == f"{material.session_id}:n:e\u0301"
    assert decoded.messages[0].text == "é"


def test_duplicate_direct_material_record_ids_are_refused_before_anchor_overwrite() -> None:
    material = SessionMaterial(
        origin=Origin.CLAUDE_CODE_SESSION,
        native_id="s",
        messages=(
            MessageInput(native_id="m", position=0, role=Role.USER, text="first"),
            MessageInput(native_id="m", position=1, role=Role.USER, text="second"),
        ),
    )
    with pytest.raises(MaterialProtocolError):
        encode_session_revision(material, revision_created_at="2026-01-01T00:00:00Z")


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin", "chatgpt-export"),
        ("native_id", "other"),
        ("revision_id", "wrong"),
        ("protocol_version", "unknown"),
        ("completeness", "invented"),
        ("sequence_rule", "unordered"),
    ],
)
def test_manifest_declarations_cannot_disagree_with_verified_bytes(field: str, value: str) -> None:
    encoded = encode_session_revision(build_small_session_material(), revision_created_at="2026-01-01T00:00:00Z")
    replace = cast(Callable[..., RevisionManifest], dataclasses.replace)
    manifest = replace(encoded.manifest, **{field: value})
    with pytest.raises(MaterialProtocolError):
        verify_revision(manifest, encoded.segments)


@pytest.mark.parametrize("field,value", [("segments", {}), ("anchors", []), ("native_id", 5)])
def test_malformed_manifest_shapes_have_typed_refusals(field: str, value: object) -> None:
    encoded = encode_session_revision(build_small_session_material(), revision_created_at="2026-01-01T00:00:00Z")
    payload = encoded.manifest.to_dict()
    payload[field] = value  # type: ignore[assignment]
    with pytest.raises(MaterialProtocolError):
        RevisionManifest.from_dict(payload)


@pytest.mark.parametrize("kind", ["attachment", "block"])
def test_byte_reconciled_orphan_material_is_refused_by_verify_and_decode(kind: str) -> None:
    encoded = encode_session_revision(build_small_session_material(), revision_created_at="2026-01-01T00:00:00Z")
    head = encoded.segments[-1]
    records = [
        r for r in iter_records(encoded.manifest, encoded.segments) if r["kind"] not in {"session", "usage", "lineage"}
    ]
    record = next(r for r in records if r["kind"] == kind)
    record["message_id"] = "claude-code-session:demo-session-1:n:ghost"
    if kind == "block":
        owner = next(
            r
            for r in records
            if r["kind"] == "message" and r["message_id"] != record["message_id"] and r["block_count"] == 1
        )
        owner["block_count"] = 0
    descriptors, segments, anchors, _counts = _pack_segments(
        [{k: v for k, v in r.items() if k != "seq"} for r in records],
        start_seq=0,
        start_segment_index=0,
        max_records_per_segment=500,
    )
    segments[-1] = head
    digest = _content_digest(head, [segments[0]])
    manifest = dataclasses.replace(
        encoded.manifest,
        segments=tuple(descriptors),
        content_digest=digest,
        revision_id=digest.polylogue_sha256,
        anchors={**{k: v for k, v in encoded.manifest.anchors.items() if v.segment_index == -1}, **anchors},
    )
    with pytest.raises(MaterialProtocolError):
        verify_revision(manifest, segments)
    with pytest.raises(MaterialProtocolError):
        decode_session_revision(manifest, segments)


@pytest.mark.parametrize("native_identity", ["", "AA", "6f746865722d6e6174697665"])
def test_byte_reconciled_false_attachment_native_identity_is_refused(native_identity: str) -> None:
    encoded = encode_session_revision(build_small_session_material(), revision_created_at="2026-01-01T00:00:00Z")
    head = encoded.segments[-1]
    records = [
        r for r in iter_records(encoded.manifest, encoded.segments) if r["kind"] not in {"session", "usage", "lineage"}
    ]
    attachment = next(r for r in records if r["kind"] == "attachment")
    attachment["native_identity"] = native_identity
    descriptors, segments, anchors, _counts = _pack_segments(
        [{k: v for k, v in r.items() if k != "seq"} for r in records],
        start_seq=0,
        start_segment_index=0,
        max_records_per_segment=500,
    )
    segments[-1] = head
    digest = _content_digest(head, [segments[0]])
    manifest = dataclasses.replace(
        encoded.manifest,
        segments=tuple(descriptors),
        content_digest=digest,
        revision_id=digest.polylogue_sha256,
        anchors={**{k: v for k, v in encoded.manifest.anchors.items() if v.segment_index == -1}, **anchors},
    )
    with pytest.raises(MaterialProtocolError, match="attachment reference identity"):
        verify_revision(manifest, segments)
    with pytest.raises(MaterialProtocolError, match="attachment reference identity"):
        decode_session_revision(manifest, segments)
