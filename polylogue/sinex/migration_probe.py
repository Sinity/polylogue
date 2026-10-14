"""Execute the Source migration's actual outbox and blob-owner consumers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

from polylogue.core.enums import Origin
from polylogue.material_protocol.v1 import SessionMaterial, encode_session_revision
from polylogue.material_protocol.v1.canonical import canonical_bytes
from polylogue.sinex.models import PublicationMode, PublicationObligation, PublicationOriginal, PublicationPayload
from polylogue.sinex.obligations import load_payload, stage_payload
from polylogue.sinex.provenance import bind_originals
from polylogue.storage.blob_liveness import inspect_blob_liveness
from polylogue.storage.sqlite.durable_change_train import _runtime_probe_source_connection


def publication_originals_runtime_probe(consumer: Callable[..., object], *, writer: bool) -> str:
    """The migration owner supplies its verified isolated Source schema."""
    encoded = encode_session_revision(
        SessionMaterial(origin=Origin.CLAUDE_CODE_SESSION, native_id="migration-probe"),
        revision_created_at="2026-01-01T00:00:00Z",
    )
    manifest = canonical_bytes(encoded.manifest.to_dict()) + b"\n"
    names = encoded.segment_filenames()
    payload = PublicationPayload(
        object_id=encoded.manifest.session_id,
        protocol_version=encoded.manifest.protocol_version,
        revision_id=encoded.manifest.revision_id,
        manifest_digest=hashlib.sha256(manifest).hexdigest(),
        manifest_bytes=manifest,
        segments=tuple((names[index], encoded.segments[index]) for index in sorted(encoded.segments)),
    )
    original = PublicationOriginal(
        "provider_native",
        "polylogue.raw:migration-probe",
        hashlib.sha256(b"neutral").hexdigest(),
        len(b"neutral"),
        "application/octet-stream",
    )
    payload = bind_originals(payload, (original,), normalizer_version="polylogue-migration-probe/v1")
    with _runtime_probe_source_connection() as conn:
        if writer:
            obligation = cast(
                PublicationObligation, consumer(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=1)
            )
            assert load_payload(conn, obligation) == payload
        else:
            obligation = stage_payload(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=1)
            assert consumer(conn, obligation) == payload
        assert "source.db.sinex_publication_originals" in inspect_blob_liveness(conn, original.blob_hash).surfaces
    return "original publication custody writes, reads and remains a direct Source blob owner"
