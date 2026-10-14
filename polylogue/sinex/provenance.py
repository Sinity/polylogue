"""Bind actual retained source materials to a normalized publication manifest."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from polylogue.material_protocol.v1.canonical import canonical_bytes, parse_json_value
from polylogue.sinex.models import PublicationOriginal, PublicationPayload

ORIGINAL_MATERIALS_COMMITMENT = "polylogue.original-materials/v1:sha256:"


def original_descriptors(originals: tuple[PublicationOriginal, ...]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    descriptors: list[dict[str, object]] = []
    for original in originals:
        identity = (original.role, original.source_ref)
        if identity in seen:
            raise ValueError("duplicate_publication_original_identity")
        seen.add(identity)
        descriptors.append(
            {
                "role": original.role,
                "source_ref": original.source_ref,
                "sha256": original.blob_hash,
                "size_bytes": original.size_bytes,
                "media_type": original.media_type,
                "acquisition": original.acquisition.to_dict(),
            }
        )
    return descriptors


def originals_commitment(originals: tuple[PublicationOriginal, ...], normalizer_version: str) -> str:
    from typing import cast

    from polylogue.core.json import JSONValue

    if type(normalizer_version) is not str or not normalizer_version.strip():
        raise ValueError("publication_normalizer_version_missing")
    payload = {"normalizer_version": normalizer_version, "originals": original_descriptors(originals)}
    return ORIGINAL_MATERIALS_COMMITMENT + hashlib.sha256(canonical_bytes(cast(JSONValue, payload))).hexdigest()


def bind_originals(
    payload: PublicationPayload, originals: tuple[PublicationOriginal, ...], *, normalizer_version: str
) -> PublicationPayload:
    """Use the declared cross-system slot without changing normalized bytes."""
    if not any(original.role == "provider_native" for original in originals):
        raise ValueError("publication_provider_original_missing")
    manifest = parse_json_value(payload.manifest_bytes)
    content = manifest.get("content_digest") if isinstance(manifest, dict) else None
    if not isinstance(content, dict):
        raise ValueError("publication_manifest_content_digest_missing")
    content["provider_digest"] = originals_commitment(originals, normalizer_version)
    manifest_bytes = canonical_bytes(manifest) + b"\n"
    return replace(
        payload,
        manifest_bytes=manifest_bytes,
        manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
        originals=originals,
        normalizer_version=normalizer_version,
    )


def require_originals_commitment(payload: PublicationPayload) -> None:
    manifest = parse_json_value(payload.manifest_bytes)
    content = manifest.get("content_digest") if isinstance(manifest, dict) else None
    if not isinstance(content, dict):
        raise ValueError("publication_manifest_content_digest_missing")
    if content.get("provider_digest") != originals_commitment(payload.originals, payload.normalizer_version):
        raise ValueError("publication_originals_commitment_mismatch")
