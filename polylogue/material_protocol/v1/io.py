"""Filesystem read/write for a revision directory (manifest.json + segments/).

Pure I/O glue -- no encoding/decoding logic lives here. Kept deliberately
thin so tests can also exercise encode/decode/verify entirely in memory
(dict[int, bytes]) without touching a filesystem.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.material_protocol.v1.canonical import canonical_bytes, parse_json_value
from polylogue.material_protocol.v1.constants import MANIFEST_FILENAME
from polylogue.material_protocol.v1.encode import EncodedRevision
from polylogue.material_protocol.v1.manifest import RevisionManifest


def write_revision(encoded: EncodedRevision, base_dir: Path) -> None:
    """Write manifest.json + segments/*.ndjson under *base_dir* (created if missing)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = base_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    for descriptor in encoded.manifest.segments:
        (segments_dir / descriptor.filename).write_bytes(encoded.segments[descriptor.index])
    manifest_bytes = canonical_bytes(encoded.manifest.to_dict()) + b"\n"
    (base_dir / MANIFEST_FILENAME).write_bytes(manifest_bytes)


def read_manifest(base_dir: Path) -> RevisionManifest:
    payload = parse_json_value((base_dir / MANIFEST_FILENAME).read_bytes())
    assert isinstance(payload, dict)
    return RevisionManifest.from_dict(payload)


def read_segments(base_dir: Path, manifest: RevisionManifest) -> dict[int, bytes]:
    segments_dir = base_dir / "segments"
    return {descriptor.index: (segments_dir / descriptor.filename).read_bytes() for descriptor in manifest.segments}


def read_revision(base_dir: Path) -> tuple[RevisionManifest, dict[int, bytes]]:
    manifest = read_manifest(base_dir)
    return manifest, read_segments(base_dir, manifest)


__all__ = ["read_manifest", "read_revision", "read_segments", "write_revision"]
