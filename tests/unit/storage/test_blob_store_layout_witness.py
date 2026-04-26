"""Regression test: BlobStore on-disk layout matches the committed witness.

The blob store's content-addressed scheme is durable infrastructure: any
change to it (different hash, different directory split, different framing)
silently invalidates every previously written blob. Re-running the writes
against fresh fixtures and diffing against the committed witness catches the
change before it corrupts an archive.

See `#448 <https://github.com/Sinity/polylogue/issues/448>`_.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from polylogue.proof.witnesses import WITNESS_SCHEMA_VERSION, WitnessMetadata
from polylogue.storage.blob_store import BlobStore

WITNESS_PATH = Path(__file__).resolve().parents[3] / "tests" / "witnesses" / "blob-store-layout.witness.json"
DATA_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "witnesses" / "blob-store-layout.json"


def test_committed_witness_metadata_validates() -> None:
    metadata = WitnessMetadata.read(WITNESS_PATH)
    assert metadata.validation_errors() == ()
    assert metadata.schema_version == WITNESS_SCHEMA_VERSION


def test_blob_store_layout_matches_witness(tmp_path: Path) -> None:
    fixture = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    store = BlobStore(tmp_path)
    for entry in fixture["fixtures"]:
        data = base64.b64decode(entry["content_bytes_b64"])
        sha256_hex, size = store.write_from_bytes(data)
        assert sha256_hex == entry["expected_sha256"], (
            f"Blob {entry['name']!r} hash drifted from witness — addressing scheme changed?"
        )
        assert size == entry["expected_size"], f"Blob {entry['name']!r} size drifted"
        relative_path = store.blob_path(sha256_hex).relative_to(tmp_path).as_posix()
        assert relative_path == entry["expected_relative_path"], (
            f"Blob {entry['name']!r} on-disk path drifted from witness — directory split changed?"
        )


def test_blob_store_addressing_is_content_only(tmp_path: Path) -> None:
    """Same content always lands at the same path, regardless of write order."""
    store = BlobStore(tmp_path)
    payload = b"identical bytes\n"
    h1, _ = store.write_from_bytes(payload)
    h2, _ = store.write_from_bytes(payload)
    assert h1 == h2
    assert store.blob_path(h1) == store.blob_path(h2)
