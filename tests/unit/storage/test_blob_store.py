from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from polylogue.storage.blob_store import BlobStore


def test_write_from_fileobj_round_trips_content(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"streamed blob content"

    blob_hash, blob_size = blob_store.write_from_fileobj(BytesIO(payload))

    assert blob_hash == hashlib.sha256(payload).hexdigest()
    assert blob_size == len(payload)
    assert blob_store.read_all(blob_hash) == payload


def test_write_from_fileobj_deduplicates_existing_blob(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blobs")
    payload = b"same payload"

    first_hash, first_size = blob_store.write_from_bytes(payload)
    second_hash, second_size = blob_store.write_from_fileobj(BytesIO(payload))

    assert second_hash == first_hash
    assert second_size == first_size == len(payload)
    assert blob_store.stats()["count"] == 1
