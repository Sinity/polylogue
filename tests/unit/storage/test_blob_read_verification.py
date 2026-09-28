"""Read-path blob verification re-hashes only when the file changed (polylogue-1gxyu)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore


def _store_blob(store: BlobStore, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path = store.blob_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def test_repeat_reads_do_not_rehash_an_unchanged_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: route read verification to ``verify`` again and every read re-hashes."""
    store = BlobStore(tmp_path)
    digest = _store_blob(store, b"attachment bytes")
    hashes: list[str] = []
    real_verify = store.verify

    def counting_verify(hash_hex: str) -> bool:
        hashes.append(hash_hex)
        return real_verify(hash_hex)

    monkeypatch.setattr(store, "verify", counting_verify)
    assert all(store.verify_for_read(digest) for _ in range(5))
    assert hashes == [digest]


def test_a_changed_blob_file_is_hashed_again_and_refused(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    digest = _store_blob(store, b"attachment bytes")
    assert store.verify_for_read(digest) is True

    path = store.blob_path(digest)
    path.write_bytes(b"tampered bytes!!")
    os.utime(path, ns=(1, 1))

    assert store.verify_for_read(digest) is False


def test_missing_blob_is_unreadable(tmp_path: Path) -> None:
    assert BlobStore(tmp_path).verify_for_read("0" * 64) is False


def test_a_permission_change_forces_reverification(tmp_path: Path) -> None:
    """Anti-vacuity: leave ctime out of the memo key and an unreadable blob still reads as available."""
    store = BlobStore(tmp_path)
    digest = _store_blob(store, b"attachment bytes")
    assert store.verify_for_read(digest) is True
    path = store.blob_path(digest)
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        assert store.verify_for_read(digest) is False
    finally:
        path.chmod(0o600)


def test_concurrent_cold_reads_hash_a_blob_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: drop the in-flight map and every waiting reader hashes the blob itself."""
    import threading

    store = BlobStore(tmp_path)
    digest = _store_blob(store, b"attachment bytes")
    real_verify = store.verify
    started = threading.Event()
    release = threading.Event()
    hashes: list[str] = []

    def slow_verify(hash_hex: str) -> bool:
        hashes.append(hash_hex)
        started.set()
        assert release.wait(10)
        return real_verify(hash_hex)

    monkeypatch.setattr(store, "verify", slow_verify)
    results: list[bool] = []
    readers = [threading.Thread(target=lambda: results.append(store.verify_for_read(digest))) for _ in range(6)]
    readers[0].start()
    assert started.wait(10)
    for reader in readers[1:]:
        reader.start()
    # Let the other readers reach the in-flight wait before the hash finishes.
    deadline = threading.Event()
    deadline.wait(0.2)
    release.set()
    for reader in readers:
        reader.join(10)

    assert results == [True] * 6
    assert hashes == [digest]
