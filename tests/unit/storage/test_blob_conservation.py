from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance import blob_conservation
from polylogue.storage.blob_liveness import BlobLivenessProjection
from polylogue.storage.blob_store import BlobStore


def _empty_archive(root: Path) -> None:
    sqlite3.connect(root / "source.db").close()
    sqlite3.connect(root / "index.db").close()


def test_conservation_flags_orphan_and_dangling_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _empty_archive(tmp_path)
    store = BlobStore(tmp_path / "blob")
    orphan, _ = store.write_from_bytes(b"orphan")
    dangling = hashlib.sha256(b"missing").hexdigest()
    projection = BlobLivenessProjection(
        frozenset({dangling}), owner_hashes=(("source.db.raw_sessions", frozenset({dangling})),)
    )
    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", lambda *args, **kwargs: projection)
    monkeypatch.setattr(blob_conservation, "_source_recoverability_proofs", lambda *args, **kwargs: [])

    report = blob_conservation.check_blob_conservation(tmp_path)

    assert report.orphan_blobs == 1
    assert report.orphan_sample == (orphan,)
    assert report.dangling_references == 1
    assert report.recoverable_references == 0
    assert not report.ok


def test_conservation_excludes_staged_work_from_blob_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _empty_archive(tmp_path)
    store = BlobStore(tmp_path / "blob")
    store.staging_root.mkdir(parents=True)
    (store.staging_root / "in-flight").write_bytes(b"not-yet-published")
    projection = BlobLivenessProjection(frozenset())
    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", lambda *args, **kwargs: projection)

    report = blob_conservation.check_blob_conservation(tmp_path)

    assert report.present_blobs == 0
    assert report.orphan_blobs == 0
    assert report.staged_in_flight == 1
    assert report.ok


def test_conservation_excludes_backup_prover_confirmed_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _empty_archive(tmp_path)
    recoverable = hashlib.sha256(b"recoverable").hexdigest()
    projection = BlobLivenessProjection(
        frozenset({recoverable}), owner_hashes=(("source.db.raw_sessions", frozenset({recoverable})),)
    )
    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", lambda *args, **kwargs: projection)
    monkeypatch.setattr(
        blob_conservation,
        "_source_recoverability_proofs",
        lambda *args, **kwargs: [{"blob_hash": recoverable}],
    )

    report = blob_conservation.check_blob_conservation(tmp_path)

    assert report.dangling_references == 0
    assert report.recoverable_references == 1
    assert report.ok
