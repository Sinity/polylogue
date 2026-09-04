from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from polylogue.maintenance import blob_conservation
from polylogue.storage.blob_liveness import BlobLivenessProjection
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import ActiveWriterLease, RebuildLeaseUnavailableError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _initialized_archive(root: Path) -> None:
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(root / "index.db", ArchiveTier.INDEX)


def test_conservation_flags_orphan_and_dangling_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _initialized_archive(tmp_path)
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


def test_conservation_reads_the_active_index_without_immutable_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The census follows a promoted index and includes committed WAL rows.

    Anti-vacuity: restoring the conventional path or ``immutable=True`` makes
    this assertion fail.
    """
    _initialized_archive(tmp_path)
    active_index = tmp_path / "promoted" / "index.db"
    projection = BlobLivenessProjection(frozenset())
    observed: dict[str, object] = {}

    monkeypatch.setattr(blob_conservation, "resolve_active_index_path", lambda _root: active_index)

    def project(source_db: Path, **kwargs: object) -> BlobLivenessProjection:
        observed["source_db"] = source_db
        observed.update(kwargs)
        return projection

    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", project)
    blob_conservation.check_blob_conservation(tmp_path)

    assert observed["source_db"] == tmp_path / "source.db"
    assert observed["index_db"] == active_index
    assert "immutable" not in observed


def test_conservation_refuses_a_live_archive_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The census requires a stable archive snapshot.

    Anti-vacuity: removing the writer guard would open the SQLite tiers and
    produce a result while concurrent ingestion can change its evidence.
    """
    monkeypatch.setattr(blob_conservation, "offline_writer_block_reason", lambda _config: "live pidfile PID 42")

    with pytest.raises(RuntimeError, match="requires the archive writer to be stopped: live pidfile PID 42"):
        blob_conservation.check_blob_conservation(tmp_path)


def test_conservation_holds_writer_exclusion_for_its_evidence_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An archive writer cannot begin after the offline probe passes.

    Anti-vacuity: removing the rebuild lease lets ``ActiveWriterLease`` enter
    while the census is collecting its SQLite and filesystem evidence.
    """
    _initialized_archive(tmp_path)
    projection = BlobLivenessProjection(frozenset())

    def project(*args: object, **kwargs: object) -> BlobLivenessProjection:
        writer = ActiveWriterLease(tmp_path)
        try:
            writer.acquire()
        except RebuildLeaseUnavailableError:
            return projection
        writer.close()
        pytest.fail("census did not exclude an archive writer")

    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", project)

    assert blob_conservation.check_blob_conservation(tmp_path).ok


def test_conservation_excludes_staged_work_from_blob_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _initialized_archive(tmp_path)
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


def test_conservation_treats_pending_publication_reservation_as_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _initialized_archive(tmp_path)
    store = BlobStore(tmp_path / "blob")
    reserved, _ = store.write_from_bytes(b"pending-publication")
    monkeypatch.setattr(
        blob_conservation, "project_source_blob_liveness", lambda *args, **kwargs: BlobLivenessProjection(frozenset())
    )
    monkeypatch.setattr(blob_conservation, "_source_blob_reservations", lambda *args, **kwargs: {reserved})

    report = blob_conservation.check_blob_conservation(tmp_path)

    assert report.reserved_blobs == 1
    assert report.orphan_blobs == 0
    assert report.ok


def test_conservation_excludes_backup_prover_confirmed_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _initialized_archive(tmp_path)
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


def test_conservation_rejects_a_corrupt_referenced_blob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A canonical filename alone cannot satisfy conservation.

    Anti-vacuity: counting namespace-shaped files as present would make this
    report pass.
    """
    _initialized_archive(tmp_path)
    store = BlobStore(tmp_path / "blob")
    blob_hash, _ = store.write_from_bytes(b"intact")
    store.blob_path(blob_hash).write_bytes(b"truncated")
    projection = BlobLivenessProjection(
        frozenset({blob_hash}), owner_hashes=(("source.db.raw_sessions", frozenset({blob_hash})),)
    )
    monkeypatch.setattr(blob_conservation, "project_source_blob_liveness", lambda *args, **kwargs: projection)
    monkeypatch.setattr(blob_conservation, "_source_recoverability_proofs", lambda *args, **kwargs: [])

    report = blob_conservation.check_blob_conservation(tmp_path)

    assert report.corrupt_blobs == 1
    assert report.corrupt_sample == (blob_hash,)
    assert report.dangling_references == 1
    assert not report.ok
