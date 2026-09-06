"""Both directions of blob/reference conservation.

The stubbed cases below pin one branch of ``check_blob_conservation`` each.
The two ``seeded_archive`` cases at the end run the real projection and the
real backup prover against archives the production ingest route built, which
is the only way to show that the check partitions a genuine archive rather
than the fixtures handed to it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.maintenance import blob_conservation
from polylogue.maintenance.blob_conservation import check_blob_conservation
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_liveness import BlobLivenessProjection
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.index_generation import ActiveWriterLease, RebuildLeaseUnavailableError

_SEEDED_SESSION_ID = "11111111-2222-3333-4444-555555555555"


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


def test_conservation_reads_the_active_index_without_immutable_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The census follows a promoted index and includes committed WAL rows.

    Anti-vacuity: restoring the conventional path or ``immutable=True`` makes
    this assertion fail.
    """
    _empty_archive(tmp_path)
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
    _empty_archive(tmp_path)
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


def test_conservation_treats_pending_publication_reservation_as_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _empty_archive(tmp_path)
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


def test_conservation_rejects_a_corrupt_referenced_blob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A canonical filename alone cannot satisfy conservation.

    Anti-vacuity: counting namespace-shaped files as present would make this
    report pass.
    """
    _empty_archive(tmp_path)
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


async def _seed_archive(workspace_env: dict[str, Path]) -> Path:
    """Ingest one synthetic transcript through the production live route."""
    root = workspace_env["data_root"] / "projects"
    project = root / "-conservation"
    project.mkdir(parents=True)
    transcript = project / f"{_SEEDED_SESSION_ID}.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "user",
                    "uuid": "u1",
                    "sessionId": _SEEDED_SESSION_ID,
                    "timestamp": "2026-07-20T10:00:00Z",
                    "message": {"role": "user", "content": "hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "sessionId": _SEEDED_SESSION_ID,
                    "timestamp": "2026-07-20T10:00:01Z",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "index.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, suffixes=(".jsonl",)),),
        cursor=CursorStore(workspace_env["data_root"] / "cursor.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        metrics = await processor.ingest_files([transcript], emit_event=False)
        assert metrics.succeeded_file_count == 1
    finally:
        await archive.close()
    return transcript


def _clone(workspace_env: dict[str, Path], name: str) -> Path:
    clone = workspace_env["data_root"] / name
    shutil.copytree(workspace_env["archive_root"], clone)
    return clone


def _canonical_blob_files(archive_root: Path) -> list[Path]:
    store = BlobStore(archive_root / "blob")
    return [
        path for path in (archive_root / "blob").rglob("*") if path.is_file() and store.staging_root not in path.parents
    ]


@pytest.mark.asyncio
async def test_blob_conservation_flags_orphan_blobs_and_dangling_references(
    workspace_env: dict[str, Path],
) -> None:
    """A file with no owning row and a reference with no provable bytes both fail."""
    transcript = await _seed_archive(workspace_env)

    conserved = check_blob_conservation(_clone(workspace_env, "clone-clean"))
    assert conserved.ok is True
    assert (conserved.orphan_blobs, conserved.dangling_references) == (0, 0)
    assert conserved.referenced_blobs == conserved.present_blobs == 1

    orphaned_root = _clone(workspace_env, "clone-orphan")
    orphan_hash, _size = BlobStore(orphaned_root / "blob").write_from_bytes(b"no row owns these bytes")
    orphaned = check_blob_conservation(orphaned_root)
    assert orphaned.ok is False
    assert orphaned.orphan_blobs == 1
    assert orphaned.orphan_sample == (orphan_hash,)
    assert orphaned.dangling_references == 0

    # Bytes gone from the store and the acquisition source gone from disk:
    # nothing left to prove the reference with.
    dangling_root = _clone(workspace_env, "clone-dangling")
    for blob in _canonical_blob_files(dangling_root):
        blob.unlink()
    displaced = transcript.with_suffix(".displaced")
    transcript.rename(displaced)
    try:
        dangling = check_blob_conservation(dangling_root)
    finally:
        displaced.rename(transcript)
    assert dangling.ok is False
    assert dangling.dangling_references == 1
    assert dangling.recoverable_references == 0
    assert dangling.orphan_blobs == 0


@pytest.mark.asyncio
async def test_blob_conservation_excuses_staged_and_recoverable_references(
    workspace_env: dict[str, Path],
) -> None:
    """In-flight staging and prover-recoverable bytes are accounted, not failures."""
    await _seed_archive(workspace_env)

    staged_root = _clone(workspace_env, "clone-staged")
    staging = BlobStore(staged_root / "blob").staging_root
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "blob-in-flight").write_bytes(b"a publish that has not committed yet")
    staged = check_blob_conservation(staged_root)
    assert staged.ok is True
    assert staged.staged_in_flight == 1
    assert staged.invalid_namespace_entries == 0
    assert staged.orphan_blobs == 0

    # Same missing bytes as the dangling case above, with the acquisition
    # source still on disk: the backup prover replays it and the reference is
    # accounted rather than reported lost.
    recoverable_root = _clone(workspace_env, "clone-recoverable")
    for blob in _canonical_blob_files(recoverable_root):
        blob.unlink()
    recoverable = check_blob_conservation(recoverable_root)
    assert recoverable.ok is True
    assert recoverable.recoverable_references == 1
    assert recoverable.dangling_references == 0
    assert recoverable.present_blobs == 0
