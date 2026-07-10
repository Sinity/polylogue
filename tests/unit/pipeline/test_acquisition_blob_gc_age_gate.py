"""Anti-vacuity proofs for blob publication receipts and GC races."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from polylogue.archive.message.roles import Role
from polylogue.config import Source
from polylogue.core.enums import Origin, Provider
from polylogue.pipeline.services.acquisition import AcquisitionService
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.blob_gc import MIN_AGE_S, BlobGCResult, run_blob_gc_report
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
    BlobPublicationReceipt,
    BlobPublicationReservationStore,
    exclude_archive_blob_publishers,
    reconcile_blob_publication_reservations,
)
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.asyncio
@pytest.mark.frozen_clock_modules("polylogue.pipeline.services.acquisition_records")
async def test_slow_following_source_cannot_age_uncommitted_blob_into_gc(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    frozen_clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured 61-second prefetch window is protected before persistence."""
    del workspace_env
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    source_root = tmp_path / "chatgpt-export"
    source_root.mkdir()
    (source_root / "00-first.json").write_text('{"title":"first","mapping":{}}')
    (source_root / "01-slow.json").write_text('{"title":"slow","mapping":{}}')

    initial_time = frozen_clock.time()
    original_prepare = BlobStore.prepare_from_path
    original_flush = ArchiveBlobPublisher.flush
    gc_report: BlobGCResult | None = None
    measured_window_s = 0.0

    def measured_prepare(
        store: BlobStore,
        source: Path,
        *,
        heartbeat: object | None = None,
    ) -> PreparedBlob:
        nonlocal measured_window_s
        prepared = original_prepare(store, source, heartbeat=heartbeat)  # type: ignore[arg-type]
        if source.name == "01-slow.json":
            frozen_clock.advance(MIN_AGE_S + 1)
            measured_window_s = frozen_clock.time() - initial_time
        return prepared

    def measured_flush(publisher: ArchiveBlobPublisher) -> tuple[BlobPublicationReceipt, ...]:
        nonlocal gc_report
        receipts = original_flush(publisher)
        if receipts and gc_report is None:
            first = receipts[0]
            os.utime(publisher._store.blob_path(first.blob_hash), (initial_time, initial_time))
            gc_report = run_blob_gc_report(archive_root / "source.db", archive_root / "blob", max_batch=10)
        return receipts

    monkeypatch.setattr(BlobStore, "prepare_from_path", measured_prepare)
    monkeypatch.setattr(ArchiveBlobPublisher, "flush", measured_flush)
    try:
        result = await AcquisitionService(backend).acquire_sources([Source(name="chatgpt", path=source_root)])
    finally:
        await backend.close()

    assert result.acquired == 2
    assert measured_window_s == MIN_AGE_S + 1
    assert gc_report is not None
    assert gc_report.deleted_count == 0
    assert gc_report.skipped_reserved == 1
    assert run_blob_gc_report(archive_root / "source.db", archive_root / "blob").deleted_count == 0


def test_two_same_hash_publishers_consume_only_their_own_receipt(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload = b"same content, independent publishers"
    first = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    second = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    blob_hash, size = first.write_from_bytes(payload)
    second_hash, _ = second.write_from_bytes(payload)
    first_receipt = first.receipt_id(blob_hash)
    second_receipt = second.receipt_id(second_hash)
    first.flush()
    second.flush()
    assert first_receipt and second_receipt and first_receipt != second_receipt

    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="p2.json",
            source_index=0,
            blob_hash=bytes.fromhex(blob_hash),
            blob_size=size,
            acquired_at_ms=1,
            raw_id="publisher-two",
            blob_publication_receipt_id=second_receipt,
        )
        conn.execute("DELETE FROM blob_refs WHERE ref_id = 'publisher-two'")
        conn.execute("DELETE FROM raw_sessions WHERE raw_id = 'publisher-two'")
        remaining = conn.execute(
            "SELECT publication_id FROM blob_publication_reservations ORDER BY publication_id"
        ).fetchall()
    assert remaining == [(first_receipt,)]

    store = BlobStore(archive_root / "blob")
    os.utime(store.blob_path(blob_hash), (1_700_000_000, 1_700_000_000))
    report = run_blob_gc_report(archive_root / "source.db", store.root)
    assert report.deleted_count == 0
    assert report.skipped_reserved == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="p1.json",
            source_index=0,
            blob_hash=bytes.fromhex(blob_hash),
            blob_size=size,
            acquired_at_ms=2,
            raw_id="publisher-one",
            blob_publication_receipt_id=first_receipt,
        )
    assert store.exists(blob_hash)


def test_live_publisher_missing_path_receipt_is_retained_by_reconciliation(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    publish_entered = Event()
    allow_publish = Event()

    class PausedStore(BlobStore):
        def publish_many(self, prepared):  # type: ignore[no-untyped-def]
            publish_entered.set()
            assert allow_publish.wait(timeout=2)
            return super().publish_many(prepared)

    store = PausedStore(archive_root / "blob")
    publisher = ArchiveBlobPublisher(archive_root / "source.db", store.root, store=store)
    blob_hash, _ = publisher.write_from_bytes(b"paused before final path")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(publisher.flush)
        assert publish_entered.wait(timeout=2)
        outcome = reconcile_blob_publication_reservations(archive_root / "source.db", store.root)
        assert outcome.cleared_missing == 0
        assert outcome.retained_missing == 1
        allow_publish.set()
        future.result(timeout=2)
    assert store.exists(blob_hash)
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1


def test_reconciliation_with_writer_exclusion_clears_only_terminal_receipts(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    store = BlobStore(archive_root / "blob")
    publisher = ArchiveBlobPublisher(source_db, store.root)
    missing_hash, missing_size = publisher.write_from_bytes(b"missing terminal publication")
    referenced_hash, referenced_size = publisher.write_from_bytes(b"referenced terminal publication")
    unresolved_hash, _ = publisher.write_from_bytes(b"unresolved live-looking publication")
    publisher.flush()
    store.blob_path(missing_hash).unlink()
    with sqlite3.connect(source_db) as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="referenced.json",
            source_index=0,
            blob_hash=bytes.fromhex(referenced_hash),
            blob_size=referenced_size,
            acquired_at_ms=2,
            raw_id="referenced-publication",
        )

    with exclude_archive_blob_publishers(source_db) as exclusion:
        outcome = reconcile_blob_publication_reservations(
            source_db,
            store.root,
            index_db_path=archive_root / "index.db",
            writer_exclusion=exclusion,
        )

    assert missing_size > 0
    assert outcome.cleared_missing == 1
    assert outcome.cleared_referenced == 1
    assert outcome.unresolved == 1
    with sqlite3.connect(source_db) as conn:
        remaining = conn.execute("SELECT lower(hex(blob_hash)) FROM blob_publication_reservations").fetchall()
    assert remaining == [(unresolved_hash,)]


def test_failed_reservation_batch_never_publishes_final_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    blob_hash, _ = publisher.write_from_bytes(b"must remain private")

    def fail_reservation(_store: object, _receipts: object) -> None:
        raise RuntimeError("reservation unavailable")

    monkeypatch.setattr(BlobPublicationReservationStore, "reserve_many", fail_reservation)
    with pytest.raises(RuntimeError, match="reservation unavailable"):
        publisher.flush()
    assert not BlobStore(archive_root / "blob").exists(blob_hash)
    publisher.discard_pending()


def test_publisher_batches_reservations_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    calls: list[int] = []
    original = BlobPublicationReservationStore.reserve_many

    def observe(store: BlobPublicationReservationStore, receipts):  # type: ignore[no-untyped-def]
        calls.append(len(receipts))
        assert all(not BlobStore(archive_root / "blob").exists(item.blob_hash) for item in receipts)
        return original(store, receipts)

    monkeypatch.setattr(BlobPublicationReservationStore, "reserve_many", observe)
    for index in range(200):
        publisher.write_from_bytes(f"payload-{index}".encode())
    receipts = publisher.flush()
    assert calls == [200]
    assert len(receipts) == 200


def test_gc_dry_run_does_not_block_concurrent_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage import blob_gc

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    store = BlobStore(archive_root / "blob")
    orphan_hash, _ = store.write_from_bytes(b"old orphan")
    os.utime(store.blob_path(orphan_hash), (1_700_000_000, 1_700_000_000))
    enumeration_entered = Event()
    allow_enumeration = Event()
    original_candidates = blob_gc._candidate_blobs

    def paused_candidates(*args, **kwargs):  # type: ignore[no-untyped-def]
        enumeration_entered.set()
        assert allow_enumeration.wait(timeout=2)
        return original_candidates(*args, **kwargs)

    monkeypatch.setattr(blob_gc, "_candidate_blobs", paused_candidates)
    with ThreadPoolExecutor(max_workers=2) as executor:
        gc_future = executor.submit(run_blob_gc_report, archive_root / "source.db", store.root, 10, dry_run=True)
        assert enumeration_entered.wait(timeout=2)
        publisher = ArchiveBlobPublisher(archive_root / "source.db", store.root)
        publisher.write_from_bytes(b"concurrent publisher")
        publish_future = executor.submit(publisher.flush)
        assert publish_future.result(timeout=1)
        allow_enumeration.set()
        assert gc_future.result(timeout=2).dry_run


def test_destructive_gc_serializes_final_recheck_and_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage import blob_gc

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload = b"old orphan concurrently reacquired"
    store = BlobStore(archive_root / "blob")
    blob_hash, _ = store.write_from_bytes(payload)
    os.utime(store.blob_path(blob_hash), (1_700_000_000, 1_700_000_000))
    recheck_entered = Event()
    allow_recheck = Event()
    original_references = blob_gc._reference_surfaces

    def paused_recheck(*args, **kwargs):  # type: ignore[no-untyped-def]
        recheck_entered.set()
        assert allow_recheck.wait(timeout=2)
        return original_references(*args, **kwargs)

    monkeypatch.setattr(blob_gc, "_reference_surfaces", paused_recheck)
    publisher = ArchiveBlobPublisher(archive_root / "source.db", store.root)
    publisher.write_from_bytes(payload)
    with ThreadPoolExecutor(max_workers=2) as executor:
        gc_future = executor.submit(run_blob_gc_report, archive_root / "source.db", store.root, 10)
        assert recheck_entered.wait(timeout=2)
        publish_future = executor.submit(publisher.flush)
        assert not publish_future.done()
        allow_recheck.set()
        assert gc_future.result(timeout=2).deleted_count == 1
        publish_future.result(timeout=2)
    assert store.exists(blob_hash)


def test_index_only_attachment_consumes_receipt_after_index_commit(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    payload = b"index-only attachment"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="index-only-attachment",
        title="attachment",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="see it", position=0)],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="a1",
                message_provider_id="m1",
                name="evidence.txt",
                inline_bytes=payload,
            )
        ],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_parsed(session)

    blob_hash = hashlib.sha256(payload).digest()
    with sqlite3.connect(archive_root / "source.db") as source_conn:
        assert source_conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
    with sqlite3.connect(archive_root / "index.db") as index_conn:
        assert (
            index_conn.execute("SELECT COUNT(*) FROM attachments WHERE blob_hash = ?", (blob_hash,)).fetchone()[0] == 1
        )
