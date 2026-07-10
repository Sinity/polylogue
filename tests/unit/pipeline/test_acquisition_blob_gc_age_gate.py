"""Regression harness for blob publication versus durable raw references."""

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
from polylogue.core.enums import Provider
from polylogue.pipeline.services.acquisition_streams import iter_raw_record_stream
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.blob_gc import MIN_AGE_S, BlobGCResult, run_blob_gc_report
from polylogue.storage.blob_publication import (
    BlobPublicationReservationStore,
    reconcile_blob_publication_reservations,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
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
    """A later streamed artifact must not expose an earlier blob to GC.

    ``iter_source_raw_stream`` drains a source batch before yielding its first
    record to persistence.  Advancing the controlled clock while the second
    provider-shaped artifact is streamed models an arbitrarily slow read
    without sleeping or allocating a large file.
    """
    del workspace_env
    archive_root = tmp_path / "archive"
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    source_root = tmp_path / "chatgpt-export"
    source_root.mkdir()
    (source_root / "00-first.json").write_text('{"title":"first","mapping":{}}')
    (source_root / "01-slow.json").write_text('{"title":"slow","mapping":{}}')

    original_write = BlobStore.write_from_path
    first_blob_hash: str | None = None
    gc_report: BlobGCResult | None = None
    deleted_before_commit = False

    def measured_write(
        store: BlobStore,
        source: Path,
        *,
        heartbeat: object | None = None,
    ) -> tuple[str, int]:
        nonlocal first_blob_hash, gc_report, deleted_before_commit
        blob_hash, blob_size = original_write(store, source, heartbeat=heartbeat)  # type: ignore[arg-type]
        if source.name == "00-first.json":
            first_blob_hash = blob_hash
            os.utime(store.blob_path(blob_hash), (frozen_clock.time(), frozen_clock.time()))
        elif source.name == "01-slow.json":
            assert first_blob_hash is not None
            frozen_clock.advance(MIN_AGE_S + 1)
            gc_report = run_blob_gc_report(
                archive_root / "source.db",
                store.root,
                max_batch=10,
            )
            deleted_before_commit = not store.exists(first_blob_hash)
        return blob_hash, blob_size

    monkeypatch.setattr(BlobStore, "write_from_path", measured_write)
    try:
        records = [
            record
            async for record in iter_raw_record_stream(
                Source(name="chatgpt", path=source_root),
                blob_root=archive_root / "blob",
            )
        ]
        for record in records:
            await backend.save_raw_session(record)
    finally:
        await backend.close()

    after_reference = run_blob_gc_report(
        archive_root / "source.db",
        archive_root / "blob",
        max_batch=10,
    )

    assert len(records) == 2
    assert first_blob_hash is not None
    assert gc_report is not None
    assert gc_report.deleted_count == 0
    assert gc_report.skipped_reserved == 1
    assert not deleted_before_commit
    assert BlobStore(archive_root / "blob").exists(first_blob_hash)
    assert after_reference.deleted_count == 0
    assert after_reference.skipped_referenced >= 1


def test_publication_callback_precedes_new_and_deduplicated_visibility(tmp_path: Path) -> None:
    root = tmp_path / "blob"
    observations: list[bool] = []

    def observe(blob_hash: str, _size_bytes: int) -> None:
        observations.append(BlobStore(root).exists(blob_hash))

    store = BlobStore(root, before_publish=observe)
    first_hash, _ = store.write_from_bytes(b"same payload")
    second_hash, _ = store.write_from_bytes(b"same payload")

    assert first_hash == second_hash
    assert observations == [False, True]


def test_failed_reservation_never_publishes_final_blob(tmp_path: Path) -> None:
    root = tmp_path / "blob"

    def fail_reservation(_blob_hash: str, _size_bytes: int) -> None:
        raise RuntimeError("reservation unavailable")

    store = BlobStore(root, before_publish=fail_reservation)
    with pytest.raises(RuntimeError, match="reservation unavailable"):
        store.write_from_bytes(b"must remain temporary")

    assert list(BlobStore(root).iter_all()) == []


def test_crash_reconciliation_clears_only_provably_safe_reservations(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    del backend
    store = BlobStore(archive_root / "blob")
    reservations = BlobPublicationReservationStore.create(archive_root / "source.db")

    missing_hash = "1" * 64
    reservations.reserve(missing_hash, 0)
    unresolved_hash, unresolved_size = store.write_from_bytes(b"unresolved published bytes")
    reservations.reserve(unresolved_hash, unresolved_size)
    referenced_hash, referenced_size = store.write_from_bytes(b"already referenced bytes")
    reservations.reserve(referenced_hash, referenced_size)

    conn = sqlite3.connect(archive_root / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO blob_refs (
                blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
            ) VALUES (?, 'ref', 'raw_payload', 'fixture', ?, 1)
            """,
            (bytes.fromhex(referenced_hash), referenced_size),
        )
        conn.commit()
    finally:
        conn.close()

    outcome = reconcile_blob_publication_reservations(
        archive_root / "source.db",
        archive_root / "blob",
    )

    assert outcome.cleared_missing == 1
    assert outcome.cleared_referenced == 1
    assert outcome.unresolved == 1
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        remaining = conn.execute("SELECT hex(blob_hash) FROM blob_publication_reservations").fetchall()
    finally:
        conn.close()
    assert remaining == [(unresolved_hash.upper(),)]


def test_gc_first_serializes_unlink_before_deduplicated_republication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    backend = SQLiteBackend(db_path=archive_root / "index.db")
    del backend
    plain_store = BlobStore(archive_root / "blob")
    payload = b"old orphan concurrently reacquired"
    blob_hash, _ = plain_store.write_from_bytes(payload)
    target = plain_store.blob_path(blob_hash)
    os.utime(target, (1_700_000_000, 1_700_000_000))

    unlink_entered = Event()
    allow_unlink = Event()
    reservation_attempted = Event()
    real_unlink = Path.unlink

    def controlled_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == target:
            unlink_entered.set()
            assert allow_unlink.wait(timeout=2)
        real_unlink(path, missing_ok=missing_ok)

    reservations = BlobPublicationReservationStore.create(archive_root / "source.db")

    def reserve(blob_hash_arg: str, size_bytes: int) -> None:
        reservation_attempted.set()
        reservations.reserve(blob_hash_arg, size_bytes)

    publishing_store = BlobStore(archive_root / "blob", before_publish=reserve)
    monkeypatch.setattr(Path, "unlink", controlled_unlink)

    with ThreadPoolExecutor(max_workers=2) as executor:
        gc_future = executor.submit(
            run_blob_gc_report,
            archive_root / "source.db",
            archive_root / "blob",
            10,
        )
        assert unlink_entered.wait(timeout=2)
        publish_future = executor.submit(publishing_store.write_from_bytes, payload)
        assert reservation_attempted.wait(timeout=2)
        assert not publish_future.done()
        allow_unlink.set()
        gc_result = gc_future.result(timeout=2)
        published_hash, _ = publish_future.result(timeout=2)

    assert gc_result.deleted_count == 1
    assert published_hash == blob_hash
    assert plain_store.exists(blob_hash)
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM blob_publication_reservations WHERE blob_hash = ?",
                (bytes.fromhex(blob_hash),),
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_direct_archive_writer_converts_inline_attachment_reservation_to_source_ref(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    attachment_payload = b"durably referenced attachment"
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="reserved-attachment",
        title="reserved attachment",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="see attachment",
                position=0,
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="attachment-1",
                message_provider_id="m1",
                name="evidence.txt",
                inline_bytes=attachment_payload,
            )
        ],
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        archive.write_raw_and_parsed_result(
            session,
            payload=b'{"title":"reserved attachment"}',
            source_path="/fixtures/session.json",
            acquired_at_ms=1_700_000_000_000,
        )

    attachment_hash = hashlib.sha256(attachment_payload).hexdigest()
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ? AND ref_type = 'attachment'",
                (bytes.fromhex(attachment_hash),),
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
    finally:
        conn.close()

    store = BlobStore(archive_root / "blob")
    os.utime(store.blob_path(attachment_hash), (1_700_000_000, 1_700_000_000))
    report = run_blob_gc_report(archive_root / "source.db", store.root, max_batch=10)
    assert report.deleted_count == 0
    assert report.skipped_referenced >= 1
    assert store.exists(attachment_hash)
