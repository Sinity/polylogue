"""Captured material claims cross real publication and Source transactions."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.storage_faults import ArchiveStorageFaultError
from polylogue.storage.blob_publication import ArchiveBlobPublisher, BlobPublicationReceipt
from polylogue.storage.materials import (
    PreparedMaterial,
    _prepared_material_from_record,
    _prepared_material_record,
    admit_material,
    prepare_material,
    publish_prepared_materials,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError, record_excised_blob_hash
from polylogue.storage.sqlite.write_lease import current_write_lease, write_lease

pytestmark = pytest.mark.uses_real_clock("prepared file seals use actual inode metadata")


@pytest.fixture
def material_archive(tmp_path: Path) -> Iterator[tuple[sqlite3.Connection, ArchiveBlobPublisher]]:
    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    with sqlite3.connect(root / "source.db") as conn:
        yield conn, publisher
    publisher.discard_pending()


def test_material_publication_uses_bounded_pages_and_exact_claims(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, publisher = material_archive
    observed_pages: list[int] = []
    flush = publisher.flush

    def record_flush() -> tuple[BlobPublicationReceipt, ...]:
        assert not conn.in_transaction
        observed_pages.append(len(publisher._pending))
        return flush()

    monkeypatch.setattr(publisher, "flush", record_flush)

    def prepared_rows() -> Iterator[PreparedMaterial]:
        for index in range(257):
            # Preparation runs before this test's writer phase. A production
            # page cursor supplies these already sealed values instead.
            yield preparations[index]

    assert current_write_lease() is None
    preparations = [
        prepare_material(
            blob_store=publisher,
            source_uri=f"https://example.test/{index}",
            referrer_ref=f"message:{index}",
            payload=b"same captured bytes",
        )
        for index in range(257)
    ]
    with write_lease("synthetic-material-publication", archive_root=publisher.source_db_path.parent):
        publish_prepared_materials(prepared_rows())
        assert observed_pages == [256, 1]
        assert publisher._latest_receipt_by_hash == {}
        reservations = conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0]
        assert reservations == 257
        for index, prepared in enumerate(preparations):
            admit_material(conn, prepared=prepared, observed_at_ms=index, commit=False)
        conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 257
    assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


def test_changed_prepared_file_refuses_before_any_source_reservation(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
) -> None:
    conn, publisher = material_archive
    prepared = prepare_material(
        blob_store=publisher, source_uri="https://example.test/item", referrer_ref="message:item", payload=b"original"
    )
    assert prepared.blob is not None
    os.chmod(prepared.blob.temporary_path, 0o600)
    prepared.blob.temporary_path.write_bytes(b"mutation")
    with write_lease("synthetic-material-publication", archive_root=publisher.source_db_path.parent):
        with pytest.raises(ArchiveStorageFaultError):
            publish_prepared_materials((prepared,))
    assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 0
    prepared.discard()


def test_source_rollback_keeps_exact_receipt_for_retry(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
) -> None:
    conn, publisher = material_archive
    prepared = prepare_material(
        blob_store=publisher, source_uri="https://example.test/item", referrer_ref="message:item", payload=b"captured"
    )
    with write_lease("synthetic-material-publication", archive_root=publisher.source_db_path.parent):
        publish_prepared_materials((prepared,))
        admit_material(conn, prepared=prepared, observed_at_ms=1, commit=False)
        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 1
        admitted = admit_material(conn, prepared=prepared, observed_at_ms=2)
    assert admitted.material_id == prepared.material_id
    assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


def test_excision_between_publish_and_apply_is_typed_and_does_not_admit(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
) -> None:
    conn, publisher = material_archive
    prepared = prepare_material(
        blob_store=publisher, source_uri="https://example.test/item", referrer_ref="message:item", payload=b"captured"
    )
    assert prepared.blob is not None
    with write_lease("synthetic-material-publication", archive_root=publisher.source_db_path.parent):
        publish_prepared_materials((prepared,))
        record_excised_blob_hash(
            conn,
            blob_hash=bytes.fromhex(prepared.blob.hash_hex),
            reason="synthetic excision",
            actor="test",
            excised_at_ms=1,
        )
        conn.commit()
        with pytest.raises(ContentExcisedError):
            admit_material(conn, prepared=prepared, observed_at_ms=2)
    assert conn.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 0


def test_sealed_claim_roundtrip_publishes_and_consumes_the_original_receipt(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
) -> None:
    conn, publisher = material_archive
    original = prepare_material(
        blob_store=publisher, source_uri="https://example.test/item", referrer_ref="message:item", payload=b"captured"
    )
    restored = _prepared_material_from_record(_prepared_material_record(original), publisher)
    assert restored.publication_claim is not None
    assert original.publication_claim is not None
    assert restored.publication_claim.receipt == original.publication_claim.receipt
    with write_lease("synthetic-material-publication", archive_root=publisher.source_db_path.parent):
        publish_prepared_materials((restored,))
        assert conn.execute("SELECT publication_id FROM blob_publication_reservations").fetchone()[0] == (
            original.publication_claim.receipt.publication_id
        )
        # The same sealed row remains consumable after its private file moved.
        restored_after_publication = _prepared_material_from_record(_prepared_material_record(original), publisher)
        admit_material(conn, prepared=restored_after_publication, observed_at_ms=1)
    assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0


def test_sealed_claim_cannot_be_restored_under_a_fresh_publisher(
    material_archive: tuple[sqlite3.Connection, ArchiveBlobPublisher],
) -> None:
    _conn, publisher = material_archive
    prepared = prepare_material(
        blob_store=publisher, source_uri="https://example.test/item", referrer_ref="message:item", payload=b"captured"
    )
    fresh = ArchiveBlobPublisher(publisher.source_db_path, publisher.root)
    try:
        with pytest.raises(ValueError):
            _prepared_material_from_record(_prepared_material_record(prepared), fresh)
    finally:
        prepared.discard()
