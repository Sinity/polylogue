"""No publication route can put excised bytes back on disk.

The excision ledger is read in the reservation's own write transaction, so
every ``ArchiveBlobPublisher`` caller -- raw acquisition, parse-time
snapshots, attachment convergence -- inherits the refusal (polylogue-u6jyu).

Anti-vacuity: drop the ``_excised_hashes`` filter in ``reserve_many`` and the
excised payload is reserved and published under its content hash.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import record_excised_blob_hash


def test_an_excised_payload_is_never_published(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    excised = b"bytes the operator excised"
    kept = b"bytes that stay"
    with sqlite3.connect(root / "source.db") as source:
        record_excised_blob_hash(
            source,
            blob_hash=hashlib.sha256(excised).digest(),
            reason="synthetic excision",
            actor="test",
            excised_at_ms=1,
        )

    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    excised_hex, _ = publisher.write_from_bytes(excised)
    kept_hex, _ = publisher.write_from_bytes(kept)
    receipts = publisher.flush()

    assert [receipt.blob_hash for receipt in receipts] == [kept_hex]
    assert not publisher.exists(excised_hex)
    assert publisher.read_all(kept_hex) == kept
    assert publisher.receipt_id(excised_hex) is None
    assert not any((root / "blob" / ".staging").iterdir())
    with sqlite3.connect(root / "source.db") as source:
        reserved = {
            bytes(row[0]).hex() for row in source.execute("SELECT blob_hash FROM blob_publication_reservations")
        }
    assert reserved == {kept_hex}
