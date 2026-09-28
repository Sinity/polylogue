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


def test_an_excised_sqlite_snapshot_is_a_typed_excision_not_a_parse_failure(tmp_path: Path) -> None:
    """A parse route that reads its snapshot back after flushing stops typed.

    Anti-vacuity: drop ``require_published`` after the Hermes state-db flush
    and ``parse_state_db`` opens the discarded snapshot path, so the call
    raises ``FileNotFoundError`` and the walk records a cursor failure.
    """
    import pytest

    from polylogue.sources.source_parsing import parse_one_source_path
    from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError
    from tests.unit.sources.test_hermes_import_explain import _write_state_db

    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    state_db = tmp_path / "state.db"
    _write_state_db(state_db)
    probe = ArchiveBlobPublisher(root / "source.db", root / "blob")
    snapshot_hash = snapshot_sqlite_to_blob(state_db, probe).blob_hash
    probe.discard_pending()
    with sqlite3.connect(root / "source.db") as source:
        record_excised_blob_hash(
            source,
            blob_hash=bytes.fromhex(snapshot_hash),
            reason="synthetic excision",
            actor="test",
            excised_at_ms=1,
        )

    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    with pytest.raises(ContentExcisedError) as refused:
        list(
            parse_one_source_path(
                str(state_db),
                file_mtime=None,
                source_name="hermes",
                sidecar_data={},
                capture_raw=True,
                blob_store=publisher,
            )
        )
    assert refused.value.blob_hash.hex() == snapshot_hash
    assert not publisher.exists(snapshot_hash)


def test_reading_a_refused_hash_is_a_typed_excision(tmp_path: Path) -> None:
    """Anti-vacuity: let ``blob_path`` fall through for a refused hash and a
    reader gets the staging path that flush deleted (``FileNotFoundError``)."""
    import pytest

    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    excised = b"bytes a later reader asks for"
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
    publisher.flush()

    assert publisher.exists(excised_hex) is False
    with pytest.raises(ContentExcisedError):
        publisher.read_all(excised_hex)


def _claude_code_line(session_id: str, text: str) -> bytes:
    import json

    record = {
        "type": "user",
        "uuid": f"{session_id}-message-1",
        "parentUuid": None,
        "sessionId": session_id,
        "timestamp": "2026-07-10T00:00:00Z",
        "message": {"role": "user", "content": text},
    }
    return (json.dumps(record) + "\n").encode("utf-8")


def test_an_excised_grouped_raw_capture_is_refused_and_a_zip_keeps_its_other_members(tmp_path: Path) -> None:
    """Raw capture of refused bytes stops typed, per file and per ZIP member.

    Anti-vacuity: drop ``require_published`` after the grouped flush and the
    plain file yields a ``RawSessionData`` naming discarded bytes; drop it in
    the ZIP member route and the excised member is still emitted.
    """
    import zipfile

    import pytest

    from polylogue.sources.source_parsing import parse_one_source_path
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "archive"
    with ArchiveStore(root, initialize=True, read_only=False):
        pass
    excised = _claude_code_line("excised-session", "bytes the operator excised")
    kept = _claude_code_line("kept-session", "bytes that stay")
    with sqlite3.connect(root / "source.db") as source:
        record_excised_blob_hash(
            source,
            blob_hash=hashlib.sha256(excised).digest(),
            reason="synthetic excision",
            actor="test",
            excised_at_ms=1,
        )
    plain = tmp_path / "excised.jsonl"
    plain.write_bytes(excised)
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    with pytest.raises(ContentExcisedError):
        list(
            parse_one_source_path(
                str(plain),
                file_mtime=None,
                source_name="claude-code",
                sidecar_data={},
                capture_raw=True,
                blob_store=publisher,
            )
        )

    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("projects/p/excised.jsonl", excised)
        zf.writestr("projects/p/kept.jsonl", kept)
    pairs = list(
        parse_one_source_path(
            str(bundle),
            file_mtime=None,
            source_name="claude-code",
            sidecar_data={},
            capture_raw=True,
            blob_store=publisher,
        )
    )
    captured = {raw.blob_hash for raw, _session in pairs if raw is not None}
    assert hashlib.sha256(kept).hexdigest() in captured
    assert hashlib.sha256(excised).hexdigest() not in captured
