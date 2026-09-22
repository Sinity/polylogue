"""Attachment coverage is proven by content, not by a file existing at a path.

``scan_attachment_coverage`` decided ``acquired_with_bytes_count`` with
``BlobStore.exists``. An object overwritten with bytes the recorded
``blob_hash`` no longer names therefore counted as corroborated: both debt
counters stayed zero and the attachment-coverage check printed the green
summary that every acquired attachment has its bytes. That is the exact
contradicted-object state the attachment re-bind probe
(``operations/attachment_convergence._surviving_blob_ref``) already refuses to
trust, decided by re-hashing for the same reason.

Anti-vacuity: put ``blob_store.exists`` back in the corroboration branch of
``polylogue/storage/blob_integrity.py`` and
``test_overwritten_object_is_not_coverage`` goes red -- the report returns
``acquired_with_bytes_count == 1`` with ``ok`` true. The intact and missing
cases pin the opposite direction so "call everything corrupt" cannot pass.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.maintenance.archive_verification import _check_attachment_coverage_at_index_path
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.blob_integrity import scan_attachment_coverage
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_PAYLOAD = b"attachment bytes the archive claims it fetched"


def _seed_acquired_attachment(index_db: Path, store: BlobStore) -> tuple[str, str]:
    """Write one ``acquired`` attachment and return ``(attachment_id, blob_hash)``."""
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="s1",
        title="s1",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="here is a file",
                position=0,
                variant_index=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="here is a file")],
            )
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="att-acquired",
                message_provider_id="m0",
                name="note.txt",
                mime_type="text/plain",
                inline_bytes=_PAYLOAD,
            )
        ],
    )
    blob_hash, blob_size = store.write_from_bytes(_PAYLOAD)
    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    try:
        write_parsed_session_to_archive(
            conn,
            session,
            preacquired_attachment_blobs={
                id(session.attachments[0]): (bytes.fromhex(blob_hash), blob_size, "acquired")
            },
        )
        attachment_id = str(conn.execute("SELECT attachment_id FROM attachments").fetchone()["attachment_id"])
    finally:
        conn.close()
    return attachment_id, blob_hash


class TestAttachmentCoverageProvesContent:
    def test_overwritten_object_is_not_coverage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        index_db = tmp_path / "index.db"
        store = BlobStore(tmp_path / "blob")
        monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
        attachment_id, blob_hash = _seed_acquired_attachment(index_db, store)

        # The canonical path survives; its content does not.
        store.blob_path(blob_hash).write_bytes(b"different bytes entirely")

        report = scan_attachment_coverage(index_db, store=store, sample_size=5)
        assert report.acquired_count == 1
        assert report.acquired_with_bytes_count == 0, "a contradicted object was certified as coverage"
        assert report.acquired_corrupt_count == 1
        assert report.acquired_corrupt_sample == (attachment_id,)
        assert report.acquired_missing_blob_count == 0, "the object is present; it is not missing"
        assert report.ok is False
        assert report.to_dict()["acquired_corrupt_count"] == 1

    def test_the_coverage_check_reports_the_contradiction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verification check must not print its green summary for this archive."""
        index_db = tmp_path / "index.db"
        store = BlobStore(tmp_path / "blob")
        monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
        _attachment_id, blob_hash = _seed_acquired_attachment(index_db, store)
        store.blob_path(blob_hash).write_bytes(b"different bytes entirely")

        check = _check_attachment_coverage_at_index_path(tmp_path, index_db, 5)
        assert check.count == 1
        assert check.status.value == "error"
        assert "corrupt=1" in check.summary

    def test_intact_object_is_still_coverage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Opposite direction: an honest archive must stay green."""
        index_db = tmp_path / "index.db"
        store = BlobStore(tmp_path / "blob")
        monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
        _seed_acquired_attachment(index_db, store)

        report = scan_attachment_coverage(index_db, store=store, sample_size=5)
        assert report.acquired_with_bytes_count == 1
        assert report.acquired_corrupt_count == 0
        assert report.ok is True

        check = _check_attachment_coverage_at_index_path(tmp_path, index_db, 5)
        assert check.count == 0
        assert check.status.value == "ok"

    def test_absent_object_is_still_missing_not_corrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Opposite direction: the missing bucket must not be swallowed by the new one."""
        index_db = tmp_path / "index.db"
        store = BlobStore(tmp_path / "blob")
        monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
        _attachment_id, blob_hash = _seed_acquired_attachment(index_db, store)
        store.blob_path(blob_hash).unlink()

        report = scan_attachment_coverage(index_db, store=store, sample_size=5)
        assert report.acquired_missing_blob_count == 1
        assert report.acquired_corrupt_count == 0
        assert report.acquired_with_bytes_count == 0
        assert report.ok is False
