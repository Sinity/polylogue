"""Attachment reacquisition across capture revisions (polylogue-4zqh3).

An upload-only claude.ai ``files`` reference records a name and a size but no
bytes, so its first ingest is honestly ``unfetched``. When a later revision of
the same capture carries the payload as ``extracted_content``, the bytes must
land as an ``acquired`` blob under the *same* attachment identity — a second
identity would strand the original reference and double-count the attachment.

Anti-vacuity: drop ``extracted_content`` from the ``files`` branch of
``attachment_from_meta`` and the second revision stays ``unfetched``; fold
acquisition state into ``_attachment_id`` and the two revisions mint different
identities, growing the reference count.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.claude import parse_ai
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

SESSION_UUID = "reacquisition-session"
FILE_UUID = "upload-only-file"
PAYLOAD = "restored attachment payload\n"
PAYLOAD_BYTES = PAYLOAD.encode("utf-8")


def _capture(*, extracted_content: str | None) -> dict[str, object]:
    """One claude.ai capture revision carrying a single upload-only reference."""
    file_record: dict[str, object] = {
        "file_uuid": FILE_UUID,
        "uuid": FILE_UUID,
        "file_kind": "blob",
        "file_name": "restored.md",
        "size_bytes": len(PAYLOAD_BYTES),
        "path": "/mnt/user-data/uploads/restored.md",
        "success": True,
    }
    if extracted_content is not None:
        file_record["extracted_content"] = extracted_content
    return {
        "uuid": SESSION_UUID,
        "name": "Attachment reacquisition",
        "chat_messages": [
            {
                "uuid": "m0",
                "sender": "human",
                "text": "Please read this.",
                "files": [file_record],
            }
        ],
    }


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _preacquired(store: BlobStore, session: ParsedSession) -> dict[int, tuple[bytes | None, int, str]]:
    acquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        acquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    return acquired


def _attachment_state(conn: sqlite3.Connection) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(
        "SELECT attachment_id, display_name, byte_count, blob_hash, acquisition_status FROM attachments"
    ).fetchone()
    assert row is not None
    return row


def _ref_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0])


def test_upload_only_reference_gains_bytes_at_a_stable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = _connect(tmp_path / "index.db")

    before = parse_ai(_capture(extracted_content=None), "fallback")
    write_parsed_session_to_archive(conn, before, preacquired_attachment_blobs=_preacquired(store, before))

    unfetched = _attachment_state(conn)
    assert unfetched["acquisition_status"] == "unfetched"
    assert unfetched["blob_hash"] is None
    assert unfetched["byte_count"] == len(PAYLOAD_BYTES)
    identity = str(unfetched["attachment_id"])
    assert _ref_count(conn) == 1

    after = parse_ai(_capture(extracted_content=PAYLOAD), "fallback")
    write_parsed_session_to_archive(conn, after, preacquired_attachment_blobs=_preacquired(store, after))

    acquired = _attachment_state(conn)
    assert str(acquired["attachment_id"]) == identity
    assert acquired["acquisition_status"] == "acquired"
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(PAYLOAD_BYTES).digest()
    assert acquired["byte_count"] == len(PAYLOAD_BYTES)
    assert store.read_all(hashlib.sha256(PAYLOAD_BYTES).hexdigest()) == PAYLOAD_BYTES
    assert _ref_count(conn) == 1


def test_replaying_the_acquired_revision_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = _connect(tmp_path / "index.db")

    for _ in range(2):
        session = parse_ai(_capture(extracted_content=PAYLOAD), "fallback")
        write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=_preacquired(store, session))

    acquired = _attachment_state(conn)
    assert acquired["acquisition_status"] == "acquired"
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(PAYLOAD_BYTES).digest()
    assert _ref_count(conn) == 1
